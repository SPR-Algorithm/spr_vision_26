#!/usr/bin/env python3
"""Validate real Buff S3/S4/S5 JSONL and optional synchronized recordings.

No frame-rate assumption is used for measurements. A recording's nominal AVI
duration is reported separately from its IMU capture duration. A live JSONL can
contain more frames than the Recorder saved; only replay logs are frame-aligned
with an explicitly supplied sidecar.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter
import json
import math
from pathlib import Path
import sys
from typing import Any


UNIT_TOLERANCE = 1e-3
GEOMETRY_TOLERANCE_M = 1e-4
MAX_PAIR_GAP_S = 0.2
CONTOURS = ("armor_contour", "r_mark_contour", "light_arm_contour")


def number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def vector(value: Any, size: int = 3) -> bool:
    return isinstance(value, list) and len(value) == size and all(number(x) for x in value)


def norm(value: list[float]) -> float:
    return math.sqrt(sum(x * x for x in value))


def dot(left: list[float], right: list[float]) -> float:
    return sum(x * y for x, y in zip(left, right))


def angle_deg(left: list[float], right: list[float], quaternion: bool = False) -> float:
    cosine = dot(left, right) / (norm(left) * norm(right))
    if quaternion:
        cosine = abs(cosine)  # q and -q represent the same orientation.
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine)))) * (2 if quaternion else 1)


def stats(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "median": None, "p95": None, "max": None}

    def quantile(fraction: float) -> float:
        index = (len(ordered) - 1) * fraction
        lower = int(index)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)

    return {"count": len(ordered), "median": quantile(0.5), "p95": quantile(0.95), "max": ordered[-1]}


class Checks:
    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.examples: list[dict[str, Any]] = []

    def fail(self, code: str, location: str, detail: str) -> None:
        self.counts[code] += 1
        if len(self.examples) < 20:
            self.examples.append({"check": code, "location": location, "detail": detail})

    def finite_tree(self, value: Any, location: str) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            self.fail("nonfinite", location, "numeric value must be finite")
        elif isinstance(value, dict):
            for key, item in value.items():
                self.finite_tree(item, f"{location}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                self.finite_tree(item, f"{location}[{index}]")

    def unit_vector(self, value: Any, location: str, size: int = 3) -> bool:
        if not vector(value, size):
            self.fail("invalid_vector", location, f"expected {size} finite numbers")
            return False
        if abs(norm(value) - 1) > UNIT_TOLERANCE:
            self.fail("unit_norm", location, f"norm={norm(value):.9g}, tolerance={UNIT_TOLERANCE}")
            return False
        return True


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no frame records")
    return rows


def capture_time(row: dict[str, Any]) -> float | None:
    # Relative time is also the original capture time in replay logs.
    for name in ("timestamp_relative_s", "timestamp_steady_s"):
        if number(row.get(name)):
            return float(row[name])
    return None


def original_frame_index(row: dict[str, Any]) -> int | None:
    if type(row.get("input_frame_index")) is int:
        return row["input_frame_index"]
    return row["frame"] - 1 if type(row.get("frame")) is int else None


def check_pose(pose: dict[str, Any], blades: list[Any], checks: Checks, where: str) -> None:
    center = pose.get("position_world_m")
    if not vector(center):
        checks.fail("pose_position", where, "position_world_m requires three finite numbers")
    normal_ok = checks.unit_vector(pose.get("plane_normal_world"), f"{where}.plane_normal_world")
    phase_ok = checks.unit_vector(pose.get("phase_zero_direction_world"), f"{where}.phase_zero_direction_world")
    quaternion = pose.get("orientation_world_wxyz")
    quaternion_ok = False
    if quaternion is not None:
        quaternion_ok = checks.unit_vector(quaternion, f"{where}.orientation_world_wxyz", 4)
    normal = pose.get("plane_normal_world")
    phase = pose.get("phase_zero_direction_world")
    if normal_ok and phase_ok and abs(dot(normal, phase)) > UNIT_TOLERANCE:
        checks.fail("pose_axes_perpendicular", where, f"normal dot phase_zero={dot(normal, phase):.9g}")
    if quaternion_ok:
        w, x, y, z = [value / norm(quaternion) for value in quaternion]
        orientation_normal = [1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)]
        orientation_phase = [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)]
        if normal_ok and norm([a - b for a, b in zip(normal, orientation_normal)]) > UNIT_TOLERANCE:
            checks.fail("pose_orientation_normal", where, "orientation X axis differs from plane normal")
        if phase_ok and norm([a - b for a, b in zip(phase, orientation_phase)]) > UNIT_TOLERANCE:
            checks.fail("pose_orientation_phase", where, "orientation Z axis differs from phase-zero direction")
    radius = pose.get("blade_radius_m")
    if not number(radius) or radius <= 0:
        checks.fail("pose_radius", where, "blade_radius_m must be finite and positive")
    error = pose.get("reprojection_error_px")
    if error is not None and (not number(error) or error < 0):
        checks.fail("anchor_error", where, "reprojection_error_px must be finite and nonnegative")
    slots = []
    for index, blade in enumerate(blades):
        location = f"{where}.blades[{index}]"
        if not isinstance(blade, dict):
            checks.fail("blade_object", location, "expected a blade object")
            continue
        slot = blade.get("slot")
        if type(slot) is not int or not 1 <= slot <= 5:
            checks.fail("blade_slot", location, "slot must be an integer in [1,5]")
        slots.append(slot)
        blade_center = blade.get("armor_center_world_m")
        if not vector(blade_center):
            checks.fail("blade_position", location, "armor_center_world_m requires three finite numbers")
        elif vector(center):
            offset = [x - y for x, y in zip(blade_center, center)]
            if number(radius) and abs(norm(offset) - radius) > GEOMETRY_TOLERANCE_M:
                checks.fail("blade_radius_distance", location, f"distance={norm(offset):.9g}, radius={radius}")
            if normal_ok and abs(dot(offset, normal)) > GEOMETRY_TOLERANCE_M:
                checks.fail("blade_in_plane", location, f"normal distance={dot(offset, normal):.9g} m")
    if len(set(str(x) for x in slots)) != len(slots):
        checks.fail("blade_duplicate_slot", where, "slots must be unique within a reconstruction")


def first_number(mapping: dict[str, Any], names: tuple[str, ...]) -> float | None:
    for name in names:
        if number(mapping.get(name)):
            return float(mapping[name])
    return None


def predict_motion(motion: dict[str, Any], future_s: float) -> float | None:
    """The exact formula serialized by auto_buff_observation_debug.cpp."""
    if motion.get("model") == "CONSTANT_VELOCITY":
        phase, velocity = motion.get("continuous_phase_rad"), motion.get("angular_velocity_rad_s")
        return phase + future_s * velocity if number(phase) and number(velocity) else None
    if motion.get("model") != "HARMONIC_PHASE":
        return None
    parameters = motion.get("harmonic_phase")
    reference, epoch = motion.get("reference_timestamp_steady_s"), motion.get("model_epoch_steady_s")
    keys = ("cos_amplitude_rad", "sin_amplitude_rad", "linear_rate_rad_s", "phase_offset_rad", "angular_frequency_rad_s")
    if not isinstance(parameters, dict) or not all(number(parameters.get(key)) for key in keys) or not number(reference) or not number(epoch):
        return None
    a, b, rate, offset, frequency = [parameters[key] for key in keys]
    t = reference - epoch + future_s
    return a * math.cos(frequency * t) + b * math.sin(frequency * t) + rate * t + offset


def holdout_motion(rows: list[dict[str, Any]], maximum_basis_deg: float) -> dict[str, Any]:
    """Score future measurements, never future fitted or predicted phases."""
    accepted = []
    for row in rows:
        motion, timestamp = row.get("motion"), capture_time(row)
        if isinstance(motion, dict) and timestamp is not None and number(motion.get("observed_phase_rad")):
            accepted.append((timestamp, motion))
    times = [sample[0] for sample in accepted]
    rejected: Counter[str] = Counter()
    candidates = 0
    errors = []
    fivefold_errors = []
    unfiltered_errors = []
    time_errors = []
    basis_changes = []
    methods: Counter[str] = Counter()
    period = 2 * math.pi / 5
    for row in rows:
        motion, timestamp = row.get("motion"), capture_time(row)
        if not isinstance(motion, dict) or motion.get("converged") is not True:
            continue
        candidates += 1
        prediction = predict_motion(motion, 0.5)
        offset = motion.get("cumulative_switch_offset_rad")
        if timestamp is None or prediction is None or not number(motion.get("predicted_phase_0_5s_rad")):
            rejected["missing_explicit_prediction_model"] += 1
            continue
        if not number(offset) or type(motion.get("resets")) is not int:
            rejected["missing_canonical_offset_or_reset_id"] += 1
            continue
        desired_time = timestamp + 0.5
        right = bisect_left(times, desired_time)
        if right < len(times) and abs(times[right] - desired_time) <= 1e-8:
            selected = [(right, 1.0)]
            method = "exact"
            actual_time = times[right]
        elif 0 < right < len(times) and 0 < times[right] - times[right - 1] <= MAX_PAIR_GAP_S:
            fraction = (desired_time - times[right - 1]) / (times[right] - times[right - 1])
            selected = [(right - 1, 1 - fraction), (right, fraction)]
            method = "interpolated"
            actual_time = desired_time
        else:
            nearest = min((index for index in (right - 1, right) if 0 <= index < len(times)), key=lambda index: abs(times[index] - desired_time), default=None)
            if nearest is None or abs(times[nearest] - desired_time) > 0.05:
                rejected["no_future_observation_within_0_05_s"] += 1
                continue
            selected = [(nearest, 1.0)]
            method = "nearest_within_0_05_s"
            actual_time = times[nearest]
        end = selected[-1][0]
        start = bisect_left(times, timestamp)
        segment = accepted[start:end + 1]
        if not segment or segment[-1][0] <= timestamp:
            rejected["no_future_observation"] += 1
            continue
        segment_times = [timestamp] + [item[0] for item in segment]
        if any(later - earlier > MAX_PAIR_GAP_S for earlier, later in zip(segment_times, segment_times[1:])):
            rejected["accepted_observation_gap_over_0_2_s"] += 1
            continue
        if any(item.get("resets") != motion["resets"] for _, item in segment):
            rejected["reset_changed"] += 1
            continue
        if any(not number(accepted[index][1].get("cumulative_switch_offset_rad")) for index, _ in selected):
            rejected["future_canonical_offset_missing"] += 1
            continue
        measured = sum(weight * (accepted[index][1]["observed_phase_rad"] - accepted[index][1]["cumulative_switch_offset_rad"]) for index, weight in selected)
        signed_error = prediction - offset - measured
        unfiltered_errors.append(abs(signed_error))
        maximum_change = 0.0
        valid_basis = True
        for name in ("plane_normal_world", "phase_zero_direction_world"):
            initial = motion.get(name)
            if not vector(initial) or norm(initial) <= 0:
                valid_basis = False
                break
            for _, sample in segment:
                value = sample.get(name)
                if not vector(value) or norm(value) <= 0:
                    valid_basis = False
                    break
                maximum_change = max(maximum_change, angle_deg(initial, value))
        if not valid_basis:
            rejected["missing_valid_basis"] += 1
            continue
        basis_changes.append(maximum_change)
        if maximum_change > maximum_basis_deg:
            rejected["basis_change_over_limit"] += 1
            continue
        errors.append(abs(signed_error))
        fivefold_errors.append(abs((signed_error + period / 2) % period - period / 2))
        methods[method] += 1
        time_errors.append(abs(actual_time - desired_time))
    return {
        "status": "evaluated" if errors else "not_evaluated",
        "candidate_converged_frames": candidates,
        "evaluated_pairs": len(errors),
        "canonical_absolute_phase_error_rad": stats(errors),
        "canonical_absolute_phase_error_deg": stats([math.degrees(x) for x in errors]),
        "fivefold_equivalent_error_deg_auxiliary": stats([math.degrees(x) for x in fivefold_errors]),
        "before_basis_filter_error_deg_auxiliary": stats([math.degrees(x) for x in unfiltered_errors]),
        "basis_change_deg": stats(basis_changes),
        "maximum_basis_change_deg": maximum_basis_deg,
        "future_timestamp_error_s": stats(time_errors),
        "observation_matching": dict(methods),
        "rejected_candidates": dict(rejected),
        "canonical_definition": "observed or predicted continuous phase minus that frame's cumulative_switch_offset_rad; primary residual is not wrapped by blade spacing",
        "matching_rule": "0.5 s future observed phase, interpolation across <=0.2 s accepted gaps or nearest within 0.05 s; same reset ID and basis within the declared limit throughout the interval",
        "identity_note": "phase forecast diagnostics without independent blade-identity ground truth; fivefold equivalence is an auxiliary metric, not a physical target hit test",
    }


def analyze(rows: list[dict[str, Any]], checks: Checks) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    quality: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    geometry_rejected: Counter[str] = Counter()
    state_rejected: Counter[str] = Counter()
    contours = {name: {"usable": 0, "reported": 0, "reject_reason": Counter()} for name in CONTOURS}
    statuses: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    motion_status: Counter[str] = Counter()
    motion_models: Counter[str] = Counter()
    errors: list[float] = []
    pnp_errors: list[float] = []
    chamfer_errors: list[float] = []
    contour_samples: list[float] = []
    motion_statistics: dict[str, list[float]] = {key: [] for key in ("samples", "span_s", "fit_rms_rad", "angular_velocity_rad_s", "continuous_phase_rad", "resets", "blade_switches", "cumulative_switch_offset_rad")}
    harmonic_statistics: dict[str, list[float]] = {}
    processor_modes: Counter[str] = Counter()
    processor_frames = processor_nontransmitted = processor_hold = processor_idle_zero = processor_finite = 0
    intervals: list[float] = []
    center_steps: list[float] = []
    normal_steps: list[float] = []
    phase_steps: list[float] = []
    imu_steps: list[float] = []
    plots: dict[str, list[Any]] = {key: [] for key in ("pose_t", "anchor", "pair_t", "center", "normal", "motion_t", "phase", "model_phase", "omega", "converged", "reset")}
    eligible = ge_2_blades = success = eligible_success = invalid_imu = invalid_observations = 0
    gaps = missing_frames = pair_skipped = reference_changes = reference_pairs = slot_changes = 0
    motion_frames = converged = motion_target_frames = 0
    previous: dict[str, Any] | None = None
    previous_time: float | None = None
    valid_times = []
    worst_normal_pairs = []
    for index, row in enumerate(rows):
        where = f"row {index + 1} (frame {row.get('frame', '?')})"
        checks.finite_tree(row, where)
        processor = row.get("processor_check")
        if processor is not None and not isinstance(processor, dict):
            checks.fail("processor_check_schema", where, "processor_check must be an object or null")
        if isinstance(processor, dict):
            processor_frames += 1
            mode = processor.get("mode")
            processor_modes[str(mode)] += 1
            if type(mode) is not int or mode not in (0, 1):
                checks.fail("processor_mode", where, "processor mode must be integer 0 or 1")
            processor_nontransmitted += processor.get("transmitted") is False
            if processor.get("transmitted") is not False:
                checks.fail("processor_transmitted", where, "read-only processor check must explicitly report transmitted=false")
            for key in ("finite", "idle_zero", "measured_attitude_hold"):
                if processor.get(key) is not True:
                    checks.fail(f"processor_{key}", where, f"processor_check.{key} must be true")
            processor_finite += processor.get("finite") is True
            processor_idle_zero += processor.get("idle_zero") is True
            processor_hold += processor.get("measured_attitude_hold") is True
            attitude_finite = number(processor.get("yaw")) and number(processor.get("pitch"))
            if not attitude_finite:
                checks.fail("processor_attitude_finite", where, "processor yaw and pitch must both be finite numbers")
            if mode == 0 and (processor.get("yaw") != 0 or processor.get("pitch") != 0):
                checks.fail("processor_idle_command_zero", where, "mode 0 requires exactly zero yaw and pitch")
            if mode == 1:
                gimbal = row.get("gimbal_state")
                if not isinstance(gimbal, dict) or not number(gimbal.get("yaw")) or not number(gimbal.get("pitch")):
                    checks.fail("processor_measured_attitude_missing", where, "mode 1 requires this frame's finite measured gimbal yaw and pitch")
                elif processor.get("yaw") != gimbal["yaw"] or processor.get("pitch") != gimbal["pitch"]:
                    checks.fail("processor_measured_attitude_mismatch", where, "mode 1 yaw and pitch must exactly equal this frame's measured gimbal state")
        frame = row.get("frame")
        if type(frame) is not int or frame < 1:
            checks.fail("frame_index", where, "frame must be a positive integer")
        timestamp = capture_time(row)
        if timestamp is None:
            checks.fail("timestamp", where, "no finite capture timestamp")
        else:
            valid_times.append(timestamp)
        dt = timestamp - previous_time if timestamp is not None and previous_time is not None else None
        consecutive = False
        if previous is not None:
            old_frame = previous.get("frame")
            if type(frame) is int and type(old_frame) is int:
                if frame <= old_frame:
                    checks.fail("frame_order", where, f"frame {frame} follows {old_frame}")
            current_index, old_index = original_frame_index(row), original_frame_index(previous)
            if current_index is not None and old_index is not None:
                if current_index <= old_index:
                    checks.fail("input_frame_order", where, f"original input index {current_index} follows {old_index}")
                missing_frames += max(0, current_index - old_index - 1)
                consecutive = current_index == old_index + 1
            if dt is not None:
                if dt <= 0:
                    checks.fail("timestamp_order", where, f"capture dt={dt:.9g} s must be positive")
                else:
                    intervals.append(dt)
                    gaps += dt > MAX_PAIR_GAP_S
        imu = row.get("q_wxyz")
        imu_ok = checks.unit_vector(imu, f"{where}.q_wxyz", 4)
        invalid_imu += row.get("imu_valid") is False
        observations = row.get("observations", [])
        if not isinstance(observations, list):
            checks.fail("observations", where, "observations must be an array")
            observations = []
        accepted = 0
        for observation in observations:
            if not isinstance(observation, dict) or not isinstance(observation.get("refined"), dict):
                checks.fail("observation_schema", where, "observation needs refined diagnostics")
                continue
            refined = observation["refined"]
            value = str(refined.get("quality", "MISSING"))
            quality[value] += 1
            invalid_observations += value == "INVALID"
            accepted += value in ("GOOD", "DEGRADED")
            rejected[str(refined.get("reject_reason", "MISSING"))] += 1
            geometry_rejected[str(refined.get("geometry_reject_reason", "MISSING"))] += 1
            state_rejected[str(refined.get("state_reject_reason") or "NONE")] += 1
            if value in ("GOOD", "DEGRADED"):
                points = refined.get("points")
                if not isinstance(points, list) or len(points) != 5 or not all(vector(p, 2) for p in points):
                    checks.fail("accepted_points", where, "accepted observation needs five finite 2D points")
            for name in CONTOURS:
                contour = refined.get(name)
                if isinstance(contour, dict):
                    contours[name]["reported"] += 1
                    contours[name]["usable"] += contour.get("usable") is True
                    contours[name]["reject_reason"][str(contour.get("reject_reason", "MISSING"))] += 1
                    if contour.get("usable") is True and (not number(contour.get("point_count")) or contour["point_count"] <= 0):
                        checks.fail("usable_contour_empty", where, f"{name} is usable without points")
        if row.get("valid_blades") != accepted:
            checks.fail("accepted_blade_count", where, f"valid_blades={row.get('valid_blades')}, accepted observations={accepted}")
        ge_2_blades += accepted >= 2
        is_eligible = accepted >= 2 and row.get("imu_valid") is not False
        eligible += is_eligible
        statuses[str(row.get("reconstruction_status", "MISSING"))] += 1
        reconstruction = row.get("reconstruction")
        pose = reconstruction.get("pose") if isinstance(reconstruction, dict) else None
        if isinstance(pose, dict):
            success += 1
            eligible_success += is_eligible
            if not is_eligible:
                checks.fail("reconstruction_eligibility", where, "pose produced without >=2 accepted blades and valid IMU")
            blades = reconstruction.get("blades", [])
            if not isinstance(blades, list):
                checks.fail("reconstruction_blades", where, "blades must be an array")
                blades = []
            check_pose(pose, blades, checks, where)
            sources[str(pose.get("source", "MISSING"))] += 1
            error = pose.get("reprojection_error_px")
            if number(error):
                errors.append(error)
                if timestamp is not None:
                    plots["pose_t"].append(timestamp)
                    plots["anchor"].append(error)
            if number(pose.get("pnp_reprojection_error_px")):
                pnp_errors.append(pose["pnp_reprojection_error_px"])
            if number(pose.get("contour_sample_count")):
                contour_samples.append(pose["contour_sample_count"])
            for key in ("contour_error_px", "chamfer_error_px", "chamfer_cost_px"):
                if number(pose.get(key)):
                    chamfer_errors.append(pose[key])
                    break
            old_reconstruction = previous.get("reconstruction") if previous else None
            old_pose = old_reconstruction.get("pose") if isinstance(old_reconstruction, dict) else None
            if isinstance(old_pose, dict):
                if consecutive and dt is not None and 0 < dt <= MAX_PAIR_GAP_S:
                    center, old_center = pose.get("position_world_m"), old_pose.get("position_world_m")
                    normal, old_normal = pose.get("plane_normal_world"), old_pose.get("plane_normal_world")
                    phase, old_phase = pose.get("phase_zero_direction_world"), old_pose.get("phase_zero_direction_world")
                    center_step = norm([x - y for x, y in zip(center, old_center)]) if vector(center) and vector(old_center) else None
                    normal_step = angle_deg(normal, old_normal) if vector(normal) and vector(old_normal) and norm(normal) > 0 and norm(old_normal) > 0 else None
                    if center_step is not None:
                        center_steps.append(center_step)
                    if normal_step is not None:
                        normal_steps.append(normal_step)
                        worst_normal_pairs.append({"from_frame": previous["frame"], "to_frame": frame, "dt_s": dt, "step_deg": normal_step})
                    if vector(phase) and vector(old_phase) and norm(phase) > 0 and norm(old_phase) > 0:
                        phase_steps.append(angle_deg(phase, old_phase))
                    plots["pair_t"].append(timestamp)
                    plots["center"].append(center_step)
                    plots["normal"].append(normal_step)
                    slots = sorted(b.get("slot", -1) for b in blades if isinstance(b, dict) and type(b.get("slot")) is int)
                    old_slots = sorted(b.get("slot", -1) for b in old_reconstruction.get("blades", []) if isinstance(b, dict) and type(b.get("slot")) is int)
                    slot_changes += slots != old_slots
                    for key in ("reference_observation_index", "reference_index"):
                        if key in reconstruction and key in old_reconstruction:
                            reference_pairs += 1
                            reference_changes += reconstruction[key] != old_reconstruction[key]
                            break
                else:
                    pair_skipped += 1
        if previous is not None and consecutive and dt is not None and 0 < dt <= MAX_PAIR_GAP_S:
            old_imu = previous.get("q_wxyz")
            if imu_ok and vector(old_imu, 4) and norm(old_imu) > 0:
                imu_steps.append(angle_deg(imu, old_imu, quaternion=True))
        motion = row.get("motion")
        if isinstance(motion, dict):
            motion_frames += 1
            for key, values in motion_statistics.items():
                if number(motion.get(key)):
                    values.append(motion[key])
            parameters = motion.get("harmonic_phase")
            if isinstance(parameters, dict) and motion.get("model") == "HARMONIC_PHASE":
                for key, value in parameters.items():
                    if number(value):
                        harmonic_statistics.setdefault(key, []).append(value)
            motion_status[str(motion.get("status", "MISSING"))] += 1
            target = motion.get("target")
            if isinstance(target, dict):
                motion_target_frames += 1
            model = motion.get("model", motion.get("motion_model"))
            if model is None and isinstance(target, dict):
                model = target.get("motion", target.get("motion_model", target.get("model")))
            if isinstance(model, dict):
                motion_models[str(model.get("type", model.get("kind", "OBJECT")))] += 1
            elif model is not None:
                motion_models[str(model)] += 1
            # A status string alone is not proof of convergence.
            is_converged = motion.get("converged") is True
            if isinstance(target, dict):
                is_converged = is_converged or target.get("converged") is True
                if isinstance(model, dict):
                    is_converged = is_converged or model.get("converged") is True
            converged += is_converged
            if is_converged:
                for key in ("continuous_phase_rad", "angular_velocity_rad_s"):
                    if not number(motion.get(key)) and not isinstance(target, dict):
                        checks.fail("converged_motion_value", where, f"converged motion requires finite {key}")
                prediction = predict_motion(motion, 0.5)
                logged_prediction = motion.get("predicted_phase_0_5s_rad")
                if prediction is not None and number(logged_prediction) and abs(prediction - logged_prediction) > 1e-5:
                    checks.fail("motion_prediction_formula", where, f"model prediction={prediction}, logged={logged_prediction}")
                instantaneous = predict_motion(motion, 0.0)
                if instantaneous is not None and number(motion.get("continuous_phase_rad")) and abs(instantaneous - motion["continuous_phase_rad"]) > 1e-5:
                    checks.fail("motion_instantaneous_phase", where, "continuous phase differs from the model evaluated at its reference timestamp")
                reference = motion.get("reference_timestamp_steady_s")
                capture_steady = row.get("timestamp_steady_s")
                if number(reference) and number(capture_steady) and abs(reference - capture_steady) > 1e-6:
                    checks.fail("motion_reference_timestamp", where, "model reference timestamp differs from this frame's capture timestamp")
                if motion.get("model") == "HARMONIC_PHASE" and isinstance(parameters, dict):
                    frequency = parameters.get("angular_frequency_rad_s")
                    if not number(frequency) or frequency <= 0:
                        checks.fail("motion_harmonic_frequency", where, "harmonic angular frequency must be finite and positive")
                if isinstance(motion.get("pose"), dict):
                    predicted_point = motion.get("predicted_armor_center_0_5s_world_m")
                    predicted_blades = [{"slot": 1, "armor_center_world_m": predicted_point}] if "predicted_armor_center_0_5s_world_m" in motion else []
                    check_pose(motion["pose"], predicted_blades, checks, f"{where}.motion.pose")
            phase = first_number(motion, ("observed_phase_rad", "continuous_phase_rad", "phase_rad"))
            if "observed_phase_rad" in motion:
                phase = motion["observed_phase_rad"] if number(motion["observed_phase_rad"]) else None
            if phase is not None and number(motion.get("cumulative_switch_offset_rad")):
                phase -= motion["cumulative_switch_offset_rad"]
            omega = first_number(motion, ("angular_velocity_rad_s", "omega_rad_s"))
            if isinstance(target, dict):
                if phase is None and "observed_phase_rad" not in motion:
                    phase = first_number(target, ("continuous_phase_rad", "phase_rad"))
                if omega is None:
                    omega = first_number(target, ("angular_velocity_rad_s", "omega_rad_s"))
            if timestamp is not None:
                plots["motion_t"].append(timestamp)
                plots["phase"].append(phase)
                model_phase = predict_motion(motion, 0.0) if is_converged else None
                if model_phase is not None and number(motion.get("cumulative_switch_offset_rad")):
                    model_phase -= motion["cumulative_switch_offset_rad"]
                plots["model_phase"].append(model_phase)
                plots["omega"].append(omega)
                plots["converged"].append(is_converged)
                plots["reset"].append(motion.get("resets"))
            for name in ("plane_normal_world", "normal_world", "phase_zero_direction_world"):
                if name in motion and motion[name] is not None:
                    checks.unit_vector(motion[name], f"{where}.motion.{name}")
            normal, zero = motion.get("plane_normal_world"), motion.get("phase_zero_direction_world")
            if vector(normal) and vector(zero) and abs(dot(normal, zero)) > UNIT_TOLERANCE:
                checks.fail("motion_axes_perpendicular", where, f"normal dot phase_zero={dot(normal, zero):.9g}")
        previous, previous_time = row, timestamp
    duration = valid_times[-1] - valid_times[0] if len(valid_times) > 1 else 0.0
    report = {
        "frames": len(rows),
        "timing": {"capture_duration_s": duration, "capture_intervals_s": stats(intervals), "gaps_over_0_2_s": gaps, "missing_frame_indices": missing_frames, "source": "capture timestamps; AVI FPS is not used"},
        "imu": {"invalid_flag_frames": invalid_imu, "consecutive_orientation_step_deg": stats(imu_steps)},
        "s1": {"processor_check_frames": processor_frames, "modes": dict(processor_modes), "nontransmitted_frames": processor_nontransmitted, "finite_flag_frames": processor_finite, "idle_zero_flag_frames": processor_idle_zero, "measured_attitude_hold_flag_frames": processor_hold, "check_note": "mode 0 requires zero yaw/pitch; mode 1 exactly holds this frame's measured gimbal yaw/pitch; all checks are read-only and transmitted must be false; absent legacy diagnostics are not failed"},
        "s3": {"observations": sum(quality.values()), "quality": dict(quality), "invalid_observations": invalid_observations, "reject_reason": dict(rejected), "geometry_reject_reason": dict(geometry_rejected), "state_reject_reason": dict(state_rejected), "contours": contours, "contour_note": "usable counts describe S3 availability, not proof of downstream contour optimization"},
        "s4": {"frames_ge_2_blades": ge_2_blades, "eligible_frames_ge_2_blades_and_valid_imu": eligible, "success_frames": success, "success_on_eligible_frames": eligible_success, "success_rate_on_eligible": eligible_success / eligible if eligible else None, "status": dict(statuses), "pose_source": dict(sources), "anchor_reprojection_error_px": stats(errors), "pnp_initial_anchor_error_px": stats(pnp_errors), "chamfer_error_px": stats(chamfer_errors), "contour_sample_count": stats(contour_samples), "error_semantics": "reprojection_error_px is final point-anchor error, pnp_reprojection_error_px is initial PnP anchor error, contour_error_px is measured contour distance; usable S3 contours alone do not establish Chamfer refinement", "world_center_step_m": stats(center_steps), "directed_normal_step_deg": stats(normal_steps), "phase_zero_direction_step_deg": stats(phase_steps), "pair_rule": "adjacent JSONL rows, original input_frame_index (fallback frame-1) consecutive, 0 < capture dt <= 0.2 s, both reconstructed", "skipped_pose_pairs": pair_skipped, "largest_normal_steps": sorted(worst_normal_pairs, key=lambda p: p["step_deg"], reverse=True)[:5], "slot_set_change_pairs": slot_changes, "explicit_reference_index_pairs": reference_pairs, "explicit_reference_index_change_pairs": reference_changes if reference_pairs else None, "association_note": "slot sets, frame-local reference indices and phase-zero changes do not prove persistent physical blade identity"},
        "s5": {"motion_frames": motion_frames, "target_frames": motion_target_frames, "explicitly_converged_frames": converged, "status": dict(motion_status), "models": dict(motion_models), "statistics": {key: stats(value) for key, value in motion_statistics.items()}, "harmonic_parameters": {key: stats(value) for key, value in harmonic_statistics.items()}},
    }
    return report, plots


def read_imu(path: Path, checks: Checks) -> tuple[list[tuple[float, list[float]]], dict[str, Any]]:
    samples = []
    norms = []
    intervals = []
    with path.open(encoding="utf-8") as stream:
        for row_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                values = [float(x) for x in line.split()]
            except ValueError as error:
                raise ValueError(f"{path}:{row_number}: expected 'seconds w x y z'") from error
            if len(values) != 5 or not all(number(x) for x in values):
                raise ValueError(f"{path}:{row_number}: expected five finite values 'seconds w x y z'")
            timestamp, quaternion = values[0], values[1:]
            if samples:
                dt = timestamp - samples[-1][0]
                intervals.append(dt)
                if dt <= 0:
                    checks.fail("sidecar_timestamp_order", f"{path}:{row_number}", f"dt={dt} s must be positive")
            checks.unit_vector(quaternion, f"{path}:{row_number}.q_wxyz", 4)
            norms.append(norm(quaternion))
            samples.append((timestamp, quaternion))
    if not samples:
        raise ValueError(f"{path}: no IMU rows")
    return samples, {"path": str(path), "frames": len(samples), "capture_duration_s": samples[-1][0] - samples[0][0], "capture_intervals_s": stats(intervals), "quaternion_norm": stats(norms)}


def read_video(path: Path) -> dict[str, Any]:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("--video requires the existing OpenCV Python module (cv2)") from error
    video = cv2.VideoCapture(str(path))
    if not video.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    try:
        fps = video.get(cv2.CAP_PROP_FPS)
        metadata_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
        count = 0
        while True:
            ok, _ = video.read()
            if not ok:
                break
            count += 1
    finally:
        video.release()
    if count == 0:
        raise ValueError(f"{path}: no decodable video frames")
    return {"path": str(path), "decoded_frames": count, "metadata_frames": int(metadata_count) if number(metadata_count) else None, "nominal_fps": fps if number(fps) and fps > 0 else None, "nominal_playback_duration_s": count / fps if number(fps) and fps > 0 else None, "duration_note": "nominal encoded playback duration, not the real capture duration"}


def check_media(rows: list[dict[str, Any]], video_path: Path | None, imu_path: Path | None, checks: Checks) -> dict[str, Any]:
    result: dict[str, Any] = {}
    video = read_video(video_path) if video_path else None
    samples, imu = read_imu(imu_path, checks) if imu_path else ([], None)
    if video is not None:
        result["video"] = video
    if imu is not None:
        result["imu_sidecar"] = imu
    if video is not None and imu is not None:
        matched = video["decoded_frames"] == len(samples)
        result["video_imu_frame_counts_equal"] = matched
        if not matched:
            checks.fail("video_imu_frame_count", "recording", f"video={video['decoded_frames']}, imu={len(samples)}")
    replay = []
    for row in rows:
        source = row.get("input")
        if isinstance(source, str) and source != "camera" and source:
            if video_path and Path(source).resolve() != video_path.resolve():
                checks.fail("replay_input_video", f"frame {row.get('frame')}", f"log input={source}, supplied video={video_path}")
            replay.append(row)
    if not replay:
        result["jsonl_alignment"] = "live log: independent capture stream; Recorder may save a subset, so row.frame is not an AVI index"
        return result
    aligned = 0
    maximum_index = -1
    for row in replay:
        frame = row.get("frame")
        index = row.get("input_frame_index", frame - 1 if type(frame) is int else None)
        where = f"replay frame {frame}"
        if type(index) is not int or index < 0:
            checks.fail("replay_input_frame_index", where, "expected a nonnegative input_frame_index or positive frame")
            continue
        maximum_index = max(maximum_index, index)
        if video is not None and index >= video["decoded_frames"]:
            checks.fail("replay_video_bounds", where, f"input index {index} exceeds video frames")
        if not samples:
            continue
        if index >= len(samples):
            checks.fail("replay_imu_bounds", where, f"input index {index} exceeds IMU rows")
            continue
        timestamp, quaternion = samples[index]
        actual_time = row.get("timestamp_relative_s")
        if number(actual_time) and abs(actual_time - timestamp) > 1e-6:
            checks.fail("replay_imu_timestamp_alignment", where, f"log={actual_time}, sidecar={timestamp} s")
        actual_q = row.get("q_wxyz")
        if vector(actual_q, 4) and norm(actual_q) > 0 and norm(quaternion) > 0 and angle_deg(actual_q, quaternion, quaternion=True) > 1e-4:
            checks.fail("replay_imu_quaternion_alignment", where, "log and indexed sidecar quaternion differ")
        aligned += 1
    result["jsonl_alignment"] = {"kind": "replay", "checked_imu_rows": aligned, "largest_input_frame_index": maximum_index, "subset_allowed": True, "index_source": "input_frame_index (0-based), otherwise frame-1"}
    return result


def plot_report(path: Path, plots: dict[str, list[Any]], threshold: float | None) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("--plot requires matplotlib; JSON validation has no plotting dependency") from error
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    axes[0, 0].plot(plots["pose_t"], plots["anchor"], linewidth=0.7)
    axes[0, 0].set_ylabel("Anchor reprojection error (px)")
    axes[0, 1].plot(plots["pair_t"], plots["center"], linewidth=0.7)
    axes[0, 1].set_ylabel("World center step (m)")
    axes[1, 0].plot(plots["pair_t"], plots["normal"], linewidth=0.7)
    axes[1, 0].set_ylabel("Directed normal step (deg)")
    if threshold is not None:
        axes[1, 0].axhline(threshold, color="red", linestyle="--", label=f"limit {threshold:g} deg")
        axes[1, 0].legend()
    if any(value is not None for value in plots["phase"]):
        for converged, color, label in ((False, "tab:blue", "Observed, collecting"), (True, "tab:green", "Observed, converged")):
            selected = [(t, phase) for t, phase, state in zip(plots["motion_t"], plots["phase"], plots["converged"]) if phase is not None and state is converged]
            if selected:
                axes[1, 1].scatter([item[0] for item in selected], [item[1] for item in selected], s=3, color=color, label=label)

        def segmented(values: list[Any]) -> tuple[list[Any], list[Any]]:
            timestamps, separated = [], []
            for index, (timestamp, value) in enumerate(zip(plots["motion_t"], values)):
                if index and plots["reset"][index] != plots["reset"][index - 1]:
                    timestamps.append(timestamp)
                    separated.append(None)
                timestamps.append(timestamp)
                separated.append(value)
            return timestamps, separated

        if any(value is not None for value in plots["model_phase"]):
            model_times, model_values = segmented(plots["model_phase"])
            axes[1, 1].plot(model_times, model_values, linewidth=0.8, color="black", label="Calculated model")
        axes[1, 1].set_ylabel("Canonical phase within reset segment (rad)")
        axes[1, 1].legend(fontsize="small")
        if any(value is not None for value in plots["omega"]):
            velocity = axes[1, 1].twinx()
            velocity_times, velocity_values = segmented(plots["omega"])
            velocity.plot(velocity_times, velocity_values, color="tab:orange", linewidth=0.7)
            velocity.set_ylabel("Angular velocity (rad/s)", color="tab:orange")
            velocity.tick_params(axis="y", colors="tab:orange")
    else:
        axes[1, 1].text(0.5, 0.5, "No logged S5 phase", ha="center", va="center", transform=axes[1, 1].transAxes)
    for axis in axes.flat:
        axis.set_xlabel("Capture timestamp (s)")
        axis.grid(alpha=0.25)
    figure.suptitle("Real Buff observation and reconstruction diagnostics")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path, help="real capture or replay JSONL")
    parser.add_argument("--output", type=Path, help="save the full JSON report; stdout then contains a short summary")
    parser.add_argument("--video", type=Path, help="optional real AVI; decode and count actual frames")
    parser.add_argument("--imu", type=Path, help="optional synchronized 'seconds w x y z' text sidecar")
    parser.add_argument("--max-normal-step-deg", type=float, help="opt-in diagnostic limit on adjacent directed world normals")
    parser.add_argument("--max-motion-basis-change-deg", type=float, default=5.0, help="holdout filter on normal/phase-zero basis changes (default 5 degrees; does not itself fail validation)")
    parser.add_argument("--max-holdout-phase-p95-deg", type=float, help="opt-in limit on canonical 0.5 s phase P95; also requires evaluable pairs")
    parser.add_argument("--require-motion", action="store_true", help="fail unless at least one frame explicitly reports a converged motion model")
    parser.add_argument("--plot", type=Path, help="optional static PNG diagnostics (requires matplotlib)")
    args = parser.parse_args()
    for name in ("max_normal_step_deg", "max_motion_basis_change_deg", "max_holdout_phase_p95_deg"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0):
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
    try:
        rows = load_rows(args.jsonl)
        checks = Checks()
        report, plots = analyze(rows, checks)
        report["s5"]["holdout_0_5_s"] = holdout_motion(rows, args.max_motion_basis_change_deg)
        report = {"schema_version": 1, "input": str(args.jsonl), **report}
        if args.video or args.imu:
            report["recording"] = check_media(rows, args.video, args.imu, checks)
        if args.max_normal_step_deg is not None:
            maximum = report["s4"]["directed_normal_step_deg"]["max"]
            report["s4"]["requested_normal_step_limit_deg"] = args.max_normal_step_deg
            if maximum is not None and maximum > args.max_normal_step_deg:
                checks.fail("normal_step_limit", "S4", f"max={maximum:.6g} deg exceeds requested {args.max_normal_step_deg:g} deg")
        if args.require_motion and report["s5"]["explicitly_converged_frames"] == 0:
            checks.fail("motion_required", "S5", "no frame explicitly reports a converged motion model")
        if args.max_holdout_phase_p95_deg is not None:
            phase_error = report["s5"]["holdout_0_5_s"]["canonical_absolute_phase_error_deg"]["p95"]
            if phase_error is None:
                checks.fail("holdout_phase_required", "S5", "requested phase error limit requires evaluable 0.5 s future observations")
            elif phase_error > args.max_holdout_phase_p95_deg:
                checks.fail("holdout_phase_p95_limit", "S5", f"P95={phase_error:.6g} deg exceeds requested {args.max_holdout_phase_p95_deg:g} deg")
        report["checks"] = {"passed": not checks.counts, "failure_counts": dict(checks.counts), "examples": checks.examples, "policy": "hard data/geometry invariants; accuracy limits apply only when explicitly requested", "unit_tolerance": UNIT_TOLERANCE, "geometry_tolerance_m": GEOMETRY_TOLERANCE_M}
        if args.plot:
            plot_report(args.plot, plots, args.max_normal_step_deg)
            report["plot"] = str(args.plot)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            summary = {"passed": report["checks"]["passed"], "frames": report["frames"], "s4_success": report["s4"]["success_frames"], "s4_eligible": report["s4"]["eligible_frames_ge_2_blades_and_valid_imu"], "anchor_median_px": report["s4"]["anchor_reprojection_error_px"]["median"], "max_normal_step_deg": report["s4"]["directed_normal_step_deg"]["max"], "s5_converged": report["s5"]["explicitly_converged_frames"], "failure_counts": dict(checks.counts), "report": str(args.output)}
            print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
        else:
            print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return 0 if report["checks"]["passed"] else 1
    except (OSError, ValueError, RuntimeError) as error:
        print(f"buff_real_validation: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
