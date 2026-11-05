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
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any


UNIT_TOLERANCE = 1e-3
GEOMETRY_TOLERANCE_M = 1e-4
MAX_PAIR_GAP_S = 0.2
CONTOURS = ("armor_contour", "r_mark_contour", "light_arm_contour")
INFERENCE_TIMINGS = ("upload_ms", "preprocess_ms", "inference_ms", "download_ms", "postprocess_ms", "draw_ms", "total_ms")
COMPARISON_LIMITS = {
    "minimum_detection_matching_rate": 0.99,
    "maximum_five_point_p95_px": 1.0,
    "maximum_five_point_distance_px": 3.0,
    "maximum_s4_success_rate_drop_percentage_points": 1.0,
    "maximum_normal_step_deg": 20.0,
    "maximum_s5_holdout_p95_degradation_deg": 1.0,
}


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


def file_sha256(path: Path) -> str:
    return files_sha256([path])


def files_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def inference_report(rows: list[dict[str, Any]], checks: Checks) -> dict[str, Any]:
    timings = {key: [] for key in INFERENCE_TIMINGS}
    warmed_timings = {key: [] for key in INFERENCE_TIMINGS}
    configurations = {}
    warmup_iterations: Counter[str] = Counter()
    inferred = excluded = 0
    for row in rows:
        inference = row.get("inference")
        if inference is None:
            continue
        where = f"frame {row.get('frame')} inference"
        if not isinstance(inference, dict):
            checks.fail("inference_schema", where, "inference must be an object or null")
            continue
        configured_warmup = row.get("warmup_iterations", 0)
        warmup_iterations[str(configured_warmup)] += 1
        exclude = inference.get("warmup") is True or (inferred == 0 and not (type(configured_warmup) is int and configured_warmup > 0))
        inferred += 1
        excluded += exclude
        identity = {key: inference.get(key) for key in ("backend", "model_path", "device", "precision", "model_sha256", "input_shape", "output_shape")}
        serialized = json.dumps(identity, sort_keys=True)
        configurations.setdefault(serialized, {**identity, "frames": 0})["frames"] += 1
        for name in ("input_shape", "output_shape"):
            shape = inference.get(name)
            if shape is not None and (not isinstance(shape, list) or not shape or any(type(size) is not int or size <= 0 for size in shape)):
                checks.fail("inference_shape", where, f"{name} must contain positive integer dimensions")
        values = inference.get("timings", {})
        if not isinstance(values, dict):
            checks.fail("inference_timings", where, "timings must be an object")
            continue
        for key in INFERENCE_TIMINGS:
            value = values.get(key)
            if value is None:
                continue
            if not number(value) or value < 0:
                checks.fail("inference_timing_value", where, f"{key} must be null or a finite nonnegative duration")
                continue
            timings[key].append(value)
            if not exclude:
                warmed_timings[key].append(value)
    provenance = []
    for identity in configurations.values():
        path_text = identity.get("model_path")
        if not isinstance(path_text, str) or not path_text:
            checks.fail("inference_model_path", "inference", "model_path must identify the model artifact used for inference")
            continue
        path = Path(path_text)
        item = {"model_path": path_text, "model_exists": path.is_file(), "logged_model_sha256": identity.get("model_sha256")}
        logged_hash = identity.get("model_sha256")
        if not isinstance(logged_hash, str) or re.fullmatch(r"[0-9a-fA-F]{64}", logged_hash) is None:
            checks.fail("inference_model_hash_missing", path_text, "inference diagnostics require a nonempty 64-digit model SHA256")
        if path.is_file():
            item["actual_model_file_sha256"] = file_sha256(path)
            if path.suffix.lower() == ".xml":
                weights = path.with_suffix(".bin")
                item["model_hash_definition"] = "SHA256(XML bytes followed by BIN bytes)"
                item["weights_path"] = str(weights)
                if weights.is_file():
                    item["actual_model_sha256"] = files_sha256([path, weights])
                else:
                    checks.fail("inference_model_weights_missing", path_text, f"cannot verify aggregate IR model hash; missing {weights}")
            elif path.suffix.lower() == ".onnx":
                item["model_hash_definition"] = "SHA256(ONNX file bytes)"
                item["actual_model_sha256"] = item["actual_model_file_sha256"]
            if "actual_model_sha256" in item and isinstance(logged_hash, str) and item["actual_model_sha256"] != logged_hash.lower():
                checks.fail("inference_model_hash", path_text, "logged source-model SHA256 differs from the actual ONNX or XML||BIN model")
        else:
            checks.fail("inference_model_file_missing", path_text, "cannot verify logged model SHA256 because the model artifact is unavailable")
        manifest_path = Path(str(path) + ".json")
        if str(identity.get("backend", "")).lower() == "tensorrt" and manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                raise ValueError(f"{manifest_path}: engine manifest must be an object")
            item["manifest"] = str(manifest_path)
            item["manifest_precision"] = manifest.get("precision")
            item["manifest_model_sha256"] = manifest.get("model_sha256")
            engine_hash = manifest.get("engine_sha256")
            if engine_hash and item.get("actual_model_file_sha256") != engine_hash:
                checks.fail("engine_manifest_hash", path_text, "engine file SHA256 differs from its build manifest")
            if logged_hash and logged_hash != manifest.get("model_sha256"):
                checks.fail("engine_logged_model_hash", path_text, "logged source-model SHA256 differs from the build manifest")
            precision = identity.get("precision")
            if isinstance(precision, str) and precision.lower() in ("fp32", "fp16") and precision.lower() != manifest.get("precision"):
                checks.fail("engine_logged_precision", path_text, "logged engine precision differs from the build manifest")
            source = manifest.get("onnx_path")
            if isinstance(source, str) and Path(source).is_file():
                item["actual_onnx_sha256"] = file_sha256(Path(source))
                if item["actual_onnx_sha256"] != manifest.get("model_sha256"):
                    checks.fail("engine_source_model_hash", path_text, "source ONNX SHA256 differs from its engine manifest")
            else:
                item["source_onnx_available"] = False
        provenance.append(item)
    return {
        "frames": inferred,
        "configurations": list(configurations.values()),
        "warmup_iterations": dict(warmup_iterations),
        "timing_excluded_frames": excluded,
        "timing_included_frames": inferred - excluded,
        "warmup_exclusion_rule": "explicit inference.warmup frames; otherwise exclude only the first logged detector frame when no positive detector-only warmup_iterations was recorded; positive warmup runs occur before scoring rows",
        "timings_ms_all_logged_frames": {key: stats(values) for key, values in timings.items()},
        "timings_ms_excluding_warmup": {key: stats(values) for key, values in warmed_timings.items()},
        "timing_note": "nullable stage timings remain unavailable; backend stage boundaries may differ; total_ms is detector host wall time excluding drawing",
        "model_provenance": provenance,
    }


def config_snapshot_signature(path: Path, allow_backend_changes: bool) -> tuple[str, list[str]]:
    """Compare the YAML emitter's snapshot without needing a YAML dependency.

    All non-backend content, including scalar/sequence spelling and ordering,
    remains significant. A formatting-only change may need manual review; it
    can never silently hide a threshold or calibration change.
    """
    allowed = {"device", "model", "buff_inference.backend", "buff_inference.model", "buff_inference.engine", "buff_inference.device", "buff_inference.cuda_device"}
    stack = []
    retained, excluded = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = re.match(r"^(\s*)([A-Za-z_][A-Za-z_0-9]*):\s*(.*)$", line)
        if match:
            indentation, key, value = len(match[1]), match[2], match[3]
            while stack and stack[-1][0] >= indentation:
                stack.pop()
            key_path = ".".join([item[1] for item in stack] + [key])
            if not value:
                stack.append((indentation, key))
                continue
            if allow_backend_changes and key_path in allowed:
                excluded.append(key_path)
                continue
            retained.append((key_path, value))
        else:
            retained.append((".".join(item[1] for item in stack), line.strip()))
    serialized = json.dumps(retained, ensure_ascii=False)
    return hashlib.sha256(serialized.encode()).hexdigest(), excluded


def comparison_identity(candidate: list[dict[str, Any]], baseline: list[dict[str, Any]], checks: Checks) -> dict[str, Any]:
    result = {"timestamp_tolerance_s": 1e-9, "quaternion_component_tolerance": 1e-10, "absolute_steady_epochs_compared": False}
    for key, allow_backend in (("camera_config_snapshot", True), ("buff_config_snapshot", False)):
        sets = []
        descriptions = []
        for label, rows in (("candidate", candidate), ("baseline", baseline)):
            paths = {row.get(key) for row in rows if isinstance(row.get(key), str)}
            signatures, details = set(), []
            if len(paths) == 0 or any(not row.get(key) for row in rows):
                checks.fail("comparison_config_snapshot_missing", label, f"immutable {key} is required for A/B configuration checks")
            for text in sorted(paths):
                path = Path(text)
                if not path.is_file():
                    checks.fail("comparison_config_snapshot_missing", label, f"cannot read immutable snapshot {path}")
                    continue
                signature, excluded = config_snapshot_signature(path, allow_backend)
                signatures.add(signature)
                details.append({"path": text, "comparison_sha256": signature, "backend_keys_excluded": excluded})
            sets.append(signatures)
            descriptions.append({"side": label, "snapshots": details})
        equal = bool(sets[0]) and sets[0] == sets[1]
        result[key] = {"equal_after_declared_backend_exclusions": equal, "sides": descriptions}
        if sets[0] and sets[1] and not equal:
            checks.fail("comparison_configuration_changed", key, "threshold/calibration snapshot content differs beyond declared backend/model/device keys")
    inputs = []
    for label, rows in (("candidate", candidate), ("baseline", baseline)):
        names = {row.get("input") for row in rows if isinstance(row.get("input"), str)}
        if len(names) != 1 or "camera" in names:
            checks.fail("comparison_replay_input", label, "A/B requires one explicit offline recording in each log")
        text = next(iter(names)) if len(names) == 1 else None
        path = Path(text).resolve() if text and text != "camera" else None
        inputs.append(path)
    if all(path is not None for path in inputs):
        equal = inputs[0] == inputs[1]
        hashes = []
        if not equal and all(path.is_file() for path in inputs):
            hashes = [file_sha256(path) for path in inputs]
            equal = hashes[0] == hashes[1]
        result["recording"] = {"candidate": str(inputs[0]), "baseline": str(inputs[1]), "same_recording": equal, "content_sha256_when_paths_differ": hashes}
        if not equal:
            checks.fail("comparison_recording_mismatch", "A/B", "logs refer to different recording paths/content")
    return result


def frame_map(rows: list[dict[str, Any]], label: str, checks: Checks) -> dict[int, dict[str, Any]]:
    mapped = {}
    for row in rows:
        index = row.get("input_frame_index")
        if type(index) is not int or index < 0:
            checks.fail("comparison_input_frame_index", label, "strict A/B alignment requires a nonnegative explicit input_frame_index on every row")
        elif index in mapped:
            checks.fail("comparison_duplicate_input_frame", label, f"input_frame_index={index} occurs twice")
        else:
            mapped[index] = row
    return mapped


def detector_objects(row: dict[str, Any], label: str, checks: Checks) -> list[dict[str, Any]]:
    objects = []
    for index, observation in enumerate(row.get("observations", [])):
        raw = observation.get("raw") if isinstance(observation, dict) else None
        if not isinstance(raw, dict) or not isinstance(raw.get("points"), list) or len(raw["points"]) != 5 or not all(vector(point, 2) for point in raw["points"]):
            checks.fail("comparison_raw_points", label, f"frame {row.get('input_frame_index')} observation {index} needs five finite raw points")
            continue
        category = raw.get("class_id", raw.get("activation"))
        if category is None or not number(raw.get("confidence")):
            checks.fail("comparison_raw_class_confidence", label, "raw class and finite confidence are required")
            continue
        objects.append({"points": raw["points"], "category": str(category), "confidence": raw["confidence"], "index": index})
    return objects


def five_point_distances(left: dict[str, Any], right: dict[str, Any]) -> list[float]:
    return [math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(left["points"], right["points"])]


def minimum_distance_pairs(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Maximum-cardinality, minimum-total-five-point-distance assignment."""
    if not left or not right:
        return []
    if len(left) > len(right):
        return [(j, i) for i, j in minimum_distance_pairs(right, left)]
    if len(right) > 12:
        raise ValueError("A/B assignment supports at most 12 detections per frame/class; inspect the unexpectedly large detector output")
    costs = [[sum(five_point_distances(a, b)) / 5 for b in right] for a in left]

    @lru_cache(maxsize=None)
    def assign(index: int, used: int) -> tuple[float, tuple[tuple[int, int], ...]]:
        if index == len(left):
            return 0.0, ()
        best = (math.inf, ())
        for other in range(len(right)):
            if not (used & (1 << other)):
                cost, pairs = assign(index + 1, used | (1 << other))
                candidate = (costs[index][other] + cost, ((index, other),) + pairs)
                if candidate[0] < best[0]:
                    best = candidate
        return best

    return list(assign(0, 0)[1])


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
    evaluated_errors = []
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
        evaluated_errors.append({"input_frame_index": original_frame_index(row), "absolute_error_deg": math.degrees(abs(signed_error)), "future_matching": method})
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
        "evaluated_errors_by_input_frame": evaluated_errors,
        "canonical_definition": "observed or predicted continuous phase minus that frame's cumulative_switch_offset_rad; primary residual is not wrapped by blade spacing",
        "matching_rule": "0.5 s future observed phase, interpolation across <=0.2 s accepted gaps or nearest within 0.05 s; same reset ID and basis within the declared limit throughout the interval",
        "identity_note": "phase forecast diagnostics without independent blade-identity ground truth; fivefold equivalence is an auxiliary metric, not a physical target hit test",
    }


def compare_reports(candidate_rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]], candidate_report: dict[str, Any], baseline_report: dict[str, Any], checks: Checks, enforce: bool) -> dict[str, Any]:
    identity = comparison_identity(candidate_rows, baseline_rows, checks)
    candidate = frame_map(candidate_rows, "candidate", checks)
    baseline = frame_map(baseline_rows, "baseline", checks)
    indices = sorted(candidate.keys() & baseline.keys())
    missing_frames = sorted(baseline.keys() - candidate.keys())
    additional_frames = sorted(candidate.keys() - baseline.keys())
    if missing_frames or additional_frames or not indices:
        checks.fail("comparison_frame_coverage", "A/B", f"aligned={len(indices)}, missing candidate={len(missing_frames)}, extra candidate={len(additional_frames)}; identical frame coverage is required")
    baseline_count = candidate_count = matched_count = class_mismatches = 0
    baseline_classes: Counter[str] = Counter()
    candidate_classes: Counter[str] = Counter()
    distances, confidence_differences, confidence_signed = [], [], []
    point_distances = [[] for _ in range(5)]
    delta_examples = []
    frame_detection_changes = []
    success_changes = []
    paired_anchor_deltas = []
    paired_center_distances = []
    paired_normal_angles = []
    pose_source_pairs: Counter[str] = Counter()
    common_pose_frames = 0
    for index in indices:
        new, old = candidate[index], baseline[index]
        new_time, old_time = new.get("timestamp_relative_s"), old.get("timestamp_relative_s")
        if not number(new_time) or not number(old_time) or abs(new_time - old_time) > identity["timestamp_tolerance_s"]:
            checks.fail("comparison_capture_timestamp", f"input frame {index}", f"relative capture timestamps differ: candidate={new_time}, baseline={old_time}")
        new_q, old_q = new.get("q_wxyz"), old.get("q_wxyz")
        if not vector(new_q, 4) or not vector(old_q, 4) or max(abs(a - b) for a, b in zip(new_q, old_q)) > identity["quaternion_component_tolerance"]:
            checks.fail("comparison_capture_quaternion", f"input frame {index}", "original wxyz quaternion components differ")
        new_mode = new.get("mode") if isinstance(new.get("mode"), dict) else {}
        old_mode = old.get("mode") if isinstance(old.get("mode"), dict) else {}
        if new_mode.get("effective") != old_mode.get("effective") or new.get("pose_prior_enabled") != old.get("pose_prior_enabled"):
            checks.fail("comparison_pipeline_settings", f"input frame {index}", "effective Buff mode or temporal-pose-prior setting differs")
        new_objects = detector_objects(new, "candidate", checks)
        old_objects = detector_objects(old, "baseline", checks)
        candidate_count += len(new_objects)
        baseline_count += len(old_objects)
        candidate_classes.update(item["category"] for item in new_objects)
        baseline_classes.update(item["category"] for item in old_objects)
        frame_matched = 0
        new_matched_indices, old_matched_indices = set(), set()
        frame_pairs = []
        for category in {item["category"] for item in new_objects + old_objects}:
            new_class = [item for item in new_objects if item["category"] == category]
            old_class = [item for item in old_objects if item["category"] == category]
            pairs = minimum_distance_pairs(new_class, old_class)
            frame_matched += len(pairs)
            for left, right in pairs:
                new_object, old_object = new_class[left], old_class[right]
                new_matched_indices.add(new_object["index"])
                old_matched_indices.add(old_object["index"])
                errors = five_point_distances(new_object, old_object)
                frame_pairs.append({"category": category, "candidate_observation_index": new_object["index"], "baseline_observation_index": old_object["index"], "five_point_distances_px": errors, "confidence_candidate_minus_baseline": new_object["confidence"] - old_object["confidence"]})
                distances.extend(errors)
                for values, error in zip(point_distances, errors):
                    values.append(error)
                difference = new_object["confidence"] - old_object["confidence"]
                confidence_signed.append(difference)
                confidence_differences.append(abs(difference))
        matched_count += frame_matched
        frame_detection_changes.append({"input_frame_index": index, "baseline_detection_count": len(old_objects), "candidate_detection_count": len(new_objects), "matched_same_class": frame_matched, "matches": frame_pairs, "missing_in_candidate": [item for item in old_objects if item["index"] not in old_matched_indices], "additional_in_candidate": [item for item in new_objects if item["index"] not in new_matched_indices]})
        old_reconstruction, new_reconstruction = old.get("reconstruction"), new.get("reconstruction")
        old_pose = old_reconstruction.get("pose") if isinstance(old_reconstruction, dict) else None
        new_pose = new_reconstruction.get("pose") if isinstance(new_reconstruction, dict) else None
        if isinstance(old_pose, dict) != isinstance(new_pose, dict):
            success_changes.append({"input_frame_index": index, "baseline_success": isinstance(old_pose, dict), "candidate_success": isinstance(new_pose, dict), "baseline_status": old.get("reconstruction_status"), "candidate_status": new.get("reconstruction_status")})
        if isinstance(old_pose, dict) and isinstance(new_pose, dict):
            common_pose_frames += 1
            pose_source_pairs[f"{old_pose.get('source')} -> {new_pose.get('source')}"] += 1
            if number(old_pose.get("reprojection_error_px")) and number(new_pose.get("reprojection_error_px")):
                paired_anchor_deltas.append(new_pose["reprojection_error_px"] - old_pose["reprojection_error_px"])
            if vector(old_pose.get("position_world_m")) and vector(new_pose.get("position_world_m")):
                paired_center_distances.append(norm([a - b for a, b in zip(new_pose["position_world_m"], old_pose["position_world_m"])]))
            if vector(old_pose.get("plane_normal_world")) and vector(new_pose.get("plane_normal_world")) and norm(old_pose["plane_normal_world"]) > 0 and norm(new_pose["plane_normal_world"]) > 0:
                paired_normal_angles.append(angle_deg(new_pose["plane_normal_world"], old_pose["plane_normal_world"]))
        # This unconstrained assignment only describes potential class changes;
        # it never contributes to the same-class matching score above.
        class_mismatches += sum(new_objects[left]["category"] != old_objects[right]["category"] for left, right in minimum_distance_pairs(new_objects, old_objects))
        if frame_matched != len(new_objects) or frame_matched != len(old_objects):
            if len(delta_examples) < 20:
                delta_examples.append({"input_frame_index": index, "baseline_detections": len(old_objects), "candidate_detections": len(new_objects), "matched_same_class": frame_matched})
    for label, extra_indices, mapped in (("candidate", additional_frames, candidate), ("baseline", missing_frames, baseline)):
        for index in extra_indices:
            objects = detector_objects(mapped[index], label, checks)
            frame_detection_changes.append({"input_frame_index": index, "unaligned_side": label, "baseline_detection_count": len(objects) if label == "baseline" else 0, "candidate_detection_count": len(objects) if label == "candidate" else 0, "matched_same_class": 0, "matches": [], "missing_in_candidate": objects if label == "baseline" else [], "additional_in_candidate": objects if label == "candidate" else []})
    recall = matched_count / baseline_count if baseline_count else (1.0 if candidate_count == 0 else 0.0)
    precision = matched_count / candidate_count if candidate_count else (1.0 if baseline_count == 0 else 0.0)
    matching_rate = min(recall, precision)
    distance_statistics = stats(distances)
    baseline_success = sum(isinstance((baseline[index].get("reconstruction") or {}).get("pose"), dict) for index in indices)
    candidate_success = sum(isinstance((candidate[index].get("reconstruction") or {}).get("pose"), dict) for index in indices)
    success_drop = 100 * (baseline_success - candidate_success) / len(indices) if indices else None
    baseline_holdout = baseline_report["s5"]["holdout_0_5_s"]
    candidate_holdout = candidate_report["s5"]["holdout_0_5_s"]
    baseline_errors = {item["input_frame_index"]: item["absolute_error_deg"] for item in baseline_holdout["evaluated_errors_by_input_frame"]}
    candidate_errors = {item["input_frame_index"]: item["absolute_error_deg"] for item in candidate_holdout["evaluated_errors_by_input_frame"]}
    common_motion = sorted(baseline_errors.keys() & candidate_errors.keys())
    baseline_common = stats([baseline_errors[index] for index in common_motion])
    candidate_common = stats([candidate_errors[index] for index in common_motion])
    common_degradation = candidate_common["p95"] - baseline_common["p95"] if common_motion else None
    all_baseline_p95 = baseline_holdout["canonical_absolute_phase_error_deg"]["p95"]
    all_candidate_p95 = candidate_holdout["canonical_absolute_phase_error_deg"]["p95"]
    full_degradation = all_candidate_p95 - all_baseline_p95 if all_baseline_p95 is not None and all_candidate_p95 is not None else None
    gates = []

    def gate(name: str, measured: float | None, limit: float, relation: str, reason: str = "") -> None:
        passed = None if measured is None else (measured >= limit if relation == ">=" else measured <= limit)
        gates.append({"criterion": name, "measured": measured, "relation": relation, "limit": limit, "passed": passed, "not_assessed_reason": reason if measured is None else ""})
        if enforce and passed is False:
            checks.fail(f"comparison_{name}", "A/B engineering gate", f"measured={measured:.9g}, required {relation} {limit:g}")

    gate("detection_matching_rate", matching_rate, COMPARISON_LIMITS["minimum_detection_matching_rate"], ">=")
    gate("five_point_p95_px", distance_statistics["p95"], COMPARISON_LIMITS["maximum_five_point_p95_px"], "<=", "no same-class matched detections")
    gate("five_point_max_px", distance_statistics["max"], COMPARISON_LIMITS["maximum_five_point_distance_px"], "<=", "no same-class matched detections")
    gate("s4_success_rate_drop_percentage_points", success_drop, COMPARISON_LIMITS["maximum_s4_success_rate_drop_percentage_points"], "<=", "no aligned input frames")
    gate("normal_step_deg", candidate_report["s4"]["directed_normal_step_deg"]["max"], COMPARISON_LIMITS["maximum_normal_step_deg"], "<=", "no consecutive reconstructed original input frames")
    gate("s5_holdout_p95_degradation_deg_common_frames", common_degradation, COMPARISON_LIMITS["maximum_s5_holdout_p95_degradation_deg"], "<=", "no common evaluable 0.5 s holdout frames")
    gate("s5_holdout_p95_degradation_deg_all_eligible", full_degradation, COMPARISON_LIMITS["maximum_s5_holdout_p95_degradation_deg"], "<=", "one or both logs have no evaluable holdout pairs; selection fractions are reported")
    if enforce and baseline_errors and not common_motion:
        checks.fail("comparison_s5_holdout_coverage", "A/B", "baseline has evaluable motion holdouts but candidate has none on the same original input frames")

    def count_changes(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
        return {key: {"baseline": old.get(key, 0), "candidate": new.get(key, 0), "candidate_minus_baseline": new.get(key, 0) - old.get(key, 0)} for key in sorted(old.keys() | new.keys())}

    def metric_comparison(key: str) -> dict[str, Any]:
        old, new = baseline_report["s4"][key], candidate_report["s4"][key]
        return {"baseline": old, "candidate": new, "candidate_minus_baseline": {measure: new[measure] - old[measure] if number(new.get(measure)) and number(old.get(measure)) else None for measure in ("count", "median", "p95", "max")}}

    contour_comparison = {}
    for name in CONTOURS:
        old, new = baseline_report["s3"]["contours"][name], candidate_report["s3"]["contours"][name]
        contour_comparison[name] = {"baseline": old, "candidate": new, "usable_count_delta": new["usable"] - old["usable"], "reported_count_delta": new["reported"] - old["reported"], "baseline_usable_fraction": old["usable"] / old["reported"] if old["reported"] else None, "candidate_usable_fraction": new["usable"] / new["reported"] if new["reported"] else None, "reject_reason_changes": count_changes(old["reject_reason"], new["reject_reason"])}
    return {
        "baseline_input": baseline_report["input"],
        "report_only": not enforce,
        "default_engineering_limits": COMPARISON_LIMITS,
        "engineering_gates_enforced": enforce,
        "engineering_gates": gates,
        "all_assessed_engineering_gates_passed": all(item["passed"] is not False for item in gates),
        "unassessed_engineering_gates": [item["criterion"] for item in gates if item["passed"] is None],
        "identity_and_configuration": identity,
        "alignment": {"candidate_frames": len(candidate_rows), "baseline_frames": len(baseline_rows), "aligned_input_frames": len(indices), "missing_candidate_input_frames": missing_frames, "additional_candidate_input_frames": additional_frames},
        "detections": {"baseline": baseline_count, "candidate": candidate_count, "matched_same_class": matched_count, "missing_in_candidate": baseline_count - matched_count, "additional_in_candidate": candidate_count - matched_count, "baseline_recall": recall, "candidate_precision": precision, "matching_rate_minimum_recall_precision": matching_rate, "class_counts_baseline": dict(baseline_classes), "class_counts_candidate": dict(candidate_classes), "class_mismatches_spatial_assignment_auxiliary": class_mismatches, "confidence_absolute_delta": stats(confidence_differences), "confidence_signed_candidate_minus_baseline": stats(confidence_signed), "all_five_point_distance_px": distance_statistics, "per_point_distance_px": {name: stats(values) for name, values in zip(("top", "right", "bottom", "left", "R"), point_distances)}, "mismatch_examples": delta_examples, "all_frame_detection_changes": sorted(frame_detection_changes, key=lambda item: item["input_frame_index"]), "frame_details_complete": True, "assignment": "per original frame and identical raw class; maximum-cardinality one-to-one assignment minimizing mean Euclidean distance across all five ordered raw points; physical blade identity is not inferred"},
        "s3": {"baseline_observations": baseline_report["s3"]["observations"], "candidate_observations": candidate_report["s3"]["observations"], "quality_changes": count_changes(baseline_report["s3"]["quality"], candidate_report["s3"]["quality"]), "reject_reason_changes": count_changes(baseline_report["s3"]["reject_reason"], candidate_report["s3"]["reject_reason"]), "geometry_reject_reason_changes": count_changes(baseline_report["s3"]["geometry_reject_reason"], candidate_report["s3"]["geometry_reject_reason"]), "state_reject_reason_changes": count_changes(baseline_report["s3"]["state_reject_reason"], candidate_report["s3"]["state_reject_reason"]), "contours": contour_comparison},
        "s4": {"denominator": "all identical aligned input frames, so detector losses cannot disappear from the denominator", "baseline_success": baseline_success, "candidate_success": candidate_success, "baseline_success_rate": baseline_success / len(indices) if indices else None, "candidate_success_rate": candidate_success / len(indices) if indices else None, "success_rate_drop_percentage_points": success_drop, "baseline_eligible_frames": baseline_report["s4"]["eligible_frames_ge_2_blades_and_valid_imu"], "candidate_eligible_frames": candidate_report["s4"]["eligible_frames_ge_2_blades_and_valid_imu"], "baseline_success_rate_on_eligible": baseline_report["s4"]["success_rate_on_eligible"], "candidate_success_rate_on_eligible": candidate_report["s4"]["success_rate_on_eligible"], "anchor_reprojection_error_px": metric_comparison("anchor_reprojection_error_px"), "pnp_initial_anchor_error_px": metric_comparison("pnp_initial_anchor_error_px"), "chamfer_error_px": metric_comparison("chamfer_error_px"), "contour_sample_count": metric_comparison("contour_sample_count"), "pose_source_changes": count_changes(baseline_report["s4"]["pose_source"], candidate_report["s4"]["pose_source"]), "baseline_normal_step_deg": baseline_report["s4"]["directed_normal_step_deg"], "candidate_normal_step_deg": candidate_report["s4"]["directed_normal_step_deg"], "all_success_changes": success_changes, "common_pose_frames": common_pose_frames, "common_pose_source_pairs": dict(pose_source_pairs), "common_frame_anchor_candidate_minus_baseline_px": stats(paired_anchor_deltas), "common_frame_center_distance_m": stats(paired_center_distances), "common_frame_normal_angle_deg": stats(paired_normal_angles)},
        "s5": {"baseline_converged_frames": baseline_report["s5"]["explicitly_converged_frames"], "candidate_converged_frames": candidate_report["s5"]["explicitly_converged_frames"], "baseline_holdout_pairs": len(baseline_errors), "candidate_holdout_pairs": len(candidate_errors), "baseline_selection_fraction_of_converged": len(baseline_errors) / baseline_report["s5"]["explicitly_converged_frames"] if baseline_report["s5"]["explicitly_converged_frames"] else None, "candidate_selection_fraction_of_converged": len(candidate_errors) / candidate_report["s5"]["explicitly_converged_frames"] if candidate_report["s5"]["explicitly_converged_frames"] else None, "baseline_rejected_candidates": baseline_holdout["rejected_candidates"], "candidate_rejected_candidates": candidate_holdout["rejected_candidates"], "baseline_basis_limit_deg": baseline_holdout["maximum_basis_change_deg"], "candidate_basis_limit_deg": candidate_holdout["maximum_basis_change_deg"], "common_input_frame_pairs": len(common_motion), "baseline_common_frame_error_deg": baseline_common, "candidate_common_frame_error_deg": candidate_common, "common_frame_p95_degradation_deg": common_degradation, "baseline_all_selected_error_deg": baseline_holdout["canonical_absolute_phase_error_deg"], "candidate_all_selected_error_deg": candidate_holdout["canonical_absolute_phase_error_deg"], "all_selected_p95_degradation_deg": full_degradation},
        "baseline_inference": baseline_report.get("inference", {}),
        "report_only_note": "report-only disables only engineering comparison gates; both logs' hard invariants and recording/configuration/time/quaternion alignment are still enforced",
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
    parser.add_argument("--compare", type=Path, help="baseline real replay JSONL; positional JSONL is the candidate; enforce declared A/B limits")
    parser.add_argument("--comparison-report-only", action="store_true", help="report A/B engineering differences without enforcing their limits; data/configuration/alignment invariants still apply")
    parser.add_argument("--video", type=Path, help="optional real AVI; decode and count actual frames")
    parser.add_argument("--imu", type=Path, help="optional synchronized 'seconds w x y z' text sidecar")
    parser.add_argument("--max-normal-step-deg", type=float, help="opt-in diagnostic limit on adjacent directed world normals")
    parser.add_argument("--max-motion-basis-change-deg", type=float, default=5.0, help="holdout filter on normal/phase-zero basis changes (default 5 degrees; does not itself fail validation)")
    parser.add_argument("--max-holdout-phase-p95-deg", type=float, help="opt-in limit on canonical 0.5 s phase P95; also requires evaluable pairs")
    parser.add_argument("--require-motion", action="store_true", help="fail unless at least one frame explicitly reports a converged motion model")
    parser.add_argument("--plot", type=Path, help="optional static PNG diagnostics (requires matplotlib)")
    args = parser.parse_args()
    if args.comparison_report_only and args.compare is None:
        parser.error("--comparison-report-only requires --compare")
    for name in ("max_normal_step_deg", "max_motion_basis_change_deg", "max_holdout_phase_p95_deg"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0):
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
    try:
        rows = load_rows(args.jsonl)
        checks = Checks()
        report, plots = analyze(rows, checks)
        report["s5"]["holdout_0_5_s"] = holdout_motion(rows, args.max_motion_basis_change_deg)
        report["inference"] = inference_report(rows, checks)
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
        if args.compare:
            baseline_rows = load_rows(args.compare)
            baseline_checks = Checks()
            baseline_report, _ = analyze(baseline_rows, baseline_checks)
            baseline_report["s5"]["holdout_0_5_s"] = holdout_motion(baseline_rows, args.max_motion_basis_change_deg)
            baseline_report["inference"] = inference_report(baseline_rows, baseline_checks)
            baseline_report["input"] = str(args.compare)
            for code, count in baseline_checks.counts.items():
                checks.counts[f"baseline_{code}"] += count
            for example in baseline_checks.examples:
                if len(checks.examples) < 20:
                    checks.examples.append({**example, "location": "baseline " + example["location"]})
            report["comparison"] = compare_reports(rows, baseline_rows, report, baseline_report, checks, enforce=not args.comparison_report_only)
            report["comparison"]["baseline_hard_invariants"] = {"passed": not baseline_checks.counts, "failure_counts": dict(baseline_checks.counts)}
        report["checks"] = {"passed": not checks.counts, "failure_counts": dict(checks.counts), "examples": checks.examples, "policy": "hard data/geometry invariants plus explicit single-run accuracy limits; --compare also enforces its unchanged declared engineering limits unless --comparison-report-only", "unit_tolerance": UNIT_TOLERANCE, "geometry_tolerance_m": GEOMETRY_TOLERANCE_M}
        if args.plot:
            plot_report(args.plot, plots, args.max_normal_step_deg)
            report["plot"] = str(args.plot)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            summary = {"passed": report["checks"]["passed"], "frames": report["frames"], "s4_success": report["s4"]["success_frames"], "s4_eligible": report["s4"]["eligible_frames_ge_2_blades_and_valid_imu"], "anchor_median_px": report["s4"]["anchor_reprojection_error_px"]["median"], "max_normal_step_deg": report["s4"]["directed_normal_step_deg"]["max"], "s5_converged": report["s5"]["explicitly_converged_frames"], "failure_counts": dict(checks.counts), "report": str(args.output)}
            if "comparison" in report:
                comparison = report["comparison"]
                summary["comparison"] = {"report_only": comparison["report_only"], "matching_rate": comparison["detections"]["matching_rate_minimum_recall_precision"], "five_point_p95_px": comparison["detections"]["all_five_point_distance_px"]["p95"], "s4_success_drop_percentage_points": comparison["s4"]["success_rate_drop_percentage_points"], "common_s5_pairs": comparison["s5"]["common_input_frame_pairs"], "s5_common_p95_degradation_deg": comparison["s5"]["common_frame_p95_degradation_deg"]}
            print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
        else:
            print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return 0 if report["checks"]["passed"] else 1
    except (OSError, ValueError, RuntimeError) as error:
        print(f"buff_real_validation: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
