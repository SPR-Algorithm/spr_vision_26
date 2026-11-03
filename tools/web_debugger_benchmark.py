#!/usr/bin/env python3
"""Run bounded CLOSED-LOOP vehicle measurements, including normal control/fire output.

The executable owns the camera/serial port. This runner only creates passive WebSocket
viewers; it never changes tuning parameters. Run from the repository root.
"""
import argparse
import base64
import json
import math
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import threading
import time


class Viewer(threading.Thread):
    def __init__(self, port, path, stalled=False):
        super().__init__(daemon=True)
        self.port, self.path, self.stalled = port, path, stalled
        self.stop_event = threading.Event()
        self.received = 0
        self.connected = False
        self.error = None
        self.video_frames = 0
        self.telemetry_samples = 0
        self.valid_targets = 0
        self.control_samples = 0
        self.buffer = b""

    def parse_frames(self, chunk):
        self.buffer += chunk
        while len(self.buffer) >= 2:
            opcode = self.buffer[0] & 15
            length = self.buffer[1] & 127
            header = 2
            if length == 126:
                if len(self.buffer) < 4:
                    return
                length = struct.unpack("!H", self.buffer[2:4])[0]
                header = 4
            elif length == 127:
                if len(self.buffer) < 10:
                    return
                length = struct.unpack("!Q", self.buffer[2:10])[0]
                header = 10
            if length > 16 * 1024 * 1024:
                raise RuntimeError("oversized server frame")
            if len(self.buffer) < header + length:
                return
            payload = self.buffer[header:header + length]
            self.buffer = self.buffer[header + length:]
            if opcode == 2:
                self.video_frames += 1
            elif opcode == 1:
                message = json.loads(payload)
                if message.get("type") == "telemetry":
                    samples = message.get("samples", [])
                    self.telemetry_samples += len(samples)
                    self.valid_targets += sum(bool(s.get("target_valid")) for s in samples)
                    self.control_samples += sum(bool(s.get("control")) for s in samples)

    def run(self):
        sock = None
        try:
            deadline = time.monotonic() + 25
            while not self.stop_event.is_set() and time.monotonic() < deadline:
                try:
                    sock = socket.create_connection(("127.0.0.1", self.port), timeout=0.5)
                    break
                except OSError:
                    self.stop_event.wait(0.1)
            if sock is None:
                raise RuntimeError("WebSocket server did not become ready")
            sock.settimeout(0.2)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
            key = base64.b64encode(os.urandom(16)).decode()
            sock.sendall((f"GET {self.path} HTTP/1.1\r\nHost: localhost\r\n"
                          f"Upgrade: websocket\r\nConnection: Upgrade\r\n"
                          f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
            header = b""
            while b"\r\n\r\n" not in header and not self.stop_event.is_set():
                try:
                    chunk = sock.recv(65536)
                except socket.timeout:
                    continue
                if not chunk:
                    raise RuntimeError("server closed during handshake")
                header += chunk
            if b" 101 " not in header.split(b"\r\n", 1)[0]:
                raise RuntimeError("WebSocket upgrade failed")
            self.connected = True
            if self.stalled:
                self.stop_event.wait()
                return
            initial = header.split(b"\r\n\r\n", 1)[1]
            self.received += len(initial)
            self.parse_frames(initial)
            while not self.stop_event.is_set():
                try:
                    chunk = sock.recv(65536)
                except socket.timeout:
                    continue
                if not chunk:
                    break
                self.received += len(chunk)
                self.parse_frames(chunk)
        except (OSError, RuntimeError, ValueError) as exc:
            if not self.stop_event.is_set():
                self.error = str(exc)
        finally:
            if sock is not None:
                sock.close()


def process_sample(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().split(")", 1)[1].split()
        rss_kib = int(stat[21]) * os.sysconf("SC_PAGE_SIZE") / 1024
        cpu_seconds = (int(stat[11]) + int(stat[12])) / os.sysconf("SC_CLK_TCK")
        return time.monotonic(), cpu_seconds, rss_kib
    except (OSError, IndexError, ValueError):
        return None


def stop_process(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_case(args, mode, repeat):
    label = f"{args.label}_{mode}_{repeat}"
    output = args.output_dir / f"{label}.json"
    log = args.output_dir / f"{label}.log"
    viewers = []
    if mode in ("video", "both", "slow"):
        viewers.append(Viewer(args.port, "/ws"))
    if mode in ("telemetry", "both", "slow"):
        viewers.append(Viewer(args.port, "/ws/telemetry"))
    if mode == "slow":
        viewers.extend([Viewer(args.port, "/ws", True),
                        Viewer(args.port, "/ws/telemetry", True)])
    command = [str(args.binary), str(args.config), "--headless=true",
               f"--web={'off' if mode == 'off' else 'on'}",
               f"--benchmark-seconds={args.seconds}",
               f"--benchmark-warmup={args.warmup}",
               f"--benchmark-output={output}"]
    print(f"{label}: starting closed-loop measurement", flush=True)
    with log.open("w") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        for viewer in viewers:
            viewer.start()
        samples = []
        start = time.monotonic()
        timed_out = False
        try:
            while process.poll() is None:
                sample = process_sample(process.pid)
                if sample:
                    samples.append(sample)
                if time.monotonic() - start > args.warmup + args.seconds + 25:
                    timed_out = True
                    break
                time.sleep(0.2)
        finally:
            stop_process(process)
            for viewer in viewers:
                viewer.stop_event.set()
            for viewer in viewers:
                viewer.join(timeout=2)
    if timed_out or process.returncode != 0 or not output.exists():
        failure = {"label": label, "status": "failed", "timed_out": timed_out,
                   "returncode": process.returncode, "log": str(log), "command": command}
        output.write_text(json.dumps(failure, indent=2) + "\n")
        raise RuntimeError(f"{label} failed; inspect {log}")
    result = json.loads(output.read_text())
    # Linux steady_clock and Python monotonic share CLOCK_MONOTONIC. Keep only
    # process samples within the application's actual measurement interval.
    measured = [sample for sample in samples
                if result["measurement_start_monotonic"] <= sample[0]
                <= result["measurement_end_monotonic"]]
    resource = {}
    if len(measured) > 1:
        elapsed = measured[-1][0] - measured[0][0]
        resource = {"cpu_percent_one_core": 100 * (measured[-1][1] - measured[0][1]) / elapsed,
                    "rss_peak_mib": max(s[2] for s in measured) / 1024,
                    "scope": "200 ms samples within measurement interval; excludes viewer processes"}
    result.update({"label": args.label, "mode": mode, "repeat": repeat,
                   "process_resources": resource,
                   "viewers": [{"path": v.path, "stalled": v.stalled,
                                "connected": v.connected, "bytes": v.received,
                                "video_frames": v.video_frames,
                                "telemetry_samples": v.telemetry_samples,
                                "valid_targets": v.valid_targets,
                                "control_samples": v.control_samples,
                                "error": v.error} for v in viewers]})
    if mode == "slow":
        counters = result.get("debug_run_counters", {})
        result["backpressure_observed"] = {
            "video": counters.get("video_client_dropped", 0) > 0,
            "telemetry": counters.get("telemetry_client_dropped", 0) > 0}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{label}: {result['vision_fps']:.2f} fps", flush=True)
    if not result.get("measurement_complete") or not result.get("measured_frames"):
        raise RuntimeError(f"{label}: incomplete measurement or no camera frames")
    if any(not v.connected or v.error for v in viewers):
        raise RuntimeError(f"{label}: a viewer failed; inspect {output}")
    for viewer in viewers:
        if viewer.stalled:
            continue
        count = viewer.video_frames if viewer.path == "/ws" else viewer.telemetry_samples
        if not count:
            raise RuntimeError(f"{label}: no data on {viewer.path}; inspect {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("build/auto_aim_debug_mpc"))
    parser.add_argument("--config", type=Path, default=Path("configs/sentry.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="current")
    parser.add_argument("--modes", default="off,no_client,telemetry,both,slow")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--warmup", type=float, default=5)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    args.binary = args.binary.resolve()
    args.config = args.config.resolve()
    args.output_dir = args.output_dir.resolve()
    modes = args.modes.split(",")
    if (any(m not in {"off", "no_client", "video", "telemetry", "both", "slow"} for m in modes)
            or not math.isfinite(args.seconds) or not math.isfinite(args.warmup)
            or args.seconds <= 0 or args.warmup < 0 or args.repeats < 1):
        parser.error("invalid modes or measurement duration")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for repeat in range(1, args.repeats + 1):
        for mode in modes:
            run_case(args, mode, repeat)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, KeyboardInterrupt) as exc:
        raise SystemExit(str(exc))
