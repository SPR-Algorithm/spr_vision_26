#!/usr/bin/env python3
"""Build a local Buff TensorRT engine without running inference.

Example: python3 tools/build_buff_engine.py --precision=fp32 --dry-run
The source ONNX and existing engine are never modified by a failed build.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import ctypes.util
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_trtexec(explicit: str | None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    else:
        for variable in ("TensorRT_ROOT", "TENSORRT_ROOT"):
            if os.environ.get(variable):
                candidates.append(Path(os.environ[variable]) / "bin/trtexec")
        found = shutil.which("trtexec")
        if found:
            candidates.append(Path(found))
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return path.resolve()
    raise ValueError("Cannot find executable trtexec; pass --trtexec=/path/to/trtexec or set TensorRT_ROOT")


def version_integer(text: object) -> int | None:
    match = re.search(r"(\d+)\.(\d+)", str(text))
    return int(match[1]) * 1000 + int(match[2]) * 10 if match else None


def cuda_api_version(library: str, function: str) -> int | None:
    try:
        name = ctypes.util.find_library(library)
        if not name:
            return None
        handle = ctypes.CDLL(name)
        query = getattr(handle, function)
        query.argtypes = [ctypes.POINTER(ctypes.c_int)]
        query.restype = ctypes.c_int
        version = ctypes.c_int()
        return version.value if query(ctypes.byref(version)) == 0 else None
    except (OSError, AttributeError):
        return None


def tensorrt_version(trtexec: Path, build_log: str) -> str:
    # SDK bin/trtexec may resolve into targets/<architecture>/bin. Search its
    # ancestors rather than assuming the resolved binary's grandparent is the SDK.
    roots = list(trtexec.resolve().parents)
    roots.extend(Path(os.environ[name]) for name in ("TensorRT_ROOT", "TENSORRT_ROOT") if os.environ.get(name))
    for root in roots:
        header = root / "include/NvInferVersion.h"
        if not header.is_file():
            continue
        macros = dict(re.findall(r"^\s*#define\s+(\w+)\s+([A-Za-z_0-9]+)", header.read_text(encoding="utf-8", errors="replace"), re.MULTILINE))

        def resolve_macro(name: str) -> str | None:
            seen = set()
            while name in macros and name not in seen:
                seen.add(name)
                name = macros[name]
                if name.isdigit():
                    return name
            return None

        parts = [resolve_macro(f"NV_TENSORRT_{part}") for part in ("MAJOR", "MINOR", "PATCH", "BUILD")]
        if all(part is not None for part in parts):
            return ".".join(parts)
    encoded = re.search(r"TensorRT\s+v(\d{5,7})\]\s*\[b(\d+)\]", build_log)
    if encoded:
        version, build = int(encoded[1]), encoded[2]
        return f"{version // 10000}.{version % 10000 // 100}.{version % 100}.{build}"
    match = re.search(r"TensorRT(?:\s+version)?\s*[:v]?\s*(\d+\.\d+\.\d+(?:\.\d+)?)", build_log, re.IGNORECASE)
    return match[1] if match else "unknown"


def environment_metadata(trtexec: Path, device: int, build_log: str) -> dict:
    version = tensorrt_version(trtexec, build_log)
    gpu = {"name": None, "compute_capability": None, "device_index": device}
    driver_version = None
    try:
        completed = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,compute_cap,driver_version", "--format=csv,noheader,nounits"], check=False, capture_output=True, text=True, timeout=10)
        devices = list(csv.reader(completed.stdout.splitlines())) if completed.returncode == 0 else []
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        selected = visible[device].strip() if visible != [""] and device < len(visible) else str(device)
        for values in devices:
            if len(values) == 5:
                index, uuid, name, capability, driver = [value.strip() for value in values]
                if selected in (index, uuid):
                    gpu.update(name=name, compute_capability=capability)
                    driver_version = driver
                    break
    except (OSError, subprocess.TimeoutExpired):
        pass
    if gpu["name"] is None:
        match = re.search(r"Selected Device:\s*(.+)", build_log)
        gpu["name"] = match[1].strip() if match else None
    if gpu["compute_capability"] is None:
        match = re.search(r"Compute Capability:\s*(\d+)\.(\d+)", build_log)
        gpu["compute_capability"] = f"{match[1]}.{match[2]}" if match else None
    toolkit_version = None
    cuda_root = Path(os.environ.get("CUDA_HOME", os.environ.get("CUDA_PATH", "/usr/local/cuda")))
    toolkit_file = cuda_root / "version.json"
    if toolkit_file.is_file():
        try:
            toolkit = json.loads(toolkit_file.read_text())
            toolkit_version = version_integer(toolkit.get("cuda", {}).get("version"))
        except (OSError, ValueError, AttributeError):
            pass
    return {
        "trt_version": version,
        "gpu": gpu,
        "cuda": {"runtime_version": cuda_api_version("cudart", "cudaRuntimeGetVersion"), "driver_version": cuda_api_version("cuda", "cuDriverGetVersion"), "toolkit_version": toolkit_version, "nvidia_driver_version": driver_version},
    }


def publish_pair(engine_temporary: Path, metadata_temporary: Path, output: Path, metadata_path: Path) -> None:
    """Publish only complete artifacts, retaining rollback links until both renames succeed."""
    destinations = ((engine_temporary, output), (metadata_temporary, metadata_path))
    backups = []
    replaced = []
    try:
        for _, destination in destinations:
            if destination.exists():
                descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".backup", dir=destination.parent)
                os.close(descriptor)
                backup = Path(name)
                backup.unlink()
                os.link(destination, backup)
            else:
                backup = None
            backups.append((destination, backup))
        for source, destination in destinations:
            replaced.append(destination)
            os.replace(source, destination)
    except BaseException:
        for destination, backup in reversed(backups):
            if destination in replaced:
                if backup is None:
                    destination.unlink(missing_ok=True)
                else:
                    os.replace(backup, destination)
        raise
    finally:
        for _, backup in backups:
            if backup is not None:
                backup.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, default=Path("assets/buffer.onnx"))
    parser.add_argument("--output", type=Path, help="engine path (default build/buff_engines/buffer.<precision>.engine)")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--trtexec", help="explicit TensorRT trtexec executable")
    parser.add_argument("--device", type=int, default=0, help="logical CUDA device index (default 0)")
    parser.add_argument("--dry-run", action="store_true", help="print the planned command without creating any files or directories")
    args = parser.parse_args()
    if args.device < 0:
        parser.error("--device must be nonnegative")
    temporary_paths = []
    try:
        source = args.onnx.expanduser().resolve()
        if not source.is_file():
            raise ValueError(f"ONNX model does not exist: {source}")
        trtexec = find_trtexec(args.trtexec)
        output = (args.output or Path(f"build/buff_engines/buffer.{args.precision}.engine")).expanduser().resolve()
        metadata_path = Path(str(output) + ".json")
        if output == source or metadata_path == source:
            raise ValueError("Engine/metadata output must not overwrite the source ONNX")
        engine_temporary = output.with_name(f".{output.name}.BUILD_TMP")
        if not args.dry_run:
            output.parent.mkdir(parents=True, exist_ok=True)
            descriptor, name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
            os.close(descriptor)
            engine_temporary = Path(name)
            temporary_paths.append(engine_temporary)
        command = [str(trtexec), f"--onnx={source}", f"--saveEngine={engine_temporary}", f"--device={args.device}", "--skipInference", "--noTF32", "--inputIOFormats=fp32:chw", "--outputIOFormats=fp32:chw", "--memPoolSize=workspace:1024"]
        if args.precision == "fp16":
            command.append("--fp16")
        if args.dry_run:
            print(json.dumps({"dry_run": True, "precision": args.precision, "output": str(output), "metadata": str(metadata_path), "command": command}, indent=2))
            return 0
        source_hash = sha256(source)
        print(json.dumps({"building": str(output), "precision": args.precision, "command": command}), flush=True)
        lines = []
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace") as process:
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    lines.append(line)
                returncode = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
        if returncode != 0:
            raise RuntimeError(f"trtexec failed with exit code {returncode}; existing engine and metadata were preserved")
        if not engine_temporary.is_file() or engine_temporary.stat().st_size == 0:
            raise RuntimeError("trtexec returned success without a nonempty engine; existing output was preserved")
        if sha256(source) != source_hash:
            raise RuntimeError("Source ONNX changed during the build; refusing to publish this engine")
        metadata = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(), "precision": args.precision, "model_sha256": source_hash, "onnx_path": str(source), "engine_sha256": sha256(engine_temporary), "engine_path": str(output), "command": command, "trtexec_path": str(trtexec), "input_io_format": "fp32:chw", "output_io_format": "fp32:chw", "tf32_enabled": False, "workspace_mib": 1024, **environment_metadata(trtexec, args.device, "".join(lines))}
        descriptor, name = tempfile.mkstemp(prefix=f".{metadata_path.name}.", suffix=".tmp", dir=output.parent)
        metadata_temporary = Path(name)
        temporary_paths.append(metadata_temporary)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        publish_pair(engine_temporary, metadata_temporary, output, metadata_path)
        print(json.dumps({"built": str(output), "metadata": str(metadata_path), "precision": args.precision, "model_sha256": source_hash}))
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"build_buff_engine: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("build_buff_engine: interrupted; unpublished temporary files removed", file=sys.stderr)
        return 130
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
