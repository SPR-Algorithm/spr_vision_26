# Task 1 report: Stage 0 build baseline and unified entry

## Implementation

- Added the target-scoped `OPENCV_DISABLE_EIGEN_TENSOR_SUPPORT` definition to `auto_buff`, which propagates to its consumers.
- Added `BuffInput`, `BuffActivation`, and the sole public `BuffProcessor::process` output path. The placeholder validates image, confidence, quaternion, gimbal state, gimbal mode, ordered in-bounds non-degenerate points, and activation/mode compatibility. It emits only mode 0 or mode 1.
- Added reset handling for gimbal-mode transitions, timestamp regression, and a 500 ms tracking gap.
- Enforced the `VisionToGimbal` packed ABI as exactly 28 bytes and retained the default `SP`/`0xef` framing; invalid output is value-initialized, so all six motion floats are exactly zero.
- Moved both auto_buff debug programs to `processor.process(input)` followed by `gimbal.send(output)`. Removed `io::Command` and `auto_aim::Plan` as Aimer's public auto_buff output APIs.
- Corrected the gimbal Eigen include spelling so auto_buff debug targets do not mix `/usr/local` and system Eigen headers.

## TDD evidence

RED command:

```text
cmake --build build --target buff_processor_protocol_test -j2
```

Result: expected failure, `fatal error: tasks/auto_buff/buff_processor.hpp: No such file or directory`. The same invocation built the legacy `auto_buff` object successfully after the target-scoped Tensor-disable definition, isolating the missing public API as the RED condition.

GREEN command:

```text
cmake --build build --target buff_processor_protocol_test -j2 && ./build/buff_processor_protocol_test
```

Result: successful build and zero-exit protocol test. It verifies valid small/big inputs map to mode 1; conflicting or out-of-bounds inputs map to mode 0 with six zero motion floats; returned framing is `SP`/`0xef`; and `sizeof(io::VisionToGimbal) == 28`.

## Verification

```text
cmake --build build --target buff_processor_protocol_test auto_buff auto_buff_debug auto_buff_debug_mpc auto_buff_test buff_detector_test auto_aim_test gimbal_test standard -j2
./build/buff_processor_protocol_test
git diff --check
```

Result: every listed target built successfully, the protocol test exited zero, and the whitespace check passed. CMake reported the expected environment condition: ROS2 Jazzy is present, so Humble-specific targets were skipped by existing project logic.

## Self-review

- Confirmed no `io::Command`, `auto_aim::Plan`, `.aim(`, or `.mpc_aim(` remains in the auto_buff implementation, two debug callers, or the auto_buff diagnostic source.
- Confirmed none of the five protected vehicle YAML files was edited or staged.
- `clang-format` is unavailable in this environment (`clang-format: command not found`); source compiles with the repository's existing formatting conventions.

## Concerns

- The debug adapters currently use detector confidence `1.0F`, because the legacy `PowerRune` transport does not retain `YOLO11_BUFF::Object::prob`. Stage 1's promised standard five-point observation output should carry that real confidence instead.
