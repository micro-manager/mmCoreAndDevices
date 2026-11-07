# Linux build and validation - 2026-11-05

The Linux adapter is **experimental**. It was cross-compiled and statically
validated on Windows. It has not been loaded by Linux Micro-Manager or tested
with a camera under Linux. Existing camera regression results apply to the
Windows adapter. **macOS is not supported (Intel or Apple silicon).**

## Target and output

- Ubuntu 24.04, x86_64; baseline x86_64 CPU instructions.
- Emergent eSDK 4.08.05, supplied in the Ubuntu 24.04 / kernel 6.17.0 package.
- Micro-Manager Device API 75 / Module API 10.
- Output: `libmmgr_dal_EmergentCamera.so.0`, 221456 bytes.
- SHA-256: `7ABF2F44F403D07D0FA9C8106B7B204E7CF2B6EFF8C08508AFAF9A5DA2197646`.

Use the SDK's supported NIC, kernel and drivers on the deployment machine.
Install the binary alongside Micro-Manager's other `libmmgr_dal_*.so.0` adapters
and retain the `.so.0` suffix. Follow [the adapter README](../README.md) for runtime
library paths, configuration and a writable `Settings - File` location.

## Linux source changes

- Read negotiated host NIC speed from `/sys/class/net/<interface>/speed`.
  Link state is checked before reporting the value. An unavailable sysfs value
  is reported explicitly, rather than treated as image throughput.
- Locate `Emergent.ini` beside the loaded adapter using `dladdr`.
- Use a process-specific temporary settings filename before atomic rename.
- Add a standalone Linux CMake configuration that builds the adapter and
  MMDevice support code, with a versioned SONAME and `--no-undefined`.

## Native Linux rebuild

Install a C++17-capable GNU compiler, CMake, the matching Linux eSDK, and the
Micro-Manager source matching the intended runtime. From `DeviceAdapters/EmergentCamera` (the adapter root, not `src/`):

```bash
cmake -S . -B build/linux-x86_64 -DCMAKE_BUILD_TYPE=Release \
  -DMM_SOURCE_DIR=/path/to/mmCoreAndDevices \
  -DEVT_SDK_ROOT=/opt/EVT/eSDK
cmake --build build/linux-x86_64 --parallel
```

The output is `build/linux-x86_64/libmmgr_dal_EmergentCamera.so.0`. If the SDK will be
installed at a different location on the deployment machine, also set
`EVT_SDK_RUNTIME_ROOT` to that target path when configuring.

## Completed checks

- CMake configuration and Ninja build passed using Windows-hosted Zig 0.14.1
  / Clang 19.1.7, with Ubuntu 24.04 GNU C++ headers and runtime libraries.
- All nine adapter/MMDevice translation units compiled and linked.
- ELF64 x86_64 shared object; correct SONAME, runtime search path and all nine
  required Micro-Manager entry points verified.
- All 156 strong imports resolve against
  the original vendor SDK and Ubuntu libraries, including
  32 Emergent C++ imports.
- Dependency names contain Linux library names without Windows build paths.
- Adapter symbol requirements include GLIBC 2.34 and GLIBCXX 3.4.29; SDK
  dependencies can impose additional requirements, so this is not a claim of
  compatibility with older distributions.
- Windows Release x64 recompilation passed after the portability changes.
  The published Windows DLL is the existing hardware-tested binary.

The SDK DSOs omit SONAME. For cross-linking, private build copies were given
their normal Linux SONAMEs so Zig would not embed resolved Windows paths in
the adapter. Validation uses the original SDK exports. The package contains
the adapter source, build files, documentation, tests, and platform binaries,
without vendor SDK libraries.

Machine-readable verification is in [linux-verification.json](../tests/results/2026-11-05/linux-verification.json).
Portable build tools, checksummed dependency provenance and detailed logs are
retained in `S:\Projects\Emergent\EmergentCamera-linux`.

## Pending Linux validation

Load the adapter in Linux Micro-Manager; verify Snap, Live, finite sequences,
ROI/binning, pixel formats, settings save/restore, GPIO and PTP with the camera.
Physical trigger timing, synchronized PTP accuracy and sustained capture
throughput require the actual camera, NIC and deployment network.
