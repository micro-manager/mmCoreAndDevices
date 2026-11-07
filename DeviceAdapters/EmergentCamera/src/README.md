# Adapter source

| File | Purpose |
| --- | --- |
| `Module.cpp` | Micro-Manager module entry points and device registration |
| `EmergentCamera.h` | Camera device, property actions, and sequence-thread declarations |
| `EmergentCamera.cpp` | Camera control, properties, acquisition, GPIO/PTP, and platform network information |
| `EmergentSettings.cpp` | Per-camera INI persistence and platform-specific settings-file location |

These four implementation/header files come from the local project
associated with the 2026-11-05 validation. Only copyright and BSD-3-Clause
SPDX headers have been added; the runtime implementation is unchanged. The adapter root contains
the build definitions: `../EmergentCamera.vcxproj` for Windows and
`../CMakeLists.txt` for Linux. Both refer to this `src/` directory.

Micro-Manager headers/support code come from `MMDevice` in the containing
repository. Emergent headers and libraries come from a separately installed
classic eSDK. Neither dependency is duplicated in this directory.

See [Windows build instructions](../docs/BUILD_WINDOWS.md) and
[Linux build instructions](../docs/BUILD_LINUX.md).


## License and original author

Copyright (c) 2026, Dr. Volodymyr Cherkas. Licensed under
[BSD-3-Clause](../LICENSE). Retain the original copyright notice, license
conditions, and disclaimer in source redistributions, including public forks
and modified versions. See [NOTICE](../NOTICE).
