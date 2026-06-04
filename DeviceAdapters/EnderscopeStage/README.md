# EnderscopeStage Micro-Manager Adapter

| Summary: | Interfaces with Enderscope/Marlin-compatible motion controllers over serial G-code |
| --- | --- |
| Author: | Jerome Mutterer and Erwan Grandgirard |
| License: | Same as the hosting Micro-Manager build/distribution |
| Platforms: | Should work on all platforms (serial interface) |
| Devices: | EnderscopeXYStage, EnderscopeZStage |
| Since version: | Local prototype (2026-05) |
| Serial port settings: | Typical: `115200 8N1`, handshaking off |

This directory contains a new Micro-Manager device adapter that follows the structure of `Marzhauser-LStep/LStep.cpp` while targeting the `Stage` behavior in `enderscope.py` (Marlin-like G-code transport).

## Devices Exported

- `EnderscopeXYStage` (`MM::XYStageDevice`)
- `EnderscopeZStage` (`MM::StageDevice`)

## Command Mapping to `enderscope.Stage`

- `SetPositionUm(...)` -> `G90` then `G0 ...` then `M400`
- `SetRelativePositionUm(...)` -> `G91` then `G0 ...` then `M400`
- `GetPosition...` -> `M114` (parses `X:.. Y:.. Z:..`)
- `Home()` -> `G28 X Y` (XY stage) / `G28 Z` (Z stage) then `M400`
- `Stop()` -> `M410`

All stage coordinates are converted between Micro-Manager um and Enderscope mm.

## Pre-initialization Properties

- `Port` (serial port)
- `ReadTimeoutMs` (default `1000`)

The serial baud rate is configured on the COM port device itself (its own
`BaudRate` property), not on the stage device. Set the port to match the
controller (typically `115200`).

## Notes

- Adapter currently reports unbounded limits.
- Adapter uses synchronous motion (`M400`) and returns `Busy() == false`.
- `Home()` homes only the axes owned by each device; it never falls back to a
  full `G28` home-all, so the XY device never moves Z and vice versa.
- To compile in `mmCoreAndDevices`, place this folder under `DeviceAdapters/` and add it to that repository's build configuration (CMake or VS project lists, depending on platform/build system).

## References

1. **EnderScope: a low-cost 3D printer-based scanning microscope for microplastic detection.** *Philosophical Transactions of the Royal Society A*, 2024. <https://doi.org/10.1098/rsta.2023.0214>

2. **Enderscope.py: A library for computational imaging using the EnderScope automated microscope.** *SoftwareX*, 2025. <https://doi.org/10.1016/j.softx.2025.102210>

