# EmergentCamera verification - 2026-11-05

Developer: **Dr. Volodymyr Cherkas**<br>
Company: **Advanced Optical Devices** - [www.ao-devices.com](https://www.ao-devices.com)<br>
Manufacturer: **Emergent** - [www.emergentvisiontec.com](https://www.emergentvisiontec.com)

Hardware: HE-25000SBL-T-M, serial 2017395, firmware 3.35,
192.168.206.206. Application: Micro-Manager 2.0.3 (20261008), Windows x64.
Tests use the installed MMCoreJ, vendor eSDK and real camera.

Release x64 builds complete without compiler warnings or errors. The full
camera regression has **170 PASS records and zero failures**. Six additional
settings-persistence test groups also passed. Scripts and complete results
are in [../tests/](../tests/) and [the archived results](../tests/results/2026-11-05/).

## Current behavior

The browser receives this order directly from the adapter, without numeric
prefixes or changes to Micro-Manager:

Camera Info -> temperatures -> Acquisition -> Exposure -> Gain -> auto-control
ROI -> Readout -> ROI/Binning -> Trigger including GPIO -> PTP -> image
controls -> Settings.

The first-start frame-rate limit defaults to **Off**. Explicitly saving an On
preference restores that preference at the next startup. Bit-depth choices
display in ascending order: ADC 8/10 and pixel format 8/10/12. The actual
Micro-Manager PropertyItem sorter and visible dropdowns were checked.

Connection information shows the camera IP, Ethernet 2, and 10 Gbit/s host NIC
link rate. This is not a measurement of image throughput or a remote switch
port. Developer/website and manufacturer/website appear together near the top.

The camera has fixed GPO0/GPO1 output directions and GPI2/GPI3 input directions.
The SDK/XML has no direction switch. The selected default routing is GPI2
acquisition start on a rising edge, and GPO1 exposure active with active-high
polarity. Saved settings override defaults during startup restore.

## Hardware regression

| Feature | Verified behavior |
| --- | --- |
| Discovery/identity | Exact-IP connection; model, serial and firmware reads |
| Binning | Main-window/Core Binning; 1x gives 5328x4608 and 2x gives 2664x2304; ROI limits follow both directions |
| Pixel formats | Mono8/10/12 in both binning modes; image length, byte count and significant depth |
| ROI | Alignment, centering, clearing, invalid-input rejection and subsequent capture |
| Exposure | Core/property/hardware readback agreement, auto exposure, applied-value rounding |
| Frame rate | Off default, limit on/off, configured-rate readback and captures |
| Acquisition | Snap and finite sequences in Continuous/MultiFrame with Internal/Software triggering |
| Long sequences | 1,003 frames at 30 fps across complete hardware bursts |
| Live/restart | Repeated sequences, MultiFrame Live, stop/recovery, busy protection |
| External wait | Prompt cancellation of an external-trigger wait and subsequent internal capture |
| ADC | Bit8/Bit10, each with all advertised native unpacked mono output formats |
| Gain/auto controls | Gain, PGA, offset, auto targets/rates/limits and auto-control ROI write/read/restore; integer increment alignment |
| GPIO/trigger | All advertised enum values and representative numeric values; polarity mapping and GPI state reads |
| Orientation | Reverse X/Y write/read and successful capture after each setting |
| LUT | Enable read/write; entry 17 write/read and restoration of value/index |
| Temperatures | Sensor, camera and PHY readings |
| Metadata | Vendor hardware timestamp tag on every sequence image; frame ID also emitted |
| Presentation | Unique unnumbered names, group order, ascending bit-depth dropdowns and attribution |

On this exact model/firmware, the SDK advertises AcquisitionFrameCount up to
1000 and reads a written 1000 back as 1000, but repeated acquisition attempts
deliver exactly 232 frames, consistent with an 8-bit counter wrapping. The
adapter limits this model/firmware to 255-frame hardware bursts and splits
longer Micro-Manager sequences. The 1,003-frame regression passes with that
workaround.

## Emergent.ini save and startup restore

Installed file: `C:\Program Files\Micro-Manager-2.0\Emergent.ini`.
The path defaults to the adapter DLL folder. Sections use camera serial
numbers with a model check. Other cameras' sections are preserved. Writes
replace the file through a temporary file.

Automatic save occurs on clean Micro-Manager close/device unload. Immediate
save is available using `Settings - Save = Save now`, with Live stopped.
`Settings - Active File` and `Settings - Status` show the file and outcome.
Pre-init `Settings - File` / `Settings - Remember` allow an alternate absolute
path or disabling automatic persistence. A writable folder is required.

The persistence checks cover:

- explicit save, serial isolation and preservation of another camera section;
- automatic shutdown save, followed by a fresh Core/camera initialization
  restoring ROI, binning, ADC/pixel format, exposure, gain, frame-rate policy,
  trigger and GPIO choices, followed by successful capture;
- unsupported-version file reporting and preservation, while capture remains usable;
- invalid numeric exposure and ROI entries skipped with an explanatory status;
- file-write failure reported without preventing capture;
- restoration of the recorded small-ROI camera setup and preparation of the
  initial installed INI.

Normal application close was also tested: changing exposure from 0.1 to 0.2 ms
in the main window and closing Micro-Manager wrote `0.2000` to the installed
INI. Relaunching Micro-Manager reported restoration of 49 settings in CoreLog.
The main window visibly restored 0.2 ms after restart. Exposure was then
returned to 0.1 ms and saved through the installed `Settings - Save` control.
`Settings - Status` displayed a successful save; reading the installed file
confirmed 0.1000 ms, frame-rate limit Off, binning 1 and the original ROI.
The final browser order, including Gain immediately after Exposure and GPIO
under Trigger, was verified visually. The browser was left open for review.

The initial profile retains binning 1, ROI 240x32 at 1640,2048, exposure 0.1 ms
and PTP TwoStep, with frame-rate limit Off as requested. Hardware can normalize
dependent values to its current limits and increments. Runtime commands,
read-only information, images and LUT table entries are excluded. A crash
before saving does not guarantee retention of the last edits. Configuration
presets/scripts applied after initialization can override restored values.

## PTP and physical test limits

Both OneStep and TwoStep were accepted by the camera. In direct SDK checks,
six samples per mode at two-second intervals all reported Listening, zero
offset and a zero latched PTP clock. No valid master lock (Slave) or
cross-device synchronization accuracy was demonstrated. Zero offset while
Listening is not an accuracy measurement. PTP scheduled gate acquisition is
not exposed by this adapter.

Physical GPIO waveforms and response to actual external trigger pulses were
not measured. Gain calibration, automatic brightness convergence, LUT image
response and geometric reversal require controlled illumination or a target;
the regression establishes API and capture behavior for those controls.

## Deployment

Installed DLL SHA-256:
`7E7A946E4D9814F99EA0B1820EE56E00096D0FD7FE957D6C340A06F06D1ADBF7`.

Project files: `S:\Development\EmergentCamera`.
The installed `.cfg` is migrated as needed using `property-renames.json`.
Previous DLL, source and configuration files are preserved under
`S:\Projects\Emergent\EmergentCamera-validation\backups\before-settings-persistence-20261105`.


## Linux build - 2026-11-05

An experimental Ubuntu 24.04 x86_64 binary was cross-compiled using the
supplied Emergent eSDK 4.08.05. CMake/Ninja build, ELF exports, SONAME and all
156 required symbol resolutions passed.
Windows Release x64 recompilation also passed after the guarded Linux changes.
Linux runtime and camera tests have **not** been run. The hardware results
above describe Windows only. **macOS is not supported.**

Linux binary SHA-256: `7ABF2F44F403D07D0FA9C8106B7B204E7CF2B6EFF8C08508AFAF9A5DA2197646`.
See [BUILD_LINUX.md](BUILD_LINUX.md) and [linux-verification.json](../tests/results/2026-11-05/linux-verification.json).


## Publication layout - 2026-11-06

The original 2026-11-05 results above are preserved. Source files are now in
`../src/`, Windows and Linux binaries are in separate `../binaries/` folders,
and archived result files are under `../tests/results/2026-11-05/`.
Recorded installation paths and backup locations describe the original test
machine. No camera regression was rerun during this repository reorganization.
