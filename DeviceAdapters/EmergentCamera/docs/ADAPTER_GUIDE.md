# EmergentCamera Micro-Manager device adapter

**Platform status:** Windows x64 is hardware-tested. The Ubuntu 24.04 x86_64
binary is experimental and cross-compiled; Linux runtime and camera testing
are pending. **macOS is not supported (Intel or Apple silicon).**
See [BUILD_LINUX.md](BUILD_LINUX.md) for Linux build and validation details.

Developer: **Dr. Volodymyr Cherkas**<br>
Company: **Advanced Optical Devices** — [www.ao-devices.com](https://www.ao-devices.com)
Manufacturer: **Emergent** — [www.emergentvisiontec.com](https://www.emergentvisiontec.com)

This adapter targets the installed Emergent Vision Technologies eSDK and a
monochrome GigE camera selected by IP address. The default IP is
`192.168.207.207`. Its interfaces were checked against the installed
Micro-Manager 2.0 core (`MMCore 12.5.0`, Device API 75, Module API 10).

Implemented functionality:

- exact-IP discovery and camera identification;
- snap acquisition and Micro-Manager sequence acquisition;
- exposure in milliseconds (translated to the eSDK `Exposure` value in
  microseconds);
- hardware ROI with the camera's width, height, and offset increments;
- native unpacked `Mono8`, `Mono10`, and `Mono12` output;
- hardware 1x1 / 2x2 binning in the main Micro-Manager window and Core API;
- ADC modes, analog/PGA gain, offset, auto gain and auto exposure controls;
- ROI centering, image reversal, lookup table, GPIO and trigger configuration;
- internal/software-triggered Snap, finite sequences, and Live acquisition;
- PTP mode/status, signed offset, and latched 64-bit camera clock diagnostics;
- exact camera timestamps and frame IDs in sequence metadata;
- one-component 8-bit or 16-bit Micro-Manager image buffers;
- clean stream/buffer teardown and detailed eSDK errors in CoreLog.

The connected camera was probed with the vendor API while this adapter was
written:

- model: `HE-25000SBL-T-M`;
- serial: `2017395`;
- firmware: `3.26`;
- sensor: `5328 x 4608`;
- supported native mono formats: `Mono8`, `Mono10`, and `Mono12`;
- ROI increments: 8 pixels horizontally and 16 pixels vertically;
- exposure range: 0.094 to 244.205 ms.

The hardware regression on 2026-11-05 used the same camera with firmware
`3.35` at `192.168.206.206`. Exposure limits depend on ADC, binning, ROI and
frame rate and are queried from the camera. The `Camera Info` properties
identify the developer, company, manufacturer and their websites, the connected
camera IP, and the host network interface with its negotiated link rate.
`Connection Speed` is the host NIC's line rate, not measured image throughput
or a remote switch-port measurement. It currently reads 10 Gbit/s on Ethernet 2.

## Property browser and GPIO

Property names use descriptive groups without numeric prefixes. The adapter
enumerates them as Camera Info, temperatures, Acquisition, Exposure, Gain,
Readout, ROI/Binning, Trigger (including GPIO), PTP, image controls, and Settings.
This order is supplied directly to Micro-Manager's browser. ADC and pixel
format choices start with their bit depth, for example `8-bit (Bit8)` and
`12-bit (Mono12)`, allowing Micro-Manager's numeric-prefix sorter to show them
in ascending 8/10/12 order. The parenthesized token is the native SDK value.

The connected HE-25000 has fixed line directions:

| Line | Direction | Default / control |
| --- | --- | --- |
| GPIO 0 / GPO0 | Output | Output signal and active-high/active-low polarity |
| GPIO 1 / GPO1 | Output | Exposure active, active high |
| GPIO 2 / GPI2 | Input | Acquisition start, rising edge |
| GPIO 3 / GPI3 | Input | Input mode, state and debounce |

The `Trigger - GPIO n - Direction` fields are read-only because this camera provides no
programmable direction node. GPO0/GPO1 cannot be switched to inputs and
GPI2/GPI3 cannot be switched to outputs. Use `Trigger - Start Input` to select
the acquisition-start input. Internal triggering remains the startup mode;
choose `Trigger - Mode = External` to acquire from external input pulses.
If saved settings exist, the saved trigger/GPIO choices replace these defaults.
GPO polarity now uses explicit `Active high` / `Active low` labels, mapped to
the vendor boolean values true / false respectively.

See the vendor's [GPIO pinout](https://docs.emergentvisiontec.com/cameras/gpio-ports-schematics)
and [output polarity](https://docs.emergentvisiontec.com/camera-features/gpo-n-polarity).

## Reusing settings with Emergent.ini

The adapter now saves imaging and trigger settings to `Emergent.ini` beside
the adapter DLL. In the installed setup this is
`C:\Program Files\Micro-Manager-2.0\Emergent.ini`. Each camera has its own
`[Camera:serial-number]` section and a model check. Micro-Manager's `.cfg`
continues to select devices, camera IPs and Core roles; the INI stores the
camera's last working runtime settings.

Settings are restored during camera initialization and saved automatically
on a **clean Micro-Manager close or device unload**. For an immediate save,
stop Live and set `Settings - Save` to `Save now` in the property browser.
`Settings - Status` reports restore/save results and file-access errors;
`Settings - Active File` shows the exact file in use. A crash before saving
does not guarantee retention of the latest edits.

In the Hardware Configuration Wizard, the pre-initialization properties
`Settings - File` and `Settings - Remember` allow an absolute alternate path
or disabling automatic restore/save. This is also useful for isolated tests.
The chosen folder must permit writing. Existing sections for other serials
are preserved, and updates replace the INI using a temporary file.

The first-start frame-rate limit is **Off**. An explicitly saved On preference
is restored with its requested rate on later starts. Saved values are applied
in dependency order: ADC/pixel format, binning/ROI, scalar controls, rate
policy, auto controls, PTP and trigger mode. The camera may normalize values
to its current increments and limits. Invalid/unavailable entries are skipped
and logged. Malformed or unsupported-version files are reported without
preventing camera initialization and are preserved on automatic shutdown.

Runtime commands, read-only identity/temperature/clock information, and LUT
index/value edits are excluded. The INI remembers LUT enable but is not a
backup of the camera's complete lookup table. Acquisition buffers/images are
never stored in it. Retained external trigger or PTP settings still require
the corresponding input pulses or clock master.

Example format (the installed file contains this camera's actual settings):

```ini
[Emergent]
FormatVersion=1

[Camera:2017395]
Model=HE-25000SBL-T-M
Binning=1
Readout - Pixel Type=12-bit (Mono12)
Exposure - Time (ms)=0.1000
Acquisition - Frame Rate Limit=Off
RequestedFrameRate=664
Trigger - Start Input=GPI_2
Trigger - GPIO 1 - Output Signal=Exposure
Trigger - GPIO 1 - Output Polarity=Active high
```

Hardware Configuration Startup presets or later scripts can override restored
values by setting the corresponding properties after initialization.

## Main-window binning and compatibility

Micro-Manager requires a property named exactly `Binning` to populate its
main-window dropdown and operate `core.setBinning()`. This adapter provides
`Binning` with the hardware-supported values `1` and `2`.
The previous numbered aliases have been removed from the browser. The supplied
[`property-renames.json`](../examples/property-renames.json) maps the old property names to their replacements;
update scripts and other saved configurations using that map. Existing hardware configurations must be migrated separately; copying the
adapter binary does not modify a `.cfg` file.
ADC/Pixel Type preset values also need the displayed numeric-leading labels.
Changing binning resets the image to the full frame for that binning factor
and refreshes ROI, frame-rate and exposure limits. Stop Live before changing
properties directly through the Core; the Micro-Manager GUI handles this for
its own controls.

Integer camera features are aligned to their reported hardware increments.
Exposure and gain readbacks report the value actually applied by the camera.

On the tested HE-25000SBL-T-M firmware 3.35, a hardware frame count of 1000
reads back as 1000 but captures 232 frames. This revision limits this specific
model/firmware to hardware bursts of at most 255 frames and splits longer
Micro-Manager sequences into complete bursts. Live acquisition uses one-frame
bursts in MultiFrame mode so it can be stopped between completed bursts.

## PTP diagnostics

The property browser exposes:

- `PTP - Mode`: `Off`, `OneStep`, `TwoStep`, as supported by the camera;
- `PTP - Status`: read-only camera synchronization state;
- `PTP - Offset (ns)`: signed camera-to-master clock offset;
- `PTP - Camera Clock (ns)`: timestamp latched before reading its two
  32-bit registers, returned as decimal text to preserve all 64 bits.

`Listening` means that the camera has not locked to a PTP clock; a valid lock
is reported as `Slave`. A zero offset in `Listening` is not evidence of
synchronization. Choose OneStep or TwoStep to match the PTP grandmaster.
The adapter does not start a grandmaster or configure the host network.

Each sequence image includes `Emergent-HardwareTimestampNs` and
`Emergent-FrameId` tags. These preserve the vendor frame's timestamp and ID;
the timestamp is useful as a synchronized PTP timestamp only after a valid
lock. The standard Micro-Manager elapsed-time metadata remains host elapsed
time. PTP scheduled acquisition gate times are not implemented by this
adapter's trigger selector.

On the current network, both PTP modes were accepted but remained
`Listening`; no valid PTP lock or cross-device accuracy was demonstrated.
See [TEST_REPORT.md](TEST_REPORT.md) for the checks performed and their limits.

Vendor references:
[PTP status](https://docs.emergentvisiontec.com/camera-features/ptpstatus),
[PTP offset](https://docs.emergentvisiontec.com/camera-features/ptpoffset),
[trigger modes](https://docs.emergentvisiontec.com/cameras/trigger-modes-for-area-scan-cameras).

## Build and installation

See the [Windows build instructions](BUILD_WINDOWS.md),
[Linux build instructions](BUILD_LINUX.md), and
[platform-specific installation instructions](../README.md).
The implementation is in `../src/`; build files are in the adapter root.

## Configure and test

1. Close eCapture; only one application should control the camera.
2. Start Micro-Manager and open **Devices > Hardware Configuration Wizard**.
3. Add **EmergentCamera / EmergentCamera**.
4. Leave **Camera IP Address** at `192.168.207.207` and initialize the device.
5. Set it as the Core camera, then test **Snap** and **Live** first with
   `Readout - Pixel Type = 8-bit (Mono8)` and a short exposure.

`Mono10` and `Mono12` are returned as 16-bit little-endian grayscale buffers;
the significant bit depth reported to Micro-Manager is 10 or 12 respectively.
Packed formats are deliberately omitted because the camera provides their
unpacked equivalents, avoiding a conversion and extra buffer copy.

For loading/runtime errors, enable **Tools > Options > Debug Logging** and
inspect the Micro-Manager CoreLog. eSDK failures include their numeric error
code in the log.

See [the hardware test instructions](../tests/README.md) to repeat the
Windows regression with a connected camera. Archived results apply to
2026-11-05 and are not a substitute for validation on a new setup.
