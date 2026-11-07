# EmergentCamera for Micro-Manager

Micro-Manager device adapter for Emergent monochrome cameras. Supports Snap,
Live, sequences, exposure, binning, hardware ROI, gain and trigger/GPIO controls.
Camera settings can be saved and restored between Micro-Manager sessions.

**Developer:** Dr. Volodymyr Cherkas

**Company:** [Advanced Optical Devices](https://www.ao-devices.com)

**Manufacturer:** [Emergent Vision Technologies](https://www.emergentvisiontec.com)

Advanced Optical Devices is a system integrator and Emergent reseller in
Poland and the European Union.

## Project layout

```text
EmergentCamera/
  README.md
  LICENSE                           BSD-3-Clause; Dr. Volodymyr Cherkas
  NOTICE                            Original author attribution
  CMakeLists.txt                     Linux adapter build
  EmergentCamera.vcxproj             Windows adapter build
  EmergentCamera.vcxproj.filters
  src/                              C++ implementation and header
  binaries/
    windows-x64/                     Hardware-tested Windows DLL
    linux-x86_64/                    Experimental Linux shared library
    SHA256SUMS
  docs/                             Build instructions, user guide, test report
  examples/                         Settings example and property migration map
  tests/                            Hardware test scripts
    results/2026-11-05/              Archived regression and static-check results
  licenses/                         Micro-Manager third-party license notice
```

The adapter directory was renamed from `Emergent` to `EmergentCamera`.
The module/device name remains **EmergentCamera**, so existing hardware
configurations do not need a device-name change for this directory rename.
Download the binary for your operating system; the SDK is installed separately.
Checksums are in [binaries/SHA256SUMS](binaries/SHA256SUMS).

## Compatibility

**macOS is not supported, including Intel Macs and Apple silicon.**

| Platform | Status |
| --- | --- |
| Windows x64 | Hardware-tested with the configuration below |
| Linux x86_64, Ubuntu 24.04 | Experimental; cross-compiled and statically checked; Linux runtime and camera tests pending |
| macOS | Not supported |

Tested with **Micro-Manager 2.0.3 (20261008), Windows x64, Emergent eSDK
4.08.06/4.08.02**, and an **HE-25000SBL-T-M** camera with firmware **3.35**. Other
camera models and SDK versions have not been validated with the Windows binary but should work.

The Linux binary uses **Emergent eSDK 4.08.05 for Ubuntu 24.04 x64**.
Both binaries require **Micro-Manager Device API 75 / Module API 10**.

## Windows installation

1. Install [Micro-Manager](https://micro-manager.org/Download_Micro-Manager_Latest_Release)
   for Windows x64 and the compatible
   [Emergent eSDK](https://www.emergentvisiontec.com/downloads/).
   This binary was tested with eSDK 4.08.06; obtain that version from Emergent
   or test newest version available.
2. Configure the camera and network interface using Emergent's instructions in eCapture.
   Confirm that the camera captures in eCapture, then close eCapture and Micro-Manager.
3. Download [mmgr_dal_EmergentCamera.dll](binaries/windows-x64/mmgr_dal_EmergentCamera.dll) and copy it
   into the folder containing `ImageJ.exe`, normally:

   ```text
   C:\Program Files\Micro-Manager-2.0\
   ```

4. Start Micro-Manager. Open **Devices > Hardware Configuration Wizard**, add
   **EmergentCamera / EmergentCamera**, and select the discovered camera.
   Verify **Camera IP Address**, or choose **Manual IP** and enter its actual
   address. The adapter's fallback address, `192.168.207.207`, is not a universal
   camera address.
5. Select the device as the default camera, finish the wizard, and save the
   hardware `.cfg` file. For the first **Snap/Live** test, use
   **Trigger - Mode = Internal** and **PTP - Mode = Off**.

If the adapter is unavailable, check CoreLog, install the
[Microsoft Visual C++ v14 x64 runtime](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist),
and ensure these SDK folders are on `PATH` (adjust for your SDK location), then
restart Micro-Manager:

```text
C:\Program Files\EVT\eSDK\bin
C:\Program Files\EVT\eSDK\genicam\bin\Win64_x64
C:\Program Files\EVT\eSDK\ffmpeg\bin
```

## Linux installation (experimental)

1. Install Linux x86_64 Micro-Manager with the matching device/module APIs,
   and [Emergent eSDK](https://www.emergentvisiontec.com/downloads/)
   **4.08.05 for Ubuntu 24.04 x64**. Use its supported kernel and NIC drivers;
   the supplied SDK package targets kernel **6.17.0**. Confirm capture using
   the vendor software, then close it.
2. Download [libmmgr_dal_EmergentCamera.so.0](binaries/linux-x86_64/libmmgr_dal_EmergentCamera.so.0)
   and place it beside Micro-Manager's other `libmmgr_dal_*.so.0` adapters.
   Keep the `.so.0` filename.
3. Make the SDK libraries available before starting Micro-Manager. For the
   default SDK installation, run:

   ```bash
   export LD_LIBRARY_PATH="/opt/EVT/eSDK/lib:/opt/EVT/eSDK/genicam/bin/Linux64_x64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
   ```

4. Run `ldd` on the downloaded adapter and resolve any `not found` dependencies.
   Start Micro-Manager from that terminal and configure the camera as described
   in Windows steps 4-5 above.

## Settings

Exposure and binning are available in the main window. Other controls are in
**Devices > Device Property Browser**. Frame-rate limit defaults to **Off**;
an explicitly saved On preference is restored on the next startup.

`Emergent.ini` is created beside the adapter binary and stores settings separately for
each camera serial number. Settings are saved on a **clean close/device
unload** and restored at initialization. To save immediately, stop Live and
set **Settings - Save = Save now**. Check **Settings - Status** for the result.

If the adapter folder is not writable, set **Settings - File** to an
absolute path in a writable folder during hardware configuration.
**Settings - Remember = Off** disables automatic save/restore. The `.cfg`
file still selects hardware and camera IP addresses.

## Notes

- Supported output formats are unpacked Mono8, Mono10 and Mono12. The latter
  two use 16-bit image buffers. Colour and packed formats are not implemented.
- On the tested camera, GPO0/GPO1 are fixed outputs and GPI2/GPI3 are fixed
  inputs. Defaults are GPI2 acquisition start and GPO1 exposure active,
  active high. Saved settings override defaults. See the
  [manufacturer's GPIO pinout](https://docs.emergentvisiontec.com/cameras/gpio-ports-schematics)
  before connecting external equipment.
- PTP diagnostics are exposed, but master lock and synchronization accuracy
  have not been verified. Scheduled PTP acquisition is not implemented.


## Source and documentation

- [C++ source](src/) and [source layout](src/README.md).
- [Windows build instructions](docs/BUILD_WINDOWS.md).
- [Linux build and validation](docs/BUILD_LINUX.md).
- [Detailed adapter guide](docs/ADAPTER_GUIDE.md), including properties,
  GPIO, PTP, hardware burst limits, and settings persistence.
- [2026-11-05 test report](docs/TEST_REPORT.md) and
  [archived results](tests/results/2026-11-05/).
- [Hardware test instructions](tests/README.md).
- [Settings example](examples/Emergent.ini.example) and
  [property migration map](examples/property-renames.json).

The runtime implementation and published binaries are unchanged. Copyright
and SPDX license headers have been added to the source.
Camera tests were performed on **2026-11-05**, as recorded in the report;
they were not repeated on the PC used for this reorganization. The archived
Windows functional results contain 170 PASS records with zero failures and
six passing persistence groups. Linux build/static checks passed, while
Linux runtime and camera validation remain pending.

## Licensing and attribution

**BSD-3-Clause** — [LICENSE](LICENSE).

Copyright (c) 2026, **Dr. Volodymyr Cherkas**.

Public forks and redistributed modifications must retain this copyright
notice naming the original author, the license conditions, and the disclaimer.
Binary redistributions must reproduce them in their accompanying documentation
and/or other materials. The author's name may not be used for endorsement
without specific prior written permission. See [NOTICE](NOTICE) and
[licensing and attribution guidance](docs/LICENSING.md).

Existing Micro-Manager code retains its own copyright and license, reproduced
in [licenses/MMDevice-BSD-3-Clause.txt](licenses/MMDevice-BSD-3-Clause.txt).
The vendor SDK is not included and remains subject to Emergent's terms.
