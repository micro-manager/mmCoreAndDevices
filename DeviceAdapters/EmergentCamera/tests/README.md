# Hardware tests and archived results

The original Windows camera regression and settings-persistence tests passed
on **2026-11-05**. Full output is kept in [results/2026-11-05/](results/2026-11-05/).
Linux results are static/build verification, not Linux camera tests.

## Running the Windows hardware regression

A connected camera, compatible eSDK, and installed Micro-Manager are required.
Close Micro-Manager and eCapture first. From the adapter root:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tests\run-tests.ps1 `
  -CameraIP 'YOUR_CAMERA_IP'

powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tests\run-tests.ps1 `
  -CameraIP 'YOUR_CAMERA_IP' -Persistence
```

The default adapter directory is `../binaries/windows-x64/` relative to the
test scripts. Supply `-AdapterPath` to test a rebuilt DLL and
`-MicroManagerPath` for a nondefault Micro-Manager installation.
The scripts exercise and restore camera settings; the persistence script
includes assertions for the original camera serial and must be adapted for
a different camera. New outputs are written beside the scripts, separately
from archived results, and are ignored by Git.

The optional `-Probe` and `-Capabilities` modes use the original SDK probe
scripts. They retain their original SDK and Python 3.12 installation paths;
adjust those paths for another test machine. `-ConfigCheck` uses
`EmergentCam-migrated.cfg`, a camera-specific test fixture, not a universal
hardware configuration.

## Archived files

| File | Contents |
| --- | --- |
| `functional-results.txt` | 170 PASS records, zero failures |
| `persistence-results.txt` | Six passing persistence groups, zero failures |
| `camera-capabilities.json` | Vendor API feature/camera snapshot |
| `linux-verification.json` | Linux binary hash, exports/imports, dependencies, and pending runtime flags |
| `linux-toolchain-manifest.json` | Checksummed toolchain and SDK package provenance |

Archived files are preserved as recorded, including original-machine paths.
See [the test report](../docs/TEST_REPORT.md) for physical GPIO/PTP limits.
