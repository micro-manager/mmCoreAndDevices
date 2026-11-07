# Building EmergentCamera on Windows x64

## Requirements

- Visual Studio 2022 with Desktop development with C++, the v143 toolset,
  and a Windows 10/11 SDK.
- The `mmCoreAndDevices` source tree and its normal Windows build dependencies.
- The classic Emergent eSDK development files (`include/`, `lib/`, and runtime).
  The archived Windows hardware regression records eSDK **4.08.06**; the
  existing public compatibility notes also list **4.08.02**. eSDK Pro is not
  required. Other releases require validation on the deployment setup.
- A Micro-Manager runtime matching Device API **75** / Module API **10**.

## Project layout and build

Keep this whole adapter directory at
`mmCoreAndDevices/DeviceAdapters/EmergentCamera/`. Its Visual Studio project
references the repository's `MMDevice` project and shared build properties
using paths relative to the adapter root. The C++ files are in `src/`.

1. Open `mmCoreAndDevices/micromanager.sln` in Visual Studio 2022.
2. Choose **Release** and **x64**.
3. Add `DeviceAdapters/EmergentCamera/EmergentCamera.vcxproj` with
   **Add > Existing Project** if it is not already present in your local solution.
4. The default SDK directory is `C:\Program Files\EVT\eSDK`.
   Set the MSBuild property `ESDK_ROOT` if your installation differs.
5. Build the **EmergentCamera** project. Its dependency builds MMDevice.

The default solution output is
`mmCoreAndDevices/build/Release/x64/mmgr_dal_EmergentCamera.dll`.
The project links `EmergentCamera.lib`, `EmergentGigEVision.lib`,
`EmergentGenICam.lib`, `EmergentUtil.lib`, and Windows `Iphlpapi.lib`.

Build products stay in `build/`; the checked-in `binaries/windows-x64/` DLL
is the separately preserved, hardware-tested artifact. Building does not
automatically replace that published binary.

## Installation and validation

Close Micro-Manager and copy your rebuilt DLL into its installation directory.
Follow [the installation instructions](../README.md#windows-installation) for
SDK runtime paths. Use [the hardware test runner](../tests/README.md) with
`-AdapterPath` pointing to the directory containing the rebuilt DLL.
