# Micro-Manager Device Adapter for IDS Peak cameras
Micro-Manager is an application to control microscope hardware, such as cameras. It includes a hardware abstraction layer written in C++ and a user interface written in Java. Micro-Manager needs a translation layer between the driver (written by the manufacturer) and Micro-Manager's C++ backend. This translation layer is called a "Device Adapter" (Micro-Manager has chosen not to call it a "Driver" to distinguish it from the libraries provided by the manufacturers of the devices).

This GitHub repository contains a Device Adapter for IDS cameras. It contains both the already built .dll (mmgr_dal_IDSPeak.dll) and the C++/h files to build it yourself. Instructions can be found below. The Device Adapter was tested with multiple IDS USB3-3040CP-HQ Rev 2.2 on Windows 10, and not with any other camera or operating system. Although camera model should not matter, there might be assumed default settings that might not be universally present. A different operating system could be more problematic. 

## Using the precompiled .dll
As of Dec 11 2023, the IDS Peak device adapter is included in the nightly build of Micro-Manager. Greatly simplifying the use of IDS cameras with Micro-Manager. The following steps will guide you through the process of "installing" the .dll, allowing you to use IDS cameras with Micro-Manager.
1. Install the most recent nightly build of Micro-Manager (or at least a version after Dec 11 2023). **NOTE: it is NOT included in version 2.0.0, as it dates from  July 13, 2021. Use the following link to the latest nightly builds instead** https://micro-manager.org/Micro-Manager_Nightly_Builds.
2. Download and install "IDS Peak" from the IDS website https://en.ids-imaging.com/downloads.html
3. Copy all .dll and .lib files from the IDS Peak software into the root folder of Micro-Manager. The .dll and .lib files can be found in ".\PATH_TO_INSTALL\IDS\ids_peak\comfort_sdk\api\lib\x86_64" (.\PATH_TO_INSTALL is the folder you installed IDS Peak, e.g. "C:\Program Files").
4. Start Micro-Manager and either walk through the Hardware configuration wizard. Make sure the camera is plugged into a USB3 port before launching Micro-Manager.

## Building the device adapter yourself
More advanced users could build the device adapter themselves, allowing them to tailor the device adapter to their needs. If you just plan to use the default device adapter, there is no benefit to building it yourself. Below you will find a brief walkthrough on building the device adapter.
1. First, follow the Micro-Manager guide on building Micro-Manager (https://micro-manager.org/Building_MM_on_Windows) and on setting up a Visual Studio environment to building device adapters (https://micro-manager.org/Visual_Studio_project_settings_for_device_adapters).
2. In Visual Studio, right-click the project you created in step 1 and choose **Properties**. Under **Configuration Properties > C/C++ > General** add the following folder to the **Additional Include Directories**: ".\ids_peak\comfort_sdk\api\include". You can exit the **Properties** interface now.
3. Right-click the **Project** again, and click **Add > Existing Item**. Browse to ".\ids_peak\comfort_sdk\api\lib\x86_64" and add **"ids_peak_comfort_c.lib"**.
4. Right-click the **Header Files** tab under your project, and click **Add > Existing Item**, add the **"IDSPeak.h"** file.
5. Right-click the **Source Files** tab under your project, and click **Add > Existing Item**, add the **"IDSPeak.cpp"** file.
6. Right-click the **Project**, and click **Build**. The .dll will now be build, and should finish without any warnings/errors.
7. The .dll file can now be found in ".\micro-manager\mmCoreAndDevices\build\Debug\x64".
8. Now you have compiled the .dll, follow the steps under "Using the precompiled .dll" to enable Micro-Manager to communicate with IDS cameras.
9. If you want to use the .dll on a PC other than the one used to build the .dll, it is best to set the Solution Configuration to "Release" (that way the other PC doesn't require an install of Microsoft Visual Studio 2019). The .dll can than be found in ".\micro-manager\mmCoreAndDevices\build\Release\x64".

## Features
- Imaging in grayscale (8 and 16bit) and 32bit RGBA. One can switch between 8bit grayscale and 32bit RGBA in **Device -> Device Property Browser -> IDSCam - PixelType**
- Multi-camera support. One can load multiple cameras cameras from the Hardware configuration wizard. The settings of each camera can be changed separately. To record simultaneously with IDS cameras, one should also load the "Utilities->Multi camera" Device Adapter. The "Multi Camera" device adapter should be listed as the default camera. After finishing the "Hardware configuration wizard". One should go to the "Device Property Browser" and under "Multi Camera - Physical Camera X" select the desired cameras.

# User guide
For more detailed information on how to use IDS cameras with Micro-Manager, please check https://micro-manager.org/IDSPeak

## Future features
- **None planned**
Note that these are just ideas, no promises are made that these will be implemented in a timely manner (or at all). Other suggestions are more than welcome, either create a github issue or send an email to lars.kool@espci.fr

## Acknowledgements
This Device Adapter was developed by Lars Kool at Institut Pierre-Gilles de Gennes (Paris, France).