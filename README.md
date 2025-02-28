# kmdouglass's Fork of mmCoreAndDevices
Custom device adapters that are not officially part of the Micro-Manager project.

## Contents

This fork provides the following additional device adapters:

- [SkeletonDevice](DeviceAdapters/SkeletonDevice/) : A minimal, general device adapter that works but does nothing. Used as a starting point for new adapters.
- [SkeletonSerial](DeviceAdapters/SkeletonSerial/) : A minimal device adapter for serial port communications

## Instructions

- Set up the development environment for Micro-Manager if you haven't already
- Open the [micromanager.sln](./micromanager.sln) file in Visual Studio 2022
- Right click `Solution 'Micromanager'` in the Solution Explorer and select `Add` > `Existing Project...`
- Navigate to the folder containing the code for the Device Adapter that you want to build
- Select the adapter's `.vcxproj` file and open it
- Navigate to the solution in the Solution Explorer and right click it. Select `Properties...`
- Ensure that the `Platform Toolset` is set to `Visual Studio 2019 (v142)`
- Toggle the Solution Configuration to either `Debug` or `Release` depending on whether you are testing. (`Release` should be used for experiments because it is generally faster.)
- Right click the solution and select `Build`
- Navigate to the output directory containing the DLL, for example: `C:\Users\douglass\src\projects\micro-manager\mmCoreAndDevices\build\Release\x64\mmgr_dal_SkeletonDevice.dll`
- Copy the DLL to your Micro-Manager root directory
- Start Micro-Manager and open the `Devices` > `Hardware Configuration Wizard...`
- In Step 2, verify that the device adapter appears in the list of `Available Devices`
- Add the device and test

# Original README

# mmCoreAndDevices
The c++ code at the core of the Micro-Manager project.

## API Docs
[Main Page](https://micro-manager.org/apidoc/MMCore/latest/index.html)

If you are using a scripting language to control a microscope through the CMMCore object
then you are likely looking for the [CMMCore API](https://micro-manager.org/apidoc/MMCore/latest/class_c_m_m_core.html)

### Building on Windows
The windows project uses the following properties which may be overridden in the MSBuild command line using the `/property:name=value` switch:

    MM_3RDPARTYPUBLIC: The file path of the publically available repository of 3rd party dependencies
    MM_3RDPARTYPRIVATE: The file path of the repository of 3rd party dependencies which cannot be made publically available
    MM_BOOST_INCLUDEDIR: The include directory for Boost.
    MM_BOOST_LIBDIR:  The lib directory for Boost.
    MM_SWIG:  The location of `swig.exe`
    MM_PROTOBUF_INCLUDEDIR: The include directory for Google's `protobuf`
    MM_PROTOBUF_LIBDIR: The lib directory for Google's `protobuf`
    MM_PROTOC: The location of `protoc.exe` for Googles `protobuf`
    MM_BUILDDIR: The directory that build artifacts will be stored in.
	
To see the default values of each property please view `MMCommon.props`

### Building on Mac and  Linux

The easiest way to build on Mac or Linux is to clone the [micro-manager](https://github.com/micro-manager/micro-manager) repository and use this repo as a submodule. 


Then follow the [instructions](https://github.com/micro-manager/micro-manager/blob/main/doc/how-to-build.md#building-on-unix) for building micro-manager which will also build this repo.

You can avoid building the micro-manager parts and only build MMCore and the device adapters by using the following configure command: `./configure --without-java`.

The other thing to note is that `make install` may require the use of `sudo` unless you used the `--prefix=` option for configure.

#### Using your own fork
If you want to make changes to this repo then you need to update the submodule to point to your fork. After you set that up you can work in the submodule as if it were a standalone git repository.

From the top level of the `micro-manager` folder
```bash
git clone git@github.com:micro-manager/micro-manager.git
cd micro-manager
git submodule set-url mmCoreAndDevices <git url of your fork>
git submodule update --init --recursive
```
