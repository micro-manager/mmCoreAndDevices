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
