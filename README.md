# mmCoreAndDevices
The c++ code at the core of the Micro-Manager project.

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
*TODO*

### Branches
This repository consists of two primary branches: `main` and `privateMain`.  
Changes made to the `main` branch are merged into `privateMain`. The only difference between the two branches is that `privateMain` includes a private repository as a submodule containing device adapter code which is not permitted to be made publicly available.  
Note: Please do not modify public files in the `privateMain` branch. Changes made in `main` should be merged into `privateMain` but never the other way around.  
Note: After checking out the `privateMain` branch it may be necessary to run the following command to make sure that all submodules are up to date in your local copy: `git submodule update --init --recursive`.
