# CMMCore
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
    OutDir: The directory that build artifacts will be stored in.
    IntDir: The directory that biuld intermediates to be stored in.
	
To see the default values of each property please view `MMCommon.props`