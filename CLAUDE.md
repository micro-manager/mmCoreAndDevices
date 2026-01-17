# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Micro-Manager is Open Source software for microscope control. This repository contains the C++ core components:

- **MMCore**: Device abstraction layer providing the top-level API for applications to control microscope hardware (see `MMCore/MMCore.h`)
- **MMDevice**: Device API defining the interface that device adapters must implement (see `MMDevice/MMDevice.h` and `MMDevice/DeviceBase.h`)
- **MMCoreJ_wrap**: SWIG-generated Java wrapper for MMCore
- **DeviceAdapters/**: Publicly accessible device adapter source code
- **SecretDeviceAdapters/**: Non-public device adapters (separate private repository referenced via git submodule)

### Architecture

MMCore implements two APIs:
- **Top API** (`MMCore/MMCore.h`): Used by applications to control abstracted microscope hardware. Public methods use lowercase naming (e.g., `getConfiguration()`) to be language-agnostic for SWIG wrapping.
- **Bottom API** (`MMDevice/MMDevice.h`): Implemented by device adapters to communicate with physical devices.

Both APIs maintain a pure C-interface for cross-platform compatibility and dynamic loading.

Device adapters are compiled into dynamically loadable libraries (`.dll`/`.so`/`.dylib`) that MMCore loads at runtime.

## Build Commands

### Windows (Visual Studio)

Primary solution file: `micromanager.sln`

Build using MSBuild or Visual Studio. MSBuild properties can be overridden via `/property:name=value`:
- `MM_3RDPARTYPUBLIC`: Path to public 3rd party dependencies (default: `../../3rdpartypublic`)
- `MM_3RDPARTYPRIVATE`: Path to private 3rd party dependencies (default: `../../3rdparty`)
- `MM_BOOST_INCLUDEDIR`: Boost include directory
- `MM_BOOST_LIBDIR`: Boost lib directory
- `MM_BUILDDIR`: Build artifacts directory

See `buildscripts/VisualStudio/MMCommon.props` for default values.

### macOS and Linux (Autotools)

Standard GNU Autotools workflow:

```bash
# From repository root
./configure                    # Configure build (detects available SDKs)
make                          # Build all components
make install                  # Install (may require sudo)
```

Build only MMCore and device adapters (skip Java components):
```bash
./configure --without-java
make
```

The build system auto-detects which device adapters can be built based on available vendor SDKs and headers.

### Experimental Meson Build (justfile)

Requires: `just`, `meson`, `ninja`

```bash
just                          # List available commands
just build-mmdevice          # Build MMDevice
just build-mmcore            # Build MMCore (depends on MMDevice)
just test-mmdevice           # Test MMDevice
just test-mmcore             # Test MMCore
just test                    # Run all tests
just clean                   # Clean build artifacts
```

Quick one-liner with uv:
```bash
uvx --from rust-just --with meson --with ninja just test
```

## Testing

Tests use Catch2 framework and are located in `unittest/` subdirectories:
- `MMCore/unittest/` - Core tests
- `MMDevice/unittest/` - Device API tests
- Some device adapters have their own tests (e.g., `DeviceAdapters/HamiltonMVP/unittest/`)

Run tests via meson (see justfile commands above) or through IDE test runners.

## Device Adapter Development

### Base Classes

Device adapters inherit from specialized base classes in `MMDevice/DeviceBase.h`:
- `CCameraBase<T>` - Cameras
- `CShutterBase<T>` - Shutters
- `CStageBase<T>` - Z-stages
- `CXYStageBase<T>` - XY stages
- `CStateDeviceBase<T>` - Filter wheels, turrets
- `CGenericBase<T>` - Generic devices
- `CAutoFocusBase<T>` - Autofocus devices
- `CImageProcessorBase<T>` - Image processors
- `CHubBase<T>` - Device hubs (multi-device controllers)

All base classes use the CRTP pattern: `class MyDevice : public CCameraBase<MyDevice>`

### Property System

Device configuration uses a property-based system. Properties are created in `Initialize()`:
```cpp
// Create action handler
CPropertyAction* pAct = new CPropertyAction(this, &MyDevice::OnProperty);
// Create property with allowed values
CreateProperty("PropertyName", "DefaultValue", MM::String, false, pAct);
AddAllowedValue("PropertyName", "Value1");
AddAllowedValue("PropertyName", "Value2");
```

### Vendor SDK Integration

Many device adapters depend on proprietary vendor SDKs:
- SDK headers/libraries are in `../../3rdpartypublic/` (public) or `../../3rdparty/` (private)
- Build system automatically disables adapters when SDKs are unavailable
- Each adapter's `configure.ac` or `.vcxproj` specifies SDK requirements

Example adapters to study:
- `DemoCamera` - Full-featured reference implementation without hardware dependencies
- `Arduino` - Simple serial communication pattern
- `PVCAM` - Camera with vendor SDK integration
- `ASITiger` - Multi-device hub pattern

### Threading Considerations

Camera adapters typically use separate threads for image acquisition:
- Implement proper locking around shared state
- Use MMDevice threading primitives (`MMThreadLock` in `DeviceThreads.h`)
- Circular buffers handle asynchronous image streaming (see `MMCore/CircularBuffer.h`)

### Error Handling

Use standard error codes from `MMDevice/MMDeviceConstants.h`:
- `DEVICE_OK` - Success
- `DEVICE_ERR` - Generic error
- Define adapter-specific error codes starting at 100+

Common error message constants are defined in `MMDevice/DeviceBase.h` (e.g., `g_Msg_SERIAL_TIMEOUT`).

## Key Files for Understanding the System

- `MMCore/MMCore.h` - Main public API
- `MMDevice/MMDevice.h` - Device interface definitions
- `MMDevice/DeviceBase.h` - Base classes and utility templates
- `DeviceAdapters/DemoCamera/DemoCamera.cpp` - Reference implementation
- `buildscripts/VisualStudio/MMCommon.props` - Windows build configuration defaults

## Coding style
- All indents are 3 spaces (no tab characters).
- Curly braces open in the same line in Java and in a new line in C++ (see examples).
- Class names begin with uppercase, and with each word capitalized, e.g. MyFirstClass.
- Function names use the same convention except that in Java they begin with lowercase and in C++ with uppercase, e.g. MyFunc() in C++ and myFunc() in Java.
- All variables begin with lower case, e.g. myVar.
- Class member variables begin with lowercase and end with underscore, e.g. memberVar_.
- Do not use this->memberVar_ idiom unless it is absolutely necessary to avoid confusion.
- Static constants in C++: const char* const g_Keyword_Description.
- Static constants in Java: static final String METADATA_FILE_NAME.
- if/else, for, and while statements should include curly braces, even if they are only followed by a single line.

## CRITICAL: File Editing on Windows
MANDATORY: Always use Backslashes on Windows for File Paths
When using Edit or MultiEdit on WIndows, you MUST use backslashes (\) in the file paths, NOT forward slashes (/).
WRONG:
Edit(file_path: "D:/repos/project/file.tsx", ...)
MultiEdit(file_path: "D:/repos/project/file.tsx", ...)
CORRECT:
Edit(file_path: "D:\repos\project\file.tsx", ...)
MultiEdit(file_path: "D:\repos\project\file.tsx", ...)
