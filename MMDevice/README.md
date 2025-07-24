# MMDevice

MMDevice defines the interface between Micro-Manager device adapters and
MMCore. It also contains default implementations (in `DeviceBase.h`) for device
adapter functionality.

The device interface is a C++ interface designed to use limited features so
that binary compatibility can be maintained between compiler versions and
settings (such as MSVC Debug vs Release runtimes). This relies on the fact that
platforms maintain a stable ABI (application binary interface) for C constructs
and C++ class and vtable layout.

## Repositories

This code is available in two Git repositories:

- https://github.com/micro-manager/mmCoreAndDevices, in subdirectory
  `MMDevice`. This is the official source for MMDevice.

- https://github.com/micro-manager/mmdevice, which is a mirror of the above; we
  do not currently accept pull requests or issues on this repository.

## Building MMDevice

MMDevice is usually built as part of Micro-Manager, using its build system
(MSVC on Windows, GNU Autotools on macOS/Linux); files for that are included in
this directory (but are not intended for use in isolation).

We are in the process of modularizing the build systems of Micro-Manager's
components, and MMDevice also has cross-platform build scripts using Meson;
these can be used to build just MMDevice and run its unit tests:

```sh
git clone https://github.com/micro-manager/mmdevice.git
cd mmdevice
meson setup --vsenv builddir  # --vsenv recommended on Windows
meson compile -C builddir
meson test -C builddir
```
