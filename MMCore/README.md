# MMCore

MMCore is Micro-Manager's hardware abstraction layer, written in C++.

It provides APIs to load device adapters, control devices, and acquire data. It
also implements buffering for image acquisition.

Currently, MMCore is mainly intended to be used from Python (via pymmcore) or
Java (via MMCoreJ), but it can be used as a C++ library as well.

## Repositories

This code is available in two Git repositories:

- https://github.com/micro-manager/mmCoreAndDevices, in subdirectory `MMCore`.
  This is the official source for MMCore.

- https://github.com/micro-manager/mmcore, which is a mirror of the above; we
  do not currently accept pull requests or issues on this repository.

(The eventual goal is for MMCore to be removed from mmCoreAndDevices and for
the `mmcore` repository to become the official source. Until then,
mmCoreAndDevices remains the official source.)

## Building MMCore

The "supported" ways to use MMCore are via MMCoreJ (built as part of
Micro-Manager) or via pymmcore(-plus). This directory contains files for the
traditional build systems used by Micro-Manager (MSVC on Windows, GNU Autotools
on macOS/Linux), but those are not designed to allow building just MMCore in
isolation.

We are in the process of modularizing the build systems of Micro-Manager's
components, and MMCore now also has cross-platform build scripts using Meson.

This can be used to build just MMCore using the `mmcore` repository:

```sh
git clone https://github.com/micro-manager/mmcore.git
cd mmcore
meson setup --vsenv builddir  # --vsenv recommended on Windows
meson compile -C builddir
meson test -C builddir
```

Note that the above way of building will use the latest `mmdevice` as a
dependency. To control the exact version of `mmdevice` to use, you will need to
copy it into `subprojects/` before running Meson (as is done in
mmCoreAndDevices's CI).

If using MMCore as a C++ library, it should be noted that it needs to be built
as a static library. Building as a DLL or shared library would require defining
which symbols are public, and this is not currently available.
