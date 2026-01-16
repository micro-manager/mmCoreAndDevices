# MMCoreJ

MMCoreJ provides Java bindings for MMCore, Micro-Manager's hardware abstraction
layer, written in C++.

## Building MMCoreJ

The currently "supported" way to build MMCoreJ is to run the full Micro-Manager
build (using Ant on Windows and Autoconf/Automake elsewhere).

However, we are working to make MMCoreJ an independent project with its own
build system using Meson for the C++ parts and Maven for the Java parts.

You can test the new build system as follows. Note that SWIG version 2.x or 3.x
(not 4.x) must be available on the path.

```sh
# Copy over matching sources for MMDevice and MMCore (otherwise they will be
# fetched by Meson but may not be the correct versions).
rm -rf subprojects/mmdevice subprojects/mmcore
cp -R ../MMDeivce subprojects/mmdevice
cp -R ../MMCore subprojects/mmcore

meson setup --vsenv builddir  # --vsenv recommended on Windows
meson compile -C builddir
mvn package
```

This should place the built JARs in target/. There is a main JAR containing the
Java classes and a separate per-OS/architecture "natives" JAR containing the
native library.

## Native Library Loading

MMCoreJ requires a native library containing the C++ MMCore. The library is
loaded automatically when the `CMMCore` class is first accessed.

There is a "modern" and "legacy" search order for the native library. The
modern way can be used when MMCoreJ is built by Meson + Maven; when it fails,
loading falls back to the legacy search order. (As of this writing, MMStudio
still uses the legacy method.)

### Modern Search Order (for Meson build)

**Library file names:**

- **Linux**: `libmmcorej.so`
- **macOS**: `libmmcorej.dylib`
- **Windows**: `mmcorej.dll`

The native library `mmcorej` is located in the following order:

1. **JAR resource extraction**: The library is extracted from JAR resources at
   `/natives/<os>/<arch>/` where `<os>` is `linux`, `macos`, or `windows` and
   `<arch>` is `arm64` or `x86_64`. These are provided as "natives" classifier
   JARs (e.g., `MMCoreJ-<version>-natives-macos-arm64.jar`).

2. **System library path**: If not found as a JAR resource, `java.library.path`
   is searched.

If the above fails, the legacy loading mechanism is used (see below).

### Legacy Search Order

**Library file names:**

- **Linux**: `libMMCoreJ_wrap.so`
- **macOS**: `libMMCoreJ_wrap.jnilib` or `.dylib`
- **Windows**: `MMCoreJ_wrap.dll`

The legacy loading mechanism searches for `MMCoreJ_wrap` as follows:

1. **System property path**: If `mmcorej.library.path` is set, the library is
   loaded exclusively from that directory (no fallback to other locations).

2. **Relative to the JAR**: These paths are searched in order:
   - The directory containing the MMCoreJ JAR
   - The parent directory of the JAR
   - _[deprecated]_ `../mm/<platform>/` relative to the JAR
   - The grandparent directory of the JAR
   - _[deprecated]_ `../../mm/<platform>/` relative to the JAR

   Where `<platform>` is one of: `macosx`, `win32`, `win64`, `linux32`,
   `linux64`. The `mm/<platform>/` locations are obsolete and were never used
   by Micro-Manager.

3. **Compile-time path**: _[deprecated]_ A path optionally set at build time
   (Autoconf/Automake build only).

4. **System default**: `java.library.path` is used as a last resort.
