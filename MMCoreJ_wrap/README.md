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
