# This Justfile defines tasks for building and testing MMDevice and MMCore projects.
# It is only for the experimental meson-build system, and is not yet used in production.
#
# To use it, first install Just: https://github.com/casey/just, for example:
#   $ uv tool install just
#   # or
#   $ brew install just
# Make sure you also have meson and ninja installed, for example:
#   $ uv tool install meson && uv tool install ninja
#   # or
#   $ brew install meson ninja
#
# Then run any command with `just <command>`.  Or `just --list` to see all available commands.

# Build MMDevice
build-mmdevice:
    meson setup MMDevice/builddir MMDevice --reconfigure --buildtype debug
    meson compile -C MMDevice/builddir

# Build MMCore (depends on build-mmdevice)
build-mmcore: build-mmdevice
    cp -R MMDevice MMCore/subprojects
    meson setup MMCore/builddir MMCore --reconfigure --buildtype debug
    meson compile -C MMCore/builddir

# Test MMDevice (depends on build-mmdevice)
test-mmdevice:
    if [ ! -d MMDevice/builddir ]; then just build-mmdevice; fi
    meson test -C MMDevice/builddir --print-errorlogs

# Test MMCore (depends on build-mmcore)
test-mmcore:
    if [ ! -d MMCore/builddir ]; then just build-mmcore; fi
    meson test -C MMCore/builddir

# Run all tests
test: test-mmdevice test-mmcore

# Clean build artifacts
clean:
    rm -rf MMDevice/builddir
    rm -rf MMCore/builddir
    rm -rf MMCore/subprojects/MMDevice
