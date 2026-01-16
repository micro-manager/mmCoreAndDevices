#!/usr/bin/env bash

# This script is designed to run inside a manylinux_2_28 container.

set -euxo pipefail

yum install -y java-1.8.0-openjdk-devel

export PATH="/opt/python/cp314-cp314/bin:$PATH"
pip install meson ninja 'swig<4'

meson setup builddir --buildtype=release
meson compile -C builddir
