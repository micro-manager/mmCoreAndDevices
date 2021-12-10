#!/bin/bash

# For use by Micro-Manager team members and continuous integration scripts
# only: run this script to clone (if necessary) and check out the matching
# commit of SecretDeviceAdapters.
#
# The SecretDeviceAdapters working tree is simply placed at the root of the
# mmCoreAndDevices working tree, _not_ as a submodule.
#
# If there is an existing clone of SecretDeviceAdapters, it is used regardless
# of its settings and which remote it points to.

set -e

if [ ! -f secret-device-adapters-commit ]; then
    echo The secret-device-adapters-commit file is missing.
    exit 1
fi

repo_root=$(dirname "$0")
pushd "$repo_root" >/dev/null
sda_sha=$(cat secret-device-adapters-commit)

if [ -d ./SecretDeviceAdapters ]; then
    echo Using existing SecretDeviceAdapters working tree
else
    if [ "$1" = "use_ssh" ]; then
        echo Cloning via SSH...
        git clone git@github.com:micro-manager/SecretDeviceAdapters.git
    else
        echo Cloning via HTTPS...
        git clone https://github.com/micro-manager/SecretDeviceAdapters.git
    fi
fi

cd SecretDeviceAdapters
git fetch
git checkout $sda_sha

popd >/dev/null
