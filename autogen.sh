#!/bin/sh


echo "Bootstrapping autoconf/automake build system ..." 1>&2

# Subdirectory must be present, even if empty, to prevent automake errors.
mkdir -p SecretDeviceAdapters

autoreconf --force --install --verbose

if [ $? -eq 0 ] # Command succeeded.
then
	echo "Bootstrapping complete; now you can run ./configure" 1>&2
else
	echo "Bootstrapping failed"
fi