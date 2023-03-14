#!/bin/sh
##before install, to find the directory where the app is installed, and confirm the directory is exist, then copy
##file to directory. otherwise exit 1, and then the installer will exit with error

var=/Applications/Micro-Manager1.4

if [ -d "$var" ]; then
    echo app dir exist
else
    echo app dir not exist
    exit 1
fi

exit 0