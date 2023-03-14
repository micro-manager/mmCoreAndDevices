#!/bin/sh
##
appDir=/Applications/Micro-Manager1.4
tempDir=/Users/Shared/MMTemp
if [ -d "$tempDir" ]; then
    echo temp dir exist
else
    echo temp dir not exist
    exit 1
fi

if [ -f "$tempDir/libmmgr_dal_Veroptics" ]; then
    echo lib
else
    exit 1
fi


if [ -f "$tempDir/libusb-1.0.0.dylib" ]; then
    echo libusb
else
exit 1
fi


echo $tempDir
echo $appDir
mv "$tempDir/libmmgr_dal_Veroptics" "$appDir"
mv "$tempDir/libusb-1.0.0.dylib" "$appDir"
rm -r "$tempDir"
exit 0