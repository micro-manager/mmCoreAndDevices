#!/bin/sh

## Remove
Count=0

appPath=/Applications/Micro-Manager1.4
echo $appPath

if [ -d "$appPath" ]; then
    echo app dir exist
else
    echo app dir not exist
    exit 1
fi

#echo "Do you want to remove Micro-Manager Veroptics driver?<y/N>"
read -p "Do you want to remove Micro-Manager Veroptics driver?<y/N>" prompt
if [[ "$prompt" == "y" || "$prompt" == "Y" ]]
then
echo start remove
else
	exit 0
fi

Path="$appPath/libmmgr_dal_Veroptics"
if [ -f "$Path" ]; then
    sudo rm "$Path"
    Count=$[Count+1]
fi

Path="$appPath/libusb-1.0.0.dylib"
if [ -f "$Path" ]; then
    sudo rm "$Path"
    Count=$[Count+1]
fi

## >0
if [ $Count -gt 0 ]; then
    echo "x2camera ASICamera driver is removed($Count files)"
else
    echo no file need to remove
fi

exit 0