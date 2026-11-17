///////////////////////////////////////////////////////////////////////////////
// FILE:          ASITigerComm.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   ASI TIGER controller adapter
//
// COPYRIGHT:     Applied Scientific Instrumentation, Eugene OR
//
// LICENSE:       This file is distributed under the BSD license.
//
//                This file is distributed in the hope that it will be useful,
//                but WITHOUT ANY WARRANTY; without even the implied warranty
//                of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//                IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//                CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//                INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.
//
// AUTHOR:        Jon Daniels (jon@asiimaging.com) 09/2013
//

#pragma once

#include "ASIHub.h"
#include "MMDevice.h"
#include "DeviceBase.h"

///////////////////////////////////////////////////////
// ***** TIGER-specific *******************************
///////////////////////////////////////////////////////

class CTigerCommHub : public ASIHub {
public:
    CTigerCommHub();
    ~CTigerCommHub() = default;

    // Device API
    int Initialize();

    // Hub API
    bool SupportsDeviceDetection();
    MM::DeviceDetectionStatus DetectDevice();
    int DetectInstalledDevices();

private:
    int TalkToTiger();
};
