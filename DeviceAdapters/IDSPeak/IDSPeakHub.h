////////////////////////////////////////////////////////////////////////////////
// FILE:          IDSPeakHub.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Driver for IDS peak series cameras
//
//                Based on IDS peak SDK and Micro-manager DemoCamera example
//                tested with SDK version 2.5
//
// AUTHOR:        Lars Kool, Institut Pierre-Gilles de Gennes
//
// YEAR:          2025
//                
// VERSION:       2.0.1
//
// LICENSE:       This file is distributed under the BSD license.
//                License text is included with the source distribution.
//
//                This file is distributed in the hope that it will be useful,
//                but WITHOUT ANY WARRANTY; without even the implied warranty
//                of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//                IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//                CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//                INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.
//
// LAST UPDATE:   13.11.2025 LK
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "DeviceBase.h"

extern const char* g_IDSPeakCameraName;
extern const char* g_IDSPeakHubName;

class IDSPeakHub : public HubBase<IDSPeakHub>
{
public:
    IDSPeakHub();
    ~IDSPeakHub();

    // Device API
    void GetName(char* pName) const;
    bool Busy();
    int Initialize();
    int Shutdown();

    // HUB api
    int DetectInstalledDevices();

private:
    // Data members
    bool initialized_;
    bool busy_;
    int nCameras_;
};
