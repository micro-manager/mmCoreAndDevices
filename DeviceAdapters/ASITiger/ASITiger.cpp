///////////////////////////////////////////////////////////////////////////////
// FILE:          ASITiger.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   ASI Tiger MODULE_API items and ASIUtility class
//                Note this is for the "Tiger" MM set of adapters, which should
//                  work for more than just the TG-1000 "Tiger" controller
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
// BASED ON:      ASIStage.cpp, ASIFW1000.cpp, Arduino.cpp, and DemoCamera.cpp
//
//

#include "ASITiger.h"
#include "ASITigerComm.h"
#include "ASIXYStage.h"
#include "ASIZStage.h"
#include "ASIClocked.h"
#include "ASIFWheel.h"
#include "ASIScanner.h"
#include "ASIPiezo.h"
#include "ASICRISP.h"
#include "ASILED.h"
#include "ASIPLogic.h"
#include "ASIPmt.h"
#include "ASILens.h"
#include "ASIDac.h"
#include "ASIDacXYStage.h"
#include <cstdio>
#include <string>
#include "MMDevice.h"
#include "DeviceBase.h"
#include <iostream>
#include <sstream>
#include <vector>


// TODO add in support for other devices, each time modifying these places
//    name constant declarations in the corresponding .h file
//    MODULE_API MM::Device* CreateDevice(const char* deviceName) in this file
//    DetectInstalledDevices in TigerComm (or other hub)


///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////

MODULE_API void InitializeModuleData()
{
    RegisterDevice(g_TigerCommHubName, MM::HubDevice, g_TigerCommHubDescription);
}


MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
    std::string deviceStr = deviceName;
    if (deviceName == 0)
    {
        return 0;
    }
    else if (strcmp(deviceName, g_TigerCommHubName) == 0)
    {
        return new CTigerCommHub;
    }
    else if (deviceStr.compare(0, strlen(g_XYStageDeviceName), (std::string)g_XYStageDeviceName) == 0)
    {
        return new CXYStage(deviceName);
    }
    else if (deviceStr.compare(0, strlen(g_ZStageDeviceName), (std::string)g_ZStageDeviceName) == 0)
    {
        return new CZStage(deviceName);
    }
    else if (deviceStr.compare(0, strlen(g_FSliderDeviceName), (std::string)g_FSliderDeviceName) == 0)
    {
        return new CFSlider(deviceName);
    }
    else if (deviceStr.compare(0, strlen(g_TurretDeviceName), (std::string)g_TurretDeviceName) == 0)
    {
        return new CTurret(deviceName);
    }
    else if (deviceStr.compare(0, strlen(g_PortSwitchDeviceName), (std::string)g_PortSwitchDeviceName) == 0)
    {
        return new CPortSwitch(deviceName);
    }
    else if (deviceStr.compare(0, strlen(g_FWheelDeviceName), (std::string)g_FWheelDeviceName) == 0)
    {
        return new CFWheel(deviceName);
    }
    else if (deviceStr.compare(0, strlen(g_ScannerDeviceName), (std::string)g_ScannerDeviceName) == 0)
    {
        return new CScanner(deviceName);
    }
    else if (deviceStr.compare(0, strlen(g_MMirrorDeviceName), (std::string)g_MMirrorDeviceName) == 0)
    {
        return new CScanner(deviceName);  // this for compatibility with old config files
    }
    else if (deviceStr.compare(0, strlen(g_PiezoDeviceName), (std::string)g_PiezoDeviceName) == 0)
    {
        return new CPiezo(deviceName);
    }
    else if (deviceStr.compare(0, strlen(g_CRISPDeviceName), (std::string)g_CRISPDeviceName) == 0)
    {
        return new CCRISP(deviceName);
    }
    else if (deviceStr.compare(0, strlen(g_LEDDeviceName), (std::string)g_LEDDeviceName) == 0)
    {
        return new CLED(deviceName);
    }
    else if (deviceStr.compare(0, strlen(g_PLogicDeviceName), (std::string)g_PLogicDeviceName) == 0)
    {
        return new CPLogic(deviceName);
    }
    else if (deviceStr.compare(0, strlen(g_PMTDeviceName), (std::string)g_PMTDeviceName) == 0)
    {
        return new CPMT(deviceName);
    }
    else if (deviceStr.compare(0, strlen(g_LensDeviceName), (std::string)g_LensDeviceName) == 0)
    {
        return new CLens(deviceName);
    }
    else if (deviceStr.compare(0, strlen(g_DacXYStageDeviceName), (std::string)g_DacXYStageDeviceName) == 0)
    {
        return new CDACXYStage(deviceName);
    }
    else if (deviceStr.compare(0, strlen(g_DacDeviceName), (std::string)g_DacDeviceName) == 0)
    {
        return new CDAC(deviceName);
    }
    else
    {
        return 0;
    }
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
    delete pDevice;
}

