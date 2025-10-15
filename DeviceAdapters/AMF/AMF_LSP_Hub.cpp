///////////////////////////////////////////////////////////////////////////////
// FILE:          AMF_LSP.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Device adapter for AMF LSP devices
//                 
// AUTHOR:        Lars Kool, Institut Pierre-Gilles de Gennes, Paris, France
//
// YEAR:          2025
//                
// VERSION:       0.1
//
// LICENSE:       This file is distributed under the BSD license.
//                License text is included with the source distribution.
//
//                This file is distributed in the hope that it will be useful,
//                but WITHOUT ANY WARRANTY; without even the implied warranty
//                of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//                IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//                CONTRIBUTORS BE   LIABLE FOR ANY DIRECT, INDIRECT,
//                INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.
//
//LAST UPDATE:    09.09.2025 LK

#include "AMF_LSP_Hub.h"
#include "AMF_RVM.h"
#include "AMF_Commands.h"
#include "AMF_LSP_Pump.h"


///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////

MODULE_API void InitializeModuleData()
{
    RegisterDevice(AMF_LSP_Hub_Name, MM::HubDevice, "Hub for AMF LSP devices.");
    RegisterDevice(AMF_RVM_Name, MM::StateDevice, "AMF Rotary Valve Module.");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
    if (!deviceName) {
        return 0; // Trying to create nothing, return nothing
    }
    if (strcmp(deviceName, AMF_LSP_Hub_Name) == 0) {
        return new AMF_LSP_Hub(); // Create Hub
    }
    if (strncmp(deviceName, AMF_RVM_Name, strlen(AMF_RVM_Name)) == 0) {
        return new AMF_RVM(); // Create valve
    }
    if (strncmp(deviceName, AMF_LSP_Pump_Name, strlen(AMF_LSP_Pump_Name)) == 0) {
        return new AMF_LSP_Pump(); // Create pump
    }
    return 0; // If an unexpected name is provided, return nothing
}

MODULE_API void DeleteDevice(MM::Device* pDevice) {
    delete pDevice;
}

///////////////////////////////////////////////////////////////////////////////
// AMF Hub API
///////////////////////////////////////////////////////////////////////////////

AMF_LSP_Hub::AMF_LSP_Hub() {
    // COM Port
    CPropertyAction* pAct = new CPropertyAction(this, &AMF_LSP_Hub::OnPort);
    CreateStringProperty(MM::g_Keyword_Port, "Undefined", false, pAct, true);
}

int AMF_LSP_Hub::Initialize() {
    if (initialized_) { return DEVICE_OK; }

    // Not much to do here
    initialized_ = true;
    return DEVICE_OK;
}

void AMF_LSP_Hub::GetName(char* pName) const {
	CDeviceUtils::CopyLimitedString(pName, AMF_LSP_Hub_Name);
}

int AMF_LSP_Hub::DetectInstalledDevices() {
    ClearInstalledDevices();
    InitializeModuleData();

    MM::Device* pDev = CreateDevice(AMF_RVM_Name);
    if (pDev) { AddInstalledDevice(pDev); }
    else {
        LogMessage("Could not create AMF RVM device.");
        return DEVICE_ERR;
    }

    pDev = CreateDevice(AMF_LSP_Pump_Name);
    if (pDev) { AddInstalledDevice(pDev); }
    else {
        LogMessage("Could not create AMF LSP Pump.");
        return DEVICE_ERR;
    }
    return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// AMF Hub API
///////////////////////////////////////////////////////////////////////////////

int AMF_LSP_Hub::OnPort(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(port_.c_str());
    }
    else if (eAct == MM::AfterSet)
    {
        if (!initialized_)
            pProp->Get(port_);
    }

    return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// AMF LSP API
///////////////////////////////////////////////////////////////////////////////

int AMF_LSP_Hub::GetPort(std::string& port) {
    port = port_;
    return DEVICE_OK;
}
