///////////////////////////////////////////////////////////////////////////////
// FILE:          AMF_Hub.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Hub for AMF Devices
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
//LAST UPDATE:    04.04.2025 LK

#include "AMF_Hub.h"
#include "AMF_RVM.h"

///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////

MODULE_API void InitializeModuleData()
{
    RegisterDevice(g_AMF_Hub_Name, MM::HubDevice, "Hub for AMF devices.");
    RegisterDevice(g_AMF_RVM_Name, MM::StateDevice, "AMF Rotary Valve Module.");
    RegisterDevice(g_AMF_Test_Name, MM::GenericDevice, "AMF Test Device");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
    if (!deviceName) {
        return 0; // Trying to create nothing, return nothing
    }
    if (strcmp(deviceName, g_AMF_Hub_Name) == 0) {
        return new AMF_Hub(); // Create Hub
    }
    if (strncmp(deviceName, g_AMF_RVM_Name, strlen(g_AMF_RVM_Name)) == 0) {
        return new AMF_RVM(); // Create channel
    }
    return 0; // If an unexpected name is provided, return nothing
}

MODULE_API void DeleteDevice(MM::Device* pDevice) {
    delete pDevice;
}

///////////////////////////////////////////////////////////////////////////////
// AMF Hub API
///////////////////////////////////////////////////////////////////////////////

AMF_Hub::AMF_Hub() {
    // Number of RVMs connected
    CPropertyAction* pAct = new CPropertyAction(this, &AMF_Hub::OnRVMCount);
    CreateIntegerProperty("Number of RVMs", 0, false, pAct, true);

    // Number of LSPs connected (not yet supported)
    //pAct = new CPropertyAction(this, &AMF_Hub::OnLSPCount);
    //nRet = CreateIntegerProperty("Number of LSPs", 0, false, pAct, true);
    //assert(nRet == DEVICE_OK);
}

int AMF_Hub::Initialize() {
    initialized_ = true;
    return DEVICE_OK;
}

void AMF_Hub::GetName(char* pName) const {
	CDeviceUtils::CopyLimitedString(pName, g_AMF_Hub_Name);
}

int AMF_Hub::DetectInstalledDevices() {
    ClearInstalledDevices();
    InitializeModuleData();

    // Add RVMs
    for (int i = 0; i < nRVM_; i++) {
        MM::Device* pDev = CreateDevice(g_AMF_RVM_Name);
        LogMessage("Creating RVM");
        if (pDev) { AddInstalledDevice(pDev); }
        else { LogMessage("RVM creation failed."); }
    }

    return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// AMF Hub Action Handlers
///////////////////////////////////////////////////////////////////////////////

int AMF_Hub::OnRVMCount(MM::PropertyBase* pProp, MM::ActionType eAct) {
    if (eAct == MM::BeforeGet) {
        pProp->Set(nRVM_);
    }
    else if (eAct == MM::AfterSet) {
        pProp->Get(nRVM_);
    }
    return DEVICE_OK;
}

int AMF_Hub::OnLSPCount(MM::PropertyBase* pProp, MM::ActionType eAct) {
    if (eAct == MM::BeforeGet) {
        pProp->Set(nLSP_);
    }
    else if (eAct == MM::AfterSet) {
        pProp->Get(nLSP_);
    }
    return DEVICE_OK;
}

