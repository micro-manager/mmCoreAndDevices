////////////////////////////////////////////////////////////////////////////////
// FILE:          IDSPeakCamera.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Driver for IDS peak series of USB cameras
//
//                Based on IDS peak SDK and Micro-manager DemoCamera example
//                tested with SDK version 2.5
//                Requires Micro-manager Device API 71 or higher!
//                
// AUTHOR:        Lars Kool, Institut Pierre-Gilles de Gennes
//
// YEAR:          2023
//                
// VERSION:       1.1.1
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
// LAST UPDATE:   03.12.2024 LK
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "IDSPeakHub.h"
#include "IDSPeakCamera.h"
#include "DeviceBase.h"
#include <string>
#include <iostream>

#include <peak/peak.hpp>
#include <peak_ipl/peak_ipl.hpp>

const char* g_IDSPeakCameraName = "IDSCam";
const char* g_IDSPeakHubName = "IDS Peak Hub";

///////////////////////////////////////////////////////////////////////////////
//  MMDevice API
///////////////////////////////////////////////////////////////////////////////

MODULE_API void InitializeModuleData()
{
    RegisterDevice(g_IDSPeakHubName, MM::HubDevice, "Hub for IDS cameras");
    RegisterDevice(g_IDSPeakCameraName, MM::CameraDevice, "Device adapter for IDS peak cameras");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
    std::cout << deviceName << std::endl;
    if (!deviceName) {
        return 0; // Trying to create nothing, return nothing
    }
    if (strcmp(deviceName, g_IDSPeakHubName) == 0) {
        return new IDSPeakHub(); // Create Hub
    }
    if (strncmp(deviceName, g_IDSPeakCameraName, strlen(g_IDSPeakCameraName)) == 0) {
        std::string name_s = deviceName;
        std::string substr = name_s.substr(strlen(g_IDSPeakCameraName), strlen(deviceName));
        int deviceIdx = std::stoi(substr);
        return new CIDSPeakCamera(deviceIdx); // Create channel
    }
    return 0; // If an unexpected name is provided, return nothing
}

MODULE_API void DeleteDevice(MM::Device* device)
{
    delete device;
}

////////////////////////////////////////////////////////////////////////////////
// IDSPeakHub class
// Hub for IDS Peak cameras
////////////////////////////////////////////////////////////////////////////////

IDSPeakHub::IDSPeakHub() :
    initialized_(false),
    busy_(false)
{}

IDSPeakHub::~IDSPeakHub()
{
    Shutdown();
}

////////////////////////////////////////////////////////////////////////////////
// MM::Device API
////////////////////////////////////////////////////////////////////////////////

/**
* Copies the devicename to the provided char buffer.
* Required by the MM::Device API.
* @params name - Pointer to allocated char buffer.
*/
void IDSPeakHub::GetName(char* name) const
{
    // Return the name used to refer to this device adapter
    CDeviceUtils::CopyLimitedString(name, g_IDSPeakHubName);
}

/**
* Returns whether the Hub is busy executing a command.
* This is never the case, so it always returns false.
* @returns Boolean - Always returns false
*/
bool IDSPeakHub::Busy()
{
    return false;
}


/**
* Initializes the hub and the IDS library, and detects the number of IDS cameras connected.
* Note that it does not initialize any of the cameras, this is only done after the user
* selects the camera in the Hardware Configuration Wizard.
* @returns Integer status code - Returns DEVICE_OK on success
*/
int IDSPeakHub::Initialize()
{
    if (initialized_) { return DEVICE_OK; }

    // Name
    int ret = CreateStringProperty(MM::g_Keyword_Name, g_IDSPeakHubName, true);
    if (DEVICE_OK != ret) { return ret; }

    // Description
    ret = CreateStringProperty(MM::g_Keyword_Description, "Hub for IDS Peak cameras", true);
    if (DEVICE_OK != ret) { return ret; }

    try
    {
        // The initialize function is reference counted, so no reason to keep track of
        // it ourselves.
        peak::Library::Initialize();
		auto& deviceManager = peak::DeviceManager::Instance();
		deviceManager.Update();
		nCameras_ = (int)deviceManager.Devices().size();
    }
    catch (std::exception& e)
    {
        LogMessage("IDS exception: Could not initialize the IDS library.");
        LogMessage(e.what());
        return DEVICE_ERR;
    }

    initialized_ = true;
    return DEVICE_OK;
}

/**
* Shuts down (unloads) the hub. 
* @returns Integer status code - Returns DEVICE_OK on success
*/
int IDSPeakHub::Shutdown()
{
    if (!initialized_) { return DEVICE_OK; }
    peak::Library::Close();
    initialized_ = false;
    return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////
// MM::Hub API
////////////////////////////////////////////////////////////////////////////////

/**
* Detects the number of installed devices.
* Since the Initialize function already determined the number of connected cameras
* this function just generates that many camera devices
* @returns Integer status code - Returns DEVICE_OK on success
*/
int IDSPeakHub::DetectInstalledDevices()
{
    ClearInstalledDevices();
    for (int i = 0; i < nCameras_; i++) {
        MM::Device* device = new CIDSPeakCamera(i);
        if (device) {
            AddInstalledDevice(device);
        }
    }
    return DEVICE_OK;
}
