///////////////////////////////////////////////////////////////////////////////
// FILE:          FluigentPressureController.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Device adapter for Fluigent pressure controllers
//                
// AUTHOR:        Lars Kool, Institut Pierre-Gilles de Gennes
//
// YEAR:          2025
//                
// VERSION:       1.0
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
//LAST UPDATE:    08.04.2025 LK

#include "FluigentPressureController.h"

#include "DeviceBase.h"
#include "DeviceThreads.h"
#include "ModuleInterface.h"

#include <string>
#include <vector>

#include "fgt_SDK.h"

const char* g_FluigentChannelName = "FluigentChannel";
const char* g_FluigentHubName = "FluigentHub";
const char* g_Calibrate = "Calibrate";
const char* g_Imposed = "Imposed Pressure";
const char* g_Measured = "Measured Pressure";

///////////////////////////////////////////////////////////////////////////////
//  MMDevice API
///////////////////////////////////////////////////////////////////////////////


MODULE_API void InitializeModuleData()
{
    RegisterDevice(g_FluigentHubName, MM::HubDevice, "Hub for Fluigent pressure controllers");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
    if (!deviceName)
    { 
        return 0; // Trying to create nothing, return nothing
    }
    const std::string name(deviceName);
    if (name == g_FluigentHubName)
    {
        return new FluigentHub(); // Create Hub
    }
    const std::size_t chanPrefixLen = std::strlen(g_FluigentChannelName);
    if (name.substr(0, chanPrefixLen) == g_FluigentChannelName)
    {
        int idx = std::stoi(name.substr(chanPrefixLen));
        return new FluigentChannel(idx);
    }
    return 0; // If an unexpected name is provided, return nothing
}

MODULE_API void DeleteDevice(MM::Device* device)
{
    delete device;
}

///////////////////////////////////////////////////////////////////////////////
// FluigentHub class
// Hub for Fluigent Pressure Controller devices
///////////////////////////////////////////////////////////////////////////////

FluigentHub::FluigentHub() :
    initialized_(false),
    busy_(false),
    nDevices_(0),
    nChannels_(0),
    errorCode_(0),
    calibrate_("None")
{}

FluigentHub::~FluigentHub()
{
    if (initialized_)
        fgt_close();
}

///////////////////////////////////////////////////////////////////////////////
// FluigentHub class
// MMDevice API
///////////////////////////////////////////////////////////////////////////////

void FluigentHub::GetName(char* name) const
{
    // Return the name used to refer to this device adapter
    std::string deviceName = g_FluigentHubName;
    CDeviceUtils::CopyLimitedString(name, g_FluigentHubName);
}

int FluigentHub::Initialize()
{
    // Name
    int ret = CreateStringProperty(MM::g_Keyword_Name, g_FluigentHubName, true);
    if (DEVICE_OK != ret)
        return ret;

    // Description
    ret = CreateStringProperty(MM::g_Keyword_Description, "Hub for Fluigent pressure controllers", true);
    if (DEVICE_OK != ret)
        return ret;

    // Detect number of pressure controllers (MFCS, MFCS-EZ, LINE-UP)
    unsigned char nDevicesDetected;
    nDevicesDetected = fgt_detect(SNs_, instrumentTypes_);
    for (int i = 0; i < nDevicesDetected; i++)
    {
        switch (instrumentTypes_[i])
        {
        case 1: // MFCS
            nDevices_++;
            break;
        case 2: // MFCS-EZ
            nDevices_++;
            break;
        case 4: // LineUP
            nDevices_++;
            break;
        default: // Remove other devices from list
            SNs_[i] = 0;
        }
    }
    LogMessage("Number of devices detected: " + std::to_string(nDevicesDetected));

    // Initialize pressure controllers
    errorCode_ = fgt_initEx(SNs_);
    if (errorCode_ != 0) { return DEVICE_ERR; }

    // Set system-wide pressure unit to the unit chosen during startup (default is kPa)
    fgt_set_sessionPressureUnit((char*)"kPa");

    // Detect total number of pressure channels
    unsigned char nChannelsTemp;
    errorCode_ = fgt_get_pressureChannelCount(&nChannelsTemp);
    nChannels_ = (int)nChannelsTemp;
    LogMessage("Number of channels detected: " + std::to_string(nChannels_));

    // Get channel information
    fgt_get_pressureChannelsInfo(channelInfo_);

    // Create calibration property
    std::vector<std::string> allowedNames = { "All", "None" };
    for (size_t i = 0; i < nChannels_; i++) {
        if (channelInfo_[i].InstrType != fgt_INSTRUMENT_TYPE::None) {
            allowedNames.push_back(std::to_string(i));
        }
    }
    CPropertyAction* pAct = new CPropertyAction(this, &FluigentHub::OnCalibrate);
    ret = CreateStringProperty("Calibrate", "None", false, pAct);
    SetAllowedValues("Calibrate", allowedNames);

    initialized_ = true;
    return DEVICE_OK;
}

int FluigentHub::Shutdown()
{
    // Close the communication with Fluigent devices
    fgt_close();
    initialized_ = false;
    return DEVICE_OK;
}

int FluigentHub::DetectInstalledDevices()
{
    // Automatically add all discovered pumps
    ClearInstalledDevices();
    for (int i = 0; i < nChannels_; i++) {
        MM::PressurePump* pPump = new FluigentChannel(i);
        if (pPump)
            AddInstalledDevice(pPump);
    }
    return DEVICE_OK;
}

int FluigentHub::OnCalibrate(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    int ret = DEVICE_ERR;
    switch (eAct)
    {
    case MM::AfterSet:
    {
        pProp->Get(calibrate_);
        if (calibrate_ == "All") {
            // Calibrate all channels, one by one
            for (int i = 0; i < nChannels_; i++) {
                fgt_calibratePressure(channelInfo_[i].indexID);
            }
        }
        else if (calibrate_ != "None") {
            // Calibrate specific channel
            fgt_calibratePressure(channelInfo_[atoi(calibrate_.c_str())].indexID);
        }
        calibrate_ = "None";
        ret = DEVICE_OK;
    }break;
    case MM::BeforeGet:
    {
        // Refresh the displayed value
        ret = DEVICE_OK;
        pProp->Set(calibrate_.c_str());
    }break;
    }
    return ret;
    return DEVICE_OK;
}

int FluigentHub::GetNChannels(int& nChannels) {
    nChannels = nChannels_;
    return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// FluigentChannel class
// Fluigent Pressure Controller Channel
///////////////////////////////////////////////////////////////////////////////

FluigentChannel::FluigentChannel(int idx) :
    initialized_(false),
    busy_(false),
    errorCode_(0)
{
    idx_ = idx;
};

///////////////////////////////////////////////////////////////////////////////
// FluigentChannel class
// MMDevice API
///////////////////////////////////////////////////////////////////////////////

void FluigentChannel::GetName(char* name) const
{
    // Return the name used to refer to this device adapter
    CDeviceUtils::CopyLimitedString(name, (g_FluigentChannelName + std::to_string(idx_)).c_str());
}

int FluigentChannel::Initialize()
{
    // Name
    int ret = CreateStringProperty(MM::g_Keyword_Name, (g_FluigentChannelName + std::to_string(idx_)).c_str(), true);
    if (DEVICE_OK != ret)
        return ret;

    // Description
    ret = CreateStringProperty(MM::g_Keyword_Description, "Fluigent Pressure Controller Channel", true);
    if (DEVICE_OK != ret)
        return ret;

    // Link with Hub
    CreateHubIDProperty();
    FluigentHub* pHub = static_cast<FluigentHub*>(GetParentHub());
    if (pHub)
    {
        char hubLabel[MM::MaxStrLength];
        pHub->GetLabel(hubLabel);
        SetParentID(hubLabel); // for backward comp.
    }
    else
        LogMessage("No Hub found!");

    // Get channelinfo for this specific channel
    fgt_CHANNEL_INFO tempInfo[256];
    fgt_get_pressureChannelsInfo(tempInfo);
    channelInfo_ = tempInfo[idx_];

    // Serial Number
    ret = CreateIntegerProperty("Serial Number", channelInfo_.indexID, true);
    if (DEVICE_OK != ret)
        return ret;

    // Imposed pressure
    GetPressureLimits(Pmin_, Pmax_);
    CPropertyAction* pAct = new CPropertyAction(this, &FluigentChannel::OnImposedPressure);
    ret = CreateFloatProperty("Imposed Pressure", 0, false, pAct);
    SetPropertyLimits("Imposed Pressure", Pmin_, Pmax_);

    // Measured pressure
    pAct = new CPropertyAction(this, &FluigentChannel::OnMeasuredPressure);
    ret = CreateFloatProperty("Measured Pressure", 0, true, pAct);
    return ret;
}

int FluigentChannel::Stop() {
    SetPressureKPa(0);
    Pimp_ = 0;
    return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// FluigentChannel class
// Action handlers
///////////////////////////////////////////////////////////////////////////////

int FluigentChannel::OnImposedPressure(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    int ret = DEVICE_ERR;
    switch (eAct)
    {
    case MM::AfterSet:
    {
        // Sets a new pressure, and immediately updates the meausured pressure
        double Ptemp;
        pProp->Get(Ptemp);
        Pimp_ = (float)Ptemp;
        SetPressureKPa(Pimp_);
        GetPressureKPa(Pmeas_);
        ret = DEVICE_OK;
    }break;
    case MM::BeforeGet:
    {
        // Simply get the imposed pressure value
        pProp->Set(Pimp_);
        ret = DEVICE_OK;
    }break;
    }
    return ret;
}

int FluigentChannel::OnMeasuredPressure(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    int ret = DEVICE_OK;
    switch (eAct)
    {
        // This function is read-only
    case MM::BeforeGet:
    {
        // Get the measured pressure, and refresh the value
        ret = GetPressureKPa(Pmeas_);
        pProp->Set(Pmeas_);
    }break;
    }
    return ret;

}

///////////////////////////////////////////////////////////////////////////////
// FluigentChannel class
// MMPump API
///////////////////////////////////////////////////////////////////////////////

int FluigentChannel::GetPressureKPa(double& P) {
    float temp = 0;
    int ret = fgt_get_pressure(channelInfo_.indexID, &temp);
    P = (double)temp;
    return ret;
}

int FluigentChannel::SetPressureKPa(double P)
{
    int ret = fgt_set_pressure(channelInfo_.indexID, (float)P);
    return ret;
}

int FluigentChannel::Calibrate()
{
    fgt_calibratePressure(channelInfo_.indexID);
    int ret = DEVICE_OK;
    return ret;
}

///////////////////////////////////////////////////////////////////////////////
// FluigentChannel class
// Utility methods
///////////////////////////////////////////////////////////////////////////////

int FluigentChannel::GetPressureLimits(double& Pmin, double& Pmax)
{
    int ret = DEVICE_OK;
    float minTemp = 0;
    float maxTemp = 0;
    fgt_get_pressureRange(channelInfo_.indexID, &minTemp, &maxTemp);
    Pmin = (double)minTemp;
    Pmax = (double)maxTemp;
    return ret;
}