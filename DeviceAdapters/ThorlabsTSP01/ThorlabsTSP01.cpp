///////////////////////////////////////////////////////////////////////////////
// FILE:          ThorlabsTSP01.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters - Thorlabs TSP01revB adapter
//-----------------------------------------------------------------------------
// DESCRIPTION:   This device adapter interfaces with Thorlabs thermometer TSP01-revB
//                Developed using PM100x adapter and Thorlabs' C++ code examples with some ChatGPT proof-reading
//                
// AUTHOR:        Andrey Andreev, Ellison Medical Institute, 2025
//                aandreev@emila.org

// TODO: move the polling interval to properties (currently once per second)

#include "ThorlabsTSP01.h"
#include <stdlib.h>
#include <time.h>


const char* g_TSP01Name = "ThorlabsTSP01";
const char* g_On = "On";
const char* g_Off = "Off";
const char* internal_temp_prop_name = "USBDeviceTemp";
const char* internal_humidity_prop_name = "USBDeviceHumidity";
const char* probe1_prop_name = "TempProbe1";
const char* probe2_prop_name = "TempProbe2";
int polling_interval_sec = 1;

///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////
MODULE_API void InitializeModuleData()
{
    RegisterDevice(g_TSP01Name, MM::GenericDevice, "ThorLabs Temperature Sensor");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
    if (deviceName == 0)
        return 0;

    if (strcmp(deviceName, g_TSP01Name) == 0)
    {
        return new ThorlabsTSP01();
    }

    return 0;
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
    delete pDevice;
}


ThorlabsTSP01::ThorlabsTSP01() :
    initialized_(false),
    deviceName_("ThorlabsTSP01"),
    instrHdl_(VI_NULL),
    latestTemp_(24.0),
    latestHumidity_(50.0),
    latestTemp_probe1_(24.0),
    latestTemp_probe2_(24.0),
    keepRunning_(false)
{

    CPropertyAction* pAct = new CPropertyAction(this, &ThorlabsTSP01::OnTSPName);
    CreateProperty("Thermometer", deviceName_.c_str(), MM::String, false, pAct, true);
    
}


ThorlabsTSP01::~ThorlabsTSP01()
{
}

int ThorlabsTSP01::Initialize()
{
    static ViChar buffer[VI_FIND_BUFLEN];
    ViUInt32 count;
    ViStatus err = TLTSPB_findRsrc(VI_NULL, &count);
    if (err != VI_SUCCESS || count == 0)
        return (int)err;
    
    err = TLTSPB_getRsrcName(VI_NULL, 0, buffer);
    if (err != VI_SUCCESS)
        return (int)err;
    
    err = TLTSPB_init(buffer, VI_ON, VI_ON, &instrHdl_);
    if(err != VI_SUCCESS)
        return (int)err;
    
    ViChar modelName[TLTSP_BUFFER_SIZE];
    ViChar serialNumber[TLTSP_BUFFER_SIZE];
    ViChar manufacturerName[TLTSP_BUFFER_SIZE];
    ViPBoolean resourceInUse = 0;

    err = TLTSPB_getRsrcInfo(instrHdl_, 0, modelName, serialNumber, manufacturerName, resourceInUse);

    if (err != DEVICE_OK)
        return (int)err;
    
    CreateStringProperty("Sensor Serial Number", serialNumber, true);
    int ret = CreateFloatProperty(internal_temp_prop_name, latestTemp_, true);
    if (ret != DEVICE_OK)
        return ret;

    ret = CreateFloatProperty(internal_humidity_prop_name, latestHumidity_, true);
    if (ret != DEVICE_OK)
        return ret;

    ret = CreateFloatProperty(probe1_prop_name, latestTemp_probe1_, true);
    if (ret != DEVICE_OK)
        return ret;
    ret = CreateFloatProperty(probe2_prop_name, latestTemp_probe2_, true);
    if (ret != DEVICE_OK)
        return ret;

    keepRunning_ = true;
    tempThread_ = std::thread(&ThorlabsTSP01::BackgroundTemperatureUpdate, this);

    initialized_ = true;
    return DEVICE_OK;
}


int ThorlabsTSP01::Shutdown()
{
    keepRunning_ = false;
    if (tempThread_.joinable())
        tempThread_.join();

    TLTSPB_close(instrHdl_);  // if applicable
    return DEVICE_OK;
}

void ThorlabsTSP01::GetName(char* pszName) const {
    CDeviceUtils::CopyLimitedString(pszName, deviceName_.c_str());
}

bool ThorlabsTSP01::Busy() { return false; }

void ThorlabsTSP01::BackgroundTemperatureUpdate()
{
    while (keepRunning_)
    {
        ViReal64 temp;
        ViReal64 humi;
        ViStatus err = TLTSPB_measTemperature(instrHdl_, TLTSP_TEMPER_CHANNEL_1, &temp);
        if (err == VI_SUCCESS and latestTemp_ != temp)
        {
            std::lock_guard<std::mutex> lock(tempMutex_);
            latestTemp_ = temp;
            OnPropertyChanged(internal_temp_prop_name, CDeviceUtils::ConvertToString(temp));
        }

        err = TLTSPB_measTemperature(instrHdl_, TLTSP_TEMPER_CHANNEL_2, &temp);
        if (err == VI_SUCCESS and latestTemp_probe1_ != temp)
        {
            std::lock_guard<std::mutex> lock(tempMutex_);
            latestTemp_probe1_ = temp;
            OnPropertyChanged(probe1_prop_name, CDeviceUtils::ConvertToString(temp));
        }
        err = TLTSPB_measTemperature(instrHdl_, TLTSP_TEMPER_CHANNEL_3, &temp);
        if (err == VI_SUCCESS and latestTemp_probe2_ != temp)
        {
            std::lock_guard<std::mutex> lock(tempMutex_);
            latestTemp_probe2_ = temp;
            OnPropertyChanged(probe2_prop_name, CDeviceUtils::ConvertToString(temp));
        }

        err = TLTSPB_measHumidity(instrHdl_, &humi);
        if (err == VI_SUCCESS and latestHumidity_ != temp)
        {
            std::lock_guard<std::mutex> lock(tempMutex_);
            latestHumidity_ = humi;
            OnPropertyChanged(internal_humidity_prop_name, CDeviceUtils::ConvertToString(humi));
        }


        std::this_thread::sleep_for(std::chrono::seconds(polling_interval_sec));
    }
}

int ThorlabsTSP01::OnTSPName(MM::PropertyBase* pProp, MM::ActionType eAct) { return DEVICE_OK; }
