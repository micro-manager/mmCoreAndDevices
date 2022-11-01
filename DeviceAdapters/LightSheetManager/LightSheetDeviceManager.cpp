/*
 * Project: Light Sheet Device Manager
 * License: BSD 3-clause, see license.txt
 * Author: Brandon Simpson (brandon@asiimaging.com)
 * Copyright (c) 2022, Applied Scientific Instrumentation
 */

#include "LightSheetDeviceManager.h"

LightSheetDeviceManager::LightSheetDeviceManager() :
    initialized_(false),
    geometryType_("diSPIM"), // TODO: why do the defaults not work? is it the property action?
    lightSheetType_("Scanned"),
    numImagingPaths_(1),
    numIlluminationPaths_(1),
    numSimultaneousCameras_(1)
{
    // call the base class method to setup default error codes/messages
    InitializeDefaultErrorMessages();

    // pre-init properties
    CPropertyAction* pAct = nullptr;
    
    pAct = new CPropertyAction(this, &LightSheetDeviceManager::OnMicroscopeGeometry);
    CreateProperty(gMicroscopeGeometry, "diSPIM", MM::String, false, pAct, true);
    SetAllowedValues(gMicroscopeGeometry, geometry_.GetGeometryTypes());

    pAct = new CPropertyAction(this, &LightSheetDeviceManager::OnNumSimultaneousCameras);
    CreateProperty(gSimultaneousCameras, "1", MM::Integer, false, pAct, true);
    SetPropertyLimits(gSimultaneousCameras, 1, INT_MAX);

    pAct = new CPropertyAction(this, &LightSheetDeviceManager::OnNumImagingPaths);
    CreateProperty(gImagingPaths, "1", MM::Integer, false, pAct, true);
    SetPropertyLimits(gImagingPaths, 1, INT_MAX);

    pAct = new CPropertyAction(this, &LightSheetDeviceManager::OnNumIlluminationPaths);
    CreateProperty(gIlluminationPaths, "1", MM::Integer, false, pAct, true);
    SetPropertyLimits(gIlluminationPaths, 1, INT_MAX);

    pAct = new CPropertyAction(this, &LightSheetDeviceManager::OnLightSheetType);
    CreateProperty(gLightSheetType, gLightSheetTypeScanned, MM::String, false, pAct, true);
    AddAllowedValue(gLightSheetType, gLightSheetTypeScanned);
    AddAllowedValue(gLightSheetType, gLightSheetTypeStatic);
}

LightSheetDeviceManager::~LightSheetDeviceManager()
{
    Shutdown();
}

void LightSheetDeviceManager::GetName(char* name) const
{
    CDeviceUtils::CopyLimitedString(name, gDeviceName);
}

int LightSheetDeviceManager::Shutdown()
{
    if (initialized_)
    {
        initialized_ = false;
    }
    return DEVICE_OK;
}

bool LightSheetDeviceManager::Busy()
{
    return false;
}

int LightSheetDeviceManager::Initialize()
{
    // generic device adapter properties
    CreateStringProperty(MM::g_Keyword_Name, gDeviceName, true);
    CreateStringProperty(MM::g_Keyword_Description, gDeviceDescription, true);

    // read-only version number
    CreateStringProperty(gVersionNumberPropertyName, gVersionNumber, true);

    // collect arrays of available devices in the hardware configuration
    devices_ =
    {
        {MM::XYStageDevice, GetLoadedDevicesOfType(MM::XYStageDevice)},
        {MM::StageDevice, GetLoadedDevicesOfType(MM::StageDevice)},
        {MM::ShutterDevice, GetLoadedDevicesOfType(MM::ShutterDevice)},
        {MM::GalvoDevice, GetLoadedDevicesOfType(MM::GalvoDevice)},
        {MM::CameraDevice, GetLoadedDevicesOfType(MM::CameraDevice)}
    };

    // create properties using the device map and device arrays
    CreateDeviceProperties(geometry_.GetDeviceMap(geometryType_));

    // we no longer need the device map or device arrays
    geometry_.ClearDeviceMap();
    devices_.clear();

    initialized_ = true;
    return DEVICE_OK;
}

void LightSheetDeviceManager::CreateDeviceProperties(std::map<std::string, MM::DeviceType> deviceMap)
{
    // create properties from the device map
    for (const auto& device : deviceMap)
    {
        const std::string propertyName = device.first;
        const MM::DeviceType deviceType = device.second;

        // create camera properties and skip to the next property
        if (propertyName == gImagingCameraPropertyName)
        {
            CreateCameraProperties();
            continue;
        }

        // skip the "IllumBeam" property if in static light sheet mode
        if (propertyName == gIllumBeamPropertyName && lightSheetType_ == gLightSheetTypeStatic)
        {
            continue;
        }

        // create properties based on the property name prefix
        if (numImagingPaths_ > 1 && StringStartsWith(propertyName, gImagingPrefix))
        {
            // create multiple properties => "Imaging1, Imaging2, ... ImagingN"
            CreatePrefixProperties(propertyName, gImagingPrefix, numImagingPaths_, deviceType);
        }
        else if (numIlluminationPaths_ > 1 && StringStartsWith(propertyName, gIllumPrefix))
        {
            // create multiple properties => "Illum1, Illum2, ... IllumN"
            CreatePrefixProperties(propertyName, gIllumPrefix, numIlluminationPaths_, deviceType);
        }
        else
        {
            // create single property => no prefix found at the start of the string
            CreateStringProperty(propertyName.c_str(), gUndefined, false);
            SetAllowedValues(propertyName.c_str(), devices_.at(deviceType));
        }
    }
}

void LightSheetDeviceManager::CreateCameraProperties()
{
    if (numSimultaneousCameras_ == 1 && numImagingPaths_ == 1)
    {
        // single imaging path and single camera => "ImagingCamera"
        CreateStringProperty(gImagingCameraPropertyName, gUndefined, false);
        SetAllowedValues(gImagingCameraPropertyName, devices_.at(MM::CameraDevice));
    }
    else if (numSimultaneousCameras_ == 1 && numImagingPaths_ > 1)
    {
        // multiple imaging paths and one camera per imaging path => "Imaging1Camera, Imaging2Camera"
        for (int imagingPath = 1; imagingPath <= numImagingPaths_; imagingPath++)
        {
            const std::string propertyName = "Imaging" + std::to_string(imagingPath) + "Camera";
            CreateStringProperty(propertyName.c_str(), gUndefined, false);
            SetAllowedValues(propertyName.c_str(), devices_.at(MM::CameraDevice));
        }
    }
    else
    {
        // multiple simultaneous cameras per imaging path =>
        // "Imaging1Camera1, Imaging1Camera2, Imaging2Camera1, Imaging2Camera2"
        for (int imagingPath = 1; imagingPath <= numImagingPaths_; imagingPath++)
        {
            for (int cameraNum = 1; cameraNum <= numSimultaneousCameras_; cameraNum++)
            {
                const std::string propertyName = 
                    "Imaging" + std::to_string(imagingPath) + "Camera" + std::to_string(cameraNum);
                CreateStringProperty(propertyName.c_str(), gUndefined, false);
                SetAllowedValues(propertyName.c_str(), devices_.at(MM::CameraDevice));
            }
        }
    }
}

// TODO: use rfind for now: use starts_with when we have C++20 support
bool LightSheetDeviceManager::StringStartsWith(const std::string& str, const std::string& searchTerm) const
{
    return str.rfind(searchTerm, 0) == 0;
}

std::vector<std::string> LightSheetDeviceManager::GetLoadedDevicesOfType(const MM::DeviceType deviceType)
{
    // start with undefined for no device
    std::vector<std::string> devices;
    devices.push_back(gUndefined);
    // get all loaded devices of MM::DeviceType
    unsigned int index = 0;
    char deviceName[MM::MaxStrLength];
    for (;;)
    {
        GetLoadedDeviceOfType(deviceType, deviceName, index++);
        if (strlen(deviceName))
        {
            devices.push_back(deviceName);
        }
        else
        {
            break;
        }
    }
    return devices;
}

void LightSheetDeviceManager::CreatePrefixProperties(
    const std::string& propertyName, const std::string& prefix,
    const int numProperties, MM::DeviceType deviceType)
{
    const std::string end = propertyName.substr(prefix.size());
    for (int i = 0; i < numProperties; i++)
    {
        const std::string newPropertyName = prefix + std::to_string(i + 1) + end;
        CreateStringProperty(newPropertyName.c_str(), gUndefined, false);
        SetAllowedValues(newPropertyName.c_str(), devices_.at(deviceType));
    }
}

// Pre-init Property Actions

int LightSheetDeviceManager::OnNumImagingPaths(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(numImagingPaths_);
    }
    else if (eAct == MM::AfterSet)
    {
        long numPaths;
        pProp->Get(numPaths);
        numImagingPaths_ = numPaths;
    }
    return DEVICE_OK;
}

int LightSheetDeviceManager::OnNumIlluminationPaths(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(numIlluminationPaths_);
    }
    else if (eAct == MM::AfterSet)
    {
        long numPaths;
        pProp->Get(numPaths);
        numIlluminationPaths_ = numPaths;
    }
    return DEVICE_OK;
}

int LightSheetDeviceManager::OnNumSimultaneousCameras(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(numSimultaneousCameras_);
    }
    else if (eAct == MM::AfterSet)
    {
        long numPaths;
        pProp->Get(numPaths);
        numSimultaneousCameras_ = numPaths;
    }
    return DEVICE_OK;
}

int LightSheetDeviceManager::OnMicroscopeGeometry(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(geometryType_.c_str());
    }
    else if (eAct == MM::AfterSet)
    {
        std::string geometryType;
        pProp->Get(geometryType);
        geometryType_ = geometryType;
    }
    return DEVICE_OK;
}

int LightSheetDeviceManager::OnLightSheetType(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(lightSheetType_.c_str());
    }
    else if (eAct == MM::AfterSet)
    {
        std::string lightSheetType;
        pProp->Get(lightSheetType);
        lightSheetType_ = lightSheetType;
    }
    return DEVICE_OK;
}
