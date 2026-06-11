/*
 * Project: Light Sheet Device Manager
 * License: BSD 3-clause, see license.txt
 * Author: Brandon Simpson (brandon@asiimaging.com)
 * Copyright (c) 2022, Applied Scientific Instrumentation
 */

#include "LightSheetDeviceManager.h"

namespace {
namespace Properties {
    constexpr char IlluminationPaths[] = "IlluminationPaths";
    constexpr char ImagingPaths[] = "ImagingPaths";
    constexpr char LightSheetType[] = "LightSheetType";
    constexpr char MicroscopeGeometry[] =  "MicroscopeGeometry";
    constexpr char SimultaneousCameras[] =  "SimultaneousCameras";
} // namespace Properties
namespace Props = Properties;
} // namespace

LightSheetDeviceManager::LightSheetDeviceManager() {
    // call the base method to setup default errors
    InitializeDefaultErrorMessages();

    // create pre-init properties
    CreateNumIlluminationPathsProperty();
    CreateNumImagingPathsProperty();
    CreateLightSheetTypeProperty();
    CreateMicroscopeGeometryProperty();
    CreateNumSimultaneousCamerasProperty();
}

LightSheetDeviceManager::~LightSheetDeviceManager() {
    Shutdown();
}

void LightSheetDeviceManager::GetName(char* name) const {
    CDeviceUtils::CopyLimitedString(name, gDeviceName);
}

int LightSheetDeviceManager::Shutdown() {
    initialized_ = false;
    return DEVICE_OK;
}

// Return false, this "device" can never be busy.
bool LightSheetDeviceManager::Busy() {
    return false;
}

int LightSheetDeviceManager::Initialize() {
    // generic device adapter properties
    CreateStringProperty(MM::g_Keyword_Name, gDeviceName, true);
    CreateStringProperty(MM::g_Keyword_Description, gDeviceDescription, true);

    // read-only version number
    CreateStringProperty(gVersionNumberPropertyName, gVersionNumber, true);

    // collect arrays of available devices in the hardware configuration
    devices_ = {
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

void LightSheetDeviceManager::CreateDeviceProperties(const std::map<std::string, MM::DeviceType>& deviceMap) {
    // create properties from the device map
    for (const auto& device : deviceMap) {
        const std::string propertyName = device.first;
        const MM::DeviceType deviceType = device.second;

        // create camera properties and skip to the next property
        if (propertyName == gImagingCameraPropertyName) {
            CreateCameraProperties();
            continue;
        }

        // skip the "IllumBeam" property if in static light sheet mode
        if (propertyName == gIllumBeamPropertyName && lightSheetType_ == gLightSheetTypeStatic) {
            continue;
        }

        // create properties based on the property name prefix
        if (numImagingPaths_ > 1 && StartsWith(propertyName, gImagingPrefix)) {
            // create multiple properties => "Imaging1, Imaging2, ... ImagingN"
            CreatePrefixProperties(propertyName, gImagingPrefix, numImagingPaths_, deviceType);
        } else if (numIlluminationPaths_ > 1 && StartsWith(propertyName, gIllumPrefix)) {
            // create multiple properties => "Illum1, Illum2, ... IllumN"
            CreatePrefixProperties(propertyName, gIllumPrefix, numIlluminationPaths_, deviceType);
        } else {
            // create single property => no prefix found at the start of the string
            CreateStringProperty(propertyName.c_str(), gUndefined, false);
            SetAllowedValues(propertyName.c_str(), devices_.at(deviceType));
        }
    }
}

void LightSheetDeviceManager::CreateCameraProperties() {
    if (numSimultaneousCameras_ == 1 && numImagingPaths_ == 1) {
        // single imaging path and single camera => "ImagingCamera"
        CreateStringProperty(gImagingCameraPropertyName, gUndefined, false);
        SetAllowedValues(gImagingCameraPropertyName, devices_.at(MM::CameraDevice));
    } else if (numSimultaneousCameras_ == 1 && numImagingPaths_ > 1) {
        // multiple imaging paths and one camera per imaging path => "Imaging1Camera, Imaging2Camera"
        for (int imagingPath = 1; imagingPath <= numImagingPaths_; imagingPath++) {
            const std::string propertyName = "Imaging" + std::to_string(imagingPath) + "Camera";
            CreateStringProperty(propertyName.c_str(), gUndefined, false);
            SetAllowedValues(propertyName.c_str(), devices_.at(MM::CameraDevice));
        }
    } else {
        // multiple simultaneous cameras; multiple imaging paths =>
        // "Imaging1Camera1", "Imaging1Camera2", "Imaging2Camera1", "Imaging2Camera2"
        // multiple simultaneous cameras; single imaging path =>
        // "ImagingCamera1", "ImagingCamera2"
        for (int imagingPath = 1; imagingPath <= numImagingPaths_; imagingPath++) {
            for (int cameraNum = 1; cameraNum <= numSimultaneousCameras_; cameraNum++) {
                std::string propertyName;
                if (numImagingPaths_ > 1) {
                    propertyName = "Imaging" + std::to_string(imagingPath)
                        + "Camera" + std::to_string(cameraNum);
                } else {
                    propertyName = "ImagingCamera" + std::to_string(cameraNum);
                }
                CreateStringProperty(propertyName.c_str(), gUndefined, false);
                SetAllowedValues(propertyName.c_str(), devices_.at(MM::CameraDevice));
            }
        }
    }
}

// TODO: use rfind for now: use starts_with when we have C++20 support
bool LightSheetDeviceManager::StartsWith(const std::string& str, const std::string& searchTerm) const {
    return str.rfind(searchTerm, 0) == 0;
}

std::vector<std::string> LightSheetDeviceManager::GetLoadedDevicesOfType(const MM::DeviceType deviceType) {
    // start with undefined for no device
    std::vector<std::string> devices;
    devices.push_back(gUndefined);
    // get all loaded devices of MM::DeviceType
    unsigned int index = 0;
    char deviceName[MM::MaxStrLength];
    for (;;) {
        GetLoadedDeviceOfType(deviceType, deviceName, index++);
        if (deviceName[0] != '\0') {
            devices.push_back(deviceName);
        } else {
            break;
        }
    }
    return devices;
}

void LightSheetDeviceManager::CreatePrefixProperties(const std::string& propertyName,
    const std::string& prefix, const int numProperties, MM::DeviceType deviceType) {

    const std::string end = propertyName.substr(prefix.size());
    for (int i = 0; i < numProperties; i++) {
        const std::string newPropertyName = prefix + std::to_string(i + 1) + end;
        CreateStringProperty(newPropertyName.c_str(), gUndefined, false);
        SetAllowedValues(newPropertyName.c_str(), devices_.at(deviceType));
    }
}

// Pre-init Properties

void LightSheetDeviceManager::CreateNumImagingPathsProperty() {
    const std::string propertyName = "ImagingPaths";
    CreateIntegerProperty(
        propertyName.c_str(), numImagingPaths_, false,
        new MM::ActionLambda([this](MM::PropertyBase* pProp, MM::ActionType eAct) {
            if (eAct == MM::BeforeGet) {
                pProp->Set(numImagingPaths_);
            } else if (eAct == MM::AfterSet) {
                long numPaths;
                pProp->Get(numPaths);
                numImagingPaths_ = numPaths;
            }
            return DEVICE_OK;
        }),
        true
    );
    SetPropertyLimits(propertyName.c_str(), 1, 4);
}

void LightSheetDeviceManager::CreateNumIlluminationPathsProperty() {
    CreateIntegerProperty(
        Props::IlluminationPaths, numIlluminationPaths_, false,
        new MM::ActionLambda([this](MM::PropertyBase* pProp, MM::ActionType eAct) {
            if (eAct == MM::BeforeGet) {
                pProp->Set(numIlluminationPaths_);
            } else if (eAct == MM::AfterSet) {
                long numPaths;
                pProp->Get(numPaths);
                numIlluminationPaths_ = numPaths;
            }
            return DEVICE_OK;
        }),
        true
    );
    SetPropertyLimits(Props::IlluminationPaths, 1, 4);
}

void LightSheetDeviceManager::CreateLightSheetTypeProperty() {
    CreateStringProperty(
        Props::LightSheetType, lightSheetType_.c_str(), false,
        new MM::ActionLambda([this](MM::PropertyBase* pProp, MM::ActionType eAct) {
            if (eAct == MM::BeforeGet) {
                pProp->Set(lightSheetType_.c_str());
            } else if (eAct == MM::AfterSet) {
                std::string lightSheetType;
                pProp->Get(lightSheetType);
                lightSheetType_ = lightSheetType;
            }
            return DEVICE_OK;
        }),
        true
    );
    AddAllowedValue(Props::LightSheetType, gLightSheetTypeScanned);
    AddAllowedValue(Props::LightSheetType, gLightSheetTypeStatic);
}

void LightSheetDeviceManager::CreateMicroscopeGeometryProperty() {
    CreateStringProperty(
        Props::MicroscopeGeometry, geometryType_.c_str(), false,
        new MM::ActionLambda([this](MM::PropertyBase* pProp, MM::ActionType eAct) {
            if (eAct == MM::BeforeGet) {
                pProp->Set(geometryType_.c_str());
            } else if (eAct == MM::AfterSet) {
                std::string geometryType;
                pProp->Get(geometryType);
                geometryType_ = geometryType;
            }
            return DEVICE_OK;
        }),
        true
    );
    std::vector<std::string> allowedValues = geometry_.GetGeometryTypes();
    SetAllowedValues(Props::MicroscopeGeometry, allowedValues);
}

void LightSheetDeviceManager::CreateNumSimultaneousCamerasProperty() {
    CreateIntegerProperty(
        Props::SimultaneousCameras, numSimultaneousCameras_, false,
        new MM::ActionLambda([this](MM::PropertyBase* pProp, MM::ActionType eAct) {
            if (eAct == MM::BeforeGet) {
                pProp->Set(numSimultaneousCameras_);
            } else if (eAct == MM::AfterSet) {
                long numCameras;
                pProp->Get(numCameras);
                numSimultaneousCameras_ = numCameras;
            }
            return DEVICE_OK;
        }),
        true
    );
    SetPropertyLimits(Props::SimultaneousCameras, 1, 4);
}
