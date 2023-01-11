/*
 * Project: Light Sheet Device Manager
 * License: BSD 3-clause, see license.txt
 * Author: Brandon Simpson (brandon@asiimaging.com)
 * Copyright (c) 2022, Applied Scientific Instrumentation
 */

#ifndef _LIGHTSHEET_DEVICE_MANAGER_H_
#define _LIGHTSHEET_DEVICE_MANAGER_H_

#include "MMDevice.h"
#include "DeviceBase.h"
#include "LightSheetManager.h"
#include "MicroscopeGeometry.h"
#include <vector>
#include <string>

class LightSheetDeviceManager : public CGenericBase<LightSheetDeviceManager>
{
public:
    LightSheetDeviceManager();
    ~LightSheetDeviceManager();

    // Device API
    int Initialize();
    int Shutdown();
    bool Busy();
    void GetName(char* name) const;

    // Pre-init Property Actions
    int OnNumImagingPaths(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnNumIlluminationPaths(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnNumSimultaneousCameras(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnMicroscopeGeometry(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnLightSheetType(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
    // Create properties for all properties in the device map.
    void CreateDeviceProperties(std::map<std::string, MM::DeviceType> deviceMap);

    // Create properties for the imaging camera based on the number of imaging paths and simultaneous cameras on each imaging path.
    void CreateCameraProperties();

    // Returns true if the string starts with the search term.
    bool StringStartsWith(const std::string& str, const std::string& searchTerm) const;

    // Returns the device names of all devices of MM::DeviceType loaded in the hardware configuration as strings.
    std::vector<std::string> GetLoadedDevicesOfType(const MM::DeviceType deviceType);

    // Create multiple properties with the light path prefix.
    void CreatePrefixProperties(const std::string& propertyName, const std::string& prefix, 
        const int numProperties, MM::DeviceType deviceType);

    std::map<MM::DeviceType, std::vector<std::string>> devices_;
    MicroscopeGeometry geometry_;
    bool initialized_;

    // pre-init properties
    std::string geometryType_;   // type of microscope geometry
    std::string lightSheetType_; // scanned or static sheet
    long numImagingPaths_;
    long numIlluminationPaths_;
    long numSimultaneousCameras_;

};

#endif // _LIGHTSHEET_DEVICE_MANAGER_H_