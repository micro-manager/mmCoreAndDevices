/*
 * Project: Light Sheet Device Manager
 * License: BSD 3-clause, see license.txt
 * Author: Brandon Simpson (brandon@asiimaging.com)
 * Copyright (c) 2022, Applied Scientific Instrumentation
 */

#include "MicroscopeGeometry.h"

MicroscopeGeometry::MicroscopeGeometry()
{
    CreateDeviceMap();
    CreateGeometryTypes();
}

void MicroscopeGeometry::ClearDeviceMap()
{
    deviceMap_.clear();
    geometryTypes_.clear();
}

void MicroscopeGeometry::CreateGeometryTypes()
{
    for (const auto& device : deviceMap_)
    {
        geometryTypes_.push_back(device.first);
    }
}

// The return value is a reference so that it can be used in SetAllowedValues
std::vector<std::string>& MicroscopeGeometry::GetGeometryTypes()
{
    return geometryTypes_;
}

std::map<std::string, MM::DeviceType> MicroscopeGeometry::GetDeviceMap(const std::string geometryType)
{
    return deviceMap_.at(geometryType);
}

// Refer to the header file for detailed instructions on how add 
// new microscope geometries to the device map.
void MicroscopeGeometry::CreateDeviceMap()
{
    deviceMap_ =
    {
        {
            "diSPIM",
            {
                {"SampleXY", MM::XYStageDevice},
                {"SampleZ", MM::StageDevice},
                {"TriggerCamera", MM::ShutterDevice},
                {"TriggerLaser", MM::ShutterDevice},
                {"InvertedFocus", MM::StageDevice},
                {"InvertedCamera", MM::CameraDevice},
                {"InvertedShutter", MM::ShutterDevice},
                {"IllumSlice", MM::GalvoDevice},
                {"IllumBeam", MM::GalvoDevice},
                {"ImagingFocus", MM::StageDevice},
                {"ImagingCamera", MM::CameraDevice}
            },
        },
        {
            "iSPIM",
            {
                {"SampleXY", MM::XYStageDevice},
                {"SampleZ", MM::StageDevice},
                {"TriggerCamera", MM::ShutterDevice},
                {"TriggerLaser", MM::ShutterDevice},
                {"InvertedFocus", MM::StageDevice},
                {"InvertedCamera", MM::CameraDevice},
                {"InvertedShutter", MM::ShutterDevice},
                {"IllumSlice", MM::GalvoDevice},
                {"IllumBeam", MM::GalvoDevice},
                {"ImagingFocus", MM::StageDevice},
                {"ImagingCamera", MM::CameraDevice}
            }
        },
        {
            "oSPIM",
            {
                {"SampleXY", MM::XYStageDevice},
                {"SampleZ", MM::StageDevice},
                {"SampleH", MM::StageDevice},
                {"TriggerCamera", MM::ShutterDevice},
                {"TriggerLaser", MM::ShutterDevice},
                {"InvertedFocus", MM::StageDevice},
                {"InvertedCamera", MM::CameraDevice},
                {"InvertedShutter", MM::ShutterDevice},
                {"IllumSlice", MM::GalvoDevice},
                {"IllumBeam", MM::GalvoDevice},
                {"ImagingFocus", MM::StageDevice},
                {"ImagingCamera", MM::CameraDevice}
            }
        },
        {
            "mesoSPIM",
            {
                {"SampleXY", MM::XYStageDevice},
                {"SampleZ", MM::StageDevice},
                {"SampleRotation", MM::StageDevice},
                {"TriggerCamera", MM::ShutterDevice},
                {"TriggerLaser", MM::ShutterDevice},
                {"IllumSlice", MM::GalvoDevice},
                {"IllumBeam", MM::GalvoDevice},
                {"IllumFocus", MM::StageDevice},
                {"ImagingFocus", MM::StageDevice},
                {"ImagingCamera", MM::CameraDevice}
            }
        },
        {
            "OpenSPIM-L",
            {
                {"SampleXY", MM::XYStageDevice},
                {"SampleZ", MM::XYStageDevice},
                {"SampleRotation", MM::StageDevice},
                {"TriggerCamera", MM::ShutterDevice},
                {"TriggerLaser", MM::ShutterDevice},
                {"ImagingCamera", MM::CameraDevice}
            }
        },
        {
            "SCAPE",
            {
                {"SampleXY", MM::XYStageDevice},
                {"SampleZ", MM::StageDevice},
                {"TriggerCamera", MM::ShutterDevice},
                {"TriggerLaser", MM::ShutterDevice},
                {"IllumSlice", MM::GalvoDevice},
                {"IllumBeam", MM::GalvoDevice},
                {"ImagingFocus", MM::StageDevice},
                {"ImagingCamera", MM::CameraDevice},
                {"PreviewCamera", MM::CameraDevice}
            }
        }
    };
}
