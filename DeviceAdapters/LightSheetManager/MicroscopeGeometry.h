/*
 * Project: Light Sheet Device Manager
 * License: BSD 3-clause, see license.txt
 * Author: Brandon Simpson (brandon@asiimaging.com)
 * Copyright (c) 2022, Applied Scientific Instrumentation
 */

#ifndef _MICROSCOPE_GEOMETRY_H_
#define _MICROSCOPE_GEOMETRY_H_

#include "MMDevice.h"
#include "DeviceBase.h"
#include <map>

// [ How To Add New Microscope Geometries ]
//
// To add a new microscope geometry type to the device adapter you add a new 
// entry in the deviceMap_ variable in the CreateDeviceMap() function.
//
//
//
//

// This class contains the data needed to map logical devices to 
// physical devices for each microscope geometry type. This mapping 
// is used by the LightSheetManager plugin.
class MicroscopeGeometry
{
public:
	// The constructor populates both deviceMap_ and geometryTypes_.
	MicroscopeGeometry();
	
	// Clears both deviceMap_ and geometryTypes_ and is called after 
	// device adapter initialization is successful.
	void ClearDeviceMap();

	// Returns the list of microscope geometry types as strings.
	std::vector<std::string>& GetGeometryTypes();

	// Returns a map of property names mapped to Micro-Manager device types.
	std::map<std::string, MM::DeviceType> GetDeviceMap(const std::string geometryType);

	// This is the function where you can define new microscope geometry types.
	void CreateDeviceMap();
private:

	// Creates the list of microscope geometry types based on the device map.
	void CreateGeometryTypes();

	std::map<std::string, std::map<std::string, MM::DeviceType>> deviceMap_;
	std::vector<std::string> geometryTypes_;
};

#endif _MICROSCOPE_GEOMETRY_H_