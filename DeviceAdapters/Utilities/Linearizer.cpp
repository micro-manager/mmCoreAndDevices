///////////////////////////////////////////////////////////////////////////////
// FILE:          Linearizer.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Utility device that linearizes the output of a physical device
//
// AUTHOR:        Nico Stuurman, nstuurman@altoslabs.com, 7/12/2022
// COPYRIGHT:     Altos Labs, 2022
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

#ifdef _WIN32
// Prevent windows.h from defining min and max macros,
// which clash with std::min and std::max.
#define NOMINMAX
#endif

#include <fstream>
#include "Utilities.h"
#include <iostream>

extern const char* g_DeviceNameLinearizer;
extern const char* g_NoDevice;


Linearizer::Linearizer() :
   initialized_(false),
   deviceName_(""),
   jsonFilePath_(""),
   jsonData_()
{
   InitializeDefaultErrorMessages();

   SetErrorText(ERR_INVALID_DEVICE_NAME, "Please select a valid  device");
   SetErrorText(ERR_TIMEOUT, "Device was busy.  Try increasing the Core-Timeout property");
   SetErrorText(ERR_FILE_UNPARSABLE, "Failed to parse calibration file");

   // Name                                                                   
   CreateProperty(MM::g_Keyword_Name, g_DeviceNameLinearizer, MM::String, true);

   // Description                                                            
   CreateProperty(MM::g_Keyword_Description, "Device Linearizing the output of another device", MM::String, true);

   CPropertyAction* pAct = new CPropertyAction(this, &Linearizer::OnJSONFile);
   CreateProperty("CalibrationFile", jsonFilePath_.c_str(), MM::String, false, pAct, true);
}

Linearizer::~Linearizer() 
{
   Shutdown();
}

int Linearizer::Shutdown()
{
   channelData_.clear();
   initialized_ = false;
   return DEVICE_OK;
}

void Linearizer::GetName(char* Name) const
{
   CDeviceUtils::CopyLimitedString(Name, g_DeviceNameLinearizer);
}

bool Linearizer::Busy()
{
   // TODO: implement correctly
   return false;
}

int Linearizer::Initialize()
{
   // get list with available devices. 
   char deviceName[MM::MaxStrLength];
   unsigned int deviceIterator = 0;
   for (;;)
   {
      GetLoadedDeviceOfType(MM::AnyType, deviceName, deviceIterator++);
      if (0 < strlen(deviceName))
      {
         availableDevices_.push_back(std::string(deviceName));
      }
      else
         break;
   }

   std::vector<std::string>::iterator it;
   it = availableDevices_.begin();
   availableDevices_.insert(it, g_NoDevice);

   CPropertyAction* pAct = new CPropertyAction(this, &Linearizer::OnDevice);
   std::string defaultDevice = g_NoDevice;
   CreateProperty("State Device", defaultDevice.c_str(), MM::String, false, pAct, false);
   if (availableDevices_.size() >= 1)
      SetAllowedValues("State Device", availableDevices_);
   else
      return ERR_NO_STATE_DEVICE_FOUND;

   SetProperty("State Device", defaultDevice.c_str());

   initialized_ = true;

   return DEVICE_OK;
}

int Linearizer::OnJSONFile(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(jsonFilePath_.c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      std::string jsonFilePath;
      pProp->Get(jsonFilePath);
      std::ifstream jsonFile;
      jsonFile.open(jsonFilePath);
      if (!jsonFile.is_open())
      {
         jsonFilePath = jsonFilePath + ".json";
         jsonFile.open(jsonFilePath);
      }
      if (jsonFile.is_open())
      {
         jsonFilePath_ = jsonFilePath;
         jsonFile >> jsonData_;
         int nrElements = jsonData_.size();
         if (nrElements != 3) {
            return ERR_FILE_UNPARSABLE;
         }
         std::cout << nrElements;
         nrElements = jsonData_.size();
         if (nrElements != 3)
            return ERR_FILE_UNPARSABLE;
         json element0 = jsonData_[0];
         if (!element0.contains("wavelengths") || !element0.contains("wavelength_devices")
            || !element0.contains("wavelength_device_property")
            || !element0.contains("x_unit")
            || !element0.contains("y_unit"))
         {
            return ERR_FILE_UNPARSABLE;
         }
         std::vector<float> wavelengths = element0["wavelengths"];
         std::vector<std::string> wavelengthDevices = element0["wavelength_devices"];
         if (wavelengths.size() != wavelengthDevices.size()) 
            return ERR_FILE_UNPARSABLE;
         std::string wavelengthDeviceProperty = element0["wavelength_device_property"];
         std::string xUnit = element0["x_unit"];
         std::string yUnit = element0["y_unit"];

         channelData_.clear();
         for (int i = 0; i < wavelengths.size(); i++) {
            ChannelData* channelData = new ChannelData();
            channelData->deviceName = wavelengthDevices[i];
            channelData->xValuePropertyName = element0["wavelength_device_property"];
            channelData->xUnit = xUnit;
            channelData->yUnit = yUnit;
            channelData->wavelength = wavelengths[i];
            channelData_.push_back(*channelData);
         }

         json element1 = jsonData_[1];
         for (json element : element1)
         {
            if (!element.contains(xUnit) || !element.contains(yUnit) || !element.contains("wavelength"))
            {
               return ERR_FILE_UNPARSABLE;
            }
            float wavelength = element["wavelength"];
            bool found = false;
            for (std::vector<ChannelData>::iterator it = channelData_.begin(); it != channelData_.end() && !found; ++it)
            {
               if (it->wavelength == wavelength) {
                  std::vector<float> volts = element[xUnit];
                  it->xValues = volts;
                  std::vector<float> means = element[yUnit];
                  float max = *std::max_element(std::begin(means), std::end(means));
                  if (it->yUnit == "W" &&  max < 1.0)
                  {
                     it->yUnit = "mW";
                     for (float val : means)
                        it->yValues.push_back(val * 1000.0);
                  }
                  else
                  {
                     it->yValues = means;
                  }
                  found = true;
               }
            }

         }

      }
   }
   return DEVICE_OK;
}

int Linearizer::OnDevice(MM::PropertyBase* pProp, MM::ActionType eAct) 
{
   return DEVICE_OK;
}

int Linearizer::OnUnit(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   return DEVICE_OK;
}

int Linearizer::OnValue(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   return DEVICE_OK;
}