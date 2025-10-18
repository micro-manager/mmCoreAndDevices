///////////////////////////////////////////////////////////////////////////////
// FILE:          AutoFocus.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Various 'Meta-Devices' that add to or combine functionality of 
//                physcial devices.
//
// AUTHOR:        Nico Stuurman, nico.stuurman@ucsf.edu 2025
// COPYRIGHT:     University of California, San Francisco, 2008-2025
//                2015-2016, Open Imaging, Inc.
//                Altos Labs, 2022
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

#include "Utilities.h"

AutoFocus::AutoFocus() :
   initialized_(false),
   continuousFocusing_(false),
   offset_(0.0),
   workingDistance_(0.0),
   maxStepSizeUm_(10.0),
   stepSizeUm_(1.0),
   algorithm_(0)
{
   InitializeDefaultErrorMessages();
   SetErrorText(ERR_NO_PHYSICAL_CAMERA, "No physical camera found.  Please select a valid camera in the Camera property.");
   SetErrorText(ERR_AUTOFOCUS_NOT_SUPPORTED, "The selected camera does not support AutoFocus.");
   SetErrorText(ERR_NO_SHUTTER_DEVICE_FOUND, "No Shutter device found.  Please select a valid shutter in the Shutter property.");
   SetErrorText(ERR_NO_AUTOFOCUS_DEVICE, "No AutoFocus Device selected");
   SetErrorText(ERR_NO_AUTOFOCUS_DEVICE_FOUND, "No AutoFocus Device loaded");
   // Name
   CreateProperty(MM::g_Keyword_Name, "AutoFocus", MM::String, true);
   // Description
   CreateProperty(MM::g_Keyword_Description, "Hardware-based autofocus device that uses a shutter and a camera to determine the location/size of the reflection spot", MM::String, true);
}

AutoFocus::~AutoFocus()
{
   if (initialized_)
      Shutdown();
}

void AutoFocus::GetName(char* name) const
{
   CDeviceUtils::CopyLimitedString(name, "AutoFocus");
}

int AutoFocus::Initialize()
{
   // get list with available shutter devices.
   char deviceName[MM::MaxStrLength];
   unsigned int deviceIterator = 0;
   for (;;)
   {
      GetLoadedDeviceOfType(MM::ShutterDevice, deviceName, deviceIterator++);
      if (0 < strlen(deviceName))
      {
         availableShutters_.push_back(std::string(deviceName));
      }
      else
         break;
   }
   CPropertyAction* pAct = new CPropertyAction(this, &AutoFocus::OnShutter);
   std::string defaultShutter = "Undefined";
   if (availableShutters_.size() >= 1)
      defaultShutter = availableShutters_[0];
   CreateProperty("Shutter", defaultShutter.c_str(), MM::String, false, pAct, false);
   if (availableShutters_.size() >= 1)
      SetAllowedValues("Shutter", availableShutters_);
   else
      return ERR_NO_SHUTTER_DEVICE_FOUND;
   // This is needed, otherwise Shutter_ is not always set resulting in crashes
   // This could lead to strange problems if multiple shutter devices are loaded
   SetProperty("Shutter", defaultShutter.c_str());
   // get list with available physical cameras.
   deviceIterator = 0;
   for (;;)
   {
      GetLoadedDeviceOfType(MM::CameraDevice, deviceName, deviceIterator++);
      if (0 < strlen(deviceName))
      {
         availableCameras_.push_back(std::string(deviceName));
      }
      else
         break;
   }
   pAct = new CPropertyAction(this, &AutoFocus::OnCamera);
   std::string defaultCamera = "Undefined";
   if (availableCameras_.size() >= 1)
      defaultCamera = availableCameras_[0];
   CreateProperty("Camera", defaultCamera.c_str(), MM::String, false, pAct, false);
   if (availableCameras_.size() >= 1)
      SetAllowedValues("Camera", availableCameras_);
   else
      return ERR_NO_PHYSICAL_CAMERA;
   // This is needed, otherwise Camera_ is not always set resulting in crashes
   // This could lead to strange problems if multiple camera devices are loaded
   SetProperty("Camera", defaultCamera.c_str());

   pAct = new CPropertyAction(this, &AutoFocus::OnAlgorithm);
   CreateProperty("Algorithm", "", MM::String, false, pAct);
   AddAllowedValue("Algorithm", "Standard");


   initialized_ = true;
   return DEVICE_OK;

 }

int AutoFocus::OnShutter(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet) {
      pProp->Set(shutter_.c_str());
   }
   else if (eAct == MM::AfterSet) {
      pProp->Get(shutter_);
   }
   return DEVICE_OK;
}

int AutoFocus::OnCamera(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet) {
      pProp->Set(camera_.c_str());
   }
   else if (eAct == MM::AfterSet) {
      pProp->Get(camera_);
   }
   return DEVICE_OK;
}



