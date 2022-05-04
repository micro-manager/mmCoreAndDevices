///////////////////////////////////////////////////////////////////////////////
// FILE:          AutoFocusStage.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Various 'Meta-Devices' that add to or combine functionality of 
//                physcial devices.
//
// AUTHOR:        Nico Stuurman, nico@cmp.ucsf.edu, 11/07/2008
//                DAXYStage by Ed Simmon, 11/28/2011
//                Nico Stuurman, nstuurman@altoslabs.com, 4/22/2022
// COPYRIGHT:     University of California, San Francisco, 2008
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

extern const char* g_DeviceNameAutoFocusStage;



AutoFocusStage::AutoFocusStage() :
   AutoFocusDeviceName_(""),
   initialized_(false)
{
   InitializeDefaultErrorMessages();

   SetErrorText(ERR_INVALID_DEVICE_NAME, "Please select a valid AutoFocus device");
   SetErrorText(ERR_NO_AUTOFOCUS_DEVICE, "No AutoFocus Device selected");
   SetErrorText(ERR_NO_AUTOFOCUS_DEVICE_FOUND, "No AutoFocus Device loaded");

   // Name                                                                   
   CreateProperty(MM::g_Keyword_Name, g_DeviceNameAutoFocusStage, MM::String, true);

   // Description                                                            
   CreateProperty(MM::g_Keyword_Description, "AutoFocus offset treated as a ZStage", MM::String, true);

}

AutoFocusStage::~AutoFocusStage()
{
}

void AutoFocusStage::GetName(char* name) const
{
   CDeviceUtils::CopyLimitedString(name, g_DeviceNameAutoFocusStage);
}

int AutoFocusStage::Initialize()
{
   // get list with available AutoFocus devices.
   // TODO: this is a initialization parameter, which makes it harder for the end-user to set up!
   char deviceName[MM::MaxStrLength];
   unsigned int deviceIterator = 0;
   for (;;)
   {
      GetLoadedDeviceOfType(MM::AutoFocusDevice, deviceName, deviceIterator++);
      if (0 < strlen(deviceName))
      {
         availableAutoFocusDevices_.push_back(std::string(deviceName));
      }
      else
         break;
   }

   CPropertyAction* pAct = new CPropertyAction(this, &AutoFocusStage::OnAutoFocusDevice);
   std::string defaultAutoFocus = "Undefined";
   if (availableAutoFocusDevices_.size() >= 1)
      defaultAutoFocus = availableAutoFocusDevices_[0];
   CreateProperty("AutoFocus Device", defaultAutoFocus.c_str(), MM::String, false, pAct, false);
   if (availableAutoFocusDevices_.size() >= 1)
      SetAllowedValues("AutoFocus Device", availableAutoFocusDevices_);
   else
      return ERR_NO_AUTOFOCUS_DEVICE_FOUND;

   // This is needed, otherwise DeviceAUtofocus_ is not always set resulting in crashes
   // This could lead to strange problems if multiple AutoFocus devices are loaded
   SetProperty("AutoFocus Device", defaultAutoFocus.c_str());

   int ret = UpdateStatus();
   if (ret != DEVICE_OK)
      return ret;

   std::ostringstream tmp;
   tmp << AutoFocusDeviceName_;
   LogMessage(tmp.str().c_str());

   initialized_ = true;

   return DEVICE_OK;
}

int AutoFocusStage::Shutdown()
{
   if (initialized_)
      initialized_ = false;

   return DEVICE_OK;
}

bool AutoFocusStage::Busy()
{
   MM::AutoFocus* AutoFocusDevice = (MM::AutoFocus*)GetDevice(AutoFocusDeviceName_.c_str());
   if (AutoFocusDevice != 0)
      return AutoFocusDevice->Busy();

   // If we are here, there is a problem.  No way to report it.
   return false;
}

/*
 * Sets the position of the stage in um relative to the position of the origin
 */
int AutoFocusStage::SetPositionUm(double pos)
{
   MM::AutoFocus* AutoFocusDevice = (MM::AutoFocus*)GetDevice(AutoFocusDeviceName_.c_str());
   if (AutoFocusDevice == 0)
      return ERR_NO_AUTOFOCUS_DEVICE;

   return AutoFocusDevice->SetOffset(pos);
}

/*
 * Reports the current position of the stage in um relative to the origin
 */
int AutoFocusStage::GetPositionUm(double& pos)
{
   MM::AutoFocus* AutoFocusDevice = (MM::AutoFocus*)GetDevice(AutoFocusDeviceName_.c_str());
   if (AutoFocusDevice == 0)
      return ERR_NO_AUTOFOCUS_DEVICE;

   return  AutoFocusDevice->GetOffset(pos);;
}

/*
 * Sets a voltage (in mV) on the DA, relative to the minimum Stage position
 * The origin is NOT taken into account
 */
int AutoFocusStage::SetPositionSteps(long /* steps */)
{
   MM::AutoFocus* AutoFocusDevice = (MM::AutoFocus*)GetDevice(AutoFocusDeviceName_.c_str());
   if (AutoFocusDevice == 0)
      return ERR_NO_AUTOFOCUS_DEVICE;

   return  DEVICE_UNSUPPORTED_COMMAND;
}

int AutoFocusStage::GetPositionSteps(long& /*steps */)
{
   MM::AutoFocus* AutoFocusDevice = (MM::AutoFocus*)GetDevice(AutoFocusDeviceName_.c_str());
   if (AutoFocusDevice == 0)
      return ERR_NO_AUTOFOCUS_DEVICE;

   return  DEVICE_UNSUPPORTED_COMMAND;
}

/*
 * Sets the origin (relative position 0) to the current absolute position
 */
int AutoFocusStage::SetOrigin()
{
   MM::AutoFocus* AutoFocusDevice = (MM::AutoFocus*)GetDevice(AutoFocusDeviceName_.c_str());
   if (AutoFocusDevice == 0)
      return ERR_NO_AUTOFOCUS_DEVICE;

   return  DEVICE_UNSUPPORTED_COMMAND;
}

int AutoFocusStage::GetLimits(double& /*min*/, double& /*max*/)
{
   MM::AutoFocus* AutoFocusDevice = (MM::AutoFocus*)GetDevice(AutoFocusDeviceName_.c_str());
   if (AutoFocusDevice == 0)
      return ERR_NO_AUTOFOCUS_DEVICE;

   return  DEVICE_UNSUPPORTED_COMMAND;
}


///////////////////////////////////////
// Action Interface
//////////////////////////////////////
int AutoFocusStage::OnAutoFocusDevice(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(AutoFocusDeviceName_.c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      std::string AutoFocusDeviceName;
      pProp->Get(AutoFocusDeviceName);
      MM::AutoFocus* AutoFocusDevice = (MM::AutoFocus*)GetDevice(AutoFocusDeviceName.c_str());
      if (AutoFocusDevice != 0) {
         AutoFocusDeviceName_ = AutoFocusDeviceName;
      }
      else
         return ERR_INVALID_DEVICE_NAME;
   }
   return DEVICE_OK;
}
