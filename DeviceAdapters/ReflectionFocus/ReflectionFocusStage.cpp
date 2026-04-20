///////////////////////////////////////////////////////////////////////////////
// FILE:          ReflectionFocusStage.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Stage wrapper device that exposes ReflectionFocus offset as a Z-stage.
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

#include "ReflectionFocus.h"

extern const char* g_DeviceNameReflectionFocusStage;



ReflectionFocusStage::ReflectionFocusStage() :
   AutoFocusDeviceName_(""),
   initialized_(false)
{
   InitializeDefaultErrorMessages();

   SetErrorText(ERR_INVALID_DEVICE_NAME, "Please select a valid AutoFocus device");
   SetErrorText(ERR_NO_AUTOFOCUS_DEVICE, "No AutoFocus Device selected");
   SetErrorText(ERR_NO_AUTOFOCUS_DEVICE_FOUND, "No AutoFocus Device loaded");

   // Name                                                                   
   CreateProperty(MM::g_Keyword_Name, g_DeviceNameReflectionFocusStage, MM::String, true);

   // Description                                                            
   CreateProperty(MM::g_Keyword_Description, "AutoFocus offset treated as a ZStage", MM::String, true);

}

ReflectionFocusStage::~ReflectionFocusStage()
{
}

void ReflectionFocusStage::GetName(char* name) const
{
   CDeviceUtils::CopyLimitedString(name, g_DeviceNameReflectionFocusStage);
}

int ReflectionFocusStage::Initialize()
{
   // get list with available ReflectionFocus devices.
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

   CPropertyAction* pAct = new CPropertyAction(this, &ReflectionFocusStage::OnReflectionFocusDevice);
   std::string defaultAutoFocus = "Undefined";
   if (availableAutoFocusDevices_.size() >= 1)
      defaultAutoFocus = availableAutoFocusDevices_[0];
   CreateProperty("AutoFocus Device", defaultAutoFocus.c_str(), MM::String, false, pAct, false);
   if (availableAutoFocusDevices_.size() >= 1)
      SetAllowedValues("AutoFocus Device", availableAutoFocusDevices_);
   else
      return ERR_NO_AUTOFOCUS_DEVICE_FOUND;

   // This is needed, otherwise AutoFocusDeviceName_ is not always set resulting in crashes
   // This could lead to strange problems if multiple ReflectionFocus devices are loaded
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

int ReflectionFocusStage::Shutdown()
{
   if (initialized_)
   {
      // Unregister from ReflectionFocus device
      if (AutoFocusDeviceName_ != "" && AutoFocusDeviceName_ != "Undefined")
      {
         MM::AutoFocus* afDevice = (MM::AutoFocus*)GetDevice(AutoFocusDeviceName_.c_str());
         if (afDevice != nullptr)
         {
            ReflectionFocus* pAutoFocus = dynamic_cast<ReflectionFocus*>(afDevice);
            if (pAutoFocus != nullptr)
            {
               pAutoFocus->UnregisterStage(this);
            }
         }
      }

      initialized_ = false;
   }

   return DEVICE_OK;
}

bool ReflectionFocusStage::Busy()
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
int ReflectionFocusStage::SetPositionUm(double pos)
{
   MM::AutoFocus* AutoFocusDevice = (MM::AutoFocus*)GetDevice(AutoFocusDeviceName_.c_str());
   if (AutoFocusDevice == 0)
      return ERR_NO_AUTOFOCUS_DEVICE;

   return AutoFocusDevice->SetOffset(pos);
}

/*
 * Reports the current position of the stage in um relative to the origin
 */
int ReflectionFocusStage::GetPositionUm(double& pos)
{
   MM::AutoFocus* AutoFocusDevice = (MM::AutoFocus*)GetDevice(AutoFocusDeviceName_.c_str());
   if (AutoFocusDevice == 0)
      return ERR_NO_AUTOFOCUS_DEVICE;

   return AutoFocusDevice->GetOffset(pos);
}

/*
 * Sets a voltage (in mV) on the DA, relative to the minimum Stage position
 * The origin is NOT taken into account
 */
int ReflectionFocusStage::SetPositionSteps(long /* steps */)
{
   MM::AutoFocus* AutoFocusDevice = (MM::AutoFocus*)GetDevice(AutoFocusDeviceName_.c_str());
   if (AutoFocusDevice == 0)
      return ERR_NO_AUTOFOCUS_DEVICE;

   return  DEVICE_UNSUPPORTED_COMMAND;
}

int ReflectionFocusStage::GetPositionSteps(long& /*steps */)
{
   MM::AutoFocus* AutoFocusDevice = (MM::AutoFocus*)GetDevice(AutoFocusDeviceName_.c_str());
   if (AutoFocusDevice == 0)
      return ERR_NO_AUTOFOCUS_DEVICE;

   return  DEVICE_UNSUPPORTED_COMMAND;
}

/*
 * Sets the origin (relative position 0) to the current absolute position
 */
int ReflectionFocusStage::SetOrigin()
{
   MM::AutoFocus* AutoFocusDevice = (MM::AutoFocus*)GetDevice(AutoFocusDeviceName_.c_str());
   if (AutoFocusDevice == 0)
      return ERR_NO_AUTOFOCUS_DEVICE;

   return  DEVICE_UNSUPPORTED_COMMAND;
}

int ReflectionFocusStage::GetLimits(double& /*min*/, double& /*max*/)
{
   MM::AutoFocus* AutoFocusDevice = (MM::AutoFocus*)GetDevice(AutoFocusDeviceName_.c_str());
   if (AutoFocusDevice == 0)
      return ERR_NO_AUTOFOCUS_DEVICE;

   return  DEVICE_UNSUPPORTED_COMMAND;
}


///////////////////////////////////////
// Action Interface
//////////////////////////////////////
int ReflectionFocusStage::OnReflectionFocusDevice(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(AutoFocusDeviceName_.c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      // Unregister from previous ReflectionFocus device if any
      if (AutoFocusDeviceName_ != "" && AutoFocusDeviceName_ != "Undefined")
      {
         MM::AutoFocus* oldDevice = (MM::AutoFocus*)GetDevice(AutoFocusDeviceName_.c_str());
         if (oldDevice != nullptr)
         {
            // Cast to ReflectionFocus to access RegisterStage/UnregisterStage
            ReflectionFocus* pAutoFocus = dynamic_cast<ReflectionFocus*>(oldDevice);
            if (pAutoFocus != nullptr)
            {
               pAutoFocus->UnregisterStage(this);
            }
         }
      }

      std::string AutoFocusDeviceName;
      pProp->Get(AutoFocusDeviceName);
      MM::AutoFocus* AutoFocusDevice = (MM::AutoFocus*)GetDevice(AutoFocusDeviceName.c_str());
      if (AutoFocusDevice != 0) {
         AutoFocusDeviceName_ = AutoFocusDeviceName;

         // Register with new ReflectionFocus device
         if (AutoFocusDeviceName_ != "" && AutoFocusDeviceName_ != "Undefined")
         {
            ReflectionFocus* pAutoFocus = dynamic_cast<ReflectionFocus*>(AutoFocusDevice);
            if (pAutoFocus != nullptr)
            {
               pAutoFocus->RegisterStage(this);
            }
         }
      }
      else
         return ERR_INVALID_DEVICE_NAME;
   }
   return DEVICE_OK;
}
