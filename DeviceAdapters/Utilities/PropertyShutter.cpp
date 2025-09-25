///////////////////////////////////////////////////////////////////////////////
// FILE:          PropertyShutter.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Various 'Meta-Devices' that add to or combine functionality of 
//                physcial devices.
//
// AUTHOR:        Based on DAShutter by Nico Stuurman
//                Extended for general property control by Alex Landolt
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
#include <cmath>
#include <string>

extern const char* g_DeviceNamePropertyShutter;
extern const char* g_NoDevice;

// Error codes specific to PropertyShutter
#define ERR_NO_TARGET_DEVICE         10015
#define ERR_NO_TARGET_PROPERTY       10016
#define ERR_PROPERTY_NOT_FOUND       10017

PropertyShutter::PropertyShutter() :
   targetDeviceName_(g_NoDevice),
   targetPropertyName_(""),
   openValue_("1"),
   closedValue_("0"),
   invertLogic_(false),
   shutterDelay_(0),
   initialized_(false)
{
   InitializeDefaultErrorMessages();

   SetErrorText(ERR_NO_TARGET_DEVICE, "No target device selected");
   SetErrorText(ERR_NO_TARGET_PROPERTY, "No target property selected");  
   SetErrorText(ERR_PROPERTY_NOT_FOUND, "Target property not found on device");

   // Name                                                                   
   CreateProperty(MM::g_Keyword_Name, g_DeviceNamePropertyShutter, MM::String, true);

   // Description                                                            
   CreateProperty(MM::g_Keyword_Description, "Controls any device property as a shutter", MM::String, true);
}

PropertyShutter::~PropertyShutter()
{
   Shutdown();
}

void PropertyShutter::GetName(char* Name) const
{
   CDeviceUtils::CopyLimitedString(Name, g_DeviceNamePropertyShutter);
}

int PropertyShutter::Initialize()
{
   // Get list of all available devices
   availableDevices_.clear();
   availableDevices_.push_back(g_NoDevice);
   
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

   // Target Device property
   CPropertyAction* pAct = new CPropertyAction(this, &PropertyShutter::OnTargetDevice);
   CreateProperty("Target Device", g_NoDevice, MM::String, false, pAct, false);
   SetAllowedValues("Target Device", availableDevices_);

   // Target Property (will be populated when device is selected)
   pAct = new CPropertyAction(this, &PropertyShutter::OnTargetProperty);
   CreateProperty("Target Property", "", MM::String, false, pAct, false);

   // Open Value - value to set when shutter is open
   pAct = new CPropertyAction(this, &PropertyShutter::OnOpenValue);
   CreateProperty("Open Value", "1", MM::String, false, pAct, false);

   // Closed Value - value to set when shutter is closed  
   pAct = new CPropertyAction(this, &PropertyShutter::OnClosedValue);
   CreateProperty("Closed Value", "0", MM::String, false, pAct, false);

   // Shutter Delay - additional delay after opening/closing (in milliseconds)
   pAct = new CPropertyAction(this, &PropertyShutter::OnShutterDelay);
   CreateProperty("Shutter Delay (ms)", "0", MM::Integer, false, pAct, false);
   SetPropertyLimits("Shutter Delay (ms)", 0, 5000);

   // Invert Logic - swap open/closed behavior
   pAct = new CPropertyAction(this, &PropertyShutter::OnInvertLogic);
   CreateProperty("Invert Logic", "0", MM::Integer, false, pAct, false);
   AddAllowedValue("Invert Logic", "0");
   AddAllowedValue("Invert Logic", "1");

   // State property for direct control
   pAct = new CPropertyAction(this, &PropertyShutter::OnState);
   CreateProperty("State", "0", MM::Integer, false, pAct);
   AddAllowedValue("State", "0");
   AddAllowedValue("State", "1");

   int ret = UpdateStatus();
   if (ret != DEVICE_OK)
      return ret;

   initialized_ = true;

   return DEVICE_OK;
}

bool PropertyShutter::Busy()
{
   if (targetDeviceName_ == g_NoDevice || targetDeviceName_.empty())
      return false;

   MM::Device* targetDevice = GetDevice(targetDeviceName_.c_str());
   if (targetDevice != nullptr)
      return targetDevice->Busy();

   return false;
}

int PropertyShutter::SetOpen(bool open)
{
   if (targetDeviceName_ == g_NoDevice || targetDeviceName_.empty())
      return ERR_NO_TARGET_DEVICE;
      
   if (targetPropertyName_.empty())
      return ERR_NO_TARGET_PROPERTY;

   MM::Device* targetDevice = GetDevice(targetDeviceName_.c_str());
   if (targetDevice == nullptr)
      return ERR_NO_TARGET_DEVICE;

   // Apply invert logic if needed
   bool effectiveOpen = invertLogic_ ? !open : open;
   std::string valueToSet = effectiveOpen ? openValue_ : closedValue_;
   
   int ret = targetDevice->SetProperty(targetPropertyName_.c_str(), valueToSet.c_str());
   
   if (ret == DEVICE_OK)
   {
      // Small delay to let device update
      CDeviceUtils::SleepMs(10);
      
      // Additional configurable delay after shutter operation
      if (shutterDelay_ > 0)
      {
         CDeviceUtils::SleepMs(shutterDelay_);
      }
      
      char actualValue[MM::MaxStrLength];
      int checkRet = targetDevice->GetProperty(targetPropertyName_.c_str(), actualValue);
      if (checkRet == DEVICE_OK)
      {
         // Notify about the target device property change to update GUI
         GetCoreCallback()->OnPropertyChanged(targetDevice, targetPropertyName_.c_str(), actualValue);
      }
      
      // Notify about shutter change
      GetCoreCallback()->OnShutterOpenChanged(this, open);
   }
      
   return ret;
}

int PropertyShutter::GetOpen(bool& open)
{
   if (targetDeviceName_ == g_NoDevice || targetDeviceName_.empty())
   {
      open = false;
      return DEVICE_OK;
   }
      
   if (targetPropertyName_.empty())
   {
      open = false;
      return DEVICE_OK;
   }

   MM::Device* targetDevice = GetDevice(targetDeviceName_.c_str());
   if (targetDevice == nullptr)
   {
      open = false;
      return DEVICE_OK;
   }

   char currentValue[MM::MaxStrLength];
   int ret = targetDevice->GetProperty(targetPropertyName_.c_str(), currentValue);
   if (ret != DEVICE_OK)
   {
      open = false;
      return ret;
   }

   // Check if current value matches open value
   // Handle both string and numeric comparisons
   bool propertyIsOpen = false;
   std::string currentStr(currentValue);
   
   if (openValue_ == currentStr)
   {
      // Exact string match
      propertyIsOpen = true;
   }
   else
   {
      // Try numeric comparison for floating point values
      try
      {
         double currentVal = std::stod(currentStr);
         double openVal = std::stod(openValue_);
         propertyIsOpen = (std::abs(currentVal - openVal) < 1e-6);
      }
      catch (...)
      {
         // If conversion fails, stick with string comparison result
         propertyIsOpen = false;
      }
   }
   
   // Apply invert logic if needed
   open = invertLogic_ ? !propertyIsOpen : propertyIsOpen;
   
   return DEVICE_OK;
}

void PropertyShutter::UpdateAllowedProperties()
{
   availableProperties_.clear();
   
   if (targetDeviceName_ == g_NoDevice || targetDeviceName_.empty())
   {
      SetAllowedValues("Target Property", availableProperties_);
      return;
   }

   MM::Device* targetDevice = GetDevice(targetDeviceName_.c_str());
   if (targetDevice == nullptr)
   {
      SetAllowedValues("Target Property", availableProperties_);
      return;
   }

   // Get all properties of the target device
   for (unsigned int i = 0; i < targetDevice->GetNumberOfProperties(); i++)
   {
      char propName[MM::MaxStrLength];
      targetDevice->GetPropertyName(i, propName);
      
      // Only include non-read-only properties
      bool readOnly = false;
      targetDevice->GetPropertyReadOnly(propName, readOnly);
      if (!readOnly)
      {
         availableProperties_.push_back(std::string(propName));
      }
   }

   SetAllowedValues("Target Property", availableProperties_);
}

///////////////////////////////////////
// Action Interface
///////////////////////////////////////

int PropertyShutter::OnTargetDevice(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(targetDeviceName_.c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      std::string deviceName;
      pProp->Get(deviceName);
      targetDeviceName_ = deviceName;
      
      // Update available properties for the new device
      UpdateAllowedProperties();
      
      // Reset target property when device changes
      targetPropertyName_ = "";
      SetProperty("Target Property", "");
   }
   return DEVICE_OK;
}

int PropertyShutter::OnTargetProperty(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(targetPropertyName_.c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(targetPropertyName_);
   }
   return DEVICE_OK;
}

int PropertyShutter::OnOpenValue(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(openValue_.c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(openValue_);
   }
   return DEVICE_OK;
}

int PropertyShutter::OnClosedValue(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(closedValue_.c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(closedValue_);
   }
   return DEVICE_OK;
}

int PropertyShutter::OnShutterDelay(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(shutterDelay_);
   }
   else if (eAct == MM::AfterSet)
   {
      long delay;
      pProp->Get(delay);
      shutterDelay_ = delay;
   }
   return DEVICE_OK;
}

int PropertyShutter::OnInvertLogic(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      long invertValue = invertLogic_ ? 1 : 0;
      pProp->Set(invertValue);
   }
   else if (eAct == MM::AfterSet)
   {
      long invertValue;
      pProp->Get(invertValue);
      invertLogic_ = (invertValue == 1);
   }
   return DEVICE_OK;
}

int PropertyShutter::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      bool open;
      int ret = GetOpen(open);
      if (ret != DEVICE_OK)
         return ret;
      long state = open ? 1 : 0;
      pProp->Set(state);
   }
   else if (eAct == MM::AfterSet)
   {
      long state;
      pProp->Get(state);
      bool open = (state == 1);
      return SetOpen(open);
   }
   return DEVICE_OK;
}
