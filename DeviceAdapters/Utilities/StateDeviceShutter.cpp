///////////////////////////////////////////////////////////////////////////////
// FILE:          StateDeviceShutter.cpp
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

extern const char* g_DeviceNameStateDeviceShutter;
extern const char* g_NoDevice;


StateDeviceShutter::StateDeviceShutter() :
   stateDeviceName_(""),
   initialized_(false),
   lastMoveStartTime_(0, 0)
{
   InitializeDefaultErrorMessages();

   SetErrorText(ERR_INVALID_DEVICE_NAME, "Please select a valid State device");
   SetErrorText(ERR_NO_STATE_DEVICE, "No State Device selected");
   SetErrorText(ERR_NO_STATE_DEVICE_FOUND, "No State Device loaded");
   SetErrorText(ERR_TIMEOUT, "Device was busy.  Try increasing the Core-Timeout property");

   // Name                                                                   
   CreateProperty(MM::g_Keyword_Name, g_DeviceNameStateDeviceShutter, MM::String, true);

   // Description                                                            
   CreateProperty(MM::g_Keyword_Description, "State device that is used as a shutter", MM::String, true);

   EnableDelay(true);
}

StateDeviceShutter::~StateDeviceShutter()
{
   Shutdown();
}

void StateDeviceShutter::GetName(char* Name) const
{
   CDeviceUtils::CopyLimitedString(Name, g_DeviceNameStateDeviceShutter);
}

int StateDeviceShutter::Initialize()
{
   // get list with available DA devices. 
   char deviceName[MM::MaxStrLength];
   unsigned int deviceIterator = 0;
   for (;;)
   {
      GetLoadedDeviceOfType(MM::StateDevice, deviceName, deviceIterator++);
      if (0 < strlen(deviceName))
      {
         availableStateDevices_.push_back(std::string(deviceName));
      }
      else
         break;
   }

   std::vector<std::string>::iterator it;
   it = availableStateDevices_.begin();
   availableStateDevices_.insert(it, g_NoDevice);

   CPropertyAction* pAct = new CPropertyAction(this, &StateDeviceShutter::OnStateDevice);
   std::string defaultStateDevice = g_NoDevice;
   CreateProperty("State Device", defaultStateDevice.c_str(), MM::String, false, pAct, false);
   if (availableStateDevices_.size() >= 1)
      SetAllowedValues("State Device", availableStateDevices_);
   else
      return ERR_NO_STATE_DEVICE_FOUND;

   SetProperty("State Device", defaultStateDevice.c_str());

   initialized_ = true;

   return DEVICE_OK;
}

bool StateDeviceShutter::Busy()
{
   MM::State* stateDevice = (MM::State*)GetDevice(stateDeviceName_.c_str());
   if (stateDevice != 0 && stateDevice->Busy())
      return true;

   MM::MMTime delay(GetDelayMs() * 1000.0);
   if (GetCoreCallback()->GetCurrentMMTime() < lastMoveStartTime_ + delay)
      return true;

   return false;
}

/*
 * Opens or closes the shutter.
 */
int StateDeviceShutter::SetOpen(bool open)
{
   MM::State* stateDevice = (MM::State*)GetDevice(stateDeviceName_.c_str());
   if (stateDevice == 0)
      return DEVICE_OK;

   int ret = WaitWhileBusy();
   if (ret != DEVICE_OK)
      return ret;

   lastMoveStartTime_ = GetCoreCallback()->GetCurrentMMTime();
   return stateDevice->SetGateOpen(open);
}

int StateDeviceShutter::GetOpen(bool& open)
{
   MM::State* stateDevice = (MM::State*)GetDevice(stateDeviceName_.c_str());
   if (stateDevice == 0)
      return DEVICE_OK;

   int ret = WaitWhileBusy();
   if (ret != DEVICE_OK)
      return ret;

   return stateDevice->GetGateOpen(open);
}

int StateDeviceShutter::WaitWhileBusy()
{
   MM::State* stateDevice = (MM::State*)GetDevice(stateDeviceName_.c_str());
   if (stateDevice == 0)
      return DEVICE_OK;

   bool busy = true;
   char timeout[MM::MaxStrLength];
   GetCoreCallback()->GetDeviceProperty("Core", "TimeoutMs", timeout);
   MM::MMTime dTimeout = MM::MMTime(atof(timeout) * 1000.0);
   MM::MMTime start = GetCoreCallback()->GetCurrentMMTime();
   while (busy && (GetCoreCallback()->GetCurrentMMTime() - start) < dTimeout)
      busy = Busy();

   if (busy)
      return ERR_TIMEOUT;

   return DEVICE_OK;
}

///////////////////////////////////////
// Action Interface
//////////////////////////////////////
int StateDeviceShutter::OnStateDevice(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(stateDeviceName_.c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      // Avoid leaving a State device in the closed positions!
      SetOpen(true);

      std::string stateDeviceName;
      pProp->Get(stateDeviceName);
      if (stateDeviceName == g_NoDevice) {
         stateDeviceName_ = g_NoDevice;
      }
      else {
         MM::State* stateDevice = (MM::State*)GetDevice(stateDeviceName.c_str());
         if (stateDevice != 0) {
            stateDeviceName_ = stateDeviceName;
         }
         else {
            return ERR_INVALID_DEVICE_NAME;
         }
      }

      // Start with gate closed
      SetOpen(false);
   }
   return DEVICE_OK;
}

