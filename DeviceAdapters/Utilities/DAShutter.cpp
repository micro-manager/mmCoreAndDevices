///////////////////////////////////////////////////////////////////////////////
// FILE:          DAShutter.cpp
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

extern const char* g_DeviceNameDAShutter;
extern const char* g_NoDevice;


DAShutter::DAShutter() :
   DADeviceName_(g_NoDevice),
   initialized_(false)
{
   InitializeDefaultErrorMessages();

   SetErrorText(ERR_INVALID_DEVICE_NAME, "Please select a valid DA device");
   SetErrorText(ERR_NO_DA_DEVICE, "No DA Device selected");
   SetErrorText(ERR_NO_DA_DEVICE_FOUND, "No DA Device loaded");

   // Name                                                                   
   CreateProperty(MM::g_Keyword_Name, g_DeviceNameDAShutter, MM::String, true);

   // Description                                                            
   CreateProperty(MM::g_Keyword_Description, "DA device that is used as a shutter", MM::String, true);

}

DAShutter::~DAShutter()
{
   Shutdown();
}

void DAShutter::GetName(char* Name) const
{
   CDeviceUtils::CopyLimitedString(Name, g_DeviceNameDAShutter);
}

int DAShutter::Initialize()
{
   // get list with available DA devices.
   // TODO: this is a initialization parameter, which makes it harder for the end-user to set up!
   availableDAs_.clear();
   availableDAs_.push_back(g_NoDevice);
   char deviceName[MM::MaxStrLength];
   unsigned int deviceIterator = 0;
   for (;;)
   {
      GetLoadedDeviceOfType(MM::SignalIODevice, deviceName, deviceIterator++);
      if (0 < strlen(deviceName))
      {
         availableDAs_.push_back(std::string(deviceName));
      }
      else
         break;
   }

   CPropertyAction* pAct = new CPropertyAction(this, &DAShutter::OnDADevice);
   CreateProperty("DA Device", g_NoDevice, MM::String, false, pAct, false);
   SetAllowedValues("DA Device", availableDAs_);
   SetProperty("DA Device", availableDAs_[0].c_str());

   pAct = new CPropertyAction(this, &DAShutter::OnState);
   CreateProperty("State", "0", MM::Integer, false, pAct);
   AddAllowedValue("State", "0");
   AddAllowedValue("State", "1");

   int ret = UpdateStatus();
   if (ret != DEVICE_OK)
      return ret;

   initialized_ = true;

   return DEVICE_OK;
}

bool DAShutter::Busy()
{
   MM::SignalIO* da = (MM::SignalIO*)GetDevice(DADeviceName_.c_str());
   if (da != 0)
      return da->Busy();

   // If we are here, there is a problem.  No way to report it.
   return false;
}

/*
 * Opens or closes the shutter.  Remembers voltage from the 'open' position
 */
int DAShutter::SetOpen(bool open)
{
   MM::SignalIO* da = (MM::SignalIO*)GetDevice(DADeviceName_.c_str());
   int ret = DEVICE_OK;
   if (da != 0)
      ret = da->SetGateOpen(open);
   if (ret == DEVICE_OK)
      GetCoreCallback()->OnShutterOpenChanged(this, open);
   return ret;
}

int DAShutter::GetOpen(bool& open)
{
   MM::SignalIO* da = (MM::SignalIO*)GetDevice(DADeviceName_.c_str());
   if (da != 0)
      return da->GetGateOpen(open);
   open = false;
   return DEVICE_OK;   
}

///////////////////////////////////////
// Action Interface
//////////////////////////////////////
int DAShutter::OnDADevice(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(DADeviceName_.c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      // Make sure that the "old" DA device is open:
      if (DADeviceName_ != g_NoDevice)
         SetOpen(true);

      std::string DADeviceName;
      pProp->Get(DADeviceName);
      MM::SignalIO* da = (MM::SignalIO*)GetDevice(DADeviceName.c_str());
      if (da != 0) {
         DADeviceName_ = DADeviceName;
      }
      else
         DADeviceName_ = g_NoDevice;

      // Gates are open by default.  Start with shutter closed:
      if (DADeviceName_ != g_NoDevice)
         SetOpen(false);
   }
   return DEVICE_OK;
}


int DAShutter::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      bool open;
      int ret = GetOpen(open);
      if (ret != DEVICE_OK)
         return ret;
      long state = 0;
      if (open)
         state = 1;
      pProp->Set(state);
   }
   else if (eAct == MM::AfterSet)
   {
      long state;
      pProp->Get(state);
      bool open = false;
      if (state == 1)
         open = true;
      return SetOpen(open);
   }
   return DEVICE_OK;
}
