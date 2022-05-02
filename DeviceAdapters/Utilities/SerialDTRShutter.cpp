///////////////////////////////////////////////////////////////////////////////
// FILE:          SerialDTRShutter.cpp
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

extern const char* g_DeviceNameSerialDTRShutter;
extern const char* g_invertedLogicString;
extern const char* g_normalLogicString;


SerialDTRShutter::SerialDTRShutter() :
   port_(""),
   invertedLogic_(false),
   initialized_(false),
   lastMoveStartTime_(0, 0)
{
   InitializeDefaultErrorMessages();

   SetErrorText(ERR_INVALID_DEVICE_NAME, "Please select a valid port");
   SetErrorText(ERR_TIMEOUT, "Device was busy.  Try increasing the Core-Timeout property");

   // Name                                                                   
   CreateProperty(MM::g_Keyword_Name, g_DeviceNameSerialDTRShutter, MM::String, true);

   // Description                                                            
   CreateProperty(MM::g_Keyword_Description, "Serial port DTR used as a shutter", MM::String, true);

   // Port
   CPropertyAction* pAct = new CPropertyAction(this, &SerialDTRShutter::OnPort);
   CreateProperty(MM::g_Keyword_Port, "Undefined", MM::String, false, pAct, true);

   // Logic 
   pAct = new CPropertyAction(this, &SerialDTRShutter::OnLogic);
   CreateProperty("Logic", g_invertedLogicString, MM::String, false, pAct, true);
   AddAllowedValue("Logic", g_invertedLogicString);
   AddAllowedValue("Logic", g_normalLogicString);

   EnableDelay(true);
}

SerialDTRShutter::~SerialDTRShutter()
{
   Shutdown();
}

void SerialDTRShutter::GetName(char* Name) const
{
   CDeviceUtils::CopyLimitedString(Name, g_DeviceNameSerialDTRShutter);
}

int SerialDTRShutter::Initialize()
{
   initialized_ = true;

   return DEVICE_OK;
}

bool SerialDTRShutter::Busy()
{
   MM::Device* portDevice = GetCoreCallback()->GetDevice(this, port_.c_str());
   if (portDevice != 0 && portDevice->Busy())
      return true;

   MM::MMTime delay(GetDelayMs() * 1000.0);
   if (GetCoreCallback()->GetCurrentMMTime() < lastMoveStartTime_ + delay)
      return true;

   return false;
}

/*
 * Opens or closes the shutter.
 */
int SerialDTRShutter::SetOpen(bool open)
{
   MM::Device* portDevice = GetCoreCallback()->GetDevice(this, port_.c_str());
   if (portDevice == 0)
      return DEVICE_OK;

   std::string state = "Enable";
   if (!open || (open && invertedLogic_))
   {
      state = "Disable";
   }

   lastMoveStartTime_ = GetCoreCallback()->GetCurrentMMTime();
   return portDevice->SetProperty("DTR", state.c_str());
}

int SerialDTRShutter::GetOpen(bool& open)
{
   MM::Device* portDevice = GetCoreCallback()->GetDevice(this, port_.c_str());
   if (portDevice == 0)
      return DEVICE_OK;

   char buf[MM::MaxStrLength];
   int ret = portDevice->GetProperty("DTR", buf);
   if (ret != DEVICE_OK)
   {
      return ret;
   }
   if (strcmp(buf, "Enable") == 0)
   {
      open = !invertedLogic_;
   }
   else { //"Disable"
      open = invertedLogic_;
   }
   return DEVICE_OK;
}

int SerialDTRShutter::WaitWhileBusy()
{
   MM::Device* portDevice = GetCoreCallback()->GetDevice(this, port_.c_str());
   if (portDevice == 0)
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
int SerialDTRShutter::OnPort(MM::PropertyBase* pProp, MM::ActionType pAct)
{
   if (pAct == MM::BeforeGet)
   {
      pProp->Set(port_.c_str());
   }
   else if (pAct == MM::AfterSet)
   {
      pProp->Get(port_);
   }
   return DEVICE_OK;
}

int SerialDTRShutter::OnLogic(MM::PropertyBase* pProp, MM::ActionType pAct)
{
   if (pAct == MM::BeforeGet)
   {
      if (invertedLogic_)
         pProp->Set(g_invertedLogicString);
      else
         pProp->Set(g_normalLogicString);
   }
   else if (pAct == MM::AfterSet)
   {
      std::string logic;
      pProp->Get(logic);
      if (logic.compare(g_invertedLogicString) == 0)
         invertedLogic_ = true;
      else invertedLogic_ = false;
   }
   return DEVICE_OK;
}
