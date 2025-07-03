///////////////////////////////////////////////////////////////////////////////
// FILE:          FilterWheel.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Device adapter for Zaber's X-FWR series filter wheels.
//
// AUTHOR:        Soleil Lapierre (contact@zaber.com)

// COPYRIGHT:     Zaber Technologies, 2016

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

#ifdef WIN32
#pragma warning(disable: 4355)
#endif

#include "FilterWheel.h"

const char* g_FilterWheelName = "FilterWheel";
const char* g_FilterWheelDescription = "Zaber Filter Wheel";

using namespace std;

FilterWheel::FilterWheel()
: ZaberBase(this)
, deviceAddress_(1)
, numPositions_(0)
, changedTime_(0.0)
{
   this->LogMessage("FilterWheel::FilterWheel\n", true);

   InitializeDefaultErrorMessages();
   ZaberBase::setErrorMessages([&](auto code, auto message) { this->SetErrorText(code, message); });

   EnableDelay(); // signals that the delay setting will be used

   // Pre-initialization properties
   CreateProperty(MM::g_Keyword_Name, g_FilterWheelName, MM::String, true);

   CreateProperty(MM::g_Keyword_Description, "Zaber filter wheel device adapter", MM::String, true);

   CPropertyAction* pAct = new CPropertyAction(this, &FilterWheel::PortGetSet);
   CreateProperty("Zaber Serial Port", port_.c_str(), MM::String, false, pAct, true);

   pAct = new CPropertyAction (this, &FilterWheel::DeviceAddressGetSet);
   CreateIntegerProperty("Controller Device Number", deviceAddress_, false, pAct, true);
   SetPropertyLimits("Controller Device Number", 1, 99);
}


FilterWheel::~FilterWheel()
{
   this->LogMessage("FilterWheel::~FilterWheel\n", true);
   Shutdown();
}


///////////////////////////////////////////////////////////////////////////////
// Stage & Device API methods
///////////////////////////////////////////////////////////////////////////////

void FilterWheel::GetName(char* name) const
{
   CDeviceUtils::CopyLimitedString(name, g_FilterWheelName);
}


int FilterWheel::Initialize()
{
   if (initialized_)
   {
      return DEVICE_OK;
   }

   core_ = GetCoreCallback();

   this->LogMessage("FilterWheel::Initialize\n", true);

   // Make the Filter Wheel detect the current holder.
   auto ret = SendAndPollUntilIdle(deviceAddress_, 0, "tools detectholder");
   if (ret != DEVICE_OK)
   {
      this->LogMessage("Attempt to detect filter holder type failed; is this device an X-FWR?\n", true);
      return ret;
   }

   // Get the number of positions and the current position.
   long index = -1;
   ret = GetRotaryIndexedDeviceInfo(deviceAddress_, 0, numPositions_, index);
   if (ret != DEVICE_OK)
   {
      this->LogMessage("Attempt to detect filter wheel state and number of positions failed.\n", true);
      return ret;
   }

   CreateIntegerProperty("Number of Positions", numPositions_, true, 0, false);

   CPropertyAction* pAct = new CPropertyAction(this, &FilterWheel::PositionGetSet);
   CreateIntegerProperty(MM::g_Keyword_State, index, false, pAct, false);

   pAct = new CPropertyAction (this, &FilterWheel::DelayGetSet);
   ret = CreateProperty(MM::g_Keyword_Delay, "0.0", MM::Float, false, pAct);
   if (ret != DEVICE_OK)
   {
      return ret;
   }

   pAct = new CPropertyAction (this, &CStateBase::OnLabel);
   ret = CreateProperty(MM::g_Keyword_Label, "", MM::String, false, pAct);
   if (ret != DEVICE_OK)
   {
      return ret;
   }

   ret = UpdateStatus();
   if (ret != DEVICE_OK)
   {
      return ret;
   }

   if (ret == DEVICE_OK)
   {
      initialized_ = true;
      return DEVICE_OK;
   }
   else
   {
      return ret;
   }
}


int FilterWheel::Shutdown()
{
   this->LogMessage("FilterWheel::Shutdown\n", true);

   if (initialized_)
   {
      initialized_ = false;
   }

   return DEVICE_OK;
}


bool FilterWheel::Busy()
{
   this->LogMessage("FilterWheel::Busy\n", true);

   MM::MMTime interval = GetCurrentMMTime() - changedTime_;
   MM::MMTime delay(GetDelayMs()*1000.0);

   if (interval < delay)
      return true;
   else
      return IsBusy(deviceAddress_);
}


int FilterWheel::GetPositionLabel(long pos, char* label) const
{
   if (DEVICE_OK != CStateDeviceBase<FilterWheel>::GetPositionLabel(pos, label))
   {
      std::string str("Filter ");
      char numBuf[15];
      snprintf(numBuf, 15, "%ld", pos + 1);
      str.append(numBuf);
      CDeviceUtils::CopyLimitedString(label, str.c_str());
   }

   return DEVICE_OK;
}


///////////////////////////////////////////////////////////////////////////////
// Action handlers
// Handle changes and updates to property values.
///////////////////////////////////////////////////////////////////////////////

int FilterWheel::DelayGetSet(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(this->GetDelayMs());
   }
   else if (eAct == MM::AfterSet)
   {
      double delay;
      pProp->Get(delay);
      this->SetDelayMs(delay);
   }

   return DEVICE_OK;
}

int FilterWheel::PortGetSet(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   ostringstream os;
   os << "FilterWheel::PortGetSet(" << pProp << ", " << eAct << ")\n";
   this->LogMessage(os.str().c_str(), false);

   if (eAct == MM::BeforeGet)
   {
      pProp->Set(port_.c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      if (initialized_)
      {
			resetConnection();
      }

      pProp->Get(port_);
   }

   return DEVICE_OK;
}


int FilterWheel::DeviceAddressGetSet(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   this->LogMessage("FilterWheel::DeviceAddressGetSet\n", true);

   if (eAct == MM::AfterSet)
   {
      pProp->Get(deviceAddress_);
   }
   else if (eAct == MM::BeforeGet)
   {
      pProp->Set(deviceAddress_);
   }

   return DEVICE_OK;
}


int FilterWheel::PositionGetSet(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   this->LogMessage("FilterWheel::PositionGetSet\n", true);

   if (eAct == MM::BeforeGet)
   {
      long index = 1;
      if (initialized_)
      {
         int ret = GetSetting(deviceAddress_, 0, "motion.index.num", index);
         if (ret != DEVICE_OK)
         {
            return ret;
         }
      }

      // MM uses 0-based indices for states, but the Zaber Filter Wheel
      // numbers position indices starting at 1.
      pProp->Set(index - 1);
   }
   else if (eAct == MM::AfterSet)
   {
      long index;
      pProp->Get(index);

      if (initialized_)
      {
         if ((index >= 0) && (index < numPositions_))
		 {
            int ret = SendMoveCommand(deviceAddress_, 0, "index", index + 1);
            if (ret != DEVICE_OK)
            {
               return ret;
            }
            changedTime_ = GetCurrentMMTime();
		 }
         else
         {
			this->LogMessage("Requested position is outside the legal range.\n", true);
			return DEVICE_UNKNOWN_POSITION;
         }
      }
   }

   return DEVICE_OK;
}
