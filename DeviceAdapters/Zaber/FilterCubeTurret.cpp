///////////////////////////////////////////////////////////////////////////////
// FILE:          FilterCubeTurret.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Device adapter for Zaber's X-FCR series filter cube turrets
//                for microscopes.
//
// AUTHOR:        Soleil Lapierre (contact@zaber.com)

// COPYRIGHT:     Zaber Technologies Inc., 2019

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

#include "FilterCubeTurret.h"

const char* g_FilterTurretName = "FilterCubeTurret";
const char* g_FilterTurretDescription = "Zaber Filter Cube Turret";

using namespace std;

FilterCubeTurret::FilterCubeTurret()
: ZaberBase(this)
, deviceAddress_(1)
, numPositions_(0)
, changedTime_(0.0)
{
   this->LogMessage("FilterCubeTurret::FilterCubeTurret\n", true);

   InitializeDefaultErrorMessages();
   ZaberBase::setErrorMessages([&](auto code, auto message) { this->SetErrorText(code, message); });

   EnableDelay(); // signals that the delay setting will be used

   // Pre-initialization properties
   CreateProperty(MM::g_Keyword_Name, g_FilterTurretName, MM::String, true);

   CreateProperty(MM::g_Keyword_Description, "Zaber filter cube turret device adapter", MM::String, true);

   CPropertyAction* pAct = new CPropertyAction(this, &FilterCubeTurret::PortGetSet);
   CreateProperty("Zaber Serial Port", port_.c_str(), MM::String, false, pAct, true);

   pAct = new CPropertyAction (this, &FilterCubeTurret::DeviceAddressGetSet);
   CreateIntegerProperty("Controller Device Number", deviceAddress_, false, pAct, true);
   SetPropertyLimits("Controller Device Number", 1, 99);
}


FilterCubeTurret::~FilterCubeTurret()
{
   this->LogMessage("FilterCubeTurret::~FilterCubeTurret\n", true);
   Shutdown();
}


///////////////////////////////////////////////////////////////////////////////
// Stage & Device API methods
///////////////////////////////////////////////////////////////////////////////

void FilterCubeTurret::GetName(char* name) const
{
   CDeviceUtils::CopyLimitedString(name, g_FilterTurretName);
}


int FilterCubeTurret::Initialize()
{
   if (initialized_)
   {
      return DEVICE_OK;
   }

   core_ = GetCoreCallback();

   this->LogMessage("FilterCubeTurret::Initialize\n", true);

   // Home the device to make sure it has its index.
   auto ret = SendAndPollUntilIdle(deviceAddress_, 0, "home");
   if (ret != DEVICE_OK)
   {
      this->LogMessage("Attempt to detect filter holder type failed; is this device an X-FWR?\n", true);
      return ret;
   }

   long index = -1;
   ret = GetRotaryIndexedDeviceInfo(deviceAddress_, 0, numPositions_, index);
   if (ret != DEVICE_OK)
   {
      this->LogMessage("Attempt to detect filter cube turret state and number of positions failed.\n", true);
      return ret;
   }

   CreateIntegerProperty("Number of Positions", numPositions_, true, 0, false);

   CPropertyAction* pAct = new CPropertyAction(this, &FilterCubeTurret::PositionGetSet);
   CreateIntegerProperty(MM::g_Keyword_State, index, false, pAct, false);

   pAct = new CPropertyAction (this, &FilterCubeTurret::DelayGetSet);
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


int FilterCubeTurret::Shutdown()
{
   this->LogMessage("FilterWFilterCubeTurretheel::Shutdown\n", true);

   if (initialized_)
   {
      initialized_ = false;
   }

   return DEVICE_OK;
}


bool FilterCubeTurret::Busy()
{
   this->LogMessage("FilterCubeTurret::Busy\n", true);

   MM::MMTime interval = GetCurrentMMTime() - changedTime_;
   MM::MMTime delay(GetDelayMs()*1000.0);

   if (interval < delay)
   {
	   return true;
   }
   else
   {
      return IsBusy(deviceAddress_);
   }
}


int FilterCubeTurret::GetPositionLabel(long pos, char* label) const
{
   if (DEVICE_OK != CStateDeviceBase<FilterCubeTurret>::GetPositionLabel(pos, label))
   {
      std::string str("Filter Cube ");
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

int FilterCubeTurret::DelayGetSet(MM::PropertyBase* pProp, MM::ActionType eAct)
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


int FilterCubeTurret::PortGetSet(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   ostringstream os;
   os << "FilterCubeTurret::PortGetSet(" << pProp << ", " << eAct << ")\n";
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


int FilterCubeTurret::DeviceAddressGetSet(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   this->LogMessage("FilterCubeTurret::DeviceAddressGetSet\n", true);

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


int FilterCubeTurret::PositionGetSet(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   this->LogMessage("FilterCubeTurret::PositionGetSet\n", true);

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
