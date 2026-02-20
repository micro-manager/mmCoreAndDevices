///////////////////////////////////////////////////////////////////////////////
// FILE:          DemoStateDevices.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   The example implementation of state devices for the demo
//                camera. Simulates filter wheel, state device, light path,
//                and objective turret devices.
//
// AUTHOR:        Nenad Amodaj, nenad@amodaj.com, 06/08/2005
//
// COPYRIGHT:     University of California, San Francisco, 2006
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

#include "DemoCamera.h"
#include <cstdio>
#include <sstream>

// External names used by the rest of the system
extern const char* g_WheelDeviceName;
extern const char* g_StateDeviceName;
extern const char* g_LightPathDeviceName;
extern const char* g_ObjectiveDeviceName;

///////////////////////////////////////////////////////////////////////////////
// CDemoFilterWheel implementation
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CDemoFilterWheel::CDemoFilterWheel()
{
   InitializeDefaultErrorMessages();
   SetErrorText(ERR_UNKNOWN_POSITION, "Requested position not available in this device");
   EnableDelay(); // signals that the delay setting will be used
   // parent ID display
   CreateHubIDProperty();
}

CDemoFilterWheel::~CDemoFilterWheel()
{
   Shutdown();
}

void CDemoFilterWheel::GetName(char* Name) const
{
   CDeviceUtils::CopyLimitedString(Name, g_WheelDeviceName);
}


int CDemoFilterWheel::Initialize()
{
   DemoHub* pHub = static_cast<DemoHub*>(GetParentHub());
   if (pHub)
   {
      char hubLabel[MM::MaxStrLength];
      pHub->GetLabel(hubLabel);
      SetParentID(hubLabel); // for backward comp.
   }
   else
      LogMessage(NoHubError);

   if (initialized_)
      return DEVICE_OK;

   // set property list
   // -----------------

   // Name
   int ret = CreateStringProperty(MM::g_Keyword_Name, g_WheelDeviceName, true);
   if (DEVICE_OK != ret)
      return ret;

   // Description
   ret = CreateStringProperty(MM::g_Keyword_Description, "Demo filter wheel driver", true);
   if (DEVICE_OK != ret)
      return ret;

   // Set timer for the Busy signal, or we'll get a time-out the first time we check the state of the shutter, for good measure, go back 'delay' time into the past
   changedTime_ = GetCurrentMMTime();

   // Gate Closed Position
   ret = CreateIntegerProperty(MM::g_Keyword_Closed_Position, 0, false);
   if (ret != DEVICE_OK)
      return ret;

   // create default positions and labels
   const int bufSize = 1024;
   char buf[bufSize];
   for (long i=0; i<numPos_; i++)
   {
      snprintf(buf, bufSize, "State-%ld", i);
      SetPositionLabel(i, buf);
      snprintf(buf, bufSize, "%ld", i);
      AddAllowedValue(MM::g_Keyword_Closed_Position, buf);
   }

   // State
   // -----
   CPropertyAction* pAct = new CPropertyAction (this, &CDemoFilterWheel::OnState);
   ret = CreateIntegerProperty(MM::g_Keyword_State, 0, false, pAct);
   if (ret != DEVICE_OK)
      return ret;

   // Label
   // -----
   pAct = new CPropertyAction (this, &CStateBase::OnLabel);
   ret = CreateStringProperty(MM::g_Keyword_Label, "", false, pAct);
   if (ret != DEVICE_OK)
      return ret;

   ret = UpdateStatus();
   if (ret != DEVICE_OK)
      return ret;

   initialized_ = true;

   return DEVICE_OK;
}

bool CDemoFilterWheel::Busy()
{
   MM::MMTime interval = GetCurrentMMTime() - changedTime_;
   MM::MMTime delay(GetDelayMs()*1000.0);
   if (interval < delay)
      return true;
   else
      return false;
}


int CDemoFilterWheel::Shutdown()
{
   if (initialized_)
   {
      initialized_ = false;
   }
   return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// Action handlers
///////////////////////////////////////////////////////////////////////////////

int CDemoFilterWheel::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(position_);
      // nothing to do, let the caller to use cached property
   }
   else if (eAct == MM::AfterSet)
   {
      // Set timer for the Busy signal
      changedTime_ = GetCurrentMMTime();

      long pos;
      pProp->Get(pos);
      if (pos >= numPos_ || pos < 0)
      {
         pProp->Set(position_); // revert
         return ERR_UNKNOWN_POSITION;
      }

      position_ = pos;
   }

   return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// CDemoStateDevice implementation
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CDemoStateDevice::CDemoStateDevice()
{
   InitializeDefaultErrorMessages();
   SetErrorText(ERR_UNKNOWN_POSITION, "Requested position not available in this device");
   EnableDelay(); // signals that the delay setting will be used

   // Number of positions
   // -----
   CPropertyAction* pAct = new CPropertyAction (this, &CDemoStateDevice::OnNumberOfStates);
   CreateIntegerProperty("Number of positions", 0, false, pAct, true);

   // parent ID display
   CreateHubIDProperty();

}

CDemoStateDevice::~CDemoStateDevice()
{
   Shutdown();
}

void CDemoStateDevice::GetName(char* Name) const
{
   CDeviceUtils::CopyLimitedString(Name, g_StateDeviceName);
}


int CDemoStateDevice::Initialize()
{
   DemoHub* pHub = static_cast<DemoHub*>(GetParentHub());
   if (pHub)
   {
      char hubLabel[MM::MaxStrLength];
      pHub->GetLabel(hubLabel);
      SetParentID(hubLabel); // for backward comp.
   }
   else
      LogMessage(NoHubError);

   if (initialized_)
      return DEVICE_OK;

   // set property list
   // -----------------

   // Name
   int ret = CreateStringProperty(MM::g_Keyword_Name, g_StateDeviceName, true);
   if (DEVICE_OK != ret)
      return ret;

   // Description
   ret = CreateStringProperty(MM::g_Keyword_Description, "Demo state device driver", true);
   if (DEVICE_OK != ret)
      return ret;

   // Set timer for the Busy signal, or we'll get a time-out the first time we check the state of the shutter, for good measure, go back 'delay' time into the past
   changedTime_ = GetCurrentMMTime();

   // Gate Closed Position
   ret = CreateIntegerProperty(MM::g_Keyword_Closed_Position, 0, false);
   if (ret != DEVICE_OK)
       return ret;

   // create default positions and labels
   const int bufSize = 1024;
   char buf[bufSize];
   for (long i=0; i<numPos_; i++)
   {
      snprintf(buf, bufSize, "State-%ld", i);
      SetPositionLabel(i, buf);
      snprintf(buf, bufSize, "%ld", i);
      AddAllowedValue(MM::g_Keyword_Closed_Position, buf);
   }

   // State
   // -----
   CPropertyAction* pAct = new CPropertyAction (this, &CDemoStateDevice::OnState);
   ret = CreateIntegerProperty(MM::g_Keyword_State, 0, false, pAct);
   if (ret != DEVICE_OK)
      return ret;

   // Label
   // -----
   pAct = new CPropertyAction (this, &CStateBase::OnLabel);
   ret = CreateStringProperty(MM::g_Keyword_Label, "", false, pAct);
   if (ret != DEVICE_OK)
      return ret;

   // Sequence
   // -----
   pAct = new CPropertyAction(this, &CDemoStateDevice::OnSequence);
   ret = CreateProperty("Sequence", "Off", MM::String, false, pAct);
   if (ret != DEVICE_OK)
       return ret;
   AddAllowedValue("Sequence", "On");
   AddAllowedValue("Sequence", "Off");


   ret = UpdateStatus();
   if (ret != DEVICE_OK)
      return ret;

   initialized_ = true;

   return DEVICE_OK;
}

bool CDemoStateDevice::Busy()
{
    MM::MMTime interval = GetCurrentMMTime() - changedTime_;
    MM::MMTime delay(GetDelayMs() * 1000.0);
    if (interval < delay)
        return true;
    else
        return false;
}

int CDemoStateDevice::Shutdown()
{
   if (initialized_)
   {
      initialized_ = false;
   }
   return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// Action handlers
///////////////////////////////////////////////////////////////////////////////

int CDemoStateDevice::OnSequence(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        if (sequenceOn_)
            pProp->Set("On");
        else
            pProp->Set("Off");
    }
    else if (eAct == MM::AfterSet)
    {
        std::string state;
        pProp->Get(state);
        if (state == "On")
            sequenceOn_ = true;
        else
            sequenceOn_ = false;
    }
    return DEVICE_OK;
}

int CDemoStateDevice::SetGateOpen(bool open)
{
    if (gateOpen_ != open) {
        gateOpen_ = open;
    }
    return DEVICE_OK;
}

int CDemoStateDevice::GetGateOpen(bool& open)
{
    open = gateOpen_;
    return DEVICE_OK;
}


int CDemoStateDevice::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(position_);
      // nothing to do, let the caller to use cached property
   }
   else if (eAct == MM::AfterSet)
   {
      // Set timer for the Busy signal
      changedTime_ = GetCurrentMMTime();

      long pos;
      pProp->Get(pos);
      if (pos >= numPos_ || pos < 0)
      {
         pProp->Set(position_); // revert
         return ERR_UNKNOWN_POSITION;
      }

      if (gateOpen_) {
          if ((pos == position_ && !isClosed_)) {
              return DEVICE_OK;
          }
          isClosed_ = false;
      }

      else if (!isClosed_) {
          isClosed_ = true;
      }

      position_ = pos;
      return DEVICE_OK;
   }
   else if (eAct == MM::IsSequenceable)
   {
       if (sequenceOn_)
           pProp->SetSequenceable(numPatterns_);
       else
           pProp->SetSequenceable(0);
       return DEVICE_OK;
   }
   return DEVICE_OK;
}

int CDemoStateDevice::OnNumberOfStates(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(numPos_);
   }
   else if (eAct == MM::AfterSet)
   {
      if (!initialized_)
         pProp->Get(numPos_);
   }

   return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// CDemoLightPath implementation
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CDemoLightPath::CDemoLightPath()
{
   InitializeDefaultErrorMessages();
   // parent ID display
   CreateHubIDProperty();
}

CDemoLightPath::~CDemoLightPath()
{
   Shutdown();
}

void CDemoLightPath::GetName(char* Name) const
{
   CDeviceUtils::CopyLimitedString(Name, g_LightPathDeviceName);
}


int CDemoLightPath::Initialize()
{
   DemoHub* pHub = static_cast<DemoHub*>(GetParentHub());
   if (pHub)
   {
      char hubLabel[MM::MaxStrLength];
      pHub->GetLabel(hubLabel);
      SetParentID(hubLabel); // for backward comp.
   }
   else
      LogMessage(NoHubError);

   if (initialized_)
      return DEVICE_OK;

   // set property list
   // -----------------

   // Name
   int ret = CreateStringProperty(MM::g_Keyword_Name, g_LightPathDeviceName, true);
   if (DEVICE_OK != ret)
      return ret;

   // Description
   ret = CreateStringProperty(MM::g_Keyword_Description, "Demo light-path driver", true);
   if (DEVICE_OK != ret)
      return ret;

   // create default positions and labels
   const int bufSize = 1024;
   char buf[bufSize];
   for (long i=0; i<numPos_; i++)
   {
      snprintf(buf, bufSize, "State-%ld", i);
      SetPositionLabel(i, buf);
   }

   // State
   // -----
   CPropertyAction* pAct = new CPropertyAction (this, &CDemoLightPath::OnState);
   ret = CreateIntegerProperty(MM::g_Keyword_State, 0, false, pAct);
   if (ret != DEVICE_OK)
      return ret;

   // Label
   // -----
   pAct = new CPropertyAction (this, &CStateBase::OnLabel);
   ret = CreateStringProperty(MM::g_Keyword_Label, "", false, pAct);
   if (ret != DEVICE_OK)
      return ret;

   ret = UpdateStatus();
   if (ret != DEVICE_OK)
      return ret;

   initialized_ = true;

   return DEVICE_OK;
}

int CDemoLightPath::Shutdown()
{
   if (initialized_)
   {
      initialized_ = false;
   }
   return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// Action handlers
///////////////////////////////////////////////////////////////////////////////

int CDemoLightPath::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      // nothing to do, let the caller to use cached property
   }
   else if (eAct == MM::AfterSet)
   {
      long pos;
      pProp->Get(pos);
      if (pos >= numPos_ || pos < 0)
      {
         pProp->Set(position_); // revert
         return ERR_UNKNOWN_POSITION;
      }
      position_ = pos;
   }

   return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// CDemoObjectiveTurret implementation
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CDemoObjectiveTurret::CDemoObjectiveTurret()
{
   SetErrorText(ERR_IN_SEQUENCE, "Error occurred while executing sequence");
   SetErrorText(ERR_SEQUENCE_INACTIVE, "Sequence triggered, but sequence is not running");
   InitializeDefaultErrorMessages();
   // parent ID display
   CreateHubIDProperty();
}

CDemoObjectiveTurret::~CDemoObjectiveTurret()
{
   Shutdown();
}

void CDemoObjectiveTurret::GetName(char* Name) const
{
   CDeviceUtils::CopyLimitedString(Name, g_ObjectiveDeviceName);
}


int CDemoObjectiveTurret::Initialize()
{
   DemoHub* pHub = static_cast<DemoHub*>(GetParentHub());
   if (pHub)
   {
      char hubLabel[MM::MaxStrLength];
      pHub->GetLabel(hubLabel);
      SetParentID(hubLabel); // for backward comp.
   }
   else
      LogMessage(NoHubError);

   if (initialized_)
      return DEVICE_OK;

   // set property list
   // -----------------

   // Name
   int ret = CreateStringProperty(MM::g_Keyword_Name, g_ObjectiveDeviceName, true);
   if (DEVICE_OK != ret)
      return ret;

   // Description
   ret = CreateStringProperty(MM::g_Keyword_Description, "Demo objective turret driver", true);
   if (DEVICE_OK != ret)
      return ret;

   // create default positions and labels
   const int bufSize = 1024;
   char buf[bufSize];
   for (long i=0; i<numPos_; i++)
   {
      snprintf(buf, bufSize, "Objective-%c",'A'+ (char)i);
      SetPositionLabel(i, buf);
   }

   // State
   // -----
   CPropertyAction* pAct = new CPropertyAction (this, &CDemoObjectiveTurret::OnState);
   ret = CreateIntegerProperty(MM::g_Keyword_State, 0, false, pAct);
   if (ret != DEVICE_OK)
      return ret;

   // Label
   // -----
   pAct = new CPropertyAction (this, &CStateBase::OnLabel);
   ret = CreateStringProperty(MM::g_Keyword_Label, "", false, pAct);
   if (ret != DEVICE_OK)
      return ret;

   // Triggers to test sequence capabilities
   pAct = new CPropertyAction (this, &CDemoObjectiveTurret::OnTrigger);
   ret = CreateStringProperty("Trigger", "-", false, pAct);
   AddAllowedValue("Trigger", "-");
   AddAllowedValue("Trigger", "+");

   ret = UpdateStatus();
   if (ret != DEVICE_OK)
      return ret;

   initialized_ = true;

   return DEVICE_OK;
}

int CDemoObjectiveTurret::Shutdown()
{
   if (initialized_)
   {
      initialized_ = false;
   }
   return DEVICE_OK;
}



///////////////////////////////////////////////////////////////////////////////
// Action handlers
///////////////////////////////////////////////////////////////////////////////

int CDemoObjectiveTurret::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      // nothing to do, let the caller to use cached property
   }
   else if (eAct == MM::AfterSet)
   {
      long pos;
      pProp->Get(pos);
      if (pos >= numPos_ || pos < 0)
      {
         pProp->Set(position_); // revert
         return ERR_UNKNOWN_POSITION;
      }
      position_ = pos;
      std::ostringstream os;
      os << position_;
      OnPropertyChanged("State", os.str().c_str());
      char label[MM::MaxStrLength];
      GetPositionLabel(position_, label);
      OnPropertyChanged("Label", label);
   }
   else if (eAct == MM::IsSequenceable)
   {
      pProp->SetSequenceable(sequenceMaxSize_);
   }
   else if (eAct == MM::AfterLoadSequence)
   {
      sequence_ = pProp->GetSequence();
      // DeviceBase.h checks that the vector is smaller than sequenceMaxSize_
   }
   else if (eAct == MM::StartSequence)
   {
      if (sequence_.size() > 0) {
         sequenceIndex_ = 0;
         sequenceRunning_ = true;
      }
   }
   else if (eAct  == MM::StopSequence)
   {
      sequenceRunning_ = false;
   }

   return DEVICE_OK;
}

int CDemoObjectiveTurret::OnTrigger(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set("-");
   } else if (eAct == MM::AfterSet) {
      if (!sequenceRunning_)
         return ERR_SEQUENCE_INACTIVE;
      std::string tr;
      pProp->Get(tr);
      if (tr == "+") {
         if (sequenceIndex_ < sequence_.size()) {
            std::string state = sequence_[sequenceIndex_];
            int ret = SetProperty("State", state.c_str());
            if (ret != DEVICE_OK)
               return ERR_IN_SEQUENCE;
            sequenceIndex_++;
            if (sequenceIndex_ >= sequence_.size()) {
               sequenceIndex_ = 0;
            }
         } else
         {
            return ERR_IN_SEQUENCE;
         }
      }
   }
   return DEVICE_OK;
}
