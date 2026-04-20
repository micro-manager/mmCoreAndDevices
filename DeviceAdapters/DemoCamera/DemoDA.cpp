///////////////////////////////////////////////////////////////////////////////
// FILE:          DemoDA.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Demo DA device implementation
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
#include <sstream>

extern const char* g_DADeviceName;
extern const char* g_DA2DeviceName;

/****
* Demo DA device
*/

DemoDA::DemoDA (uint8_t n) : n_(n)
{
   SetErrorText(ERR_SEQUENCE_INACTIVE, "Sequence triggered, but sequence is not running");

   // parent ID display
   CreateHubIDProperty();
}

DemoDA::~DemoDA() {
}

void DemoDA::GetName(char* name) const
{
   if (n_ == 0)
      CDeviceUtils::CopyLimitedString(name, g_DADeviceName);
   else if (n_ == 1)
      CDeviceUtils::CopyLimitedString(name, g_DA2DeviceName);
   else // bad!
      CDeviceUtils::CopyLimitedString(name, "ERROR");
}

int DemoDA::Initialize()
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

   // Triggers to test sequence capabilities
   CPropertyAction* pAct = new CPropertyAction (this, &DemoDA::OnTrigger);
   CreateStringProperty("Trigger", "-", false, pAct);
   AddAllowedValue("Trigger", "-");
   AddAllowedValue("Trigger", "+");

   pAct = new CPropertyAction(this, &DemoDA::OnVoltage);
   CreateFloatProperty("Voltage", 0, false, pAct);
   SetPropertyLimits("Voltage", 0.0, 10.0);

   pAct = new CPropertyAction(this, &DemoDA::OnRealVoltage);
   CreateFloatProperty("Real Voltage", 0, true, pAct);

   return DEVICE_OK;
}

int DemoDA::SetGateOpen(bool open)
{
   open_ = open;
   if (open_)
      gatedVolts_ = volt_;
   else
      gatedVolts_ = 0;

   return DEVICE_OK;
}

int DemoDA::GetGateOpen(bool& open)
{
   open = open_;
   return DEVICE_OK;
}

int DemoDA::SetSignal(double volts)
{
   volt_ = volts;
   if (open_)
      gatedVolts_ = volts;
   std::stringstream s;
   s << "Voltage set to " << volts;
   LogMessage(s.str(), false);
   return DEVICE_OK;
}

int DemoDA::GetSignal(double& volts)
{
   volts = volt_;
   return DEVICE_OK;
}

int DemoDA::SendDASequence()
{
   (const_cast<DemoDA*> (this))->SetSentSequence();
   return DEVICE_OK;
}

// private
void DemoDA::SetSentSequence()
{
   sentSequence_ = nascentSequence_;
   nascentSequence_.clear();
}

int DemoDA::ClearDASequence()
{
   nascentSequence_.clear();
   return DEVICE_OK;
}

int DemoDA::AddToDASequence(double voltage)
{
   nascentSequence_.push_back(voltage);
   return DEVICE_OK;
}

int DemoDA::OnTrigger(MM::PropertyBase* pProp, MM::ActionType eAct)
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
         if (sequenceIndex_ < sentSequence_.size()) {
            double voltage = sentSequence_[sequenceIndex_];
            int ret = SetSignal(voltage);
            if (ret != DEVICE_OK)
               return ERR_IN_SEQUENCE;
            sequenceIndex_++;
            if (sequenceIndex_ >= sentSequence_.size()) {
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

int DemoDA::OnVoltage(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      double volts = 0.0;
      GetSignal(volts);
      pProp->Set(volts);
   }
   else if (eAct == MM::AfterSet)
   {
      double volts = 0.0;
      pProp->Get(volts);
      SetSignal(volts);
   }
   return DEVICE_OK;
}

int DemoDA::OnRealVoltage(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(gatedVolts_);
   }
   return DEVICE_OK;
}
