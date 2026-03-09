///////////////////////////////////////////////////////////////////////////////
// FILE:          StageStateDevice.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Stage used as a discrete state device. Maps N named positions
//                to stage coordinates (um), either equally spaced or
//                individually configured.
//
// COPYRIGHT:     2026 Board of Regents of the University of Wisconsin System
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

#ifdef _WIN32
#define NOMINMAX
#endif

#include "Utilities.h"

#include <string>

extern const char* g_DeviceNameStageStateDevice;
extern const char* g_NoDevice;

static const char* g_EquallySpaced = "Equally Spaced";
static const char* g_Individual = "Individual";


StageStateDevice::StageStateDevice() :
   numberOfEngagedPositions_(2),
   equallySpaced_(true),
   position0Um_(0.0),
   positionSpacingUm_(1.0),
   currentPosition_(2),
   initialized_(false),
   lastChangeTime_(0, 0)
{
   CPropertyAction* pAct = new CPropertyAction(this,
      &StageStateDevice::OnNumberOfPositions);
   CreateIntegerProperty("NumberOfPositions",
      static_cast<long>(numberOfEngagedPositions_),
      false, pAct, true);
   for (int i = 2; i <= 10; ++i)
      AddAllowedValue("NumberOfPositions", std::to_string(i).c_str());

   pAct = new CPropertyAction(this, &StageStateDevice::OnPositionMode);
   CreateStringProperty("Position Mode", g_EquallySpaced, false, pAct, true);
   AddAllowedValue("Position Mode", g_EquallySpaced);
   AddAllowedValue("Position Mode", g_Individual);

   EnableDelay(true);
}


StageStateDevice::~StageStateDevice()
{
   Shutdown();
}


int StageStateDevice::Initialize()
{
   if (initialized_)
      return DEVICE_OK;

   // PhysicalStage property
   std::vector<std::string> stageDevices;
   char deviceName[MM::MaxStrLength];
   unsigned int deviceIterator = 0;
   for (;;)
   {
      GetLoadedDeviceOfType(MM::StageDevice, deviceName, deviceIterator++);
      if (0 < strlen(deviceName))
         stageDevices.push_back(std::string(deviceName));
      else
         break;
   }

   CPropertyAction* pAct = new CPropertyAction(this,
      &StageStateDevice::OnPhysicalStage);
   int ret = CreateStringProperty("PhysicalStage", "", false, pAct);
   if (ret != DEVICE_OK)
      return ret;
   AddAllowedValue("PhysicalStage", "");
   for (const auto& s : stageDevices)
      AddAllowedValue("PhysicalStage", s.c_str());

   // Position properties
   if (equallySpaced_)
   {
      pAct = new CPropertyAction(this, &StageStateDevice::OnPosition0);
      ret = CreateFloatProperty("Position0(um)", position0Um_, false, pAct);
      if (ret != DEVICE_OK)
         return ret;

      pAct = new CPropertyAction(this, &StageStateDevice::OnPositionSpacing);
      ret = CreateFloatProperty("PositionSpacing(um)", positionSpacingUm_, false, pAct);
      if (ret != DEVICE_OK)
         return ret;
   }
   else
   {
      positionsUm_.resize(numberOfEngagedPositions_, 0.0);
      for (unsigned int i = 0; i < numberOfEngagedPositions_; ++i)
      {
         CPropertyActionEx* pActEx = new CPropertyActionEx(this,
            &StageStateDevice::OnIndividualPosition, i);
         std::string propName = "Position-" + std::to_string(i) + "(um)";
         ret = CreateFloatProperty(propName.c_str(), 0.0, false, pActEx);
         if (ret != DEVICE_OK)
            return ret;
      }
   }

   // Position labels
   for (unsigned int i = 0; i < numberOfEngagedPositions_; ++i)
      SetPositionLabel(i, std::to_string(i).c_str());
   SetPositionLabel(numberOfEngagedPositions_, "Disengaged");

   currentPosition_ = static_cast<long>(numberOfEngagedPositions_);

   // State property
   pAct = new CPropertyAction(this, &StageStateDevice::OnState);
   ret = CreateIntegerProperty(MM::g_Keyword_State, currentPosition_, false, pAct);
   if (ret != DEVICE_OK)
      return ret;
   SetPropertyLimits(MM::g_Keyword_State, 0, numberOfEngagedPositions_);

   // Label property
   pAct = new CPropertyAction(this, &StageStateDevice::OnLabel);
   ret = CreateStringProperty(MM::g_Keyword_Label, "Disengaged", false, pAct);
   if (ret != DEVICE_OK)
      return ret;

   // Closed position (gate support)
   ret = CreateIntegerProperty(MM::g_Keyword_Closed_Position, 0, false);
   if (ret != DEVICE_OK)
      return ret;
   SetPropertyLimits(MM::g_Keyword_Closed_Position, 0, numberOfEngagedPositions_ - 1);

   initialized_ = true;
   return DEVICE_OK;
}


int StageStateDevice::Shutdown()
{
   if (!initialized_)
      return DEVICE_OK;

   initialized_ = false;
   return DEVICE_OK;
}


void StageStateDevice::GetName(char* name) const
{
   CDeviceUtils::CopyLimitedString(name, g_DeviceNameStageStateDevice);
}


bool StageStateDevice::Busy()
{
   MM::Stage* stage = static_cast<MM::Stage*>(
      GetDevice(stageDeviceLabel_.c_str()));
   if (stage && stage->Busy())
      return true;

   MM::MMTime delay(GetDelayMs() * 1000.0);
   if (GetCurrentMMTime() < lastChangeTime_ + delay)
      return true;

   return false;
}


unsigned long StageStateDevice::GetNumberOfPositions() const
{
   return numberOfEngagedPositions_ + 1;
}


double StageStateDevice::PositionForState(long state) const
{
   if (equallySpaced_)
      return position0Um_ + state * positionSpacingUm_;
   else
      return positionsUm_[state];
}


int StageStateDevice::OnNumberOfPositions(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(static_cast<long>(numberOfEngagedPositions_));
   }
   else if (eAct == MM::AfterSet)
   {
      long num;
      pProp->Get(num);
      numberOfEngagedPositions_ = static_cast<unsigned int>(num);
   }
   return DEVICE_OK;
}


int StageStateDevice::OnPositionMode(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(equallySpaced_ ? g_EquallySpaced : g_Individual);
   }
   else if (eAct == MM::AfterSet)
   {
      std::string val;
      pProp->Get(val);
      equallySpaced_ = (val == g_EquallySpaced);
   }
   return DEVICE_OK;
}


int StageStateDevice::OnPhysicalStage(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(stageDeviceLabel_.c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(stageDeviceLabel_);
      currentPosition_ = static_cast<long>(numberOfEngagedPositions_);
      SetProperty(MM::g_Keyword_State, std::to_string(currentPosition_).c_str());
   }
   return DEVICE_OK;
}


int StageStateDevice::OnPosition0(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(position0Um_);
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(position0Um_);
      currentPosition_ = static_cast<long>(numberOfEngagedPositions_);
      SetProperty(MM::g_Keyword_State, std::to_string(currentPosition_).c_str());
   }
   return DEVICE_OK;
}


int StageStateDevice::OnPositionSpacing(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(positionSpacingUm_);
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(positionSpacingUm_);
      currentPosition_ = static_cast<long>(numberOfEngagedPositions_);
      SetProperty(MM::g_Keyword_State, std::to_string(currentPosition_).c_str());
   }
   return DEVICE_OK;
}


int StageStateDevice::OnIndividualPosition(MM::PropertyBase* pProp, MM::ActionType eAct, long index)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(positionsUm_[index]);
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(positionsUm_[index]);
      currentPosition_ = static_cast<long>(numberOfEngagedPositions_);
      SetProperty(MM::g_Keyword_State, std::to_string(currentPosition_).c_str());
   }
   return DEVICE_OK;
}


int StageStateDevice::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(currentPosition_);
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(currentPosition_);

      if (currentPosition_ == static_cast<long>(numberOfEngagedPositions_))
         return DEVICE_OK;

      bool gateOpen;
      GetGateOpen(gateOpen);
      long stateToApply = currentPosition_;
      if (!gateOpen)
         GetProperty(MM::g_Keyword_Closed_Position, stateToApply);

      MM::Stage* stage = static_cast<MM::Stage*>(
         GetDevice(stageDeviceLabel_.c_str()));
      if (stage)
      {
         double targetUm = PositionForState(stateToApply);
         int ret = stage->SetPositionUm(targetUm);
         lastChangeTime_ = GetCurrentMMTime();
         if (ret != DEVICE_OK)
            return ret;
      }
   }
   else if (eAct == MM::IsSequenceable)
   {
      pProp->SetSequenceable(0);
   }
   return DEVICE_OK;
}
