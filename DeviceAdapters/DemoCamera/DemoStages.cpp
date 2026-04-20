///////////////////////////////////////////////////////////////////////////////
// FILE:          DemoStages.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   The example implementation of stage devices for the demo
//                camera. Simulates single axis (Z) stage and XY stage devices.
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
#include <cmath>
#include <algorithm>

// External names used by the rest of the system
extern const char* g_StageDeviceName;
extern const char* g_XYStageDeviceName;

// Global variable for intensity factor
extern double g_IntensityFactor_;

///////////////////////////////////////////////////////////////////////////////
// CDemoStage implementation
// ~~~~~~~~~~~~~~~~~~~~~~~~~

CDemoStage::CDemoStage()
{
   InitializeDefaultErrorMessages();
   SetErrorText(ERR_UNKNOWN_POSITION, "Position out of range");

   // parent ID display
   CreateHubIDProperty();
}

CDemoStage::~CDemoStage()
{
   Shutdown();
}

void CDemoStage::GetName(char* Name) const
{
   CDeviceUtils::CopyLimitedString(Name, g_StageDeviceName);
}

int CDemoStage::Initialize()
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
   int ret = CreateStringProperty(MM::g_Keyword_Name, g_StageDeviceName, true);
   if (DEVICE_OK != ret)
      return ret;

   // Description
   ret = CreateStringProperty(MM::g_Keyword_Description, "Demo stage driver", true);
   if (DEVICE_OK != ret)
      return ret;

   // Position
   // --------
   CPropertyAction* pAct = new CPropertyAction (this, &CDemoStage::OnPosition);
   ret = CreateFloatProperty(MM::g_Keyword_Position, 0, false, pAct);
   if (ret != DEVICE_OK)
      return ret;

   // Sequenceability
   // --------
   pAct = new CPropertyAction (this, &CDemoStage::OnSequence);
   ret = CreateStringProperty("UseSequences", "No", false, pAct);
   AddAllowedValue("UseSequences", "No");
   AddAllowedValue("UseSequences", "Yes");
   if (ret != DEVICE_OK)
      return ret;

   ret = UpdateStatus();
   if (ret != DEVICE_OK)
      return ret;

   initialized_ = true;

   return DEVICE_OK;
}

int CDemoStage::Shutdown()
{
   if (initialized_)
   {
      initialized_ = false;
   }
   return DEVICE_OK;
}

int CDemoStage::SetPositionUm(double pos)
{
   if (pos > upperLimit_ || lowerLimit_ > pos)
   {
      return ERR_UNKNOWN_POSITION;
   }
   pos_um_ = pos;
   SetIntensityFactor(pos);
   return OnStagePositionChanged(pos_um_);
}

// Have "focus" (i.e. max intensity) at Z=0, getting gradually dimmer as we
// get further away, without ever actually hitting 0.
// We cap the intensity factor to between .1 and 1.
void CDemoStage::SetIntensityFactor(double pos)
{
   pos = fabs(pos);
   g_IntensityFactor_ = std::max(.1, std::min(1.0, 1.0 - .2 * log(pos)));
}

int CDemoStage::IsStageSequenceable(bool& isSequenceable) const
{
   isSequenceable = sequenceable_;
   return DEVICE_OK;
}

int CDemoStage::GetStageSequenceMaxLength(long& nrEvents) const
{
   if (!sequenceable_) {
      return DEVICE_UNSUPPORTED_COMMAND;
   }

   nrEvents = 2000;
   return DEVICE_OK;
}

int CDemoStage::StartStageSequence()
{
   if (!sequenceable_) {
      return DEVICE_UNSUPPORTED_COMMAND;
   }

   return DEVICE_OK;
}

int CDemoStage::StopStageSequence()
{
   if (!sequenceable_) {
      return DEVICE_UNSUPPORTED_COMMAND;
   }

   return DEVICE_OK;
}

int CDemoStage::ClearStageSequence()
{
   if (!sequenceable_) {
      return DEVICE_UNSUPPORTED_COMMAND;
   }

   return DEVICE_OK;
}

int CDemoStage::AddToStageSequence(double /* position */)
{
   if (!sequenceable_) {
      return DEVICE_UNSUPPORTED_COMMAND;
   }

   return DEVICE_OK;
}

int CDemoStage::SendStageSequence()
{
   if (!sequenceable_) {
      return DEVICE_UNSUPPORTED_COMMAND;
   }

   return DEVICE_OK;
}


///////////////////////////////////////////////////////////////////////////////
// Action handlers
///////////////////////////////////////////////////////////////////////////////

int CDemoStage::OnPosition(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      std::stringstream s;
      s << pos_um_;
      pProp->Set(s.str().c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      double pos;
      pProp->Get(pos);
      if (pos > upperLimit_ || lowerLimit_ > pos)
      {
         pProp->Set(pos_um_); // revert
         return ERR_UNKNOWN_POSITION;
      }
      pos_um_ = pos;
      SetIntensityFactor(pos);
   }

   return DEVICE_OK;
}

int CDemoStage::OnSequence(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      std::string answer = "No";
      if (sequenceable_)
         answer = "Yes";
      pProp->Set(answer.c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      std::string answer;
      pProp->Get(answer);
      if (answer == "Yes")
         sequenceable_ = true;
      else
         sequenceable_ = false;
   }
   return DEVICE_OK;
}
///////////////////////////////////////////////////////////////////////////////
// CDemoXYStage implementation
// ~~~~~~~~~~~~~~~~~~~~~~~~~

CDemoXYStage::CDemoXYStage()
{
   InitializeDefaultErrorMessages();

   // parent ID display
   CreateHubIDProperty();
}

CDemoXYStage::~CDemoXYStage()
{
   Shutdown();
}

void CDemoXYStage::GetName(char* Name) const
{
   CDeviceUtils::CopyLimitedString(Name, g_XYStageDeviceName);
}

int CDemoXYStage::Initialize()
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
   int ret = CreateStringProperty(MM::g_Keyword_Name, g_XYStageDeviceName, true);
   if (DEVICE_OK != ret)
      return ret;

   // Description
   ret = CreateStringProperty(MM::g_Keyword_Description, "Demo XY stage driver", true);
   if (DEVICE_OK != ret)
      return ret;

   CPropertyAction* pAct = new CPropertyAction(this, &CDemoXYStage::OnVelocity);
   ret = CreateFloatProperty("Velocity", 10.0, false, pAct, false);
   if (ret != DEVICE_OK)
      return ret;

   ret = UpdateStatus();
   if (ret != DEVICE_OK)
      return ret;

   initialized_ = true;

   return DEVICE_OK;
}

int CDemoXYStage::Shutdown()
{
   if (initialized_)
   {
      initialized_ = false;
   }
   return DEVICE_OK;
}

bool CDemoXYStage::Busy()
{
   if (timeOutTimer_ == 0)
      return false;
   if (timeOutTimer_->expired(GetCurrentMMTime()))
   {
      // delete(timeOutTimer_);
      return false;
   }
   return true;
}

int CDemoXYStage::SetPositionSteps(long x, long y)
{
   MM::MMTime currentTime = GetCurrentMMTime();
   double newTargetX = x * stepSize_um_;
   double newTargetY = y * stepSize_um_;

   // If a move is in progress, compute the intermediate position and cancel the old move.
   if (timeOutTimer_ != nullptr && !timeOutTimer_->expired(currentTime))
   {
      double currentPosX, currentPosY;
      ComputeIntermediatePosition(currentTime, currentPosX, currentPosY);
      startPosX_um_ = currentPosX;
      startPosY_um_ = currentPosY;
      delete timeOutTimer_;
      timeOutTimer_ = nullptr;
   }
   else
   {
      // No move in progress; start from the last settled position.
      startPosX_um_ = posX_um_;
      startPosY_um_ = posY_um_;
   }

   // Set the new target.
   targetPosX_um_ = newTargetX;
   targetPosY_um_ = newTargetY;

   // Calculate the distance and determine the move duration (in ms)
   double difX = targetPosX_um_ - startPosX_um_;
   double difY = targetPosY_um_ - startPosY_um_;
   double distance = sqrt((difX * difX) + (difY * difY));
   moveDuration_ms_ = (long)(distance / velocity_);
   if (moveDuration_ms_ < 1)
      moveDuration_ms_ = 1;  // enforce a minimum duration

   moveStartTime_ = currentTime;
   timeOutTimer_ = new MM::TimeoutMs(currentTime, moveDuration_ms_);

   // Optionally, notify listeners of the starting position (as an acknowledgement)
   int ret = OnXYStagePositionChanged(startPosX_um_, startPosY_um_);
   if (ret != DEVICE_OK)
      return ret;

   return DEVICE_OK;
}

int CDemoXYStage::GetPositionSteps(long& x, long& y)
{
   MM::MMTime currentTime = GetCurrentMMTime();
   if (timeOutTimer_ != nullptr && !timeOutTimer_->expired(currentTime))
   {
      double currentPosX, currentPosY;
      ComputeIntermediatePosition(currentTime, currentPosX, currentPosY);
      x = (long)(currentPosX / stepSize_um_);
      y = (long)(currentPosY / stepSize_um_);
   }
   else
   {
      // Movement complete; ensure final position is set.
      if (timeOutTimer_ != nullptr)
      {
         posX_um_ = targetPosX_um_;
         posY_um_ = targetPosY_um_;
         delete timeOutTimer_;
         timeOutTimer_ = nullptr;
      }
      x = (long)(posX_um_ / stepSize_um_);
      y = (long)(posY_um_ / stepSize_um_);
   }
   return DEVICE_OK;
}

int CDemoXYStage::SetRelativePositionSteps(long x, long y)
{
   long xSteps, ySteps;
   GetPositionSteps(xSteps, ySteps);

   return this->SetPositionSteps(xSteps+x, ySteps+y);
}

// currentTime: the current time
// currentPosX, currentPosY: output parameters for the computed position in microns
void CDemoXYStage::ComputeIntermediatePosition(
   const MM::MMTime& currentTime, double& currentPosX, double& currentPosY)
{
   double elapsed_ms = (currentTime - moveStartTime_).getMsec();
   double fraction = elapsed_ms / moveDuration_ms_;
   if (fraction > 1.0)
      fraction = 1.0;
   currentPosX = startPosX_um_ + fraction * (targetPosX_um_ - startPosX_um_);
   currentPosY = startPosY_um_ + fraction * (targetPosY_um_ - startPosY_um_);
}

void CDemoXYStage::CommitCurrentIntermediatePosition_(const MM::MMTime& now)
{
   if (timeOutTimer_ && !timeOutTimer_->expired(now))
   {
      // freeze where we *are* now
      ComputeIntermediatePosition(now, posX_um_, posY_um_);
      (void)OnXYStagePositionChanged(posX_um_, posY_um_);
   }
   // Drop the timer so Busy() instantly goes idle
   delete timeOutTimer_;
   timeOutTimer_ = nullptr;
}

int CDemoXYStage::Stop()
{
   MM::MMTime now = GetCurrentMMTime();
   CommitCurrentIntermediatePosition_(now);
   return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// Action handlers
///////////////////////////////////////////////////////////////////////////////

int CDemoXYStage::OnVelocity(MM::PropertyBase* pProp, MM::ActionType eAct){
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(velocity_);
   }
   else if (eAct == MM::AfterSet)
   {
      double newVelocity;
      pProp->Get(newVelocity);
      // Enforce a minimum positive velocity
      if (newVelocity <= 0.0)
         newVelocity = 0.1;
      velocity_ = newVelocity;
   }
   return DEVICE_OK;
}
