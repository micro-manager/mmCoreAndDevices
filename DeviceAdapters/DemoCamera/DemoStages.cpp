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
#ifndef _WIN32
#include <sys/time.h>
#endif

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

   pAct = new CPropertyAction(this, &CDemoXYStage::OnUsesCallbacks);
   ret = CreateStringProperty("UsesCallbacks", "Yes", false, pAct);
   if (ret != DEVICE_OK)
      return ret;
   AddAllowedValue("UsesCallbacks", "Yes");
   AddAllowedValue("UsesCallbacks", "No");

   ret = UpdateStatus();
   if (ret != DEVICE_OK)
      return ret;

   initialized_ = true;

   StartPollingThread();

   return DEVICE_OK;
}

int CDemoXYStage::Shutdown()
{
   if (initialized_)
   {
      StopPollingThread();
      initialized_ = false;
   }
   return DEVICE_OK;
}

bool CDemoXYStage::Busy()
{
   MMThreadGuard guard(moveLock_);
   if (timeOutTimer_ == nullptr)
      return false;
   return !timeOutTimer_->expired(GetCurrentMMTime());
}

int CDemoXYStage::SetPositionSteps(long x, long y)
{
   MM::MMTime currentTime = GetCurrentMMTime();
   double newTargetX = x * stepSize_um_;
   double newTargetY = y * stepSize_um_;

   double startX, startY;
   {
      MMThreadGuard guard(moveLock_);

      // Determine the current position to use as the start of the new move.
      if (timeOutTimer_ != nullptr)
      {
         if (!timeOutTimer_->expired(currentTime))
         {
            // Move still in progress: interpolate current position.
            ComputeIntermediatePosition(currentTime, startPosX_um_, startPosY_um_);
         }
         else
         {
            // Move completed but posX_um_/posY_um_ not yet updated (e.g. no polling).
            startPosX_um_ = posX_um_ = targetPosX_um_;
            startPosY_um_ = posY_um_ = targetPosY_um_;
         }
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

      startX = startPosX_um_;
      startY = startPosY_um_;
   }

   // Notify listeners of the starting position (as an acknowledgement).
   int ret = OnXYStagePositionChanged(startX, startY);
   if (ret != DEVICE_OK)
      return ret;

   // Wake the polling thread immediately so it starts reporting mid-move positions.
   if (pollingThread_ != nullptr)
      pollingThread_->NotifyMoveStarted();

   return DEVICE_OK;
}

int CDemoXYStage::GetPositionSteps(long& x, long& y)
{
   MMThreadGuard guard(moveLock_);
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

int CDemoXYStage::SetRelativePositionUm(double dx, double dy)
{
   // GetPositionUm routes through GetPositionSteps + ConvertPositionStepsToUm,
   // so it returns the true interpolated position in adapter/user coordinates
   // (with origin and mirroring applied), matching the coordinate space that
   // SetPositionUm expects.
   double currentX, currentY;
   int ret = GetPositionUm(currentX, currentY);
   if (ret != DEVICE_OK)
      return ret;
   return SetPositionUm(currentX + dx, currentY + dy);
}

// Must be called with moveLock_ held.
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

int CDemoXYStage::Stop()
{
   double posX, posY;
   {
      MMThreadGuard guard(moveLock_);
      MM::MMTime now = GetCurrentMMTime();
      if (timeOutTimer_ != nullptr)
      {
         if (!timeOutTimer_->expired(now))
            ComputeIntermediatePosition(now, posX_um_, posY_um_);
         else
         {
            posX_um_ = targetPosX_um_;
            posY_um_ = targetPosY_um_;
         }
         delete timeOutTimer_;
         timeOutTimer_ = nullptr;
      }
      posX = posX_um_;
      posY = posY_um_;
   }
   // Call outside the lock to avoid re-entrancy issues with the core callback.
   (void)OnXYStagePositionChanged(posX, posY);
   return DEVICE_OK;
}

void CDemoXYStage::StartPollingThread()
{
   stopPollingThread_ = false;
   pollingThread_ = new PollingThread(this);
   pollingThread_->activate();
}

void CDemoXYStage::StopPollingThread()
{
   if (pollingThread_ != nullptr)
   {
      stopPollingThread_ = true;
      pollingThread_->NotifyMoveStarted();  // wake thread so it sees the stop flag immediately
      pollingThread_->wait();
      delete pollingThread_;
      pollingThread_ = nullptr;
   }
}

int CDemoXYStage::OnUsesCallbacks(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(usesCallbacks_ ? "Yes" : "No");
   }
   else if (eAct == MM::AfterSet)
   {
      std::string val;
      pProp->Get(val);
      bool enable = (val == "Yes");
      if (enable == usesCallbacks_)
         return DEVICE_OK;
      usesCallbacks_ = enable;
      if (enable)
         StartPollingThread();
      else
         StopPollingThread();
   }
   return DEVICE_OK;
}

CDemoXYStage::PollingThread::PollingThread(CDemoXYStage* stage) : stage_(stage)
{
#ifdef _WIN32
   moveEvent_ = CreateEvent(NULL, FALSE, FALSE, NULL); // auto-reset
#else
   pthread_mutex_init(&eventMutex_, NULL);
   pthread_cond_init(&eventCond_, NULL);
   eventSignaled_ = false;
#endif
}

CDemoXYStage::PollingThread::~PollingThread()
{
#ifdef _WIN32
   CloseHandle(moveEvent_);
#else
   pthread_cond_destroy(&eventCond_);
   pthread_mutex_destroy(&eventMutex_);
#endif
}

void CDemoXYStage::PollingThread::NotifyMoveStarted()
{
#ifdef _WIN32
   SetEvent(moveEvent_);
#else
   pthread_mutex_lock(&eventMutex_);
   eventSignaled_ = true;
   pthread_cond_signal(&eventCond_);
   pthread_mutex_unlock(&eventMutex_);
#endif
}

void CDemoXYStage::PollingThread::WaitForMoveOrTimeout(unsigned long timeoutMs)
{
#ifdef _WIN32
   WaitForSingleObject(moveEvent_, timeoutMs);
#else
   struct timeval tv;
   gettimeofday(&tv, NULL);
   struct timespec ts;
   ts.tv_sec  = tv.tv_sec + timeoutMs / 1000;
   ts.tv_nsec = tv.tv_usec * 1000L + (timeoutMs % 1000) * 1000000L;
   if (ts.tv_nsec >= 1000000000L) { ts.tv_sec++; ts.tv_nsec -= 1000000000L; }
   pthread_mutex_lock(&eventMutex_);
   while (!eventSignaled_)
      if (pthread_cond_timedwait(&eventCond_, &eventMutex_, &ts) != 0)
         break;
   eventSignaled_ = false;
   pthread_mutex_unlock(&eventMutex_);
#endif
}

int CDemoXYStage::PollingThread::svc()
{
   while (!stage_->stopPollingThread_)
   {
      // When idle, sleep up to 1 s — NotifyMoveStarted() wakes us immediately.
      WaitForMoveOrTimeout(1000);

      // Drive position callbacks for the duration of the move.
      while (!stage_->stopPollingThread_)
      {
         double posX = 0.0, posY = 0.0;
         bool report = false;
         bool moving = false;

         {
            MMThreadGuard guard(stage_->moveLock_);
            if (stage_->timeOutTimer_ != nullptr)
            {
               MM::MMTime now = stage_->GetCurrentMMTime();
               if (!stage_->timeOutTimer_->expired(now))
               {
                  stage_->ComputeIntermediatePosition(now, posX, posY);
                  moving = true;
               }
               else
               {
                  // Move just completed — snap to target and fire one final callback.
                  stage_->posX_um_ = stage_->targetPosX_um_;
                  stage_->posY_um_ = stage_->targetPosY_um_;
                  delete stage_->timeOutTimer_;
                  stage_->timeOutTimer_ = nullptr;
                  posX = stage_->posX_um_;
                  posY = stage_->posY_um_;
               }
               report = true;
            }
         }

         if (report)
            (void)stage_->OnXYStagePositionChanged(posX, posY);

         if (!moving)
            break;  // move finished; go back to waiting for the next one

         CDeviceUtils::SleepMs(50);
      }
   }
   return 0;
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
