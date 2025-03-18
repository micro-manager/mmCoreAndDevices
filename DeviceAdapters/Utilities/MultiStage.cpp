///////////////////////////////////////////////////////////////////////////////
// FILE:          MultiStage.cpp
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

#include <algorithm>

extern const char* g_DeviceNameMultiStage;
extern const char* g_Undefined;
extern const char* g_SyncNow;


MultiStage::MultiStage() :
   nrPhysicalStages_(2),
   simulatedStepSizeUm_(0.1),
   initialized_(false)
{
   InitializeDefaultErrorMessages();
   SetErrorText(ERR_INVALID_DEVICE_NAME, "Invalid stage device");
   SetErrorText(ERR_AUTOFOCUS_NOT_SUPPORTED, "Cannot use autofocus offset device as physical stage");
   SetErrorText(ERR_NO_PHYSICAL_STAGE, "No physical stage assigned");

   CreateStringProperty(MM::g_Keyword_Name, g_DeviceNameMultiStage, true);
   CreateStringProperty(MM::g_Keyword_Description,
      "Combines multiple physical 1D stages into a single 1D stage", true);

   // Number of stages is a pre-init property because we need to create the
   // corresponding post-init properties for the affine transofrm of each
   // physical stage
   CreateIntegerProperty("NumberOfPhysicalStages", nrPhysicalStages_, false,
      new CPropertyAction(this, &MultiStage::OnNrStages),
      true);
   for (unsigned i = 0; i < 8; ++i)
   {
      AddAllowedValue("NumberOfPhysicalStages", std::to_string(i + 1).c_str());
   }

   CreateFloatProperty("SimulatedStepSizeUm", simulatedStepSizeUm_, false,
      new CPropertyAction(this, &MultiStage::OnStepSize),
      true);
}


MultiStage::~MultiStage()
{
   Shutdown();
}


void MultiStage::GetName(char* name) const
{
   CDeviceUtils::CopyLimitedString(name, g_DeviceNameMultiStage);
}


int MultiStage::Initialize()
{
   if (initialized_)
      return DEVICE_OK;

   for (unsigned i = 0; i < nrPhysicalStages_; ++i)
   {
      usedStages_.push_back(g_Undefined);
      stageScalings_.push_back(1.0);
      stageTranslations_.push_back(0.0);
   }

   std::vector<std::string> availableStages;
   availableStages.push_back(g_Undefined);
   char stageLabel[MM::MaxStrLength];
   unsigned int index = 0;
   for (;;)
   {
      GetLoadedDeviceOfType(MM::StageDevice, stageLabel, index++);
      if (strlen(stageLabel))
      {
         availableStages.push_back(stageLabel);
      }
      else
      {
         break;
      }
   }

   for (unsigned i = 0; i < nrPhysicalStages_; ++i)
   {
      const std::string displayIndex = std::to_string(i + 1);

      const std::string propPhysStage("PhysicalStage-" + displayIndex);
      CreateStringProperty(propPhysStage.c_str(), usedStages_[i].c_str(), false,
         new CPropertyActionEx(this, &MultiStage::OnPhysicalStage, i));
      SetAllowedValues(propPhysStage.c_str(), availableStages);

      const std::string propScaling("Scaling-" + displayIndex);
      CreateFloatProperty(propScaling.c_str(), stageScalings_[i], false,
         new CPropertyActionEx(this, &MultiStage::OnScaling, i));

      const std::string propTranslation("TranslationUm-" + displayIndex);
      CreateFloatProperty(propTranslation.c_str(), stageTranslations_[i], false,
         new CPropertyActionEx(this, &MultiStage::OnTranslationUm, i));
   }

   CreateStringProperty("BringPositionsIntoSync", "", false,
      new CPropertyAction(this, &MultiStage::OnBringIntoSync));
   AddAllowedValue("BringPositionsIntoSync", "");
   AddAllowedValue("BringPositionsIntoSync", g_SyncNow);

   initialized_ = true;
   return DEVICE_OK;
}


int MultiStage::Shutdown()
{
   if (!initialized_)
      return DEVICE_OK;

   usedStages_.clear();
   stageScalings_.clear();
   stageTranslations_.clear();

   return DEVICE_OK;
}


bool MultiStage::Busy()
{
   for (std::vector<std::string>::iterator it = usedStages_.begin(),
      end = usedStages_.end();
      it != end;
      ++it)
   {
      MM::Stage* stage = (MM::Stage*)GetDevice((*it).c_str());
      if (!stage)
         continue;

      if (stage->Busy())
         return true;
   }
   return false;
}


int MultiStage::Stop()
{
   // Return last error encountered, but make sure Stop() is attempted on all
   // stages.
   int ret = DEVICE_OK;

   for (std::vector<std::string>::iterator it = usedStages_.begin(),
      end = usedStages_.end();
      it != end;
      ++it)
   {
      MM::Stage* stage = (MM::Stage*)GetDevice((*it).c_str());
      if (!stage)
         continue;

      int err = stage->Stop();
      if (err != DEVICE_OK)
         ret = err;
   }

   return ret;
}


int MultiStage::Home()
{
   for (std::vector<std::string>::iterator it = usedStages_.begin(),
      end = usedStages_.end();
      it != end;
      ++it)
   {
      MM::Stage* stage = (MM::Stage*)GetDevice((*it).c_str());
      if (!stage)
         continue;

      int err = stage->Home();
      if (err != DEVICE_OK)
         return err;
   }
   return DEVICE_OK;
}


int MultiStage::SetPositionUm(double pos)
{
   for (unsigned i = 0; i < nrPhysicalStages_; ++i)
   {
      MM::Stage* stage = (MM::Stage*)GetDevice(usedStages_[i].c_str());
      if (!stage)
         continue;

      double physicalPos = stageScalings_[i] * pos + stageTranslations_[i];
      int err = stage->SetPositionUm(physicalPos);
      if (err != DEVICE_OK)
         return err;
   }
   return DEVICE_OK;
}


int MultiStage::SetRelativePositionUm(double d)
{
   for (unsigned i = 0; i < nrPhysicalStages_; ++i)
   {
      MM::Stage* stage = (MM::Stage*)GetDevice(usedStages_[i].c_str());
      if (!stage)
         continue;

      double physicalRelPos = stageScalings_[i] * d;
      int err = stage->SetRelativePositionUm(physicalRelPos);
      if (err != DEVICE_OK)
         return err;
   }
   return DEVICE_OK;
}


int MultiStage::GetPositionUm(double& pos)
{
   // TODO We should allow user to select which stage to use for position
   // readout. For now, it is the first physical stage assigned.
   for (unsigned i = 0; i < nrPhysicalStages_; ++i)
   {
      MM::Stage* stage = (MM::Stage*)GetDevice(usedStages_[i].c_str());
      if (!stage)
         continue;

      double physicalPos;
      int err = stage->GetPositionUm(physicalPos);
      if (err != DEVICE_OK)
         return err;

      pos = (physicalPos - stageTranslations_[i]) / stageScalings_[i];
      return DEVICE_OK;
   }

   return ERR_NO_PHYSICAL_STAGE;
}


int MultiStage::SetPositionSteps(long steps)
{
   double posUm = simulatedStepSizeUm_ * steps;
   return SetPositionUm(posUm);
}


int MultiStage::GetPositionSteps(long& steps)
{
   double posUm;
   int err = GetPositionUm(posUm);
   if (err != DEVICE_OK)
      return err;
   steps = Round(posUm / simulatedStepSizeUm_);
   return DEVICE_OK;
}


int MultiStage::GetLimits(double& lower, double& upper)
{
   // Return the range where all physical stages can go; error if any have
   // error.

   // It's hard to get INFINITY in a pre-C++11-compatible and
   // compiler-independent way.
   double maxLower = -1e300, minUpper = +1e300;
   bool hasStage = false;
   for (unsigned i = 0; i < nrPhysicalStages_; ++i)
   {
      MM::Stage* stage = (MM::Stage*)GetDevice(usedStages_[i].c_str());
      if (!stage)
         continue;

      double physicalLower, physicalUpper;
      int err = stage->GetLimits(physicalLower, physicalUpper);
      if (err != DEVICE_OK)
         return err;

      hasStage = true;

      double tmpLower = (physicalLower - stageTranslations_[i]) / stageScalings_[i];
      double tmpUpper = (physicalUpper - stageTranslations_[i]) / stageScalings_[i];
      if (tmpUpper < tmpLower)
      {
         std::swap(tmpLower, tmpUpper);
      }

      maxLower = std::max(maxLower, tmpLower);
      minUpper = std::min(minUpper, tmpUpper);
   }

   // We don't want to return the infinite range if no physical stages are
   // assigned.
   if (!hasStage)
      return ERR_NO_PHYSICAL_STAGE;

   lower = maxLower;
   upper = minUpper;
   return DEVICE_OK;
}


bool MultiStage::IsContinuousFocusDrive() const
{
   // We disallow setting physical stages to autofocus stages, so always return
   // false.
   return false;
}


int MultiStage::IsStageSequenceable(bool& isSequenceable) const
{
   bool hasStage = false;
   for (std::vector<std::string>::const_iterator it = usedStages_.begin(),
      end = usedStages_.end();
      it != end;
      ++it)
   {
      MM::Stage* stage = (MM::Stage*)GetDevice((*it).c_str());
      if (!stage)
         continue;

      hasStage = true;

      bool flag;
      int err = stage->IsStageSequenceable(flag);
      if (err != DEVICE_OK)
         return err;

      if (!flag)
      {
         isSequenceable = false;
         return DEVICE_OK;
      }
   }

   if (!hasStage)
      return ERR_NO_PHYSICAL_STAGE;

   isSequenceable = true;
   return DEVICE_OK;
}


int MultiStage::GetStageSequenceMaxLength(long& nrEvents) const
{
   long minNrEvents = LONG_MAX;
   bool hasStage = false;
   for (std::vector<std::string>::const_iterator it = usedStages_.begin(),
      end = usedStages_.end();
      it != end;
      ++it)
   {
      MM::Stage* stage = (MM::Stage*)GetDevice((*it).c_str());
      if (!stage)
         continue;

      hasStage = true;

      long nr;
      int err = stage->GetStageSequenceMaxLength(nr);
      if (err != DEVICE_OK)
         return err;

      minNrEvents = std::min(minNrEvents, nr);
   }

   if (!hasStage)
      return ERR_NO_PHYSICAL_STAGE;

   nrEvents = minNrEvents;
   return DEVICE_OK;
}


int MultiStage::StartStageSequence()
{
   // Keep track of started stages in order to stop upon error
   std::vector<MM::Stage*> startedStages;

   int err;
   for (std::vector<std::string>::const_iterator it = usedStages_.begin(),
      end = usedStages_.end();
      it != end;
      ++it)
   {
      MM::Stage* stage = (MM::Stage*)GetDevice((*it).c_str());
      if (!stage)
         continue;

      err = stage->StartStageSequence();
      if (err != DEVICE_OK)
         goto error;

      startedStages.push_back(stage);
   }
   return DEVICE_OK;

error:
   while (!startedStages.empty())
   {
      startedStages.back()->StopStageSequence();
      startedStages.pop_back();
   }
   return err;
}


int MultiStage::StopStageSequence()
{
   int lastErr = DEVICE_OK;
   for (std::vector<std::string>::const_iterator it = usedStages_.begin(),
      end = usedStages_.end();
      it != end;
      ++it)
   {
      MM::Stage* stage = (MM::Stage*)GetDevice((*it).c_str());
      if (!stage)
         continue;

      int err = stage->StopStageSequence();
      if (err != DEVICE_OK)
         lastErr = err;
      // Try to stop all even after error
   }
   return lastErr;
}


int MultiStage::ClearStageSequence()
{
   int lastErr = DEVICE_OK;
   for (std::vector<std::string>::iterator it = usedStages_.begin(),
      end = usedStages_.end();
      it != end;
      ++it)
   {
      MM::Stage* stage = (MM::Stage*)GetDevice((*it).c_str());
      if (!stage)
         continue;

      int err = stage->ClearStageSequence();
      if (err != DEVICE_OK)
         lastErr = err;
   }
   return lastErr;
}


int MultiStage::AddToStageSequence(double pos)
{
   for (unsigned i = 0; i < nrPhysicalStages_; ++i)
   {
      MM::Stage* stage = (MM::Stage*)GetDevice(usedStages_[i].c_str());
      if (!stage)
         continue;

      double physicalPos = stageScalings_[i] * pos + stageTranslations_[i];
      int err = stage->AddToStageSequence(physicalPos);
      if (err != DEVICE_OK)
         return err;
   }
   return DEVICE_OK;
}


int MultiStage::SendStageSequence()
{
   for (std::vector<std::string>::iterator it = usedStages_.begin(),
      end = usedStages_.end();
      it != end;
      ++it)
   {
      MM::Stage* stage = (MM::Stage*)GetDevice((*it).c_str());
      if (!stage)
         continue;

      int err = stage->SendStageSequence();
      if (err != DEVICE_OK)
         return err;
   }
   return DEVICE_OK;
}


int MultiStage::OnNrStages(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(static_cast<long>(nrPhysicalStages_));
   }
   else if (eAct == MM::AfterSet)
   {
      long n;
      pProp->Get(n);
      nrPhysicalStages_ = n;
   }
   return DEVICE_OK;
}


int MultiStage::OnStepSize(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(simulatedStepSizeUm_);
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(simulatedStepSizeUm_);
   }
   return DEVICE_OK;
}


int MultiStage::OnPhysicalStage(MM::PropertyBase* pProp, MM::ActionType eAct, long i)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(usedStages_[i].c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      std::string stageLabel;
      pProp->Get(stageLabel);

      if (stageLabel == g_Undefined)
      {
         usedStages_[i] = g_Undefined;
      }
      else
      {
         MM::Stage* stage = (MM::Stage*)GetDevice(stageLabel.c_str());
         if (!stage)
         {
            pProp->Set(g_Undefined);
            return ERR_INVALID_DEVICE_NAME;
         }
         if (stage->IsContinuousFocusDrive())
         {
            pProp->Set(g_Undefined);
            return ERR_AUTOFOCUS_NOT_SUPPORTED;
         }
         usedStages_[i] = stageLabel;
      }
   }
   return DEVICE_OK;
}


int MultiStage::OnScaling(MM::PropertyBase* pProp, MM::ActionType eAct, long i)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(stageScalings_[i]);
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(stageScalings_[i]);
   }
   return DEVICE_OK;
}


int MultiStage::OnTranslationUm(MM::PropertyBase* pProp, MM::ActionType eAct, long i)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(stageTranslations_[i]);
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(stageTranslations_[i]);
   }
   return DEVICE_OK;
}


int MultiStage::OnBringIntoSync(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set("");
   }
   else if (eAct == MM::AfterSet)
   {
      std::string s;
      pProp->Get(s);
      if (s == g_SyncNow)
      {
         int err;
         double pos;
         err = GetPositionUm(pos);
         if (err != DEVICE_OK)
            return err;
         err = SetPositionUm(pos);
         if (err != DEVICE_OK)
            return err;
      }
      pProp->Set("");
   }
   return DEVICE_OK;
}

