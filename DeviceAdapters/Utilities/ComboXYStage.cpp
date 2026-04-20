///////////////////////////////////////////////////////////////////////////////
// FILE:          ComboXYStage.cpp
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

extern const char* g_DeviceNameComboXYStage;
extern const char* g_Undefined;



ComboXYStage::ComboXYStage() :
   simulatedXStepSizeUm_(0.01),
   simulatedYStepSizeUm_(0.01),
   initialized_(0)
{
   InitializeDefaultErrorMessages();
   SetErrorText(ERR_INVALID_DEVICE_NAME, "Invalid stage device");
   SetErrorText(ERR_NO_PHYSICAL_STAGE, "Physical X and/or Y stage not assigned");

   CreateStringProperty(MM::g_Keyword_Name, g_DeviceNameComboXYStage, true);
   CreateStringProperty(MM::g_Keyword_Description,
      "Combines multiple physical 1D stages into a single 1D stage", true);

   CreateFloatProperty("SimulatedXStepSizeUm", simulatedXStepSizeUm_, false,
      new CPropertyActionEx(this, &ComboXYStage::OnStepSize, 0),
      true);
   CreateFloatProperty("SimulatedYStepSizeUm", simulatedYStepSizeUm_, false,
      new CPropertyActionEx(this, &ComboXYStage::OnStepSize, 1),
      true);
}


ComboXYStage::~ComboXYStage()
{
   Shutdown();
}


void ComboXYStage::GetName(char* name) const
{
   CDeviceUtils::CopyLimitedString(name, g_DeviceNameComboXYStage);
}


int ComboXYStage::Initialize()
{
   if (initialized_)
      return DEVICE_OK;

   for (int i = 0; i < 2; ++i)
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

   for (unsigned i = 0; i < 2; ++i)
   {
      const std::string displayAxis((i == 0) ? "X" : "Y");

      const std::string propPhysStage("PhysicalStage-" + displayAxis);
      CreateStringProperty(propPhysStage.c_str(), usedStages_[i].c_str(), false,
         new CPropertyActionEx(this, &ComboXYStage::OnPhysicalStage, i));
      SetAllowedValues(propPhysStage.c_str(), availableStages);

      const std::string propScaling("Scaling-" + displayAxis);
      CreateFloatProperty(propScaling.c_str(), stageScalings_[i], false,
         new CPropertyActionEx(this, &ComboXYStage::OnScaling, i));

      const std::string propTranslation("TranslationUm-" + displayAxis);
      CreateFloatProperty(propTranslation.c_str(), stageTranslations_[i], false,
         new CPropertyActionEx(this, &ComboXYStage::OnTranslationUm, i));
   }

   initialized_ = true;
   return DEVICE_OK;
}


int ComboXYStage::Shutdown()
{
   if (!initialized_)
      return DEVICE_OK;

   usedStages_.clear();
   stageScalings_.clear();
   stageTranslations_.clear();

   return DEVICE_OK;
}


bool ComboXYStage::Busy()
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


int ComboXYStage::Stop()
{
   // Return last error encountered, but make sure Stop() is attempted on both
   // axes.
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


int ComboXYStage::Home()
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


int ComboXYStage::SetPositionSteps(long x, long y)
{
   LogMessage(("SetPositionSteps(" + std::to_string(x) + ", " + std::to_string(y) + ")").c_str(), true);

   for (int i = 0; i < 2; ++i)
   {
      const long posSteps = (i == 0) ? x : y;

      MM::Stage* stage = (MM::Stage*)GetDevice(usedStages_[i].c_str());
      if (!stage)
         continue;

      const double& simulatedStepSizeUm = (i == 0) ?
         simulatedXStepSizeUm_ : simulatedYStepSizeUm_;
      double logicalPosUm = static_cast<double>(posSteps) * simulatedStepSizeUm;
      double physicalPosUm = stageScalings_[i] * logicalPosUm + stageTranslations_[i];
      int err = stage->SetPositionUm(physicalPosUm);
      if (err != DEVICE_OK)
         return err;
   }
   return DEVICE_OK;
}


int ComboXYStage::GetPositionSteps(long& x, long& y)
{
   for (int i = 0; i < 2; ++i)
   {
      long& posSteps = (i == 0) ? x : y;
      const double& simulatedStepSizeUm = (i == 0) ?
         simulatedXStepSizeUm_ : simulatedYStepSizeUm_;

      MM::Stage* stage = (MM::Stage*)GetDevice(usedStages_[i].c_str());
      if (!stage)
      {
         // We can't make this an error because stage position is frequently
         // requested before anybody has a chance to set the physical stages.
         posSteps = 0;
         continue;
      }

      double physicalPosUm;
      int err = stage->GetPositionUm(physicalPosUm);
      if (err != DEVICE_OK)
         return err;

      double logicalPosUm = (physicalPosUm - stageTranslations_[i]) / stageScalings_[i];
      posSteps = Round(logicalPosUm / simulatedStepSizeUm);
   }

   LogMessage(("GetPositionSteps() -> (" + std::to_string(x) + ", " + std::to_string(y) + ")").c_str(), true);
   return DEVICE_OK;
}


int ComboXYStage::GetLimitsUm(double& xMin, double& xMax, double& yMin, double& yMax)
{
   for (int i = 0; i < 2; ++i)
   {
      double& lower = (i == 0) ? xMin : yMin;
      double& upper = (i == 0) ? xMax : yMax;

      // If client code cares about stage limits, it is probably dangerous to
      // give it fake values.
      MM::Stage* stage = (MM::Stage*)GetDevice(usedStages_[i].c_str());
      if (!stage)
         return ERR_NO_PHYSICAL_STAGE;

      double physicalLower, physicalUpper;
      int err = stage->GetLimits(physicalLower, physicalUpper);
      if (err != DEVICE_OK)
         return err;

      lower = (physicalLower - stageTranslations_[i]) / stageScalings_[i];
      upper = (physicalUpper - stageTranslations_[i]) / stageScalings_[i];
      if (upper < lower)
      {
         std::swap(lower, upper);
      }
   }
   return DEVICE_OK;
}


int ComboXYStage::GetStepLimits(long& xMin, long& xMax, long& yMin, long& yMax)
{
   double xMinUm, xMaxUm, yMinUm, yMaxUm;
   int err = GetLimitsUm(xMinUm, xMaxUm, yMinUm, yMaxUm);
   if (err != DEVICE_OK)
      return err;

   xMin = Round(xMinUm / simulatedXStepSizeUm_);
   xMax = Round(xMaxUm / simulatedXStepSizeUm_);
   yMin = Round(yMinUm / simulatedYStepSizeUm_);
   yMax = Round(yMaxUm / simulatedYStepSizeUm_);
   return DEVICE_OK;
}


int ComboXYStage::IsXYStageSequenceable(bool& isSequenceable) const
{
   for (int i = 0; i < 2; ++i)
   {
      MM::Stage* stage = (MM::Stage*)GetDevice(usedStages_[i].c_str());
      if (!stage)
         return ERR_NO_PHYSICAL_STAGE;

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
   isSequenceable = true;
   return DEVICE_OK;
}


int ComboXYStage::GetXYStageSequenceMaxLength(long& nrEvents) const
{
   long minNrEvents = LONG_MAX;
   for (int i = 0; i < 2; ++i)
   {
      MM::Stage* stage = (MM::Stage*)GetDevice(usedStages_[i].c_str());
      if (!stage)
         return ERR_NO_PHYSICAL_STAGE;

      long nr;
      int err = stage->GetStageSequenceMaxLength(nr);
      if (err != DEVICE_OK)
         return err;

      minNrEvents = std::min(minNrEvents, nr);
   }

   nrEvents = minNrEvents;
   return DEVICE_OK;
}


int ComboXYStage::StartXYStageSequence()
{
   int startedStages = 0;
   int err;
   for (int i = 0; i < 2; ++i)
   {
      MM::Stage* stage = (MM::Stage*)GetDevice(usedStages_[i].c_str());
      if (!stage)
      {
         err = ERR_NO_PHYSICAL_STAGE;
         goto error;
      }

      err = stage->StartStageSequence();
      if (err != DEVICE_OK)
         goto error;
      ++startedStages;
   }
   return DEVICE_OK;

error:
   while (startedStages > 0)
   {
      MM::Stage* stage = (MM::Stage*)GetDevice(usedStages_[--startedStages].c_str());
      stage->StopStageSequence();
   }
   return err;
}


int ComboXYStage::StopXYStageSequence()
{
   int lastErr = DEVICE_OK;
   for (int i = 0; i < 2; ++i)
   {
      MM::Stage* stage = (MM::Stage*)GetDevice(usedStages_[i].c_str());
      if (!stage)
         continue;

      int err = stage->StopStageSequence();
      if (err != DEVICE_OK)
         lastErr = err;
      // Try to stop all even after error or missing stage
   }
   return lastErr;
}


int ComboXYStage::ClearXYStageSequence()
{
   int lastErr = DEVICE_OK;
   for (int i = 0; i < 2; ++i)
   {
      MM::Stage* stage = (MM::Stage*)GetDevice(usedStages_[i].c_str());
      if (!stage)
         continue;

      int err = stage->ClearStageSequence();
      if (err != DEVICE_OK)
         lastErr = err;
   }
   return lastErr;
}


int ComboXYStage::AddToXYStageSequence(double positionX, double positionY)
{
   for (int i = 0; i < 2; ++i)
   {
      MM::Stage* stage = (MM::Stage*)GetDevice(usedStages_[i].c_str());
      if (!stage)
         return ERR_NO_PHYSICAL_STAGE;
   }
   for (int i = 0; i < 2; ++i)
   {
      MM::Stage* stage = (MM::Stage*)GetDevice(usedStages_[i].c_str());
      const double& logicalPos = (i == 0) ? positionX : positionY;
      double physicalPos = stageScalings_[i] * logicalPos + stageTranslations_[i];
      int err = stage->AddToStageSequence(physicalPos);
      if (err != DEVICE_OK)
         return err;
   }
   return DEVICE_OK;
}


int ComboXYStage::SendXYStageSequence()
{
   for (int i = 0; i < 2; ++i)
   {
      MM::Stage* stage = (MM::Stage*)GetDevice(usedStages_[i].c_str());
      if (!stage)
         return ERR_NO_PHYSICAL_STAGE;
   }
   for (int i = 0; i < 2; ++i)
   {
      MM::Stage* stage = (MM::Stage*)GetDevice(usedStages_[i].c_str());
      int err = stage->SendStageSequence();
      if (err != DEVICE_OK)
         return err;
   }
   return DEVICE_OK;
}


int ComboXYStage::OnPhysicalStage(MM::PropertyBase* pProp, MM::ActionType eAct, long xy)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(usedStages_[xy].c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      std::string stageLabel;
      pProp->Get(stageLabel);

      if (stageLabel == g_Undefined)
      {
         usedStages_[xy] = g_Undefined;
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
         usedStages_[xy] = stageLabel;
      }
   }
   return DEVICE_OK;
}


int ComboXYStage::OnStepSize(MM::PropertyBase* pProp, MM::ActionType eAct, long xy)
{
   double& simulatedStepSizeUm = (xy == 0) ?
      simulatedXStepSizeUm_ : simulatedYStepSizeUm_;

   if (eAct == MM::BeforeGet)
   {
      pProp->Set(simulatedStepSizeUm);
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(simulatedStepSizeUm);
   }

   return DEVICE_OK;
}


int ComboXYStage::OnScaling(MM::PropertyBase* pProp, MM::ActionType eAct, long xy)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(stageScalings_[xy]);
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(stageScalings_[xy]);
   }
   return DEVICE_OK;
}


int ComboXYStage::OnTranslationUm(MM::PropertyBase* pProp, MM::ActionType eAct, long xy)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(stageTranslations_[xy]);
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(stageTranslations_[xy]);
   }
   return DEVICE_OK;
}
