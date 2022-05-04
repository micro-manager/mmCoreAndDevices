///////////////////////////////////////////////////////////////////////////////
// FILE:          SingleAxisStage.cpp
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

extern const char* g_DeviceNameSingleAxisStage;
extern const char* g_Undefined;



SingleAxisStage::SingleAxisStage() :
   useXaxis_(true),
   simulatedStepSizeUm_(0.1),
   initialized_(false),
   usedStage_(g_Undefined)
{
   InitializeDefaultErrorMessages();
   SetErrorText(ERR_INVALID_DEVICE_NAME, "Invalid stage device");
   SetErrorText(ERR_AUTOFOCUS_NOT_SUPPORTED, "Cannot use autofocus offset device as physical stage");
   SetErrorText(ERR_NO_PHYSICAL_STAGE, "No physical stage assigned");

   CreateStringProperty(MM::g_Keyword_Name, g_DeviceNameSingleAxisStage, true);
   CreateStringProperty(MM::g_Keyword_Description,
      "Use single axis of XY stage as a logical 1D stage", true);

   CreateFloatProperty("SimulatedStepSizeUm", simulatedStepSizeUm_, false,
      new CPropertyAction(this, &SingleAxisStage::OnStepSize),
      true);
}


SingleAxisStage::~SingleAxisStage()
{
   Shutdown();
}


void SingleAxisStage::GetName(char* name) const
{
   CDeviceUtils::CopyLimitedString(name, g_DeviceNameSingleAxisStage);
}


int SingleAxisStage::Initialize()
{
   if (initialized_)
      return DEVICE_OK;

   std::vector<std::string> availableStages;
   availableStages.push_back(g_Undefined);
   char stageLabel[MM::MaxStrLength];
   unsigned int index = 0;
   for (;;)
   {
      GetLoadedDeviceOfType(MM::XYStageDevice, stageLabel, index++);
      if (strlen(stageLabel))
      {
         availableStages.push_back(stageLabel);
      }
      else
      {
         break;
      }
   }

   const std::string propPhysStage("PhysicalStage");
   CreateStringProperty(propPhysStage.c_str(), usedStage_.c_str(), false,
      new CPropertyAction(this, &SingleAxisStage::OnPhysicalStage));
   SetAllowedValues(propPhysStage.c_str(), availableStages);

   const std::string propAxisUsed("PhysicalAxis");
   CreateStringProperty(propAxisUsed.c_str(), "X", false,
      new CPropertyAction(this, &SingleAxisStage::OnAxisUsed));
   AddAllowedValue(propAxisUsed.c_str(), "X");
   AddAllowedValue(propAxisUsed.c_str(), "Y");

   initialized_ = true;
   return DEVICE_OK;
}


int SingleAxisStage::Shutdown()
{
   if (!initialized_)
      return DEVICE_OK;

   return DEVICE_OK;
}


bool SingleAxisStage::Busy()
{
   MM::XYStage* stage = (MM::XYStage*)GetDevice(usedStage_.c_str());
   if (!stage)
      return DEVICE_OK;

   return stage->Busy();
}


int SingleAxisStage::Stop()
{
   MM::XYStage* stage = (MM::XYStage*)GetDevice(usedStage_.c_str());
   if (!stage)
      return DEVICE_OK;

   return stage->Stop();
}


int SingleAxisStage::SetPositionUm(double pos)
{
   MM::XYStage* stage = (MM::XYStage*)GetDevice(usedStage_.c_str());
   if (!stage)
      return ERR_NO_PHYSICAL_STAGE;

   double xpos, ypos;
   int err = stage->GetPositionUm(xpos, ypos);
   if (err != DEVICE_OK)
      return err;
   if (useXaxis_)
   {
      return stage->SetPositionUm(pos, ypos);
   }
   else
   {
      return stage->SetPositionUm(xpos, pos);
   }
}


int SingleAxisStage::SetRelativePositionUm(double d)
{
   MM::XYStage* stage = (MM::XYStage*)GetDevice(usedStage_.c_str());
   if (!stage)
      return ERR_NO_PHYSICAL_STAGE;

   if (useXaxis_)
   {
      return stage->SetRelativePositionUm(d, 0.0);
   }
   else
   {
      return stage->SetRelativePositionUm(0.0, d);
   }
}


int SingleAxisStage::GetPositionUm(double& pos)
{
   // not sure why this is called on startup but not the MultiStage version
   // for now just ignore situation where we don't have a physical stage defined

   MM::XYStage* stage = (MM::XYStage*)GetDevice(usedStage_.c_str());
   if (!stage)
      return DEVICE_OK;

   double xpos, ypos;
   int err = stage->GetPositionUm(xpos, ypos);
   if (err != DEVICE_OK)
      return err;
   if (useXaxis_)
   {
      pos = xpos;
   }
   else
   {
      pos = ypos;
   }
   return DEVICE_OK;
}


int SingleAxisStage::SetPositionSteps(long steps)
{
   double posUm = simulatedStepSizeUm_ * steps;
   return SetPositionUm(posUm);
}


int SingleAxisStage::GetPositionSteps(long& steps)
{
   double posUm;
   int err = GetPositionUm(posUm);
   if (err != DEVICE_OK)
      return err;
   steps = Round(posUm / simulatedStepSizeUm_);
   return DEVICE_OK;
}


int SingleAxisStage::GetLimits(double& lower, double& upper)
{
   MM::XYStage* stage = (MM::XYStage*)GetDevice(usedStage_.c_str());
   if (!stage)
      return ERR_NO_PHYSICAL_STAGE;

   double xMin, xMax, yMin, yMax;
   int err = stage->GetLimitsUm(xMin, xMax, yMin, yMax);
   if (err != DEVICE_OK)
      return err;
   if (useXaxis_)
   {
      lower = xMin;
      upper = xMax;
   }
   else
   {
      lower = yMin;
      upper = yMax;
   }
   return DEVICE_OK;
}


bool SingleAxisStage::IsContinuousFocusDrive() const
{
   // We disallow setting physical stages to autofocus stages, so always return
   // false.
   return false;
}


int SingleAxisStage::IsStageSequenceable(bool& isSequenceable) const
{
   MM::XYStage* stage = (MM::XYStage*)GetDevice(usedStage_.c_str());
   if (!stage)
      return ERR_NO_PHYSICAL_STAGE;

   return stage->IsXYStageSequenceable(isSequenceable);
}


int SingleAxisStage::GetStageSequenceMaxLength(long& nrEvents) const
{
   MM::XYStage* stage = (MM::XYStage*)GetDevice(usedStage_.c_str());
   if (!stage)
      return ERR_NO_PHYSICAL_STAGE;

   return stage->GetXYStageSequenceMaxLength(nrEvents);
}


int SingleAxisStage::StartStageSequence()
{
   MM::XYStage* stage = (MM::XYStage*)GetDevice(usedStage_.c_str());
   if (!stage)
      return ERR_NO_PHYSICAL_STAGE;

   return stage->StartXYStageSequence();
}


int SingleAxisStage::StopStageSequence()
{
   MM::XYStage* stage = (MM::XYStage*)GetDevice(usedStage_.c_str());
   if (!stage)
      return ERR_NO_PHYSICAL_STAGE;

   return stage->StopXYStageSequence();
}


int SingleAxisStage::ClearStageSequence()
{
   MM::XYStage* stage = (MM::XYStage*)GetDevice(usedStage_.c_str());
   if (!stage)
      return ERR_NO_PHYSICAL_STAGE;

   return stage->ClearXYStageSequence();
}


int SingleAxisStage::AddToStageSequence(double pos)
{
   MM::XYStage* stage = (MM::XYStage*)GetDevice(usedStage_.c_str());
   if (!stage)
      return ERR_NO_PHYSICAL_STAGE;

   double xpos, ypos;
   int err = stage->GetPositionUm(xpos, ypos);
   if (err != DEVICE_OK)
      return err;
   if (useXaxis_)
   {
      return stage->AddToXYStageSequence(pos, ypos);
   }
   else
   {
      return stage->AddToXYStageSequence(xpos, pos);
   }
}


int SingleAxisStage::SendStageSequence()
{
   MM::XYStage* stage = (MM::XYStage*)GetDevice(usedStage_.c_str());
   if (!stage)
      return ERR_NO_PHYSICAL_STAGE;

   return stage->SendXYStageSequence();
}


int SingleAxisStage::OnAxisUsed(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      if (useXaxis_)
      {
         pProp->Set("X");
      }
      else
      {
         pProp->Set("Y");
      }
   }
   else if (eAct == MM::AfterSet)
   {
      std::string tmpstr;
      pProp->Get(tmpstr);
      useXaxis_ = (tmpstr.compare("X") == 0);
   }
   return DEVICE_OK;
}


int SingleAxisStage::OnStepSize(MM::PropertyBase* pProp, MM::ActionType eAct)
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


int SingleAxisStage::OnPhysicalStage(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(usedStage_.c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      std::string stageLabel;
      pProp->Get(stageLabel);

      if (stageLabel == g_Undefined)
      {
         usedStage_ = g_Undefined;
      }
      else
      {
         MM::XYStage* stage = (MM::XYStage*)GetDevice(stageLabel.c_str());
         if (!stage)
         {
            pProp->Set(g_Undefined);
            return ERR_INVALID_DEVICE_NAME;
         }
         usedStage_ = stageLabel;
      }
   }
   return DEVICE_OK;
}
