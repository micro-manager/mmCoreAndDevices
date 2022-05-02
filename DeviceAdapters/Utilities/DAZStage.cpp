///////////////////////////////////////////////////////////////////////////////
// FILE:          DAZStage.cpp
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

extern const char* g_DeviceNameDAZStage;
extern const char* g_PropertyMinUm;
extern const char* g_PropertyMaxUm;


DAZStage::DAZStage() :
   DADeviceName_(""),
   initialized_(false),
   minDAVolt_(0.0),
   maxDAVolt_(10.0),
   minStageVolt_(0.0),
   maxStageVolt_(5.0),
   minStagePos_(0.0),
   maxStagePos_(200.0),
   pos_(0.0),
   originPos_(0.0)
{
   InitializeDefaultErrorMessages();

   SetErrorText(ERR_INVALID_DEVICE_NAME, "Please select a valid DA device");
   SetErrorText(ERR_NO_DA_DEVICE, "No DA Device selected");
   SetErrorText(ERR_VOLT_OUT_OF_RANGE, "The DA Device cannot set the requested voltage");
   SetErrorText(ERR_POS_OUT_OF_RANGE, "The requested position is out of range");
   SetErrorText(ERR_NO_DA_DEVICE_FOUND, "No DA Device loaded");

   // Name                                                                   
   CreateProperty(MM::g_Keyword_Name, g_DeviceNameDAZStage, MM::String, true);

   // Description                                                            
   CreateProperty(MM::g_Keyword_Description, "ZStage controlled with voltage provided by a DA board", MM::String, true);

   CPropertyAction* pAct = new CPropertyAction(this, &DAZStage::OnStageMinVolt);
   CreateProperty("Stage Low Voltage", "0", MM::Float, false, pAct, true);

   pAct = new CPropertyAction(this, &DAZStage::OnStageMaxVolt);
   CreateProperty("Stage High Voltage", "5", MM::Float, false, pAct, true);

   pAct = new CPropertyAction(this, &DAZStage::OnStageMinPos);
   CreateProperty(g_PropertyMinUm, "0", MM::Float, false, pAct, true);

   pAct = new CPropertyAction(this, &DAZStage::OnStageMaxPos);
   CreateProperty(g_PropertyMaxUm, "200", MM::Float, false, pAct, true);
}

DAZStage::~DAZStage()
{
}

void DAZStage::GetName(char* Name) const
{
   CDeviceUtils::CopyLimitedString(Name, g_DeviceNameDAZStage);
}

int DAZStage::Initialize()
{
   // get list with available DA devices.  
   // TODO: this is a initialization parameter, which makes it harder for the end-user to set up!
   char deviceName[MM::MaxStrLength];
   availableDAs_.clear();
   unsigned int deviceIterator = 0;
   for (;;)
   {
      GetLoadedDeviceOfType(MM::SignalIODevice, deviceName, deviceIterator++);
      if (0 < strlen(deviceName))
      {
         availableDAs_.push_back(std::string(deviceName));
      }
      else
         break;
   }



   CPropertyAction* pAct = new CPropertyAction(this, &DAZStage::OnDADevice);
   std::string defaultDA = "Undefined";
   if (availableDAs_.size() >= 1)
      defaultDA = availableDAs_[0];
   CreateProperty("DA Device", defaultDA.c_str(), MM::String, false, pAct, false);
   if (availableDAs_.size() >= 1)
      SetAllowedValues("DA Device", availableDAs_);
   else
      return ERR_NO_DA_DEVICE_FOUND;

   // This is needed, otherwise DeviceDA_ is not always set resulting in crashes
   // This could lead to strange problems if multiple DA devices are loaded
   SetProperty("DA Device", defaultDA.c_str());

   pAct = new CPropertyAction(this, &DAZStage::OnPosition);
   CreateProperty(MM::g_Keyword_Position, "0.0", MM::Float, false, pAct);
   double minPos = 0.0;
   int ret = GetProperty(g_PropertyMinUm, minPos);
   assert(ret == DEVICE_OK);
   double maxPos = 0.0;
   ret = GetProperty(g_PropertyMaxUm, maxPos);
   assert(ret == DEVICE_OK);
   SetPropertyLimits(MM::g_Keyword_Position, minPos, maxPos);

   ret = UpdateStatus();
   if (ret != DEVICE_OK)
      return ret;

   std::ostringstream tmp;
   tmp << DADeviceName_;
   LogMessage(tmp.str().c_str());

   MM::SignalIO* da = (MM::SignalIO*)GetDevice(DADeviceName_.c_str());
   if (da != 0)
      da->GetLimits(minDAVolt_, maxDAVolt_);

   originPos_ = minStagePos_;

   initialized_ = true;

   return DEVICE_OK;
}

int DAZStage::Shutdown()
{
   if (initialized_)
      initialized_ = false;

   return DEVICE_OK;
}

bool DAZStage::Busy()
{
   MM::SignalIO* da = (MM::SignalIO*)GetDevice(DADeviceName_.c_str());
   if (da != 0)
      return da->Busy();

   // If we are here, there is a problem.  No way to report it.
   return false;
}

/*
 * Sets the position of the stage in um relative to the position of the origin
 */
int DAZStage::SetPositionUm(double pos)
{
   MM::SignalIO* da = (MM::SignalIO*)GetDevice(DADeviceName_.c_str());
   if (da == 0)
      return ERR_NO_DA_DEVICE;

   double volt = (pos - minStagePos_) / (maxStagePos_ - minStagePos_) * (maxStageVolt_ - minStageVolt_) + minStageVolt_;
   if (volt > maxStageVolt_ || volt < minStageVolt_)
      return ERR_POS_OUT_OF_RANGE;

   pos_ = pos;
   return da->SetSignal(volt);
}

/*
 * Reports the current position of the stage in um relative to the origin
 */
int DAZStage::GetPositionUm(double& pos)
{
   MM::SignalIO* da = (MM::SignalIO*)GetDevice(DADeviceName_.c_str());
   if (da == 0)
      return ERR_NO_DA_DEVICE;

   double volt;
   int ret = da->GetSignal(volt);
   if (ret != DEVICE_OK)
      // DA Device cannot read, set position from cache
      pos = pos_;
   else
   {
      pos = (volt - minStageVolt_) / (maxStageVolt_ - minStageVolt_) * (maxStagePos_ - minStagePos_) + minStagePos_;
      pos_ = pos;
   }

   return DEVICE_OK;
}

/*
 * Sets a voltage (in mV) on the DA, relative to the minimum Stage position
 * The origin is NOT taken into account
 */
int DAZStage::SetPositionSteps(long steps)
{
   MM::SignalIO* da = (MM::SignalIO*)GetDevice(DADeviceName_.c_str());
   if (da == 0)
      return ERR_NO_DA_DEVICE;

   // Interpret steps to be mV
   double volt = minStageVolt_ + (steps / 1000.0);
   if (volt < maxStageVolt_)
      da->SetSignal(volt);
   else
      return ERR_VOLT_OUT_OF_RANGE;

   pos_ = (volt - minStageVolt_) / (maxStageVolt_ - minStageVolt_) * (maxStagePos_ - minStagePos_) + minStagePos_;

   return DEVICE_OK;
}

int DAZStage::GetPositionSteps(long& steps)
{
   MM::SignalIO* da = (MM::SignalIO*)GetDevice(DADeviceName_.c_str());
   if (da == 0)
      return ERR_NO_DA_DEVICE;

   double volt;
   int ret = da->GetSignal(volt);
   if (ret != DEVICE_OK)
      steps = (long)((pos_ - minStagePos_) / (maxStagePos_ - minStagePos_) * (maxStageVolt_ - minStageVolt_) * 1000.0);
   else
      steps = (long)((volt - minStageVolt_) * 1000.0);

   return DEVICE_OK;
}

/*
 * Sets the origin (relative position 0) to the current absolute position
 */
int DAZStage::SetOrigin()
{
   MM::SignalIO* da = (MM::SignalIO*)GetDevice(DADeviceName_.c_str());
   if (da == 0)
      return ERR_NO_DA_DEVICE;
   /*
   double volt;
   int ret = DADevice_->GetSignal(volt);
   if (ret != DEVICE_OK)
      return ret;

   // calculate absolute current position:
   originPos_ = volt/(maxStageVolt_ - minStageVolt_) * (maxStagePos_ - minStagePos_);

   if (originPos_ < minStagePos_ || originPos_ > maxStagePos_)
      return ERR_POS_OUT_OF_RANGE;
   */

   return DEVICE_OK;
}

int DAZStage::GetLimits(double& min, double& max)
{
   min = minStagePos_;
   max = maxStagePos_;
   return DEVICE_OK;
}

int DAZStage::IsStageSequenceable(bool& isSequenceable) const
{
   MM::SignalIO* da = (MM::SignalIO*)GetDevice(DADeviceName_.c_str());
   if (da == 0)
      return ERR_NO_DA_DEVICE;
   return da->IsDASequenceable(isSequenceable);
}

int DAZStage::GetStageSequenceMaxLength(long& nrEvents) const
{
   MM::SignalIO* da = (MM::SignalIO*)GetDevice(DADeviceName_.c_str());
   if (da == 0)
      return ERR_NO_DA_DEVICE;
   return da->GetDASequenceMaxLength(nrEvents);
}

int DAZStage::StartStageSequence()
{
   MM::SignalIO* da = (MM::SignalIO*)GetDevice(DADeviceName_.c_str());
   if (da == 0)
      return ERR_NO_DA_DEVICE;
   return da->StartDASequence();
}

int DAZStage::StopStageSequence()
{
   MM::SignalIO* da = (MM::SignalIO*)GetDevice(DADeviceName_.c_str());
   if (da == 0)
      return ERR_NO_DA_DEVICE;
   return da->StopDASequence();
}

int DAZStage::ClearStageSequence()
{
   MM::SignalIO* da = (MM::SignalIO*)GetDevice(DADeviceName_.c_str());
   if (da == 0)
      return ERR_NO_DA_DEVICE;
   return da->ClearDASequence();
}

int DAZStage::AddToStageSequence(double pos)
{
   MM::SignalIO* da = (MM::SignalIO*)GetDevice(DADeviceName_.c_str());
   if (da == 0)
      return ERR_NO_DA_DEVICE;

   double voltage = (pos - minStagePos_) / (maxStagePos_ - minStagePos_) * (maxStageVolt_ - minStageVolt_) + minStageVolt_;

   if (voltage > maxStageVolt_)
      voltage = maxStageVolt_;
   else if (voltage < minStageVolt_)
      voltage = minStageVolt_;

   return da->AddToDASequence(voltage);
}

int DAZStage::SendStageSequence()
{
   MM::SignalIO* da = (MM::SignalIO*)GetDevice(DADeviceName_.c_str());
   if (da == 0)
      return ERR_NO_DA_DEVICE;
   return da->SendDASequence();
}


///////////////////////////////////////
// Action Interface
//////////////////////////////////////
int DAZStage::OnDADevice(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(DADeviceName_.c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      std::string DADeviceName;
      pProp->Get(DADeviceName);
      MM::SignalIO* da = (MM::SignalIO*)GetDevice(DADeviceName.c_str());
      if (da != 0) {
         DADeviceName_ = DADeviceName;
      }
      else
         return ERR_INVALID_DEVICE_NAME;
      if (initialized_)
         da->GetLimits(minDAVolt_, maxDAVolt_);
   }
   return DEVICE_OK;
}

int DAZStage::OnPosition(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      double pos;
      int ret = GetPositionUm(pos);
      if (ret != DEVICE_OK)
         return ret;
      pProp->Set(pos);
   }
   else if (eAct == MM::AfterSet)
   {
      double pos;
      pProp->Get(pos);
      return SetPositionUm(pos);
   }
   return DEVICE_OK;
}

int DAZStage::OnStageMinVolt(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(minStageVolt_);
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(minStageVolt_);
   }
   return DEVICE_OK;
}

int DAZStage::OnStageMaxVolt(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(maxStageVolt_);
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(maxStageVolt_);
   }
   return DEVICE_OK;
}

int DAZStage::OnStageMinPos(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(minStagePos_);
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(minStagePos_);
   }
   return DEVICE_OK;
}

int DAZStage::OnStageMaxPos(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(maxStagePos_);
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(maxStagePos_);
   }
   return DEVICE_OK;
}
