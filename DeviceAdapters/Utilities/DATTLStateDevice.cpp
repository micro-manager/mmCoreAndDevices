///////////////////////////////////////////////////////////////////////////////
// FILE:          MultiDAStateDevice.cpp
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

extern const char* g_DeviceNameDATTLStateDevice;
extern const char* g_normalLogicString;
extern const char* g_invertedLogicString;
extern const char* g_InvertLogic;
extern const char* g_TTLVoltage;
extern const char* g_3_3;
extern const char* g_5_0;


DATTLStateDevice::DATTLStateDevice() :
   numberOfDADevices_(1),
   initialized_(false),
   mask_(0L),
   invert_(false),
   ttlVoltage_(3.3),
   lastChangeTime_(0, 0)
{
   CPropertyAction* pAct = new CPropertyAction(this,
      &DATTLStateDevice::OnNumberOfDADevices);
   CreateIntegerProperty("NumberOfDADevices",
      static_cast<long>(numberOfDADevices_),
      false, pAct, true);
   for (int i = 1; i <= 8; ++i)
   {
      AddAllowedValue("NumberOfDADevices", std::to_string(i).c_str());
   }

   EnableDelay(true);
}


DATTLStateDevice::~DATTLStateDevice()
{
   Shutdown();
}


int DATTLStateDevice::Initialize()
{
   if (initialized_)
      return DEVICE_OK;

   daDeviceLabels_.clear();
   for (unsigned int i = 0; i < numberOfDADevices_; ++i)
   {
      daDeviceLabels_.push_back("");
   }

   // Get labels of DA (SignalIO) devices
   std::vector<std::string> daDevices;
   char deviceName[MM::MaxStrLength];
   unsigned int deviceIterator = 0;
   for (;;)
   {
      GetLoadedDeviceOfType(MM::SignalIODevice, deviceName, deviceIterator++);
      if (0 < strlen(deviceName))
      {
         daDevices.push_back(std::string(deviceName));
      }
      else
         break;
   }

   for (unsigned int i = 0; i < numberOfDADevices_; ++i)
   {
      const std::string propName = "DADevice-" + std::to_string(i);
      CPropertyActionEx* pAct = new CPropertyActionEx(this,
         &DATTLStateDevice::OnDADevice, i);
      int ret = CreateStringProperty(propName.c_str(), "", false, pAct);
      if (ret != DEVICE_OK)
         return ret;

      AddAllowedValue(propName.c_str(), "");
      for (std::vector<std::string>::const_iterator it = daDevices.begin(),
         end = daDevices.end(); it != end; ++it)
      {
         AddAllowedValue(propName.c_str(), it->c_str());
      }
   }

   int numPos = GetNumberOfPositions();
   for (int i = 0; i < numPos; ++i)
   {
      SetPositionLabel(i, std::to_string(i).c_str());
   }

   CPropertyAction* pAct = new CPropertyAction(this, &DATTLStateDevice::OnState);
   int ret = CreateIntegerProperty(MM::g_Keyword_State, 0, false, pAct);
   if (ret != DEVICE_OK)
      return ret;
   SetPropertyLimits(MM::g_Keyword_State, 0, numPos - 1);

   pAct = new CPropertyAction(this, &DATTLStateDevice::OnInvert);
   ret = CreateStringProperty(g_InvertLogic, g_normalLogicString, false, pAct);
   if (ret != DEVICE_OK)
      return ret;
   AddAllowedValue(g_InvertLogic, g_normalLogicString);
   AddAllowedValue(g_InvertLogic, g_invertedLogicString);

   pAct = new CPropertyAction(this, &DATTLStateDevice::OnTTLLevel);
   ret = CreateStringProperty(g_TTLVoltage, g_3_3, false, pAct);
   if (ret != DEVICE_OK)
      return ret;
   AddAllowedValue(g_TTLVoltage, g_3_3);
   AddAllowedValue(g_TTLVoltage, g_5_0);

   pAct = new CPropertyAction(this, &DATTLStateDevice::OnLabel);
   ret = CreateStringProperty(MM::g_Keyword_Label, "0", false, pAct);
   if (ret != DEVICE_OK)
      return ret;

   ret = CreateIntegerProperty(MM::g_Keyword_Closed_Position, 0, false);
   if (ret != DEVICE_OK)
      return ret;

   initialized_ = true;
   return DEVICE_OK;
}


int DATTLStateDevice::Shutdown()
{
   if (!initialized_)
      return DEVICE_OK;

   daDeviceLabels_.clear();

   initialized_ = false;
   return DEVICE_OK;
}


void DATTLStateDevice::GetName(char* name) const
{
   CDeviceUtils::CopyLimitedString(name, g_DeviceNameDATTLStateDevice);
}


bool DATTLStateDevice::Busy()
{
   // We are busy if any of the underlying DA devices are busy, OR
   // the delay interval has not yet elapsed.

   for (unsigned int i = 0; i < numberOfDADevices_; ++i)
   {
      MM::SignalIO* da = static_cast<MM::SignalIO*>(GetDevice(daDeviceLabels_[i].c_str()));
      if (da && da->Busy())
         return true;
   }

   MM::MMTime delay(GetDelayMs() * 1000.0);
   if (GetCurrentMMTime() < lastChangeTime_ + delay)
      return true;

   return false;
}


unsigned long DATTLStateDevice::GetNumberOfPositions() const
{
   return 1 << numberOfDADevices_;
}


int DATTLStateDevice::OnNumberOfDADevices(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(static_cast<long>(numberOfDADevices_));
   }
   else if (eAct == MM::AfterSet)
   {
      long num;
      pProp->Get(num);
      numberOfDADevices_ = num;
   }
   return DEVICE_OK;
}


int DATTLStateDevice::OnDADevice(MM::PropertyBase* pProp, MM::ActionType eAct, long index)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(daDeviceLabels_[index].c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      std::string da;
      pProp->Get(da);
      daDeviceLabels_[index] = da;
   }
   return DEVICE_OK;
}


int DATTLStateDevice::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(mask_);
   }
   else if (eAct == MM::AfterSet)
   {
      bool gateOpen;
      GetGateOpen(gateOpen);
      pProp->Get(mask_);
      long mask = mask_;
      if (!gateOpen)
         GetProperty(MM::g_Keyword_Closed_Position, mask);

      for (unsigned int i = 0; i < numberOfDADevices_; ++i)
      {
         MM::SignalIO* da = static_cast<MM::SignalIO*>(GetDevice(daDeviceLabels_[i].c_str()));
         if (da)
         {
            int ret;
            if (invert_)
               ret = da->SetSignal((mask & (1 << i)) ? 0.0 : ttlVoltage_);
            else
               ret = da->SetSignal((mask & (1 << i)) ? ttlVoltage_ : 0.0);
            lastChangeTime_ = GetCurrentMMTime();
            if (ret != DEVICE_OK)
               return ret;
         }
      }
   }
   else if (eAct == MM::IsSequenceable)
   {
      bool allSequenceable = true;
      long maxSeqLen = LONG_MAX;
      for (unsigned int i = 0; i < numberOfDADevices_; ++i)
      {
         MM::SignalIO* da = static_cast<MM::SignalIO*>(GetDevice(daDeviceLabels_[i].c_str()));
         if (da)
         {
            bool sequenceable = false;
            int ret = da->IsDASequenceable(sequenceable);
            if (ret != DEVICE_OK)
               return ret;
            if (sequenceable)
            {
               long daMaxLen = 0;
               ret = da->GetDASequenceMaxLength(daMaxLen);
               if (ret != DEVICE_OK)
                  return ret;
               if (daMaxLen < maxSeqLen)
                  maxSeqLen = daMaxLen;
            }
            else
            {
               allSequenceable = false;
            }
         }
      }
      if (maxSeqLen == LONG_MAX) // No device?
         maxSeqLen = 0;
      pProp->SetSequenceable(maxSeqLen);
   }
   else if (eAct == MM::AfterLoadSequence)
   {
      std::vector<std::string> sequence = pProp->GetSequence();
      std::vector<long> values;
      for (std::vector<std::string>::const_iterator it = sequence.begin(),
         end = sequence.end(); it != end; ++it)
      {
         try
         {
            values.push_back(std::stol(*it));
         }
         catch (const std::invalid_argument&)
         {
            return DEVICE_ERR;
         }
         catch (const std::out_of_range&)
         {
            return DEVICE_ERR;
         }
      }

      for (unsigned int i = 0; i < numberOfDADevices_; ++i)
      {
         MM::SignalIO* da = static_cast<MM::SignalIO*>(GetDevice(daDeviceLabels_[i].c_str()));

         if (da)
         {
            int ret = da->ClearDASequence();
            if (ret != DEVICE_OK)
               return ret;
            for (std::vector<long>::const_iterator it = values.begin(),
               end = values.end(); it != end; ++it)
            {
               if (invert_)
                  ret = da->AddToDASequence((*it & (1 << i)) ? 0.0 : ttlVoltage_);
               else
                  ret = da->AddToDASequence((*it & (1 << i)) ? ttlVoltage_ : 0.0);
               if (ret != DEVICE_OK)
                  return ret;
            }
            ret = da->SendDASequence();
            if (ret != DEVICE_OK)
               return ret;
         }
      }
   }
   else if (eAct == MM::StartSequence)
   {
      for (unsigned int i = 0; i < numberOfDADevices_; ++i)
      {
         MM::SignalIO* da = static_cast<MM::SignalIO*>(GetDevice(daDeviceLabels_[i].c_str()));

         if (da)
         {
            int ret = da->StartDASequence();
            if (ret != DEVICE_OK)
               return ret;
         }
      }
   }
   else if (eAct == MM::StopSequence)
   {
      for (unsigned int i = 0; i < numberOfDADevices_; ++i)
      {
         MM::SignalIO* da = static_cast<MM::SignalIO*>(GetDevice(daDeviceLabels_[i].c_str()));

         if (da)
         {
            int ret = da->StopDASequence();
            if (ret != DEVICE_OK)
               return ret;
         }
      }
   }
   return DEVICE_OK;
}


int DATTLStateDevice::OnInvert(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      if (invert_)
         pProp->Set(g_invertedLogicString);
      else
         pProp->Set(g_normalLogicString);
   }
   else if (eAct == MM::AfterSet) {
      std::string invertString;
      pProp->Get(invertString);
      invert_ = (invertString != g_normalLogicString);
      // TODO: Set State property with cached value to let it invert the output.
      SetPosition(mask_);
   }
   return DEVICE_OK;
}

int DATTLStateDevice::OnTTLLevel(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      char buffer[8];
      snprintf(buffer, sizeof(buffer), "%.1f", ttlVoltage_);
      pProp->Set(buffer);
   }
   else if (eAct == MM::AfterSet)
   {
      std::string val;
      pProp->Get(val);
      ttlVoltage_ = atof(val.c_str());
      SetProperty(MM::g_Keyword_State, CDeviceUtils::ConvertToString(mask_));
   }
   return DEVICE_OK;
}
