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

extern const char* g_DeviceNameMultiDAStateDevice;



// Enables use of DAs as TTLs (i.e. as a state device)
MultiDAStateDevice::MultiDAStateDevice() :
   numberOfDADevices_(1),
   minVoltage_(0.0),
   maxVoltage_(5.0),
   initialized_(false),
   mask_(0L),
   lastChangeTime_(0, 0)
{
   CPropertyAction* pAct = new CPropertyAction(this,
      &MultiDAStateDevice::OnNumberOfDADevices);
   CreateIntegerProperty("NumberOfDADevices",
      static_cast<long>(numberOfDADevices_),
      false, pAct, true);
   for (int i = 1; i <= 8; ++i)
   {
      AddAllowedValue("NumberOfDADevices", std::to_string(i).c_str());
   }

   pAct = new CPropertyAction(this, &MultiDAStateDevice::OnMinVoltage);
   CreateFloatProperty("MinimumVoltage", minVoltage_, false, pAct, true);
   pAct = new CPropertyAction(this, &MultiDAStateDevice::OnMaxVoltage);
   CreateFloatProperty("MaximumVoltage", maxVoltage_, false, pAct, true);

   EnableDelay(true);
}


MultiDAStateDevice::~MultiDAStateDevice()
{
   Shutdown();
}


int MultiDAStateDevice::Initialize()
{
   if (initialized_)
      return DEVICE_OK;

   daDeviceLabels_.clear();
   voltages_.clear();
   for (unsigned int i = 0; i < numberOfDADevices_; ++i)
   {
      daDeviceLabels_.push_back("");
      voltages_.push_back(0.0);
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
         &MultiDAStateDevice::OnDADevice, i);
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

   if (minVoltage_ > maxVoltage_)
   {
      std::swap(minVoltage_, maxVoltage_);
   }

   CPropertyAction* pAct = new CPropertyAction(this, &MultiDAStateDevice::OnState);
   int ret = CreateIntegerProperty(MM::g_Keyword_State, 0, false, pAct);
   if (ret != DEVICE_OK)
      return ret;
   SetPropertyLimits(MM::g_Keyword_State, 0, numPos - 1);

   pAct = new CPropertyAction(this, &MultiDAStateDevice::OnLabel);
   ret = CreateStringProperty(MM::g_Keyword_Label, "0", false, pAct);
   if (ret != DEVICE_OK)
      return ret;

   ret = CreateIntegerProperty(MM::g_Keyword_Closed_Position, 0, false);
   if (ret != DEVICE_OK)
      return ret;

   for (unsigned int i = 0; i < numberOfDADevices_; ++i)
   {
      const std::string propName = "DADevice-" + std::to_string(i) + "-Voltage";
      CPropertyActionEx* pActEx = new CPropertyActionEx(this, &MultiDAStateDevice::OnVoltage, i);
      ret = CreateFloatProperty(propName.c_str(), voltages_[i], false, pActEx);
      if (ret != DEVICE_OK)
         return ret;

      ret = SetPropertyLimits(propName.c_str(), minVoltage_, maxVoltage_);
      if (ret != DEVICE_OK)
         return ret;
   }

   initialized_ = true;
   return DEVICE_OK;
}


int MultiDAStateDevice::Shutdown()
{
   if (!initialized_)
      return DEVICE_OK;

   daDeviceLabels_.clear();
   voltages_.clear();

   initialized_ = false;
   return DEVICE_OK;
}


void MultiDAStateDevice::GetName(char* name) const
{
   CDeviceUtils::CopyLimitedString(name, g_DeviceNameMultiDAStateDevice);
}


bool MultiDAStateDevice::Busy()
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


unsigned long MultiDAStateDevice::GetNumberOfPositions() const
{
   return 1 << numberOfDADevices_;
}


int MultiDAStateDevice::OnNumberOfDADevices(MM::PropertyBase* pProp, MM::ActionType eAct)
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


int MultiDAStateDevice::OnMinVoltage(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(minVoltage_);
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(minVoltage_);
   }
   return DEVICE_OK;
}


int MultiDAStateDevice::OnMaxVoltage(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(maxVoltage_);
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(maxVoltage_);
   }
   return DEVICE_OK;
}


int MultiDAStateDevice::OnDADevice(MM::PropertyBase* pProp, MM::ActionType eAct, long index)
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


int MultiDAStateDevice::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(mask_);
   }
   else if (eAct == MM::AfterSet)
   {
      bool gateOpen;
      GetGateOpen(gateOpen);
      long gatedMask = 0;
      pProp->Get(gatedMask);
      long mask = gatedMask;
      if (!gateOpen)
      {
         GetProperty(MM::g_Keyword_Closed_Position, mask);
      }

      for (unsigned int i = 0; i < numberOfDADevices_; ++i)
      {
         MM::SignalIO* da = static_cast<MM::SignalIO*>(GetDevice(daDeviceLabels_[i].c_str()));

         if (da)
         {
            double volts = voltages_[i];
            int ret = da->SetSignal((mask & (1 << i)) ? volts : 0.0);
            lastChangeTime_ = GetCurrentMMTime();
            if (ret != DEVICE_OK)
               return ret;
         }
      }
      mask_ = gatedMask;
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
               double volts = voltages_[i];
               ret = da->AddToDASequence(*it & (1 << i) ? volts : 0.0);
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


int MultiDAStateDevice::OnVoltage(MM::PropertyBase* pProp, MM::ActionType eAct, long i)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(voltages_[i]);
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(voltages_[i]);

      bool gateOpen;
      GetGateOpen(gateOpen);

      MM::SignalIO* da = static_cast<MM::SignalIO*>(GetDevice(daDeviceLabels_[i].c_str()));

      if (gateOpen && da && (mask_ & (1 << i)))
      {
         int ret = da->SetSignal(voltages_[i]);
         lastChangeTime_ = GetCurrentMMTime();
         if (ret != DEVICE_OK)
            return ret;
      }
   }
   return DEVICE_OK;
}
