// DESCRIPTION:   Drive multiple analog and digital outputs on NI DAQ
// AUTHOR:        Mark Tsuchida, 2015, Nico Stuurman 2022
// COPYRIGHT:     2015-2016, Open Imaging, Inc., 2022 Altos Labs
// LICENSE:       This library is free software; you can redistribute it and/or
//                modify it under the terms of the GNU Lesser General Public
//                License as published by the Free Software Foundation; either
//                version 2.1 of the License, or (at your option) any later
//                version.
//
//                This library is distributed in the hope that it will be
//                useful, but WITHOUT ANY WARRANTY; without even the implied
//                warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//                PURPOSE.  See the GNU Lesser General Public License for more
//                details.
//
//                You should have received a copy of the GNU Lesser General
//                Public License along with this library; if not, write to the
//                Free Software Foundation, Inc., 51 Franklin Street, Fifth
//                Floor, Boston, MA  02110-1301  USA

#include "NIDAQ.h"

#include "ModuleInterface.h"

//
// MultiAnalogOutPort
//

NIAnalogOutputPort::NIAnalogOutputPort(const std::string& port) :
   ErrorTranslator(21000, 21999, &NIAnalogOutputPort::SetErrorText),
   niPort_(port),
   initialized_(false),
   gateOpen_(true),
   gatedVoltage_(0.0),
   sequenceRunning_(false),
   minVolts_(0.0),
   maxVolts_(5.0),
   neverSequenceable_(false),
   task_(0)
{
   InitializeDefaultErrorMessages();
   SetErrorText(ERR_SEQUENCE_RUNNING, "A sequence is running on this port.  Please stop this sequence first.");
   SetErrorText(ERR_SEQUENCE_TOO_LONG, "Sequence is too long. Try increasing sequence length in the Hub device.");
   SetErrorText(ERR_SEQUENCE_ZERO_LENGTH, "Sequence has length zero.");
   SetErrorText(ERR_VOLTAGE_OUT_OF_RANGE, "Requested voltage is out of range");
   SetErrorText(ERR_NONUNIFORM_CHANNEL_VOLTAGE_RANGES, "Non-uniform channel voltage ranges.");
   SetErrorText(ERR_VOLTAGE_RANGE_EXCEEDS_DEVICE_LIMITS, "Requested voltage range exceeds the device's limits.");

   CPropertyAction* pAct = new CPropertyAction(this, &NIAnalogOutputPort::OnMinVolts);
   CreateFloatProperty("MinVolts", minVolts_, false, pAct, true);
   pAct = new CPropertyAction(this, &NIAnalogOutputPort::OnMaxVolts);
   CreateFloatProperty("MaxVolts", maxVolts_, false, pAct, true);

   pAct = new CPropertyAction(this, &NIAnalogOutputPort::OnSequenceable);
   CreateStringProperty("Sequencing", g_UseHubSetting, false, pAct, true);
   AddAllowedValue("Sequencing", g_UseHubSetting);
   AddAllowedValue("Sequencing", g_Never);
}


NIAnalogOutputPort::~NIAnalogOutputPort()
{
   Shutdown();
}


int NIAnalogOutputPort::Initialize()
{
   if (initialized_)
      return DEVICE_OK;

   // Check that the voltage range is allowed (since we cannot
   // enforce this when creating pre-init properties)
   double minVolts, maxVolts;
   int err = GetHub()->GetVoltageLimits(minVolts, maxVolts);
   if (err != DEVICE_OK)
      return TranslateHubError(err);
   LogMessage("Device voltage limits: " + boost::lexical_cast<std::string>(minVolts) +
      " to " + boost::lexical_cast<std::string>(maxVolts), true);
   if (minVolts_ < minVolts || maxVolts_ > maxVolts)
      return ERR_VOLTAGE_RANGE_EXCEEDS_DEVICE_LIMITS;

   CPropertyAction* pAct = new CPropertyAction(this, &NIAnalogOutputPort::OnVoltage);
   err = CreateFloatProperty("Voltage", gatedVoltage_, false, pAct);
   if (err != DEVICE_OK)
      return err;
   err = SetPropertyLimits("Voltage", minVolts_, maxVolts_);
   if (err != DEVICE_OK)
      return err;

   err = StartOnDemandTask(gateOpen_ ? gatedVoltage_ : 0.0);
   if (err != DEVICE_OK)
      return err;

   return DEVICE_OK;
}


int NIAnalogOutputPort::Shutdown()
{
   if (!initialized_)
      return DEVICE_OK;

   int err = StopTask();

   unsentSequence_.clear();
   sentSequence_.clear();

   initialized_ = false;
   return err;
}


void NIAnalogOutputPort::GetName(char* name) const
{
   CDeviceUtils::CopyLimitedString(name,
      (g_DeviceNameNIDAQAOPortPrefix + niPort_).c_str());
}


int NIAnalogOutputPort::SetGateOpen(bool open)
{
   if (open && !gateOpen_)
   {
      int err = StartOnDemandTask(gatedVoltage_);
      if (err != DEVICE_OK)
         return err;
   }
   else if (!open && gateOpen_)
   {
      int err = StartOnDemandTask(0.0);
      if (err != DEVICE_OK)
         return err;
   }

   gateOpen_ = open;
   return DEVICE_OK;
}


int NIAnalogOutputPort::GetGateOpen(bool& open)
{
   open = gateOpen_;
   return DEVICE_OK;
}


int NIAnalogOutputPort::SetSignal(double volts)
{
   if (volts < minVolts_ || volts > maxVolts_)
      return ERR_VOLTAGE_OUT_OF_RANGE;

   gatedVoltage_ = volts;
   if (gateOpen_)
   {
      int err = StartOnDemandTask(gatedVoltage_);
      if (err != DEVICE_OK)
         return err;
   }
   return DEVICE_OK;
}


int NIAnalogOutputPort::GetLimits(double& minVolts, double& maxVolts)
{
   minVolts = minVolts_;
   maxVolts = maxVolts_;
   return DEVICE_OK;
}


int NIAnalogOutputPort::IsDASequenceable(bool& isSequenceable) const
{
   if (neverSequenceable_)
      return false;

   // Translation from hub error code skipped (since this never fails)
   return GetHub()->IsSequencingEnabled(isSequenceable);
}


int NIAnalogOutputPort::GetDASequenceMaxLength(long& maxLength) const
{
   // Translation from hub error code skipped (since this never fails)
   return GetHub()->GetSequenceMaxLength(maxLength);
}


int NIAnalogOutputPort::StartDASequence()
{
   if (task_)
      StopTask();

   sequenceRunning_ = true;

   int err = GetHub()->StartAOSequenceForPort(niPort_, sentSequence_);
   if (err != DEVICE_OK)
      return TranslateHubError(err);

   return DEVICE_OK;
}


int NIAnalogOutputPort::StopDASequence()
{
   int err = GetHub()->StopAOSequenceForPort(niPort_);
   if (err != DEVICE_OK)
      return TranslateHubError(err);

   sequenceRunning_ = false;

   // Recover voltage from before sequencing started, so that we
   // are back in sync
   err = StartOnDemandTask(gateOpen_ ? gatedVoltage_ : 0.0);
   if (err != DEVICE_OK)
      return err;

   return DEVICE_OK;
}


int NIAnalogOutputPort::ClearDASequence()
{
   unsentSequence_.clear();
   return DEVICE_OK;
}


int NIAnalogOutputPort::AddToDASequence(double voltage)
{
   if (voltage < minVolts_ || voltage > maxVolts_)
      return ERR_VOLTAGE_OUT_OF_RANGE;

   unsentSequence_.push_back(voltage);
   return DEVICE_OK;
}


int NIAnalogOutputPort::SendDASequence()
{
   if (sequenceRunning_)
      return ERR_SEQUENCE_RUNNING;

   sentSequence_ = unsentSequence_;
   // We don't actually "write" the sequence here, because writing
   // needs to take place once the correct task has been set up for
   // all of the AO channels.
   return DEVICE_OK;
}


int NIAnalogOutputPort::OnMinVolts(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(minVolts_);
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(minVolts_);
   }
   return DEVICE_OK;
}


int NIAnalogOutputPort::OnMaxVolts(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(maxVolts_);
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(maxVolts_);
   }
   return DEVICE_OK;
}


int NIAnalogOutputPort::OnSequenceable(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(neverSequenceable_ ? g_Never : g_UseHubSetting);
   }
   else if (eAct == MM::AfterSet)
   {
      std::string s;
      pProp->Get(s);
      neverSequenceable_ = (s == g_Never);
   }
   return DEVICE_OK;
}


int NIAnalogOutputPort::OnVoltage(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(gatedVoltage_);
   }
   else if (eAct == MM::AfterSet)
   {
      double voltage;
      pProp->Get(voltage);
      int err = SetSignal(voltage);
      if (err != DEVICE_OK)
         return err;
   }
   return DEVICE_OK;
}


int NIAnalogOutputPort::TranslateHubError(int err)
{
   if (err == DEVICE_OK)
      return DEVICE_OK;
   char buf[MM::MaxStrLength];
   if (GetHub()->GetErrorText(err, buf))
      return NewErrorCode(buf);
   return NewErrorCode("Unknown hub error");
}


int NIAnalogOutputPort::StartOnDemandTask(double voltage)
{
   if (sequenceRunning_)
      return ERR_SEQUENCE_RUNNING;

   if (task_)
   {
      int err = StopTask();
      if (err != DEVICE_OK)
         return err;
   }

   LogMessage("Starting on-demand task", true);

   int32 nierr = DAQmxCreateTask(NULL, &task_);
   if (nierr != 0)
   {
      LogMessage(GetNIDetailedErrorForMostRecentCall().c_str());
      return TranslateNIError(nierr);
   }
   LogMessage("Created task", true);

   nierr = DAQmxCreateAOVoltageChan(task_, niPort_.c_str(), NULL, minVolts_, maxVolts_,
      DAQmx_Val_Volts, NULL);
   if (nierr != 0)
   {
      LogMessage(GetNIDetailedErrorForMostRecentCall().c_str());
      goto error;
   }
   LogMessage("Created AO voltage channel", true);

   float64 samples[1];
   samples[0] = voltage;
   int32 numWritten = 0;
   nierr = DAQmxWriteAnalogF64(task_, 1,
      true, DAQmx_Val_WaitInfinitely, DAQmx_Val_GroupByChannel,
      samples, &numWritten, NULL);
   if (nierr != 0)
   {
      LogMessage(GetNIDetailedErrorForMostRecentCall().c_str());
      goto error;
   }
   if (numWritten != 1)
   {
      LogMessage("Failed to write voltage");
      // This is presumably unlikely; no error code here
      goto error;
   }
   LogMessage(("Wrote voltage with task autostart: " +
      boost::lexical_cast<std::string>(voltage) + " V").c_str(), true);

   return DEVICE_OK;

error:
   DAQmxClearTask(task_);
   task_ = 0;
   int err;
   if (nierr != 0)
   {
      LogMessage("Failed; task cleared");
      err = TranslateNIError(nierr);
   }
   else
   {
      err = DEVICE_ERR;
   }
   return err;
}


int NIAnalogOutputPort::StopTask()
{
   if (!task_)
      return DEVICE_OK;

   int32 nierr = DAQmxClearTask(task_);
   if (nierr != 0)
      return TranslateNIError(nierr);
   task_ = 0;
   LogMessage("Stopped task", true);

   return DEVICE_OK;
}