// DESCRIPTION:   Drive multiple analog and digital outputs on NI DAQ
//                Based on NI-MultiAnalog device adapter by Mark Tsuchida
// AUTHOR:        Mark Tsuchida, 2015, Nico Stuurman, 2022
// COPYRIGHT:     2015-2016, Open Imaging, Inc., Altos Labs 2022
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

#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/math/common_factor_rt.hpp>
#include <boost/scoped_array.hpp>


const char* g_DeviceNameNIDAQHub = "NIDAQHub";
const char* g_DeviceNameNIDAQAOPortPrefix = "NIDAQAO-";
const char* g_DeviceNameNIDAQDOPortPrefix = "NIDAQDO-";

const char* g_On = "On";
const char* g_Off = "Off";
const char* g_Low = "Low";
const char* g_High = "High";

const char* g_Never = "Never";
const char* g_UseHubSetting = "Use hub setting";

const int ERR_SEQUENCE_RUNNING = 2001;
const int ERR_SEQUENCE_TOO_LONG = 2002;
const int ERR_SEQUENCE_ZERO_LENGTH = 2003;
const int ERR_VOLTAGE_OUT_OF_RANGE = 2004;
const int ERR_NONUNIFORM_CHANNEL_VOLTAGE_RANGES = 2005;
const int ERR_VOLTAGE_RANGE_EXCEEDS_DEVICE_LIMITS = 2006;
const int ERR_UNKNOWN_PINS_PER_PORT = 2007;
const int ERR_INVALID_REQUEST = 2008;



MODULE_API void InitializeModuleData()
{
   RegisterDevice(g_DeviceNameNIDAQHub, MM::HubDevice, "NIDAQ analog and digital output");
}


MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
   if (deviceName == 0)
      return 0;

   if (strcmp(deviceName, g_DeviceNameNIDAQHub) == 0)
   {
      return new NIDAQHub;
   }
   else if (std::string(deviceName).
      substr(0, strlen(g_DeviceNameNIDAQAOPortPrefix)) ==
      g_DeviceNameNIDAQAOPortPrefix)
   {
      return new NIAnalogOutputPort(std::string(deviceName).
         substr(strlen(g_DeviceNameNIDAQAOPortPrefix)));
   }
   else if (std::string(deviceName).substr(0, strlen(g_DeviceNameNIDAQDOPortPrefix)) ==
       g_DeviceNameNIDAQDOPortPrefix)
   {
      return new DigitalOutputPort(std::string(deviceName).
         substr(strlen(g_DeviceNameNIDAQDOPortPrefix)));
   }

   return 0;
}


MODULE_API void DeleteDevice(MM::Device* pDevice)
{
   delete pDevice;
}



NIDAQHub::NIDAQHub () :
   ErrorTranslator(20000, 20999, &NIDAQHub::SetErrorText),
   initialized_(false),
   maxSequenceLength_(1024),
   sequencingEnabled_(false),
   sequenceRunning_(false),
   minVolts_(0.0),
   maxVolts_(5.0),
   sampleRateHz_(10000.0),
   aoTask_(0),
   doTask_(0),
   doHub8_(0),
   doHub16_(0),
   doHub32_(0)
{
   // discover devices available on this computer and list them here
   std::string defaultDeviceName = "";
   int32 stringLength = DAQmxGetSysDevNames(NULL, 0);
   std::vector<std::string> result;
   if (stringLength > 0)
   {
      char* deviceNames = new char[stringLength];
      int32 nierr = DAQmxGetSysDevNames(deviceNames, stringLength);
      if (nierr == 0)
      {
         LogMessage(deviceNames, false);
         boost::split(result, deviceNames, boost::is_any_of(", "),
            boost::token_compress_on);
         defaultDeviceName = result[0];
      }
      else
      {
         LogMessage("No NIDAQ devicename found, false");
      }
      delete[] deviceNames;
   }

  
   CPropertyAction* pAct = new CPropertyAction(this, &NIDAQHub::OnDevice);
   int err = CreateStringProperty("Device", defaultDeviceName.c_str(), false, pAct, true);
   if (result.size() > 0)
   {
      for (std::string device : result)
      {
         AddAllowedValue("Device", device.c_str());
      }
   }

   pAct = new CPropertyAction(this, &NIDAQHub::OnMaxSequenceLength);
   err = CreateIntegerProperty("MaxSequenceLength",
      static_cast<long>(maxSequenceLength_), false, pAct, true);
}


NIDAQHub::~NIDAQHub()
{
   Shutdown();
}


int NIDAQHub::Initialize()
{
   if (initialized_)
      return DEVICE_OK;

   if (!GetParentHub())
      return DEVICE_ERR;


   // Dynamically determine name of ChangeDetectionEvent for this device
   niChangeDetection_ = "/" + niDeviceName_ + "/ChangeDetectionEvent";
   niSampleClock_ = "/" + niDeviceName_ + "/do/SampleClock";

   // Determine the possible voltage range
   int err = GetVoltageRangeForDevice(niDeviceName_, minVolts_, maxVolts_);
   if (err != DEVICE_OK)
      return err;

   CPropertyAction* pAct = new CPropertyAction(this, &NIDAQHub::OnSequencingEnabled);
   err = CreateStringProperty("Sequence", sequencingEnabled_ ? g_On : g_Off, false, pAct);
   if (err != DEVICE_OK)
      return err;
   AddAllowedValue("Sequence", g_On);
   AddAllowedValue("Sequence", g_Off);

   std::vector<std::string> doPorts = GetDigitalPortsForDevice(niDeviceName_);
   if (doPorts.size() > 0)
   {
      // we could check if we actually have ports of these kinds, but the cost of instantiating all is low
      doHub8_ = new NIDAQDOHub<uInt8>(this);
      doHub16_ = new NIDAQDOHub<uInt16>(this);
      doHub32_ = new NIDAQDOHub<uInt32>(this);
   }

   std::vector<std::string> triggerPorts = GetAOTriggerTerminalsForDevice(niDeviceName_);
   if (!triggerPorts.empty())
   {
      niTriggerPort_ = triggerPorts[0];
      pAct = new CPropertyAction(this, &NIDAQHub::OnTriggerInputPort);
      err = CreateStringProperty("AOTriggerInputPort", niTriggerPort_.c_str(), false, pAct);
      if (err != DEVICE_OK)
         return err;
      for (std::vector<std::string>::const_iterator it = triggerPorts.begin(),
            end = triggerPorts.end();
            it != end; ++it)
      {
         AddAllowedValue("AOTriggerInputPort", it->c_str());
      }

      pAct = new CPropertyAction(this, &NIDAQHub::OnSampleRate);
      err = CreateFloatProperty("SampleRateHz", sampleRateHz_, false, pAct);
      if (err != DEVICE_OK)
         return err;
   }

   err = SwitchTriggerPortToReadMode();
   if (err != DEVICE_OK)
   {
      LogMessage("Failed to switch device " + niDeviceName_ + ", port " + niTriggerPort_ + " to read mode.");
      // do not return an error to allow the user to switch the triggerport to something that works
   }

   initialized_ = true;
   return DEVICE_OK;
}


int NIDAQHub::Shutdown()
{
   if (!initialized_)
      return DEVICE_OK;

   int err = StopTask(aoTask_);

   physicalAOChannels_.clear();
   aoChannelSequences_.clear();

   if (doHub8_ != 0)
      delete doHub8_;
   else if (doHub16_ != 0)
      delete doHub16_;
   else if (doHub32_ != 0)
      delete  doHub32_;

   initialized_ = false;
   return err;
}


void NIDAQHub::GetName(char* name) const
{
   CDeviceUtils::CopyLimitedString(name, g_DeviceNameNIDAQHub);
}


int NIDAQHub::DetectInstalledDevices()
{
   std::vector<std::string> aoPorts =
      GetAnalogPortsForDevice(niDeviceName_);

   for (std::vector<std::string>::const_iterator it = aoPorts.begin(), end = aoPorts.end();
      it != end; ++it)
   {
      MM::Device* pDevice =
         ::CreateDevice((g_DeviceNameNIDAQAOPortPrefix + *it).c_str());
      if (pDevice)
      {
         AddInstalledDevice(pDevice);
      }
   }

   std::vector<std::string> doPorts = GetDigitalPortsForDevice(niDeviceName_);

   for (std::vector<std::string>::const_iterator it = doPorts.begin(), end = doPorts.end();
       it != end; ++it)
   {
       MM::Device* pDevice =
           ::CreateDevice((g_DeviceNameNIDAQDOPortPrefix + *it).c_str());
       if (pDevice)
       {
           AddInstalledDevice(pDevice);
       }
   }

   return DEVICE_OK;
}


int NIDAQHub::GetVoltageLimits(double& minVolts, double& maxVolts)
{
   minVolts = minVolts_;
   maxVolts = maxVolts_;
   return DEVICE_OK;
}


int NIDAQHub::StartAOSequenceForPort(const std::string& port,
   const std::vector<double> sequence)
{
   int err = StopTask(aoTask_);
   if (err != DEVICE_OK)
      return err;

   err = AddAOPortToSequencing(port, sequence);
   if (err != DEVICE_OK)
      return err;

   err = StartAOSequencingTask();
   if (err != DEVICE_OK)
      return err;
   // We don't restart the task without this port on failure.
   // There is little point in doing so.

   return DEVICE_OK;
}


int NIDAQHub::StopAOSequenceForPort(const std::string& port)
{
   int err = StopTask(aoTask_);
   if (err != DEVICE_OK)
      return err;
   sequenceRunning_ = false;
   RemoveAOPortFromSequencing(port);
   // We do not restart sequencing for the remaining ports,
   // since it is meaningless (we can't preserve their state).
   
   // Make sure that the input trigger pin has a high impedance (i.e. does not 
   // somehow become an output pin
   return SwitchTriggerPortToReadMode();
}


int NIDAQHub::SwitchTriggerPortToReadMode()
{
   int err = StopTask(aoTask_);
   if (err != DEVICE_OK)
      return err;

   int32 nierr = DAQmxCreateTask((niDeviceName_ + "TriggerPinReadTask").c_str(), &aoTask_);
   if (nierr != 0)
      return TranslateNIError(nierr);
   LogMessage("Created Trigger pin read task", true);
   nierr = DAQmxCreateDIChan(aoTask_, niTriggerPort_.c_str(), "tIn", DAQmx_Val_ChanForAllLines);
   if (nierr != 0)
      return TranslateNIError(nierr);
   nierr = DAQmxStartTask(aoTask_);
   if (nierr != 0)
      return TranslateNIError(nierr);

   return DEVICE_OK;
}


int NIDAQHub::IsSequencingEnabled(bool& flag) const
{
   flag = sequencingEnabled_;
   return DEVICE_OK;
}


int NIDAQHub::GetSequenceMaxLength(long& maxLength) const
{
   maxLength = static_cast<long>(maxSequenceLength_);
   return DEVICE_OK;
}


int NIDAQHub::AddAOPortToSequencing(const std::string& port,
   const std::vector<double> sequence)
{
   if (sequence.size() > maxSequenceLength_)
      return ERR_SEQUENCE_TOO_LONG;

   RemoveAOPortFromSequencing(port);

   physicalAOChannels_.push_back(port);
   aoChannelSequences_.push_back(sequence);
   return DEVICE_OK;
}


void NIDAQHub::RemoveAOPortFromSequencing(const std::string& port)
{
   // We assume a given port appears at most once in physicalChannels_
   size_t n = physicalAOChannels_.size();
   for (size_t i = 0; i < n; ++i)
   {
      if (physicalAOChannels_[i] == port) {
         physicalAOChannels_.erase(physicalAOChannels_.begin() + i);
         aoChannelSequences_.erase(aoChannelSequences_.begin() + i);
         break;
      }
   }
}


int NIDAQHub::GetVoltageRangeForDevice(
   const std::string& device, double& minVolts, double& maxVolts)
{
   const int MAX_RANGES = 64;
   float64 ranges[2 * MAX_RANGES];
   for (int i = 0; i < MAX_RANGES; ++i)
   {
      ranges[2 * i] = 0.0;
      ranges[2 * i + 1] = 0.0;
   }

   int32 nierr = DAQmxGetDevAOVoltageRngs(device.c_str(), ranges,
      sizeof(ranges) / sizeof(float64));
   if (nierr != 0)
   {
      LogMessage(GetNIDetailedErrorForMostRecentCall().c_str());
      return TranslateNIError(nierr);
   }

   minVolts = ranges[0];
   maxVolts = ranges[1];
   for (int i = 0; i < MAX_RANGES; ++i)
   {
      if (ranges[2 * i] == 0.0 && ranges[2 * i + 1] == 0.0)
         break;
      LogMessage(("Possible voltage range " +
         boost::lexical_cast<std::string>(ranges[2 * i]) + " V to " +
         boost::lexical_cast<std::string>(ranges[2 * i + 1]) + " V").c_str(),
         true);
      if (ranges[2 * i + 1] > maxVolts)
      {
         minVolts = ranges[2 * i];
         maxVolts = ranges[2 * i + 1];
      }
   }
      LogMessage(("Selected voltage range " +
         boost::lexical_cast<std::string>(minVolts) + " V to " +
         boost::lexical_cast<std::string>(maxVolts) + " V").c_str(),
         true);

   return DEVICE_OK;
}


std::vector<std::string>
NIDAQHub::GetAOTriggerTerminalsForDevice(const std::string& device)
{
   std::vector<std::string> result;
      
   char ports[4096];
   int32 nierr = DAQmxGetDevTerminals(device.c_str(), ports, sizeof(ports));
   if (nierr == 0)
   {
      std::vector<std::string> terminals;
      boost::split(terminals, ports, boost::is_any_of(", "),
         boost::token_compress_on);

      // Only return the PFI terminals.
      for (std::vector<std::string>::const_iterator
         it = terminals.begin(), end = terminals.end();
         it != end; ++it)
      {
         if (it->find("PFI") != std::string::npos)
         {
            result.push_back(*it);
         }
      }
   }
   else
   {
      LogMessage(GetNIDetailedErrorForMostRecentCall().c_str());
      LogMessage("Cannot get list of trigger ports");
   }

   return result;
}


std::vector<std::string>
NIDAQHub::GetAnalogPortsForDevice(const std::string& device)
{
   std::vector<std::string> result;

   char ports[4096];
   int32 nierr = DAQmxGetDevAOPhysicalChans(device.c_str(), ports, sizeof(ports));
   if (nierr == 0)
   {
      boost::split(result, ports, boost::is_any_of(", "),
         boost::token_compress_on);
   }
   else
   {
      LogMessage(GetNIDetailedErrorForMostRecentCall().c_str());
      LogMessage("Cannot get list of analog ports");
   }

   return result;
}

std::vector<std::string>
NIDAQHub::GetDigitalPortsForDevice(const std::string& device)
{
    std::vector<std::string> result;

    char ports[4096];
    int32 nierr = DAQmxGetDevDOPorts(device.c_str(), ports, sizeof(ports));
    if (nierr == 0)
    {
        boost::split(result, ports, boost::is_any_of(", "),
            boost::token_compress_on);
    }
    else
    {
        LogMessage(GetNIDetailedErrorForMostRecentCall().c_str());
        LogMessage("Cannot get list of digital ports");
    }

    return result;
}


std::string NIDAQHub::GetPhysicalChannelListForSequencing(std::vector<std::string> channels) const
{
   std::string ret;
   for (std::vector<std::string>::const_iterator begin = channels.begin(),
      end = channels.end(), it = begin;
      it != end; ++it)
   {
      if (it != begin)
         ret += ", ";
      ret += *it;
   }
   return ret;
}


template<typename T>
inline int NIDAQHub::GetLCMSamplesPerChannel(size_t& seqLen, std::vector<std::vector<T>> channelSequences) const
{
   // Use an arbitrary but reasonable limit to prevent
   // overflow or excessive memory consumption.
   const uint64_t factorLimit = 2 << 14;

   uint64_t len = 1;
   for (unsigned int i = 0; i < channelSequences.size(); ++i)
   {
      uint64_t channelSeqLen = channelSequences[i].size();
      if (channelSeqLen > factorLimit)
      {
         return ERR_SEQUENCE_TOO_LONG;
      }
      if (channelSeqLen == 0)
      {
         return ERR_SEQUENCE_ZERO_LENGTH;
      }
      len = boost::math::lcm(len, channelSeqLen);
      if (len > factorLimit)
      {
         return ERR_SEQUENCE_TOO_LONG;
      }
   }
   seqLen = (size_t) len;
   return DEVICE_OK;
}


template<typename T>
void NIDAQHub::GetLCMSequence(T* buffer, std::vector<std::vector<T>> sequences) const
{
   size_t seqLen;
   if (GetLCMSamplesPerChannel(seqLen, sequences) != DEVICE_OK)
      return;

   for (unsigned int i = 0; i < sequences.size(); ++i)
   {
      size_t chanOffset = seqLen * i;
      size_t chanSeqLen = sequences[i].size();
      for (unsigned int j = 0; j < seqLen; ++j)
      {
         buffer[chanOffset + j] =
            sequences[i][j % chanSeqLen];
      }
   }
}


/**
* This task will start sequencing of all analog outputs that were previously added 
* using AddAOPortToSequencing
* The triggerinputport has to be supported by the device.  
* Specifically, a trigger input terminal is of the form /Dev1/PFI0, where 
* there is a preceding slash.  Terminals that are part of an output port
* (such as Dev1/port0/line7) do not work.
* Uses DAQmxCfgSampClkTiming to transition to the next state for each 
* anlog output at each consecutive rising flank of the trigger input terminal.
* 
*/
int NIDAQHub::StartAOSequencingTask()
{
   if (aoTask_)
   {
      int err = StopTask(aoTask_);
      if (err != DEVICE_OK)
         return err;
   }

   LogMessage("Starting sequencing task", true);

   boost::scoped_array<float64> samples;

   size_t numChans = physicalAOChannels_.size();
   size_t samplesPerChan;
   int err = GetLCMSamplesPerChannel(samplesPerChan, aoChannelSequences_);
   if (err != DEVICE_OK)
      return err;

   LogMessage(boost::lexical_cast<std::string>(numChans) + " channels", true);
   LogMessage("LCM sequence length = " +
      boost::lexical_cast<std::string>(samplesPerChan), true);

   int32 nierr = DAQmxCreateTask("AOSeqTask", &aoTask_);
   if (nierr != 0)
   {
      LogMessage(GetNIDetailedErrorForMostRecentCall().c_str());
      return nierr;
   }
   LogMessage("Created task", true);

   const std::string chanList = GetPhysicalChannelListForSequencing(physicalAOChannels_);
   nierr = DAQmxCreateAOVoltageChan(aoTask_, chanList.c_str(),
      "AOSeqChan", minVolts_, maxVolts_, DAQmx_Val_Volts,
      NULL);
   if (nierr != 0)
   {
      LogMessage(GetNIDetailedErrorForMostRecentCall().c_str());
      goto error;
   }
   LogMessage(("Created AO voltage channel for: " + chanList).c_str(), true);

   nierr = DAQmxCfgSampClkTiming(aoTask_, niTriggerPort_.c_str(),
      sampleRateHz_, DAQmx_Val_Rising,
      DAQmx_Val_ContSamps, samplesPerChan);
   if (nierr != 0)
   {
      LogMessage(GetNIDetailedErrorForMostRecentCall().c_str());
      goto error;
   }
   LogMessage("Configured sample clock timing to use " + niTriggerPort_, true);

   samples.reset(new float64[samplesPerChan * numChans]);
   GetLCMSequence(samples.get(), aoChannelSequences_);

   int32 numWritten = 0;
   nierr = DAQmxWriteAnalogF64(aoTask_, static_cast<int32>(samplesPerChan),
      false, DAQmx_Val_WaitInfinitely, DAQmx_Val_GroupByChannel,
      samples.get(), &numWritten, NULL);
   if (nierr != 0)
   {
      LogMessage(GetNIDetailedErrorForMostRecentCall().c_str());
      goto error;
   }
   if (numWritten != static_cast<int32>(samplesPerChan))
   {
      LogMessage("Failed to write complete sequence");
      // This is presumably unlikely; no error code here
      goto error;
   }
   LogMessage("Wrote samples", true);

   nierr = DAQmxStartTask(aoTask_);
   if (nierr != 0)
   {
      LogMessage(GetNIDetailedErrorForMostRecentCall().c_str());
      goto error;
   }
   LogMessage("Started task", true);

   sequenceRunning_ = true;

   return DEVICE_OK;

error:
   DAQmxClearTask(aoTask_);
   aoTask_ = 0;
   err;
   if (nierr != 0)
   {
      LogMessage("Failed; task cleared");
      err = TranslateNIError(nierr);
   }
   else
   {
      err = DEVICE_ERR;
   }

   sequenceRunning_ = false;
   return err;
}


int NIDAQHub::StartDOBlankingAndOrSequence(const std::string& port, const uInt32 portWidth, const bool blankingOn, 
            const bool sequenceOn, const long& pos, const bool blankingDirection, const std::string triggerPort)
{
   if (portWidth == 8)
      return doHub8_->StartDOBlankingAndOrSequence(port, blankingOn, sequenceOn, pos, blankingDirection, triggerPort);
   else if (portWidth == 16)
      return doHub16_->StartDOBlankingAndOrSequence(port, blankingOn, sequenceOn, pos, blankingDirection, triggerPort);
   else if (portWidth == 32)
      return doHub32_->StartDOBlankingAndOrSequence(port, blankingOn, sequenceOn, pos, blankingDirection, triggerPort);

   return ERR_UNKNOWN_PINS_PER_PORT;
}


int NIDAQHub::StopDOBlankingAndSequence(const uInt32 portWidth)
{
   if (portWidth == 8)
      return doHub8_->StopDOBlankingAndSequence();
   else if (portWidth == 16)
      return doHub16_->StopDOBlankingAndSequence();
   else if (portWidth == 32)
      return doHub32_->StopDOBlankingAndSequence();

   return ERR_UNKNOWN_PINS_PER_PORT;
}


int NIDAQHub::SetDOPortState(std::string port, uInt32 portWidth, long state)
{
   if (doTask_)
   {
      int err = StopTask(doTask_);
      if (err != DEVICE_OK)
         return err;
   }

   LogMessage("Starting on-demand task", true);

   int32 nierr = DAQmxCreateTask(NULL, &doTask_);
   if (nierr != 0)
   {
      LogMessage(GetNIDetailedErrorForMostRecentCall().c_str());
      return TranslateNIError(nierr);
   }
   LogMessage("Created task", true);

   nierr = DAQmxCreateDOChan(doTask_, port.c_str(), NULL, DAQmx_Val_ChanForAllLines);
   if (nierr != 0)
   {
      LogMessage(GetNIDetailedErrorForMostRecentCall().c_str());
      goto error;
   }
   LogMessage("Created DO channel", true);

   int32 numWritten = 0;
   if (portWidth == 8)
   {
      uInt8 samples[1];
      samples[0] = (uInt8)state;
      nierr = DAQmxWriteDigitalU8(doTask_, 1,
         true, DAQmx_Val_WaitInfinitely, DAQmx_Val_GroupByChannel,
         samples, &numWritten, NULL);

   }
   else if (portWidth == 16)
   {
      uInt16 samples[1];
      samples[0] = (uInt16)state;
      nierr = DAQmxWriteDigitalU16(doTask_, 1,
         true, DAQmx_Val_WaitInfinitely, DAQmx_Val_GroupByChannel,
         samples, &numWritten, NULL);
   }
   else if (portWidth == 32)
   {
      uInt32 samples[1];
      samples[0] = (uInt32)state;
      nierr = DAQmxWriteDigitalU32(doTask_, 1,
         true, DAQmx_Val_WaitInfinitely, DAQmx_Val_GroupByChannel,
         samples, &numWritten, NULL);
   }
   else
   {
      LogMessage(("Found invalid number of pins per port: " +
         boost::lexical_cast<std::string>(portWidth)).c_str(), true);
      goto error;
   }
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
   LogMessage(("Wrote Digital out with task autostart: " +
      boost::lexical_cast<std::string>(state)).c_str(), true);

   return DEVICE_OK;

error:
   DAQmxClearTask(doTask_);
   doTask_ = 0;
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


 int NIDAQHub::StopTask(TaskHandle &task)
{
   if (!task)
      return DEVICE_OK;

   int32 nierr = DAQmxClearTask(task);
   if (nierr != 0)
      return TranslateNIError(nierr);
   task = 0;
   LogMessage("Stopped task", true);

   return DEVICE_OK;
}


int NIDAQHub::OnDevice(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(niDeviceName_.c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      std::string deviceName;
      pProp->Get(deviceName);
      niDeviceName_ = deviceName;
   }
   return DEVICE_OK;
}


int NIDAQHub::OnMaxSequenceLength(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(static_cast<long>(maxSequenceLength_));
   }
   else if (eAct == MM::AfterSet)
   {
      long maxLength;
      pProp->Get(maxLength);
      if (maxLength < 0)
      {
         maxLength = 0;
         pProp->Set(maxLength);
      }
      maxSequenceLength_ = maxLength;
   }
   return DEVICE_OK;
}


int NIDAQHub::OnSequencingEnabled(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(sequencingEnabled_ ? g_On : g_Off);
   }
   else if (eAct == MM::AfterSet)
   {
      std::string sw;
      pProp->Get(sw);
      sequencingEnabled_ = (sw == g_On);
   }
   return DEVICE_OK;
}


int NIDAQHub::OnTriggerInputPort(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(niTriggerPort_.c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      if (sequenceRunning_)
         return ERR_SEQUENCE_RUNNING;

      std::string port;
      pProp->Get(port);
      niTriggerPort_ = port;
      return SwitchTriggerPortToReadMode();
   }
   return DEVICE_OK;
}


int NIDAQHub::OnSampleRate(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(sampleRateHz_);
   }
   else if (eAct == MM::AfterSet)
   {
      if (sequenceRunning_)
         return ERR_SEQUENCE_RUNNING;

      double rateHz;
      pProp->Get(rateHz);
      if (rateHz <= 0.0)
      {
         rateHz = 1.0;
         pProp->Set(rateHz);
      }
      sampleRateHz_ = rateHz;
   }
   return DEVICE_OK;
}


//
// NIDAQDOHub
//


template<typename Tuint>
NIDAQDOHub<Tuint>::NIDAQDOHub(NIDAQHub* hub) : diTask_(0), doTask_(0), hub_(hub)
{
   if (typeid(Tuint) == typeid(uInt8))
      portWidth_ = 8;
   else if (typeid(Tuint) == typeid(uInt8))
      portWidth_ = 16;
   else if (typeid(Tuint) == typeid(uInt32))
      portWidth_ = 32;
   else
      portWidth_ = 0;
}


template<typename Tuint>
 NIDAQDOHub<Tuint>::~NIDAQDOHub<Tuint>()
{
   hub_->StopTask(doTask_);

   physicalDOChannels_.clear();
   doChannelSequences_.clear();
}


template<typename Tuint>
int NIDAQDOHub<Tuint>::AddDOPortToSequencing(const std::string& port, const std::vector<Tuint> sequence)
{
   long maxSequenceLength;
   hub_->GetSequenceMaxLength(maxSequenceLength);
   if (sequence.size() > maxSequenceLength)
      return ERR_SEQUENCE_TOO_LONG;

   RemoveDOPortFromSequencing(port);

   physicalDOChannels_.push_back(port);
   doChannelSequences_.push_back(sequence);
   return DEVICE_OK;
}


template<typename Tuint>
inline void NIDAQDOHub<Tuint>::RemoveDOPortFromSequencing(const std::string & port)
{
   size_t n = physicalDOChannels_.size();
   for (size_t i = 0; i < n; ++i)
   {
      if (physicalDOChannels_[i] == port) {
         physicalDOChannels_.erase(physicalDOChannels_.begin() + i);
         doChannelSequences_.erase(doChannelSequences_.begin() + i);
      }
   }
}


template<class Tuint>
int NIDAQDOHub<Tuint>::StartDOBlankingAndOrSequence(const std::string& port, const bool blankingOn, const bool sequenceOn, 
                                       const long& pos, const bool blankOnLow, const std::string triggerPort)
{
   if (!blankingOn && !sequenceOn)
   {
      return ERR_INVALID_REQUEST;
   }
   // First read the state of the triggerport, since we will only get changes of triggerPort, not
   // its actual state.
   bool triggerPinState;
   int err = GetPinState(triggerPort, triggerPinState);
   if (err != DEVICE_OK)
      return err;

   //Set initial state based on blankOnLow and triggerPort state
   if (blankOnLow ^ triggerPinState) 
      err = hub_->SetDOPortState(port, portWidth_, 0);
   else
      err = hub_->SetDOPortState(port, portWidth_, pos);
   if (err != DEVICE_OK)
      return err;
      
   err = hub_->StopTask(diTask_);
   if (err != DEVICE_OK)
      return err;

   int32 nierr = DAQmxCreateTask("DIChangeTask", &diTask_);
   if (nierr != 0)
   {
      return hub_->TranslateNIError(nierr);;
   }
   hub_->LogMessage("Created DI task", true);

   int32 number = 2;
   std::vector<Tuint> doSequence_;
   if (sequenceOn)
   {
      int i = 0;
      for (std::string pChan : physicalDOChannels_)
      {
         if (port == pChan)
         {
            doSequence_ = doChannelSequences_[i];
            number =  2 * (int32) doSequence_.size();
         }
         i++;
      }
   }

   nierr = DAQmxCreateDIChan(diTask_, triggerPort.c_str(), "DIBlankChan", DAQmx_Val_ChanForAllLines);
   if (nierr != 0)
   {
      return HandleTaskError(nierr);
   }
   hub_->LogMessage("Created DI channel for: " + triggerPort, true);

   // Note, if triggerPort is not part of the port, we'll likely start seeing errors here
   // This needs to be in the documentation
   nierr = DAQmxCfgChangeDetectionTiming(diTask_, triggerPort.c_str(),
          triggerPort.c_str(), DAQmx_Val_ContSamps, number);
   if (nierr != 0)
   {
      return HandleTaskError(nierr);
   }
   hub_->LogMessage("Configured change detection timing to use " + triggerPort, true);

   // this is only here to monitor the ChangeDetectionEvent, delete after debugging
   
   nierr = DAQmxExportSignal(diTask_, DAQmx_Val_ChangeDetectionEvent, hub_->niSampleClock_.c_str());
   if (nierr != 0)
   {
      return HandleTaskError(nierr);
   }
   hub_->LogMessage("Routed change detection timing to  " + hub_->niChangeDetection_, true);

   nierr = DAQmxStartTask(diTask_);
   if (nierr != 0)
   {
      return HandleTaskError(nierr);
   }
   hub_->LogMessage("Started DI task", true);
   
   // end routing changedetectionEvent
   

   // Change detection now should be running on the input port.  
   // Configure a task to use change detection as the input
   err = hub_->StopTask(doTask_);
   if (err != DEVICE_OK)
      return err;

   nierr = DAQmxCreateTask("DOBlankTask", &doTask_);
   if (nierr != 0)
   {
      return hub_->TranslateNIError(nierr);;
   }
   hub_->LogMessage("Created DO task", true);

   nierr = DAQmxCreateDOChan(doTask_, port.c_str(), "DOSeqChan", DAQmx_Val_ChanForAllLines);
   if (nierr != 0)
   {
      return HandleTaskError(nierr);
   }
   hub_->LogMessage("Created DO channel for: " + port, true);
  
   boost::scoped_array<Tuint> samples;
   samples.reset(new Tuint[number]);
   if (sequenceOn && doSequence_.size() > 0)
   {
      if (blankOnLow ^ triggerPinState)
      {
         for (uInt32 i = 0; i < doSequence_.size(); i++)
         {
            samples.get()[2 * i] = doSequence_[i];
            if (blankingOn)
               samples.get()[2 * i + 1] = 0;
            else
               samples.get()[2 * i + 1] = doSequence_[i];
         }
      }
      else
      {
         for (uInt32 i = 0; i < doSequence_.size(); i++)
         {
            if (blankingOn)
               samples.get()[2 * i] = 0;
            else 
               samples.get()[2 * i] = doSequence_[i];
            
            samples.get()[2 * i + 1] = doSequence_[i];            
         }
      }
   }
   else  // assume that blanking is on, otherwise things make no sense
   {
      if (blankOnLow ^ triggerPinState)
      {
         samples.get()[0] = (Tuint)pos;
         samples.get()[1] = 0;
      }
      else
      {
         samples.get()[0] = 0;
         samples.get()[1] = (Tuint)pos;
      }
   }

   nierr = DAQmxCfgSampClkTiming(doTask_, hub_->niChangeDetection_.c_str(),
                  hub_->sampleRateHz_, DAQmx_Val_Rising, DAQmx_Val_ContSamps, number);
   if (nierr != 0)
   {
      return HandleTaskError(nierr);
   }
   hub_->LogMessage("Configured sample clock timing to use " + hub_->niChangeDetection_, true);

   int32 numWritten = 0;
   nierr = DaqmxWriteDigital(doTask_, static_cast<int32>(number), samples.get(), &numWritten);

   if (nierr != 0)
   {
      return HandleTaskError(nierr);
   }
   if (numWritten != static_cast<int32>(number))
   {
      hub_->LogMessage("Failed to write complete sequence");
      // This is presumably unlikely; no error code here
      return HandleTaskError(nierr);
   }
   hub_->LogMessage("Wrote samples", true);

   nierr = DAQmxStartTask(doTask_);
   if (nierr != 0)
   {
      return HandleTaskError(nierr);
   }
   hub_->LogMessage("Started DO task", true);
   
   return DEVICE_OK;
}


template<class Tuint>
int NIDAQDOHub<Tuint>::GetPinState(const std::string pinDesignation, bool & state)
{
   int err = hub_->StopTask(diTask_);
   if (err != DEVICE_OK)
      return err;

   int32 nierr = DAQmxCreateTask("DIReadTriggerPinTask", &diTask_);
   if (nierr != 0)
   {
      return hub_->TranslateNIError(nierr);;
   }
   hub_->LogMessage("Created DI task", true);
   nierr = DAQmxCreateDIChan(diTask_, pinDesignation.c_str(), "tIn", DAQmx_Val_ChanForAllLines);
   if (nierr != 0)
   {
      return hub_->TranslateNIError(nierr);;
   }
   nierr = DAQmxStartTask(diTask_);
   if (nierr != 0)
   {
      return hub_->TranslateNIError(nierr);;
   }
   uInt8 readArray[1];
   int32 read;
   int32 bytesPerSample;
   nierr = DAQmxReadDigitalLines(diTask_, 1, 0, DAQmx_Val_GroupByChannel, readArray, 1, &read, &bytesPerSample, NULL);
   if (nierr != 0)
   {
      return hub_->TranslateNIError(nierr);;
   }
   state = readArray[0] != 0;
   return DEVICE_OK;
}


template<class Tuint>
int NIDAQDOHub<Tuint>::StopDOBlankingAndSequence()
{
   hub_->StopTask(doTask_); // even if this fails, we still want to stop the diTask_
   return hub_->StopTask(diTask_);
}


template<class Tuint>
int NIDAQDOHub<Tuint>::HandleTaskError(int32 niError)
{
   std::string niErrorMsg;
   if (niError != 0)
   {
      niErrorMsg = GetNIDetailedErrorForMostRecentCall();
      hub_->LogMessage(niErrorMsg.c_str());
   }
   DAQmxClearTask(diTask_);
   diTask_ = 0;
   DAQmxClearTask(doTask_);
   doTask_ = 0;
   int err = DEVICE_OK;;
   if (niError != 0)
   {
      err = hub_->TranslateNIError(niError);
      hub_->SetErrorText(err, niErrorMsg.c_str());
   }
   return err;
}


template<class Tuint>
int NIDAQDOHub<Tuint>::DaqmxWriteDigital(TaskHandle doTask_, int32 samplesPerChan, const Tuint* samples, int32* numWritten)
{
   return ERR_UNKNOWN_PINS_PER_PORT;
}


template<>
int NIDAQDOHub<uInt8>::DaqmxWriteDigital(TaskHandle doTask, int32 samplesPerChan, const uInt8* samples, int32* numWritten)
{
   return DAQmxWriteDigitalU8(doTask, samplesPerChan,
      false, DAQmx_Val_WaitInfinitely, DAQmx_Val_GroupByChannel,
      samples, numWritten, NULL);
}


template<>
int NIDAQDOHub<uInt16>::DaqmxWriteDigital(TaskHandle doTask, int32 samplesPerChan, const uInt16* samples, int32* numWritten)
{
   return DAQmxWriteDigitalU16(doTask, samplesPerChan,
      false, DAQmx_Val_WaitInfinitely, DAQmx_Val_GroupByChannel,
      samples, numWritten, NULL);
}


template<>
int NIDAQDOHub<uInt32>::DaqmxWriteDigital(TaskHandle doTask, int32 samplesPerChan, const uInt32* samples, int32* numWritten)
{
   return DAQmxWriteDigitalU32(doTask, samplesPerChan,
      false, DAQmx_Val_WaitInfinitely, DAQmx_Val_GroupByChannel,
      samples, numWritten, NULL);
}


template class NIDAQDOHub<uInt8>;
template class NIDAQDOHub<uInt16>;
template class NIDAQDOHub<uInt32>;