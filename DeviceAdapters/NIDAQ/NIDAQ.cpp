// DESCRIPTION:   Drive multiple analog outputs on NI DAQ
// AUTHOR:        Mark Tsuchida, 2015
// COPYRIGHT:     2015-2016, Open Imaging, Inc.
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



MODULE_API void InitializeModuleData()
{
   RegisterDevice(g_DeviceNameNIDAQHub, MM::HubDevice, "Multi-channel analog output");
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
   minVolts_(0.0),
   maxVolts_(5.0),
   sampleRateHz_(10000.0),
   aoTask_(0),
   doHub8_(0),
   doHub16_(0),
   doHub32_(0)
{
   // TODO: discover devices available on this computer and list them here

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

   std::vector<std::string> triggerPorts = GetTriggerPortsForDevice(niDeviceName_);
   if (!triggerPorts.empty())
   {
      niTriggerPort_ = triggerPorts[0];
      pAct = new CPropertyAction(this, &NIDAQHub::OnTriggerInputPort);
      err = CreateStringProperty("TriggerInputPort", niTriggerPort_.c_str(), false, pAct);
      if (err != DEVICE_OK)
         return err;
      for (std::vector<std::string>::const_iterator it = triggerPorts.begin(),
            end = triggerPorts.end();
            it != end; ++it)
      {
         AddAllowedValue("TriggerInputPort", it->c_str());
      }

      pAct = new CPropertyAction(this, &NIDAQHub::OnSampleRate);
      err = CreateFloatProperty("SampleRateHz", sampleRateHz_, false, pAct);
      if (err != DEVICE_OK)
         return err;
   }

   std::vector<std::string> doPorts = GetDigitalPortsForDevice(niDeviceName_);
   if (doPorts.size() > 0)
   {
      uInt32 portWidth;
      int32 nierr = DAQmxGetPhysicalChanDOPortWidth(doPorts[0].c_str(), &portWidth);
      if (portWidth == 8) {
         doHub8_ = new NIDAQDOHub<uInt8>(this);
      }
      else if (portWidth == 16)
      {
         doHub16_ = new NIDAQDOHub<uInt16>(this);
      }
      else if (portWidth == 32) {
         doHub32_ = new NIDAQDOHub<uInt32>(this);
      }
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

   std::vector<std::string> doPorts =
       GetDigitalPortsForDevice(niDeviceName_);

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
   RemoveAOPortFromSequencing(port);
   // We do not restart sequencing for the remaining ports,
   // since it is meaningless (we can't preserve their state).
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
NIDAQHub::GetTriggerPortsForDevice(const std::string& device)
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
   return err;
}

int NIDAQHub::StopDOBlanking()
{
   if (doHub8_ != 0)
      return doHub8_->StopDOBlanking();
   else if (doHub16_ != 0)
      return doHub16_->StopDOBlanking();
   else if (doHub32_ != 0)
      return doHub32_->StopDOBlanking();

   return ERR_UNKNOWN_PINS_PER_PORT;
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
      if (aoTask_)
         return ERR_SEQUENCE_RUNNING;

      std::string port;
      pProp->Get(port);
      niTriggerPort_ = port;
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
      if (aoTask_)
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
   // We assume a given port appears at most once in physicalChannels_
   size_t n = physicalDOChannels_.size();
   for (size_t i = 0; i < n; ++i)
   {
      if (physicalDOChannels_[i] == port) {
         uInt32 portWidth;
         int32 nierr = DAQmxGetPhysicalChanDOPortWidth(port.c_str(), &portWidth);
         if (nierr != 0)
         {
            hub_->LogMessage(GetNIDetailedErrorForMostRecentCall().c_str());
            return;
         }
         physicalDOChannels_.erase(physicalDOChannels_.begin() + i);
         doChannelSequences_.erase(doChannelSequences_.begin() + i);
         break;
      }
   }
}


template<typename Tuint>
int NIDAQDOHub<Tuint>::StopDOSequenceForPort(const std::string& port)
{
   int err = hub_->StopTask(doTask_);
   if (err != DEVICE_OK)
      return err;
   RemoveDOPortFromSequencing(port);
   // We do not restart sequencing for the remaining ports,
   // since it is meaningless (we can't preserve their state).
   return DEVICE_OK;
}


/*
* Starts digital output sequences for all ports requesting sequences.
* Note that in practice, most NI boards only allow sequencing on one output port (the first port).
* We may refactor this code to reflect that better.
* Starting sequences on ports higher than the first one will most likely
* result in errors.
*
*/
template<typename Tuint>
inline int NIDAQDOHub<Tuint>::StartDOSequenceForPort(const std::string& port, const std::vector<Tuint> sequence)
{
   int err = hub_->StopTask(doTask_);
   if (err != DEVICE_OK)
      return err;

   err = AddDOPortToSequencing(port, sequence);
   if (err != DEVICE_OK)
      return err;

   err = StartDOSequencingTask();
   if (err != DEVICE_OK)
      return err;
   // We don't restart the task without this port on failure.
   // There is little point in doing so.

   return DEVICE_OK;
}
   

template<typename Tuint>
int NIDAQDOHub<Tuint>::StartDOSequencingTask()
{
   if (doTask_)
   {
      int err = hub_->StopTask(doTask_);
      if (err != DEVICE_OK)
         return err;
   }

   uInt32 portWidth;
   if (typeid(Tuint) == typeid(uInt8))
      portWidth = 8;
   else if (typeid(Tuint) == typeid(uInt8))
      portWidth = 16;
   else if (typeid(Tuint) == typeid(uInt32))
      portWidth = 32;
   else
      return ERR_UNKNOWN_PINS_PER_PORT;

   hub_->LogMessage("Starting DO sequencing task", true);

   int32 nierr = DAQmxGetPhysicalChanDOPortWidth(physicalDOChannels_[0].c_str(), &portWidth);
   if (nierr != 0)
   {
      return hub_->TranslateNIError(nierr);
   }

   size_t numChans = physicalDOChannels_.size();
   size_t samplesPerChan;
   int err = hub_->GetLCMSamplesPerChannel(samplesPerChan, doChannelSequences_);

   if (err != DEVICE_OK)
      return err;

   hub_->LogMessage(boost::lexical_cast<std::string>(numChans) + " channels", true);
   hub_->LogMessage("LCM sequence length = " +
      boost::lexical_cast<std::string>(samplesPerChan), true);

   nierr = DAQmxCreateTask("DOSeqTask", &doTask_);
   if (nierr != 0)
   {
      return hub_->TranslateNIError(nierr);;
   }
   hub_->LogMessage("Created task", true);

   boost::scoped_array<Tuint> samples;

   const std::string chanList = hub_->GetPhysicalChannelListForSequencing(physicalDOChannels_);
   nierr = DAQmxCreateDOChan(doTask_, chanList.c_str(), "DOSeqChan", DAQmx_Val_ChanForAllLines);
   if (nierr != 0)
   {
      return HandleTaskError(nierr);
   }
   hub_->LogMessage(("Created DO channel for: " + chanList).c_str(), true);

   nierr = DAQmxCfgSampClkTiming(doTask_, hub_->niTriggerPort_.c_str(),
      hub_->sampleRateHz_, DAQmx_Val_Rising, DAQmx_Val_ContSamps, samplesPerChan);
   if (nierr != 0)
   {
      return HandleTaskError(nierr);
   }
   hub_->LogMessage("Configured sample clock timing to use " + hub_->niTriggerPort_, true);


   samples.reset(new Tuint[samplesPerChan * numChans]);
   hub_->GetLCMSequence(samples.get(), doChannelSequences_);
   int32 numWritten = 0;

   nierr = DaqmxWriteDigital(doTask_, static_cast<int32>(samplesPerChan), samples.get(), &numWritten);

   if (nierr != 0)
   {
      return HandleTaskError(nierr);
   }
   if (numWritten != static_cast<int32>(samplesPerChan))
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
   hub_->LogMessage("Started task", true);

   return DEVICE_OK;
}

template<class Tuint>
int NIDAQDOHub<Tuint>::StartDOBlanking(const std::string& port, const bool sequenceOn, 
                                       const long& pos, const bool blankingDirection)
{
   int err = hub_->StopTask(diTask_);
   if (err != DEVICE_OK)
      return err;

   int32 nierr = DAQmxCreateTask("DIChangeTask", &diTask_);
   if (nierr != 0)
   {
      return hub_->TranslateNIError(nierr);;
   }
   hub_->LogMessage("Created DI task", true);

   int32 number = 2;
   if (!sequenceOn)
   {

   }

   //std::string triggerPort = hub_->niTriggerPort_;

   // may first need to read the state of the triggerport, since we will only get changes, not
   // its actual state.
   std::string triggerPort = "/Dev1/port0/line7";

   nierr = DAQmxCreateDIChan(diTask_, triggerPort.c_str(), "DIBlankChan", DAQmx_Val_ChanForAllLines);
   if (nierr != 0)
   {
      return HandleTaskError(nierr);
   }
   hub_->LogMessage("Created DI channel for: " + hub_->niTriggerPort_, true);


   nierr = DAQmxCfgChangeDetectionTiming(diTask_, triggerPort.c_str(),
      triggerPort.c_str(), DAQmx_Val_ContSamps, number);
   if (nierr != 0)
   {
      return HandleTaskError(nierr);
   }
   hub_->LogMessage("Configured change detection timing to use " + hub_->niTriggerPort_, true);


   std::string changeInput = "/Dev1/ChangeDetectionEvent";
   // this is only here to monitor the ChangeDetectionEvent
   nierr = DAQmxExportSignal(diTask_, DAQmx_Val_ChangeDetectionEvent, "/Dev1/PFI0");
   if (nierr != 0)
   {
      return HandleTaskError(nierr);
   }
   hub_->LogMessage("Routed change detection timing to  " + changeInput, true);

   nierr = DAQmxStartTask(diTask_);
   if (nierr != 0)
   {
      return HandleTaskError(nierr);
   }
   hub_->LogMessage("Started DI task", true);


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
   hub_->LogMessage("Created DI task", true);


   std::string tempPort;
   tempPort = "/Dev1/port0/line0:6";

   nierr = DAQmxCreateDOChan(doTask_, tempPort.c_str(), "DOSeqChan", DAQmx_Val_ChanForAllLines);
   if (nierr != 0)
   {
      return HandleTaskError(nierr);
   }
   hub_->LogMessage("Created DO channel for: " + tempPort, true);

   uInt32 portWidth = 8;
   /*
   if (typeid(Tuint) == typeid(uInt8))
      portWidth = 8;
   else if (typeid(Tuint) == typeid(uInt8))
      portWidth = 16;
   else if (typeid(Tuint) == typeid(uInt32))
      portWidth = 32;
   else
      return ERR_UNKNOWN_PINS_PER_PORT;

   hub_->LogMessage("Starting DO sequencing task", true);

   nierr = DAQmxGetPhysicalChanDOPortWidth(physicalDOChannels_[0].c_str(), &portWidth);
   if (nierr != 0)
   {
      return hub_->TranslateNIError(nierr);
   }
   */


   boost::scoped_array<Tuint> samples;
   samples.reset(new Tuint[number]);
   samples.get()[0] = pos;
   samples.get()[1] = 0;

   nierr = DAQmxCfgSampClkTiming(doTask_, changeInput.c_str(),
      hub_->sampleRateHz_, DAQmx_Val_Rising, DAQmx_Val_ContSamps, number);
   if (nierr != 0)
   {
      return HandleTaskError(nierr);
   }
   hub_->LogMessage("Configured sample clock timing to use " + changeInput, true);

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
int NIDAQDOHub<Tuint>::StopDOBlanking()
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







DigitalOutputPort::DigitalOutputPort(const std::string& port) :
      ErrorTranslator(21000, 21999, &DigitalOutputPort::SetErrorText),
   niPort_(port),
   initialized_(false),
   sequenceRunning_(false),
   blanking_(false),
   blankOnLow_(true),
   pos_(0),
   numPos_(0),
   portWidth_(0),
   neverSequenceable_(false),
   task_(0)
{
   InitializeDefaultErrorMessages();
   SetErrorText(ERR_SEQUENCE_RUNNING, "A sequence is running on this port.  Please stop this sequence first.");
   SetErrorText(ERR_SEQUENCE_TOO_LONG, "Sequence is too long. Try increasing sequence length in the Hub device.");
   SetErrorText(ERR_SEQUENCE_ZERO_LENGTH, "Sequence has length zero.");
   SetErrorText(ERR_UNKNOWN_PINS_PER_PORT, "Only 8 and 32 pin ports are supported.");

   CPropertyAction *pAct = new CPropertyAction(this, &DigitalOutputPort::OnSequenceable);
   CreateStringProperty("Sequencing", g_UseHubSetting, false, pAct, true);
   AddAllowedValue("Sequencing", g_UseHubSetting);
   AddAllowedValue("Sequencing", g_Never);
}


DigitalOutputPort::~DigitalOutputPort()
{
    Shutdown();
}


int DigitalOutputPort::Initialize()
{
    if (initialized_)
        return DEVICE_OK;

    // Need to set all pins of the port to output pins here on in Hub
    int32 nierr = DAQmxGetPhysicalChanDOPortWidth(niPort_.c_str(), &portWidth_);
    if (nierr != 0)
    {
        LogMessage(GetNIDetailedErrorForMostRecentCall().c_str());
        return TranslateNIError(nierr);
    }
    numPos_ = (1 << portWidth_) - 1;

    CPropertyAction* pAct = new CPropertyAction(this, &DigitalOutputPort::OnState);
    CreateIntegerProperty("State", 0, false, pAct);
    SetPropertyLimits("State", 0, numPos_);

    pAct = new CPropertyAction(this, &DigitalOutputPort::OnBlanking);
    CreateStringProperty("Blanking", blanking_ ? g_On : g_Off, false, pAct);
    AddAllowedValue("Blanking", g_Off);
    AddAllowedValue("Blanking", g_On);

    pAct = new CPropertyAction(this, &DigitalOutputPort::OnBlankingTriggerDirection);
    CreateStringProperty("Blank on", blankOnLow_ ? g_Low : g_High, false, pAct);
    AddAllowedValue("Blank on", g_Low);
    AddAllowedValue("Blank on", g_High);

    return DEVICE_OK;
}


int DigitalOutputPort::Shutdown()
{
    if (!initialized_)
        return DEVICE_OK;

    int err = StopTask();

    initialized_ = false;
    return err;
}


void DigitalOutputPort::GetName(char* name) const
{
    CDeviceUtils::CopyLimitedString(name,
        (g_DeviceNameNIDAQDOPortPrefix + niPort_).c_str());
}


int DigitalOutputPort::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(pos_);
   }
   else if (eAct == MM::AfterSet)
   {
      long pos;
      pProp->Get(pos);
      int err = StartOnDemandTask(pos);
      if (err == DEVICE_OK)
         pos_ = pos;
      return err;
   }

   else if (eAct == MM::IsSequenceable)
   {
      bool isHubSequenceable;
      GetHub()->IsSequencingEnabled(isHubSequenceable);

      bool sequenceable = neverSequenceable_ ? false : isHubSequenceable;

      if (sequenceable)
      {
         long maxLength;
         GetHub()->GetSequenceMaxLength(maxLength);
         pProp->SetSequenceable(maxLength);
      }
      else
      {
         pProp->SetSequenceable(0);
      }
    }
   else if (eAct == MM::AfterLoadSequence)
   {
      if (sequenceRunning_)
         return ERR_SEQUENCE_RUNNING;

      std::vector<std::string> sequence = pProp->GetSequence();
      long maxLength;
      GetHub()->GetSequenceMaxLength(maxLength);
      if (sequence.size() > maxLength)
         return DEVICE_SEQUENCE_TOO_LARGE;

      if (portWidth_ == 8) 
      {
         sequence8_.clear();
         for (unsigned int i = 0; i < sequence.size(); i++)
         {
            std::istringstream os(sequence[i]);
            uInt8 val;
            os >> val;
            sequence8_.push_back(val);
         }
      }
      else if (portWidth_ == 16)
      {
         sequence16_.clear();
         for (unsigned int i = 0; i < sequence.size(); i++)
         {
            std::istringstream os(sequence[i]);
            uInt32 val;
            os >> val;
            sequence16_.push_back(val);
         }
      }
      else if (portWidth_ == 32)
      {
         sequence32_.clear();
         for (unsigned int i = 0; i < sequence.size(); i++)
         {
            std::istringstream os(sequence[i]);
            uInt32 val;
            os >> val;
            sequence32_.push_back(val);
         }
      }
   }

   else if (eAct == MM::StartSequence)
   {
      int err = DEVICE_OK;
      sequenceRunning_ = true;
      // TODO: set the first state of the sequence before we start?
      if (portWidth_ == 8)
      {
         err = GetHub()->getDOHub8()->StartDOSequenceForPort(niPort_, sequence8_);
      }
      else if (portWidth_ == 16)
      {
         err = GetHub()->getDOHub16()->StartDOSequenceForPort(niPort_, sequence16_);
      }
      else if (portWidth_ == 32)
      {
         err = GetHub()->getDOHub32()->StartDOSequenceForPort(niPort_, sequence32_);
      }
      if (err != DEVICE_OK)
         sequenceRunning_ = false;
      return err;
   }

   else if (eAct == MM::StopSequence)
   {
      int err = DEVICE_OK;
      sequenceRunning_ = false;
      if (portWidth_ == 8)
      {
         err =  GetHub()->getDOHub8()->StopDOSequenceForPort(niPort_);
      }
      else if (portWidth_ == 16)
      {
         err = GetHub()->getDOHub16()->StopDOSequenceForPort(niPort_);
      }
      else if (portWidth_ == 32)
      {
         err =  GetHub()->getDOHub32()->StopDOSequenceForPort(niPort_);
      }
      if (err == DEVICE_OK) {
         err = StartOnDemandTask(pos_);
      }
      return err;
   }

    return DEVICE_OK;
}

int DigitalOutputPort::OnBlanking(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(blanking_ ? g_On : g_Off);
   }
   else if (eAct == MM::AfterSet)
   {

      std::string response;
      pProp->Get(response);
      bool blanking = response == g_On;
      if (blanking_ != blanking)
      {
         // do the thing in the hub
         blanking_ = blanking;
         if (blanking_)
            GetHub()->getDOHub8()->StartDOBlanking(niPort_, false, pos_, blankOnLow_);
         else
         {
            GetHub()->StopDOBlanking();
            return StartOnDemandTask(pos_);
         }

      }
   }
   return DEVICE_OK;
}

int DigitalOutputPort::OnBlankingTriggerDirection(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      std::string response = blankOnLow_ ? g_Low : g_High;
      pProp->Set(response.c_str());
   }
   else if (eAct == MM::AfterSet)
   {

      std::string response;
      pProp->Get(response);
      bool blankOnLow = response == g_Low ? true : false;
      if (blankOnLow_ != blankOnLow)
      {
         // do the thing in the hub
         blankOnLow_ = blankOnLow;
      }
   }
   return DEVICE_OK;
}


int DigitalOutputPort::OnSequenceable(MM::PropertyBase* pProp, MM::ActionType eAct)
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


int DigitalOutputPort::StopTask()
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


int DigitalOutputPort::StartOnDemandTask(long state)
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


    nierr = DAQmxCreateDOChan(task_,
        niPort_.c_str(), NULL, DAQmx_Val_ChanForAllLines);
    if (nierr != 0)
    {
        LogMessage(GetNIDetailedErrorForMostRecentCall().c_str());
        goto error;
    }
    LogMessage("Created DO channel", true);


    int32 numWritten = 0;

    if (portWidth_ == 8)
    {
       uInt8 samples[1];
       samples[0] = (uInt8)state;
       nierr = DAQmxWriteDigitalU8(task_, 1,
          true, DAQmx_Val_WaitInfinitely, DAQmx_Val_GroupByChannel,
          samples, &numWritten, NULL);
       if (nierr != 0)
       {
          LogMessage(GetNIDetailedErrorForMostRecentCall().c_str());
          goto error;
       }
    }
    else if (portWidth_ == 32)
    {
       uInt32 samples[1];
       samples[0] = (uInt32)state;
       nierr = DAQmxWriteDigitalU32(task_, 1,
          true, DAQmx_Val_WaitInfinitely, DAQmx_Val_GroupByChannel,
          samples, &numWritten, NULL);
       if (nierr != 0)
       {
          LogMessage(GetNIDetailedErrorForMostRecentCall().c_str());
          goto error;
       }
    } 
    else
    {
       LogMessage(("Found invalid number of pins per port: " +
          boost::lexical_cast<std::string>(portWidth_)).c_str(), true);
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
