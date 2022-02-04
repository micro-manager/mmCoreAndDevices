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


const char* g_DeviceNameMultiAnalogOutHub = "NIMultiAnalogOutHub";
const char* g_DeviceNameMultiAnalogOutPortPrefix = "NIMultiAnalogOut-";
const char* g_DeviceNameDigitalOutputPortPrefix = "NIDigitalOutput-";

const char* g_On = "On";
const char* g_Off = "Off";

const char* g_Never = "Never";
const char* g_UseHubSetting = "Use hub setting";


const int ERR_SEQUENCE_RUNNING = 2001;
const int ERR_SEQUENCE_TOO_LONG = 2002;
const int ERR_SEQUENCE_ZERO_LENGTH = 2003;
const int ERR_VOLTAGE_OUT_OF_RANGE = 2004;
const int ERR_NONUNIFORM_CHANNEL_VOLTAGE_RANGES = 2005;
const int ERR_VOLTAGE_RANGE_EXCEEDS_DEVICE_LIMITS = 2006;


MODULE_API void InitializeModuleData()
{
   RegisterDevice(g_DeviceNameMultiAnalogOutHub, MM::HubDevice, "Multi-channel analog output");
}


MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
   if (deviceName == 0)
      return 0;

   if (strcmp(deviceName, g_DeviceNameMultiAnalogOutHub) == 0)
   {
      return new MultiAnalogOutHub;
   }
   else if (std::string(deviceName).
      substr(0, strlen(g_DeviceNameMultiAnalogOutPortPrefix)) ==
      g_DeviceNameMultiAnalogOutPortPrefix)
   {
      return new MultiAnalogOutPort(std::string(deviceName).
         substr(strlen(g_DeviceNameMultiAnalogOutPortPrefix)));
   }
   else if (std::string(deviceName).substr(0, strlen(g_DeviceNameDigitalOutputPortPrefix)) ==
       g_DeviceNameDigitalOutputPortPrefix)
   {
      return new DigitalOutputPort(std::string(deviceName).
         substr(strlen(g_DeviceNameDigitalOutputPortPrefix)));
   }

   return 0;
}


MODULE_API void DeleteDevice(MM::Device* pDevice)
{
   delete pDevice;
}


inline std::string GetNIError(int32 nierr)
{
   char buf[1024];
   if (DAQmxGetErrorString(nierr, buf, sizeof(buf)))
      return "[failed to get DAQmx error code]";
   return buf;
}


inline std::string GetNIDetailedErrorForMostRecentCall()
{
   char buf[1024];
   if (DAQmxGetExtendedErrorInfo(buf, sizeof(buf)))
      return "[failed to get DAQmx extended error info]";
   return buf;
}


//
// MultiAnalogOutHub
//

MultiAnalogOutHub::MultiAnalogOutHub () :
   ErrorTranslator(20000, 20999, &MultiAnalogOutHub::SetErrorText),
   initialized_(false),
   maxSequenceLength_(1024),
   sequencingEnabled_(false),
   minVolts_(0.0),
   maxVolts_(5.0),
   sampleRateHz_(10000.0),
   aoTask_(0)
{
   CPropertyAction* pAct = new CPropertyAction(this, &MultiAnalogOutHub::OnDevice);
   int err = CreateStringProperty("Device", "", false, pAct, true);

   pAct = new CPropertyAction(this, &MultiAnalogOutHub::OnMaxSequenceLength);
   err = CreateIntegerProperty("MaxSequenceLength",
      static_cast<long>(maxSequenceLength_), false, pAct, true);
}


MultiAnalogOutHub::~MultiAnalogOutHub()
{
   Shutdown();
}


int MultiAnalogOutHub::Initialize()
{
   if (initialized_)
      return DEVICE_OK;

   if (!GetParentHub())
      return DEVICE_ERR;

   // Determine the possible voltage range
   int err = GetVoltageRangeForDevice(niDeviceName_, minVolts_, maxVolts_);
   if (err != DEVICE_OK)
      return err;

   CPropertyAction* pAct = new CPropertyAction(this, &MultiAnalogOutHub::OnSequencingEnabled);
   err = CreateStringProperty("Sequence", sequencingEnabled_ ? g_On : g_Off,
      false, pAct);
   if (err != DEVICE_OK)
      return err;
   AddAllowedValue("Sequence", g_On);
   AddAllowedValue("Sequence", g_Off);

   std::vector<std::string> triggerPorts = GetTriggerPortsForDevice(niDeviceName_);
   if (!triggerPorts.empty())
   {
      niTriggerPort_ = triggerPorts[0];
      pAct = new CPropertyAction(this, &MultiAnalogOutHub::OnTriggerInputPort);
      err = CreateStringProperty("TriggerInputPort", niTriggerPort_.c_str(), false, pAct);
      if (err != DEVICE_OK)
         return err;
      for (std::vector<std::string>::const_iterator it = triggerPorts.begin(),
            end = triggerPorts.end();
            it != end; ++it)
      {
         AddAllowedValue("TriggerInputPort", it->c_str());
      }

      pAct = new CPropertyAction(this, &MultiAnalogOutHub::OnSampleRate);
      err = CreateFloatProperty("SampleRateHz", sampleRateHz_, false, pAct);
      if (err != DEVICE_OK)
         return err;
   }

   initialized_ = true;
   return DEVICE_OK;
}


int MultiAnalogOutHub::Shutdown()
{
   if (!initialized_)
      return DEVICE_OK;

   int err = StopTask(aoTask_);
   err = StopTask(doTask_);

   physicalAOChannels_.clear();
   aoChannelSequences_.clear();
   physicalDOChannels_.clear();
   doChannelSequences8_.clear();
   doChannelSequences32_.clear();

   initialized_ = false;
   return err;
}


void MultiAnalogOutHub::GetName(char* name) const
{
   CDeviceUtils::CopyLimitedString(name, g_DeviceNameMultiAnalogOutHub);
}


int MultiAnalogOutHub::DetectInstalledDevices()
{
   std::vector<std::string> aoPorts =
      GetAnalogPortsForDevice(niDeviceName_);

   for (std::vector<std::string>::const_iterator it = aoPorts.begin(), end = aoPorts.end();
      it != end; ++it)
   {
      MM::Device* pDevice =
         ::CreateDevice((g_DeviceNameMultiAnalogOutPortPrefix + *it).c_str());
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
           ::CreateDevice((g_DeviceNameDigitalOutputPortPrefix + *it).c_str());
       if (pDevice)
       {
           AddInstalledDevice(pDevice);
       }
   }

   return DEVICE_OK;
}


int MultiAnalogOutHub::GetVoltageLimits(double& minVolts, double& maxVolts)
{
   minVolts = minVolts_;
   maxVolts = maxVolts_;
   return DEVICE_OK;
}


int MultiAnalogOutHub::StartAOSequenceForPort(const std::string& port,
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


int MultiAnalogOutHub::StopAOSequenceForPort(const std::string& port)
{
   int err = StopTask(aoTask_);
   if (err != DEVICE_OK)
      return err;
   RemoveAOPortFromSequencing(port);
   // We do not restart sequencing for the remaining ports,
   // since it is meaningless (we can't preserve their state).
   return DEVICE_OK;
}


int MultiAnalogOutHub::StartDOSequenceForPort8(const std::string& port,
   const std::vector<uInt8> sequence)
{
   int err = StopTask(doTask_);
   if (err != DEVICE_OK)
      return err;

   err = AddDOPortToSequencing8(port, sequence);
   if (err != DEVICE_OK)
      return err;

   err = StartDOSequencingTask();
   if (err != DEVICE_OK)
      return err;
   // We don't restart the task without this port on failure.
   // There is little point in doing so.

   return DEVICE_OK;
}


int MultiAnalogOutHub::StartDOSequenceForPort32(const std::string& port,
   const std::vector<uInt32> sequence)
{
   int err = StopTask(doTask_);
   if (err != DEVICE_OK)
      return err;

   err = AddDOPortToSequencing32(port, sequence);
   if (err != DEVICE_OK)
      return err;

   err = StartDOSequencingTask();
   if (err != DEVICE_OK)
      return err;
   // We don't restart the task without this port on failure.
   // There is little point in doing so.

   return DEVICE_OK;
}


int MultiAnalogOutHub::StopDOSequenceForPort(const std::string& port)
{
   int err = StopTask(doTask_);
   if (err != DEVICE_OK)
      return err;
   RemoveDOPortFromSequencing(port);
   // We do not restart sequencing for the remaining ports,
   // since it is meaningless (we can't preserve their state).
   return DEVICE_OK;
}


int MultiAnalogOutHub::IsSequencingEnabled(bool& flag) const
{
   flag = sequencingEnabled_;
   return DEVICE_OK;
}


int MultiAnalogOutHub::GetSequenceMaxLength(long& maxLength) const
{
   maxLength = static_cast<long>(maxSequenceLength_);
   return DEVICE_OK;
}


int MultiAnalogOutHub::AddAOPortToSequencing(const std::string& port,
   const std::vector<double> sequence)
{
   if (sequence.size() > maxSequenceLength_)
      return ERR_SEQUENCE_TOO_LONG;

   RemoveAOPortFromSequencing(port);

   physicalAOChannels_.push_back(port);
   aoChannelSequences_.push_back(sequence);
   return DEVICE_OK;
}


void MultiAnalogOutHub::RemoveAOPortFromSequencing(const std::string& port)
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


int MultiAnalogOutHub::AddDOPortToSequencing8(const std::string& port,
   const std::vector<uInt8> sequence)
{
   if (sequence.size() > maxSequenceLength_)
      return ERR_SEQUENCE_TOO_LONG;

   RemoveAOPortFromSequencing(port);

   physicalDOChannels_.push_back(port);
   doChannelSequences8_.push_back(sequence);
   return DEVICE_OK;
}

int MultiAnalogOutHub::AddDOPortToSequencing32(const std::string& port,
   const std::vector<uInt32> sequence)
{
   if (sequence.size() > maxSequenceLength_)
      return ERR_SEQUENCE_TOO_LONG;

   RemoveAOPortFromSequencing(port);

   physicalDOChannels_.push_back(port);
   doChannelSequences32_.push_back(sequence);
   return DEVICE_OK;
}

void MultiAnalogOutHub::RemoveDOPortFromSequencing(const std::string& port)
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
            LogMessage(GetNIDetailedErrorForMostRecentCall().c_str());
            return;
         }
         physicalDOChannels_.erase(physicalDOChannels_.begin() + i);
         // this will break if the device has both 8 and 32 pinWidth ports
         if (portWidth == 8)
            doChannelSequences8_.erase(doChannelSequences8_.begin() + i);
         else if (portWidth == 32)
            doChannelSequences32_.erase(doChannelSequences32_.begin() + i);

         break;
      }
   }
}


int MultiAnalogOutHub::GetVoltageRangeForDevice(
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
MultiAnalogOutHub::GetTriggerPortsForDevice(const std::string& device)
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
MultiAnalogOutHub::GetAnalogPortsForDevice(const std::string& device)
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
MultiAnalogOutHub::GetDigitalPortsForDevice(const std::string& device)
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


std::string MultiAnalogOutHub::GetPhysicalChannelListForSequencing(std::vector<std::string> channels) const
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
inline int MultiAnalogOutHub::GetLCMSamplesPerChannel(size_t& seqLen, std::vector<std::vector<T>> channelSequences) const
{
   // Use an arbitrary but reasonable limit to prevent
   // overflow or excessive memory consumption.
   const uint64_t factorLimit = 2 << 14;

   uint64_t len = 1;
   for (unsigned int i = 0; i < channelSequences.size(); ++i)
   {
      uint64_t channelSeqLen = aoChannelSequences_[i].size();
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


void MultiAnalogOutHub::GetLCMSequence(double* buffer) const
{
   size_t seqLen;
   if (GetLCMSamplesPerChannel(seqLen, aoChannelSequences_) != DEVICE_OK)
      return;

   for (unsigned int i = 0; i < aoChannelSequences_.size(); ++i)
   {
      size_t chanOffset = seqLen * i;
      size_t chanSeqLen = aoChannelSequences_[i].size();
      for (unsigned int j = 0; j < seqLen; ++j)
      {
         buffer[chanOffset + j] =
            aoChannelSequences_[i][j % chanSeqLen];
      }
   }
}


int MultiAnalogOutHub::StartAOSequencingTask()
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
   GetLCMSequence(samples.get());

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

int MultiAnalogOutHub::StartDOSequencingTask()
{
   if (doTask_)
   {
      int err = StopTask(doTask_);
      if (err != DEVICE_OK)
         return err;
   }

   LogMessage("Starting DO sequencing task", true);

   uInt32 portWidth;
   int32 nierr = DAQmxGetPhysicalChanDOPortWidth(physicalDOChannels_[0].c_str(), &portWidth);
   if (nierr != 0)
   {
      LogMessage(GetNIDetailedErrorForMostRecentCall().c_str());
      return TranslateNIError(nierr);
   }


   size_t numChans = physicalDOChannels_.size();
   size_t samplesPerChan;
   int err;

   if (portWidth == 8) 
   {
      err = GetLCMSamplesPerChannel(samplesPerChan, doChannelSequences8_);
   }
   else if (portWidth == 32)
   {
      err = GetLCMSamplesPerChannel(samplesPerChan, doChannelSequences32_);
   }
   if (err != DEVICE_OK)
      return err;

   LogMessage(boost::lexical_cast<std::string>(numChans) + " channels", true);
   LogMessage("LCM sequence length = " +
      boost::lexical_cast<std::string>(samplesPerChan), true);

   int32 nierr = DAQmxCreateTask("DOSeqTask", &doTask_);
   if (nierr != 0)
   {
      LogMessage(GetNIDetailedErrorForMostRecentCall().c_str());
      return nierr;
   }
   LogMessage("Created task", true);

   const std::string chanList = GetPhysicalChannelListForSequencing(physicalDOChannels_);
   nierr = DAQmxCreateDOChan(doTask_, chanList.c_str(), "DOSeqChan", DAQmx_Val_Volts);
   if (nierr != 0)
   {
      LogMessage(GetNIDetailedErrorForMostRecentCall().c_str());
      goto error;
   }
   LogMessage(("Created DO voltage channel for: " + chanList).c_str(), true);

   nierr = DAQmxCfgSampClkTiming(doTask_, niTriggerPort_.c_str(),
      sampleRateHz_, DAQmx_Val_Rising, DAQmx_Val_ContSamps, samplesPerChan);
   if (nierr != 0)
   {
      LogMessage(GetNIDetailedErrorForMostRecentCall().c_str());
      goto error;
   }
   LogMessage("Configured sample clock timing to use " + niTriggerPort_, true);


   if (portWidth == 8) {
      boost::scoped_array<uInt8> samples;
      samples.reset(new uInt8[samplesPerChan * numChans]);
      GetLCMSequence(samples.get());

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
   }

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


int MultiAnalogOutHub::StopTask(TaskHandle &task)
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


int MultiAnalogOutHub::OnDevice(MM::PropertyBase* pProp, MM::ActionType eAct)
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


int MultiAnalogOutHub::OnMaxSequenceLength(MM::PropertyBase* pProp, MM::ActionType eAct)
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


int MultiAnalogOutHub::OnSequencingEnabled(MM::PropertyBase* pProp, MM::ActionType eAct)
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


int MultiAnalogOutHub::OnTriggerInputPort(MM::PropertyBase* pProp, MM::ActionType eAct)
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


int MultiAnalogOutHub::OnSampleRate(MM::PropertyBase* pProp, MM::ActionType eAct)
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
// MultiAnalogOutPort
//

MultiAnalogOutPort::MultiAnalogOutPort(const std::string& port) :
   ErrorTranslator(21000, 21999, &MultiAnalogOutPort::SetErrorText),
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
   CPropertyAction* pAct = new CPropertyAction(this, &MultiAnalogOutPort::OnMinVolts);
   CreateFloatProperty("MinVolts", minVolts_, false, pAct, true);
   pAct = new CPropertyAction(this, &MultiAnalogOutPort::OnMaxVolts);
   CreateFloatProperty("MaxVolts", maxVolts_, false, pAct, true);

   pAct = new CPropertyAction(this, &MultiAnalogOutPort::OnSequenceable);
   CreateStringProperty("Sequencing", g_UseHubSetting, false, pAct, true);
   AddAllowedValue("Sequencing", g_UseHubSetting);
   AddAllowedValue("Sequencing", g_Never);
}


MultiAnalogOutPort::~MultiAnalogOutPort()
{
   Shutdown();
}


int MultiAnalogOutPort::Initialize()
{
   if (initialized_)
      return DEVICE_OK;

   // Check that the voltage range is allowed (since we cannot
   // enforce this when creating pre-init properties)
   double minVolts, maxVolts;
   int err = GetAOHub()->GetVoltageLimits(minVolts, maxVolts);
   if (err != DEVICE_OK)
      return TranslateHubError(err);
   LogMessage("Device voltage limits: " + boost::lexical_cast<std::string>(minVolts) +
      " to " + boost::lexical_cast<std::string>(maxVolts), true);
   if (minVolts_ < minVolts || maxVolts_ > maxVolts)
      return ERR_VOLTAGE_RANGE_EXCEEDS_DEVICE_LIMITS;

   CPropertyAction* pAct = new CPropertyAction(this, &MultiAnalogOutPort::OnVoltage);
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


int MultiAnalogOutPort::Shutdown()
{
   if (!initialized_)
      return DEVICE_OK;

   int err = StopTask();

   unsentSequence_.clear();
   sentSequence_.clear();

   initialized_ = false;
   return err;
}


void MultiAnalogOutPort::GetName(char* name) const
{
   CDeviceUtils::CopyLimitedString(name,
      (g_DeviceNameMultiAnalogOutPortPrefix + niPort_).c_str());
}


int MultiAnalogOutPort::SetGateOpen(bool open)
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


int MultiAnalogOutPort::GetGateOpen(bool& open)
{
   open = gateOpen_;
   return DEVICE_OK;
}


int MultiAnalogOutPort::SetSignal(double volts)
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


int MultiAnalogOutPort::GetLimits(double& minVolts, double& maxVolts)
{
   minVolts = minVolts_;
   maxVolts = maxVolts_;
   return DEVICE_OK;
}


int MultiAnalogOutPort::IsDASequenceable(bool& isSequenceable) const
{
   if (neverSequenceable_)
      return false;

   // Translation from hub error code skipped (since this never fails)
   return GetAOHub()->IsSequencingEnabled(isSequenceable);
}


int MultiAnalogOutPort::GetDASequenceMaxLength(long& maxLength) const
{
   // Translation from hub error code skipped (since this never fails)
   return GetAOHub()->GetSequenceMaxLength(maxLength);
}


int MultiAnalogOutPort::StartDASequence()
{
   if (task_)
      StopTask();

   sequenceRunning_ = true;

   int err = GetAOHub()->StartAOSequenceForPort(niPort_, sentSequence_);
   if (err != DEVICE_OK)
      return TranslateHubError(err);

   return DEVICE_OK;
}


int MultiAnalogOutPort::StopDASequence()
{
   int err = GetAOHub()->StopAOSequenceForPort(niPort_);
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


int MultiAnalogOutPort::ClearDASequence()
{
   unsentSequence_.clear();
   return DEVICE_OK;
}


int MultiAnalogOutPort::AddToDASequence(double voltage)
{
   if (voltage < minVolts_ || voltage > maxVolts_)
      return ERR_VOLTAGE_OUT_OF_RANGE;

   unsentSequence_.push_back(voltage);
   return DEVICE_OK;
}


int MultiAnalogOutPort::SendDASequence()
{
   if (sequenceRunning_)
      return ERR_SEQUENCE_RUNNING;

   sentSequence_ = unsentSequence_;
   // We don't actually "write" the sequence here, because writing
   // needs to take place once the correct task has been set up for
   // all of the AO channels.
   return DEVICE_OK;
}


int MultiAnalogOutPort::OnMinVolts(MM::PropertyBase* pProp, MM::ActionType eAct)
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


int MultiAnalogOutPort::OnMaxVolts(MM::PropertyBase* pProp, MM::ActionType eAct)
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


int MultiAnalogOutPort::OnSequenceable(MM::PropertyBase* pProp, MM::ActionType eAct)
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


int MultiAnalogOutPort::OnVoltage(MM::PropertyBase* pProp, MM::ActionType eAct)
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


int MultiAnalogOutPort::TranslateHubError(int err)
{
   if (err == DEVICE_OK)
      return DEVICE_OK;
   char buf[MM::MaxStrLength];
   if (GetAOHub()->GetErrorText(err, buf))
      return NewErrorCode(buf);
   return NewErrorCode("Unknown hub error");
}


int MultiAnalogOutPort::StartOnDemandTask(double voltage)
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

   nierr = DAQmxCreateAOVoltageChan(task_,
      niPort_.c_str(), NULL, minVolts_, maxVolts_,
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


int MultiAnalogOutPort::StopTask()
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


DigitalOutputPort::DigitalOutputPort(const std::string& port) :
    ErrorTranslator(22000, 22999, &DigitalOutputPort::SetErrorText),
    niPort_(port),
    initialized_(false),
    sequenceRunning_(false),
    numPos_(0),
    portWidth_(0),
    neverSequenceable_(false),
    task_(0)
{
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
    CreateIntegerProperty("State", 0, false, pAct, false);
    SetPropertyLimits("State", 0, numPos_);

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
        (g_DeviceNameDigitalOutputPortPrefix + niPort_).c_str());
}


int DigitalOutputPort::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      // nothing to do, let the caller use cached property
   }
   else if (eAct == MM::AfterSet)
   {
      long pos;
      pProp->Get(pos);
      return StartOnDemandTask(pos);
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
      if (portWidth_ == 8)
      {
         return GetHub()->StartDOSequenceForPort8(niPort_, sequence8_);
      }
      else if (portWidth_ == 32)
      {
         return GetHub()->StartDOSequenceForPort32(niPort_, sequence32_);
      }
   }

   else if (eAct == MM::StopSequence)
   {
      return GetHub()->StopDOSequenceForPort(niPort_);
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
