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

#include <iostream>
#include <fstream>

#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/math/common_factor_rt.hpp>
#include <boost/scoped_array.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>


const char* g_DeviceNameNIDAQHub = "NIDAQHub";
const char* g_DeviceNameNIDAQAOPortPrefix = "NIDAQAO-";
const char* g_DeviceNameNIDAQDOPortPrefix = "NIDAQDO-";
const char* g_DeviceNameNIDAQAIPortPrefix = "NIDAQAI-";
const char* g_DeviceNameNIDAQAOBlankPrefix = "NIDAQAOBlank-";

const char* g_On = "On";
const char* g_Off = "Off";
const char* g_Low = "Low";
const char* g_High = "High";
const char* g_Never = "Never";
const char* g_UseHubSetting = "Use hub setting";
const char* g_Post = "Post";
const char* g_Pre = "Pre";

const int ERR_SEQUENCE_RUNNING = 2001;
const int ERR_SEQUENCE_TOO_LONG = 2002;
const int ERR_SEQUENCE_ZERO_LENGTH = 2003;
const int ERR_VOLTAGE_OUT_OF_RANGE = 2004;
const int ERR_NONUNIFORM_CHANNEL_VOLTAGE_RANGES = 2005;
const int ERR_VOLTAGE_RANGE_EXCEEDS_DEVICE_LIMITS = 2006;
const int ERR_UNKNOWN_PINS_PER_PORT = 2007;
const int ERR_UNEXPECTED_AMOUNT_OF_MEASUREMENTS = 2008;
const int ERR_FAILED_TO_OPEN_TRACE = 2009;
const int ERR_INVALID_REQUEST = 2010;
const int ERR_SEQUENCE_INVALID_NUMBER = 2011;



MODULE_API void InitializeModuleData()
{
   RegisterDevice(g_DeviceNameNIDAQHub, MM::HubDevice, "NIDAQ analog and digital output");
}


// Returns true if name starts with prefix, and if so stores the remainder of
// name (the part after the prefix) in suffix.
//
// Using std::string::substr(pos) directly for this throws std::out_of_range
// when pos > size(). Such an exception escaping CreateDevice() propagates
// through MMCore and the JNI boundary and terminates the JVM, so avoid it.
static bool SplitDeviceNamePrefix(const char* name, const char* prefix,
   std::string& suffix)
{
   const size_t prefixLen = strlen(prefix);
   const std::string nameStr(name);
   if (nameStr.size() < prefixLen)
      return false;
   if (nameStr.compare(0, prefixLen, prefix) != 0)
      return false;
   suffix = nameStr.substr(prefixLen);
   return true;
}


MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
   if (deviceName == 0)
      return 0;

   try
   {
      if (strcmp(deviceName, g_DeviceNameNIDAQHub) == 0)
      {
         return new NIDAQHub;
      }

      std::string port;
      if (SplitDeviceNamePrefix(deviceName, g_DeviceNameNIDAQAOPortPrefix, port))
         return new NIAnalogOutputPort(port);
      if (SplitDeviceNamePrefix(deviceName, g_DeviceNameNIDAQDOPortPrefix, port))
         return new DigitalOutputPort(port);
      if (SplitDeviceNamePrefix(deviceName, g_DeviceNameNIDAQAIPortPrefix, port))
         return new NIAnalogInputPort(port);
      if (SplitDeviceNamePrefix(deviceName, g_DeviceNameNIDAQAOBlankPrefix, port))
         return new AnalogOutputBlankingPort(port);
   }
   catch (const std::exception&)
   {
      // No device object exists yet, so LogMessage() is not available here.
      // Returning 0 makes MMCore report a normal "failed to instantiate
      // device" error rather than letting the exception kill the process.
      return 0;
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
   minVoltsOut_(0.0),
   maxVoltsOut_(5.0),
   sampleRateHz_(10000.0),
   aoTask_(0),
   doTask_(0),
   aoBlankTask_(0),
   aoBlankDiTask_(0),
   doHub8_(0),
   doHub16_(0),
   doHub32_(0),
   mThread_(0),
   expectedMaxVoltsIn_(5.0),
   expectedMinVoltsIn_(-5.0),
   traceFrequency_(100.0),
   traceAmount_(100),
   tracePath_("C:/Program Files/Micro-Manager-2.0/CoreLogs"),
   measuringTrace_(false),
   tThread_(0)
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
   // Wrapper: a C++ exception escaping Initialize() propagates through MMCore
   // and the JNI boundary and terminates the JVM with no usable diagnostic.
   // Catch it here, log what it was, and report a normal device error instead.
   try
   {
      return InitializeImpl();
   }
   catch (const std::exception& e)
   {
      LogMessage(std::string("EXCEPTION in NIDAQHub::Initialize: ") +
         typeid(e).name() + ": " + e.what());
      return DEVICE_ERR;
   }
   catch (...)
   {
      LogMessage("Unknown (non-standard) C++ exception in NIDAQHub::Initialize");
      return DEVICE_ERR;
   }
}


int NIDAQHub::InitializeImpl()
{
   if (initialized_)
      return DEVICE_OK;

   if (!GetParentHub())
      return DEVICE_ERR;


   // Dynamically determine name of ChangeDetectionEvent for this device
   niChangeDetection_ = "/" + niDeviceName_ + "/ChangeDetectionEvent";
   niSampleClock_ = "/" + niDeviceName_ + "/do/SampleClock";

   // Determine the possible voltage range
   int err = GetVoltageRangeForDevice(niDeviceName_, minVoltsOut_, maxVoltsOut_);
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

   pAct = new CPropertyAction(this, &NIDAQHub::OnExpectedMaxVoltsIn);
   err = CreateFloatProperty("Maximum expected measured Voltage", 5.0, false, pAct);
   if (err != DEVICE_OK)
       return err;

   pAct = new CPropertyAction(this, &NIDAQHub::OnExpectedMinVoltsIn);
   err = CreateFloatProperty("Minimum expected measured Voltage", -5.0, false, pAct);
   if (err != DEVICE_OK)
       return err;

   mThread_ = new InputMonitoringThread(this);

   pAct = new CPropertyAction(this, &NIDAQHub::OnTraceFrequency);
   err = CreateFloatProperty("Trace sampling frequency", 10.0, false, pAct);
   if (err != DEVICE_OK)
       return err;

   pAct = new CPropertyAction(this, &NIDAQHub::OnTraceAmount);
   err = CreateIntegerProperty("Total samples taken", 100, false, pAct);
   if (err != DEVICE_OK)
       return err;

   pAct = new CPropertyAction(this, &NIDAQHub::OnTracePath);
   err = CreateStringProperty("Trace folder", "C:/Program Files/Micro-Manager-2.0/CoreLogs", false, pAct);
   if (err != DEVICE_OK)
       return err;

   pAct = new CPropertyAction(this, &NIDAQHub::OnTraceRunning);
   err = CreateStringProperty("Trace Running", "Stopped", false, pAct);
   if (err != DEVICE_OK)
       return err;
   AddAllowedValue("Trace Running", "Stopped");
   AddAllowedValue("Trace Running", "Running");

   tThread_ = new TraceMonitoringThread(this);

   initialized_ = true;
   return DEVICE_OK;
}


void NIDAQHub::DestroyMonitoringThread(InputMonitoringThread*& t)
{
   if (t == 0)
      return;
   t->Stop();
   t->Join();
   delete t;
   t = 0;
}


void NIDAQHub::DestroyMonitoringThread(TraceMonitoringThread*& t)
{
   if (t == 0)
      return;
   t->Stop();
   t->Join();
   delete t;
   t = 0;
}


int NIDAQHub::Shutdown()
{
   // Wrapper: see NIDAQHub::Initialize(). Shutdown() is also called from
   // ~NIDAQHub(), where an escaping exception means immediate std::terminate.
   try
   {
      return ShutdownImpl();
   }
   catch (const std::exception& e)
   {
      LogMessage(std::string("EXCEPTION in NIDAQHub::Shutdown: ") +
         typeid(e).name() + ": " + e.what());
      return DEVICE_ERR;
   }
   catch (...)
   {
      LogMessage("Unknown (non-standard) C++ exception in NIDAQHub::Shutdown");
      return DEVICE_ERR;
   }
}


int NIDAQHub::ShutdownImpl()
{
   if (!initialized_)
      return DEVICE_OK;

   // Destroy both threads first, so that no worker can still be running
   // (and dereferencing this hub) while the members below are torn down.
   //
   // Order matters: the trace thread must go first. On exit its svc() calls
   // hub_->FinishTrace(), which dereferences mThread_, so destroying mThread_
   // while the trace thread is still running is a null dereference.
   DestroyMonitoringThread(tThread_);
   DestroyMonitoringThread(mThread_);

   int err = StopTask(aoTask_);

   StopTask(aoBlankTask_);
   StopTask(aoBlankDiTask_);

   physicalAOChannels_.clear();
   aoChannelSequences_.clear();

   // Independent deletes: the previous if/else-if chain leaked doHub16_ and
   // doHub32_ whenever doHub8_ was non-null, which is the normal case since
   // InitializeImpl() allocates all three together. Deleting null is a no-op.
   delete doHub8_;
   doHub8_ = 0;
   delete doHub16_;
   doHub16_ = 0;
   delete doHub32_;
   doHub32_ = 0;

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
      GetAnalogOutputPortsForDevice(niDeviceName_);

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

   std::vector<std::string> aiPorts =
       GetAnalogInputPortsForDevice(niDeviceName_);

   for (std::vector<std::string>::const_iterator it = aiPorts.begin(), end = aiPorts.end();
       it != end; ++it)
   {
       MM::Device* pDevice =
           ::CreateDevice((g_DeviceNameNIDAQAIPortPrefix + *it).c_str());
       if (pDevice)
       {
           AddInstalledDevice(pDevice);
       }
   }

   // One analog-output blanking device per board
   if (!aoPorts.empty())
   {
       MM::Device* pDevice =
           ::CreateDevice((g_DeviceNameNIDAQAOBlankPrefix + niDeviceName_).c_str());
       if (pDevice)
       {
           AddInstalledDevice(pDevice);
       }
   }

   return DEVICE_OK;
}


int NIDAQHub::GetVoltageLimits(double& minVolts, double& maxVolts)
{
   minVolts = minVoltsOut_;
   maxVolts = maxVoltsOut_;
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
NIDAQHub::GetAnalogOutputPortsForDevice(const std::string& device)
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
NIDAQHub::GetAnalogInputPortsForDevice(const std::string& device)
{
    std::vector<std::string> result;

    char ports[4096];
    int32 nierr = DAQmxGetDevAIPhysicalChans(device.c_str(), ports, sizeof(ports));
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
   const uint64_t factorLimit = channelSequences.size() * maxSequenceLength_;

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
      "AOSeqChan", minVoltsOut_, maxVoltsOut_, DAQmx_Val_Volts,
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


int NIDAQHub::ReadTriggerPinState(const std::string& triggerPort, bool& state)
{
   TaskHandle task = 0;
   int32 nierr = DAQmxCreateTask("AOBlankReadPinTask", &task);
   if (nierr != 0)
      return TranslateNIError(nierr);
   nierr = DAQmxCreateDIChan(task, triggerPort.c_str(), "tIn", DAQmx_Val_ChanForAllLines);
   if (nierr != 0)
   {
      DAQmxClearTask(task);
      return TranslateNIError(nierr);
   }
   nierr = DAQmxStartTask(task);
   if (nierr != 0)
   {
      DAQmxClearTask(task);
      return TranslateNIError(nierr);
   }
   uInt8 readArray[1];
   int32 read;
   int32 bytesPerSample;
   nierr = DAQmxReadDigitalLines(task, 1, 0, DAQmx_Val_GroupByChannel, readArray, 1, &read, &bytesPerSample, NULL);
   DAQmxClearTask(task);
   if (nierr != 0)
      return TranslateNIError(nierr);
   state = readArray[0] != 0;
   return DEVICE_OK;
}


/**
* Gate a set of analog-output channels with a hardware trigger (no sequencing).
*
* Same scheme as NIDAQDOHub::StartDOBlankingAndOrSequence: a DI change-detection
* task on triggerPort generates /Dev/ChangeDetectionEvent, which clocks an AO
* task over channelList holding a regenerating 2-sample-per-channel buffer
* {on, off}.  Each trigger edge advances the buffer, so the AO output follows
* the trigger line.
*/
int NIDAQHub::StartAOBlanking(const std::string& channelList, const size_t numChannels,
   const double onVolts, const double offVolts, const long state,
   const bool blankOnLow, const std::string& triggerPort)
{
   if (numChannels == 0)
      return ERR_INVALID_REQUEST;

   // Release the previous tasks first; the pin read below needs the trigger line.
   StopAOBlanking();

   // Change detection only reports edges, so read the current level to
   // phase-align the buffer.
   bool triggerPinState;
   int err = ReadTriggerPinState(triggerPort, triggerPinState);
   if (err != DEVICE_OK)
      return err;

   // True: the trigger sits at its blanking level, the next edge turns on.
   const bool firstSampleIsOn = (blankOnLow ^ triggerPinState);

   // A clocked AO task does not drive the pins until its first clock edge, so
   // first set the level matching the current trigger state statically, then
   // release that on-demand task (AO holds its value across the clear).
   err = SetAOPortState(channelList, numChannels, onVolts, offVolts,
      firstSampleIsOn ? 0 : state);
   if (err != DEVICE_OK)
      return err;
   StopAOBlanking();

   int32 nierr = 0;
   std::string niErrorMsg;

   nierr = DAQmxCreateTask("AOBlankDITask", &aoBlankDiTask_);
   if (nierr != 0)
      return TranslateNIError(nierr);
   LogMessage("Created AO blanking DI task", true);

   nierr = DAQmxCreateDIChan(aoBlankDiTask_, triggerPort.c_str(), "AOBlankDIChan", DAQmx_Val_ChanForAllLines);
   if (nierr != 0)
      goto error;
   LogMessage("Created DI channel for: " + triggerPort, true);

   nierr = DAQmxCfgChangeDetectionTiming(aoBlankDiTask_, triggerPort.c_str(),
      triggerPort.c_str(), DAQmx_Val_ContSamps, 2);
   if (nierr != 0)
      goto error;
   LogMessage("Configured change detection timing to use " + triggerPort, true);

   // Nobody reads this task; let its buffer wrap instead of overflowing.
   nierr = DAQmxSetReadOverWrite(aoBlankDiTask_, DAQmx_Val_OverwriteUnreadSamps);
   if (nierr != 0)
      goto error;

   nierr = DAQmxStartTask(aoBlankDiTask_);
   if (nierr != 0)
      goto error;
   LogMessage("Started AO blanking DI task", true);

   nierr = DAQmxCreateTask("AOBlankTask", &aoBlankTask_);
   if (nierr != 0)
      goto error;
   LogMessage("Created AO blanking task", true);

   nierr = DAQmxCreateAOVoltageChan(aoBlankTask_, channelList.c_str(), "AOBlankChan",
      minVoltsOut_, maxVoltsOut_, DAQmx_Val_Volts, NULL);
   if (nierr != 0)
      goto error;
   LogMessage("Created AO voltage channel for: " + channelList, true);

   nierr = DAQmxCfgSampClkTiming(aoBlankTask_, niChangeDetection_.c_str(),
      sampleRateHz_, DAQmx_Val_Rising, DAQmx_Val_ContSamps, 2);
   if (nierr != 0)
      goto error;
   LogMessage("Configured sample clock timing to use " + niChangeDetection_, true);

   nierr = DAQmxSetWriteRegenMode(aoBlankTask_, DAQmx_Val_AllowRegen);
   if (nierr != 0)
      goto error;

   err = WriteAOBlankingSamples(numChannels, onVolts, offVolts, state, firstSampleIsOn);
   if (err != DEVICE_OK)
   {
      StopAOBlanking();
      return err;
   }
   LogMessage("Wrote AO blanking samples", true);

   nierr = DAQmxStartTask(aoBlankTask_);
   if (nierr != 0)
      goto error;
   LogMessage("Started AO blanking task", true);

   return DEVICE_OK;

error:
   niErrorMsg = GetNIDetailedErrorForMostRecentCall();
   LogMessage(niErrorMsg.c_str());
   StopAOBlanking();
   err = TranslateNIError(nierr);
   SetErrorText(err, niErrorMsg.c_str());
   return err;
}


int NIDAQHub::StopAOBlanking()
{
   // Clear and null unconditionally (unlike StopTask), so a failed clear can
   // never leave a stale handle that blocks the next StartAOBlanking.
   if (aoBlankTask_) { DAQmxClearTask(aoBlankTask_); aoBlankTask_ = 0; }
   if (aoBlankDiTask_) { DAQmxClearTask(aoBlankDiTask_); aoBlankDiTask_ = 0; }
   return DEVICE_OK;
}


int NIDAQHub::GetAOBlankingEdgeCount(uInt64& count)
{
   count = 0;
   if (!aoBlankDiTask_)
      return DEVICE_OK;

   // Count on the DI side: each detected change is one acquired sample.  The AO
   // task's generated-sample counter is not usable here, as a 2-sample buffer is
   // regenerated from onboard memory and that counter does not advance.
   int32 nierr = DAQmxGetReadTotalSampPerChanAcquired(aoBlankDiTask_, &count);
   if (nierr != 0)
      return TranslateNIError(nierr);
   return DEVICE_OK;
}


int NIDAQHub::WriteAOBlankingSamples(const size_t numChannels, const double onVolts,
   const double offVolts, const long state, const bool firstSampleIsOn)
{
   if (!aoBlankTask_)
      return ERR_INVALID_REQUEST;

   // Grouped by channel: [chan0 s0, chan0 s1, chan1 s0, ...].  Active channels
   // alternate on/off, inactive ones stay off; firstSampleIsOn sets the phase.
   boost::scoped_array<float64> samples(new float64[2 * numChannels]);
   for (size_t c = 0; c < numChannels; ++c)
   {
      const bool active = ((state >> c) & 1L) != 0;
      const double onValue = active ? onVolts : offVolts;
      if (firstSampleIsOn)
      {
         samples[2 * c] = onValue;
         samples[2 * c + 1] = offVolts;
      }
      else
      {
         samples[2 * c] = offVolts;
         samples[2 * c + 1] = onValue;
      }
   }

   int32 numWritten = 0;
   int32 nierr = DAQmxWriteAnalogF64(aoBlankTask_, 2, false, DAQmx_Val_WaitInfinitely,
      DAQmx_Val_GroupByChannel, samples.get(), &numWritten, NULL);
   if (nierr != 0)
   {
      LogMessage(GetNIDetailedErrorForMostRecentCall().c_str());
      return TranslateNIError(nierr);
   }
   if (numWritten != 2)
   {
      LogMessage("Failed to write complete AO blanking buffer");
      return DEVICE_ERR;
   }
   return DEVICE_OK;
}


/**
* Statically drive the AO blanking channels: active channels output onVolts,
* inactive channels offVolts.  Used when blanking is off.
*/
int NIDAQHub::SetAOPortState(const std::string& channelList, const size_t numChannels,
   const double onVolts, const double offVolts, const long state)
{
   if (numChannels == 0)
      return ERR_INVALID_REQUEST;

   if (aoBlankTask_) { DAQmxClearTask(aoBlankTask_); aoBlankTask_ = 0; }

   int32 nierr = DAQmxCreateTask(NULL, &aoBlankTask_);
   if (nierr != 0)
      return TranslateNIError(nierr);

   nierr = DAQmxCreateAOVoltageChan(aoBlankTask_, channelList.c_str(), NULL,
      minVoltsOut_, maxVoltsOut_, DAQmx_Val_Volts, NULL);
   if (nierr != 0)
      goto error;

   {
      boost::scoped_array<float64> samples(new float64[numChannels]);
      for (size_t c = 0; c < numChannels; ++c)
      {
         const bool active = ((state >> c) & 1L) != 0;
         samples[c] = active ? onVolts : offVolts;
      }

      int32 numWritten = 0;
      nierr = DAQmxWriteAnalogF64(aoBlankTask_, 1, true, DAQmx_Val_WaitInfinitely,
         DAQmx_Val_GroupByChannel, samples.get(), &numWritten, NULL);
      if (nierr != 0)
         goto error;
      if (numWritten != 1)
      {
         LogMessage("Failed to write AO port state");
         goto error;
      }
   }

   return DEVICE_OK;

error:
   if (nierr != 0)
      LogMessage(GetNIDetailedErrorForMostRecentCall().c_str());
   DAQmxClearTask(aoBlankTask_);
   aoBlankTask_ = 0;
   if (nierr != 0)
      return TranslateNIError(nierr);
   return DEVICE_ERR;
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


int NIDAQHub::StartAIMeasuringForPort(NIAnalogInputPort* port)
{
    //check if port has not already been added
    size_t n = physicalAIChannels_.size();
    for (size_t i = 0; i < n; ++i)
    {
        if (physicalAIChannels_[i] == port)
            return DEVICE_OK;
    }
    physicalAIChannels_.push_back(port);
    int err;
    if (measuringTrace_)
    {
        DestroyMonitoringThread(tThread_);

        tThread_ = new TraceMonitoringThread(this);
        err = tThread_->Start(GetPhysicalChannelListForMeasuring(physicalAIChannels_), expectedMinVoltsIn_,
            expectedMaxVoltsIn_, (float) traceFrequency_, traceAmount_, (int) physicalAIChannels_.size());
    }
    else
    {
        DestroyMonitoringThread(mThread_);

        mThread_ = new InputMonitoringThread(this);
        err = mThread_->Start(GetPhysicalChannelListForMeasuring(physicalAIChannels_), expectedMinVoltsIn_, expectedMaxVoltsIn_);
    }
   

    return err;
}


int NIDAQHub::StopAIMeasuringForPort(NIAnalogInputPort* port)
{
    size_t n = physicalAIChannels_.size();
    for (size_t i = 0; i < n; ++i)
    {
        if (physicalAIChannels_[i] == port)
        {
            physicalAIChannels_.erase(physicalAIChannels_.begin() + i);

            if (measuringTrace_)
            {
                DestroyMonitoringThread(tThread_);

                tThread_ = new TraceMonitoringThread(this);
                int err = DEVICE_OK;
                if (n > 1)
                    err = tThread_->Start(GetPhysicalChannelListForMeasuring(physicalAIChannels_), expectedMinVoltsIn_,
                        expectedMaxVoltsIn_, (float) traceFrequency_, traceAmount_, (int) physicalAIChannels_.size());

                return err;
            }
            else
            {
                DestroyMonitoringThread(mThread_);

                mThread_ = new InputMonitoringThread(this);

                int err = DEVICE_OK;
                if (n > 1)
                    err = mThread_->Start(GetPhysicalChannelListForMeasuring(physicalAIChannels_), expectedMinVoltsIn_, expectedMaxVoltsIn_);

                return err;
            }
        }
    }
    return DEVICE_OK;
}


int NIDAQHub::UpdateAIValues(float64* values, int32 amount)
{
    if (amount != 1)
        return ERR_UNEXPECTED_AMOUNT_OF_MEASUREMENTS;

    size_t n = physicalAIChannels_.size();
    for (size_t i = 0; i < n; ++i)
    {
        physicalAIChannels_[i]->UpdateState((float) values[i]);
    }

    return DEVICE_OK;
}


std::string NIDAQHub::GetPhysicalChannelListForMeasuring(std::vector<NIAnalogInputPort*> channels)
{
    std::string result;
    size_t n = channels.size();
    for (size_t i = 0; i < n; ++i)
    {
        result += channels[i]->niPort_;
        if (i < n - 1)
            result += ", ";
    }
    return result;
}


int NIDAQHub::StartTrace()
{
    measuringTrace_ = true;

    // Destroy the trace thread before the input thread: a trace thread that is
    // still running calls hub_->FinishTrace() on exit, which dereferences
    // mThread_. (This was previously a bare "delete tThread_", which destroyed
    // a possibly still running thread without stopping or joining it first.)
    DestroyMonitoringThread(tThread_);
    DestroyMonitoringThread(mThread_);
    mThread_ = new InputMonitoringThread(this);

    tThread_ = new TraceMonitoringThread(this);
    int err  = tThread_->Start(GetPhysicalChannelListForMeasuring(physicalAIChannels_), expectedMinVoltsIn_,
        expectedMaxVoltsIn_, (float) traceFrequency_, traceAmount_, (int) physicalAIChannels_.size());

    return err;
}


int NIDAQHub::StopTrace()
{
    DestroyMonitoringThread(tThread_);
    tThread_ = new TraceMonitoringThread(this);
    measuringTrace_ = false;

    // Previously a bare "delete mThread_" (see StartTrace()).
    DestroyMonitoringThread(mThread_);
    mThread_ = new InputMonitoringThread(this);
    int err = DEVICE_OK;
    if (physicalAIChannels_.size() > 0)
        err = mThread_->Start(GetPhysicalChannelListForMeasuring(physicalAIChannels_), expectedMinVoltsIn_, expectedMaxVoltsIn_);

    return err;
}


// Called from TraceMonitoringThread::svc(), i.e. on the trace thread itself.
// It must therefore never stop or join tThread_: doing so would make the trace
// thread join itself and deadlock. Only mThread_ may be touched here.
int NIDAQHub::FinishTrace()
{
    measuringTrace_ = false;
    OnPropertiesChanged();

    int err = DEVICE_OK;
    if (physicalAIChannels_.size() > 0)
        err = mThread_->Start(GetPhysicalChannelListForMeasuring(physicalAIChannels_), expectedMinVoltsIn_, expectedMaxVoltsIn_);

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


int NIDAQHub::OnExpectedMaxVoltsIn(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(expectedMaxVoltsIn_);
    }
    else if (eAct == MM::AfterSet)
    {
        double temp_max = 5.0;
        pProp->Get(temp_max);
        expectedMaxVoltsIn_ = (float) temp_max;

        DestroyMonitoringThread(mThread_);
        mThread_ = new InputMonitoringThread(this);
        if (physicalAIChannels_.size() > 1)
            mThread_->Start(GetPhysicalChannelListForMeasuring(physicalAIChannels_), expectedMinVoltsIn_, expectedMaxVoltsIn_);

    }
    return DEVICE_OK;
}


int NIDAQHub::OnExpectedMinVoltsIn(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(expectedMinVoltsIn_);
    }
    else if (eAct == MM::AfterSet)
    {
        double temp_min = -5.0;
        pProp->Get(temp_min);
        expectedMinVoltsIn_ = (float) temp_min;

        DestroyMonitoringThread(mThread_);
        mThread_ = new InputMonitoringThread(this);
        if (physicalAIChannels_.size() > 1)
            mThread_->Start(GetPhysicalChannelListForMeasuring(physicalAIChannels_), expectedMinVoltsIn_, expectedMaxVoltsIn_);
    }
    return DEVICE_OK;
}


int NIDAQHub::OnTraceFrequency(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(traceFrequency_);
    }
    else if (eAct == MM::AfterSet)
    {
        pProp->Get(traceFrequency_);
    }
    return DEVICE_OK;
}


int NIDAQHub::OnTraceAmount(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(traceAmount_);
    }
    else if (eAct == MM::AfterSet)
    {
        pProp->Get(traceAmount_);
    }
    return DEVICE_OK;
}


int NIDAQHub::OnTracePath(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(tracePath_.c_str());
    }
    else if (eAct == MM::AfterSet)
    {
        pProp->Get(tracePath_);
    }
    return DEVICE_OK;
}


int NIDAQHub::OnTraceRunning(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(measuringTrace_? "Running" : "Stopped");
    }
    else if (eAct == MM::AfterSet)
    {
        std::string input;
        pProp->Get(input);
        if (input == "Running")
        {
            StartTrace();
        }
        else if(input == "Stopped")
        {
            StopTrace();
        }
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
   else if (typeid(Tuint) == typeid(uInt16))
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


InputMonitoringThread::InputMonitoringThread(NIDAQHub* hub) :
    stop_(false),
    started_(false),
    aiTask_(NULL)
{
    hub_ = hub;
}


InputMonitoringThread::~InputMonitoringThread()
{
    Stop();
    // Join rather than letting ~MMDeviceThreadBase detach: svc() dereferences
    // hub_, so a detached worker outliving the hub is a use-after-free.
    Join();
}


void InputMonitoringThread::Join()
{
    if (started_)
    {
        // Clear first: if wait() throws anyway, a later Join() must not retry.
        started_ = false;
        wait();
    }
}


// Clear aiTask_ and reset the handle, for use on Start()'s error paths. The
// task was already created there, so simply returning would leak it.
void InputMonitoringThread::ClearTask()
{
    if (aiTask_ != NULL)
    {
        DAQmxClearTask(aiTask_);
        aiTask_ = NULL;
    }
}


int InputMonitoringThread::Start(std::string AIChannelList, float minVal, float maxVal)
{
    stop_ = false;
    int err = DAQmxCreateTask("AnalogInputReadTask", &aiTask_);
    if (err != DEVICE_OK)
    {
        // DAQmx leaves the handle unspecified on failure; do not clear it.
        aiTask_ = NULL;
        return err;
    }

    err = DAQmxCreateAIVoltageChan(aiTask_, AIChannelList.c_str(), "", DAQmx_Val_RSE, minVal, maxVal, DAQmx_Val_Volts, NULL);
    if (err != DEVICE_OK)
    {
        ClearTask();
        return err;
    }

    activate();
    // Only after activate() returns without throwing is there a thread to join.
    started_ = true;
    return DEVICE_OK;
}


int InputMonitoringThread::svc()
{
    while (!stop_)
    {
        float64 values[128] = { 0 };
        int32 amount;
        int err = DAQmxReadAnalogF64(aiTask_, 1, 2.0, DAQmx_Val_GroupByChannel, values, 128, &amount, NULL);
        if (err != DEVICE_OK)
            return err;

        hub_->UpdateAIValues(values, amount);
        CDeviceUtils::SleepMs(100);
    }
    int err = DAQmxClearTask(aiTask_);
    if (err != DEVICE_OK)
        return err;

    return DEVICE_OK;
}


TraceMonitoringThread::TraceMonitoringThread(NIDAQHub* hub) :
    stop_(false),
    started_(false),
    totalAmount_(0),
    numberOfChannels_(0),
    CSVheader_("")
{
    hub_ = hub;
    path_ = hub_->tracePath_ + "/trace_";
}


TraceMonitoringThread::~TraceMonitoringThread()
{
    Stop();
    // See ~InputMonitoringThread().
    Join();
}


void TraceMonitoringThread::Join()
{
    if (started_)
    {
        // Clear first: if wait() throws anyway, a later Join() must not retry.
        started_ = false;
        wait();
    }
}


// See InputMonitoringThread::ClearTask(). DAQmxClearTask() also stops a task
// that was already started, so this is valid after DAQmxStartTask() too.
void TraceMonitoringThread::ClearTask()
{
    if (aiTask_ != NULL)
    {
        DAQmxClearTask(aiTask_);
        aiTask_ = NULL;
    }
}


int TraceMonitoringThread::Start(std::string AIChannelList, float minVal, float maxVal, float frequency, int numberOfSamples, int numberOfChannels)
{
    stop_ = false;
    int err = DAQmxCreateTask("AnalogInputReadTask", &aiTask_);
    if (err != DEVICE_OK)
    {
        // DAQmx leaves the handle unspecified on failure; do not clear it.
        aiTask_ = NULL;
        return err;
    }

    // Every error path below must clear aiTask_, which was created above.
    CSVheader_ = "Time, " + AIChannelList;
    err = DAQmxCreateAIVoltageChan(aiTask_, AIChannelList.c_str(), "", DAQmx_Val_RSE, minVal, maxVal, DAQmx_Val_Volts, NULL);
    if (err != DEVICE_OK)
    {
        ClearTask();
        return err;
    }

    timestep_ = 1 / frequency;
    err = DAQmxSetSampClkRate(aiTask_, frequency);
    if (err != DEVICE_OK)
    {
        ClearTask();
        return err;
    }

    err = DAQmxSetSampQuantSampMode(aiTask_, DAQmx_Val_FiniteSamps);
    if (err != DEVICE_OK)
    {
        ClearTask();
        return err;
    }

    totalAmount_ = numberOfSamples;
    err = DAQmxSetSampQuantSampPerChan(aiTask_, numberOfSamples);
    if (err != DEVICE_OK)
    {
        ClearTask();
        return err;
    }

    err = DAQmxSetSampTimingType(aiTask_, DAQmx_Val_SampClk);
    if (err != DEVICE_OK)
    {
        ClearTask();
        return err;
    }

    path_ += boost::posix_time::to_iso_string(boost::posix_time::second_clock::local_time()) + ".csv";
    numberOfChannels_ = numberOfChannels;

    err = DAQmxStartTask(aiTask_);
    if (err != DEVICE_OK)
    {
        ClearTask();
        return err;
    }

    activate();
    // Only after activate() returns without throwing is there a thread to join.
    started_ = true;
    return DEVICE_OK;
}


int TraceMonitoringThread::svc()
{
    std::ofstream trace(path_);
    if (!trace.is_open()) 
    {
        hub_->LogMessage("Could not open trace");
        return ERR_FAILED_TO_OPEN_TRACE;
    }

    trace << CSVheader_ << std::endl;
    float time = 0;
    float64 values[1024] = { 0 };

    while (!stop_ && totalAmount_ > 0)
    {
        int32 amount = 0;
        int err = DAQmxReadAnalogF64(aiTask_, DAQmx_Val_Auto, -1, DAQmx_Val_GroupByScanNumber, values, 1024, &amount, NULL);
        if (err != DEVICE_OK)
            return err;

        for (int i = 0; i < amount; i++)
        {
            trace << time << ", ";
            for (int j = 0; j < numberOfChannels_; j++)
            {
                trace << values[i * numberOfChannels_ + j];
                if (j < numberOfChannels_-1)
                    trace << ", ";
            }
            trace << std::endl;
            time += timestep_;
        }
        totalAmount_ -= amount;
        CDeviceUtils::SleepMs(100);
    }
    stop_ = true;
    trace.close();
    int err = DAQmxClearTask(aiTask_);
    if (err != DEVICE_OK)
        return err;

    err = hub_->FinishTrace();

    return err;
}