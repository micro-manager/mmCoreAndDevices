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

#pragma once

#include "DeviceBase.h"

#include "NIDAQmx.h"

#include <boost/lexical_cast.hpp>
#include <boost/utility.hpp>

#include <string>
#include <vector>


// Mix-in class for error code handling.
template <typename TDevice>
class ErrorTranslator
{
public:
   explicit ErrorTranslator(int minCode, int maxCode,
         void (TDevice::*setCodeFunc)(int, const char*)) :
      minErrorCode_(minCode),
      maxErrorCode_(maxCode),
      nextErrorCode_(minCode),
      setCodeFunc_(setCodeFunc)
   {}

   int NewErrorCode(const std::string& msg)
   {
      if (nextErrorCode_ > maxErrorCode_)
         nextErrorCode_ = minErrorCode_;
      int code = nextErrorCode_++;

      (static_cast<TDevice*>(this)->*setCodeFunc_)(code, msg.c_str());
      return code;
   }

   int TranslateNIError(int32 nierr)
   {
      char buf[1024];
      if (DAQmxGetErrorString(nierr, buf, sizeof(buf)))
         return NewErrorCode("[Cannot get DAQmx error message]");
      return NewErrorCode(buf);
   }

private:
   void (TDevice::*setCodeFunc_)(int, const char*);
   int minErrorCode_;
   int maxErrorCode_;
   int nextErrorCode_;
};

class NIDAQHub;


// template class to deal with 8 pin and 32 pin DO ports without
// too much code duplication
template <typename Tuint>
class NIDAQDOHub
{
public:
   NIDAQDOHub(NIDAQHub* hub) { hub_ = hub;  }
   int StartDOSequenceForPort(const std::string& port,
      const std::vector<Tuint> sequence);
   virtual int StopDOSequenceForPort(const std::string& port);

private:
   int AddDOPortToSequencing(const std::string& port,
      const std::vector<Tuint> sequence);
   void RemoveDOPortFromSequencing(const std::string& port);
   int StartDOSequencingTask();

   template<typename T> 
   int DaqmxWriteDigital(TaskHandle doTask_, int32 samplesPerChar, const uInt8* samples, int32* numWritten) {}

   template<uInt8>
      int DaqmxWriteDigital(TaskHandle doTask_, int32 samplesPerChar, const uInt8* samples, int32* numWritten) 
      {
         return DAQmxWriteDigitalU8(doTask_, samplesPerChan,
            false, DAQmx_Val_WaitInfinitely, DAQmx_Val_GroupByChannel,
            samples, numWritten, NULL);
      }

   template<uInt32>
      int DaqmxWriteDigital(TaskHandle doTask_, int32 samplesPerChar, const uInt32* samples, int32* numWritten)
      {
         return DAQmxWriteDigitalU32(doTask_, samplesPerChan,
            false, DAQmx_Val_WaitInfinitely, DAQmx_Val_GroupByChannel,
            samples, numWritten, NULL);
      }


   NIDAQHub* hub_;
   TaskHandle doTask_;
   std::vector<std::string> physicalDOChannels_;
   std::vector<std::vector<Tuint>> doChannelSequences_;
};


// A hub-peripheral device set for driving multiple analog output ports,
// possibly with hardware-triggered sequencing using a shared trigger input.
class NIDAQHub : public HubBase<NIDAQHub>,
   public ErrorTranslator<NIDAQHub>,
   boost::noncopyable
{
   friend NIDAQDOHub<uInt32>;
   friend NIDAQDOHub<uInt8>;
public:
   NIDAQHub();
   virtual ~NIDAQHub();

   virtual int Initialize();
   virtual int Shutdown();

   virtual void GetName(char* name) const;
   virtual bool Busy() { return false; }

   virtual int DetectInstalledDevices();

   // Interface for individual ports
   virtual int GetVoltageLimits(double& minVolts, double& maxVolts);
   virtual int StartAOSequenceForPort(const std::string& port,
      const std::vector<double> sequence);
   virtual int StopAOSequenceForPort(const std::string& port);


   virtual int IsSequencingEnabled(bool& flag) const;
   virtual int GetSequenceMaxLength(long& maxLength) const;

   NIDAQDOHub<uInt8> * getDOHub8()  {return doHub8_; }
   NIDAQDOHub<uInt32> * getDOHub32() { return doHub32_; }

   int StopTask(TaskHandle& task);

private:
   int AddAOPortToSequencing(const std::string& port,
      const std::vector<double> sequence);
   void RemoveAOPortFromSequencing(const std::string& port);

   int AddDOPortToSequencing8(const std::string& port,
      const std::vector<uInt8> sequence);
   int AddDOPortToSequencing32(const std::string& port,
      const std::vector<uInt32> sequence);
   void RemoveDOPortFromSequencing(const std::string& port);

   int GetVoltageRangeForDevice(const std::string& device,
      double& minVolts, double& maxVolts);
   std::vector<std::string> GetTriggerPortsForDevice(
      const std::string& device);
   std::vector<std::string> GetAnalogPortsForDevice(
      const std::string& device);
   std::vector<std::string> GetDigitalPortsForDevice(
       const std::string& device);
   std::string GetPhysicalChannelListForSequencing(std::vector<std::string> channels) const;
   template<typename T> int GetLCMSamplesPerChannel(size_t& seqLen, std::vector<std::vector<T>>) const;
   template<typename T> void GetLCMSequence(T* buffer, std::vector<std::vector<T>> sequences) const;

   int StartAOSequencingTask();


   // Action handlers
   int OnDevice(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnMaxSequenceLength(MM::PropertyBase* pProp, MM::ActionType eAct);

   int OnSequencingEnabled(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnTriggerInputPort(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnSampleRate(MM::PropertyBase* pProp, MM::ActionType eAct);


   bool initialized_;
   size_t maxSequenceLength_;
   bool sequencingEnabled_;

   std::string niDeviceName_;
   std::string niTriggerPort_;

   double minVolts_; // Min possible for device
   double maxVolts_; // Max possible for device
   double sampleRateHz_;

   TaskHandle aoTask_;   
   TaskHandle doTask_;

   NIDAQDOHub<uInt8> * doHub8_;
   NIDAQDOHub<uInt32> * doHub32_;

   // "Loaded" sequences for each channel
   // Invariant: physicalChannels_.size() == channelSequences_.size()
   std::vector<std::string> physicalAOChannels_; // Invariant: all unique
   std::vector<std::vector<double>> aoChannelSequences_;

   std::vector<std::string> physicalDOChannels_;

   std::vector<std::vector<uInt8>> doChannelSequences8_;
   std::vector<std::vector<uInt32>> doChannelSequences32_;
};


class MultiAnalogOutPort : public CSignalIOBase<MultiAnalogOutPort>,
   ErrorTranslator<MultiAnalogOutPort>,
   boost::noncopyable
{
public:
   MultiAnalogOutPort(const std::string& port);
   virtual ~MultiAnalogOutPort();

   virtual int Initialize();
   virtual int Shutdown();

   virtual void GetName(char* name) const;
   virtual bool Busy() { return false; }

   virtual int SetGateOpen(bool open);
   virtual int GetGateOpen(bool& open);
   virtual int SetSignal(double volts);
   virtual int GetSignal(double& /* volts */) { return DEVICE_UNSUPPORTED_COMMAND; }
   virtual int GetLimits(double& minVolts, double& maxVolts);

   virtual int IsDASequenceable(bool& isSequenceable) const;
   virtual int GetDASequenceMaxLength(long& maxLength) const;
   virtual int StartDASequence();
   virtual int StopDASequence();
   virtual int ClearDASequence();
   virtual int AddToDASequence(double);
   virtual int SendDASequence();

private:
   // Pre-init property action handlers
   int OnMinVolts(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnMaxVolts(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnSequenceable(MM::PropertyBase* pProp, MM::ActionType eAct);

   // Post-init property action handlers
   int OnVoltage(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
   NIDAQHub* GetAOHub() const
   { return static_cast<NIDAQHub*>(GetParentHub()); }
   int TranslateHubError(int err);
   int StartOnDemandTask(double voltage);
   int StopTask();

private:
   const std::string niPort_;

   bool initialized_;

   bool gateOpen_;
   double gatedVoltage_;
   bool sequenceRunning_;

   double minVolts_; // User-selected for this port
   double maxVolts_; // User-selected for this port
   bool neverSequenceable_;

   TaskHandle task_;

   std::vector<double> unsentSequence_;
   std::vector<double> sentSequence_; // Pretend "sent" to device
};

class DigitalOutputPort : public CStateDeviceBase<DigitalOutputPort>, 
    ErrorTranslator<DigitalOutputPort>
{
public:
    DigitalOutputPort(const std::string& port);
    virtual ~DigitalOutputPort();

    virtual int Initialize();
    virtual int Shutdown();

    virtual void GetName(char* name) const;
    virtual bool Busy() { return false; }

    virtual unsigned long GetNumberOfPositions()const { return numPos_; }

    // action interface
    // ----------------
    int OnState(MM::PropertyBase* pProp, MM::ActionType eAct);
    //int OnBlanking(MM::PropertyBase* pProp, MM::ActionType eAct);
    //int OnBlankingTriggerDirection(MM::PropertyBase* pProp, MM::ActionType eAct);

    //int OnSequence(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
    int OnSequenceable(MM::PropertyBase* pProp, MM::ActionType eAct);
    NIDAQHub* GetHub() const
    {
       return static_cast<NIDAQHub*>(GetParentHub());
    }
    int StartOnDemandTask(long state);
    int StopTask();

    std::string niPort_;
    bool initialized_;
    bool sequenceRunning_;
    long numPos_;
    uInt32 portWidth_;
    bool neverSequenceable_;

    // this can probably be done more elegantly using templates
    std::vector<uInt8> sequence8_;
    std::vector<uInt32> sequence32_;

    TaskHandle task_;
};


