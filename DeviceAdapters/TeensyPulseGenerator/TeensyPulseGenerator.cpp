#include "TeensyPulseGenerator.h"
#include <thread>

#ifdef WIN32
#include <winsock.h>
#else
#include <netinet/in.h> // #include <arpa/inet.h>
#endif

const char* g_RunUntilStopped = "Run_Until_Stopped";
const char* g_NrPulses = "Number_of_Pulses";

const unsigned char cmd_version = 0;
const unsigned char cmd_start = 1;
const unsigned char cmd_stop = 2;
const unsigned char cmd_interval = 3; // interval in microseconds
const unsigned char cmd_pulse_duration = 4; // in microsconds
const unsigned char cmd_wait_for_input = 5;
const unsigned char cmd_number_of_pulses = 6;


TeensyPulseGenerator::TeensyPulseGenerator() :
    initialized_(false),
    port_(""),
    interval_(100),      // Default 100ms interval
    pulseDuration_(10),   // Default 10ms pulse
    triggerMode_(false),    // Default trigger mode on
    running_(false),        // Not running initially
    runUntilStopped_(true), // Keep on pulsing until stopped
    version_(0),            // version of the firmware
    nrPulses_(1)            // Number of pulses, only relevant if !runUntilStopped_ 

{
   InitializeDefaultErrorMessages();

   // Add custom error messages
   SetErrorText(ERR_PORT_OPEN_FAILED, "Failed to open serial port");
   SetErrorText(ERR_COMMUNICATION, "Communication error with device");

   // Create serial port property
   CPropertyAction* pAct = new CPropertyAction(this, &TeensyPulseGenerator::OnPort);
   CreateProperty(MM::g_Keyword_Port, "Undefined", MM::String, false, pAct, true);
}

TeensyPulseGenerator::~TeensyPulseGenerator()
{
    Shutdown();
    // option: inter
}

void TeensyPulseGenerator::GetName(char* name) const
{
    CDeviceUtils::CopyLimitedString(name, "TeensyPulseGenerator");
}

int TeensyPulseGenerator::Initialize()
{
   if (initialized_)
       return DEVICE_OK;

    // Ensure port is set
   if (port_.empty())
       return ERR_NO_PORT_SET;

   PurgeComPort(port_.c_str());

   // get firmware version
   int ret = SendCommand(cmd_version, 0); 
   if (ret != DEVICE_OK)
      return ret;
   ret = GetResponse(0, version_);
   if (ret != DEVICE_OK)
      return ret;
   // TODO: check we can work with this firmware
   std::ostringstream os;
   os << version_;
   CreateProperty("Firmware-version", os.str().c_str(), MM::String, true);


   // Create properties

   // Interval property
   ret = Enquire(cmd_interval);
   if (ret != DEVICE_OK)
      return ret;
   uint32_t interval;
   ret = GetResponse(cmd_interval, interval);
   if (ret != DEVICE_OK)
      return ret;
   interval_ = interval / 1000.0;
   CPropertyAction* pAct = new CPropertyAction(this, &TeensyPulseGenerator::OnInterval);
   CreateFloatProperty("Interval-ms", interval_, false, pAct);

   // Pulse Duration property
   ret = Enquire(cmd_pulse_duration);
   if (ret != DEVICE_OK)
      return ret;
   uint32_t pulseDuration;
   ret = GetResponse(cmd_pulse_duration, pulseDuration);
   if (ret != DEVICE_OK)
      return ret;
   pulseDuration_ = pulseDuration / 1000.0;
   pAct = new CPropertyAction(this, &TeensyPulseGenerator::OnPulseDuration);
   CreateFloatProperty("PulseDuration-ms", pulseDuration_, false, pAct);

   // Trigger Mode property
   ret = Enquire(cmd_wait_for_input);
   if (ret != DEVICE_OK)
      return ret;
   uint32_t waitForInput;
   ret = GetResponse(cmd_wait_for_input, waitForInput);
   if (ret != DEVICE_OK)
      return ret;
   triggerMode_ = (bool) waitForInput;
   pAct = new CPropertyAction(this, &TeensyPulseGenerator::OnTriggerMode);
   CreateProperty("TriggerMode", triggerMode_ ? "On" : "Off", MM::String, false, pAct);
   AddAllowedValue("TriggerMode", "Off");
   AddAllowedValue("TriggerMode", "On");

   // Run until Stopped property
   ret = Enquire(cmd_number_of_pulses);
   if (ret != DEVICE_OK)
      return ret;
   ret = GetResponse(cmd_number_of_pulses, nrPulses_);
   if (ret != DEVICE_OK)
      return ret;
   runUntilStopped_ = nrPulses_ == 0;
   pAct = new CPropertyAction(this, &TeensyPulseGenerator::OnRunUntilStopped);
   CreateProperty(g_RunUntilStopped, runUntilStopped_ ? "On" : "Off", MM::String, false, pAct);
   AddAllowedValue(g_RunUntilStopped, "Off");
   AddAllowedValue(g_RunUntilStopped, "On");

   // Sets the Number of pulses.  Will only be used if runUntilStopped == false
   pAct = new CPropertyAction(this, &TeensyPulseGenerator::OnNrPulses);
   CreateIntegerProperty(g_NrPulses, nrPulses_, false, pAct);


   // State (Start/Stop) property
   pAct = new CPropertyAction(this, &TeensyPulseGenerator::OnState);
   CreateProperty("State", "Stop", MM::String, false, pAct);
   AddAllowedValue("State", "Stop");
   AddAllowedValue("State", "Start");

   // State (Start/Stop) property
   pAct = new CPropertyAction(this, &TeensyPulseGenerator::OnStatus);
   CreateProperty("Status", running_ ? "Active" : "Idle", MM::String, true, pAct);

   initialized_ = true;
   return DEVICE_OK;
}

int TeensyPulseGenerator::Shutdown()
{
    if (port_ != "")
    {
       // Ensure device is stopped
       const std::lock_guard<std::mutex> lock(mutex_);
       SendCommand(cmd_stop, 0);  // Stop command
    }
    initialized_ = false;
    return DEVICE_OK;
}

bool TeensyPulseGenerator::Busy()
{
    return false;  // This device doesn't have a concept of "busy"
}

int TeensyPulseGenerator::SendCommand(uint8_t cmd, uint32_t param)
{
   // This should not be needed, but just in case:
   PurgeComPort(port_.c_str());

    // Prepare buffer
   const unsigned int buflen = 5;
   unsigned char buffer[buflen];
   buffer[0] = cmd;
   // This needs work when we have a Big Endian host, the Teensy is little endian
   alignas(uint32_t) char alignedByteArray[4];
   std::memcpy(alignedByteArray, &param, 4);

   // For commands that require a parameter, convert to little-endian
   buffer[1] = alignedByteArray[0];
   buffer[2] = alignedByteArray[1];
   buffer[3] = alignedByteArray[2];
   buffer[4] = alignedByteArray[3];

   // Send 5-byte command
   return WriteToComPort(port_.c_str(), buffer, buflen);
}

int TeensyPulseGenerator::Enquire(uint8_t cmd)
{
   const unsigned int buflen = 2;
   unsigned char buffer[buflen];
   buffer[0] = 255;
   buffer[1] = cmd;

   return WriteToComPort(port_.c_str(), buffer, buflen);
}

int TeensyPulseGenerator::GetResponse(uint8_t cmd, uint32_t& param) 
{ 
   unsigned char buf[1] = { 0 };
   unsigned long read = 0;
   MM::TimeoutMs timeout = MM::TimeoutMs(GetCurrentMMTime(), 1000);
   while (read == 0) {
      if (timeout.expired(GetCurrentMMTime()))
      {
         return ERR_COMMUNICATION;
      }
      ReadFromComPort(port_.c_str(), buf, 1, read);
   }
   if (read == 1 && buf[0] == cmd)
   {
      unsigned char buf2[4] = { 0, 0, 0, 0 };
      read = 0;
      unsigned long tmpRead = 0;
      while (read < 4)
      {
         if (timeout.expired(GetCurrentMMTime()))
         {
            return ERR_COMMUNICATION;
         }
         ReadFromComPort(port_.c_str(), &buf2[read], 4 - read, tmpRead);
         read += tmpRead;
      }
      alignas(uint32_t) char alignedByteArray[4];
      std::memcpy(alignedByteArray, buf2, sizeof(buf2));

      // This needs change on a Big endian host (PC and Teensy are both little endian).
      param =  *(reinterpret_cast<uint32_t*>(alignedByteArray));

      std::ostringstream os;
      os << "Read: " << param;
      GetCoreCallback()->LogMessage(this, os.str().c_str(), true);
    } 
    else
    {
       return ERR_COMMUNICATION;
    }
   return DEVICE_OK;
}


int TeensyPulseGenerator::OnPort(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(port_.c_str());
    }
    else if (eAct == MM::AfterSet)
    {
        pProp->Get(port_);
    }
    return DEVICE_OK;
}

int TeensyPulseGenerator::OnInterval(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(interval_);
    }
    else if (eAct == MM::AfterSet)
    {
       pProp->Get(interval_);
        
       // Send interval command if initialized
       if (initialized_)
       {
          const std::lock_guard<std::mutex> lock(mutex_);
          PurgeComPort(port_.c_str());
          uint32_t interval = static_cast<uint32_t> (interval_ * 1000.0);
          int ret = SendCommand(cmd_interval, interval);
          if (ret != DEVICE_OK)
             return ret;
          uint32_t parm;
          ret = GetResponse(cmd_interval, parm);
          if (ret != DEVICE_OK)
             return ret;
          if (parm != interval)
          {
            GetCoreCallback()->LogMessage(this, "Interval sent not the same as interval echoed back", false);
            return ERR_COMMUNICATION;
          }
       }
    }
    return DEVICE_OK;
}


int TeensyPulseGenerator::OnPulseDuration(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(pulseDuration_);
    }
    else if (eAct == MM::AfterSet)
    {
       pProp->Get(pulseDuration_);
        
       // Send pulse duration command if initialized
       if (initialized_)
       {
          uint32_t pulseDurationUs =  static_cast<uint32_t>(pulseDuration_ * 1000.0);

          const std::lock_guard<std::mutex> lock(mutex_);
          int ret = SendCommand(cmd_pulse_duration, pulseDurationUs);
          if (ret != DEVICE_OK)
             return ret;
          uint32_t param;
          ret = GetResponse(cmd_pulse_duration, param);
          if (ret != DEVICE_OK)
             return ret;
          if (param != pulseDurationUs)
          {
            GetCoreCallback()->LogMessage(this, "PulseDuration sent not the same as pulseDuration echoed back", false);
            return ERR_COMMUNICATION;

          }
       }
   }
   return DEVICE_OK;
}


int TeensyPulseGenerator::OnTriggerMode(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(triggerMode_ ? "On" : "Off");
   }
   else if (eAct == MM::AfterSet)
   {
      std::string triggerModeStr;
      pProp->Get(triggerModeStr);
      triggerMode_ = (triggerModeStr == "On");
        
      // Send trigger mode command if initialized
      if (initialized_)
      {
         uint32_t sp = triggerMode_ ? 1 : 0;
         const std::lock_guard<std::mutex> lock(mutex_);
         int ret = SendCommand(cmd_wait_for_input, sp);
         if (ret != DEVICE_OK)
            return ret;
         uint32_t param;
         ret = GetResponse(cmd_wait_for_input, param);
         if (param != sp)
         {
            GetCoreCallback()->LogMessage(this, "Triggermode sent not the same as triggermode echoed back", false);
            return ERR_COMMUNICATION;
         }
      }
   }
   return DEVICE_OK;
}

int TeensyPulseGenerator::OnRunUntilStopped(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(runUntilStopped_ ? "On" : "Off");
   }
   else if (eAct == MM::AfterSet)
   {
      std::string stateStr;
      pProp->Get(stateStr);
      if (stateStr == "On" && !runUntilStopped_)
      {
         const std::lock_guard<std::mutex> lock(mutex_);
         int ret = SendCommand(cmd_number_of_pulses, static_cast<uint32_t> (0));
         if (ret != DEVICE_OK)
            return ret;
         uint32_t param;
         ret = GetResponse(cmd_number_of_pulses, param);
         if (param != 0)
         {
            GetCoreCallback()->LogMessage(this, "NrPulses sent (0) not the same as number of pulses received", false);
            return ERR_COMMUNICATION;

         }
         runUntilStopped_ = true;
      }
      else if (stateStr == "Off" && runUntilStopped_)
      {
         unsigned char cmd = cmd_number_of_pulses;
         const std::lock_guard<std::mutex> lock(mutex_);
         int ret = SendCommand(cmd, nrPulses_);
         if (ret != DEVICE_OK)
            return ret;
         uint32_t param;
         ret = GetResponse(cmd, param);
         if (nrPulses_ != param) {
            GetCoreCallback()->LogMessage(this, "NrPulses sent not the same as number of pulses received", false);
            return ERR_COMMUNICATION;
         }
         runUntilStopped_ = false;
      }
   }
   return DEVICE_OK;
}


int TeensyPulseGenerator::OnNrPulses(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set((long) nrPulses_);
   }
   else if (eAct == MM::AfterSet)
   {
      long nrPulses;
      pProp->Get(nrPulses);
      if ((unsigned) nrPulses != nrPulses_)
      {

         nrPulses_ = (unsigned) nrPulses;
         if (!runUntilStopped_)
         {
            const std::lock_guard<std::mutex> lock(mutex_);
            int ret = SendCommand(cmd_number_of_pulses, nrPulses_);
            if (ret != DEVICE_OK)
               return ret;
            uint32_t param;
            ret = GetResponse(cmd_number_of_pulses, param);
            if (nrPulses_ != param) {
               GetCoreCallback()->LogMessage(this, "NrPulses sent not the same as number of pulses received", false);
               return ERR_COMMUNICATION;
            }
         }
      }
   }
   return DEVICE_OK;
}

void TeensyPulseGenerator::CheckStatus()
{
               long microSecondWait = (long) ((nrPulses_ - 1) * interval_ + pulseDuration_);
               long wait = (microSecondWait / 1000);
   // TODO: this could be a long wait.
   // Break it up and check whether the destructor has been called so that we will not 
   // delay destruction of this object.
   Sleep(wait);
   while (running_)
   {
      {
         const std::lock_guard<std::mutex> lock(mutex_);
         Enquire(cmd_start);
         uint32_t response;
         GetResponse(cmd_start, response);
         running_ = (bool)response;
      }
      Sleep((long) (interval_ / 1000));
   }
   GetCoreCallback()->OnPropertyChanged(this, "Status", "Idle");
}

int TeensyPulseGenerator::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(running_ ? "Start" : "Stop");
    }
    else if (eAct == MM::AfterSet)
    {
        std::string stateStr;
        pProp->Get(stateStr);
        
        if (stateStr == "Start" && !running_)
        {
           {
              const std::lock_guard<std::mutex> lock(mutex_);
              int ret = SendCommand(cmd_start, 0); // Start command
              if (ret != DEVICE_OK)
                 return ret;
              uint32_t param;
              ret = GetResponse(cmd_start, param);
              if (ret != DEVICE_OK)
                 return ret;
              if (param != 1)
                 return ERR_COMMUNICATION;
           }
           running_ = true;
           if (!runUntilStopped_)
           {
              // Start a thread that waits for the estimated duraion of the pulse train, then checks 
              // whether the Teensy is done.
               std::function<void()> func = [&]() {
                  return this->CheckStatus();
               };
               singleThread_.enqueue(func);
             // std::function<void()> func = (TeensyPulseGenerator::CheckStatus, this);

              //std::thread t (&TeensyPulseGenerator::CheckStatus, this, wait);
              // Note: we will crash if the destructor of this is called while the thread is running
              // need to come up with something cleaner..
              //t.detach();
           }
        }
        else if (stateStr == "Stop" && running_)
        {
           const std::lock_guard<std::mutex> lock(mutex_);
           int ret = SendCommand(cmd_stop, 0); // Stop command
           if (ret != DEVICE_OK)
              return ret;
           uint32_t param;
           ret = GetResponse(cmd_stop, param);
           if (ret != DEVICE_OK)
              return ret;
           running_ = false;
           // param holds the number of pulses
        }
    }
    return DEVICE_OK;
}

int TeensyPulseGenerator::OnStatus(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(running_ ? "Active" : "Idle");
   }
   return DEVICE_OK;
}


///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////

MODULE_API void InitializeModuleData()
{
    RegisterDevice("TeensyPulseGenerator", MM::GenericDevice, "Teensy Pulse Generator");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
    if (deviceName == 0)
        return 0;

    if (strcmp(deviceName, "TeensyPulseGenerator") == 0)
    {
        return new TeensyPulseGenerator();
    }

    return 0;
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
    delete pDevice;
}
