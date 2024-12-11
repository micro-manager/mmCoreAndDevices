#include "TeensyPulseGenerator.h"

#ifdef WIN32
#include <winsock.h>
#else
#include <netinet/in.h> // #include <arpa/inet.h>
#endif

const char* g_RunUntilStopped = "Run_Until_Stopped";
const char* g_NrPulses = "Number_of_Pulses";

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

   // Open serial port
   PurgeComPort(port_.c_str());


   // Create properties

   // Interval property
   CPropertyAction* pAct = new CPropertyAction(this, &TeensyPulseGenerator::OnInterval);
   CreateFloatProperty("Interval-ms", interval_, false, pAct);
   //SetPropertyLimits("Interval-ms", 1, 1000000);

   // Pulse Duration property
   pAct = new CPropertyAction(this, &TeensyPulseGenerator::OnPulseDuration);
   CreateFloatProperty("PulseDuration-ms", pulseDuration_, false, pAct);
   //SetPropertyLimits("PulseDuration-us", 1, 100000);

   // Trigger Mode property
   pAct = new CPropertyAction(this, &TeensyPulseGenerator::OnTriggerMode);
   CreateProperty("TriggerMode", triggerMode_ ? "On" : "Off", MM::String, false, pAct);
   AddAllowedValue("TriggerMode", "Off");
   AddAllowedValue("TriggerMode", "On");

   // Run until Stopped property
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

   int ret = SendCommand(0, 0); 
   if (ret != DEVICE_OK)
      return ret;
   ret = GetResponse(0, version_);
   if (ret != DEVICE_OK)
      return ret;
   std::ostringstream os;
   os << version_;
   CreateProperty("Firmware-version", os.str().c_str(), MM::String, true);

   ret = UpdateStatus();
   if (ret != DEVICE_OK)
      return ret;

   initialized_ = true;
   return DEVICE_OK;
}

int TeensyPulseGenerator::Shutdown()
{
    if (port_ != "")
    {
       // Ensure device is stopped
       SendCommand(2, 0);  // Stop command
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
          PurgeComPort(port_.c_str());
          uint32_t interval = static_cast<uint32_t> (interval_ * 1000.0);
          unsigned char cmd = 3;
          int ret = SendCommand(cmd, interval);
          if (ret != DEVICE_OK)
             return ret;
          uint32_t parm;
          ret = GetResponse(cmd, parm);
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
          unsigned char cmd = 4;

          int ret = SendCommand(cmd, pulseDurationUs);
          if (ret != DEVICE_OK)
             return ret;
          uint32_t param;
          ret = GetResponse(cmd, param);
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
         unsigned char cmd = 5;
         uint32_t sp = triggerMode_ ? 1 : 0;
         int ret = SendCommand(cmd, sp);
         if (ret != DEVICE_OK)
            return ret;
         uint32_t param;
         ret = GetResponse(cmd, param);
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
         unsigned char cmd = 6;
         int ret = SendCommand(cmd, static_cast<uint32_t> (0));
         if (ret != DEVICE_OK)
            return ret;
         uint32_t param;
         ret = GetResponse(cmd, param);
         if (param != 0)
         {
            GetCoreCallback()->LogMessage(this, "NrPulses sent (0) not the same as number of pulses received", false);
            return ERR_COMMUNICATION;

         }
         runUntilStopped_ = true;
      }
      else if (stateStr == "Off" && runUntilStopped_)
      {
         unsigned char cmd = 6;
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
            unsigned char cmd = 6;
            int ret = SendCommand(cmd, nrPulses_);
            if (ret != DEVICE_OK)
               return ret;
            uint32_t param;
            ret = GetResponse(cmd, param);
            if (nrPulses_ != param) {
               GetCoreCallback()->LogMessage(this, "NrPulses sent not the same as number of pulses received", false);
               return ERR_COMMUNICATION;
            }
         }
      }
   }
   return DEVICE_OK;
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
           unsigned char cmd = 1;
           int ret = SendCommand(cmd, 0); // Start command
           if (ret != DEVICE_OK)
              return ret;
           uint32_t param;
           ret = GetResponse(cmd, param);
           if (ret != DEVICE_OK)
              return ret;
           if (param != 0)
              return ERR_COMMUNICATION;
           running_ = true;
        }
        else if (stateStr == "Stop" && running_)
        {
           unsigned char cmd = 2;
           int ret = SendCommand(cmd, 0); // Stop command
           if (ret != DEVICE_OK)
              return ret;
           uint32_t param;
           ret = GetResponse(cmd, param);
           if (ret != DEVICE_OK)
              return ret;
           running_ = false;
           // param holds the number of pulses
        }
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
