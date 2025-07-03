#include "TeensyPulseGenerator.h"
#include "CameraPulser.h"
#include <thread>

#ifdef WIN32
#include <winsock.h>
#else
#include <netinet/in.h> // #include <arpa/inet.h>
#endif

const char* g_RunUntilStopped = "Run_Until_Stopped";
const char* g_NrPulses = "Number_of_Pulses";
const char* g_NrPulsesCounted = "Number_of_Actual_Pulses";

const char* g_TeensyPulseGenerator = "TeensyPulseGenerator";

TeensyPulseGenerator::TeensyPulseGenerator() :
   initialized_(false),
   port_(""),
   teensyCom_(0),
   interval_(100),      // Default 100ms interval
   pulseDuration_(10),   // Default 10ms pulse
   triggerMode_(false),    // Default trigger mode on
   running_(false),        // Not running initially
   runUntilStopped_(true), // Keep on pulsing until stopped
   version_(0),            // version of the firmware
   nrPulses_(1),            // Number of pulses, only relevant if !runUntilStopped_ 
   nrPulsesCounted_(0)
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
    CDeviceUtils::CopyLimitedString(name, g_TeensyPulseGenerator);
}

int TeensyPulseGenerator::Initialize()
{
   if (initialized_)
       return DEVICE_OK;

    // Ensure port is set
   if (port_.empty())
       return ERR_NO_PORT_SET;

   teensyCom_ = new TeensyCom(GetCoreCallback(), (MM::Device*) this, port_.c_str());

   // Create properties

   // TODO: check we can work with this firmware
   int ret = teensyCom_->GetVersion(version_);
   if (ret != DEVICE_OK)
      return ret;
   std::ostringstream os;
   os << version_;
   CreateProperty("Firmware-version", os.str().c_str(), MM::String, true);

   // Interval property
   uint32_t interval;
   ret = teensyCom_->GetInterval(interval);
   if (ret != DEVICE_OK)
      return ret;
   interval_ = interval / 1000.0;
   CPropertyAction* pAct = new CPropertyAction(this, &TeensyPulseGenerator::OnInterval);
   CreateFloatProperty("Interval-ms", interval_, false, pAct);

   // Pulse Duration property
   uint32_t pulseDuration;
   ret = teensyCom_->GetPulseDuration(pulseDuration);
   if (ret != DEVICE_OK)
      return ret;
   pulseDuration_ = pulseDuration / 1000.0;
   pAct = new CPropertyAction(this, &TeensyPulseGenerator::OnPulseDuration);
   CreateFloatProperty("PulseDuration-ms", pulseDuration_, false, pAct);

   // Trigger Mode property
   uint32_t waitForInput;
   ret = teensyCom_->GetWaitForInput(waitForInput);
   if (ret != DEVICE_OK)
      return ret;
   triggerMode_ = (bool) waitForInput;
   pAct = new CPropertyAction(this, &TeensyPulseGenerator::OnTriggerMode);
   CreateProperty("TriggerMode", triggerMode_ ? "On" : "Off", MM::String, false, pAct);
   AddAllowedValue("TriggerMode", "Off");
   AddAllowedValue("TriggerMode", "On");

   // Run until Stopped property
   ret = teensyCom_->GetNumberOfPulses(nrPulses_);
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

   // Read only property to report number of pulses actually send out
   pAct = new CPropertyAction(this, &TeensyPulseGenerator::OnNrPulsesCounted);
   CreateIntegerProperty(g_NrPulsesCounted, nrPulsesCounted_, true, pAct);

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
       if (teensyCom_ != 0) {
          uint32_t param;
          teensyCom_->SetStop(param);
          delete (teensyCom_);
          teensyCom_ = 0; // is this automatic after delete?
       }
    }
    initialized_ = false;
    return DEVICE_OK;
}

bool TeensyPulseGenerator::Busy()
{
    return false;  // This device doesn't have a concept of "busy"
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
          uint32_t interval = static_cast<uint32_t> (interval_ * 1000.0);
          uint32_t parm;
          int ret = teensyCom_->SetInterval(interval, parm);
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
          uint32_t param;
          int ret = teensyCom_->SetPulseDuration(pulseDurationUs, param);
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
         uint32_t param;
         int ret = teensyCom_->SetWaitForInput(sp, param);
         if (ret != DEVICE_OK)
            return ret;
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
         uint32_t param;
         int ret = teensyCom_->SetNumberOfPulses(static_cast<uint32_t> (0), param);
         if (ret != DEVICE_OK)
            return ret;
         if (param != 0)
         {
            GetCoreCallback()->LogMessage(this, "NrPulses sent (0) not the same as number of pulses received", false);
            return ERR_COMMUNICATION;
         }
         runUntilStopped_ = true;
      }
      else if (stateStr == "Off" && runUntilStopped_)
      {
         uint32_t param;
         int ret = teensyCom_->SetNumberOfPulses(nrPulses_, param);
         if (ret != DEVICE_OK)
            return ret;
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
            uint32_t param;
            int ret = teensyCom_->SetNumberOfPulses(nrPulses_, param);
            if (ret != DEVICE_OK)
               return ret;
            if (nrPulses_ != param) {
               GetCoreCallback()->LogMessage(this, "NrPulses sent not the same as number of pulses received", false);
               return ERR_COMMUNICATION;
            }
         }
      }
   }
   return DEVICE_OK;
}

/**
 * Called from a seperate thread.
 */
void TeensyPulseGenerator::CheckStatus()
{
   long milliSecondWait = (long) ((nrPulses_ - 1) * interval_ + pulseDuration_);
   long waited = 0;
   long waitDuration = 500; // arbitrary wait
   // Break the wait up and check whether the destructor has been called so that we will not 
   // delay destruction of this object.
   while (waited < milliSecondWait)
   {
      if (!initialized_)
         return;
      Sleep(waitDuration);
      waited += waitDuration;
   }
   while (running_)
   {
      {
         uint32_t response;
         // ignor errors since we can not easily handle them
         // Think about reporting errors to logs
         teensyCom_->GetRunningStatus(response);
         running_ = (bool) response;
         if (!running_)
            teensyCom_->SetStop(nrPulsesCounted_);
      }
      Sleep((long) (interval_));
   }
   GetCoreCallback()->OnPropertyChanged(this, "Status", "Idle");
   char myString[10] = ""; // 10 chars is maximum for uint32_t
   sprintf(myString, "%d", (long) nrPulsesCounted_);
   GetCoreCallback()->OnPropertyChanged(this, g_NrPulsesCounted, myString);
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
              uint32_t param;
              int ret = teensyCom_->SetStart(param);
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
           }
        }
        else if (stateStr == "Stop" && running_)
        {
           int ret = teensyCom_->SetStop(nrPulsesCounted_); // Stop command
           if (ret != DEVICE_OK)
              return ret;
           running_ = false;
           char myString[10] = ""; // 10 chars is maximum for uint32_t
           sprintf(myString, "%d", (long) nrPulsesCounted_);
           GetCoreCallback()->OnPropertyChanged(this, g_NrPulsesCounted, myString);
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

int TeensyPulseGenerator::OnNrPulsesCounted(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set((long) nrPulsesCounted_);
   }
   return DEVICE_OK;
}
