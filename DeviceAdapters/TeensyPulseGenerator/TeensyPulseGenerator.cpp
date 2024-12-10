#include "TeensyPulseGenerator.h"


TeensyPulseGenerator::TeensyPulseGenerator() :
    initialized_(false),
    port_(""),
    interval_(10000),     // Default 10ms interval
    pulseDuration_(500),  // Default 500us pulse
    triggerMode_(true),   // Default trigger mode on
    running_(false)       // Not running initially
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
    CreateProperty("Interval-us", CDeviceUtils::ConvertToString(interval_), MM::Float, false, pAct);
    SetPropertyLimits("Interval-us", 1.0, 1000000.0);

    // Pulse Duration property
    pAct = new CPropertyAction(this, &TeensyPulseGenerator::OnPulseDuration);
    CreateProperty("PulseDuration-us", CDeviceUtils::ConvertToString(pulseDuration_), MM::Float, false, pAct);
    SetPropertyLimits("PulseDuration-us", 1.0, 100000.0);

    // Trigger Mode property
    pAct = new CPropertyAction(this, &TeensyPulseGenerator::OnTriggerMode);
    CreateProperty("TriggerMode", triggerMode_ ? "On" : "Off", MM::String, false, pAct);
    AddAllowedValue("TriggerMode", "Off");
    AddAllowedValue("TriggerMode", "On");

    // State (Start/Stop) property
    pAct = new CPropertyAction(this, &TeensyPulseGenerator::OnState);
    CreateProperty("State", "Stop", MM::String, false, pAct);
    AddAllowedValue("State", "Stop");
    AddAllowedValue("State", "Start");

    initialized_ = true;
    return DEVICE_OK;
}

int TeensyPulseGenerator::Shutdown()
{
    if (port_ != "")
    {
       // Ensure device is stopped
       SendCommand(2);  // Stop command
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
    // Prepare buffer
   const unsigned int buflen = 5;
   unsigned char buffer[buflen];
   buffer[0] = cmd;

   // For commands that require a parameter, convert to little-endian
   if (cmd >= 3 && cmd <= 5)
   {
      buffer[1] = param & 0xFF;
      buffer[2] = (param >> 8) & 0xFF;
      buffer[3] = (param >> 16) & 0xFF;
      buffer[4] = (param >> 24) & 0xFF;

      // Send 5-byte command
      return WriteToComPort(port_.c_str(), buffer, buflen);
   }
   else
   {
      // Send 1-byte command
      return WriteToComPort(port_.c_str(), buffer, 1);
   }
}


bool TeensyPulseGenerator::ReadResponse(std::string& response)
{
   return GetSerialAnswer(port_.c_str(), "\r\n", response);
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
           int ret = SendCommand(3, static_cast<uint32_t>(interval_));
           if (ret != DEVICE_OK)
              return ret;
           std::string response;
           if (GetSerialAnswer(port_.c_str(), "\r\n", response) != DEVICE_OK) 
           {
               return ERR_COMMUNICATION;
           }
           // TODO: check we received the correct asnwer, for now: log
           GetCoreCallback()->LogMessage(this, response.c_str(), true);
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
           int ret = SendCommand(4, static_cast<uint32_t>(interval_));
           if (ret != DEVICE_OK)
              return ret;
           std::string response;
           if (GetSerialAnswer(port_.c_str(), "\r\n", response) != DEVICE_OK) {
               return ERR_COMMUNICATION;
           }
           // TODO: check we received the correct asnwer, for now: log
           GetCoreCallback()->LogMessage(this, response.c_str(), true);
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
           int ret = SendCommand(5, triggerMode_ ? 1 : 0);
           if (ret != DEVICE_OK)
              return ret;
           std::string response;
           if (GetSerialAnswer(port_.c_str(), "\r\n", response) != DEVICE_OK) {
               return ERR_COMMUNICATION;
           }
           // TODO: check we received the correct asnwer, for now: log
           GetCoreCallback()->LogMessage(this, response.c_str(), true);
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
           int ret = SendCommand(1); // Start command
           if (ret != DEVICE_OK)
              return ret;
           std::string response;
           if (GetSerialAnswer(port_.c_str(), "\r\n", response) != DEVICE_OK) {
               return ERR_COMMUNICATION;
           }
           // TODO: check we received the correct asnwer, for now: log
           GetCoreCallback()->LogMessage(this, response.c_str(), true);
           running_ = true;
        }
        else if (stateStr == "Stop" && running_)
        {
           int ret = SendCommand(2); // Stop command
           if (ret != DEVICE_OK)
              return ret;
           std::string response;
           if (GetSerialAnswer(port_.c_str(), "\r\n", response) != DEVICE_OK) {
               return ERR_COMMUNICATION;
           }
           // TODO: check we received the correct asnwer, for now: log
           GetCoreCallback()->LogMessage(this, response.c_str(), true);
           running_ = false;
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
