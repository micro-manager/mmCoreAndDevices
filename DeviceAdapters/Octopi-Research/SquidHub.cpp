#include "squid.h"
#include "crc8.h"

#ifdef WIN32
   #define WIN32_LEAN_AND_MEAN
   #include <windows.h>
#endif




const char* g_HubDeviceName = "SquidHub";


MODULE_API void InitializeModuleData() 
{
   RegisterDevice(g_HubDeviceName, MM::HubDevice, g_HubDeviceName);
   RegisterDevice(g_LEDShutterName, MM::ShutterDevice, "LEDs");
   RegisterDevice(g_XYStageName, MM::XYStageDevice, "XY-Stage");
}


MODULE_API MM::Device* CreateDevice(const char* deviceName)
{

   if (strcmp(deviceName, g_HubDeviceName) == 0)
   {
      return new SquidHub();
   }
   else if (strcmp(deviceName, g_LEDShutterName) == 0)
   {
      return new SquidLEDShutter();
   }
   else if (strcmp(deviceName, g_XYStageName) == 0)
   {
      return new SquidXYStage();
   }

   // ...supplied name not recognized
   return 0;
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
   delete pDevice;
}


SquidHub::SquidHub() :
   initialized_(false),
   monitoringThread_(0),
   xyStageDevice_(0),
   port_("Undefined"),
   cmdNr_(1)
{
   InitializeDefaultErrorMessages();

   CPropertyAction* pAct = new CPropertyAction(this, &SquidHub::OnPort);
   CreateProperty(MM::g_Keyword_Port, "Undefined", MM::String, false, pAct, true);
   x_ = 0l;
   y_ = 0l;
   z_ = 0l;
}


SquidHub::~SquidHub()   
{
   LogMessage("Destructor called");
}


void SquidHub::GetName(char* name) const
{
   CDeviceUtils::CopyLimitedString(name, g_HubDeviceName);
}


int SquidHub::Initialize() {
   Sleep(200);

   monitoringThread_ = new SquidMonitoringThread(*this->GetCoreCallback(), *this, true);
   monitoringThread_->Start();
   
   const unsigned cmdSize = 8;
   unsigned char cmd[cmdSize];
   cmd[0] = 0x00;
   cmd[1] = 255; // CMD_SET.RESET
   for (unsigned i = 2; i < cmdSize; i++) {
      cmd[i] = 0;
   }
   int ret = SendCommand(cmd, cmdSize, &pendingCmd_);
   if (ret != DEVICE_OK) {
      return ret;
   }

   cmd[0] = 1; 
   cmd[1] = 254; // CMD_INITIALIZE_DRIVERS
   ret = SendCommand(cmd, cmdSize, &pendingCmd_);
   if (ret != DEVICE_OK) {
      return ret;
   }

   initialized_ = true;


   return DEVICE_OK;

   //SET_ILLUMINATION_LED_MATRIX = 13
   /*
   cmd[0] = 2; 
   cmd[1] = 13; 
   cmd[2] = 1;
   cmd[3] = 128;
   cmd[4] = 128;
   cmd[5] = 0;
   ret = SendCommand(cmd, cmdSize);
   if (ret != DEVICE_OK) {
      return ret;
   }

   cmd[0] = 0x03;
   cmd[1] = 10; // CMD_SET. TURN_ON_ILLUMINATION 
   for (unsigned i = 2; i < cmdSize; i++) {
      cmd[i] = 0;
   }
   ret = SendCommand(cmd, cmdSize);
   if (ret != DEVICE_OK) {
      return ret;
   }

   
   const unsigned msgLength = 24;
   unsigned char msg[msgLength];
   unsigned long read = 0;
   unsigned tries = 0;
   while (read == 0 && tries < 20) {
      Sleep(20);
      ReadFromComPort(port_.c_str(), msg, msgLength, read);
      tries++;
      if (read > 0) {
         LogMessage("Read something from serial port", false);
         std::ostringstream os;
         os << "Tries: " << tries << ", Read # of bytes: " << read;
         LogMessage(os.str().c_str(), false);
      }
   }
   if (tries >= 20) {
      LogMessage("Read nothing from serial port", false);
   }
   

   const unsigned TURN_ON_ILLUMINATION = 10;
   for (unsigned i = 0; i < cmdSize; i++) {
      cmd[i] = 0;
   }
   cmd[1] = TURN_ON_ILLUMINATION;
   ret = SendCommand(cmd, cmdSize);
   if (ret != DEVICE_OK) {
      return ret;
   }
   */
}


int SquidHub::Shutdown() {
   if (initialized_)
   {
      delete(monitoringThread_);
      initialized_ = false;
   }
   return DEVICE_OK;
}

bool SquidHub::Busy()
{
    return false;
}

bool SquidHub::SupportsDeviceDetection(void)
{
   return false;  // can implement this later

}

MM::DeviceDetectionStatus SquidHub::DetectDevice(void)
{
   // TODO
   return MM::CanCommunicate;  
}


int SquidHub::DetectInstalledDevices()
{
   if (MM::CanCommunicate == DetectDevice())
   {
      std::vector<std::string> peripherals;
      peripherals.clear();
      peripherals.push_back(g_LEDShutterName);
      peripherals.push_back(g_XYStageName);
      for (size_t i = 0; i < peripherals.size(); i++)
      {
         MM::Device* pDev = ::CreateDevice(peripherals[i].c_str());
         if (pDev)
         {
            AddInstalledDevice(pDev);
         }
      }
   }

   return DEVICE_OK;
}



int SquidHub::OnPort(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet) {
      pProp->Set(port_.c_str());
   }
   else if (eAct == MM::AfterSet) {
      if (initialized_) {
         // revert}
         pProp->Set(port_.c_str());
         return ERR_PORT_CHANGE_FORBIDDEN;
      }
      // take this port.  TODO: should we check if this is a valid port?
      pProp->Get(port_);
   }

   return DEVICE_OK;
}

/*
uint8_t SquidHub::crc8ccitt(const void* data, size_t size) {
   uint8_t val = 0;

   uint8_t* pos = (uint8_t*)data;
   uint8_t* end = pos + size;

   while (pos < end) {
      val = CRC_TABLE[val ^ *pos];
      pos++;
   }

   return val;
}
*/


int SquidHub::assignXYStageDevice(SquidXYStage* xyStageDevice)
{
   xyStageDevice_ = xyStageDevice;
   return DEVICE_OK;

}

bool SquidHub::IsCommandPending(uint8_t cmdNr)
{
   std::lock_guard<std::recursive_mutex> locker(lock_);
   return pendingCmd_ != cmdNr;
}

void SquidHub::ReceivedCommand(uint8_t cmdNr)
{
   std::lock_guard<std::recursive_mutex> locker(lock_);
   if (pendingCmd_ == cmdNr)
   {
      pendingCmd_ = 0;
   }
}

void SquidHub::SetCommandPending(uint8_t cmdNr)
{
   std::lock_guard<std::recursive_mutex> locker(lock_);
   pendingCmd_ = cmdNr;
}

int SquidHub::SendCommand(unsigned char* cmd, unsigned cmdSize, uint8_t* cmdNr)
{
   cmd[0] = cmdNr_;
   if (cmdNr_ < 255) 
      cmdNr_++;
   else 
      cmdNr_ = 1;
   cmd[cmdSize - 1] = crc8ccitt(cmd, cmdSize - 1);
   if (true) {
      std::ostringstream os;
      os << "Sending message: ";
      for (unsigned int i = 0; i < cmdSize; i++) {
         os << std::hex << (unsigned int)cmd[i] << " ";
      }
      LogMessage(os.str().c_str(), false);
   }
   *cmdNr = cmdNr_;
   SetCommandPending(cmdNr_);
   return WriteToComPort(port_.c_str(), cmd, cmdSize);
}

/**
* Helper function to send Move or Move Relative command to relevant Stage
  MOVE_X = 0
  MOVE_Y = 1
  MOVE_Z = 2
  MOVE_THETA = 3
  MOVETO_X = 6
  MOVETO_Y = 7
  MOVETO_Z = 8
*/
int SquidHub::SendMoveCommand(const int command, long steps)
{
   const unsigned cmdSize = 8;
   unsigned char cmd[cmdSize];
   for (unsigned i = 0; i < cmdSize; i++) {
      cmd[i] = 0;
   }
   cmd[1] = (unsigned char)command;
   // TODO: Fix in case we are running on a Big Endian system
   cmd[2] = steps >> 24;
   cmd[3] = (steps >> 16) & 0xFF;
   cmd[4] = (steps >> 8) & 0xFF;
   cmd[5] = steps & 0xFF;

   int ret = SendCommand(cmd, cmdSize, &cmdNr_);
   if (ret != DEVICE_OK)
      return ret;

   return DEVICE_OK;
}

int SquidHub::GetPositionSteps(long& x, long& y)
{
   x = x_.load();
   y = y_.load();
   return DEVICE_OK;
}

int SquidHub::SetPositionXSteps(long x)
{
   if (x_.load() != x)
   {
      x_.store(x);
      if (xyStageDevice_ != 0)
         xyStageDevice_->Callback(x, y_.load());
   }
   return DEVICE_OK;
}


int SquidHub::SetPositionYSteps(long y)
{
   if (y_.load() != y)
   {
      y_.store(y);
      if (xyStageDevice_ != 0)
         xyStageDevice_->Callback(x_.load(), y);
   }
   return DEVICE_OK;
}


int SquidHub::SetPositionZSteps(long z)
{
   if (z_.load() != z)
   {
      z_.store(z);
   //   this->GetCoreCallback()->OnStagePositionChanged(zStageDevice_, z);
   }
   return DEVICE_OK;
}