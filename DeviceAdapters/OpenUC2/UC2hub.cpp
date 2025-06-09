
#include "openuc2.h"
#include "UC2Hub.h"
#include "ModuleInterface.h"

#ifdef WIN32
#include <windows.h>
#endif

UC2Hub::UC2Hub() : initialized_(false), port_("Undefined")
{
   // Pre-initialization property: the serial port
   CPropertyAction* pAct = new CPropertyAction(this, &UC2Hub::OnPort);
   CreateProperty(MM::g_Keyword_Port, "Undefined", MM::String, false, pAct, true);
}

UC2Hub::~UC2Hub()
{
   Shutdown();
}

void UC2Hub::GetName(char* pszName) const
{
   CDeviceUtils::CopyLimitedString(pszName, g_HubName);
}

bool UC2Hub::Busy()
{
   // If needed, query the device to check if busy. For now, return false.
   return false;
}

bool UC2Hub::SupportsDeviceDetection(void)
{
   return true; // We can attempt to detect on open
}

MM::DeviceDetectionStatus UC2Hub::DetectDevice(void)
{
   if (port_ == "Undefined" || port_.length() == 0)
      return MM::Misconfigured;

   // Attempt some minimal communication if we like:
   // e.g. see if we can open port, flush, etc.
   // If that works, return MM::CanCommunicate.
   return MM::CanCommunicate;
}

int UC2Hub::DetectInstalledDevices()
{
   ClearInstalledDevices();

   // We can add known sub-devices:
   MM::Device* pDev = 0;

   pDev = CreateDevice(g_XYStageName);
   if (pDev) AddInstalledDevice(pDev);

   pDev = CreateDevice(g_ZStageName);
   if (pDev) AddInstalledDevice(pDev);

   pDev = CreateDevice(g_ShutterName);
   if (pDev) AddInstalledDevice(pDev);

   return DEVICE_OK;
}

int UC2Hub::Initialize()
{
   if (initialized_)
      return DEVICE_OK;

   // Example: optional firmware check
   if (!CheckFirmware()) {
      // Return error or keep going. Example:
      //return ERR_FIRMWARE_MISMATCH;
   }

   // Mark as initialized
   initialized_ = true;

   return DEVICE_OK;
}

int UC2Hub::Shutdown()
{
   initialized_ = false;
   return DEVICE_OK;
}

int UC2Hub::OnPort(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet) {
      pProp->Set(port_.c_str());
   }
   else if (eAct == MM::AfterSet) {
      if (initialized_) {
         // Once initialized, port changes are not allowed
         pProp->Set(port_.c_str());
      } else {
         pProp->Get(port_);
      }
   }
   return DEVICE_OK;
}

bool UC2Hub::CheckFirmware()
{
   // Minimal example:
   std::string cmd = R"({"task":"/state_get"})";
   std::string reply;
   int ret = SendJsonCommand(cmd, reply, false);
   if (ret != DEVICE_OK)
      return false;

   // Check for substring
   if (reply.find("UC2_Feather") != std::string::npos) {
      LogMessage("Found UC2 signature in firmware response.", true);
      return true;
   }
   return true; // or false if you want to be strict
}

int UC2Hub::SendJsonCommand(const std::string& jsonCmd, std::string& jsonReply, bool debug)
{
   std::lock_guard<std::mutex> guard(ioMutex_); // Thread safety

   // 1) Purge the port
   int ret = PurgeComPort(port_.c_str());
   if (ret != DEVICE_OK)
      return ret;

   // 2) Send
   ret = SendSerialCommand(port_.c_str(), jsonCmd.c_str(), "\n");
   if (ret != DEVICE_OK)
      return ret;

   // 3) Read line (or multiple lines). For example:
   std::string ans;
   ret = GetSerialAnswer(port_.c_str(), "\r", ans);
   if (ret != DEVICE_OK)
      return ret;

   jsonReply = ans;
   if (debug) {
      std::ostringstream os;
      os << "[UC2Hub::SendJsonCommand] Sent: " << jsonCmd << " Received: " << ans;
      LogMessage(os.str().c_str(), true);
   }
   return DEVICE_OK;
}
