#include "ModuleInterface.h"
#include "mux_distrib.h"
#include "iostream"
#include <vector>
#include <string>
#include <windows.h>

using namespace std;

#define ERR_PORT_CHANGE_FORBIDDEN	101 

const char* g_Default_String_Muxd = "Unknown";

MuxDistrib::MuxDistrib() : initialized_(false){
   // Device COM configuration
   CPropertyAction* pAct = new CPropertyAction(this, &MuxDistrib::OnPort);
   CreateProperty(MM::g_Keyword_Port, g_Default_String_Muxd, MM::String, false, pAct, true);

   // Raw UART Control
   CreateProperty("Pos", "0", MM::Integer, false, new CPropertyAction(this, &MuxDistrib::OnPos));
   CreateProperty("Response", "", MM::String, false);
}


MuxDistrib::~MuxDistrib() {
   if (initialized_) {
      Shutdown();
   }    
}

// MMDevice API
int MuxDistrib::Initialize() {
   if (initialized_) {
      return DEVICE_OK;
   }
   LogMessage("Start Initialization", false);

   if (Get("/1?23", "") != DEVICE_OK) {
      return DEVICE_ERR;
   }
   if (Get("/1?9000", "") != DEVICE_OK) {
       return DEVICE_ERR;
   }
   initialized_ = true;
   LogMessage("Device initialized successfully", false);
   return DEVICE_OK;
}

int MuxDistrib::Shutdown() {
   if (initialized_) {
      LogMessage("Device Shutdown", false);
      initialized_ = false;
   }
   return DEVICE_OK;
}

bool MuxDistrib::Busy() {
   return false;
}

void MuxDistrib::GetName(char* name) const {
   CDeviceUtils::CopyLimitedString(name, "MUX_DISTRIB");
}

// MM - Mux Distrib Custom methods
MM::DeviceDetectionStatus MuxDistrib::DetectDevice(void) {
   if (initialized_) {
      return MM::CanCommunicate;
   }
   // all conditions must be satisfied...
   MM::DeviceDetectionStatus result = MM::Misconfigured;
   char answerTO[MM::MaxStrLength];

   try
   {
      std::string portLowerCase = GetPort();
      for (std::string::iterator its = portLowerCase.begin(); its != portLowerCase.end(); ++its) {
         *its = (char)tolower(*its);
      }
      if (0 < portLowerCase.length() && 0 != portLowerCase.compare("undefined") && 0 != portLowerCase.compare("unknown")) {
         result = MM::CanNotCommunicate;
         // record the default answer time out
         GetCoreCallback()->GetDeviceProperty(GetPort().c_str(), "AnswerTimeout", answerTO);
         CDeviceUtils::SleepMs(2000);
         GetCoreCallback()->SetDeviceProperty(GetPort().c_str(), MM::g_Keyword_Handshaking, "off");
         GetCoreCallback()->SetDeviceProperty(GetPort().c_str(), MM::g_Keyword_StopBits, "1");
         GetCoreCallback()->SetDeviceProperty(GetPort().c_str(), "AnswerTimeout", "500.0");
         GetCoreCallback()->SetDeviceProperty(GetPort().c_str(), "DelayBetweenCharsMs", "0");
         MM::Device* pS = GetCoreCallback()->GetDevice(this, GetPort().c_str());

         std::vector<std::string> possibleBauds;
         possibleBauds.push_back("115200");
         for (std::vector< std::string>::iterator bit = possibleBauds.begin(); bit != possibleBauds.end(); ++bit) {
            GetCoreCallback()->SetDeviceProperty(GetPort().c_str(), MM::g_Keyword_BaudRate, (*bit).c_str());
            pS->Initialize();
            PurgeComPort(GetPort().c_str());

            if (Get("/1?23", "") == DEVICE_OK) {
               result = MM::CanCommunicate;
               LogMessage(to_string(result), false);
            }
            pS->Shutdown();
            if (MM::CanCommunicate == result) {
                break;
            }
            else {
                CDeviceUtils::SleepMs(10);
            }
         }
         GetCoreCallback()->SetDeviceProperty(GetPort().c_str(), "AnswerTimeout", answerTO);
      }
   }
   catch (...) {
      //LogMessage("Exception in DetectDevice!",false);
   }
   return result;
}

int MuxDistrib::DetectInstalledDevices() {
   if (MM::CanCommunicate == DetectDevice())
   {
      std::vector<std::string> peripherals;
      peripherals.clear();
      peripherals.push_back("MUX_DISTRIB");
      for (size_t i = 0; i < peripherals.size(); i++) {
         MM::Device* pDev = ::CreateDevice(peripherals[i].c_str());
         if (pDev) {
            LogMessage("Device created");
         }
      }
   }
   return DEVICE_OK;
}

int MuxDistrib::OnPort(MM::PropertyBase* pProp, MM::ActionType eAct) {
   if (eAct == MM::BeforeGet) {
      pProp->Set(port_.c_str());
   }
   else if (eAct == MM::AfterSet) {
      if (initialized_) {
         // revert
         pProp->Set(port_.c_str());
         return ERR_PORT_CHANGE_FORBIDDEN;
      }
      pProp->Get(port_);
   }
   return DEVICE_OK;
}

int MuxDistrib::OnPos(MM::PropertyBase* pProp, MM::ActionType eAct) {
   if (eAct == MM::BeforeGet) {
      return DEVICE_OK;
   }
   else if (eAct == MM::AfterSet) {
      string pos;
      pProp->Get(pos);
      int result = Set("/1b", pos + "R");
      return result;
   }
   return DEVICE_OK;
}

// Mux Distrib Custom methods
// ---- Raw UART Control

int MuxDistrib::Get(const string& uartCmd, const string& parameters) {
   string command = uartCmd + parameters;
   SendSerialCommand(port_.c_str(), command.c_str(), "\r");

   int ret = ReadResponse();

   if (ret != DEVICE_OK) {
      LogMessage("Failed to send command: " + command, true);
      if (SerialReconnection(10) != DEVICE_OK) {
         return DEVICE_ERR;
      }
      if (SendSerialCommand(port_.c_str(), command.c_str(), "\n") != DEVICE_OK) {
         LogMessage("Failed to send command after reconnection attempts", true);
         return DEVICE_ERR;
      }
      return DEVICE_ERR;
   }
   return DEVICE_OK;
}

int MuxDistrib::Set(const string& uartCmd, const string& parameters) {
   string command = uartCmd + parameters;
   SendSerialCommand(port_.c_str(), command.c_str(), "\r");

   int ret = ReadResponse();

   if (ret != DEVICE_OK) {
      LogMessage("Failed to send command: " + command, true);
      if (SerialReconnection(10) != DEVICE_OK) {
         return DEVICE_ERR;
        }
        if (SendSerialCommand(port_.c_str(), command.c_str(), "\n") != DEVICE_OK) {
           LogMessage("Failed to send command after reconnection attempts", true);
           return DEVICE_ERR;
        }
        return DEVICE_ERR;
   }
   return DEVICE_OK;
}

int MuxDistrib::ReadResponse() {
   string answer;
   int ret = GetSerialAnswer(port_.c_str(), "\n", answer);
   LogMessage("Device Answer: " + answer, false);
   SetProperty("Response", answer.c_str());
   if (ret != DEVICE_OK or answer.empty()) {
      return DEVICE_ERR;
   }
   return DEVICE_OK;
}

string MuxDistrib::GetPort() {
   std::string port;
   port = port_;
   return port;
}

int MuxDistrib::SerialReconnection(int maxAttempts) {
   for (int attempt = 1; attempt <= maxAttempts; attempt++) {
      LogMessage("Try to reconnect " + std::to_string(attempt) + " / " + std::to_string(maxAttempts), false);
      // No solution for now
      Sleep(1000); // Attendre 1 seconde entre les tentatives
   }
   LogMessage("Failed to reconnect after " + std::to_string(maxAttempts) + " try", true);
   return DEVICE_ERR;
}
