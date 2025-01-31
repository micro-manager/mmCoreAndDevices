#include "ModuleInterface.h"
#include "mux_distrib.h"
#include "iostream"
#include <vector>
#include <string>
#include <windows.h>

using namespace std;

#define PERIOD_ACQ 1000
#define ERR_PORT_CHANGE_FORBIDDEN	101 

const char* g_Default_String_Muxd = "Unknown";

MuxDistrib::MuxDistrib() : initialized_(false), isGet_(true), timerOn(false), timerThread(NULL), stopTimerThread(false) {
   // Device COM configuration
   CPropertyAction* pAct = new CPropertyAction(this, &MuxDistrib::OnPort);
   CreateProperty(MM::g_Keyword_Port, g_Default_String_Muxd, MM::String, false, pAct, true);

   // Raw UART Control
   CreateProperty("IsGet", "0", MM::Integer, false, new CPropertyAction(this, &MuxDistrib::OnIsGet));
   CreateProperty("Command", "", MM::String, false, new CPropertyAction(this, &MuxDistrib::OnCommand));
   CreateProperty("Parameters", "", MM::String, false);
   CreateProperty("Response", "", MM::String, false);

   // Nominal Operation Control
   //// ---- new post
   CreateProperty("NewPost", "0", MM::Integer, false);

   //// ---- debug
   CreateProperty("Debug", "0", MM::Integer, false);
   
   // ---- Timer Control
   CreateProperty("TimerOn", "0", MM::Integer, false, new CPropertyAction(this, &MuxDistrib::OnStart));
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

   if (Get("<FIRMV", "") != DEVICE_OK) {
      return DEVICE_ERR;
   }
   if (Get("<PINGA", "") != DEVICE_OK) {
      return DEVICE_ERR;
   }
   initialized_ = true;
   LogMessage("Device initialized successfully", false);
   return DEVICE_OK;
}

int MuxDistrib::Shutdown() {
   if (initialized_) {
      LogMessage("Device Shutdown", false);
      StopTimer();
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

            if (Get("<FIRMV?", "") == DEVICE_OK) {
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

int MuxDistrib::OnStart(MM::PropertyBase* pProp, MM::ActionType eAct) {
   if (eAct == MM::BeforeGet) {
      pProp->Set(static_cast<long>(timerOn ? 1 : 0));
   }
   else if (eAct == MM::AfterSet) {
      long value;
      pProp->Get(value);
      timerOn = (value != 0);

      if (timerOn) {
         return StartTimer();
      }
     else {
        return StopTimer();
     }
   }
   return DEVICE_OK;
}

int MuxDistrib::OnIsGet(MM::PropertyBase* pProp, MM::ActionType eAct) {
   if (eAct == MM::BeforeGet) {
      pProp->Set(static_cast<long>(isGet_ ? 1 : 0));
   }
   else if (eAct == MM::AfterSet) {
      long value;
      pProp->Get(value);
      isGet_ = (value != 0);
   }
   return DEVICE_OK;
}

int MuxDistrib::OnCommand(MM::PropertyBase* pProp, MM::ActionType eAct) {
   if (eAct == MM::BeforeGet) {
      return DEVICE_OK;
   }
   else if (eAct == MM::AfterSet) {
      string command;
      char parameters[MM::MaxStrLength];
      pProp->Get(command);
      GetProperty("Parameters", parameters);

      if (isGet_) {
         return Get(command, parameters);
      }
      int result = Set(command, parameters);
      string currentData = GetAllData();
      if (UpdateAllProperties(currentData) != DEVICE_OK) {
         LogMessage("Failed to update properties");
         return DEVICE_ERR;
      }
      return result;
   }
   return DEVICE_OK;
}

int MuxDistrib::OnTrigger(MM::PropertyBase* pProp, MM::ActionType eAct) {
   if (eAct == MM::AfterSet) {
      long value;
      pProp->Get(value);
      return Set("<TRIGO", to_string(value));
   }
   return DEVICE_OK;
}

// Mux Distrib Custom methods
// ---- Raw UART Control
string MuxDistrib::FormatMsg(const string& uartCmd, const string& parameters, bool isGet) {
   string buffer = isGet ? uartCmd + "?" : uartCmd + "!";
   if (!parameters.empty()) {
      buffer += parameters;
   }
   //buffer += "\n";
   LogMessage("Cmd sent: " + buffer, false);
   return buffer;
}

int MuxDistrib::Get(const string& uartCmd, const string& parameters) {
   string command = FormatMsg(uartCmd, parameters, true);
   SendSerialCommand(port_.c_str(), command.c_str(), "\n");

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
   string command = FormatMsg(uartCmd, parameters, false);
   SendSerialCommand(port_.c_str(), command.c_str(), "\n");

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

// ---- Nominal mode Control (Thread)
int MuxDistrib::StartTimer() {
   stopTimerThread = false;
   timerThread = CreateThread(NULL, 0, TimerThreadFunction, this, 0, NULL);
   if (timerThread == NULL) {
      LogMessage("Error on timer's thread making", true);
      return DEVICE_ERR;
   }
   return DEVICE_OK;
}

int MuxDistrib::StopTimer() {
   if (timerThread != NULL) {
      stopTimerThread = true;
      WaitForSingleObject(timerThread, INFINITE);
      CloseHandle(timerThread);
      timerThread = NULL;
   }
   return DEVICE_OK;
}

DWORD WINAPI MuxDistrib::TimerThreadFunction(LPVOID lpParam) {
   MuxDistrib* pDevice = static_cast<MuxDistrib*>(lpParam);
   return pDevice->ReadTimer(PERIOD_ACQ);
}

int MuxDistrib::ReadTimer(int msInterval) {
   HANDLE timerHandler = CreateWaitableTimer(NULL, FALSE, NULL);
   if (timerHandler == NULL) {
      LogMessage("Error creating timer", true);
      return DEVICE_ERR;
   }

   LARGE_INTEGER dueTime;
   dueTime.QuadPart = -(10000LL * msInterval);

   if (!SetWaitableTimer(timerHandler, &dueTime, msInterval, NULL, NULL, 0)) {
      LogMessage("Error on timer configuration", true);
      CloseHandle(timerHandler);
      return DEVICE_ERR;
   }
   int result;
   string deviceResponse;

   while (initialized_ && timerOn) {
      WaitForSingleObject(timerHandler, INFINITE);

      deviceResponse = GetAllData();
      if (deviceResponse == "ERROR") {
          break;
      }
      result = UpdateAllProperties(deviceResponse);
      if (result != DEVICE_OK) {
         LogMessage("Error update failed", true);
         break;
      }
   }

   CloseHandle(timerHandler);
   return DEVICE_OK;
}

string MuxDistrib::GetAllData() {
   int result;
   char pingaResponse[MM::MaxStrLength];

   // Get PingA Data
   result = Get("<PING_", "");
   GetProperty("Response", pingaResponse);
   if (result != DEVICE_OK) {
      LogMessage("Error on PING_ data reading, attempting reconnection", true);
      if (SerialReconnection(10) != DEVICE_OK) {
         LogMessage("Reconnection failed after multiple attempts", true);
         return "ERROR";
      }
   }
   LogMessage(pingaResponse, false);
   return pingaResponse;
}

int MuxDistrib::UpdateAllProperties(string deviceResponse) {
   string target = deviceResponse.substr(11);
   vector<string> sensorsDataTitle = { "NewPost", "Debug" };
   vector<string> sensorsValue(2);
   istringstream iss(target);
   string token;

   int i = 0;
   while (getline(iss, token, ':') && i < sensorsValue.size()) {
      sensorsValue[i] = token;
      LogMessage(sensorsDataTitle[i] + " : " + sensorsValue[i], false);
      i++;
   }

   // Jump over data corresponding to SensorMaster 
   SetProperty("NewPost", sensorsValue[0].c_str());
   SetProperty("Debug", sensorsValue[1].c_str());

   return DEVICE_OK;
}

string MuxDistrib::ExtractString(char* entry_String) {
   const char* last_Pipe = strrchr(entry_String, '|');
   if (last_Pipe) {
      last_Pipe++;
      LogMessage("Extracted value " + string(last_Pipe), false);
      return string(last_Pipe);
   }
   LogMessage("no val", false);
   return string();
}