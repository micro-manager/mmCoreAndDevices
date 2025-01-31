#include "ModuleInterface.h"
#include "mux_wire_v3.h"
#include "iostream"
#include <vector>
#include <string>
#include <windows.h>

using namespace std;

#define PERIOD_ACQ 1000
#define ERR_PORT_CHANGE_FORBIDDEN	101 

const char* g_Default_String_Muxw = "Unknown";

MuxWireV3::MuxWireV3() : initialized_(false), isGet_(true), timerOn(false), timerThread(NULL), stopTimerThread(false) {
   // Device COM configuration
   CPropertyAction* pAct = new CPropertyAction(this, &MuxWireV3::OnPort);
   CreateProperty(MM::g_Keyword_Port, g_Default_String_Muxw, MM::String, false, pAct, true);

   // Raw UART Control
   CreateProperty("IsGet", "0", MM::Integer, false, new CPropertyAction(this, &MuxWireV3::OnIsGet));
   CreateProperty("Command", "", MM::String, false, new CPropertyAction(this, &MuxWireV3::OnCommand));
   CreateProperty("Parameters", "", MM::String, false);
   CreateProperty("Response", "", MM::String, false);

   // Nominal Operation Control


   //// ---- Type of valve
   CreateProperty("Type0", "0", MM::Integer, false);
   CreateProperty("Type1", "0", MM::Integer, false);
   CreateProperty("Type2", "0", MM::Integer, false);
   CreateProperty("Type3", "0", MM::Integer, false);
   CreateProperty("Type4", "0", MM::Integer, false);
   CreateProperty("Type5", "0", MM::Integer, false);
   CreateProperty("Type6", "0", MM::Integer, false);
   CreateProperty("Type7", "0", MM::Integer, false);

   //// ---- Status of valve
   CreateProperty("Status0", "0", MM::Integer, false);
   CreateProperty("Status1", "0", MM::Integer, false);
   CreateProperty("Status2", "0", MM::Integer, false);
   CreateProperty("Status3", "0", MM::Integer, false);
   CreateProperty("Status4", "0", MM::Integer, false);
   CreateProperty("Status5", "0", MM::Integer, false);
   CreateProperty("Status6", "0", MM::Integer, false);
   CreateProperty("Status7", "0", MM::Integer, false);
   //// ---- Status of valve
   CreateProperty("Master0", "0", MM::Integer, false);
   CreateProperty("Master1", "0", MM::Integer, false);
   CreateProperty("Master2", "0", MM::Integer, false);
   CreateProperty("Master3", "0", MM::Integer, false);
   CreateProperty("Master4", "0", MM::Integer, false);
   CreateProperty("Master5", "0", MM::Integer, false);
   CreateProperty("Master6", "0", MM::Integer, false);
   CreateProperty("Master7", "0", MM::Integer, false);
   //// ---- Trigger Current Value
   CreateProperty("TriggerIn", "0", MM::Integer, false);
   CreateProperty("TriggerOut", "0", MM::Integer, false, new CPropertyAction(this, &MuxWireV3::OnTrigger));
   //// ---- Timer Control
   CreateProperty("TimerOn", "0", MM::Integer, false, new CPropertyAction(this, &MuxWireV3::OnStart));
}


MuxWireV3::~MuxWireV3() {
   if (initialized_) {
      Shutdown();
   }    
}

// MMDevice API
int MuxWireV3::Initialize() {
   if (initialized_) {
      return DEVICE_OK;
   }
   LogMessage("Start Initialization", false);

   if (Get("<FIRMV", "") != DEVICE_OK) {
      return DEVICE_ERR;
   }
   if (Get("<PING_", "") != DEVICE_OK) {
      return DEVICE_ERR;
   }
   initialized_ = true;
   LogMessage("Device initialized successfully", false);
   return DEVICE_OK;
}

int MuxWireV3::Shutdown() {
   if (initialized_) {
      LogMessage("Device Shutdown", false);
      StopTimer();
      initialized_ = false;
   }
   return DEVICE_OK;
}

bool MuxWireV3::Busy() {
   return false;
}

void MuxWireV3::GetName(char* name) const {
   CDeviceUtils::CopyLimitedString(name, "MUX_WIRE");
}

// MM - Mux Wire Custom methods
MM::DeviceDetectionStatus MuxWireV3::DetectDevice(void) {
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
            GetCoreCallback()->SetDeviceProperty(GetPort().c_str(), "AnswerTimeout", answerTO);
         }
      }
   }
   catch (...)
   {
      //LogMessage("Exception in DetectDevice!",false);
   }
   return result;
}

int MuxWireV3::DetectInstalledDevices() {
   if (MM::CanCommunicate == DetectDevice())
   {
      std::vector<std::string> peripherals;
      peripherals.clear();
      peripherals.push_back("MUX_WIRE");
      for (size_t i = 0; i < peripherals.size(); i++)
      {
         MM::Device* pDev = ::CreateDevice(peripherals[i].c_str());
         if (pDev) {
            LogMessage("Device created");
         }
      }
   }
   return DEVICE_OK;
}

int MuxWireV3::OnPort(MM::PropertyBase* pProp, MM::ActionType eAct) {
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

int MuxWireV3::OnStart(MM::PropertyBase* pProp, MM::ActionType eAct) {
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

int MuxWireV3::OnIsGet(MM::PropertyBase* pProp, MM::ActionType eAct) {
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

int MuxWireV3::OnCommand(MM::PropertyBase* pProp, MM::ActionType eAct) {
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

int MuxWireV3::OnTrigger(MM::PropertyBase* pProp, MM::ActionType eAct) {
   if (eAct == MM::AfterSet) {
      long value;
      pProp->Get(value);
      return Set("<TRIGO", to_string(value));
   }
   return DEVICE_OK;
}

// Mux Wire Custom methods
// ---- Raw UART Control
string MuxWireV3::FormatMsg(const string& uartCmd, const string& parameters, bool isGet) {
   string buffer = isGet ? uartCmd + "?" : uartCmd + "!";
   if (!parameters.empty()) {
      buffer += parameters;
   }
   //buffer += "\n";
   LogMessage("Cmd sent: " + buffer, false);
   return buffer;
}

int MuxWireV3::Get(const string& uartCmd, const string& parameters) {
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

int MuxWireV3::Set(const string& uartCmd, const string& parameters) {
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

int MuxWireV3::ReadResponse() {
   string answer;
   int ret = GetSerialAnswer(port_.c_str(), "\n", answer);
   LogMessage("Device Answer: " + answer, false);
   SetProperty("Response", answer.c_str());
   if (ret != DEVICE_OK or answer.empty()) {
      return DEVICE_ERR;
   }
   return DEVICE_OK;
}

string MuxWireV3::GetPort() {
   std::string port;
   port = port_;
   return port;
}

int MuxWireV3::SerialReconnection(int maxAttempts) {
   for (int attempt = 1; attempt <= maxAttempts; attempt++) {
      LogMessage("Try to reconnect " + std::to_string(attempt) + " / " + std::to_string(maxAttempts), false);
      // No solution for now
      Sleep(1000); // Attendre 1 seconde entre les tentatives
   }
   LogMessage("Failed to reconnect after " + std::to_string(maxAttempts) + " try", true);
   return DEVICE_ERR;
}

// ---- Nominal mode Control (Thread)
int MuxWireV3::StartTimer() {
   stopTimerThread = false;
   timerThread = CreateThread(NULL, 0, TimerThreadFunction, this, 0, NULL);
   if (timerThread == NULL) {
      LogMessage("Error creating timer thread", true);
      return DEVICE_ERR;
   }
   return DEVICE_OK;
}

int MuxWireV3::StopTimer() {
   if (timerThread != NULL) {
      stopTimerThread = true;
      WaitForSingleObject(timerThread, INFINITE);
      CloseHandle(timerThread);
      timerThread = NULL;
   }
   return DEVICE_OK;
}

DWORD WINAPI MuxWireV3::TimerThreadFunction(LPVOID lpParam) {
   MuxWireV3* pDevice = static_cast<MuxWireV3*>(lpParam);
   return pDevice->ReadTimer(PERIOD_ACQ);
}

int MuxWireV3::ReadTimer(int msInterval) {
   HANDLE timerHandle = CreateWaitableTimer(NULL, FALSE, NULL);
   if (timerHandle == NULL) {
      LogMessage("Error creating timer", true);
      return DEVICE_ERR;
   }
   LARGE_INTEGER due_time;
   due_time.QuadPart = -(10000LL * msInterval);

   if (!SetWaitableTimer(timerHandle, &due_time, msInterval, NULL, NULL, 0)) {
      LogMessage("Error on timer configuration", true);
      CloseHandle(timerHandle);
      return DEVICE_ERR;
   }
   int result;
   string deviceResponse;

   while (initialized_ && timerOn) {
      WaitForSingleObject(timerHandle, INFINITE);

      deviceResponse = GetAllData();
      if (deviceResponse == "ERROR") break;

      result = UpdateAllProperties(deviceResponse);
      if (result != DEVICE_OK) {
         LogMessage("Error update failed", true);
         break;
      }
   }
   CloseHandle(timerHandle);
   return DEVICE_OK;
}
string MuxWireV3::GetAllData() {
   int result;
   char fullResponse[MM::MaxStrLength];
   char pingaResponse[MM::MaxStrLength];
   char trigiResponse[MM::MaxStrLength];
   char trigoResponse[MM::MaxStrLength];
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
   // Get Trigger IN Data
   result = Get("<TRIGI", "");
   GetProperty("Response", trigiResponse);
   if (result != DEVICE_OK) {
      LogMessage("Error on TRIGI data reading, attempting reconnection", true);
      if (SerialReconnection(10) != DEVICE_OK) {
         LogMessage("Reconnection failed after multiple attempts", true);
         return "ERROR";
      }
   }
   // Get Trigger OUT Data
   result = Get("<TRIGO", "");
   GetProperty("Response", trigoResponse);
   if (result != DEVICE_OK) {
      LogMessage("Error on TRIGO data reading, attempting reconnection", true);
      if (SerialReconnection(10) != DEVICE_OK) {
         LogMessage("Reconnection failed after multiple attempts", true);
         return "ERROR";
      }
   }
   sprintf(fullResponse, "%s:%s:%s", pingaResponse, ExtractString(trigiResponse).c_str(), ExtractString(trigoResponse).c_str());
   LogMessage(fullResponse, false);
   return fullResponse;
}

int MuxWireV3::UpdateAllProperties(string deviceResponse) {
   string target = deviceResponse.substr(11);
   vector<string> sensorsDataTitle = { "Type0", "Type1", "Type2", "Type3", "Type4", "Type5", "Type6", "Type7", "Status0", "Status1", "Status2", "Status3", "Status4", "Status5", "Status6", "Status7", "Master0", "Master1", "Master2", "Master3", "Master4", "Master5", "Master6", "Master7", "TriggerIn", "TriggerOut"};
   vector<string> sensorsValue(26);
   istringstream iss(target);
   string token;
   int i = 0;
   while (getline(iss, token, ':') && i < sensorsValue.size()) {
      sensorsValue[i] = token;
      LogMessage(sensorsDataTitle[i] + " : " + sensorsValue[i], false);
      i++;
   }

   // Jump over data corresponding to SensorMaster 
   SetProperty("Type0", sensorsValue[0].c_str());
   SetProperty("Type1", sensorsValue[1].c_str());
   SetProperty("Type2", sensorsValue[2].c_str());
   SetProperty("Type3", sensorsValue[3].c_str());
   SetProperty("Type4", sensorsValue[4].c_str());
   SetProperty("Type5", sensorsValue[5].c_str());
   SetProperty("Type6", sensorsValue[6].c_str());
   SetProperty("Type7", sensorsValue[7].c_str());
   SetProperty("Status0", sensorsValue[8].c_str());
   SetProperty("Status1", sensorsValue[9].c_str());
   SetProperty("Status2", sensorsValue[10].c_str());
   SetProperty("Status3", sensorsValue[11].c_str());
   SetProperty("Status4", sensorsValue[12].c_str());
   SetProperty("Status5", sensorsValue[13].c_str());
   SetProperty("Status6", sensorsValue[14].c_str());
   SetProperty("Status7", sensorsValue[15].c_str());
   SetProperty("Master0", sensorsValue[16].c_str());
   SetProperty("Master1", sensorsValue[17].c_str());
   SetProperty("Master2", sensorsValue[18].c_str());
   SetProperty("Master3", sensorsValue[19].c_str());
   SetProperty("Master4", sensorsValue[20].c_str());
   SetProperty("Master5", sensorsValue[21].c_str());
   SetProperty("Master6", sensorsValue[22].c_str());
   SetProperty("Master7", sensorsValue[23].c_str());
   SetProperty("TriggerIn", sensorsValue[24].c_str());
   SetProperty("TriggerOut", sensorsValue[25].c_str());

   return DEVICE_OK;
}

string MuxWireV3::ExtractString(char* entry_string) {
   const char* last_pipe = strrchr(entry_string, '|');
   if (last_pipe) {
      last_pipe++;
      LogMessage("Extracted value " + string(last_pipe), false);
      return string(last_pipe);
   }
   LogMessage("no val", false);
   return string();
}