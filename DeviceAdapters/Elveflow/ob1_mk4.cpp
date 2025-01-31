#include "ModuleInterface.h"
#include "ob1_mk4.h"
#include "iostream"
#include <vector>
#include <string>
#include <windows.h>

using namespace std;

#define PERIOD_ACQ 1000
#define ERR_PORT_CHANGE_FORBIDDEN	101 

const char* g_Default_String_Ob1 = "Unknown";

Ob1Mk4::Ob1Mk4() : initialized_(false), isGet_(true), timerOn(false), timerThread(NULL), stopTimerThread(false) {
   // Device COM configuration
   CPropertyAction* pAct = new CPropertyAction(this, &Ob1Mk4::OnPort);
   CreateProperty(MM::g_Keyword_Port, g_Default_String_Ob1, MM::String, false, pAct, true);
    
   // Raw UART Control
   CreateProperty("IsGet", "0", MM::Integer, false, new CPropertyAction(this, &Ob1Mk4::OnIsGet));
   CreateProperty("Command", "", MM::String, false, new CPropertyAction(this, &Ob1Mk4::OnCommand));
   CreateProperty("Parameters", "", MM::String, false);
   CreateProperty("Response", "", MM::String, false);

   // Nominal Operation Control

   // ---- Sensor of Regulators Current Value
   CreateProperty("RegulatorValue0", "0", MM::Integer, false);
   CreateProperty("RegulatorValue1", "0", MM::Integer, false);
   CreateProperty("RegulatorValue2", "0", MM::Integer, false);
   CreateProperty("RegulatorValue3", "0", MM::Integer, false);
    
   // ---- Sensors Current Value
   CreateProperty("SensorValue0", "0", MM::Integer, false);
   CreateProperty("SensorValue1", "0", MM::Integer, false);
   CreateProperty("SensorValue2", "0", MM::Integer, false);
   CreateProperty("SensorValue3", "0", MM::Integer, false);

   // ---- Pressure Setpoint
   CreateProperty("PressureSetpoint0", "0", MM::Integer, false, new CPropertyAction(this, &Ob1Mk4::OnPressureSetpoint0));
   CreateProperty("PressureSetpoint1", "0", MM::Integer, false, new CPropertyAction(this, &Ob1Mk4::OnPressureSetpoint1));
   CreateProperty("PressureSetpoint2", "0", MM::Integer, false, new CPropertyAction(this, &Ob1Mk4::OnPressureSetpoint2));
   CreateProperty("PressureSetpoint3", "0", MM::Integer, false, new CPropertyAction(this, &Ob1Mk4::OnPressureSetpoint3));

   // ---- Trigger Current Value
   CreateProperty("TriggerIn", "0", MM::Integer, false);
   CreateProperty("TriggerOut", "0", MM::Integer, false, new CPropertyAction(this, &Ob1Mk4::OnTrigger));

   // ---- Timer Control
   CreateProperty("TimerOn", "0", MM::Integer, false, new CPropertyAction(this, &Ob1Mk4::OnStart));
}


Ob1Mk4::~Ob1Mk4() {
   if (initialized_)
      Shutdown();
}

// MMDevice API
int Ob1Mk4::Initialize() {
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

int Ob1Mk4::Shutdown() {
   if (initialized_) {
      LogMessage("Device Shutdown", false);
      StopTimer();
      initialized_ = false;
   }
   return DEVICE_OK;
}

bool Ob1Mk4::Busy() {
   return false;
}

void Ob1Mk4::GetName(char* name) const {
   CDeviceUtils::CopyLimitedString(name, "OB1_MK4");
}

// MM - Ob1 Mk4 Custom methods
MM::DeviceDetectionStatus Ob1Mk4::DetectDevice(void)
{
   if (initialized_) {
      return MM::CanCommunicate;
   } 
   // all conditions must be satisfied...
   MM::DeviceDetectionStatus result = MM::Misconfigured;
   char answerTo[MM::MaxStrLength];

   try {
      std::string portLowerCase = GetPort();
      for (std::string::iterator its = portLowerCase.begin(); its != portLowerCase.end(); ++its) {
         *its = (char)tolower(*its);
      }
      if (0 < portLowerCase.length() && 0 != portLowerCase.compare("undefined") && 0 != portLowerCase.compare("unknown")) {
         result = MM::CanNotCommunicate;
         // record the default answer time out
         GetCoreCallback()->GetDeviceProperty(GetPort().c_str(), "AnswerTimeout", answerTo);
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
            GetCoreCallback()->SetDeviceProperty(GetPort().c_str(), "AnswerTimeout", answerTo);
         }
      }
   }
   catch (...) {
      //LogMessage("Exception in DetectDevice!",false);
   }

   return result;
}

int Ob1Mk4::DetectInstalledDevices() {
   if (MM::CanCommunicate == DetectDevice()) {
      std::vector<std::string> peripherals;
      peripherals.clear();
      peripherals.push_back("OB1_MK4");
      for (size_t i = 0; i < peripherals.size(); i++) {
         MM::Device* pDev = ::CreateDevice(peripherals[i].c_str());
         if (pDev) {
            LogMessage("Device created");
         }
      }
   }
   return DEVICE_OK;
}

int Ob1Mk4::OnPort(MM::PropertyBase* pProp, MM::ActionType eAct) {
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

int Ob1Mk4::OnStart(MM::PropertyBase* pProp, MM::ActionType eAct) {
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

int Ob1Mk4::OnIsGet(MM::PropertyBase* pProp, MM::ActionType eAct) {
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

int Ob1Mk4::OnCommand(MM::PropertyBase* pProp, MM::ActionType eAct) {
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

int Ob1Mk4::OnPressureSetpoint0(MM::PropertyBase* pProp, MM::ActionType eAct) {
   if (eAct == MM::AfterSet) {
      long value;
      pProp->Get(value);
      int result = Set("<PRESS", ":0:" + to_string(value));
      string currentData = GetAllData();
      if (UpdateAllProperties(currentData) != DEVICE_OK) {
         LogMessage("Failed to update properties 1s after");
         return DEVICE_ERR;
      }
      return result;
   }
   return DEVICE_OK;
}

int Ob1Mk4::OnPressureSetpoint1(MM::PropertyBase* pProp, MM::ActionType eAct) {
   if (eAct == MM::AfterSet) {
      long value;
      pProp->Get(value);
      int result = Set("<PRESS", ":1:" + to_string(value));
      string currentData = GetAllData();
      if (UpdateAllProperties(currentData) != DEVICE_OK) {
         LogMessage("Failed to update properties 1s after");
         return DEVICE_ERR;
      }
      return result;
   }
   return DEVICE_OK;
}

int Ob1Mk4::OnPressureSetpoint2(MM::PropertyBase* pProp, MM::ActionType eAct) {
   if (eAct == MM::AfterSet) {
      long value;
      pProp->Get(value);
      int result = Set("<PRESS", ":2:" + to_string(value));
      string currentData = GetAllData();
      if (UpdateAllProperties(currentData) != DEVICE_OK) {
         LogMessage("Failed to update properties 1s after");
         return DEVICE_ERR;
      }
      return result;
   }
   return DEVICE_OK;
}

int Ob1Mk4::OnPressureSetpoint3(MM::PropertyBase* pProp, MM::ActionType eAct) {
   if (eAct == MM::AfterSet) {
      long value;
      pProp->Get(value);
      int result = Set("<PRESS", ":3:" + to_string(value));
      string currentData = GetAllData();
      if (UpdateAllProperties(currentData) != DEVICE_OK) {
         LogMessage("Failed to update properties 1s after");
         return DEVICE_ERR;
      }
      return result;
   }
   return DEVICE_OK;
}

int Ob1Mk4::OnTrigger(MM::PropertyBase* pProp, MM::ActionType eAct) {
   if (eAct == MM::AfterSet) {
      long value;
      pProp->Get(value);
      return Set("<TRIGO", to_string(value));
   }
   return DEVICE_OK;
}

// Ob1 Mk4 Custom methods
// ---- Raw UART Control
string Ob1Mk4::FormatMsg(const string& uartCmd, const string& parameters, bool isGet) {
   string buffer = isGet ? uartCmd + "?" : uartCmd + "!";
   if (!parameters.empty()) {
      buffer += parameters;
   }  
   //buffer += "\n";
   LogMessage("Cmd sent: " + buffer, false);
   return buffer;
}

int Ob1Mk4::Get(const string& uartCmd, const string& parameters) {
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

int Ob1Mk4::Set(const string& uartCmd, const string& parameters) {
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

int Ob1Mk4::ReadResponse() {
   string answer;
   int ret = GetSerialAnswer(port_.c_str(), "\n", answer);
   LogMessage("Device Answer: " + answer, false);
   SetProperty("Response", answer.c_str());
   if (ret != DEVICE_OK or answer.empty()) {
      return DEVICE_ERR;
   }
   return DEVICE_OK;
}

string Ob1Mk4::GetPort() {
   std::string port;
   port = port_;
   return port;
}

int Ob1Mk4::SerialReconnection(int maxAttempts) {
   for (int attempt = 1; attempt <= maxAttempts; attempt++) {
      LogMessage("Try to reconnect " + std::to_string(attempt) + " / " + std::to_string(maxAttempts), false);
      // No solution for now
      Sleep(1000); // Attendre 1 seconde entre les tentatives
   }
   LogMessage("Failed to reconnect after " + std::to_string(maxAttempts) + " try", true);
   return DEVICE_ERR;
}

// ---- Nominal mode Control (Thread)
int Ob1Mk4::StartTimer() {
   stopTimerThread = false;
   timerThread = CreateThread(NULL, 0, TimerThreadFunction, this, 0, NULL);
   if (timerThread == NULL) {
      LogMessage("Error on timer's thread making", true);
      return DEVICE_ERR;
   }
   return DEVICE_OK;
}

int Ob1Mk4::StopTimer() {
   if (timerThread != NULL) {
      stopTimerThread = true;
      WaitForSingleObject(timerThread, INFINITE);
      CloseHandle(timerThread);
      timerThread = NULL;
   }
   return DEVICE_OK;
}

DWORD WINAPI Ob1Mk4::TimerThreadFunction(LPVOID lpParam) {
   Ob1Mk4* pDevice = static_cast<Ob1Mk4*>(lpParam);
   return pDevice->ReadTimer(PERIOD_ACQ);
}

int Ob1Mk4::ReadTimer(int msInterval) {
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
   string ob1Response;

   while (initialized_ && timerOn) {
      WaitForSingleObject(timerHandler, INFINITE);

      ob1Response = GetAllData();
      if (ob1Response == "ERROR") {
         break;
      }
      result = UpdateAllProperties(ob1Response);
      if (result != DEVICE_OK) {
         LogMessage("Error update failed", true);
         break;
      }
   }

   CloseHandle(timerHandler);
   return DEVICE_OK;
}
string Ob1Mk4::GetAllData() {
   int result;
   char fullResponse[MM::MaxStrLength];
   char pingaResponse[MM::MaxStrLength];
   char trigiResponse[MM::MaxStrLength];
   char trigoResponse[MM::MaxStrLength];
   // Get PingA Data
   result = Get("<PINGA", "");
   GetProperty("Response", pingaResponse);
   if (result != DEVICE_OK) {
      LogMessage("Error on PINGA data reading, attempting reconnection", true);
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

int Ob1Mk4::UpdateAllProperties(string ob1Response) {
   string target = ob1Response.substr(11);
   vector<string> sensorsDataTitle = { "RegulatorValue0", "SensorValue0", "SensorType0", "RegulatorValue1", "SensorValue1", "SensorType1", "RegulatorValue2", "SensorValue2", "SensorType2", "RegulatorValue3", "SensorValue3", "SensorType3", "TriggerIn", "TriggerOut"};
   vector<string> sensorsValue(14);
   istringstream iss(target);
   string token;

   int i = 0;
   while (getline(iss, token, ':') && i < sensorsValue.size()) {
      sensorsValue[i] = token;
      LogMessage(sensorsDataTitle[i] + " : " + sensorsValue[i], false);
      i++;
   }
   // Jump over data corresponding to SensorType 
   SetProperty("RegulatorValue0", sensorsValue[0].c_str());
   SetProperty("SensorValue0", sensorsValue[1].c_str());
   SetProperty("RegulatorValue1", sensorsValue[3].c_str());
   SetProperty("SensorValue1", sensorsValue[4].c_str());
   SetProperty("RegulatorValue2", sensorsValue[6].c_str());
   SetProperty("SensorValue2", sensorsValue[7].c_str());
   SetProperty("RegulatorValue3", sensorsValue[9].c_str());
   SetProperty("SensorValue3", sensorsValue[10].c_str());
   SetProperty("TriggerIn", sensorsValue[12].c_str());
   SetProperty("TriggerOut", sensorsValue[13].c_str());
   return DEVICE_OK;
}

string Ob1Mk4::ExtractString(char* entryString) {
   const char* last_Pipe = strrchr(entryString, '|');
   if (last_Pipe) {
       last_Pipe++;
      LogMessage("Extracted value " + string(last_Pipe), false);
      return string(last_Pipe);
   }
   LogMessage("no val", false);
   return string();
}