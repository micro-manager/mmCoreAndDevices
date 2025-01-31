#include "DeviceBase.h"
#include "DeviceUtils.h"
#include <vector>

using namespace std;

class MuxWireV3 : public CGenericBase<MuxWireV3> {
public:
   MuxWireV3();
   ~MuxWireV3();

   // MMDevice API
   int Initialize() override;
   int Shutdown() override;
   void GetName(char* name) const override;
   bool Busy() override;

   // MM - Mux Wire Custom methods
   MM::DeviceDetectionStatus DetectDevice(void);
   int DetectInstalledDevices();
   int OnPort(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnStart(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnIsGet(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnCommand(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnTrigger(MM::PropertyBase* pProp, MM::ActionType eAct);

   // Mux Wire Custom methods
   //// ---- Raw UART Control
   string FormatMsg(const string& uartCmd, const string& parameters, bool isGet);
   int Get(const string& uartCmd, const string& parameters);
   int Set(const string& uartCmd, const string& parameters);
   int ReadResponse();
   string GetPort();

private:
   bool isGet_;
   bool initialized_;
   string port_;           // Name of the COM port

   bool timerOn;
   HANDLE timerThread;
   bool stopTimerThread;

   // Mux Wire Custom methods
   // ----  Device COM configuration
   int SerialReconnection(int maxAttempts);

   // ---- Nominal mode Control (Thread)
   int StartTimer();
   int StopTimer();
   static DWORD WINAPI TimerThreadFunction(LPVOID lpParam);
   int ReadTimer(int msInterval);
   string GetAllData();
   int UpdateAllProperties(string ob1Response);
   string ExtractString(char* entryString);
};
