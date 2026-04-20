#include "DeviceBase.h"
#include "DeviceUtils.h"
#include <vector>

using namespace std;

class MuxDistrib : public CGenericBase<MuxDistrib> {
public:
   MuxDistrib();
   ~MuxDistrib();

   // MMDevice API
   int Initialize() override;
   int Shutdown() override;
   void GetName(char* name) const override;
   bool Busy() override;

   // MM - Mux Distrib Custom methods
   MM::DeviceDetectionStatus DetectDevice(void);
   int DetectInstalledDevices();
   int OnPort(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnPos(MM::PropertyBase* pProp, MM::ActionType eAct);

   // Mux Distrib Custom methods
   // ---- Raw UART Control
   int Get(const string& uartCmd, const string& parameters);
   int Set(const string& uartCmd, const string& parameters);
   int ReadResponse();
   string GetPort();

   private:
   bool initialized_;
   string port_;           // Name of the COM port

   // Mux Distrib Custom methods
   // ----  Device COM configuration
   int SerialReconnection(int maxAttempts);
};
