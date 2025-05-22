#ifndef _UC2HUB_H_
#define _UC2HUB_H_


#include "MMDevice.h"
#include "DeviceBase.h"
#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <mutex>

class UC2Hub : public HubBase<UC2Hub>
{
public:
   UC2Hub();
   ~UC2Hub();

   // MMDevice API:
   int  Initialize();
   int  Shutdown();
   void GetName(char* pszName) const;
   bool Busy();

   // Hub API:
   bool SupportsDeviceDetection(void);
   MM::DeviceDetectionStatus DetectDevice(void);
   int  DetectInstalledDevices();

   // Action handlers
   int  OnPort(MM::PropertyBase* pProp, MM::ActionType eAct);

   // Utility for sub-devices to send JSON
   int  SendJsonCommand(const std::string& jsonCmd, std::string& jsonReply, bool debug=false);

   // Possibly check firmware
   bool CheckFirmware();

   // Provide access to port name
   std::string GetPort() { return port_; }

private:
   bool        initialized_;
   std::string port_;
   std::mutex  ioMutex_;
};

#endif // _UC2HUB_H_
