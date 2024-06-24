#ifndef _SQUID_H_
#define _SQUID_H_

#include "MMDevice.h"
#include "DeviceBase.h"


#define ERR_PORT_CHANGE_FORBIDDEN    21001 

class SquidHub : public HubBase<SquidHub>
{
public:
   SquidHub();
   ~SquidHub();

   int Initialize();
   int Shutdown();
   void GetName(char* pszName) const;
   bool Busy();

   bool SupportsDeviceDetection(void);
   MM::DeviceDetectionStatus DetectDevice(void);
   int DetectInstalledDevices();

   int OnPort(MM::PropertyBase* pProp, MM::ActionType eAct);


private:
   uint8_t crc8ccitt(const void* data, size_t size);
   int sendCommand(unsigned char* cmd, unsigned cmdSize);
   bool initialized_;
   std::string port_;
};


#endif _SQUID_H_