#ifndef _OPENUC2_SHUTTER_H_
#define _OPENUC2_SHUTTER_H_


#include "DeviceBase.h"
#include "openUC2.h"
#include <string>

class UC2Hub;

class UC2Shutter : public CShutterBase<UC2Shutter>
{
public:
   UC2Shutter();
   ~UC2Shutter();

   // MMDevice API
   int  Initialize();
   int  Shutdown();
   void GetName(char* name) const { CDeviceUtils::CopyLimitedString(name, g_ShutterName); }
   bool Busy();

   // Shutter API
   int SetOpen(bool open);
   int GetOpen(bool& open);
   int Fire(double /*deltaT*/) { return DEVICE_UNSUPPORTED_COMMAND; }

private:
   bool     initialized_;
   UC2Hub*  hub_;
   bool     open_;
};

#endif
