#include "squid.h"
#include "crc8.h"

const char* g_Squid = "Squid";


SquidHub::SquidHub() :
   initialized_(false)
{
   InitializeDefaultErrorMessages();


}

SquidHub::~SquidHub()
{
}

void SquidHub::GetName(char* name) const
{
   CDeviceUtils::CopyLimitedString(name, g_Squid);
}

int SquidHub::Initialize() {
   return DEVICE_OK;
}

int SquidHub::Shutdown() {
   return DEVICE_OK;
}

bool SquidHub::Busy()
{
    return false;
}


int SquidHub::OnPort(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet) {
      pProp->Set(port_.c_str());
   }
   else if (eAct == MM::AfterSet) {
      if (initialized_) {
         // revert
         pProp->Set(port_.c_str());
         return ERR_PORT_CHANGE_FORBIDDEN;
      }
      // take this port.  TODO: should we check if this is a valid port?
      pProp->Get(port_);
   }

   return DEVICE_OK;
}