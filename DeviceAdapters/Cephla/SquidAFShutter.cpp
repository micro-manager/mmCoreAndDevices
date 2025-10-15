#include "Squid.h"
#include <cstdint>


const char* g_AFShutterName = "CephlaAFShutter";

extern const int MCU_PINS_AF_LASER;
extern const uint8_t CMD_LENGTH;
extern const int CMD_SET_PIN_LEVEL;


SquidAFShutter::SquidAFShutter() :
   initialized_ (false),
   name_(g_AFShutterName),
   changedTime_(),
   isOpen_(false)
{
   InitializeDefaultErrorMessages();
   EnableDelay();
   CreateHubIDProperty();
}

SquidAFShutter::~SquidAFShutter() {
   if (initialized_) {
      Shutdown();
   }
}

int SquidAFShutter::Shutdown()
{
   if (initialized_) {
      initialized_ = false;
   }
   return DEVICE_OK;
}

void SquidAFShutter::GetName(char* pszName) const
{
   CDeviceUtils::CopyLimitedString(pszName, g_AFShutterName);
}

int SquidAFShutter::Initialize()
{
   hub_ = static_cast<SquidHub*>(GetParentHub());
   if (!hub_ || !hub_->IsPortAvailable()) {
      return ERR_NO_PORT_SET;
   }
   char hubLabel[MM::MaxStrLength];
   hub_->GetLabel(hubLabel);

   // OnOff
  // ------
   CPropertyAction* pAct = new CPropertyAction(this, &SquidAFShutter::OnOnOff);
   int ret = CreateProperty("OnOff", "0", MM::Integer, false, pAct);
   if (ret != DEVICE_OK)
      return ret;

   // set shutter into the off state
   //WriteToPort(0);

   std::vector<std::string> vals;
   vals.push_back("0");
   vals.push_back("1");
   ret = SetAllowedValues("OnOff", vals);
   if (ret != DEVICE_OK)
      return ret;

   SetOpen(isOpen_);  // we can not read the state from the device, at least get it in sync with us

   changedTime_ = GetCurrentMMTime();
   initialized_ = true;

   return DEVICE_OK;
}

bool SquidAFShutter::Busy() {
   return false;
}

// Shutter API
int SquidAFShutter::SetOpen(bool open)
{
   if (open)
      return SetProperty("OnOff", "1");
   else
      return SetProperty("OnOff", "0");
}

int SquidAFShutter::GetOpen(bool& open)
{
   char buf[MM::MaxStrLength];
   int ret = GetProperty("OnOff", buf);
   if (ret != DEVICE_OK)
      return ret;
   long pos = atol(buf);
   pos > 0 ? open = true : open = false;

   return DEVICE_OK;
}

int SquidAFShutter::Fire(double deltaT)
{
   return DEVICE_UNSUPPORTED_COMMAND;
}

// action interface
int SquidAFShutter::OnOnOff(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      // use cached state, there is no way to query
      pProp->Set(isOpen_ ? 1l : 0l);
   }
   else if (eAct == MM::AfterSet)
   {
      long pos;
      pProp->Get(pos);
      int ret;
      unsigned char cmd[CMD_LENGTH];
      for (unsigned i = 0; i < CMD_LENGTH; i++) {
         cmd[i] = 0;
      }
      cmd[1] = CMD_SET_PIN_LEVEL;
      cmd[2] = MCU_PINS_AF_LASER; // pin
      cmd[3] = pos == 0 ? 0 : 1; // level

      isOpen_ = pos == 1;

      ret = hub_->SendCommand(cmd, CMD_LENGTH);
      if (ret != DEVICE_OK)
         return ret;
      changedTime_ = GetCurrentMMTime();
   }
   return DEVICE_OK;
}
