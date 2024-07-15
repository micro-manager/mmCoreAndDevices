#include "squid.h"

extern const char* g_ShutterName;

extern const int CMD_TURN_ON_ILLUMINATION = 10;
extern const int CMD_TURN_OFF_ILLUMINATION = 11;
extern const int CMD_SET_ILLUMINATION = 12;
extern const int CMD_SET_ILLUMINATION_LED_MATRIX = 13;
extern const int CMD_SET_ILLUMINATION_INTENSITY_FACTOR = 17;


SquidShutter::SquidShutter() :
   initialized_(false),
   name_(g_ShutterName),
   changedTime_()
{
   InitializeDefaultErrorMessages();
   EnableDelay();

   SetErrorText(ERR_NO_PORT_SET, "Hub Device not found.  The Squid Hub device is needed to create this device");

   // Name
   int ret = CreateProperty(MM::g_Keyword_Name, g_ShutterName, MM::String, true);
   assert(DEVICE_OK == ret);

   // Description
   ret = CreateProperty(MM::g_Keyword_Description, "Squid LED-shutter driver", MM::String, true);
   assert(DEVICE_OK == ret);

   // parent ID display
   CreateHubIDProperty();

}


SquidShutter::~SquidShutter()
{
   if (initialized_)
   {
      Shutdown();
   }
}


int SquidShutter::Shutdown()
{
   if (initialized_) {
      initialized_ = false;
   }
}


void SquidShutter::GetName(char* pszName) const
{
   CDeviceUtils::CopyLimitedString(pszName, g_ShutterName);
}


int SquidShutter::Initialize()
{
   SquidHub* hub = static_cast<SquidHub*>(GetParentHub());
   if (!hub || !hub->IsPortAvailable()) {
      return ERR_NO_PORT_SET;
   }
   char hubLabel[MM::MaxStrLength];
   hub->GetLabel(hubLabel);

   // OnOff
  // ------
   CPropertyAction* pAct = new CPropertyAction(this, &SquidShutter::OnOnOff);
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

   ret = UpdateStatus();
   if (ret != DEVICE_OK)
      return ret;

   changedTime_ = GetCurrentMMTime();
   initialized_ = true;

   return DEVICE_OK;
}


bool SquidShutter::Busy()
{
   // TODO:
   return false;
}



int SquidShutter::SetOpen(bool open)
{
   std::ostringstream os;
   os << "Request " << open;
   LogMessage(os.str().c_str(), true);

   if (open)
      return SetProperty("OnOff", "1");
   else
      return SetProperty("OnOff", "0");
}

int SquidShutter::GetOpen(bool& open)
{
   char buf[MM::MaxStrLength];
   int ret = GetProperty("OnOff", buf);
   if (ret != DEVICE_OK)
      return ret;
   long pos = atol(buf);
   pos > 0 ? open = true : open = false;

   return DEVICE_OK;
}

int SquidShutter::Fire(double /*deltaT*/)
{
   return DEVICE_UNSUPPORTED_COMMAND;
}


// action interface
//int OnState(MM::PropertyBase* pProp, MM::ActionType eAct);

int SquidShutter::OnOnOff(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   SquidHub* hub = static_cast<SquidHub*>(GetParentHub());
   if (eAct == MM::BeforeGet)
   {
      // use cached state
      pProp->Set((long)hub->GetShutterState());
   }
   else if (eAct == MM::AfterSet)
   {
      long pos;
      pProp->Get(pos);
      int ret;
      const unsigned cmdSize = 8;
      unsigned char cmd[cmdSize];
      for (unsigned i = 0; i < cmdSize; i++) {
         cmd[i] = 0;
      }
      if (pos == 0)
         cmd[1] = CMD_TURN_OFF_ILLUMINATION;
      else
         cmd[1] = CMD_TURN_ON_ILLUMINATION; 

      ret = hub->sendCommand(cmd, cmdSize);
      if (ret != DEVICE_OK)
         return ret;
      changedTime_ = GetCurrentMMTime();
   }
   return DEVICE_OK;
}





