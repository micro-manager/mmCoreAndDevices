#include "squid.h"
#include <cstdint>


const char* g_ShutterName = "LEDs";

extern const int CMD_TURN_ON_ILLUMINATION;
extern const int CMD_TURN_OFF_ILLUMINATION;
extern const int CMD_SET_ILLUMINATION_LED_MATRIX;

extern const int ILLUMINATION_SOURCE_LED_ARRAY_FULL;
extern const int ILLUMINATION_SOURCE_LED_ARRAY_LEFT_HALF;
extern const int ILLUMINATION_SOURCE_LED_ARRAY_RIGHT_HALF;
extern const int ILLUMINATION_SOURCE_LED_ARRAY_LEFTB_RIGHTR;
extern const int ILLUMINATION_SOURCE_LED_ARRAY_LOW_NA;
extern const int ILLUMINATION_SOURCE_LED_ARRAY_LEFT_DOT;
extern const int ILLUMINATION_SOURCE_LED_ARRAY_RIGHT_DOT;

const std::string ILLUMINATIONS[7] = {
   "Full",
   "Left_Half",
   "Right_Half",
   "Left-Blue_Right-Red",
   "Low_NA",
   "Left_Dot",
   "Right_Dot"
};

const int illumination_source = 1; // presumably this is the lED, with lasers something else


SquidShutter::SquidShutter() :
   initialized_(false),
   name_(g_ShutterName),
   pattern_(0),
   changedTime_(), 
   intensity_ (1),
   red_(255),
   green_(255),
   blue_(255)
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
   return DEVICE_OK;
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
      //pProp->Set((long)hub->GetShutterState());
      pProp->Set(1l);
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

      ret = hub->SendCommand(cmd, cmdSize);
      if (ret != DEVICE_OK)
         return ret;
      changedTime_ = GetCurrentMMTime();
   }
   return DEVICE_OK;
}


int SquidShutter::OnPattern(MM::PropertyBase* pProp, MM::ActionType eAct) 
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(ILLUMINATIONS[pattern_].c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      long pos;
      pProp->Get(pos);
      if (pos >= 0 && pos <= 7)
      {
         pattern_ = (uint8_t) pos;
      }
      return sendIllumination(pattern_, intensity_, red_, green_, blue_);
   }
   return DEVICE_OK;
}

int SquidShutter::OnIntensity(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set((long) intensity_);
   }
   else if (eAct == MM::AfterSet)
   {
      long pos;
      pProp->Get(pos);
      if (pos >= 0 && pos <= 255)
      {
         intensity_ = (uint8_t)pos;
      }
      return sendIllumination(pattern_, intensity_, red_, green_, blue_);
   }
   return DEVICE_OK;
}

int SquidShutter::OnRed(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set((long) red_);
   }
   else if (eAct == MM::AfterSet)
   {
      long pos;
      pProp->Get(pos);
      if (pos >= 0 && pos <= 255)
      {
         red_ = (uint8_t) pos;
      }
      return sendIllumination(pattern_, intensity_, red_, green_, blue_);
   }
   return DEVICE_OK;
}

int SquidShutter::OnGreen(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set((long) green_);
   }
   else if (eAct == MM::AfterSet)
   {
      long pos;
      pProp->Get(pos);
      if (pos >= 0 && pos <= 255)
      {
         green_ = (uint8_t) pos;
      }
      return sendIllumination(pattern_, intensity_, red_, green_, blue_);
   }
   return DEVICE_OK;
}
int SquidShutter::OnBlue(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set((long) blue_);
   }
   else if (eAct == MM::AfterSet)
   {
      long pos;
      pProp->Get(pos);
      if (pos >= 0 && pos <= 255)
      {
         blue_ = (uint8_t) pos;
      }
      return sendIllumination(pattern_, intensity_, red_, green_, blue_);
   }
   return DEVICE_OK;
}


int SquidShutter::sendIllumination(uint8_t pattern, uint8_t intensity, uint8_t red, uint8_t green, uint8_t blue)
{
   SquidHub* hub = static_cast<SquidHub*>(GetParentHub());

   const unsigned cmdSize = 8;
   unsigned char cmd[cmdSize];
   for (unsigned i = 0; i < cmdSize; i++) {
      cmd[i] = 0;
   }
   cmd[1] = CMD_SET_ILLUMINATION_LED_MATRIX;
   cmd[2] = pattern;
   cmd[3] = intensity / 255 * red;
   cmd[4] = intensity / 255 * green;
   cmd[5] = intensity / 255 * blue;

   int ret = hub->SendCommand(cmd, cmdSize);
   if (ret != DEVICE_OK)
      return ret;
   changedTime_ = GetCurrentMMTime();

   return DEVICE_OK;
}





