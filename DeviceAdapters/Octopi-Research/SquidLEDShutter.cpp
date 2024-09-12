#include "squid.h"
#include <cstdint>


const char* g_LEDShutterName = "LEDs";
const char* g_OnOff = "OnOff";
const char* g_Pattern = "Pattern";
const char* g_Intensity = "Intensity";
const char* g_Red = "Red";
const char* g_Green = "Green";
const char* g_Blue = "Blue";

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


SquidLEDShutter::SquidLEDShutter() :
   hub_(0),
   initialized_(false),
   name_(g_LEDShutterName),
   pattern_(0),
   changedTime_(), 
   intensity_ (1),
   red_(255),
   green_(255),
   blue_(255),
   isOpen_(false),
   cmdNr_(0)
{
   InitializeDefaultErrorMessages();
   EnableDelay();

   SetErrorText(ERR_NO_PORT_SET, "Hub Device not found.  The Squid Hub device is needed to create this device");

   // Name
   int ret = CreateProperty(MM::g_Keyword_Name, g_LEDShutterName, MM::String, true);
   assert(DEVICE_OK == ret);

   // Description
   ret = CreateProperty(MM::g_Keyword_Description, "Squid LED-shutter driver", MM::String, true);
   assert(DEVICE_OK == ret);

   // parent ID display
   CreateHubIDProperty();

}


SquidLEDShutter::~SquidLEDShutter()
{
   if (initialized_)
   {
      Shutdown();
   }
}


int SquidLEDShutter::Shutdown()
{
   if (initialized_) {
      initialized_ = false;
   }
   return DEVICE_OK;
}


void SquidLEDShutter::GetName(char* pszName) const
{
   CDeviceUtils::CopyLimitedString(pszName, g_LEDShutterName);
}


int SquidLEDShutter::Initialize()
{
   hub_ = static_cast<SquidHub*>(GetParentHub());
   if (!hub_ || !hub_->IsPortAvailable()) {
      return ERR_NO_PORT_SET;
   }
   char hubLabel[MM::MaxStrLength];
   hub_->GetLabel(hubLabel);

   // OnOff
  // ------
   CPropertyAction* pAct = new CPropertyAction(this, &SquidLEDShutter::OnOnOff);
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

   // Pattern
   // ------
   pAct = new CPropertyAction(this, &SquidLEDShutter::OnPattern);
   ret = CreateProperty(g_Pattern, ILLUMINATIONS[0].c_str(), MM::String, false, pAct);
   if (ret != DEVICE_OK)
      return ret;

   for (uint8_t i = 0; i < 7; i++)
   {
      AddAllowedValue(g_Pattern, ILLUMINATIONS[i].c_str());
   }

   // Intensity
   // ------
   pAct = new CPropertyAction(this, &SquidLEDShutter::OnIntensity);
   ret = CreateProperty(g_Intensity, "1", MM::Integer, false, pAct);
   if (ret != DEVICE_OK)
      return ret;
   SetPropertyLimits(g_Intensity, 0, 255);

   // Red
   // ------
   pAct = new CPropertyAction(this, &SquidLEDShutter::OnRed);
   ret = CreateProperty(g_Red, "255", MM::Integer, false, pAct);
   if (ret != DEVICE_OK)
      return ret;
   SetPropertyLimits(g_Red, 0, 255);

   // Green
   // ------
   pAct = new CPropertyAction(this, &SquidLEDShutter::OnGreen);
   ret = CreateProperty(g_Green, "255", MM::Integer, false, pAct);
   if (ret != DEVICE_OK)
      return ret;
   SetPropertyLimits(g_Green, 0, 255);

   // Blue
   // ------
   pAct = new CPropertyAction(this, &SquidLEDShutter::OnBlue);
   ret = CreateProperty(g_Blue, "255", MM::Integer, false, pAct);
   if (ret != DEVICE_OK)
      return ret;
   SetPropertyLimits(g_Blue, 0, 255);

   SetOpen(isOpen_);  // we can not read the state from the device, at least get it in sync with us

   ret = UpdateStatus();
   if (ret != DEVICE_OK)
      return ret;

   changedTime_ = GetCurrentMMTime();
   initialized_ = true;

   return DEVICE_OK;
}


bool SquidLEDShutter::Busy()
{
   return hub_->IsCommandPending(cmdNr_);
}



int SquidLEDShutter::SetOpen(bool open)
{
   std::ostringstream os;
   os << "Request " << open;
   LogMessage(os.str().c_str(), true);

   if (open)
      return SetProperty("OnOff", "1");
   else
      return SetProperty("OnOff", "0");
}

int SquidLEDShutter::GetOpen(bool& open)
{
   char buf[MM::MaxStrLength];
   int ret = GetProperty("OnOff", buf);
   if (ret != DEVICE_OK)
      return ret;
   long pos = atol(buf);
   pos > 0 ? open = true : open = false;

   return DEVICE_OK;
}

int SquidLEDShutter::Fire(double /*deltaT*/)
{
   return DEVICE_UNSUPPORTED_COMMAND;
}


// action interface

int SquidLEDShutter::OnOnOff(MM::PropertyBase* pProp, MM::ActionType eAct)
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
      const unsigned cmdSize = 8;
      unsigned char cmd[cmdSize];
      for (unsigned i = 0; i < cmdSize; i++) {
         cmd[i] = 0;
      }
      if (pos == 0)
         cmd[1] = CMD_TURN_OFF_ILLUMINATION;
      else
         cmd[1] = CMD_TURN_ON_ILLUMINATION; 

      isOpen_ = pos == 1;

      ret = hub_->SendCommand(cmd, cmdSize, &cmdNr_);
      if (ret != DEVICE_OK)
         return ret;
      changedTime_ = GetCurrentMMTime();
   }
   return DEVICE_OK;
}


int SquidLEDShutter::OnPattern(MM::PropertyBase* pProp, MM::ActionType eAct) 
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(ILLUMINATIONS[pattern_].c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      std::string pattern;
      pProp->Get(pattern);
      for (uint8_t i = 0; i < 7; i++)
      {
         if (ILLUMINATIONS[i] == pattern) {
            pattern_ = i;
            return sendIllumination(pattern_, intensity_, red_, green_, blue_);
         }
      }
   }
   return DEVICE_OK;
}

int SquidLEDShutter::OnIntensity(MM::PropertyBase* pProp, MM::ActionType eAct)
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

int SquidLEDShutter::OnRed(MM::PropertyBase* pProp, MM::ActionType eAct)
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

int SquidLEDShutter::OnGreen(MM::PropertyBase* pProp, MM::ActionType eAct)
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
int SquidLEDShutter::OnBlue(MM::PropertyBase* pProp, MM::ActionType eAct)
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


int SquidLEDShutter::sendIllumination(uint8_t pattern, uint8_t intensity, uint8_t red, uint8_t green, uint8_t blue)
{
   const unsigned cmdSize = 8;
   unsigned char cmd[cmdSize];
   for (unsigned i = 0; i < cmdSize; i++) {
      cmd[i] = 0;
   }
   cmd[1] = CMD_SET_ILLUMINATION_LED_MATRIX;
   cmd[2] = pattern;
   cmd[3] = (uint8_t) ((double) intensity / 255 * green);
   cmd[4] = (uint8_t) ((double) intensity / 255 * red);
   cmd[5] = (uint8_t) ((double) intensity / 255 * blue);

   int ret = hub_->SendCommand(cmd, cmdSize, &cmdNr_);
   if (ret != DEVICE_OK)
      return ret;
   changedTime_ = GetCurrentMMTime();

   return DEVICE_OK;
}