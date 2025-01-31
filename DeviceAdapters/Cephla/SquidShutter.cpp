#include "Squid.h"
#include <cstdint>


const char* g_ShutterName = "CephlaShutter";
const char* g_OnOff = "OnOff";
const char* g_Pattern = "Pattern";
const char* g_Intensity = "Intensity";
const char* g_Red = "Red";
const char* g_Green = "Green";
const char* g_Blue = "Blue";
const char* g_HasLasers = "Has Lasers";
const char* g_488 = "488nm";

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
   "LED-Full",
   "LED-Left_Half",
   "LED-Right_Half",
   "LED-Left-Blue_Right-Red",
   "LED-Low_NA",
   "LED-Left_Dot",
   "LED-Right_Dot"
};

// laser IDs start at 11
const uint8_t NR_LASERS = 5;
const std::string LASERS[NR_LASERS] = {
      "405nm",
      "488nm",
      "638nm",
      "561nm",
      "730nm"
};

const int illumination_source = 1; // presumably this is the lED, with lasers something else


SquidShutter::SquidShutter() :
   hub_(0),
   initialized_(false),
   hasLasers_(false),
   name_(g_ShutterName),
   pattern_(0),
   changedTime_(), 
   intensity_ (1),
   red_(255),
   green_(255),
   blue_(255),
   iLaser_(),
   isOpen_(false),
   cmdNr_(0)
{
   for (int i = 0; i < NR_LASERS; i++)
      iLaser_[i] = 0;

   InitializeDefaultErrorMessages();
   EnableDelay();

   SetErrorText(ERR_NO_PORT_SET, "Hub Device not found.  The Squid Hub device is needed to create this device");

   // Name
   int ret = CreateProperty(MM::g_Keyword_Name, g_ShutterName, MM::String, true);
   assert(DEVICE_OK == ret);

   // Description
   ret = CreateProperty(MM::g_Keyword_Description, "Squid Light Control", MM::String, true);
   assert(DEVICE_OK == ret);

   CPropertyAction* pAct = new CPropertyAction(this, &SquidShutter::OnHasLasers);
   ret = CreateStringProperty(g_HasLasers, g_No, false, pAct, true);
   AddAllowedValue(g_HasLasers, g_No);
   AddAllowedValue(g_HasLasers, g_Yes);

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
   hub_ = static_cast<SquidHub*>(GetParentHub());
   if (!hub_ || !hub_->IsPortAvailable()) {
      return ERR_NO_PORT_SET;
   }
   char hubLabel[MM::MaxStrLength];
   hub_->GetLabel(hubLabel);

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

   // Pattern
   // ------
   pAct = new CPropertyAction(this, &SquidShutter::OnPattern);
   ret = CreateProperty(g_Pattern, ILLUMINATIONS[0].c_str(), MM::String, false, pAct);
   if (ret != DEVICE_OK)
      return ret;

   for (uint8_t i = 0; i < 7; i++)
   {
      AddAllowedValue(g_Pattern, ILLUMINATIONS[i].c_str());
   }
   if (hasLasers_) {
      for (uint8_t i = 0; i < 5; i++)
      {
         AddAllowedValue(g_Pattern, LASERS[i].c_str());
      }
   }

   // Intensity
   // ------
   pAct = new CPropertyAction(this, &SquidShutter::OnIntensity);
   ret = CreateProperty(g_Intensity, "1", MM::Integer, false, pAct);
   if (ret != DEVICE_OK)
      return ret;
   SetPropertyLimits(g_Intensity, 0, 255);

   // Red
   // ------
   pAct = new CPropertyAction(this, &SquidShutter::OnRed);
   ret = CreateProperty(g_Red, "255", MM::Integer, false, pAct);
   if (ret != DEVICE_OK)
      return ret;
   SetPropertyLimits(g_Red, 0, 255);

   // Green
   // ------
   pAct = new CPropertyAction(this, &SquidShutter::OnGreen);
   ret = CreateProperty(g_Green, "255", MM::Integer, false, pAct);
   if (ret != DEVICE_OK)
      return ret;
   SetPropertyLimits(g_Green, 0, 255);

   // Blue
   // ------
   pAct = new CPropertyAction(this, &SquidShutter::OnBlue);
   ret = CreateProperty(g_Blue, "255", MM::Integer, false, pAct);
   if (ret != DEVICE_OK)
      return ret;
   SetPropertyLimits(g_Blue, 0, 255);


   if (hasLasers_)
   {
      // Create Laser intensity properties for the known laser lines
      for (uint8_t i = 0; i < NR_LASERS; i++)
      {
         CPropertyActionEx* pActEx = new CPropertyActionEx(this, &SquidShutter::OnLaserIntensity, i);
         ret = CreateProperty(LASERS[i].c_str(), "0", MM::Integer, false, pActEx);
         if (ret != DEVICE_OK)
            return ret;
         SetPropertyLimits(LASERS[i].c_str(), 0, 65535);
         // Set the DACs to 0-2.5V range
         ret = hub_->SetDacGain(i, false);
         if (ret != DEVICE_OK)
            return ret;
      }

   }


   SetOpen(isOpen_);  // we can not read the state from the device, at least get it in sync with us

   //ret = UpdateStatus();
   //if (ret != DEVICE_OK)
   //   return ret;

   changedTime_ = GetCurrentMMTime();
   initialized_ = true;

   return DEVICE_OK;
}


// TODO: figure out how to get a real Busy signal
bool SquidShutter::Busy()
{
   return false;
   //return hub_->IsCommandPending(cmdNr_);
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

int SquidShutter::OnOnOff(MM::PropertyBase* pProp, MM::ActionType eAct)
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

      ret = hub_->SendCommand(cmd, cmdSize);
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
      if (pattern_ < 7)
      {
         pProp->Set(ILLUMINATIONS[pattern_].c_str());
      }
      else  if (pattern_ > 10 && pattern_ < 16)// Laser
      {
         pProp->Set(LASERS[pattern_ - 11].c_str());
      }
      else {
         return DEVICE_INVALID_PROPERTY_VALUE;
      }
   }
   else if (eAct == MM::AfterSet)
   {
      std::string illumination;
      pProp->Get(illumination);
      bool isOpen = isOpen_;
      for (uint8_t i = 0; i < 7; i++)
      {
         if (ILLUMINATIONS[i] == illumination) {
            pattern_ = i;
            if (isOpen)
               SetOpen(false);
            int ret  = sendIllumination(pattern_, intensity_, red_, green_, blue_);
            if (isOpen)
               SetOpen(true);
            return ret;
         }
      }
      for (uint8_t i = 0; i < NR_LASERS; i++)
      {
         if (LASERS[i] == illumination)
         {
            pattern_ = i + 11;
            if (isOpen)
               SetOpen(false);
            int ret = sendLaserIllumination(pattern_, iLaser_[i]);
            if (isOpen)
               SetOpen(true);
            return ret;
         }
      }
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

int SquidShutter::OnLaserIntensity(MM::PropertyBase* pProp, MM::ActionType eAct, long i)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set((long(iLaser_[i])));
   }
   else if (eAct == MM::AfterSet)
   {
      long pos;
      pProp->Get(pos);
      iLaser_[i] = (uint16_t)pos;
      if (pattern_ == i + 11)
      {
         return sendLaserIllumination(pattern_, iLaser_[i]);
      }
   }
   return DEVICE_OK;
}

int SquidShutter::OnHasLasers(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(hasLasers_ ? "Yes" : "No");
   }
   else if (eAct == MM::AfterSet)
   {
      std::string ans;
      pProp->Get(ans);
      hasLasers_ = ans == "Yes";
   }
   return DEVICE_OK;
}


int SquidShutter::sendIllumination(uint8_t pattern, uint8_t intensity, uint8_t red, uint8_t green, uint8_t blue)
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

   int ret = hub_->SendCommand(cmd, cmdSize);
   if (ret != DEVICE_OK)
      return ret;
   changedTime_ = GetCurrentMMTime();

   return DEVICE_OK;
}

int SquidShutter::sendLaserIllumination(uint8_t pattern, uint16_t intensity)
{
   const unsigned cmdSize = 8;
   unsigned char cmd[cmdSize];
   for (unsigned i = 0; i < cmdSize; i++) {
      cmd[i] = 0;
   }
   cmd[1] = CMD_SET_ILLUMINATION;
   cmd[2] = pattern;
   cmd[3] = (intensity >> 8) & 0xff;
   cmd[4] = intensity & 0xff;

   int ret = hub_->SendCommand(cmd, cmdSize);
   if (ret != DEVICE_OK)
      return ret;
   changedTime_ = GetCurrentMMTime();

   return DEVICE_OK;
}


