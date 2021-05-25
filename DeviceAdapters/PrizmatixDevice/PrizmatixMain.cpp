#include "PrizmatixMain.h"
#ifdef WIN32
   #define WIN32_LEAN_AND_MEAN
   #include <windows.h>
#endif
#include "FixSnprintf.h"


const char* g_DeviceNameHub = "prizmatix-Hub";
const char* g_DeviceOneLED = "Prizmatix Ctrl";
const char* g_DeviceMultiLED_2 = "Prizmatix Ctrl(2)";
const char* g_DeviceMultiLED_3 = "Prizmatix Ctrl(3)";
const char* g_DeviceMultiLED_4 = "Prizmatix Ctrl(4)";
const char* g_DeviceMultiLED_5 = "Prizmatix Ctrl(5)";
const char* g_DeviceMultiLED_6 = "Prizmatix Ctrl(6)";
const char* g_DeviceMultiLED_7 = "Prizmatix Ctrl(7)";
const char* g_DeviceMultiLED_8 = "Prizmatix Ctrl(8)";
const char* g_DeviceMultiLED_9 = "Prizmatix Ctrl(9)";
const char* g_DeviceMultiLED_10 = "Prizmatix Ctrl(10)";
const char* g_On = "On";
const char* g_Off = "Off";
   static MMThreadLock lock_;
///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////
MODULE_API void InitializeModuleData()
{
   RegisterDevice(g_DeviceNameHub, MM::HubDevice, "Hub (required)");
}
   
  
MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
   if (deviceName == 0)
      return 0;

   if (strcmp(deviceName, g_DeviceNameHub) == 0)
   {
      return new PrizmatixHub;
   }
   else if (strcmp(deviceName, g_DeviceOneLED) == 0)
   {
      return new PrizmatixLED(1,(char *)deviceName);
   }
   else if (strcmp(deviceName, g_DeviceMultiLED_2) == 0)
   {
     return new PrizmatixLED(2,(char *)deviceName);
   }
      else if (strcmp(deviceName, g_DeviceMultiLED_3) == 0)
   {
     return new PrizmatixLED(3,(char *)deviceName);
   }
    else if (strcmp(deviceName, g_DeviceMultiLED_4) == 0)
   {
     return new PrizmatixLED(4,(char *)deviceName);
   }   	  
	    else if (strcmp(deviceName, g_DeviceMultiLED_5) == 0)
   {
     return new PrizmatixLED(5,(char *)deviceName);
   }
	  else if (strcmp(deviceName, g_DeviceMultiLED_6) == 0)
   {
     return new PrizmatixLED(6,(char *)deviceName);
   }
		  else if (strcmp(deviceName, g_DeviceMultiLED_7) == 0)
   {
     return new PrizmatixLED(7,(char *)deviceName);
   }
	    else if (strcmp(deviceName, g_DeviceMultiLED_8) == 0)
   {
     return new PrizmatixLED(8,(char *)deviceName);
   }
		    else if (strcmp(deviceName, g_DeviceMultiLED_9) == 0)
   {
     return new PrizmatixLED(9,(char *)deviceName);
   }
		  else if (strcmp(deviceName, g_DeviceMultiLED_10) == 0)
   {
     return new PrizmatixLED(10,(char *)deviceName);
   }
   return 0;
}


MODULE_API void DeleteDevice(MM::Device* pDevice)
{
   delete pDevice;
}

///////////////////////////////////////////////////////////////////////////////
// PrizmatixHub implementation
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
//

///////////////////////////////////////////////////////////////////////////////
// PrizmatixHub implementation
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
//
PrizmatixHub::PrizmatixHub() :
   initialized_ (false),
   switchState_ (0),
   shutterState_ (0)
{
   portAvailable_ = false;
   invertedLogic_ = false;
   timedOutputActive_ = false;

   InitializeDefaultErrorMessages();

   SetErrorText(ERR_PORT_OPEN_FAILED, "Failed opening Prizmatix USB device");
   SetErrorText(ERR_BOARD_NOT_FOUND, "Did not find an Prizmatix board .  Is the Prizmatix board connected to this serial port?");
   SetErrorText(ERR_NO_PORT_SET, "Hub Device not found.  The Prizmatix Hub device is needed to create this device");
  /*??MMMMMM std::ostringstream errorText;
   errorText << "The firmware version on the Arduino is not compatible with this adapter.  Please use firmware version ";
   errorText <<  g_Min_MMVersion << " to " << g_Max_MMVersion;
   SetErrorText(ERR_VERSION_MISMATCH, errorText.str().c_str());
   ***/
   CPropertyAction* pAct = new CPropertyAction(this, &PrizmatixHub::OnPort);
   CreateProperty(MM::g_Keyword_Port, "Undefined", MM::String, false, pAct, true);
   /*???? MMMMM
   pAct = new CPropertyAction(this, &PrizmatixHub::OnLogic);
   CreateProperty("Logic", g_invertedLogicString, MM::String, false, pAct, true);

   AddAllowedValue("Logic", g_invertedLogicString);
   AddAllowedValue("Logic", g_normalLogicString);
   ***/
}

PrizmatixHub::~PrizmatixHub()
{
   Shutdown();
}

void PrizmatixHub::GetName(char* name) const
{
   CDeviceUtils::CopyLimitedString(name, g_DeviceNameHub);
}

bool PrizmatixHub::Busy()
{
   return false;
}

// private and expects caller to:
// 1. guard the port
// 2. purge the port
int PrizmatixHub::GetControllerVersion(int& version)
{
   int ret = DEVICE_OK;
   const char* command="V:0\n";
  
// SendSerialCommand
   ret = WriteToComPort(port_.c_str(), (const unsigned char*) command, strlen((char *)command));
   if (ret != DEVICE_OK)
      return ret;

   std::string answer;
   ret = GetSerialAnswer(port_.c_str(), "\r\n", answer);
   if (ret != DEVICE_OK)
      return ret ;
   int Mik=answer.find_last_of('_');
   nmLeds=atoi(answer.data()+Mik+1);
   //nmLeds=1;// MMM set from the answer 
   // Check version number of the Arduino
  
   nmLeds=2;//MMMM
   return ret;

}

bool PrizmatixHub::SupportsDeviceDetection(void)
{
   return true;
}

MM::DeviceDetectionStatus PrizmatixHub::DetectDevice(void)
{
   if (initialized_)
      return MM::CanCommunicate;

   // all conditions must be satisfied...
   MM::DeviceDetectionStatus result = MM::Misconfigured;
   char answerTO[MM::MaxStrLength];
   
   try
   {
      std::string portLowerCase = port_;
      for( std::string::iterator its = portLowerCase.begin(); its != portLowerCase.end(); ++its)
      {
         *its = (char)tolower(*its);
      }
      if( 0< portLowerCase.length() &&  0 != portLowerCase.compare("undefined")  && 0 != portLowerCase.compare("unknown") )
      {
         result = MM::CanNotCommunicate;
         // record the default answer time out
         GetCoreCallback()->GetDeviceProperty(port_.c_str(), "AnswerTimeout", answerTO);

         // device specific default communication parameters
         // for Arduino Duemilanova
         GetCoreCallback()->SetDeviceProperty(port_.c_str(), MM::g_Keyword_Handshaking, g_Off);
         GetCoreCallback()->SetDeviceProperty(port_.c_str(), MM::g_Keyword_BaudRate, "57600" );
         GetCoreCallback()->SetDeviceProperty(port_.c_str(), MM::g_Keyword_StopBits, "1");
         // Arduino timed out in GetControllerVersion even if AnswerTimeout  = 300 ms
         GetCoreCallback()->SetDeviceProperty(port_.c_str(), "AnswerTimeout", "1500.0");
         GetCoreCallback()->SetDeviceProperty(port_.c_str(), "DelayBetweenCharsMs", "0");
         MM::Device* pS = GetCoreCallback()->GetDevice(this, port_.c_str());
         pS->Initialize();
         // The first second or so after opening the serial port, the Arduino is waiting for firmwareupgrades.  Simply sleep 2 seconds.
         CDeviceUtils::SleepMs(2000);
         MMThreadGuard myLock(lock_);
         PurgeComPort(port_.c_str());
         int v = 0;
         int ret = GetControllerVersion(v);
         // later, Initialize will explicitly check the version #
         if( DEVICE_OK != ret )
         {
            LogMessageCode(ret,true);
         }
         else
         {
            // to succeed must reach here....
            result = MM::CanCommunicate;
         }
         pS->Shutdown();
         // always restore the AnswerTimeout to the default
         GetCoreCallback()->SetDeviceProperty(port_.c_str(), "AnswerTimeout", answerTO);

      }
   }
   catch(...)
   {
      LogMessage("Exception in DetectDevice!",false);
   }

   return result;
}


int PrizmatixHub::Initialize()
{
   // Name
   int ret;//??? MMM1 = CreateProperty(MM::g_Keyword_Name, g_DeviceNameHub, MM::String, true);
 //  if (DEVICE_OK != ret)
  //    return ret;

   // The first second or so after opening the serial port, the Arduino is waiting for firmwareupgrades.  Simply sleep 1 second.
   CDeviceUtils::SleepMs(2000);

   MMThreadGuard myLock(lock_);

   // Check that we have a controller:
   PurgeComPort(port_.c_str());
   ret = GetControllerVersion(version_);
   if( DEVICE_OK != ret)
      return ret;
   if(nmLeds <=0) return ERR_COMMUNICATION;
  
/*MMM  ?????
   if (version_ < g_Min_MMVersion || version_ > g_Max_MMVersion)
      return ERR_VERSION_MISMATCH;

   CPropertyAction* pAct = new CPropertyAction(this, &PrizmatixHub::OnVersion);
   std::ostringstream sversion;
   sversion << version_;
   CreateProperty(g_versionProp, sversion.str().c_str(), MM::Integer, true, pAct);
   */
   ret = UpdateStatus();
   if (ret != DEVICE_OK)
      return ret;

   // turn off verbose serial debug messages
   // GetCoreCallback()->SetDeviceProperty(port_.c_str(), "Verbose", "0");

   initialized_ = true;
   return DEVICE_OK;
}

int PrizmatixHub::DetectInstalledDevices()
{

   if (MM::CanCommunicate == DetectDevice()) 
   {
      std::vector<std::string> peripherals; 
      peripherals.clear();
	  char *Name=0;
	  switch(nmLeds)
	  {
			case 1: Name=(char *) g_DeviceOneLED;break;
			case 2: Name=(char *)g_DeviceMultiLED_2;break;
			case 3: Name=(char *)g_DeviceMultiLED_3;break;
			case 4: Name=(char *)g_DeviceMultiLED_4;break;
			case 5: Name=(char *)g_DeviceMultiLED_5;break;
			case 6: Name=(char *)g_DeviceMultiLED_6;break;
			case 7: Name=(char *)g_DeviceMultiLED_7;break;
			case 8: Name=(char *)g_DeviceMultiLED_8;break;
			case 9: Name=(char *)g_DeviceMultiLED_9;break;
			case 10: Name=(char *)g_DeviceMultiLED_10;break;
			default:
				Name=(char *)g_DeviceMultiLED_10;break;
	  }
	 
		peripherals.push_back(Name);
	 /*** MMMM?????
      peripherals.push_back(g_DeviceNameArduinoSwitch);
      peripherals.push_back(g_DeviceNameArduinoShutter);
      peripherals.push_back(g_DeviceNameArduinoInput);
      peripherals.push_back(g_DeviceNameArduinoDA1);
      peripherals.push_back(g_DeviceNameArduinoDA2);
	  **/
      for (size_t i=0; i < peripherals.size(); i++) 
      {
         MM::Device* pDev = ::CreateDevice(peripherals[i].c_str());
         if (pDev) 
         {
            AddInstalledDevice(pDev);
         }
      }
   }

   return DEVICE_OK;
}



int PrizmatixHub::Shutdown()
{
   initialized_ = false;
   return DEVICE_OK;
}

int PrizmatixHub::OnPort(MM::PropertyBase* pProp, MM::ActionType pAct)
{
   if (pAct == MM::BeforeGet)
   {
      pProp->Set(port_.c_str());
   }
   else if (pAct == MM::AfterSet)
   {
      pProp->Get(port_);
      portAvailable_ = true;
   }
   return DEVICE_OK;
}

int PrizmatixHub::OnVersion(MM::PropertyBase* pProp, MM::ActionType pAct)
{
   if (pAct == MM::BeforeGet)
   {
      pProp->Set((long)version_);
   }
   return DEVICE_OK;
}
/*** MMM?????
int PrizmatixHub::OnLogic(MM::PropertyBase* pProp, MM::ActionType pAct)
{
   if (pAct == MM::BeforeGet)
   {
      if (invertedLogic_)
         pProp->Set(g_invertedLogicString);
      else
         pProp->Set(g_normalLogicString);
   } else if (pAct == MM::AfterSet)
   {
      std::string logic;
      pProp->Get(logic);
      if (logic.compare(g_invertedLogicString)==0)
         invertedLogic_ = true;
      else invertedLogic_ = false;
   }
   return DEVICE_OK;
}
***/
////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////
// PrizmatixLED implementation
// ~~~~~~~~~~~~~~~~~~~~~~

PrizmatixLED::PrizmatixLED(int nmLeds_,char *Name) :
      busy_(false), 
      minV_(0.0), 
      maxV_(100.0), 
      volts_(0.0),
      gatedVolts_(0.0),
      nmLeds(nmLeds_), 
   //   maxChannel_(10),
      gateOpen_(true)
{
   InitializeDefaultErrorMessages();

   // add custom error messages
   SetErrorText(ERR_UNKNOWN_POSITION, "Invalid position (state) specified");
   SetErrorText(ERR_INITIALIZE_FAILED, "Initialization of the device failed");
   SetErrorText(ERR_WRITE_FAILED, "Failed to write data to the device");
   SetErrorText(ERR_CLOSE_FAILED, "Failed closing the device");
   SetErrorText(ERR_NO_PORT_SET, "Hub Device not found.  The Prizmatix Hub device is needed to create this device");

   /* Channel property is not needed
   CPropertyAction* pAct = new CPropertyAction(this, &PrizmatixLED::OnChannel);
   CreateProperty("Channel", channel_ == 1 ? "1" : "2", MM::Integer, false, pAct, true);
   for (int i=1; i<= 2; i++){
      std::ostringstream os;
      os << i;
      AddAllowedValue("Channel", os.str().c_str());
   }
  

   CPropertyAction* pAct = new CPropertyAction(this, &PrizmatixLED::OnMaxVolt);
   CreateProperty("MaxVolt", "5.0", MM::Float, false, pAct, true);
 //  MM::Hub* hub = GetParentHub();
  //  PrizmatixHub* PHub = dynamic_cast<PrizmatixHub *>(hub);
     */
   name_ = std::string(Name);// PHub->GetName();//channel_ == 1 ? g_DeviceNameArduinoDA1 : g_DeviceNameArduinoDA2;
  
   // Description
   int nRet = CreateProperty(MM::g_Keyword_Description, "Prizmatix Control", MM::String, true);
   assert(DEVICE_OK == nRet);
    // ???    D:0,1
   // Name
 //MM !!!  nRet = CreateProperty(MM::g_Keyword_Name, name_.c_str(), MM::String, true);
// MM 11   assert(DEVICE_OK == nRet);

   // parent ID display
   CreateHubIDProperty();
}

PrizmatixLED::~PrizmatixLED()
{
   Shutdown();
}

void PrizmatixLED::GetName(char* name) const
{
   CDeviceUtils::CopyLimitedString(name, name_.c_str());
}


int PrizmatixLED::Initialize()
{
   PrizmatixHub* hub = static_cast<PrizmatixHub*>(GetParentHub());
   if (!hub || !hub->IsPortAvailable()) {
      return ERR_NO_PORT_SET;
   }
   char hubLabel[MM::MaxStrLength];
   hub->GetLabel(hubLabel);
   SetParentID(hubLabel); // for backward comp.


   /// FirmWareName
    // ???    D:0,1
   // Name
   {  // Firmware property
			char *Com="V:1";
			hub->SendSerialCommandH(Com);			      
			std::string answer;
			int  ret = hub->GetSerialAnswerH(  answer);
			int NumFirm=atoi(answer.data()+2);
			char NameF[15];
			switch(NumFirm)
			{
				 
				case 1: strcpy(NameF,"UHPTLCC-USB");break;
				case 2: strcpy(NameF,"UHPTLCC-USB-STBL");break;
				case 3: strcpy(NameF,"FC-LED");break;
				case 4: strcpy(NameF,"Combi-LED");break;
				case 5: strcpy(NameF,"UHP-M-USB");break;
				case 6: strcpy(NameF,"UHP-F-USB");break;
				case 7: strcpy(NameF,"UHP-F-USB");break;
				default:
					NumFirm=0;break;
			}
			if(NumFirm > 0)
				CreateProperty("Firmware Name", NameF, MM::String, true);
			if(NumFirm==2)
			{ // Add Stbl
			}
   }
   
 
   ///


       char* command="S:0\n";
  
// SendSerialCommand
	 hub->SendSerialCommandH(command);
 
	   long nmWrite;
     
	    std::string answer;
   int  ret = hub->GetSerialAnswerH(  answer);
   const char * NameLeds=answer.data();
   nmWrite=answer.length();
   // set property list
   // -----------------
   
   // State
   // -----
   int nRet;
   int Mik=1;
   int Until;
   
   for(int i=0;i< nmLeds ;i++)
   {
		  // CPropertyAction
			   CPropertyActionEx* pAct = new CPropertyActionEx (this, &PrizmatixLED::OnPowerLEDEx,i);
		   char Name[20],StateName[20];
		   Until=Mik+1;
		   while(Until< nmWrite && NameLeds[Until] !=',') Until++;
		   if( Mik+1 < nmWrite && Mik  <Until )
		   {
				memcpy(Name,NameLeds+Mik,Until-Mik);
				Name[Until-Mik]=0;
				Mik=Until+1;
				  while(Mik< nmWrite && NameLeds[Mik] ==' ') Mik++;
		   }
		   else
				sprintf(Name,"LED%d",i);  
			nRet= CreateProperty(Name, "0", MM::Integer,  false, pAct);
			 SetPropertyLimits(Name, 0, 100);
			  ValLeds[i]=0;
			  OnOffLeds[i]=0;
		//-----
					CPropertyActionEx* pAct5 = new CPropertyActionEx (this, &PrizmatixLED::OnOfOnEx,i);
						sprintf(StateName,"State %s",Name); 
					 ret = CreateProperty(StateName, "0", MM::Integer, false, pAct5);
				   if (ret != DEVICE_OK)
					  return ret;

				   AddAllowedValue(StateName, "0");
				   AddAllowedValue(StateName, "1");
		//-----
		
			 
		  
   }
   nRet = UpdateStatus();
   if (nRet != DEVICE_OK)
      return nRet;

   initialized_ = true;

   return DEVICE_OK;
}

int PrizmatixLED::Shutdown()
{
     PrizmatixHub* hub = static_cast<PrizmatixHub*>(GetParentHub());
	 if(hub) hub->SendSerialCommandH("P:0");
   initialized_ = false;
   return DEVICE_OK;
}

int PrizmatixLED::WriteToPort(unsigned long value)
{
   PrizmatixHub* hub = static_cast<PrizmatixHub*>(GetParentHub());
   if (!hub || !hub->IsPortAvailable())
      return ERR_NO_PORT_SET;

   MMThreadGuard myLock(hub->GetLock());

   hub->PurgeComPortH();
   char Buf[100],StrNum[1024];
   strcpy(Buf,"P:");
   for(int i=0;i< nmLeds;i++)
   {
	  if( OnOffLeds[i]==0)
		 strcat(Buf,"0");
	  else
		 strcat(Buf,_itoa(ValLeds[i],StrNum,10));
	   strcat(Buf,",");
   }
    hub->SendSerialCommandH(Buf);
		/***
   int ret = hub->WriteToComPortH((const unsigned char*) command, 4);
   if (ret != DEVICE_OK)
      return ret;

   MM::MMTime startTime = GetCurrentMMTime();
   unsigned long bytesRead = 0;
   unsigned char answer[4];
   while ((bytesRead < 4) && ( (GetCurrentMMTime() - startTime).getMsec() < 2500)) {
      unsigned long bR;
      ret = hub->ReadFromComPortH(answer + bytesRead, 4 - bytesRead, bR);
      if (ret != DEVICE_OK)
         return ret;
      bytesRead += bR;
   }
   if (answer[0] != 3)
      return ERR_COMMUNICATION;
	  ****/
   hub->SetTimedOutput(false);

   return DEVICE_OK;
}

/***
int PrizmatixLED::WriteSignal(double volts)
{
   long value = (long) ( (volts - minV_) / maxV_ * 4095);

   std::ostringstream os;
    os << "Volts: " << volts << " Max Voltage: " << maxV_ << " digital value: " << value;
    LogMessage(os.str().c_str(), true);

   return WriteToPort(value);
}

int PrizmatixLED::SetSignal(double volts)
{
   volts_ = volts;
   if (gateOpen_) {
      gatedVolts_ = volts_;
      return WriteSignal(volts_);
   } else {
      gatedVolts_ = 0;
   }

   return DEVICE_OK;
}
*/
int PrizmatixLED::SetGateOpen(bool open)
{
   if (open) {
      gateOpen_ = true;
      gatedVolts_ = volts_;
      return DEVICE_OK;//WriteSignal(volts_);
   } 
   gateOpen_ = false;
   gatedVolts_ = 0;
   return  DEVICE_OK;

}
 
///////////////////////////////////////////////////////////////////////////////
// Action handlers
///////////////////////////////////////////////////////////////////////////////

int PrizmatixLED::OnVolts(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      // nothing to do, let the caller use cached property
   }
   else if (eAct == MM::AfterSet)
   {
      double volts;
      pProp->Get(volts);
 
    //  return SetSignal(volts);
	     return WriteToPort(0);
   }

   return DEVICE_OK;
}
 int PrizmatixLED::OnOfOnEx(MM::PropertyBase* pProp, MM::ActionType eAct,long Param)
 {
	  if (eAct == MM::BeforeGet)
   {
      // nothing to do, let the caller use cached property
   }
   else if (eAct == MM::AfterSet)
   {
	   long pos;
		pProp->Get(pos);
		OnOffLeds[Param]=pos;
		WriteToPort(0);
	 }
	   return DEVICE_OK;
 }
int PrizmatixLED::OnPowerLEDEx(MM::PropertyBase* pProp, MM::ActionType eAct,long Param)
{
   if (eAct == MM::BeforeGet)
   {
      // nothing to do, let the caller use cached property
   }
   else if (eAct == MM::AfterSet)
   {
      double volts;
      pProp->Get(volts);
     	ValLeds[Param]= floor((volts*4095/100));
    //  return SetSignal(volts);
	     return WriteToPort(0);
   }

   return DEVICE_OK;
}
/*
int PrizmatixLED::OnMaxVolt(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(maxV_);
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(maxV_);
      if (HasProperty("Volts"))
         SetPropertyLimits("Volts", 0.0, maxV_);

   }
   return DEVICE_OK;
}
/* MMMMM ???
int PrizmatixLED::OnChannel(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set((long int)channel_);
   }
   else if (eAct == MM::AfterSet)
   {
      long channel;
      pProp->Get(channel);
      if (channel >=1 && ( (unsigned) channel <=maxChannel_) )
         channel_ = channel;
   }
   return DEVICE_OK;
}
**/

///////////////////////////////////////////////////////////////////////////////