#include "PrizmatixMain.h"
#ifdef WIN32
   #define WIN32_LEAN_AND_MEAN
   #include <windows.h>
#endif
#include "FixSnprintf.h"


const char* g_DeviceNameHub = "prizmatix-Hub";
const char* g_DeviceOneLED = "Prizmatix Ctrl";
 
const char* g_On = "On";
const char* g_Off = "Off";

MMThreadLock PrizmatixHub::lock_;
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
  
   CPropertyAction* pAct = new CPropertyAction(this, &PrizmatixHub::OnPort);
   CreateProperty(MM::g_Keyword_Port, "Undefined", MM::String, false, pAct, true);
  
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
  
   ret = UpdateStatus();
   if (ret != DEVICE_OK)
      return ret;
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
	/**  switch(nmLeds)**/
	
	  Name=(char *) g_DeviceOneLED;
		peripherals.push_back(Name);
	 
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
 
///////////////////////////////////////////////////////////////////////////////
// PrizmatixLED implementation
// ~~~~~~~~~~~~~~~~~~~~~~

PrizmatixLED::PrizmatixLED(int nmLeds_,char *Name) :
      nmLeds(nmLeds_) 
 
{
	
   InitializeDefaultErrorMessages();

   // add custom error messages
 
   SetErrorText(ERR_INITIALIZE_FAILED, "Initialization of the device failed");
   SetErrorText(ERR_WRITE_FAILED, "Failed to write data to the device");
   SetErrorText(ERR_CLOSE_FAILED, "Failed closing the device");
   SetErrorText(ERR_NO_PORT_SET, "Hub Device not found.  The Prizmatix Hub device is needed to create this device");
   name_ = std::string(Name);
  
   // Description
   int nRet = CreateProperty(MM::g_Keyword_Description, "Prizmatix Control", MM::String, true);
   assert(DEVICE_OK == nRet);
  
   // Name
 // nRet = CreateProperty(MM::g_Keyword_Name, name_.c_str(), MM::String, true);
 //  assert(DEVICE_OK == nRet);

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
   
 	myHub= static_cast<PrizmatixHub*>(GetParentHub());
   if (!myHub || !myHub->IsPortAvailable()) {
      return ERR_NO_PORT_SET;
   }
   nmLeds=myHub->GetNmLeds();
   char hubLabel[MM::MaxStrLength];
   myHub->GetLabel(hubLabel);
   SetParentID(hubLabel); // for backward comp.

   {  // Firmware property
			char *Com="V:1";
			myHub->SendSerialCommandH(Com);			      
			std::string answer;
			int  ret = myHub->GetSerialAnswerH(  answer);
			int NumFirm=atoi(answer.data()+2);			
			char NameF[25];
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
				  
			   CPropertyAction* pAct = new CPropertyAction (this, &PrizmatixLED::OnSTBL);
			    ret = CreateProperty("STBL", "0", MM::Integer, false, pAct);
				   if (ret != DEVICE_OK)
					  return ret;

				   AddAllowedValue("STBL", "0");
				   AddAllowedValue("STBL", "1");
			}
   }
   
 
   ///


       char* command="S:0\n";
  
// SendSerialCommand
	 myHub->SendSerialCommandH(command);
 
	   long nmWrite;
     
	    std::string answer;
   int  ret = myHub->GetSerialAnswerH(  answer);
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
	   ValLeds[i]=0;
		  OnOffLeds[i]=0;
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
     PrizmatixHub* hub = myHub;//static_cast<PrizmatixHub*>(GetParentHub());
	 if(hub) hub->SendSerialCommandH("P:0");
   initialized_ = false;
   return DEVICE_OK;
}

int PrizmatixLED::WriteToPort(char *Str)
{
   PrizmatixHub* hub =myHub;// static_cast<PrizmatixHub*>(GetParentHub());
   if (!hub || !hub->IsPortAvailable())
      return ERR_NO_PORT_SET;

   MMThreadGuard myLock(hub->GetLock());

   hub->PurgeComPortH();
   if(Str ==0)
   {
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
   }
   else
   {
	   hub->SendSerialCommandH(Str);
   }
		
   hub->SetTimedOutput(false);

   return DEVICE_OK;
}


///////////////////////////////////////////////////////////////////////////////
// Action handlers
///////////////////////////////////////////////////////////////////////////////

 int PrizmatixLED::OnOfOnEx(MM::PropertyBase* pProp, MM::ActionType eAct,long Param)
 {
	  if (eAct == MM::BeforeGet)// && OnOffLeds[Param]>=0 )
   {
      // nothing to do, let the caller use cached property
   }
   else if (eAct == MM::AfterSet )//|| OnOffLeds[Param] ==-1)
   {
	   long pos;
		pProp->Get(pos);
		OnOffLeds[Param]=pos;
		WriteToPort(0);
	 }
	   return DEVICE_OK;
 }
 int PrizmatixLED::OnSTBL(MM::PropertyBase* pProp, MM::ActionType eAct)
 {
	  if (eAct == MM::BeforeGet )//&& ValLeds[Param] >=0)
   {
      // nothing to do, let the caller use cached property
   }
   else if (eAct == MM::AfterSet)//|| ValLeds[Param]==-1)
   {
      long Stat;
      pProp->Get(Stat);
      char Buf[20];
	 sprintf(Buf,"K:1,8,%d",Stat);
  
	     return WriteToPort(Buf);
   }

   return DEVICE_OK;
 }
int PrizmatixLED::OnPowerLEDEx(MM::PropertyBase* pProp, MM::ActionType eAct,long Param)
{
   if (eAct == MM::BeforeGet )//&& ValLeds[Param] >=0)
   {
      // nothing to do, let the caller use cached property
   }
   else if (eAct == MM::AfterSet)//|| ValLeds[Param]==-1)
   {
      double volts;
      pProp->Get(volts);
     	ValLeds[Param]= floor((volts*4095/100));

	     return WriteToPort(0);
   }

   return DEVICE_OK;
}

