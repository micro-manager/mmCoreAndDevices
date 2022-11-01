///////////////////////////////////////////////////////////////////////////////
// FILE:			XLIGHTV3MM.cpp
// PROJECT:			Micro-Manager
// SUBSYSTEM:		DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:		Crestoptics XLight adapter
//                                                                                     
// AUTHOR:			ing. S. Silvestri silvestri.salvatore.ing@gmail.com, 12/09/2022
//					
//
// COPYRIGHT:		2022, Crestoptics s.r.l.
// LICENSE:			This file is distributed under the BSD license.
//					License text is included with the source distribution.
//
//					This file is distributed in the hope that it will be useful,
//					but WITHOUT ANY WARRANTY; without even the implied warranty
//					of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//					IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//					CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//					INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.

// works with V3; V3 support: work in progress !

#include "XLight_MM.h"
#include "DeviceConfig.h"

#include <string>

// External names used used by the rest of the system
// static lock
MMThreadLock XLightHub::lock_;
XLightHub* pXLightHub ;
bool HCW_state;


// ======================================== API ==================================================
// ===============================================================================================
// ----------------------------------------------------------------------------------------------
MODULE_API void InitializeModuleData()
{
	RegisterDevice(g_DichroicWheel_Name, MM::StateDevice, g_DichroicWheel_Dev_Desc); // last string : description
	RegisterDevice(g_EmissionWheel_Name, MM::StateDevice, g_EmissionWheel_Dev_Desc);
	RegisterDevice(g_ExcitationWheel_Name, MM::StateDevice, g_ExcitationWheel_Dev_Desc);
	RegisterDevice(g_SpinningSlider_Name, MM::StateDevice, g_SpinningSlider_Dev_Desc);

	RegisterDevice(g_CameraSlider_Name , MM::StateDevice, g_CameraSlider_Dev_Desc);
	RegisterDevice(g_SpinningMotor_Name , MM::StateDevice, g_SpinningMotor_Dev_Desc);

	RegisterDevice(g_EmissionIrisDeviceName , MM::GenericDevice, g_EmissionIrisDevice_Dev_Desc);
	RegisterDevice(g_IlluminationIrisDeviceName , MM::GenericDevice, g_IlluminationIrisDevice_Dev_Desc);


	RegisterDevice(g_HubDevice_Name, MM::HubDevice,g_HubDevice_Dev_Desc);
}

// ----------------------------------------------------------------------------------------------

MM::Device* CreateXLightDevice(TDevicelType eDeviceType){
	MM::Device* pDevice=NULL;
	XLightStateDevice* pStateDevice=NULL;
	IrisDevice* pIris=NULL;
	TDeviceInfo* pDeviceInfo;

	if (eDeviceType<EMISSION_IT){
		pDevice=new XLightStateDevice();
		pStateDevice= static_cast<XLightStateDevice*> (pDevice);
		pDeviceInfo=&(pStateDevice->DeviceInfo_);

	}
	else {
		pDevice=new IrisDevice();
		pIris= static_cast<IrisDevice*> (pDevice);
		pDeviceInfo=&(pIris->DeviceInfo_);
	}

	pDeviceInfo->PrefixCMD=CMDPrefix[eDeviceType-1];
	pDeviceInfo->MaxValue=MaxPositions[eDeviceType-1];
	pDeviceInfo->Working=MaxPositions[eDeviceType-1]>0;
	pDeviceInfo->DeviceType_=eDeviceType;
	pDeviceInfo->name_=DevicesName[eDeviceType-1];
	pDeviceInfo->description_=DevicesDesc[eDeviceType-1];
	return pDevice;
}
// ----------------------------------------------------------------------------------------------

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
	if (deviceName == 0)
		return 0;

	if (strcmp(deviceName, g_EmissionWheel_Name) == 0)
	{
		// create state device
		return CreateXLightDevice(EMISSION_FT);
	}
	else if (strcmp(deviceName, g_DichroicWheel_Name) == 0)
	{
		// create state device
		return CreateXLightDevice(DICHROIC_FT);
	}
	else if (strcmp(deviceName, g_ExcitationWheel_Name) == 0)
	{
		// create state device
		return CreateXLightDevice(EXCITATION_FT);
	}

	else if (strcmp(deviceName, g_SpinningSlider_Name) == 0)
	{
		// create state device
		return CreateXLightDevice(SPINNING_SLIDER);
	}

	else if (strcmp(deviceName, g_CameraSlider_Name) == 0)
	{
		// create state device
		return CreateXLightDevice(CAMERA_SLIDER);
	}
	else if (strcmp(deviceName, g_SpinningMotor_Name) == 0)
	{
		// create state device
		return CreateXLightDevice(SPINNING_MOTOR);
	}

	
	else if (strcmp(deviceName, g_EmissionIrisDeviceName) == 0)
	{
		// create state device
		return CreateXLightDevice(EMISSION_IT);
	}
	else if (strcmp(deviceName, g_IlluminationIrisDeviceName) == 0)
	{
		// create state device
		return CreateXLightDevice(ILLUMINATION_IT);
	}

	else if (strcmp(deviceName, g_HubDevice_Name) == 0)
	{
		pXLightHub=new XLightHub();
		return pXLightHub;
	}



	// ...supplied name not recognized
	//FinisciFunzione("API::CreateDevice");
	return 0;
}
// ----------------------------------------------------------------------------------------------
MODULE_API void DeleteDevice(MM::Device* pDevice)
{
	delete pDevice;
}

// ----------------------------------------------------------------------------------------------
// ======================================== Hub ==================================================
// ===============================================================================================
// ----------------------------------------------------------------------------------------------
XLightHub::XLightHub() :
initialized_(false),
	busy_(false),
	port_("Undefined")
{
	InitializeDefaultErrorMessages();

	// custom error messages
	SetErrorText(ERR_COMMAND_CANNOT_EXECUTE, "Command cannot be executed");

	// create pre-initialization properties
	// ------------------------------------

	// Port
	CPropertyAction* pAct = new CPropertyAction(this, &XLightHub::OnPort);
	CreateProperty(MM::g_Keyword_Port, "Undefined", MM::String, false, pAct, true);	  
}
// ----------------------------------------------------------------------------------------------
int XLightHub::Initialize()
{

	//	IniziaFunzione("Initialize");

	MMThreadGuard myLock(lock_);
	PurgeComPort(port_.c_str());

	int ret = GetControllerInfo();
	if( DEVICE_OK != ret)
		return ret;


	ret = UpdateStatus();
	if (ret != DEVICE_OK)
		return ret;
	initialized_ = true;

	return DEVICE_OK;
}
// ----------------------------------------------------------------------------------------------
// private and expects caller to:
// 1. guard the port
// 2. purge the port



bool XLightHub::SupportsDeviceDetection(void)
{
   return true;
}


MM::DeviceDetectionStatus XLightHub::DetectDevice(void)
{
	//	IniziaFunzione("DetectDevice");
	if (initialized_)
		return MM::CanCommunicate;

	// all conditions must be satisfied...
	MM::DeviceDetectionStatus result = MM::Misconfigured;
	char answerTO[MM::MaxStrLength];

	bool OK_9600=false;

	try
	{
		std::string portLowerCase = port_;
		for( std::string::iterator its = portLowerCase.begin(); its != portLowerCase.end(); ++its)
		{
			*its = (char)tolower(*its);
		}
		if( 0< portLowerCase.length() &&  0 != portLowerCase.compare("undefined")  && 0 != portLowerCase.compare("unknown") )
		{

			// try with 9600, if not connected try with 115200
			result = MM::CanNotCommunicate;
			// record the default answer time out
			GetCoreCallback()->GetDeviceProperty(port_.c_str(), "AnswerTimeout", answerTO);


			// device specific default communication parameters
			// for Arduino Duemilanova
			GetCoreCallback()->SetDeviceProperty(port_.c_str(), MM::g_Keyword_Handshaking, g_Off);
			GetCoreCallback()->SetDeviceProperty(port_.c_str(), MM::g_Keyword_StopBits, "1");
			// Arduino timed out in GetControllerVersion even if AnswerTimeout  = 300 ms
			GetCoreCallback()->SetDeviceProperty(port_.c_str(), "AnswerTimeout", "300.0");
			GetCoreCallback()->SetDeviceProperty(port_.c_str(), "DelayBetweenCharsMs", "0");
			MM::Device* pS = GetCoreCallback()->GetDevice(this, port_.c_str());
			int ret = DEVICE_OK;
			for (int t=0; t<2; t++){
				if (t==0)
					GetCoreCallback()->SetDeviceProperty(port_.c_str(), MM::g_Keyword_BaudRate, "9600" );
				else 
					GetCoreCallback()->SetDeviceProperty(port_.c_str(), MM::g_Keyword_BaudRate, "115200" );

				pS->Initialize();
				// The first second or so after opening the serial port, the Arduino is waiting for firmwareupgrades.  Simply sleep 2 seconds.
				CDeviceUtils::SleepMs(2000);
				MMThreadGuard myLock(lock_);
				PurgeComPort(port_.c_str());
				//int v = 0;
				ret = GetControllerInfo();
				// later, Initialize will explicitly check the version #
				if ( ret==DEVICE_OK)
					break;
			}



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
// ----------------------------------------------------------------------------------------------
int XLightHub::DetectInstalledDevices()
{  
	//	IniziaFunzione("DetectInstalledDevices");

	if (MM::CanCommunicate == DetectDevice()) 
	{
		ClearInstalledDevices();

		// make sure this method is called before we look for available devices
		InitializeModuleData();

		char hubName[MM::MaxStrLength];
		GetName(hubName); // this device name
		for (unsigned i=0; i<GetNumberOfDevices(); i++)
		{ 
			char deviceName[MM::MaxStrLength];
			bool success = GetDeviceName(i, deviceName, MM::MaxStrLength);
			if (success && (strcmp(hubName, deviceName) != 0))
			{
				MM::Device* pDev = CreateDevice(deviceName);
				if (pDev!=NULL)
					AddInstalledDevice(pDev);
			}
		}
	}
	return DEVICE_OK; 
}
// ----------------------------------------------------------------------------------------------
void XLightHub::GetName(char* pName) const
{
	CDeviceUtils::CopyLimitedString(pName, g_HubDevice_Name);
}
// ----------------------------------------------------------------------------------------------
int XLightHub::Shutdown() {
	initialized_=false;
	return DEVICE_OK;
}
// ----------------------------------------------------------------------------------------------
bool XLightHub::Busy() { 
	return busy_;
} 
// ----------------------------------------------------------------------------------------------


int XLightHub::ExecuteCmd(TCmdType eCmdType,  TDeviceInfo* pDeviceInfo, long value){
	std::string CommandStr=BuilCommand(eCmdType,pDeviceInfo, value );
	int ret=DEVICE_OK;
	ret=SendCmdString(CommandStr,15000);
	if (ret!=DEVICE_OK)
		return ret;
	return ParseAnswer(CommandStr, GetInputStr(), eCmdType, pDeviceInfo);
}
// ----------------------------------------------------------------------------------------------
std::string XLightHub::BuilCommandBase (TCmdType eCmdType,  TDeviceInfo* pDeviceInfo){
	
	std::string StrCmdBase=pDeviceInfo->PrefixCMD;
	if (eCmdType!=SETPOS_CMD )
		StrCmdBase="r"+StrCmdBase;

	if ( eCmdType==GETNUMPOS_CMD)
		StrCmdBase=StrCmdBase+"N";
	return StrCmdBase;
}
// ----------------------------------------------------------------------------------------------
std::string XLightHub::BuilCommand (TCmdType eCmdType,  TDeviceInfo* pDeviceInfo ,int value){


	std::string result=BuilCommandBase(eCmdType, pDeviceInfo);
	if (eCmdType==SETPOS_CMD){
		if ((pDeviceInfo->DeviceType_ >= EMISSION_FT) && (pDeviceInfo->DeviceType_<=EXCITATION_FT)){
			result=result+std::to_string((long long)( value+1));
		}
		else {
		result=result+std::to_string((long long) value);
		}
	}

	return result;
}
// ----------------------------------------------------------------------------------------------
int XLightHub::ParseAnswer(std::string pCmd, std::string pAnsw, TCmdType eCmdType , TDeviceInfo* pDeviceInfo){

	// creo il comando che è stato inviato
	std::string StrCmdBase=BuilCommandBase(eCmdType, pDeviceInfo);
	std::string StrAnsw= pAnsw;  
	std::string StrCmd= pCmd;  
	int ReqValue=-1;
	int RetValue=-1;
	// verifico prima che il comando sia integralmente contenuto nella risposta
	if (StrAnsw.find(StrCmdBase)!=0){
		return ERR_COMMAND_EXECUTION_ERROR;
	}
	// se arrivo qui il comando è presente nella risposta, quindi estraggo quello che rimane
	StrAnsw=StrAnsw.substr(StrCmdBase.length()); 
	try
	{
		RetValue=std::stoi ((char*)StrAnsw.c_str());
	}
	catch (...)
	{
		return ERR_COMMAND_EXECUTION_ERROR;
	}

	if (eCmdType==SETPOS_CMD){
		StrCmd=StrCmd.substr(StrCmdBase.length()); 
		// verifico che l'intero della posizione corrisponda
		try
		{
			ReqValue=std::stoi (StrCmd);
		}
		catch (...)
		{
			return ERR_COMMAND_EXECUTION_ERROR;
		}

		if (ReqValue==RetValue){
			return DEVICE_OK;
		}
		else {
			return ERR_COMMAND_EXECUTION_ERROR;
		}
	}
	else {
		if (eCmdType==GETNUMPOS_CMD){
			pDeviceInfo->MaxValue=RetValue;
		}
		else {
			pDeviceInfo->Value =pDeviceInfo->DeviceType_>=SPINNING_SLIDER?RetValue-1:RetValue;
		}
		return DEVICE_OK;
	}
}
// ----------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------
//						support funztions 
// ----------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------
int XLightHub::GetDeviceValue(TDevicelType DeviceType, TValueType ValueType, int * iValue){

	int RetValue;
	int ret=DEVICE_OK;
	bool AnswerOK;

	* iValue=0;

	std::string StrCmdBase;

	switch (ValueType){
	case WORK_VAL:{
		if (DeviceType>=EMISSION_FT && DeviceType<=EXCITATION_FT){
			StrCmdBase="r"+CMDPrefix[(int)(DeviceType)-1];
		}
		else if (DeviceType>=SPINNING_SLIDER&& DeviceType<=SPINNING_MOTOR){
			StrCmdBase="r"+CMDPrefix[(int)(DeviceType)-1]+"S";
		}
		else {
			StrCmdBase="rI";

		}
		break;
				  }
	case POS_VAL:{
		StrCmdBase="r"+CMDPrefix[(int)(DeviceType)-1];
		break;
				 }
	case RANGE_VAL:{
		StrCmdBase="r"+CMDPrefix[(int)(DeviceType)-1]+"N";
		break;
				   }
	default :
		;
	}

	ret=SendCmdString(StrCmdBase,15000);
	if (ret!=DEVICE_OK)
		return ret;

	RetValue=0;

	std::string Risp= GetInputStr();

	if (Risp!=""){

		if (GetIntFromAnswer (StrCmdBase, GetInputStr(), &AnswerOK, &RetValue)!=DEVICE_OK){
			return ret;
		}
		else {
			* iValue=RetValue;
		}
	}
	else {

		//	, ,  , , CAMERA_SLIDER, , EMISSION_IT, ILLUMINATION_IT
		if (ValueType==RANGE_VAL && DeviceOnline[DeviceType-1]){

			switch (DeviceType){
			case EMISSION_FT:{
				* iValue=8;
				break;
							 }
			case DICHROIC_FT:{
				* iValue=5; // ATENZIONE: potrebbero essere 3
				break;
							 }
			case EXCITATION_FT:{
				* iValue=8; // ATENZIONE: potrebbero essere diversi ...
				break;
							   }
			case SPINNING_SLIDER:{
				* iValue=3; // ATENZIONE: potrebbero essere 2
				break;
			case SPINNING_MOTOR:{
				* iValue=2;
				break;
								}
								 }
			default :
				* iValue=0;
				break;

			}
		}
		else {
			* iValue=0;
		}
	}
	return DEVICE_OK;
}
// ----------------------------------------------------------------------------------------------
int XLightHub::IsOnline(TDevicelType DeviceType){

	int ret=DEVICE_OK;
	int RetValue;

	// check for device status
	// ========================================= STATUS ==================================================
	ret=GetDeviceValue(DeviceType, WORK_VAL, &RetValue);
	if (ret!=DEVICE_OK)
		return ret;


	// management for Iris devices
	if (DeviceType==EMISSION_IT){
		RetValue=(RetValue&2)!=0? 1:0;
	}
	else if (DeviceType==ILLUMINATION_IT){
		RetValue=(RetValue&1)!=0? 1:0;
	}
	// ------------------- SET DEVICE CONNECTED --------------------------
	DeviceOnline[DeviceType-1]=((bool) RetValue);

	// if not online return
	if (!DeviceOnline[DeviceType-1]){
		return DEVICE_OK;
	}

	// ========================================= RANGE ==================================================
	// if online get range for device (not for spinning motor)
	if (DeviceType!=SPINNING_MOTOR){
		ret=GetDeviceValue(DeviceType, RANGE_VAL, &RetValue);
		if (ret!=DEVICE_OK)
			return ret;

		MaxPositions[DeviceType-1]=RetValue;
		if (RetValue == 0){
			// se ottengo 0 setto il dispositivo come offline
			DeviceOnline[DeviceType-1]=false;// metto a offline per sicurezza
			return DEVICE_OK;
		}
	}
	else {
		MaxPositions[DeviceType-1]=2;
	}

	// ========================================= POSITION ==================================================

	ret=GetDeviceValue(DeviceType, POS_VAL, &RetValue);
	if (ret!=DEVICE_OK)
		return ret;

	// se è un device con base 1 riduco il tutto 
	if (DeviceType>=EMISSION_FT && DeviceType<=EXCITATION_FT){
		if (RetValue==0){
			InitialPositions[DeviceType-1]=1;
			DeviceOnline[DeviceType-1]=false; // whell cant ba in posizion 0: this position is for not woking device
			return DEVICE_OK;
		}
		else {
			InitialPositions[DeviceType-1]=RetValue-1;
		}
	}
	else if (DeviceType>=SPINNING_SLIDER && DeviceType<=SPINNING_MOTOR){
		InitialPositions[DeviceType-1]=RetValue; // this devices is 0 based positions
	}
	else {
		InitialPositions[DeviceType-1]=RetValue;
	}

	return DEVICE_OK;

}
// ----------------------------------------------------------------------------------------------

int XLightHub::GetControllerInfo()
{
	// answer to v-command
	int ret=SendCmdString("v",15000);
	if (ret!=DEVICE_OK)
		return ret;

	if (GetInputStr().find("Crest driver Ver")!=0){
		return ERR_XLIGHT_NOT_FOUND;
	}

	
	// read wheel filter position to check working state
	ret=IsOnline(EMISSION_FT);
	if (ret!=DEVICE_OK)
		return ret;
	ret=IsOnline(DICHROIC_FT);
	if (ret!=DEVICE_OK){
		return ret;
	}
	ret=IsOnline(EXCITATION_FT);
	if (ret!=DEVICE_OK){
		return ret;
	}
	ret=IsOnline(SPINNING_SLIDER);
	if (ret!=DEVICE_OK){
		return ret;
	}
	ret=IsOnline(SPINNING_MOTOR);
	if (ret!=DEVICE_OK){
		return ret;
	}

	ret=IsOnline(CAMERA_SLIDER);
	if (ret!=DEVICE_OK){
		return ret;
	}
	ret=IsOnline(EMISSION_IT);
	if (ret!=DEVICE_OK){
		return ret;
	}

	ret=IsOnline(ILLUMINATION_IT);
	if (ret!=DEVICE_OK){
		return ret;
	}
	

	return ret;

}
// ----------------------------------------------------------------------------------------------

// ----------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------
//						Serial port: communication and command management 
// ----------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------


// ----------------------------------------------------------------------------------------------
void XLightHub::ClearAllRcvBuf() {
	// Read whatever has been received so far:
	unsigned long read;
	(GetCoreCallback())->ReadFromSerial(this, port_.c_str(), (unsigned char*) rcvBuf_, RCV_BUF_LENGTH, read);
	// Delete it all:
	memset(rcvBuf_, 0, RCV_BUF_LENGTH);
}
// ----------------------------------------------------------------------------------------------
int XLightHub::GetIntFromAnswer(std::string cmdbase_str, std::string answ_str, bool *ans_present, int *answ_value){

	if (answ_str.compare("\r")==0){
		*ans_present=false;
		return DEVICE_OK;
	} else if (answ_str.find(cmdbase_str)!=0){

		return ERR_COMMAND_EXECUTION_ERROR;
	}
	else {

		answ_str=answ_str.substr(cmdbase_str.length()); 
		try
		{
			*answ_value=std::stoi ((char*)answ_str.c_str());
			return DEVICE_OK;
		}
		catch (...)
		{
			return ERR_COMMAND_EXECUTION_ERROR;
		}
	}
}
// ----------------------------------------------------------------------------------------------
int  XLightHub::SendCmdString(std::string pcCmdTxt, unsigned uCmdTmOut, unsigned uRetry){
	


	
	int ret = DEVICE_OK;
	std::string MIOERRORE;
	MM::Core * pCoreCBK=GetCoreCallback();
	// empty the Rx serial buffer before sending command
	ClearAllRcvBuf();


	// send command
	rcvBuf_[0] = '\t';

	ret = pCoreCBK->SetSerialCommand(this, port_.c_str(), pcCmdTxt.c_str(), "\r");
	if (DEVICE_OK != ret)
		return ret;

	//to ensure wheel finishes movement, query the serial line waiting for echoed 
	// command executed which means XLight accomplished the movement

	MM::MMTime startTime = pCoreCBK->GetCurrentMMTime();

	while (rcvBuf_[0]=='\t')
	{
		// check the timeout if not 0
		if (uCmdTmOut!=0 && (pCoreCBK->GetCurrentMMTime() - startTime).getMsec() > uCmdTmOut){
			return ERR_COMMUNICATION_TIMEOUT;
		}
		ret = pCoreCBK->GetSerialAnswer(this, port_.c_str(), RCV_BUF_LENGTH, rcvBuf_, "\r");
		if (DEVICE_OK != ret){
			return ret;
		}
	}

	return  ret;
}
// ----------------------------------------------------------------------------------------------
std::string XLightHub::GetInputStr(){

	return rcvBuf_;
}
// ----------------------------------------------------------------------------------------------





///////////////////////////////////////////////////////////////////////////////
// Action handlers
///////////////////////////////////////////////////////////////////////////////
/*
* Sets the Serial Port to be used.
* Should be called before initialization
*/
int XLightHub::OnPort(MM::PropertyBase* pProp, MM::ActionType eAct) {


//	MM::Device* cameramia = (GetDevice("pippo"));

	if (eAct == MM::BeforeGet) {
		pProp->Set(port_.c_str());

	} else if (eAct == MM::AfterSet) {
		if (initialized_) {
			// revert
			pProp->Set(port_.c_str());
			//return ERR_PORT_CHANGE_FORBIDDEN;
		}

		bool DefaultData=port_=="Undefined";
		pProp->Get(port_);

	/*	if (DefaultData){
		//	      GetCoreCallback()->SetDeviceProperty(port_.c_str(), MM::g_Keyword_Handshaking, g_Off);
        // GetCoreCallback()->SetDeviceProperty(port_.c_str(), MM::g_Keyword_BaudRate, "115000" );
       //  GetCoreCallback()->SetDeviceProperty(port_.c_str(), MM::g_Keyword_StopBits, "1");
         // Arduino timed out in GetControllerVersion even if AnswerTimeout  = 300 ms
        // GetCoreCallback()->SetDeviceProperty(port_.c_str(), "AnswerTimeout", "30000.0");
       //  GetCoreCallback()->SetDeviceProperty(port_.c_str(), "DelayBetweenCharsMs", "0");
		}*/





		this->SetPort(port_.c_str());
	}

	return DEVICE_OK;
}
// ----------------------------------------------------------------------------
void XLightHub::SetPort(const char* port) {
	port_ = port;
}

// ==================================================================================================


XLightStateDevice::XLightStateDevice(){

	initialized_=false;
	DeviceInfo_.Connected=false;
	DeviceInfo_.Working=false;
	DeviceInfo_.PrefixCMD="*";

	DeviceInfo_.Value=-1;
	DeviceInfo_.MaxValue=0;

	DeviceInfo_.name_="Unknow";
	DeviceInfo_.description_="Unknow";

	InitializeDefaultErrorMessages();
	SetErrorText(ERR_UNKNOWN_POSITION, "Requested position not available in this device");
	// EnableDelay(); // signals that the dealy setting will be used
	// parent ID display
	CreateHubIDProperty();

}
// ----------------------------------------------------------------------------------------------
XLightStateDevice::~XLightStateDevice()
{
	Shutdown();
}
// ----------------------------------------------------------------------------------------------
unsigned long XLightStateDevice::GetNumberOfPositions()const {

return DeviceInfo_.MaxValue;
}
// ----------------------------------------------------------------------------------------------
void XLightStateDevice::GetName(char* Name) const
{
	CDeviceUtils::CopyLimitedString(Name, DeviceInfo_.name_.c_str());
}
// ----------------------------------------------------------------------------------------------
int XLightStateDevice::Initialize()
{
	DeviceInfo_.Value =InitialPositions[DeviceInfo_.DeviceType_-1]; // RIDURRE
	DeviceInfo_.MaxValue=MaxPositions[DeviceInfo_.DeviceType_-1];

	XLightHub* hub = static_cast<XLightHub*>(GetParentHub());
	if (hub)
	{
		char hubLabel[MM::MaxStrLength];
		hub->GetLabel(hubLabel);
		SetParentID(hubLabel); // for backward comp.
	}
	else
		LogMessage(NoHubError);

	if (initialized_)
		return DEVICE_OK;

	// set property list
	// -----------------

	// Name
	int ret = CreateStringProperty(MM::g_Keyword_Name, DeviceInfo_.name_.c_str(), true);
	if (DEVICE_OK != ret)
		return ret;

	// Description
	ret = CreateStringProperty(MM::g_Keyword_Description, DeviceInfo_.description_.c_str(), true);
	if (DEVICE_OK != ret)
		return ret;

	// State
	CPropertyAction * pAct = new CPropertyAction(this, &XLightStateDevice::OnState);
	ret = CreateProperty(MM::g_Keyword_State, (char *)(std::to_string((long long) DeviceInfo_.Value).c_str()), MM::Integer, false, pAct);// VALUE
	if (ret != DEVICE_OK)
		return ret;
	//CPropertyAction* pAct = new CPropertyAction (this, &XLightStateDevice::OnNumberOfStates);
	//CreateIntegerProperty("Number of positions", 0, false, pAct, true);

	// ora setto tutte le possibili posizioni
	for (int i=0; i<DeviceInfo_.MaxValue; i++){
		AddAllowedValue(MM::g_Keyword_State, (char*)(std::to_string((long long) i)).c_str());
	}

	std::string labelpos=PositionLabels[DeviceInfo_.DeviceType_-1]+std::to_string((long long)(DeviceInfo_.Value+1));// VALUE

	// Label                                                                  
	CPropertyAction * pAct1 = new CPropertyAction(this, &XLightStateDevice::OnLabel);
	ret = CreateProperty(MM::g_Keyword_Label, "Undefined", MM::String, false, pAct1);
	if (ret != DEVICE_OK)
		return ret;

	for (int i=0; i<DeviceInfo_.MaxValue; i++){
		if (DeviceInfo_.DeviceType_>=EMISSION_FT && DeviceInfo_.DeviceType_<=EXCITATION_FT )
			labelpos=PositionLabels[DeviceInfo_.DeviceType_-1]+std::to_string((long long)(i+1));
		else if (DeviceInfo_.DeviceType_==SPINNING_SLIDER){
			if (i==0)
				labelpos=PositionLabels[DeviceInfo_.DeviceType_-1]+" out";
			else 
				labelpos=PositionLabels[DeviceInfo_.DeviceType_-1]+std::to_string((long long)(i));
		}
		else if (DeviceInfo_.DeviceType_==CAMERA_SLIDER){
			if (i==DeviceInfo_.MaxValue-1)
				labelpos=PositionLabels[DeviceInfo_.DeviceType_-1]+" out";
			else 
				labelpos=PositionLabels[DeviceInfo_.DeviceType_-1]+std::to_string((long long)(i));
		}
		else if (DeviceInfo_.DeviceType_==SPINNING_MOTOR){
			if (i==0)
				labelpos="OFF";
			else 
				labelpos="ON";
		}

		SetPositionLabel(i, (char*)(labelpos.c_str()));
	}

	ret = UpdateStatus();
	if (ret != DEVICE_OK)
		return ret;

	initialized_ = true;
	return DEVICE_OK;
}
// ----------------------------------------------------------------------------------------------
bool XLightStateDevice::Busy()
{
	return false;
}
// ----------------------------------------------------------------------------------------------
int XLightStateDevice::Shutdown()
{
	initialized_ = false;
	return DEVICE_OK;
}
// ----------------------------------------------------------------------------------------------
///////////////////////////////////////////////////////////////////////////////
// Action handlers
///////////////////////////////////////////////////////////////////////////////
// ----------------------------------------------------------------------------------------------
int XLightStateDevice::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{

	if (eAct == MM::BeforeGet)
	{
		pProp->Set(DeviceInfo_.Value); // VALUE
		// nothing to do, let the caller to use cached property
	}
	else if (eAct == MM::AfterSet)
	{
		// Set timer for the Busy signal
		//		changedTime_ = GetCurrentMMTime();

		long pos;
		pProp->Get(pos);
		if (pos >= DeviceInfo_.MaxValue || pos < 0)
		{
			pProp->Set(DeviceInfo_.Value); // VALUE
			return ERR_UNKNOWN_POSITION;
		}
		if (setPosition (pos)==DEVICE_OK)
			DeviceInfo_.Value = pos;// RIDURRE
	}

	return DEVICE_OK;
}
// ----------------------------------------------------------------------------------------------
int XLightStateDevice::OnNumberOfStates(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(DeviceInfo_.MaxValue); // VALUE
	}
	else if (eAct == MM::AfterSet)
	{
		if (!initialized_)
			pProp->Get(DeviceInfo_.MaxValue);
	}

	return DEVICE_OK;
}
// ----------------------------------------------------------------------------------------------
int XLightStateDevice::setPosition (long lValue){
	TCmdType CmdType=SETPOS_CMD;
	//long ValRect=DeviceInfo_.DeviceType_>=SPINNING_SLIDER?lValue-1:lValue;
	return pXLightHub->ExecuteCmd(CmdType, &DeviceInfo_, lValue);
}
// ----------------------------------------------------------------------------------------------
int XLightStateDevice::getPosition (){
	TCmdType CmdType=GETPOS_CMD;
	return pXLightHub->ExecuteCmd(CmdType, &DeviceInfo_);
}
// ----------------------------------------------------------------------------------------------
int XLightStateDevice::getPositionsNumber (){

if (DeviceInfo_.DeviceType_==SPINNING_MOTOR){
		DeviceInfo_.MaxValue=2;
		return DEVICE_OK;
	}

	TCmdType CmdType=GETNUMPOS_CMD;
	return pXLightHub->ExecuteCmd(CmdType, &DeviceInfo_);

}
// ----------------------------------------------------------------------------------------------

// Iris device
IrisDevice::IrisDevice(){

	initialized_=false;

	DeviceInfo_.Connected=false;
	DeviceInfo_.Working=false;
	DeviceInfo_.PrefixCMD="*";

	DeviceInfo_.Value=-1;
	DeviceInfo_.MaxValue=0;

	DeviceInfo_.name_="Unknow";
	DeviceInfo_.description_="Unknow";

	InitializeDefaultErrorMessages();
	SetErrorText(ERR_UNKNOWN_POSITION, "Requested position not available in this device");
	// EnableDelay(); // signals that the dealy setting will be used
	// parent ID display
	CreateHubIDProperty();
}
// ----------------------------------------------------------------------------------------------
IrisDevice::~IrisDevice()
{
	Shutdown();
}
// ----------------------------------------------------------------------------------------------

void IrisDevice::GetName(char* Name) const
{
	CDeviceUtils::CopyLimitedString(Name, DeviceInfo_.name_.c_str());
}
// ----------------------------------------------------------------------------------------------
int IrisDevice::Initialize()
{
	int ret;
	XLightHub* hub = static_cast<XLightHub*>(GetParentHub());
	if (hub)
	{
		char hubLabel[MM::MaxStrLength];
		hub->GetLabel(hubLabel);
		SetParentID(hubLabel); // for backward comp.
	}
	else
		LogMessage(NoHubError);


	if (!initialized_) {

		DeviceInfo_.Value =InitialPositions[DeviceInfo_.DeviceType_-1];
		DeviceInfo_.MaxValue=MaxPositions[DeviceInfo_.DeviceType_-1];


		// Set property list
		// -----------------
		CPropertyAction* pAct = new CPropertyAction(this, &IrisDevice::OnSetAperture);
		ret = CreateIntegerProperty(g_IrisAperture,  (int) DeviceInfo_.Value,  false, pAct);
		if (ret != DEVICE_OK)
			return ret;


		SetPropertyLimits(g_IrisAperture, 0, DeviceInfo_.MaxValue);

		ret = UpdateStatus();
		if (ret != DEVICE_OK)
			return ret;

		initialized_ = true;
	}
	return DEVICE_OK;
}
// ----------------------------------------------------------------------------------------------
bool IrisDevice::Busy()
{
	return false;
}
// ----------------------------------------------------------------------------------------------
int IrisDevice::Shutdown()
{
	initialized_ = false;

	return DEVICE_OK;
}
// ----------------------------------------------------------------------------------------------
///////////////////////////////////////////////////////////////////////////////
// Action handlers
///////////////////////////////////////////////////////////////////////////////
// ----------------------------------------------------------------------------------------------

int IrisDevice::setaperture (long lValue){
	
	
	TCmdType CmdType=SETPOS_CMD;

	return pXLightHub->ExecuteCmd(CmdType, &DeviceInfo_, lValue);
}
// ----------------------------------------------------------------------------------------------

int IrisDevice::OnSetAperture(MM::PropertyBase* pProp, MM::ActionType eAct)
{

	if (eAct == MM::BeforeGet) {
		pProp->Set((long) DeviceInfo_.Value);
	}
	else if (eAct == MM::AfterSet) {
		//actualAperture_=actualAperture_+10;

		//actualAperture_
		long pos;
		pProp->Get(pos);
		if (pos > DeviceInfo_.MaxValue || pos < 0)
		{
			pProp->Set((long) DeviceInfo_.Value); // revert
			return ERR_UNKNOWN_POSITION;
		}
		if (setaperture (pos)==DEVICE_OK)
			DeviceInfo_.Value = pos;


	}
	return DEVICE_OK;

}
// ----------------------------------------------------------------------------------------------
