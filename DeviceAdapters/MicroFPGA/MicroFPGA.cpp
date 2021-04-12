//////////////////////////////////////////////////////////////////////////////
// FILE:          MicroFPGA.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Adapter for MicroFPGA, a FPGA platform using FPGA boards from
//                Alchitry. The adapter must be used with a special firmware, see:
//                https://github.com/jdeschamps/MicroFPGA
// COPYRIGHT:     EMBL
// LICENSE:       LGPL
//
// AUTHOR:        Joran Deschamps, EMBL, 2020
//				  
//


#include "MicroFPGA.h"
#include "ModuleInterface.h"

#ifdef WIN32
#include <windows.h>
#define snprintf _snprintf 
#endif

const char* g_DeviceNameMicroFPGAHub = "MicroFPGA-Hub";
const char* g_DeviceNameLaserTrig = "LaserTrig";
const char* g_DeviceNameAnalogInput = "AnalogInput";
const char* g_DeviceNamePWM = "PWM";
const char* g_DeviceNameTTL = "TTL";
const char* g_DeviceNameServos = "Servos";

//////////////////////////////////////////////////////////////////////////////
/// Constants that should match the ones in the firmware
const int g_version = 2;
const int g_id_au = 79;
const int g_id_cu = 29;

const int g_maxlasers = 8;
const int g_maxanaloginput = 8;
const int g_maxttl = 5;
const int g_maxpwm = 5;
const int g_maxservos = 7;

const int g_offsetaddressLaserMode = 0;
const int g_offsetaddressLaserDuration = g_offsetaddressLaserMode+g_maxlasers;
const int g_offsetaddressLaserSequence = g_offsetaddressLaserDuration+g_maxlasers;
const int g_offsetaddressTTL = g_offsetaddressLaserSequence+g_maxlasers;
const int g_offsetaddressServo = g_offsetaddressTTL+g_maxttl;
const int g_offsetaddressPWM = g_offsetaddressServo+g_maxservos;
const int g_offsetaddressAnalogInput = g_offsetaddressPWM+g_maxpwm;

const int g_address_version = 100;
const int g_address_id = 101;

// static lock
MMThreadLock MicroFPGAHub::lock_;

///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////
MODULE_API void InitializeModuleData()
{
	RegisterDevice(g_DeviceNameMicroFPGAHub, MM::HubDevice, "Hub (required)");
	RegisterDevice(g_DeviceNameLaserTrig, MM::GenericDevice, "Laser Trigger");
	RegisterDevice(g_DeviceNameAnalogInput, MM::GenericDevice, "Analog Input");
	RegisterDevice(g_DeviceNamePWM, MM::GenericDevice, "PWM Output");
	RegisterDevice(g_DeviceNameTTL, MM::GenericDevice, "TTL Output");
	RegisterDevice(g_DeviceNameServos, MM::GenericDevice, "Servos");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
	if (deviceName == 0)
		return 0;

	if (strcmp(deviceName, g_DeviceNameMicroFPGAHub) == 0)
	{
		return new MicroFPGAHub;
	}
	else if (strcmp(deviceName, g_DeviceNameLaserTrig) == 0)
	{
		return new LaserTrig;
	}
	else if (strcmp(deviceName, g_DeviceNameAnalogInput) == 0)
	{
		return new AnalogInput;
	}
	else if (strcmp(deviceName, g_DeviceNamePWM) == 0)
	{
		return new PWM; 
	}
	else if (strcmp(deviceName, g_DeviceNameTTL) == 0)
	{
		return new TTL; 
	}
	else if (strcmp(deviceName, g_DeviceNameServos) == 0)
	{
		return new Servo;
	}

	return 0;
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
	delete pDevice;
}

///////////////////////////////////////////////////////////////////////////////
// MicroFPGA Hub implementation
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
//
MicroFPGAHub::MicroFPGAHub() :
	initialized_ (false)
{
	portAvailable_ = false;

	InitializeDefaultErrorMessages();
	SetErrorText(ERR_PORT_OPEN_FAILED, "Failed opening MicroFPGA USB device.");
	SetErrorText(ERR_BOARD_NOT_FOUND, "Did not find an MicroFPGA board with the correct firmware. Is the MicroFPGA board connected to this serial port?");
	SetErrorText(ERR_NO_PORT_SET, "Hub Device not found. The MicroFPGA Hub device is needed to create this device.");
	SetErrorText(ERR_VERSION_MISMATCH, "The firmware version on the MicroFPGA is not compatible with this adapter. Please use firmware version 2.");
	SetErrorText(ERR_UNKNOWN_ID, "The board ID is unknwown.");
	SetErrorText(ERR_COMMAND_UNKNOWN, "An unknown command was sent to the MicroFPGA.");

	CPropertyAction* pAct = new CPropertyAction(this, &MicroFPGAHub::OnPort);
	CreateProperty(MM::g_Keyword_Port, "Undefined", MM::String, false, pAct, true);
}

MicroFPGAHub::~MicroFPGAHub()
{
	Shutdown();
}

void MicroFPGAHub::GetName(char* name) const
{
	CDeviceUtils::CopyLimitedString(name, g_DeviceNameMicroFPGAHub);
}

bool MicroFPGAHub::Busy()
{
	return false;
}

MM::DeviceDetectionStatus MicroFPGAHub::DetectDevice(void)
{
	// Code adapted from Arduino.cpp, Micro-Manager, written by Nico Stuurman and Karl Hoover
	if (initialized_)
		return MM::CanCommunicate;

	MM::DeviceDetectionStatus result = MM::Misconfigured;
	char answerTO[MM::MaxStrLength];

	try{
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
			GetCoreCallback()->SetDeviceProperty(port_.c_str(), MM::g_Keyword_Handshaking, "0");
			GetCoreCallback()->SetDeviceProperty(port_.c_str(), MM::g_Keyword_BaudRate, "9600" );
			GetCoreCallback()->SetDeviceProperty(port_.c_str(), MM::g_Keyword_StopBits, "1");
			GetCoreCallback()->SetDeviceProperty(port_.c_str(), "AnswerTimeout", "500.0");
			GetCoreCallback()->SetDeviceProperty(port_.c_str(), "DelayBetweenCharsMs", "0");
			MM::Device* pS = GetCoreCallback()->GetDevice(this, port_.c_str());
			pS->Initialize();

			CDeviceUtils::SleepMs(100);
			MMThreadGuard myLock(lock_);
			PurgeComPort(port_.c_str());
			long v = 0;
			int ret = GetControllerVersion(v);

			if( DEVICE_OK != ret ){
				LogMessageCode(ret,true);
			} else {
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


int MicroFPGAHub::Initialize()
{
	// Name
	int ret = CreateProperty(MM::g_Keyword_Name, g_DeviceNameMicroFPGAHub, MM::String, true);
	if (DEVICE_OK != ret)
		return ret;

	MMThreadGuard myLock(lock_);

	PurgeComPort(port_.c_str());

	// Gets controller version
	ret = GetControllerVersion(version_);
	if( DEVICE_OK != ret)
		return ret;

	// Verifies that the version of the firmware and adapter match
	if (g_version != version_){
		return ERR_VERSION_MISMATCH;
	}

	//CPropertyAction* pAct = new CPropertyAction(this, &MicroFPGAHub::OnVersion);
	std::ostringstream sversion;
	sversion << version_;
	CreateProperty("MicroFPGA version", sversion.str().c_str(), MM::Integer, true);

	// Checks the ID
	ret = GetID(id_);
	if( DEVICE_OK != ret)
		return ret;

	if (g_id_au == id_){
		CreateProperty("MicroFPGA ID", "Au", MM::String, true);
	} else if(g_id_cu == id_){
		CreateProperty("MicroFPGA ID", "Cu", MM::String, true);
	}else {
		return ERR_UNKNOWN_ID;
	}

	initialized_ = true;
	return DEVICE_OK;
}

int MicroFPGAHub::DetectInstalledDevices()
{
	// Code adapted from Arduino.cpp, Micro-Manager, written by Nico Stuurman and Karl Hoover
	if (MM::CanCommunicate == DetectDevice()) 
	{
		std::vector<std::string> peripherals; 
		peripherals.clear();
		peripherals.push_back(g_DeviceNameLaserTrig);
		
		// Only the Au has an ADC
		if(id_ == g_id_au){
			peripherals.push_back(g_DeviceNameAnalogInput);
		}

		peripherals.push_back(g_DeviceNamePWM);
		peripherals.push_back(g_DeviceNameTTL);
		peripherals.push_back(g_DeviceNameServos);
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

int MicroFPGAHub::Shutdown()
{
	initialized_ = false;
	return DEVICE_OK;
}

int MicroFPGAHub::GetControllerVersion(long& version)
{
	int ret = SendReadRequest(g_address_version);
	if (ret != DEVICE_OK)
		return ret;

	ret = ReadAnswer(version);
	if (ret != DEVICE_OK)
		return ret;

	return ret;
}

int MicroFPGAHub::GetID(long& id)
{
	int ret = SendReadRequest(g_address_id);
	if (ret != DEVICE_OK)
		return ret;

	ret = ReadAnswer(id);
	if (ret != DEVICE_OK)
		return ret;

	return ret;
}

int MicroFPGAHub::SendWriteRequest(long address, long value)
{   
	unsigned char command[9];
	command[0] = (1 << 7);	// 1 = write
	command[1] = static_cast<char>(address);	// put the least significant byte
	command[2] = static_cast<char>((address >> 8));	// put the least but one significant byte 
	command[3] = static_cast<char>((address >> 16));	
	command[4] = static_cast<char>((address >> 24)); // address and value are 32 bits long
	command[5] = static_cast<char>(value);	
	command[6] = static_cast<char>((value >> 8));	
	command[7] = static_cast<char>((value >> 16));	
	command[8] = static_cast<char>((value >> 24));	

	int ret = WriteToComPortH((const unsigned char*) command, 9);

	return ret;
}
int MicroFPGAHub::SendReadRequest(long address){
	unsigned char command[5];
	command[0] = (0 << 7);	
	command[1] = static_cast<char>(address);	
	command[2] = static_cast<char>((address >> 8));	
	command[3] = static_cast<char>((address >> 16));	
	command[4] = static_cast<char>((address >> 24));

	int ret = WriteToComPortH((const unsigned char*) command, 5);

	return ret;
}

int MicroFPGAHub::ReadAnswer(long& ans){
	unsigned char* answer = new unsigned char[4];
	
	// Code adapted from Arduino.cpp, Micro-Manager, written by Nico Stuurman and Karl Hoover
	MM::MMTime startTime = GetCurrentMMTime();  
	unsigned long bytesRead = 0;

	while ((bytesRead < 4) && ( (GetCurrentMMTime() - startTime).getMsec() < 500)) {
		unsigned long bR;
		int ret = ReadFromComPortH(answer + bytesRead, 4 - bytesRead, bR);
		if (ret != DEVICE_OK)
			return ret;
		bytesRead += bR;
	}

	// Format answer
	int tmp = answer[3];
	for(int i=1;i<4;i++){
		tmp = tmp << 8;
		tmp = tmp | answer[3-i];
	}

	ans = tmp;
	
	// If unknown command answer
	if(ans == ERR_COMMAND_UNKNOWN){
		return ERR_COMMAND_UNKNOWN;
	}

	return DEVICE_OK;
}

int MicroFPGAHub::OnPort(MM::PropertyBase* pProp, MM::ActionType pAct)
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

///////////////////////////////////////////////////////////////////////////////////////////
//////
LaserTrig::LaserTrig() :
initialized_ (false),
	busy_(false)
{
	InitializeDefaultErrorMessages();

	// Custom error messages
	SetErrorText(ERR_NO_PORT_SET, "Hub Device not found. The MicroFPGA Hub device is needed to create this device");
	SetErrorText(ERR_COMMAND_UNKNOWN, "An unknown command was sent to the MicroFPGA.");

	// Description
	int ret = CreateProperty(MM::g_Keyword_Description, "MicroFPGA laser triggering system", MM::String, true);
	assert(DEVICE_OK == ret);

	// Name
	ret = CreateProperty(MM::g_Keyword_Name, g_DeviceNameLaserTrig, MM::String, true);
	assert(DEVICE_OK == ret);

	// Number of lasers
	CPropertyAction* pAct = new CPropertyAction(this, &LaserTrig::OnNumberOfLasers);
	CreateProperty("Number of lasers", "4", MM::Integer, false, pAct, true);
	SetPropertyLimits("Number of lasers", 1, g_maxlasers);
}


LaserTrig::~LaserTrig()
{
	Shutdown();
}

void LaserTrig::GetName(char* name) const
{
	CDeviceUtils::CopyLimitedString(name, g_DeviceNameLaserTrig);
}


int LaserTrig::Initialize()
{
	// Parent ID display	
	MicroFPGAHub* hub = static_cast<MicroFPGAHub*>(GetParentHub());
	if (!hub) {
		return ERR_NO_PORT_SET;
	}
	char hubLabel[MM::MaxStrLength];
	hub->GetLabel(hubLabel);
	SetParentID(hubLabel);
	CreateHubIDProperty();

	// Allocate memory for lasers
	mode_ = new long [GetNumberOfLasers()];
	duration_ = new long [GetNumberOfLasers()];
	sequence_ = new long [GetNumberOfLasers()];

	CPropertyActionEx *pExAct;
	int nRet;

	for(unsigned int i=0;i<GetNumberOfLasers();i++){	
		mode_[i] = 0;
		duration_[i] = 0;
		sequence_[i] = 0;

		std::stringstream mode;
		std::stringstream dura;
		std::stringstream seq;
		mode << "Mode" << i;
		dura << "Duration" << i;
		seq << "Sequence" << i;

		pExAct = new CPropertyActionEx (this, &LaserTrig::OnDuration,i);
		nRet = CreateProperty(dura.str().c_str(), "0", MM::Integer, false, pExAct);
		if (nRet != DEVICE_OK)
			return nRet;
		SetPropertyLimits(dura.str().c_str(), 0, 65535);   

		pExAct = new CPropertyActionEx (this, &LaserTrig::OnMode,i);
		nRet = CreateProperty(mode.str().c_str(), "0", MM::Integer, false, pExAct);
		if (nRet != DEVICE_OK)
			return nRet;
		SetPropertyLimits(mode.str().c_str(), 0, 4);

		pExAct = new CPropertyActionEx (this, &LaserTrig::OnSequence,i);
		nRet = CreateProperty(seq.str().c_str(), "65535", MM::Integer, false, pExAct);
		if (nRet != DEVICE_OK)
			return nRet;
		SetPropertyLimits(seq.str().c_str(), 0, 65535);
	}

	nRet = UpdateStatus();
	if (nRet != DEVICE_OK)
		return nRet;

	initialized_ = true;

	return DEVICE_OK;
}

int LaserTrig::Shutdown()
{
	initialized_ = false;
	return DEVICE_OK;
}

int LaserTrig::WriteToPort(long address, long value)
{
	MicroFPGAHub* hub = static_cast<MicroFPGAHub*>(GetParentHub());
	if (!hub) {
		return ERR_NO_PORT_SET;
	}

	MMThreadGuard myLock(hub->GetLock());

	hub->PurgeComPortH();

	int ret = hub->SendWriteRequest(address, value);
	if (ret != DEVICE_OK)
		return ret;

	return DEVICE_OK;
}

int LaserTrig::ReadFromPort(long& answer)
{
	MicroFPGAHub* hub = static_cast<MicroFPGAHub*>(GetParentHub());
	if (!hub) {
		return ERR_NO_PORT_SET;
	}
	int ret = hub->ReadAnswer(answer);
	if (ret != DEVICE_OK)
		return ret;

	return DEVICE_OK;
}


///////////////////////////////////////
/////////// Action handlers
int LaserTrig::OnNumberOfLasers(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	if (pAct == MM::BeforeGet){
		pProp->Set(numlasers_);
	} else if (pAct == MM::AfterSet){
		pProp->Get(numlasers_);
	}
	return DEVICE_OK;
}

int LaserTrig::OnMode(MM::PropertyBase* pProp, MM::ActionType pAct, long laser)
{
	if (pAct == MM::BeforeGet)
	{
		MicroFPGAHub* hub = static_cast<MicroFPGAHub*>(GetParentHub());
		if (!hub){
			return ERR_NO_PORT_SET;
		}

		MMThreadGuard myLock(hub->GetLock());

		int ret = hub->SendReadRequest(g_offsetaddressLaserMode+laser);
		if (ret != DEVICE_OK)
			return ret;

		long answer;
		ret = ReadFromPort(answer);
		if (ret != DEVICE_OK)
			return ret;

		pProp->Set(answer);
		mode_[laser]=answer;
	}
	else if (pAct == MM::AfterSet)
	{
		long mode;
		pProp->Get(mode);

		int ret = WriteToPort(g_offsetaddressLaserMode+laser,mode);  
		if (ret != DEVICE_OK)
			return ret;

		mode_[laser] = mode;
	}

	return DEVICE_OK;
}

int LaserTrig::OnDuration(MM::PropertyBase* pProp, MM::ActionType pAct, long laser)
{
	if (pAct == MM::BeforeGet)
	{
		MicroFPGAHub* hub = static_cast<MicroFPGAHub*>(GetParentHub());
		if (!hub){
			return ERR_NO_PORT_SET;
		}

		MMThreadGuard myLock(hub->GetLock());

		int ret = hub->SendReadRequest(g_offsetaddressLaserDuration+laser);
		if (ret != DEVICE_OK)
			return ret;

		long answer;
		ret = ReadFromPort(answer);
		if (ret != DEVICE_OK)
			return ret;

		pProp->Set(answer);
		duration_[laser]=answer;
	}
	else if (pAct == MM::AfterSet)
	{
		long pos;
		pProp->Get(pos);

		int ret = WriteToPort(g_offsetaddressLaserDuration+laser,pos); 
		if (ret != DEVICE_OK)
			return ret;

		duration_[laser] = pos;
	}

	return DEVICE_OK;
}

int LaserTrig::OnSequence(MM::PropertyBase* pProp, MM::ActionType pAct, long laser)
{
	if (pAct == MM::BeforeGet)
	{
		MicroFPGAHub* hub = static_cast<MicroFPGAHub*>(GetParentHub());
		if (!hub){
			return ERR_NO_PORT_SET;
		}

		MMThreadGuard myLock(hub->GetLock());

		int ret = hub->SendReadRequest(g_offsetaddressLaserSequence+laser);
		if (ret != DEVICE_OK)
			return ret;

		long answer;
		ret = ReadFromPort(answer);
		if (ret != DEVICE_OK)
			return ret;

		pProp->Set(answer);
		sequence_[laser]=answer;
	}
	else if (pAct == MM::AfterSet)
	{
		long pos;
		pProp->Get(pos);

		int ret = WriteToPort(g_offsetaddressLaserSequence+laser,pos);
		if (ret != DEVICE_OK)
			return ret; 

		sequence_[laser] = pos;
	}

	return DEVICE_OK;
}


///////////////////////////////////////////////////////////////////////////////////////////
//////
TTL::TTL() :
initialized_ (false),
	busy_(false)
{
	InitializeDefaultErrorMessages();

	// Custom error messages
	SetErrorText(ERR_NO_PORT_SET, "Hub Device not found. The MicroFPGA Hub device is needed to create this device");
	SetErrorText(ERR_COMMAND_UNKNOWN, "An unknown command was sent to the MicroFPGA.");

	// Description
	int ret = CreateProperty(MM::g_Keyword_Description, "MicroFPGA TTL", MM::String, true);
	assert(DEVICE_OK == ret);

	// Name
	ret = CreateProperty(MM::g_Keyword_Name, g_DeviceNameTTL, MM::String, true);
	assert(DEVICE_OK == ret);

	// Number of TTL channels
	CPropertyAction* pAct = new CPropertyAction(this, &TTL::OnNumberOfChannels);
	CreateProperty("Number of channels", "4", MM::Integer, false, pAct, true);
	SetPropertyLimits("Number of channels", 1, g_maxttl);
}

TTL::~TTL()
{
	Shutdown();
}

void TTL::GetName(char* name) const
{
	CDeviceUtils::CopyLimitedString(name, g_DeviceNameTTL);
}


int TTL::Initialize()
{
	// Parent ID display	
	MicroFPGAHub* hub = static_cast<MicroFPGAHub*>(GetParentHub());
	if (!hub) {
		return ERR_NO_PORT_SET;
	}
	char hubLabel[MM::MaxStrLength];
	hub->GetLabel(hubLabel);
	SetParentID(hubLabel);
	CreateHubIDProperty();

	// State
	// -----

	// Allocate memory for TTLs
	state_ = new long [GetNumberOfChannels()];

	CPropertyActionEx *pExAct;
	int nRet;

	for(unsigned int i=0;i<GetNumberOfChannels();i++){
		state_[i]  = 0;

		std::stringstream sstm;
		sstm << "State" << i;

		pExAct = new CPropertyActionEx (this, &TTL::OnState,i);
		nRet = CreateProperty(sstm.str().c_str(), "0", MM::Integer, false, pExAct);
		if (nRet != DEVICE_OK)
			return nRet;
		AddAllowedValue(sstm.str().c_str(), "0");
		AddAllowedValue(sstm.str().c_str(), "1");
	}

	nRet = UpdateStatus();
	if (nRet != DEVICE_OK)
		return nRet;

	initialized_ = true;

	return DEVICE_OK;
}

int TTL::Shutdown()
{
	initialized_ = false;
	return DEVICE_OK;
}

int TTL::WriteToPort(long address, long state)
{
	MicroFPGAHub* hub = static_cast<MicroFPGAHub*>(GetParentHub());
	if (!hub) {
		return ERR_NO_PORT_SET;
	}

	MMThreadGuard myLock(hub->GetLock());

	hub->PurgeComPortH();

	int val = 0;
	if(state == 1){
		val = 1;
	} 
	int ret = hub->SendWriteRequest(address, val);
	if (ret != DEVICE_OK)
		return ret;

	return DEVICE_OK;
}

int TTL::ReadFromPort(long& answer)
{
	MicroFPGAHub* hub = static_cast<MicroFPGAHub*>(GetParentHub());
	if (!hub) {
		return ERR_NO_PORT_SET;
	}
	int ret = hub->ReadAnswer(answer);
	if (ret != DEVICE_OK)
		return ret;

	return DEVICE_OK;
}

///////////////////////////////////////
/////////// Action handlers
int TTL::OnNumberOfChannels(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	if (pAct == MM::BeforeGet){
		pProp->Set(numChannels_);
	} else if (pAct == MM::AfterSet){
		pProp->Get(numChannels_);
	}
	return DEVICE_OK;
}

int TTL::OnState(MM::PropertyBase* pProp, MM::ActionType pAct, long channel)
{
	if (pAct == MM::BeforeGet)
	{
		MicroFPGAHub* hub = static_cast<MicroFPGAHub*>(GetParentHub());
		if (!hub){
			return ERR_NO_PORT_SET;
		}

		MMThreadGuard myLock(hub->GetLock());

		int ret = hub->SendReadRequest(g_offsetaddressTTL+channel);
		if (ret != DEVICE_OK)
			return ret;

		long answer;
		ret = ReadFromPort(answer);
		if (ret != DEVICE_OK)
			return ret;

		pProp->Set(answer);
		state_[channel]=answer;
	}
	else if (pAct == MM::AfterSet)
	{
		long pos;
		pProp->Get(pos);

		int ret = WriteToPort(g_offsetaddressTTL+channel, pos); 
		if (ret != DEVICE_OK)
			return ret;

		state_[channel] = pos;
	}

	return DEVICE_OK;
}


///////////////////////////////////////////////////////////////////////////////////////////
//////
Servo::Servo() :
initialized_ (false),
	busy_(false)
{
	InitializeDefaultErrorMessages();

	// Custom error messages
	SetErrorText(ERR_NO_PORT_SET, "Hub Device not found. The MicroFPGA Hub device is needed to create this device");
	SetErrorText(ERR_COMMAND_UNKNOWN, "An unknown command was sent to the MicroFPGA.");

	// Description
	int ret = CreateProperty(MM::g_Keyword_Description, "MicroFPGA Servo controller", MM::String, true);
	assert(DEVICE_OK == ret);

	// Name
	ret = CreateProperty(MM::g_Keyword_Name, g_DeviceNameServos, MM::String, true);
	assert(DEVICE_OK == ret);

	// Number of servos
	CPropertyAction* pAct = new CPropertyAction(this, &Servo::OnNumberOfServos);
	CreateProperty("Number of Servos", "4", MM::Integer, false, pAct, true);
	SetPropertyLimits("Number of Servos", 1, g_maxservos);
}

Servo::~Servo()
{
	Shutdown();
}

void Servo::GetName(char* name) const
{
	CDeviceUtils::CopyLimitedString(name, g_DeviceNameServos);
}


int Servo::Initialize()
{
	// Parent ID display	
	MicroFPGAHub* hub = static_cast<MicroFPGAHub*>(GetParentHub());
	if (!hub) {
		return ERR_NO_PORT_SET;
	}
	char hubLabel[MM::MaxStrLength];
	hub->GetLabel(hubLabel);
	SetParentID(hubLabel);
	CreateHubIDProperty();

	// State
	// -----

	// Allocate memory for servos
	position_ = new long [GetNumberOfServos()];

	CPropertyActionEx *pExAct;
	int nRet;

	for(unsigned int i=0;i<GetNumberOfServos();i++){	
		position_[i] = 0;

		std::stringstream sstm;
		sstm << "Position" << i;

		pExAct = new CPropertyActionEx (this, &Servo::OnPosition,i);
		nRet = CreateProperty(sstm.str().c_str(), "0", MM::Integer, false, pExAct);
		if (nRet != DEVICE_OK)
			return nRet;
		SetPropertyLimits(sstm.str().c_str(), 0, 65535);
	}

	nRet = UpdateStatus();
	if (nRet != DEVICE_OK)
		return nRet;

	initialized_ = true;

	return DEVICE_OK;
}

int Servo::Shutdown()
{
	initialized_ = false;
	return DEVICE_OK;
}

int Servo::WriteToPort(long address, long value)
{
	MicroFPGAHub* hub = static_cast<MicroFPGAHub*>(GetParentHub());
	if (!hub) {
		return ERR_NO_PORT_SET;
	}

	MMThreadGuard myLock(hub->GetLock());

	hub->PurgeComPortH();


	int ret = hub->SendWriteRequest(address, value);
	if (ret != DEVICE_OK)
		return ret;

	return DEVICE_OK;
}

int Servo::ReadFromPort(long& answer)
{
	MicroFPGAHub* hub = static_cast<MicroFPGAHub*>(GetParentHub());
	if (!hub) {
		return ERR_NO_PORT_SET;
	}
	int ret = hub->ReadAnswer(answer);
	if (ret != DEVICE_OK)
		return ret;

	return DEVICE_OK;
}


///////////////////////////////////////
/////////// Action handlers
int Servo::OnNumberOfServos(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	if (pAct == MM::BeforeGet){
		pProp->Set(numServos_);
	} else if (pAct == MM::AfterSet){
		pProp->Get(numServos_);
	}
	return DEVICE_OK;
}

int Servo::OnPosition(MM::PropertyBase* pProp, MM::ActionType pAct, long servo)
{
	if (pAct == MM::BeforeGet)
	{
		MicroFPGAHub* hub = static_cast<MicroFPGAHub*>(GetParentHub());
		if (!hub){
			return ERR_NO_PORT_SET;
		}

		MMThreadGuard myLock(hub->GetLock());

		int ret = hub->SendReadRequest(g_offsetaddressServo+servo);
		if (ret != DEVICE_OK)
			return ret;

		long answer;
		ret = ReadFromPort(answer);
		if (ret != DEVICE_OK)
			return ret;


		pProp->Set(answer);
		position_[servo]=answer;
	}
	else if (pAct == MM::AfterSet)
	{
		long pos;
		pProp->Get(pos);

		int ret = WriteToPort(g_offsetaddressServo+servo,pos); 
		if (ret != DEVICE_OK)
			return ret;

		position_[servo] = pos;
	}

	return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////////////////
//////
PWM::PWM() :
initialized_ (false),
	busy_(false)
{
	InitializeDefaultErrorMessages();

	// Custom error messages
	SetErrorText(ERR_NO_PORT_SET, "Hub Device not found. The MicroFPGA Hub device is needed to create this device");
	SetErrorText(ERR_COMMAND_UNKNOWN, "An unknown command was sent to the MicroFPGA.");

	// Description
	int ret = CreateProperty(MM::g_Keyword_Description, "MicroFPGA PWM controller", MM::String, true);
	assert(DEVICE_OK == ret);

	// Name
	ret = CreateProperty(MM::g_Keyword_Name, g_DeviceNamePWM, MM::String, true);
	assert(DEVICE_OK == ret);

	// Number of PWM channels
	CPropertyAction* pAct = new CPropertyAction(this, &PWM::OnNumberOfChannels);
	CreateProperty("Number of PWM", "1", MM::Integer, false, pAct, true);
	SetPropertyLimits("Number of PWM", 1, g_maxpwm);
}

PWM::~PWM()
{
	Shutdown();
}

void PWM::GetName(char* name) const
{
	CDeviceUtils::CopyLimitedString(name, g_DeviceNamePWM);
}


int PWM::Initialize()
{
	// Parent ID display	
	MicroFPGAHub* hub = static_cast<MicroFPGAHub*>(GetParentHub());
	if (!hub) {
		return ERR_NO_PORT_SET;
	}
	char hubLabel[MM::MaxStrLength];
	hub->GetLabel(hubLabel);
	SetParentID(hubLabel);
	CreateHubIDProperty();

	// State
	// -----

	// Allocate memory for channels
	state_ = new long [GetNumberOfChannels()];

	CPropertyActionEx *pExAct;
	int nRet;

	for(unsigned int i=0;i<GetNumberOfChannels();i++){
		state_[i] = 0;

		std::stringstream sstm;
		sstm << "Position" << i;

		pExAct = new CPropertyActionEx (this, &PWM::OnState,i);
		nRet = CreateProperty(sstm.str().c_str(), "0", MM::Integer, false, pExAct);
		if (nRet != DEVICE_OK)
			return nRet;
		SetPropertyLimits(sstm.str().c_str(), 0, 255);
	}

	nRet = UpdateStatus();
	if (nRet != DEVICE_OK)
		return nRet;

	initialized_ = true;

	return DEVICE_OK;
}

int PWM::Shutdown()
{
	initialized_ = false;
	return DEVICE_OK;
}

int PWM::WriteToPort(long address, long position)
{
	MicroFPGAHub* hub = static_cast<MicroFPGAHub*>(GetParentHub());
	if (!hub) {
		return ERR_NO_PORT_SET;
	}

	MMThreadGuard myLock(hub->GetLock());

	hub->PurgeComPortH();

	int ret = hub->SendWriteRequest(address, position);
	if (ret != DEVICE_OK)
		return ret;

	return DEVICE_OK;
}

int PWM::ReadFromPort(long& answer)
{
	MicroFPGAHub* hub = static_cast<MicroFPGAHub*>(GetParentHub());
	if (!hub) {
		return ERR_NO_PORT_SET;
	}
	int ret = hub->ReadAnswer(answer);
	if (ret != DEVICE_OK)
		return ret;

	return DEVICE_OK;
}

///////////////////////////////////////
/////////// Action handlers
int PWM::OnNumberOfChannels(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	if (pAct == MM::BeforeGet){
		pProp->Set(numChannels_);
	} else if (pAct == MM::AfterSet){
		pProp->Get(numChannels_);
	}
	return DEVICE_OK;
}

int PWM::OnState(MM::PropertyBase* pProp, MM::ActionType pAct, long channel)
{
	if (pAct == MM::BeforeGet)
	{
		MicroFPGAHub* hub = static_cast<MicroFPGAHub*>(GetParentHub());
		if (!hub){
			return ERR_NO_PORT_SET;
		}

		MMThreadGuard myLock(hub->GetLock());

		int ret = hub->SendReadRequest(g_offsetaddressPWM+channel);
		if (ret != DEVICE_OK)
			return ret;

		long answer;
		ret = ReadFromPort(answer);
		if (ret != DEVICE_OK)
			return ret;

		pProp->Set(answer);
		state_[channel]=answer;
	}
	else if (pAct == MM::AfterSet)
	{
		long pos;
		pProp->Get(pos);

		if(pos<0 || pos>255){
			pos = 0;
		}

		int ret = WriteToPort(g_offsetaddressPWM+channel,pos); 
		if (ret != DEVICE_OK)
			return ret;

		state_[channel] = pos;
	}

	return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////////////////
//////
AnalogInput::AnalogInput() :
initialized_ (false)
{
	InitializeDefaultErrorMessages();

	// Custom error messages
	SetErrorText(ERR_NO_PORT_SET, "Hub Device not found. The MicroFPGA Hub device is needed to create this device");
	SetErrorText(ERR_COMMAND_UNKNOWN, "An unknown command was sent to the MicroFPGA.");

	// Description
	int ret = CreateProperty(MM::g_Keyword_Description, "MicroFPGA AnalogInput", MM::String, true);
	assert(DEVICE_OK == ret);

	// Name
	ret = CreateProperty(MM::g_Keyword_Name, g_DeviceNameAnalogInput, MM::String, true);
	assert(DEVICE_OK == ret);

	// Number of Analog input channels
	CPropertyAction* pAct = new CPropertyAction(this, &AnalogInput::OnNumberOfChannels);
	CreateProperty("Number of channels", "3", MM::Integer, false, pAct, true);
	SetPropertyLimits("Number of channels", 1, g_maxanaloginput);
}

AnalogInput::~AnalogInput()
{
	Shutdown();
}

void AnalogInput::GetName(char* name) const
{
	CDeviceUtils::CopyLimitedString(name, g_DeviceNameAnalogInput);
}

bool AnalogInput::Busy()
{
	return false;
}

int AnalogInput::Initialize()
{
	// Parent ID display	
	MicroFPGAHub* hub = static_cast<MicroFPGAHub*>(GetParentHub());
	if (!hub) {
		return ERR_NO_PORT_SET;
	}
	char hubLabel[MM::MaxStrLength];
	hub->GetLabel(hubLabel);
	SetParentID(hubLabel);
	CreateHubIDProperty();

	// State
	// -----

	// Allocate memory for inputs
	state_ = new long [GetNumberOfChannels()];

	CPropertyActionEx *pExAct;
	int nRet;

	for(unsigned int i=0;i<GetNumberOfChannels();i++){
		state_[i] = 0;

		std::stringstream sstm;
		sstm << "AnalogInput" << i;

		pExAct = new CPropertyActionEx (this, &AnalogInput::OnAnalogInput,i);
		nRet = CreateProperty(sstm.str().c_str(), "0", MM::Integer, true, pExAct);
		if (nRet != DEVICE_OK)
			return nRet;
	}

	nRet = UpdateStatus();
	if (nRet != DEVICE_OK)
		return nRet;

	initialized_ = true;

	return DEVICE_OK;
}

int AnalogInput::Shutdown()
{
	initialized_ = false;
	return DEVICE_OK;
}

int AnalogInput::WriteToPort(long address)
{
	MicroFPGAHub* hub = static_cast<MicroFPGAHub*>(GetParentHub());
	if (!hub) {
		return ERR_NO_PORT_SET;
	}

	MMThreadGuard myLock(hub->GetLock());

	hub->PurgeComPortH();

	int ret = hub->SendReadRequest(address);
	if (ret != DEVICE_OK)
		return ret;

	return DEVICE_OK;
}

int AnalogInput::ReadFromPort(long& answer)
{
	MicroFPGAHub* hub = static_cast<MicroFPGAHub*>(GetParentHub());
	if (!hub) {
		return ERR_NO_PORT_SET;
	}
	int ret = hub->ReadAnswer(answer);
	if (ret != DEVICE_OK)
		return ret;

	return DEVICE_OK;
}

///////////////////////////////////////
/////////// Action handlers
int AnalogInput::OnNumberOfChannels(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	if (pAct == MM::BeforeGet){
		pProp->Set(numChannels_);
	} else if (pAct == MM::AfterSet){
		pProp->Get(numChannels_);
	}
	return DEVICE_OK;
}

int AnalogInput::OnAnalogInput(MM::PropertyBase* pProp, MM::ActionType pAct, long channel)
{
	if (pAct == MM::BeforeGet){
		MicroFPGAHub* hub = static_cast<MicroFPGAHub*>(GetParentHub());
		if (!hub){
			return ERR_NO_PORT_SET;
		}

		MMThreadGuard myLock(hub->GetLock());

		int ret = hub->SendReadRequest(g_offsetaddressAnalogInput+channel);
		if (ret != DEVICE_OK)
			return ret;

		long answer;
		ret = ReadFromPort(answer);
		if (ret != DEVICE_OK)
			return ret;


		pProp->Set(answer);
		state_[channel]=answer;
	}
	return DEVICE_OK;
}
