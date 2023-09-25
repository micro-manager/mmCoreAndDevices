#include "OxxiusCombinerHub.h"
#include "GenericLaser.h"
#include "OxxiusShutter.h"
#include "OxxiusMDual.h"
#include "OxxiusFlipMirror.h"
#include "OxxiusLBX.h"
#include "OxxiusLCX.h"
#include "CoherentObis.h"
#include "Cobolt08_01.h"

#include "ModuleInterface.h"

using namespace std;


// Oxxius devices
const char* g_OxxiusCombinerDeviceName = "Combiner";

const char* g_LCXLaserDeviceName = "LCX laser source";
const char* g_LCXLaser1DeviceName = "LCX laser source 1";
const char* g_LCXLaser2DeviceName = "LCX laser source 2";
const char* g_LCXLaser3DeviceName = "LCX laser source 3";
const char* g_LCXLaser4DeviceName = "LCX laser source 4";
const char* g_LCXLaser5DeviceName = "LCX laser source 5";
const char* g_LCXLaser6DeviceName = "LCX laser source 6";

const char* g_LBXLaserDeviceName = "LBX laser source";
const char* g_LBXLaser1DeviceName = "LBX laser source 1";
const char* g_LBXLaser2DeviceName = "LBX laser source 2";
const char* g_LBXLaser3DeviceName = "LBX laser source 3";
const char* g_LBXLaser4DeviceName = "LBX laser source 4";
const char* g_LBXLaser5DeviceName = "LBX laser source 5";
const char* g_LBXLaser6DeviceName = "LBX laser source 6";

const char* g_OBISLaserDeviceName = "Obis laser source";
const char* g_ObisLaser1DeviceName = "Obis laser source 1";
const char* g_ObisLaser2DeviceName = "Obis laser source 2";
const char* g_ObisLaser3DeviceName = "Obis laser source 3";
const char* g_ObisLaser4DeviceName = "Obis laser source 4";
const char* g_ObisLaser5DeviceName = "Obis laser source 5";
const char* g_ObisLaser6DeviceName = "Obis laser source 6";

const char* g_OxxiusShutterDeviceName = "Shutter";
const char* g_OxxiusShutter1DeviceName = "Shutter 1";
const char* g_OxxiusShutter2DeviceName = "Shutter 2";
const char* g_OxxiusMDualDeviceName = "MDual";
const char* g_OxxiusMDualADeviceName = "MDual A";
const char* g_OxxiusMDualBDeviceName = "MDual B";
const char* g_OxxiusMDualCDeviceName = "MDual C";
const char* g_OxxiusFlipMirrorDeviceName = "Flip-Mirror";
const char* g_OxxiusFlipMirror1DeviceName = "Flip-Mirror 1";
const char* g_OxxiusFlipMirror2DeviceName = "Flip-Mirror 2";




const char* g_slotPrefix[7] = { "","L1 ","L2 ","L3 ","L4 ","L5 ","L6 " };

const char* convertable[3] = { "A", "B", "C" };

///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////
MODULE_API void InitializeModuleData()
{
	RegisterDevice(g_OxxiusCombinerDeviceName, MM::HubDevice, "Oxxius laser combiner controlled through serial interface");

	RegisterDevice(g_LCXLaser1DeviceName, MM::ShutterDevice, "LCX Laser on slot 1");
	RegisterDevice(g_LCXLaser2DeviceName, MM::ShutterDevice, "LCX Laser on slot 2");
	RegisterDevice(g_LCXLaser3DeviceName, MM::ShutterDevice, "LCX Laser on slot 3");
	RegisterDevice(g_LCXLaser4DeviceName, MM::ShutterDevice, "LCX Laser on slot 4");
	RegisterDevice(g_LCXLaser5DeviceName, MM::ShutterDevice, "LCX Laser on slot 5");
	RegisterDevice(g_LCXLaser6DeviceName, MM::ShutterDevice, "LCX Laser on slot 6");

	RegisterDevice(g_LBXLaser1DeviceName, MM::ShutterDevice, "LBX Laser on slot 1");
	RegisterDevice(g_LBXLaser2DeviceName, MM::ShutterDevice, "LBX Laser on slot 2");
	RegisterDevice(g_LBXLaser3DeviceName, MM::ShutterDevice, "LBX Laser on slot 3");
	RegisterDevice(g_LBXLaser4DeviceName, MM::ShutterDevice, "LBX Laser on slot 4");
	RegisterDevice(g_LBXLaser5DeviceName, MM::ShutterDevice, "LBX Laser on slot 5");
	RegisterDevice(g_LBXLaser6DeviceName, MM::ShutterDevice, "LBX Laser on slot 6");

	RegisterDevice(g_ObisLaser1DeviceName, MM::ShutterDevice, "Obis Laser on slot 1");
	RegisterDevice(g_ObisLaser2DeviceName, MM::ShutterDevice, "Obis Laser on slot 2");
	RegisterDevice(g_ObisLaser3DeviceName, MM::ShutterDevice, "Obis Laser on slot 3");
	RegisterDevice(g_ObisLaser4DeviceName, MM::ShutterDevice, "Obis Laser on slot 4");
	RegisterDevice(g_ObisLaser5DeviceName, MM::ShutterDevice, "Obis Laser on slot 5");
	RegisterDevice(g_ObisLaser6DeviceName, MM::ShutterDevice, "Obis Laser on slot 6");

	RegisterDevice(g_OxxiusShutter1DeviceName, MM::ShutterDevice, "E-m shutter on channel 1");
	RegisterDevice(g_OxxiusShutter2DeviceName, MM::ShutterDevice, "E-m shutter on channel 2");
	RegisterDevice(g_OxxiusMDualADeviceName, MM::GenericDevice, "M-Dual on channel A");
	RegisterDevice(g_OxxiusMDualBDeviceName, MM::GenericDevice, "M-Dual on channel B");
	RegisterDevice(g_OxxiusMDualCDeviceName, MM::GenericDevice, "M-Dual on channel C");
	RegisterDevice(g_OxxiusFlipMirror1DeviceName, MM::StateDevice, "Flip-Mirror on slot 1");
	RegisterDevice(g_OxxiusFlipMirror2DeviceName, MM::StateDevice, "Flip-Mirror on slot 2");

}

MODULE_API MM::Device* CreateDevice(const char* deviceNameChar)
{
	if (deviceNameChar == 0)
		return 0;

	std::string deviceNameAndSlot = string(deviceNameChar);

	if (strcmp(deviceNameChar, g_OxxiusCombinerDeviceName) == 0) {
		return new OxxiusCombinerHub();
	}
	else if (deviceNameAndSlot.compare(0, strlen(g_LCXLaserDeviceName), g_LCXLaserDeviceName) == 0) {
		return new OxxiusLCX(deviceNameChar);
	}
	else if (deviceNameAndSlot.compare(0, strlen(g_LBXLaserDeviceName), g_LBXLaserDeviceName) == 0) {
		return new OxxiusLBX(deviceNameChar);
	}
	else if (deviceNameAndSlot.compare(0, strlen(g_OxxiusShutterDeviceName), g_OxxiusShutterDeviceName) == 0) {
		return new OxxiusShutter(deviceNameChar);
	}
	else if (deviceNameAndSlot.compare(0, strlen(g_OxxiusMDualDeviceName), g_OxxiusMDualDeviceName) == 0) {
		return new OxxiusMDual(deviceNameChar);
	}
	else if (deviceNameAndSlot.compare(0, strlen(g_OxxiusFlipMirrorDeviceName), g_OxxiusFlipMirrorDeviceName) == 0) {
		return new OxxiusFlipMirror(deviceNameChar);
	}
	else if (deviceNameAndSlot.compare(0, strlen(g_OBISLaserDeviceName), g_OBISLaserDeviceName) == 0) {
		return new CoherentObis(deviceNameChar);
	}
	return 0;
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
	delete pDevice;
}

///////////////////////////////////////////////////////////////////////////////
//
// Oxxius combiner implementation
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
///////////////////////////////////////////////////////////////////////////////


OxxiusCombinerHub::OxxiusCombinerHub() : initialized_(false)
{
	// Initializing private variables
	serialNumber_ = "";
	installedDevices_ = 0;
	serialAnswer_ = "";
	interlockClosed_ = false;
	keyActivated_ = false;
	maxtemperature_ = 20.0;

	InitializeDefaultErrorMessages();
	SetErrorText(ERR_COMBINER_NOT_FOUND, "Hub Device not found.  The peer device is expected to be a Oxxius combiner");


	// Create pre-initialization properties
	// ------------------------------------

	// Communication port
	CPropertyAction* pAct = new CPropertyAction(this, &OxxiusCombinerHub::OnPort);
	CreateProperty(MM::g_Keyword_Port, "Undefined", MM::String, false, pAct, true);
}

OxxiusCombinerHub::~OxxiusCombinerHub()
{
	Shutdown();
}

void OxxiusCombinerHub::GetName(char* name) const
{
	CDeviceUtils::CopyLimitedString(name, g_OxxiusCombinerDeviceName);
}

bool OxxiusCombinerHub::Busy()
{
	return false;

}


int OxxiusCombinerHub::Initialize()
{
	if (!initialized_) {

		// Set proprety list
		 // - - - - - - - - -

		 // Name and description of the combiner:
		RETURN_ON_MM_ERROR(CreateProperty(MM::g_Keyword_Name, g_OxxiusCombinerDeviceName, MM::String, true));
		RETURN_ON_MM_ERROR(CreateProperty(MM::g_Keyword_Description, "Oxxius L6Cc/L4Cc combiner", MM::String, true));

		// Serial number of the combiner:
		CPropertyAction* pAct = new CPropertyAction(this, &OxxiusCombinerHub::OnSerialNumber);
		RETURN_ON_MM_ERROR(CreateProperty(MM::g_Keyword_HubID, serialNumber_.c_str(), MM::String, true, pAct));

		// Interlock circuit:
		pAct = new CPropertyAction(this, &OxxiusCombinerHub::OnInterlock);
		RETURN_ON_MM_ERROR(CreateProperty("Interlock circuit", "", MM::String, true, pAct));
		 
		// Emission key:
		pAct = new CPropertyAction(this, &OxxiusCombinerHub::OnEmissionKey);
		RETURN_ON_MM_ERROR(CreateProperty("EmissionKey", "", MM::String, true, pAct));

		// Temperature:
		pAct = new CPropertyAction(this, &OxxiusCombinerHub::OnTemperature);
		RETURN_ON_MM_ERROR(CreateProperty("MaxTemperature", "", MM::Float, true, pAct));


		if (!IsCallbackRegistered())
			return DEVICE_NO_CALLBACK_REGISTERED;

		// Enumerates the installed AOMs and their position
		bool AOM1en = false, AOM2en = false;
		unsigned int ver = 0;

		RETURN_ON_MM_ERROR(QueryCommand(this, GetCoreCallback(), NO_SLOT, "AOM1 EN", false));
		ParseforBoolean(AOM1en);

		RETURN_ON_MM_ERROR(QueryCommand(this, GetCoreCallback(), NO_SLOT, "AOM2 EN", false));
		ParseforBoolean(AOM2en);

		RETURN_ON_MM_ERROR(QueryCommand(this, GetCoreCallback(), NO_SLOT, "SV?", false));
		ParseforVersion(ver);

		// A position equal to "0" stands for an absence of modulator
		if (AOM1en) {
			bool adcom = false;
			string command = "";

			if (ver < 1016) { //version check 
				adcom = true;
				command = "AOM1 PO";
			}
			else {
				adcom = false;
				command = "AOM1 POS";
			}
			RETURN_ON_MM_ERROR(QueryCommand(this, GetCoreCallback(), NO_SLOT, command.c_str(), adcom));
			ParseforInteger(AOM1pos_);
		}
		if (AOM2en) {
			bool adcom = false;
			string command = "";

			if (ver < 1016) { //version check 
				adcom = true;
				command = "AOM2 PO";
			}
			else {
				adcom = false;
				command = "AOM2 POS";
			}
			RETURN_ON_MM_ERROR(QueryCommand(this, GetCoreCallback(), NO_SLOT, command.c_str(), adcom));
			ParseforInteger(AOM1pos_);
		}


		//Mpa position retreive
		for (unsigned int i = 1; i <= MAX_NUMBER_OF_SLOTS; i++) {
			string command = "IP";
			std::stringstream ss;
			ss << i;
			command += ss.str();

			RETURN_ON_MM_ERROR(QueryCommand(this, GetCoreCallback(), NO_SLOT, command.c_str(), true));
			if (serialAnswer_ != "????") {
				mpa[i] = 1;
			}
		}



		RETURN_ON_MM_ERROR(UpdateStatus());

		initialized_ = true;

		// RETURN_ON_MM_ERROR( DetectInstalledDevices() );
	}
	return DEVICE_OK;
}



int OxxiusCombinerHub::DetectInstalledDevices()
{
	if (initialized_) {

		// Enumerates the lasers (or devices) present on the combiner
		unsigned int masque = 1;
		unsigned int repartition = 0;

		//sending command ?CL
		RETURN_ON_MM_ERROR(QueryCommand(this, GetCoreCallback(), NO_SLOT, "?CL", false));
		ParseforInteger(repartition);

		for (unsigned int querySlot = 1; querySlot <= MAX_NUMBER_OF_SLOTS; querySlot++) {
			if ((repartition & masque) != 0) {
				string answer;
				// A laser source is listed, now querying for detailed information (model, etc)

				std::string detailedInfo, serialNumber;

				//send command to get devices information
				RETURN_ON_MM_ERROR(QueryCommand(this, GetCoreCallback(), querySlot, "INF?", false));
				ParseforString(detailedInfo);

				if (detailedInfo != "timeout") {
					std::ostringstream nameSlotModel;

					//get the model (LCX or LBX or third-party)
					std::string model = detailedInfo.substr(0, 3);
					
					if (model == "LBX" || model == "LSX") {
						nameSlotModel << g_LBXLaserDeviceName << " " << querySlot;
					}
					else if (model == "LCX" || model == "LPX") {
						nameSlotModel << g_LCXLaserDeviceName << " " << querySlot;
					}
					else if (detailedInfo == "ERR-100") {
						//Dans le OBIS
						nameSlotModel << g_OBISLaserDeviceName << " " << querySlot;
					}

					MM::Device* pDev = ::CreateDevice(nameSlotModel.str().c_str());
					if (pDev) {
						AddInstalledDevice(pDev);
						installedDevices_++;
					}
				}
			}
			masque <<= 1;		// Left-shift the bit mask and repeat
		}

		// Creating Devices for the two electro-mechanical shutters:
		for (unsigned int channel = 1; channel <= 2; channel++) {
			std::ostringstream nameModelChannel;
			nameModelChannel << g_OxxiusShutterDeviceName << " " << channel;

			MM::Device* pDev = ::CreateDevice(nameModelChannel.str().c_str());
			if (pDev) {
				AddInstalledDevice(pDev);
				installedDevices_++;
			}
		}

		// Creating Devices for the "Flip mirror" or MDUAL modules:
		/*unsigned int FM1type = 0, FM2type = 0;
		RETURN_ON_MM_ERROR(QueryCommand(this, GetCoreCallback(), NO_SLOT, "FM1C", false));
		ParseforInteger(FM1type);
		RETURN_ON_MM_ERROR(QueryCommand(this, GetCoreCallback(), NO_SLOT, "FM2C", false));
		ParseforInteger(FM2type);*/


		// MDUAL module detection
		for (unsigned int j = 0; j <= 2; j++) {
			std::string MDSlot;
			std::ostringstream com;
			com << "IP" << convertable[j];

			RETURN_ON_MM_ERROR(QueryCommand(this, GetCoreCallback(), NO_SLOT, com.str().c_str(), true));
			ParseforString(MDSlot);

			com.str("");
			com.clear();
			com << g_OxxiusMDualDeviceName << " " << convertable[j];

			if (MDSlot != "????") {
				MM::Device* pDev = ::CreateDevice(com.str().c_str());
				if (pDev) {
					AddInstalledDevice(pDev);
					installedDevices_++;
				}

			}

		}

		//Flip mirror module creation 

		for (unsigned int j = 1; j <= 2; j++) {
			unsigned int FMpresence;
			std::ostringstream queryPhrase,FMName ;
			queryPhrase << "FM" << j << "C";

			RETURN_ON_MM_ERROR(QueryCommand(this, GetCoreCallback(), NO_SLOT, queryPhrase.str().c_str(), false));
			ParseforInteger(FMpresence);

			FMName << g_OxxiusFlipMirrorDeviceName << " " << j;
			if ( (FMpresence == 1) || (FMpresence == 2) || (FMpresence == 3) ) {
				MM::Device* pDev = ::CreateDevice(FMName.str().c_str());
				if (pDev) {
					AddInstalledDevice(pDev);
					installedDevices_++;
				}
			}
		}

	} // if (initialized_)

	return DEVICE_OK;
}


int OxxiusCombinerHub::Shutdown()
{
	initialized_ = false;
	return DEVICE_OK;
}


///////////////////////////////////////////////////////////////////////////////
// Action handlers
///////////////////////////////////////////////////////////////////////////////

int OxxiusCombinerHub::OnPort(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(port_.c_str());
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(port_);
		pProp->Set(port_.c_str());
	}
	return DEVICE_OK;
}


int OxxiusCombinerHub::OnSerialNumber(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	if (pAct == MM::BeforeGet) {
		QueryCommand(this, GetCoreCallback(), NO_SLOT, "HID?", false);
		ParseforString(serialNumber_);
		pProp->Set(serialNumber_.c_str());
	}

	return DEVICE_OK;
}


int OxxiusCombinerHub::OnInterlock(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	if (pAct == MM::BeforeGet) {
		QueryCommand(this, GetCoreCallback(), NO_SLOT, "INT?", false);
		ParseforBoolean(interlockClosed_);

		if (interlockClosed_) {
			pProp->Set("Closed");
		}
		else {
			pProp->Set("Open");
		}
	}

	return DEVICE_OK;
}


int OxxiusCombinerHub::OnEmissionKey(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	if (pAct == MM::BeforeGet) {
		QueryCommand(this, GetCoreCallback(), NO_SLOT, "KEY?", false);
		ParseforBoolean(keyActivated_);

		if (keyActivated_) {
			pProp->Set("Armed");
		}
		else {
			pProp->Set("Disarmed");
		}
	}

	return DEVICE_OK;
}


int OxxiusCombinerHub::OnTemperature(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	if (pAct == MM::BeforeGet) {
		// create an array of the temperatures of all the lasers
		std::vector<float> temperature_array;
		float temp_var;

		for (int i = 1; i <= MAX_NUMBER_OF_SLOTS; i++) {
			
			if (i != GetObPos()) {
				QueryCommand(this, GetCoreCallback(), i, "?BT", false); //for LBX and LCX lasers
				ParseforFloat(temp_var);
			}
			else {
				QueryCommand(this, GetCoreCallback(), i, "SOUR:TEMP:BAS?", false); // For OBIS laser
				ParseforTemperature(temp_var);
			}
			temperature_array.push_back(temp_var);
		}

		float maxTemp = temperature_array[0];
		for (auto i : temperature_array) {
			if (i > maxTemp) {
				maxTemp = i;
			}
		}
		maxtemperature_ = maxTemp;

		pProp->Set(static_cast<double>(maxtemperature_));

	}
	return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// Generic methods
///////////////////////////////////////////////////////////////////////////////

void OxxiusCombinerHub::LogError(int id, MM::Device* device, MM::Core* core, const char* functionName) //print log messages
{
	std::ostringstream os;
	char deviceName[MM::MaxStrLength];
	device->GetName(deviceName);
	os << "Error " << id << ", " << deviceName << ", " << functionName << endl;
	core->LogMessage(device, os.str().c_str(), false);
}


/**
 * Sends a serial command to a given slot, then stores the result in the receive buffer.
 */
int OxxiusCombinerHub::QueryCommand(MM::Device* device, MM::Core* core, const unsigned int destinationSlot, const char* command, bool adco)
{
	// First check: if the command string is empty, do nothing and return "DEVICE_OK"
	if (strcmp(command, "") == 0) return DEVICE_OK;

	char rcvBuf_[RCV_BUF_LENGTH];
	// Compose the command to be sent to the combiner
	std::string strCommand, strHZIn, strHZOut;
	strCommand.assign(g_slotPrefix[destinationSlot]);
	strCommand.append(command);
	strHZIn.assign(g_slotPrefix[destinationSlot]);
	strHZIn.append("HZ 9876");
	strHZOut.assign(g_slotPrefix[destinationSlot]);
	strHZOut.append("HZ 0");

	/*
		std::ostringstream InfoMessage;
		InfoMessage << "Now sending command :";
		InfoMessage << string(strCommand.c_str());
		LogError(DEVICE_OK, device, core, InfoMessage.str().c_str());
	*/
	std::ostringstream InfoMessage2;
	InfoMessage2 << "Send: " << command << " Received: ";

	// Preambule for specific commands
	if (adco) {
		int ret = core->SetSerialCommand(device, port_.c_str(), strHZIn.c_str(), "\r\n");
		ret = core->GetSerialAnswer(device, port_.c_str(), RCV_BUF_LENGTH, rcvBuf_, "\r\n");
		if (ret != DEVICE_OK) {
			LogError(ret, device, core, "QueryCommand-SetSerialCommand - preambule");
			return ret;
		}
	}

	// Send command through the serial interface
	int ret = core->SetSerialCommand(device, port_.c_str(), strCommand.c_str(), "\r\n");
	if (ret != DEVICE_OK) {
		LogError(ret, device, core, "QueryCommand-SetSerialCommand");
		return ret;
	}

	// Get a response
	ret = core->GetSerialAnswer(device, port_.c_str(), RCV_BUF_LENGTH, rcvBuf_, "\r\n");

	InfoMessage2 << rcvBuf_;
	/* DEBUG ONLY */
	// LogError(DEVICE_OK, device, core, InfoMessage2.str().c_str());

	if (ret != DEVICE_OK) {
		LogError(ret, device, core, "QueryCommand-GetSerialAnswer");

		// Keep on trying until we either get our answer, or 3 seconds have passed
		int maxTimeMs = 3000;
		// Wait for a (increasing) delay between each try
		int delayMs = 10;
		// Keep track of how often we tried
		int counter = 1;
		bool done = false;
		MM::MMTime startTime(core->GetCurrentMMTime()); // Let's keep in mind that MMTime is counted in microseconds

		while (!done) {
			counter++;
			ret = core->GetSerialAnswer(device, port_.c_str(), RCV_BUF_LENGTH, rcvBuf_, "\r\n");
			if ((ret == DEVICE_OK) || ((core->GetCurrentMMTime() - startTime) > (maxTimeMs * 1000.0)))
				done = true;
			else {
				CDeviceUtils::SleepMs(delayMs);
				delayMs *= 2;
			}
		}
		ostringstream os;
		if (ret == DEVICE_OK)
			os << "QueryCommand-GetSerialAnswer: Succeeded reading from serial port after trying " << counter << "times.";
		else
			os << "QueryCommand-GetSerialAnswer: Failed reading from serial port after trying " << counter << "times.";

		core->LogMessage(device, os.str().c_str(), true);

		serialAnswer_.assign(rcvBuf_);
		return ret;
	}
	serialAnswer_.assign(rcvBuf_);
	ret = core->PurgeSerial(device, port_.c_str());

	/*	if( strcmp(serialAnswer_, "timeout") == 0)	{
			std::ostringstream syntaxErrorMessage;
			syntaxErrorMessage << "Time out received against sent command '";
			syntaxErrorMessage << string(strCommand.c_str());
			syntaxErrorMessage << "'";

			LogError(DEVICE_SERIAL_TIMEOUT, device, core, syntaxErrorMessage.str().c_str());
			return DEVICE_SERIAL_TIMEOUT;
		}
	*/
	// Epilogue for specific commands
	if (adco) {
		int retEpi = core->SetSerialCommand(device, port_.c_str(), strHZOut.c_str(), "\r\n");
		retEpi = core->GetSerialAnswer(device, port_.c_str(), RCV_BUF_LENGTH, rcvBuf_, "\r\n");
		if (retEpi != DEVICE_OK) {
			LogError(retEpi, device, core, "QueryCommand-SetSerialCommand - Epilogue");
			return retEpi;
		}
	}

	return DEVICE_OK;
}


int OxxiusCombinerHub::ParseforBoolean(bool& Bval)
{
	unsigned int intAnswer = (unsigned int)atoi(serialAnswer_.c_str());
	Bval = (intAnswer == 1);

	serialAnswer_.clear();

	return DEVICE_OK;
}


int OxxiusCombinerHub::ParseforFloat(float& Dval)
{
	Dval = (float)atof(serialAnswer_.c_str());

	serialAnswer_.clear();

	return DEVICE_OK;
}

int OxxiusCombinerHub::ParseforDouble(double& Dval)
{
	Dval = (double)atof(serialAnswer_.c_str());

	serialAnswer_.clear();

	return DEVICE_OK;
}



int OxxiusCombinerHub::ParseforInteger(unsigned int& Ival)
{
	Ival = (unsigned int)atoi(serialAnswer_.c_str());

	serialAnswer_.clear();

	return DEVICE_OK;
}


int OxxiusCombinerHub::ParseforString(std::string& Sval)
{
	Sval.assign(serialAnswer_);

	serialAnswer_.clear();

	return DEVICE_OK;
}


int OxxiusCombinerHub::ParseforVersion(unsigned int& Vval) //cast the string into a comparable int
{
	std::string temp1;
	std::string temp2(serialAnswer_);

	for (unsigned int i = 0; i <= (temp2.length()) - 1; i++) {
		if (temp2.at(i) != '.') {
			temp1 += temp2.at(i);
		}
	}

	stringstream s(temp1);
	s >> Vval;
	serialAnswer_.clear();

	return DEVICE_OK;
}


int OxxiusCombinerHub::ParseforPercent(double& Pval) //cast the string into a comparable int
{
	std::string percentage;
	std::size_t found;

	percentage.assign(serialAnswer_);

	found = percentage.find("%");
	if (found != std::string::npos) {
		Pval = atof(percentage.substr(0, found).c_str());
	}

	return DEVICE_OK;
}


int OxxiusCombinerHub::ParseforTemperature(float& Tval) // cast a Celsius temperature string into a comparable float
{
	std::string temperature;
	std::size_t found;

	temperature.assign(serialAnswer_);

	found = temperature.find("C");
	if (found != std::string::npos) {
		Tval = atof(temperature.substr(0, found).c_str());
	}

	return DEVICE_OK;
}

int OxxiusCombinerHub::ParseforChar(char* Nval)
{
	strcpy(Nval, serialAnswer_.c_str());
	serialAnswer_.clear();

	return DEVICE_OK;
}

bool OxxiusCombinerHub::GetAOMpos1(unsigned int slot)
{
	bool res = false;

	if (slot == AOM1pos_) {
		res = true;
	}

	return res;
}

bool OxxiusCombinerHub::GetAOMpos2(unsigned int slot)
{
	bool res = false;

	if (slot == AOM2pos_) {
		res = true;
	}

	return res;
}

bool OxxiusCombinerHub::GetMPA(unsigned int slot) {
	bool res = false;

	if (mpa[slot] == 1) {
		res = true;
	}

	return res;
}

int OxxiusCombinerHub::GetObPos() {
	return obPos_;
}
