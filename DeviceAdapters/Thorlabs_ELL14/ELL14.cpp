///////////////////////////////////////////////////////////////////////////////
//// FILE:    	ELL14.cpp
//// PROJECT:	MicroManage
//// SUBSYSTEM:  DeviceAdapters
////-----------------------------------------------------------------------------
//// DESCRIPTION: Implementation of the Thorlabs' Elliptec Rotation Mount
//// https://www.thorlabs.de/newgrouppage9.cfm?objectgroup_ID=12829
////           	 
//// AUTHOR: Manon Paillat, 2022
//// developped under the supervision of Florian Ströhl
//// Contact: florian.strohl@uit.no
//

#ifdef WIN32
#include <windows.h>
#endif

#include "ELL14.h"
#include <string>
#include "../MMDevice/ModuleInterface.h"
#include <sstream>
//#include <thread>

//  ELL14 device
const char* g_ELL14Name = " ELL14";
///////////////////////////////////////////////////////////////////////////////
using namespace std;

///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////

MODULE_API void InitializeModuleData()
{
	RegisterDevice(g_ELL14Name, MM::GenericDevice, g_ELL14Name);
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
	if (deviceName == 0)
		return 0;

	if (strcmp(deviceName, g_ELL14Name) == 0) {
		return new  ELL14();
	}

	return 0;
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
	delete pDevice;
}


///////////////////////////////////////////////////////////////////////////////
// Thorlabs Rotation Mount device adapter
//
ELL14::ELL14() :
	initialized_(false),
	port_("Undefined"),
	channel_("0"),
	pos_(0.0),
	offset_(0.0),
	homeDir_(rotDirection::FW),
	jogStep_(90),
	jogDir_(rotDirection::FW),
	relativeMove_(90.0),
	pulsesPerRev_(1),
	maxReplyTimeMs_(1200)
{

	InitializeDefaultErrorMessages();

	SetErrorText(ERR_COMMUNICATION_TIME_OUT, "Communication time-out. Is the channel set correctly?");
	SetErrorText(ERR_MECHANICAL_TIME_OUT, "Mechanical time-out.");
	SetErrorText(ERR_COMMAND_ERROR_OR_NOT_SUPPORTED, "Unsupported or unknown command.");
	SetErrorText(ERR_VALUE_OUT_OF_RANGE, "Value out of range.");
	SetErrorText(ERR_MODULE_ISOLATED, "Module isolated.");
	SetErrorText(ERR_MODULE_OUT_OF_ISOLATION, "Module out of isolation.");
	SetErrorText(ERR_INITIALIZING_ERROR, "Initializing error.");
	SetErrorText(ERR_THERMAL_ERROR, "Thermal error.");
	SetErrorText(ERR_BUSY, "Busy.");
	SetErrorText(ERR_SENSOR_ERROR, "Sensor error.");
	SetErrorText(ERR_MOTOR_ERROR, "Motor error.");
	SetErrorText(ERR_OUT_OF_RANGE, "Out of range.");
	SetErrorText(ERR_OVER_CURRENT_ERROR, "Over-current error.");
	SetErrorText(ERR_UNKNOWN_ERROR, "Unknown error (error code >13).");

	// create pre-initialization properties
	// ------------------------------------

	// Description                                                       	 
	CreateProperty(MM::g_Keyword_Description, "Thorlab's ELL14 Rotation Mount", MM::String, true);

	// Port                                                              	 
	CPropertyAction* pAct = new CPropertyAction(this, &ELL14::OnPort);
	CreateProperty(MM::g_Keyword_Port, "Undefined", MM::String, false, pAct, true);

	// Channel
	string channels[] = { "0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F" };
	vector<string> channels_vec;
	for (int i = 0; i < 16; i++) {
		channels_vec.push_back(channels[i]); // Adds a new element at the end of the vector
	}
	SetAllowedValues("Channel", channels_vec);

	pAct = new CPropertyAction(this, &ELL14::OnChannel);
	CreateProperty("Channel", "0", MM::String, false, pAct, true); // Default adress : 0
}

ELL14::~ELL14()
{
	Shutdown();
}

///////////////////////////////////////////////////////////////////////////////
// Action handlers
///////////////////////////////////////////////////////////////////////////////

int  ELL14::OnPort(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet) {
		pProp->Set(port_.c_str());
	}
	else if (eAct == MM::AfterSet) {
		if (initialized_) {
			// revert
			pProp->Set(port_.c_str());
			return ERR_PORT_CHANGE_FORBIDDEN;
		}

		pProp->Get(port_);
	}

	return DEVICE_OK;
}

int  ELL14::OnChannel(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::AfterSet)
	{
		string channel;
		pProp->Get(channel);

		channel_ = channel;
	}

	return DEVICE_OK;
}

int  ELL14::OnPosition(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet) {
		stringstream ss;
		ss << pos_;
		pProp->Set(ss.str().c_str());
	}
	else if (eAct == MM::AfterSet) {
		long pos;
		pProp->Get(pos);
		pos = modulo2pi(pos);
		int ret = SetPosition(pos);
		if (ret != DEVICE_OK)
			return ret;
	}
	return DEVICE_OK;
}

int  ELL14::OnHomeValue(MM::PropertyBase* pProp, MM::ActionType eAct) {
	if (eAct == MM::AfterSet) {
		long offset;
		pProp->Get(offset);
		int ret = SetOffset(offset);
		if (ret != DEVICE_OK)
			return ret;
	}

	return DEVICE_OK;
}

int  ELL14::OnHome(MM::PropertyBase* pProp, MM::ActionType eAct) {
	if (eAct == MM::AfterSet) {
		string stringDir;
		pProp->Get(stringDir);
		if (stringDir.compare("Clockwise") == 0)
			homeDir_ = rotDirection::FW;
		else
			homeDir_ = rotDirection::BW;

		int ret = Home();
		if (ret != DEVICE_OK)
			return ret;
	}
	return DEVICE_OK;
}

int  ELL14::OnRelativeMove(MM::PropertyBase* pProp, MM::ActionType eAct) {
	if (eAct == MM::AfterSet) {
		double relativeMove;
		pProp->Get(relativeMove);
		int ret = SetRelativePosition(relativeMove);
		if (ret != DEVICE_OK)
			return ret;
	}
	return DEVICE_OK;
}

int  ELL14::OnJogStep(MM::PropertyBase* pProp, MM::ActionType eAct) {
	if (eAct == MM::AfterSet) {
		double jogStep;
		pProp->Get(jogStep);
		int ret = SetJogStep(jogStep);
		if (ret != DEVICE_OK)
			return ret;
	}
	return DEVICE_OK;
}

int  ELL14::OnJog(MM::PropertyBase* pProp, MM::ActionType eAct) {
	if (eAct == MM::AfterSet) {
		string stringDir;
		pProp->Get(stringDir);
		if (stringDir.compare("Clockwise") == 0)
			jogDir_ = rotDirection::FW;
		else
			jogDir_ = rotDirection::BW;
		int ret = Jog();
		if (ret != DEVICE_OK)
			return ret;
	}
	return DEVICE_OK;
}

int  ELL14::OnSearchFrequencies(MM::PropertyBase* pProp, MM::ActionType eAct) {
	if (eAct == MM::BeforeGet) {
		pProp->Set("No Action");
	}
	else if (eAct == MM::AfterSet) {
		string stringDir;
		pProp->Get(stringDir);
		if (stringDir.compare("Launch Research") == 0) {
			int ret = SearchFrequencies();
			if (ret != DEVICE_OK)
				return ret;
		}
	}
	return DEVICE_OK;
}

////////////////////////////////////////
int  ELL14::Initialize()
{
	if (initialized_)
		return DEVICE_OK;

	// set property list
	// -----------------

	// ID
	string id;
	getID(&id, &pulsesPerRev_);
	int nRet = CreateProperty("ID", id.c_str(), MM::String, true);
	if (nRet != DEVICE_OK)
		return nRet;

	// Name
	int ret;
	ret = CreateStringProperty(MM::g_Keyword_Name, g_ELL14Name, true);
	if (DEVICE_OK != ret)
		return ret;

	// Position
	// --------
	ret = GetPosition(pos_);
	if (ret != DEVICE_OK)
		return ret;
	CPropertyAction* pAct = new CPropertyAction(this, &ELL14::OnPosition);
	ret = CreateFloatProperty(MM::g_Keyword_Position, pos_, false, pAct);
	if (ret != DEVICE_OK)
		return ret;
	SetPropertyLimits("Position", 0, 359.99);

	// Offset value
	// -----------------
	ret = GetOffset(offset_);
	if (ret != DEVICE_OK)
		return ret;
	pAct = new CPropertyAction(this, &ELL14::OnHomeValue);
	ret = CreateFloatProperty(MM::g_Keyword_Offset, offset_, false, pAct);
	if (ret != DEVICE_OK)
		return ret;
	SetPropertyLimits("Home offset", 0, 359.99);

	// Homing
	// ------
	pAct = new CPropertyAction(this, &ELL14::OnHome);
	const char* homeDir = "Clockwise";
	ret = CreateStringProperty("Home", homeDir, false, pAct);
	if (ret != DEVICE_OK)
		return ret;
	AddAllowedValue("Home", "Clockwise");
	AddAllowedValue("Home", "Counterclockwise");

	// Relative Move
	// -------------
	pAct = new CPropertyAction(this, &ELL14::OnRelativeMove);
	ret = CreateFloatProperty("Relative Move", relativeMove_, false, pAct);
	if (ret != DEVICE_OK)
		return ret;
	SetPropertyLimits("Relative Move", -359.99, 359.99);

	// Jog Step
	// --------
	ret = GetJogStep(jogStep_);
	if (ret != DEVICE_OK)
		return ret;
	pAct = new CPropertyAction(this, &ELL14::OnJogStep);
	ret = CreateFloatProperty("Jog Step", jogStep_, false, pAct);
	if (ret != DEVICE_OK)
		return ret;
	SetPropertyLimits("Jog Step", 0, 359.99);

	// Jogging
	// -------
	pAct = new CPropertyAction(this, &ELL14::OnJog);
	ret = CreateStringProperty("Jog", "Clockwise", false, pAct);
	if (ret != DEVICE_OK)
		return ret;
	AddAllowedValue("Jog", "Clockwise");
	AddAllowedValue("Jog", "Counterclockwise");

	// Search best frequencies
	// -----------------------
	pAct = new CPropertyAction(this, &ELL14::OnSearchFrequencies);
	ret = CreateStringProperty("Frequencies Optimization", "No Action", false, pAct);
	if (ret != DEVICE_OK)
		return ret;
	AddAllowedValue("Frequencies Optimization", "No Action");
	AddAllowedValue("Frequencies Optimization", "Launch Research");

	initialized_ = true;
	return DEVICE_OK;
}

int  ELL14::Shutdown()
{
	if (initialized_) {
		initialized_ = false;
	}
	return DEVICE_OK;
}

bool ELL14::Busy() {
	ostringstream command;
	command << channel_ << "gs";

	// send _HOSTREQ_STATUS request
	int ret = SendSerialCommand(port_.c_str(), command.str().c_str(), "\r");
	if (ret != DEVICE_OK)
		return true;

	// receive _DEVGET_STATUS
	string answer;
	ret = GetSerialAnswer(port_.c_str(), "\r", answer);
	if (ret != DEVICE_OK)
		return true;

	// remove "\n" if start character
	string message = removeLineFeed(answer);

	int code = getStatusCode(message); //message instead of answer

	// in case of error, we postulate that device is busy
	if (code == 0) { // code "0" corresponds to no-error
		return false;
	}

	return true;
}

void  ELL14::GetName(char* Name) const
{
	CDeviceUtils::CopyLimitedString(Name, g_ELL14Name);
}

//---------------------------------------------------------------------------
// Set data
//---------------------------------------------------------------------------

// Ask the device to move to a position pos (in degree) + wait for the reply
int  ELL14::SetPosition(double pos) {
	ostringstream command;

	pos = modulo2pi(pos);
	long val = (long)(pulsesPerRev_ / 360 * pos);
	command << channel_ << "ma" << positionFromValue(val);

	int ret = SendSerialCommand(port_.c_str(), command.str().c_str(), "\r");
	if (ret != DEVICE_OK)
		return ret;

	double angle = abs(pos_ - pos);
	return ret = receivePosition(angle);
}

// Ask the device to move of a certain angle (in degree) + wait for the repaly
// The value can be negative for a counterclowise motion
int  ELL14::SetRelativePosition(double relativeMove) {
	relativeMove_ = relativeMove;

	ostringstream command;
	long val = (long)(pulsesPerRev_ / 360 * relativeMove);
	command << channel_ << "mr" << positionFromValue(val);
	int ret = SendSerialCommand(port_.c_str(), command.str().c_str(), "\r");
	if (ret != DEVICE_OK)
		return ret;

	return ret = receivePosition(abs(relativeMove));
}

// Provide the device the home position (in degree)
int  ELL14::SetOffset(double offset) {
	ostringstream command_set_value;
	long val = (long)(offset / 360 * pulsesPerRev_);
	command_set_value << channel_ << "so" << positionFromValue(val);
	int ret = SendSerialCommand(port_.c_str(), command_set_value.str().c_str(), "\r");
	if (ret != DEVICE_OK)
		return ret;

	string answer;
	ret = GetSerialAnswer(port_.c_str(), "\r", answer);
	if (ret != DEVICE_OK)
		return ret;

	string message = removeLineFeed(answer);
	if (isStatus(message)) {
		int status = getStatusCode(message);
		if (status == 0)
			return GetPosition(pos_);
		else
			return status;
	}
	else
		return ERR_UNEXPECTED_ANSWER;
}

// Provide the device the jog step (in degree) + wait for reply
int  ELL14::SetJogStep(double jogStep) {
	jogStep_ = jogStep;

	ostringstream command_set_value;
	long val = (long)(jogStep / 360 * pulsesPerRev_);
	command_set_value << channel_ << "sj" << positionFromValue(val);
	int ret = SendSerialCommand(port_.c_str(), command_set_value.str().c_str(), "\r");
	if (ret != DEVICE_OK)
		return ret;

	string answer;
	ret = GetSerialAnswer(port_.c_str(), "\r", answer);
	if (ret != DEVICE_OK)
		return ret;

	string message = removeLineFeed(answer);
	if (isStatus(message)) {
		return getStatusCode(message);
	}
	else
		return ERR_UNEXPECTED_ANSWER;
}

//---------------------------------------------------------------------------
// Get data
//---------------------------------------------------------------------------

// Ask the device to identify itself
// Analyze the response
int  ELL14::getID(string* id, double* pulsesPerRev) {
	ostringstream command;
	command << channel_ << "in";

	int ret = SendSerialCommand(port_.c_str(), command.str().c_str(), "\r");
	if (ret != DEVICE_OK)
		return ret;

	string answer;
	ret = GetSerialAnswer(port_.c_str(), "\r", answer);
	if (ret != DEVICE_OK)
		return ret;

	// remove "\n" if start character
	string message = removeLineFeed(answer);

	// check if returned an status reply
	if (isStatus(message))
		return getStatusCode(message);

	// check if it is the expected answer
	if (message.substr(1, 2).compare("IN") != 0)
		return ERR_UNEXPECTED_ANSWER;

	// check if ELL14 (in hex)
	if (message.substr(3, 2).compare("0E") != 0)
		return ERR_WRONG_DEVICE;

	*id = message.substr(3, 15); // module + serial + year + firmware
	*pulsesPerRev = positionFromHex(message.substr(25, 8));

	return DEVICE_OK;
}

// Ask the device for its position
// Analyze response and return the position in degree
int  ELL14::GetPosition(double& pos) {
	ostringstream command;
	command << channel_ << "gp";

	int ret = SendSerialCommand(port_.c_str(), command.str().c_str(), "\r");
	if (ret != DEVICE_OK)
		return ret;

	string answer;
	ret = GetSerialAnswer(port_.c_str(), "\r", answer);
	if (ret != DEVICE_OK)
		return ret;

	string message = removeLineFeed(answer);
	if (isStatus(message)) {
		return getStatusCode(message);
	}

	if (message.substr(1, 2).compare("PO") != 0)
		return ERR_UNEXPECTED_ANSWER;

	else {
		string val = message.substr(3, message.length() - 3);
		pos = (double)(positionFromHex(val) * 360.0 / pulsesPerRev_);
		return DEVICE_OK;
	}
}

// Ask the device for the registered offset
// Analyze response and return the offset in degree
int  ELL14::GetOffset(double& offset) {
	ostringstream command;
	command << channel_ << "go";

	int ret = SendSerialCommand(port_.c_str(), command.str().c_str(), "\r");
	if (ret != DEVICE_OK)
		return ret;

	string answer;
	ret = GetSerialAnswer(port_.c_str(), "\r", answer);
	if (ret != DEVICE_OK)
		return ret;

	string message = removeLineFeed(answer);
	if (isStatus(message)) {
		return getStatusCode(message);
	}

	if (message.substr(1, 2).compare("HO") != 0)
		return ERR_UNEXPECTED_ANSWER;

	else {
		string val = message.substr(3, message.length() - 3);
		offset = positionFromHex(val) * 360.0 / pulsesPerRev_;
		return DEVICE_OK;
	}
}

// Ask the device for the registered jog step
// Analyze response and return the value in degree
int  ELL14::GetJogStep(double& jogStep) {
	ostringstream command;
	command << channel_ << "gj";

	int ret = SendSerialCommand(port_.c_str(), command.str().c_str(), "\r");
	if (ret != DEVICE_OK)
		return ret;

	string answer;
	ret = GetSerialAnswer(port_.c_str(), "\r", answer);
	if (ret != DEVICE_OK)
		return ret;

	string message = removeLineFeed(answer);
	if (isStatus(message)) {
		return getStatusCode(message);
	}

	else if (message.substr(1, 2).compare("GJ") != 0)
		return ERR_UNEXPECTED_ANSWER;

	else {
		string val = message.substr(3, message.length() - 3);
		jogStep = positionFromHex(val) * 360.0 / pulsesPerRev_;
		return DEVICE_OK;
	}
}

//---------------------------------------------------------------------------
// Action handlers
//---------------------------------------------------------------------------

// Ask the device to go to its home position (offset) + wait for reply
// The rotation direction is given by homeDir_
int  ELL14::Home() {
	ostringstream command;

	command << channel_ << "ho" << static_cast<char>(homeDir_);
	int ret = SendSerialCommand(port_.c_str(), command.str().c_str(), "\r");
	if (ret != DEVICE_OK)
		return ret;

	return ret = receivePosition(360 * 1.4);
}

// Ask the device to "jog"
// The rotation direction is given by homeDir_
int  ELL14::Jog() {
	ostringstream command;

	string commandID = (jogDir_ == rotDirection::FW) ? "fw" : "bw";
	command << channel_ << commandID;
	int ret = SendSerialCommand(port_.c_str(), command.str().c_str(), "\r");
	if (ret != DEVICE_OK)
		return ret;

	return receivePosition(jogStep_);
}

// Ask the device to search for its best working frequencies (for both motors)
int  ELL14::SearchFrequencies() {
	ostringstream command1;
	command1 << channel_ << "s1";
	int ret = SendSerialCommand(port_.c_str(), command1.str().c_str(), "\r");
	if (ret != DEVICE_OK)
		return ret;

	string answer1;
	CDeviceUtils::SleepMs(5000);
	ret = GetSerialAnswer(port_.c_str(), "\r", answer1);
	if (ret != DEVICE_OK)
		return ret;

	ostringstream command2;
	command2 << channel_ << "s2";
	ret = SendSerialCommand(port_.c_str(), command2.str().c_str(), "\r");
	if (ret != DEVICE_OK)
		return ret;

	string answer2;
	CDeviceUtils::SleepMs(5000);
	ret = GetSerialAnswer(port_.c_str(), "\r", answer2);
	if (ret != DEVICE_OK)
		return ret;

	return DEVICE_OK;
}

//---------------------------------------------------------------------------
// Convenience function
//---------------------------------------------------------------------------

// Convert the long data into an ASCII sequence
string  ELL14::positionFromValue(long val) {
	ostringstream ss;
	ss << hex << val;
	string s = ss.str();
	int size = s.length();

	//mettre en majuscule
	int i = 0;
	while (s[i]) {
		s[i] = toupper(s[i]);
		i++;
	}

	//add missing characters (need 8 bytes command)
	stringstream ss_pos;
	for (int i = 0; i < 8 - size; i++) {
		ss_pos << "0";
	}
	ss_pos << s;

	return ss_pos.str();
}

// Convert ASCII data sequence into int data
int  ELL14::positionFromHex(string pos) {
	long n;

	// convert to long
	sscanf(pos.c_str(), "%x", &n);

	return n;
}

string  ELL14::removeLineFeed(string answer) {
	string message;
	if (answer.substr(0, 1).compare("\n") == 0) {
		message = answer.substr(1, answer.length() - 1);
	}
	else {
		message = answer;
	}

	return message;
}

// Remove the 3 first caracters of the command (adress (1) + command ID (2))
string  ELL14::removeCommandFlag(string message) {
	string value = message.substr(3, message.length() - 3);
	return value;
}

// Signal if the reply is a Status reply
bool  ELL14::isStatus(string message) {
	if (message.substr(1, 2).compare("GS") == 0) {
		return true;
	}
	return false;
}

// Study the get status reply and return the status code
int  ELL14::getStatusCode(string message) {
	string code = removeCommandFlag(message);

	if (code.compare("00") == 0) {
		return DEVICE_OK;
	}
	else if (code.compare("01") == 0) {
		return ERR_COMMUNICATION_TIME_OUT;
	}
	else if (code.compare("02") == 0) {
		return ERR_MECHANICAL_TIME_OUT;
	}
	else if (code.compare("03") == 0) {
		return ERR_COMMAND_ERROR_OR_NOT_SUPPORTED;
	}
	else if (code.compare("04") == 0) {
		return ERR_VALUE_OUT_OF_RANGE;
	}
	else if (code.compare("05") == 0) {
		return ERR_MODULE_ISOLATED;
	}
	else if (code.compare("06") == 0) {
		return ERR_MODULE_OUT_OF_ISOLATION;
	}
	else if (code.compare("07") == 0) {
		return ERR_INITIALIZING_ERROR;
	}
	else if (code.compare("08") == 0) {
		return ERR_THERMAL_ERROR;
	}
	else if (code.compare("09") == 0) {
		return ERR_BUSY;
	}
	else if (code.compare("0A") == 0) {
		return ERR_SENSOR_ERROR;
	}
	else if (code.compare("0B") == 0) {
		return ERR_MOTOR_ERROR;
	}
	else if (code.compare("0C") == 0) {
		return ERR_OUT_OF_RANGE;
	}
	else if (code.compare("0D") == 0) {
		return ERR_OVER_CURRENT_ERROR;
	}

	return ERR_UNKNOWN_ERROR;
}

// Verify position reply
int  ELL14::receivePosition(double angle) {
	string answer;

	// wait for the motion to be completed
	double sleepTime = angle * maxReplyTimeMs_ / 360;
	CDeviceUtils::SleepMs(sleepTime);

	int ret = GetSerialAnswer(port_.c_str(), "\r", answer);
	if (ret != DEVICE_OK)
		return ret;

	// remove "\n" if start character
	string message = removeLineFeed(answer);

	// check for status
	if (isStatus(message)) {
		return getStatusCode(message);
	}

	else if (message.substr(1, 2).compare("PO") != 0) {
		return ERR_UNEXPECTED_ANSWER;
	}

	// extract new real position
	string val = message.substr(3, message.length() - 3);
	pos_ = modulo2pi((positionFromHex(val) * 360.0 / pulsesPerRev_));

	return DEVICE_OK;
}

double  ELL14::modulo2pi(double angle) {
	double res;
	res = fmod(angle, 360.0);
	if (res < 0)
		res += 360.0;
	return res;
}