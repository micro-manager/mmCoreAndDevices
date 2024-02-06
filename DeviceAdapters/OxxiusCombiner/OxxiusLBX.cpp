#include "OxxiusLBX.h"
using namespace std;

OxxiusLBX::OxxiusLBX(const char* nameAndSlot) : initialized_(false)
{
	string tSlot = string(nameAndSlot);

	name_.assign(tSlot);// set laser name
	tSlot = tSlot.substr(tSlot.length() - 1, 1);
	slot_ = (unsigned int)atoi(tSlot.c_str());// set laser slot

	parentHub_;
	busy_ = false;
	laserOn_ = false;
	alarm_ = "";
	state_ = "";
	digitalMod_ = "";
	analogMod_ = "";
	controlMode_ = "";

	//MPA
	mpa_number = -1; //probably not needed

	// powerSetPoint_ = 0.0;
	maxRelPower_ = 100.0;
	nominalPower_ = 100.0;
	maxCurrent_ = 125.0;

	InitializeDefaultErrorMessages();
	SetErrorText(ERR_NO_PORT_SET, "Hub Device not found.  The Laser combiner is needed to create this device");

	// parent ID display
	CreateHubIDProperty();
}


OxxiusLBX::~OxxiusLBX()
{
	Shutdown();
}



int OxxiusLBX::Initialize()
{
	if (!initialized_) {

		size_t found;
		string strSlot;
		string spa;


		parentHub_ = static_cast<OxxiusCombinerHub*>(GetParentHub());
		if (!parentHub_) {
			return DEVICE_COMM_HUB_MISSING;
		}
		char hubLabel[MM::MaxStrLength];
		parentHub_->GetLabel(hubLabel);
		SetParentID(hubLabel); // for backward compatibility

		// Set property list
		// -----------------
		// Name (read only)
		RETURN_ON_MM_ERROR(CreateProperty(MM::g_Keyword_Name, name_.c_str(), MM::String, true));

		// Description (read only)
		ostringstream descriPt1;
		char sourceSerialNumber[] = "LAS-XXXXXXXXXX";
		parentHub_->QueryCommand(this, GetCoreCallback(), slot_, "HID?", false);
		parentHub_->ParseforChar(sourceSerialNumber);

		parentHub_->QueryCommand(this, GetCoreCallback(), slot_, "IP", true);
		parentHub_->ParseforString(spa);

		parentHub_->QueryCommand(this, GetCoreCallback(), slot_, "INF?", false);
		parentHub_->ParseforString(strSlot);

		// Retrieves and delete the laser's type
		found = strSlot.find("-");
		if (found != string::npos) {
			strSlot.erase(0, found + 1);
		}

		// Retrieves and define the laser's wavelength
		found = strSlot.find("-");
		if (found != string::npos) {
			waveLength = (unsigned int)atoi(strSlot.substr(0, found).c_str());
		}

		// Retrieves and define the nominal power
		strSlot.erase(0, found + 1);
		nominalPower_ = (float)atof(strSlot.substr(0, found).c_str());


		if (parentHub_->GetMPA(slot_)) {
			mpa_number = slot_;
		}
		else {
			mpa_number = -1;
		}


		// LBX model
		descriPt1 << "LBX";
		descriPt1 << " source on slot " << slot_;
		descriPt1 << ", " << sourceSerialNumber;

		RETURN_ON_MM_ERROR(CreateProperty(MM::g_Keyword_Description, descriPt1.str().c_str(), MM::String, true));

		// Alarm (read only)
		CPropertyAction* pAct = new CPropertyAction(this, &GenericLaser::OnAlarm);
		RETURN_ON_MM_ERROR(CreateProperty("Alarm", "None", MM::String, true, pAct));

		// Status (read only)
		pAct = new CPropertyAction(this, &GenericLaser::OnState);
		RETURN_ON_MM_ERROR(CreateProperty("State", "", MM::String, true, pAct));

		// Emission selector (write/read)
		pAct = new CPropertyAction(this, &GenericLaser::OnEmissionOnOff);
		RETURN_ON_MM_ERROR(CreateProperty("Emission", "", MM::String, false, pAct));
		AddAllowedValue("Emission", "ON");
		AddAllowedValue("Emission", "OFF");

		// Digital modulation selector (write/read)
		pAct = new CPropertyAction(this, &GenericLaser::OnDigitalMod);
		RETURN_ON_MM_ERROR(CreateProperty("Digital Modulation", "", MM::String, false, pAct));
		AddAllowedValue("Digital Modulation", "ON");
		AddAllowedValue("Digital Modulation", "OFF");

		// Analog modulation selector (write/read)
		pAct = new CPropertyAction(this, &GenericLaser::OnAnalogMod);
		RETURN_ON_MM_ERROR(CreateProperty("Analog Modulation", "", MM::String, false, pAct));
		AddAllowedValue("Analog Modulation", "ON");
		AddAllowedValue("Analog Modulation", "OFF");

		// Control mode selector (= APC or ACC) (write/read)
		pAct = new CPropertyAction(this, &GenericLaser::OnControlMode);
		RETURN_ON_MM_ERROR(CreateProperty("Control mode", "", MM::String, false, pAct));
		AddAllowedValue("Control mode", "ACC");
		AddAllowedValue("Control mode", "APC");

		//Fire property
		pAct = new CPropertyAction(this, &GenericLaser::OnFire);
		RETURN_ON_MM_ERROR(CreateProperty("Fire", "0", MM::Float, false, pAct));

		// Define the maximal current and power
		maxRelPower_ = 100.0;
		maxCurrent_ = 105.0;

		// Power set point (write/read)
		pAct = new CPropertyAction(this, &GenericLaser::OnPowerSetPoint);
		RETURN_ON_MM_ERROR(CreateProperty("Power set point", "0", MM::Float, false, pAct));
		SetPropertyLimits("Power set point", 0, maxRelPower_);

		// Current set point (write/read)
		pAct = new CPropertyAction(this, &GenericLaser::OnCurrentSetPoint);
		RETURN_ON_MM_ERROR(CreateProperty("Current set point", "0", MM::Float, false, pAct));
		SetPropertyLimits("Current set point", 0, maxCurrent_);

		RETURN_ON_MM_ERROR(UpdateStatus());

		initialized_ = true;
	}

	return DEVICE_OK;
}


/*
int OxxiusLBX::Shutdown()
{
	initialized_ = false;
	return DEVICE_OK;
}
*/


int OxxiusLBX::Fire(double deltaT)
{
	string activate_query = "DL 1";
	string deactivate_query = "DL 0";

	if (laserOn_ == false) {
		laserOn_ = true;
		RETURN_ON_MM_ERROR(parentHub_->QueryCommand(this, GetCoreCallback(), slot_, activate_query.c_str(), false));
	}
	CDeviceUtils::SleepMs((long)(deltaT));
	laserOn_ = false;
	RETURN_ON_MM_ERROR(parentHub_->QueryCommand(this, GetCoreCallback(), slot_, deactivate_query.c_str(), false));
	//At the end, the laser is set to off

	return DEVICE_OK;
}

//RETURN_ON_MM_ERROR(parentHub_->QueryCommand(this, GetCoreCallback(), slot_, newCommand.c_str(), false));

///////////////////////////////////////////////////////////////////////////////
// Action handlers
///////////////////////////////////////////////////////////////////////////////

int OxxiusLBX::OnAlarm(MM::PropertyBase* pProp, MM::ActionType)
{
	unsigned int alarmInt = 99;
	RETURN_ON_MM_ERROR(parentHub_->QueryCommand(this, GetCoreCallback(), slot_, "?F", false));

	parentHub_->ParseforInteger(alarmInt);

	switch (alarmInt) {
	case 0:
		alarm_ = "No Alarm";
		break;
	case 1:
		alarm_ = "Out-of-bounds diode current";
		break;
	case 2:
		alarm_ = "Unexpected laser power value";
		break;
	case 3:
		alarm_ = "Out-of-bounds supply voltage";
		break;
	case 4:
		alarm_ = "Out-of-bounds internal temperature";
		break;
	case 5:
		alarm_ = "Out-of-bounds baseplate temperature";
		break;
	case 7:
		alarm_ = "Interlock circuit open";
		break;
	case 8:
		alarm_ = "Soft reset";
		break;
	default:
		alarm_ = "Other alarm";
	}

	pProp->Set(alarm_.c_str());

	return DEVICE_OK;
}


int OxxiusLBX::OnState(MM::PropertyBase* pProp, MM::ActionType)
{
	unsigned int stateInt = 99;
	RETURN_ON_MM_ERROR(parentHub_->QueryCommand(this, GetCoreCallback(), slot_, "?STA", false));

	parentHub_->ParseforInteger(stateInt);

	switch (stateInt) {
	case 1:
		state_ = "Warm-up phase";
		break;
	case 2:
		state_ = "Stand-by state";
		break;
	case 3:
		state_ = "Emission on";
		break;
	case 4:
		state_ = "Internal error";
		break;
	case 5:
		state_ = "Alarm";
		break;
	case 6:
		state_ = "Sleep state";
		break;
	default:
		state_ = "Other state";
	}

	pProp->Set(alarm_.c_str());

	return DEVICE_OK;
}


int OxxiusLBX::OnEmissionOnOff(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet) {
		unsigned int status = 0;
		ostringstream query;

		query << "?CS " << slot_;

		RETURN_ON_MM_ERROR(parentHub_->QueryCommand(this, GetCoreCallback(), NO_SLOT, query.str().c_str(), false));
		parentHub_->ParseforInteger(status);

		switch (status) {
		case 0:		// LBX model: Emission off
			laserOn_ = false;
			break;
		case 1:		// LBX model: Emission on
			laserOn_ = true;
			break;
		default:
			laserOn_ = true;
		}

		if (laserOn_) {
			pProp->Set("ON");
		}
		else {
			pProp->Set("OFF");
		}
	}
	else if (eAct == MM::AfterSet) {
		string newEmissionStatus, newCommand = "";

		pProp->Get(newEmissionStatus);

		if (newEmissionStatus.compare("ON") == 0) {
			newCommand = "DL 1";
			laserOn_ = true;
		}
		else if (newEmissionStatus.compare("OFF") == 0) {
			newCommand = "DL 0";
			laserOn_ = false;
		}

		RETURN_ON_MM_ERROR(parentHub_->QueryCommand(this, GetCoreCallback(), slot_, newCommand.c_str(), false));
	}
	return DEVICE_OK;
}


int OxxiusLBX::OnDigitalMod(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	unsigned int querySlot = NO_SLOT;

	if (eAct == MM::BeforeGet) {
		string query;
		
		querySlot = slot_;
		query = "?TTL";

		RETURN_ON_MM_ERROR(parentHub_->QueryCommand(this, GetCoreCallback(), querySlot, query.c_str(), false));

		bool digiM;
		parentHub_->ParseforBoolean(digiM);

		if (digiM)
			digitalMod_.assign("ON");
		else
			digitalMod_.assign("OFF");

		pProp->Set(digitalMod_.c_str());

	}
	else if (eAct == MM::AfterSet) {
		string newModSet, newCommand;

		pProp->Get(newModSet);
		digitalMod_.assign(newModSet);

		if (digitalMod_ == "ON") {
			querySlot = slot_;
			newCommand.assign("TTL 1");
		}

		else if (digitalMod_ == "OFF") {
			querySlot = slot_;
			newCommand.assign("TTL 0");
		}

		RETURN_ON_MM_ERROR(parentHub_->QueryCommand(this, GetCoreCallback(), querySlot, newCommand.c_str(), false));
	}
	return DEVICE_OK;
}


int OxxiusLBX::OnAnalogMod(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet) {
		string query;
		query = "?AM";
	
		RETURN_ON_MM_ERROR(parentHub_->QueryCommand(this, GetCoreCallback(), slot_, query.c_str(), false));
		bool digiM;
		parentHub_->ParseforBoolean(digiM);

		if (digiM) {
			digitalMod_.assign("ON");
		}
		else {
			digitalMod_.assign("OFF");
		}
		pProp->Set(digitalMod_.c_str());

	}
	else if (eAct == MM::AfterSet) {
		ostringstream newCommand;
		newCommand << "AM ";

		pProp->Get(digitalMod_);

		if (digitalMod_ == "OFF")
			newCommand << "0";
		else
			newCommand << "1";

		RETURN_ON_MM_ERROR(parentHub_->QueryCommand(this, GetCoreCallback(), slot_, newCommand.str().c_str(), false));
	}
	return DEVICE_OK;
}


int OxxiusLBX::OnControlMode(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet) {
		unsigned int ctrlM = 1;
		string command;

		command = "?APC";

		RETURN_ON_MM_ERROR(parentHub_->QueryCommand(this, GetCoreCallback(), slot_, command.c_str(), false));
		parentHub_->ParseforInteger(ctrlM);

		if (ctrlM == 1) {
			controlMode_.assign("APC");
		}
		else if (ctrlM == 0) {
			controlMode_.assign("ACC");
		}

		pProp->Set(controlMode_.c_str());
	}
	else if (eAct == MM::AfterSet) {
		string newCommand;

		pProp->Get(controlMode_);

		if (controlMode_ == "ACC") {
			newCommand.assign("APC 0");
		}
		else if (controlMode_ == "APC") {
			newCommand.assign("APC 1");
		}

		RETURN_ON_MM_ERROR(parentHub_->QueryCommand(this, GetCoreCallback(), slot_, newCommand.c_str(), false));
	}
	return DEVICE_OK;
}

/*
int OxxiusLBX::OnPowerSetPoint(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet) {
		string command = "?SP";
		unsigned int thisSlot = slot_;
		float absSetPoint_;


		if ((0 < mpa_number) && (mpa_number < 7)) {
			thisSlot = NO_SLOT;
			command = "?PL";
			stringstream s;
			s << mpa_number;
			command += s.str();
		}

		RETURN_ON_MM_ERROR(parentHub_->QueryCommand(this, GetCoreCallback(), thisSlot, command.c_str(), false));
		parentHub_->ParseforFloat(absSetPoint_ );

		pProp->Set(static_cast<double>( absSetPoint_ ));
	}
	else if (eAct == MM::AfterSet) {

		double GUISetPoint = 0.0;
		pProp->Get(GUISetPoint);

		if ((GUISetPoint >= 0.0) || (GUISetPoint <= nominalPower_)) {
			string command = "P";
			unsigned int thisSlot = slot_;

			ostringstream newCommand;
			char* powerSPString = new char[20];
			strcpy(powerSPString, CDeviceUtils::ConvertToString( static_cast<double>(GUISetPoint) ) );

			if ((0 < mpa_number) && (mpa_number < 7)) {
				thisSlot = NO_SLOT;
				command = "IP";
				command += CDeviceUtils::ConvertToString((int)(mpa_number));

				strcpy(powerSPString, CDeviceUtils::ConvertToString(GUISetPoint));
			}
			
			newCommand << command << " " << powerSPString;
			RETURN_ON_MM_ERROR(parentHub_->QueryCommand(this, GetCoreCallback(), thisSlot, newCommand.str().c_str(), false));
		}
		else {
			// If the value entered through the GUI is not valid, read the machine value
			OnPowerSetPoint(pProp, MM::BeforeGet);
		}
	}
	return DEVICE_OK;
}
*/


int OxxiusLBX::OnCurrentSetPoint(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet) {
		float machineSetPoint = 0.0;
		string command = "?SC";
		
		RETURN_ON_MM_ERROR(parentHub_->QueryCommand(this, GetCoreCallback(), slot_, command.c_str(), false));
		parentHub_->ParseforFloat(machineSetPoint);

		pProp->Set(machineSetPoint);
	}

	else if (eAct == MM::AfterSet) {

		double GUISetPoint = 0.0;
		pProp->Get(GUISetPoint);

		if ((GUISetPoint >= 0.0) || (GUISetPoint <= maxCurrent_)) {

			ostringstream newCommand;
			string command = "C";
			
			char* currentSPString = new char[20];
			strcpy(currentSPString, CDeviceUtils::ConvertToString(GUISetPoint));

			newCommand << command << " " << currentSPString;
			RETURN_ON_MM_ERROR(parentHub_->QueryCommand(this, GetCoreCallback(), slot_, newCommand.str().c_str(), false));
			
		}
	}
	return DEVICE_OK;
}

/*
int OxxiusLBX::OnFire(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::AfterSet) {
		double input;
		pProp->Get(input);
		return Fire(input);
	}
	return DEVICE_OK;
}
*/