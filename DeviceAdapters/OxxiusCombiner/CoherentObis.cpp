#include "CoherentObis.h"
using namespace std;

CoherentObis::CoherentObis(const char* nameAndSlot)
{
	initialized_ = false;
	
	string tSlot = string(nameAndSlot);

	name_.assign(tSlot);// set laser name
	tSlot = tSlot.substr(tSlot.length() - 1, 1);
	slot_ = (unsigned int)atoi(tSlot.c_str());// set laser slot

	parentHub_;
	busy_ = false;
	laserOn_ = false;

	alarm_ = "";

	InitializeDefaultErrorMessages();
	SetErrorText(ERR_NO_PORT_SET, "Hub Device not found.  The Laser combiner is needed to create this device");

	// parent ID display
	CreateHubIDProperty();
}


CoherentObis::~CoherentObis()
{
	Shutdown();
}


int CoherentObis::Initialize()
{

	if (!initialized_) {

		parentHub_ = static_cast<OxxiusCombinerHub*>(GetParentHub());
		if (!parentHub_) {
			return DEVICE_COMM_HUB_MISSING;
		}

		string cmdmaxpwr = "SOUR1:POW:LIM:HIGH?";
		string cmdminpwr = "SOUR1:POW:LIM:LOW?";
		parentHub_->QueryCommand(this, GetCoreCallback(), slot_, cmdmaxpwr.c_str(), true);
		parentHub_->ParseforDouble(maxPower_);
		parentHub_->QueryCommand(this, GetCoreCallback(), slot_, cmdminpwr.c_str(), true);
		parentHub_->ParseforDouble(minPower_);



		// Set property list
		// -----------------
		// Name (read only)
		RETURN_ON_MM_ERROR(CreateProperty(MM::g_Keyword_Name, name_.c_str(), MM::String, true));

		// store the device serial number
		parentHub_->QueryCommand(this, GetCoreCallback(), slot_, "SYST:INF:SNUM?", false);
		parentHub_->ParseforString(serialNumber_);

		// Description (read only)
		ostringstream descriPt1;
		parentHub_->QueryCommand(this, GetCoreCallback(), slot_, "*IDN?", false);
		parentHub_->ParseforString(description_); //example: "Coherent, Inc - OBIS 405nm 50mW C - V1.0.1 - Dec 14 2010"

		//we ignore "spa"

		//define the laser's wavelength
		parentHub_->QueryCommand(this, GetCoreCallback(), slot_, "SYST1:INF:WAV?", false);
		parentHub_->ParseforDouble(wavelength_);

		//let's skip the nominal power retrieve

		//let's assume there are no MPA or AOMs

		std::ostringstream InfoMessage1;   // Debug purposes only
		InfoMessage1 << "INFOS RECUPEREES: maxpower:"<<maxPower_<<" minpower:"<<minPower_<<" num de serie:"<<serialNumber_<<" description:"<<description_<<" wavelength:"<<wavelength_;
		GetCoreCallback()->LogMessage(this, InfoMessage1.str().c_str(), false);

		descriPt1 << "OBIS source on slot " << slot_;
		descriPt1 << ", " << string(serialNumber_);

		RETURN_ON_MM_ERROR(CreateProperty(MM::g_Keyword_Description, descriPt1.str().c_str(), MM::String, true));

		// Alarm (read only)
		CPropertyAction* pAct = new CPropertyAction(this, &GenericLaser::OnAlarm);
		RETURN_ON_MM_ERROR(CreateProperty("Alarm", "None", MM::String, true, pAct));
		//No status
		//No Control mode



		//Emission selector
		pAct = new CPropertyAction(this, &GenericLaser::OnEmissionOnOff);
		RETURN_ON_MM_ERROR(CreateProperty("Emission", "", MM::String, false, pAct));
		AddAllowedValue("Emission", "ON");
		AddAllowedValue("Emission", "OFF");

		//No Current set point ?
		maxPower_ = 100.0;

		//Set Power
		pAct = new CPropertyAction(this, &GenericLaser::OnPowerSetPoint);
		RETURN_ON_MM_ERROR(CreateProperty("Power set point", "0.0", MM::Float, false, pAct));
		SetPropertyLimits("Power set point", 0.0, maxPower_);

		//Read Power
		pAct = new CPropertyAction(this, &GenericLaser::OnPowerReadback);
		RETURN_ON_MM_ERROR(CreateProperty("Read Power point", "0.0", MM::Float, false, pAct));


		//Read Current
		pAct = new CPropertyAction(this, &GenericLaser::OnCurrentReadback);
		RETURN_ON_MM_ERROR(CreateProperty("Read Current point", "0.0", MM::Float, false, pAct));

		//Fire
		pAct = new CPropertyAction(this, &GenericLaser::OnFire);
		RETURN_ON_MM_ERROR(CreateProperty("Fire", "0", MM::Float, false, pAct));

		//Modes
		pAct = new CPropertyAction(this, &GenericLaser::OnOperatingMode);
		RETURN_ON_MM_ERROR(CreateProperty("Mode", "", MM::String, false, pAct));


		RETURN_ON_MM_ERROR(UpdateStatus());

		initialized_ = true;

	}

	return DEVICE_OK;

}


int CoherentObis::Fire(double deltaT)
{
	string activate_query = "dl 1"; //SOUR1:AM:STAT ON
	string deactivate_query = "dl 0";

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

//Handlers

int CoherentObis::OnAlarm(MM::PropertyBase* pProp, MM::ActionType)
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


int CoherentObis::OnEmissionOnOff(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet) {
		double status;
		ostringstream query;

		query << "?l";

		RETURN_ON_MM_ERROR(parentHub_->QueryCommand(this, GetCoreCallback(), slot_, query.str().c_str(), false));
		parentHub_->ParseforDouble(status);

		if (status == 1) {
			laserOn_ = true;
		}
		else {
			laserOn_ = false;
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
			newCommand = "dl 1"; //original: SOUR1:AM:STAT ON
			laserOn_ = true;
		}
		else if (newEmissionStatus.compare("OFF") == 0) {
			newCommand = "dl 0";
			laserOn_ = false;
		}

		RETURN_ON_MM_ERROR(parentHub_->QueryCommand(this, GetCoreCallback(), slot_, newCommand.c_str(), false));
	}

	return DEVICE_OK;
}

/*
int CoherentObis::OnPowerSetPoint(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet) {
		float absSetPoint_;
		string command = "SOUR1:POW:LEV:IMM:AMPL?";
		RETURN_ON_MM_ERROR(parentHub_->QueryCommand(this, GetCoreCallback(), slot_, command.c_str(), false));
		parentHub_->ParseforFloat(absSetPoint_);

		pProp->Set(absSetPoint_);
	}
	else if (eAct == MM::AfterSet) {
		double GUISetPoint = 0.0;
		pProp->Get(GUISetPoint);

		if ((GUISetPoint >= 0.0) && (GUISetPoint <= maxPower_)) { //&& instead of ||
			string command = "SOUR1:POW:LEV:IMM:AMPL ";
			command += to_string(GUISetPoint);

			RETURN_ON_MM_ERROR(parentHub_->QueryCommand(this, GetCoreCallback(), slot_, command.c_str(), false));
			
		}
		else {
			// If the value entered through the GUI is not valid, read the machine value
			OnPowerSetPoint(pProp, MM::BeforeGet);
		}
	}
	return DEVICE_OK;
}
*/

int CoherentObis::OnPowerReadback(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		string command = "SOUR1:POW:LEV:IMM:AMPL?";
		float power;
		RETURN_ON_MM_ERROR(parentHub_->QueryCommand(this, GetCoreCallback(), slot_, command.c_str(), false));
		parentHub_->ParseforFloat(power);
		pProp->Set(power);
	}
	return DEVICE_OK;
}

int CoherentObis::OnCurrentReadback(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		string command = "SOUR1:POW:CURR?";
		float current;
		RETURN_ON_MM_ERROR(parentHub_->QueryCommand(this, GetCoreCallback(), slot_, command.c_str(), false));
		parentHub_->ParseforFloat(current);
		pProp->Set(current);
	}
	return DEVICE_OK;
}

int CoherentObis::OnOperatingMode(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if(eAct==MM::BeforeGet)
	{ 
		string command = "SOUR:AM:SOUR?";
		string operating_mode;
		RETURN_ON_MM_ERROR(parentHub_->QueryCommand(this, GetCoreCallback(), slot_, command.c_str(), false));
		parentHub_->ParseforString(operating_mode);
		pProp->Set(operating_mode.c_str());
	}
	else if (eAct == MM::AfterSet) {
		string input = "";
		string command;
		pProp->Get(input);
		if (input == "CWP" || input == "CWC") {
			command = "SOUR:AM:INT ";
		}
		else if (input == "DIG" || input == "ANAL" || input == "MIX" || input == "DIGITAL" || input == "ANALOG" || input == "MIXED" || input == "DIGSO" || input == "MIXSO") {
			command = "SOUR:AM:EXT ";
		}
		command += input;
		RETURN_ON_MM_ERROR(parentHub_->QueryCommand(this, GetCoreCallback(), slot_, command.c_str(), false));
		string rep;
		parentHub_->ParseforString(rep);


		std::ostringstream Info;   /////DEBUG
		Info << "DEBUG OPERATING MODE: " << rep << "avec la commande encoyée suivante: " << command;
		GetCoreCallback()->LogMessage(this, Info.str().c_str(), false);

	}

	return DEVICE_OK;

}
/*
int CoherentObis::OnFire(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::AfterSet) {
		double input;
		pProp->Get(input);
		return Fire(input);
	}
	return DEVICE_OK;
}
*/