#include "Cobolt08_01.h"
using namespace std;

Cobolt08_01::Cobolt08_01(const char* nameAndSlot) : initialized_(false)
{
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


Cobolt08_01::~Cobolt08_01()
{
	Shutdown();
}



int Cobolt08_01::Initialize()
{

	if (!initialized_) {

		parentHub_ = static_cast<OxxiusCombinerHub*>(GetParentHub());
		if (!parentHub_) {
			return DEVICE_COMM_HUB_MISSING;
		}
		


		RETURN_ON_MM_ERROR(UpdateStatus());

		initialized_ = true;

	}

	return DEVICE_OK;

}


int Cobolt08_01::Shutdown()
{
	initialized_ = false;
	return DEVICE_OK;
}




int Cobolt08_01::Fire(double deltaT)
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

int Cobolt08_01::OnAlarm(MM::PropertyBase* pProp, MM::ActionType)
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


int Cobolt08_01::OnEmissionOnOff(MM::PropertyBase* pProp, MM::ActionType eAct)
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
