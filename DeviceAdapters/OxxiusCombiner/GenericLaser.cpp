#include "GenericLaser.h"

int GenericLaser::Shutdown()
{
	initialized_ = false;
	return DEVICE_OK;
}

void GenericLaser::GetName(char* Name) const
{
	CDeviceUtils::CopyLimitedString(Name, name_.c_str());
}

bool GenericLaser::Busy()
{
	return busy_;
}

int GenericLaser::SetOpen(bool openCommand)
{
	string activate_query = "DL 1";
	string deactivate_query = "DL 0";

	if (openCommand == false) {
		RETURN_ON_MM_ERROR(parentHub_->QueryCommand(this, GetCoreCallback(), slot_, deactivate_query.c_str(), false));
		laserOn_ = false;
	}
	else {
		RETURN_ON_MM_ERROR(parentHub_->QueryCommand(this, GetCoreCallback(), slot_, activate_query.c_str(), false));
		laserOn_ = true;
	}
	return DEVICE_OK;
}


int GenericLaser::GetOpen(bool& isOpen)
{
	unsigned int status = 0;
	ostringstream query;

	query << "?CS " << slot_;
	RETURN_ON_MM_ERROR(parentHub_->QueryCommand(this, GetCoreCallback(), NO_SLOT, query.str().c_str(), false));
	parentHub_->ParseforInteger(status);

	switch (status) {
	case 0:		// Emission is OFF
		laserOn_ = false;
		break;
	case 1:		// Emission is ON
		laserOn_ = true;
		break;
	default:	// Other cases: emission is potentially ON
		laserOn_ = true;
	}
	isOpen = laserOn_;
	return DEVICE_OK;
}



int GenericLaser::OnPowerSetPoint(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet) {
		double relSetPoint_ = 0.0;
		string powSetPointQuery = "?PPL";		// Generic query to read the power set point (percentage)
		ostringstream powSetPointPhrase;
		powSetPointPhrase << powSetPointQuery << slot_;

		RETURN_ON_MM_ERROR(parentHub_->QueryCommand(this, GetCoreCallback(), NO_SLOT, powSetPointPhrase.str().c_str(), false));
		parentHub_->ParseforPercent(relSetPoint_);

		pProp->Set(relSetPoint_);
	}
	else if (eAct == MM::AfterSet) {

		double GUISetPoint = 0.0;
		pProp->Get(GUISetPoint);

		if ((GUISetPoint >= 0.0) || (GUISetPoint <= POW_UPPER_LIMIT)) {
			string powSetPointCommand = "PPL";	// Generic query to set the laser power (percentage)

			ostringstream powSetPointPhrase;
			char* powerSPString = new char[20];
			strcpy( powerSPString, CDeviceUtils::ConvertToString(GUISetPoint) );

			powSetPointPhrase << powSetPointCommand << slot_ << " " << powerSPString;
			RETURN_ON_MM_ERROR(parentHub_->QueryCommand(this, GetCoreCallback(), NO_SLOT, powSetPointPhrase.str().c_str(), false));
		}
		else {}		// Do nothing if the GUI set point is out-of-bounds
		
	}
	return DEVICE_OK;
}


int GenericLaser::OnFire(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::AfterSet) {
		double input;
		pProp->Get(input);
		return Fire(input);
	}
	return DEVICE_OK;
}