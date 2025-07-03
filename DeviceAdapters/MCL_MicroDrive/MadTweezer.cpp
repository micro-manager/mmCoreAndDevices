/*
File:		MadTweezer.cpp
Copyright:	Mad City Labs Inc., 2023
License:	Distributed under the BSD license.
*/

#include "AcquireDevice.h"
#include "MadTweezer.h"
#include "MCL_MicroDrive.h"
#include "HandleListType.h"
#include "DeviceUtils.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <string.h>

#include <vector>
#include <iostream>

using namespace std;

MadTweezer::MadTweezer() :
	handle_(0),
	serialNumber_(0),
	pid_(0),
	axis_(0),
	encoderResolution_(0.0),
	stepSize_rad_(0.0),
	maxVelocity_(0.0),
	minVelocity_(0.0),
	location_mrad_(0.0),
	max_mrad_(0.0),
	velocity_rad_(0.0),
	units_(0),
	mode_(0),
	busy_(false),
	initialized_(false),
	encoded_(false),
	home_(false),
	direction_(1) // clockwise
{
	InitializeDefaultErrorMessages();

	// MCL error messages 
	SetErrorText(MCL_GENERAL_ERROR, "MCL Error: General Error");
	SetErrorText(MCL_DEV_ERROR, "MCL Error: Error transferring data to device");
	SetErrorText(MCL_DEV_NOT_ATTACHED, "MCL Error: Device not attached");
	SetErrorText(MCL_USAGE_ERROR, "MCL Error: Using a function from library device does not support");
	SetErrorText(MCL_DEV_NOT_READY, "MCL Error: Device not ready");
	SetErrorText(MCL_ARGUMENT_ERROR, "MCL Error: Argument out of range");
	SetErrorText(MCL_INVALID_AXIS, "MCL Error: Invalid axis");
	SetErrorText(MCL_INVALID_HANDLE, "MCL Error: Handle not valid");
	SetErrorText(MCL_INVALID_DRIVER, "MCL Error: Invalid Driver");
	SetErrorText(INVALID_VELOCITY, "Velocity must be between max and min velocity for current mode");
	SetErrorText(INVALID_LOCATION, "Location must be between 0 mrad and 6,238.0249 mrad");
}


MadTweezer::~MadTweezer()
{
	Shutdown();
}


bool MadTweezer::Busy()
{
	return busy_;
}


void MadTweezer::GetName(char* pszName) const
{
	CDeviceUtils::CopyLimitedString(pszName, g_DeviceMadTweezerName);
}

int MadTweezer::Initialize()
{
	int err = DEVICE_OK;

	HandleListLock();
	err = InitDeviceAdapter();
	HandleListUnlock();

	return err;
}


int MadTweezer::InitDeviceAdapter()
{
	if (initialized_)
		return DEVICE_OK;

	// Attempt to acquire a device/axis for this adapter.
	int ret = MCL_SUCCESS;
	ret = AcquireDeviceHandle(MADTWEEZER_TYPE, handle_, axis_);
	if (ret != MCL_SUCCESS)
		return ret;

	// Query device information
	serialNumber_ = MCL_GetSerialNumber(handle_);

	ret = MCL_GetProductID(&pid_, handle_);
	if (ret != MCL_SUCCESS)
		return ret;

	double ignore1, ignore2;
	ret = MCL_MDAxisInformation(axis_, &encoderResolution_, &stepSize_rad_, &maxVelocity_, &ignore1,
		&ignore2, &minVelocity_, &units_, handle_);
	if (ret != MCL_SUCCESS)
		return ret;

	if (maxVelocity_ <= 100)
		velocity_rad_ = MAX_VEL_HIGH_PRECISION;
	else
		velocity_rad_ = maxVelocity_;
	max_mrad_ = 2 * M_PI * 1000;

	// Create properties
	int err = DEVICE_OK;

	err = CreateMadTweezerProperties();
	if (err != DEVICE_OK)
		return err;

	err = UpdateStatus();
	if (err != DEVICE_OK)
		return err;

	initialized_ = true;

	return DEVICE_OK;
}


int MadTweezer::CreateMadTweezerProperties()
{
	int err;
	char iToChar[25];

	vector<string> yesNoList;
	yesNoList.push_back(g_Listword_No);
	yesNoList.push_back(g_Listword_Yes);

	/// Read only properties

	// Name property
	err = CreateProperty(MM::g_Keyword_Name, g_DeviceMadTweezerName, MM::String, true);
	if (err != DEVICE_OK)
		return err;

	// Description Property
	err = CreateProperty(MM::g_Keyword_Description, "Rotational Axis Driver", MM::String, true);
	if (err != DEVICE_OK)
		return err;

	// Device handle
	sprintf(iToChar, "%d", handle_);
	err = CreateProperty(g_Keyword_Handle, iToChar, MM::String, true);
	if (err != DEVICE_OK)
		return err;

	// Product ID
	sprintf(iToChar, "%hu", pid_);
	err = CreateProperty(g_Keyword_ProductID, iToChar, MM::String, true);
	if (err != DEVICE_OK)
		return err;

	// Serial Number
	sprintf(iToChar, "%d", serialNumber_);
	err = CreateProperty(g_Keyword_Serial_Num, iToChar, MM::String, true);
	if (err != DEVICE_OK)
		return err;

	// Maximum velocity
	err = CreateProperty(g_Keyword_MaxVelocityHighSpeed, "125.6637", MM::Float, true);
	if (err != DEVICE_OK)
		return err;

	// Minumum velocity
	err = CreateProperty(g_Keyword_MinVelocityHighSpeed, "1.5708", MM::Float, true);
	if (err != DEVICE_OK)
		return err;

	// Maximum velocity
	err = CreateProperty(g_Keyword_MaxVelocityHighPrecision, "6.2831", MM::Float, true);
	if (err != DEVICE_OK)
		return err;

	// Minumum velocity
	err = CreateProperty(g_Keyword_MinVelocityHighPrecision, "0.1964", MM::Float, true);
	if (err != DEVICE_OK)
		return err;

	// Home
	CPropertyAction* pAct = new CPropertyAction(this, &MadTweezer::OnHome);
	err = CreateProperty(g_Keyword_Home, g_Listword_No, MM::String, false, pAct);
	if (err != DEVICE_OK)
		return err;
	err = SetAllowedValues(g_Keyword_Home, yesNoList);


	// Location
	sprintf(iToChar, "%f", location_mrad_);
	pAct = new CPropertyAction(this, &MadTweezer::OnLocation);
	err = CreateProperty(g_Keyword_Location, iToChar, MM::Float, false, pAct);
	if (err != DEVICE_OK)
		return err;

	// Mode
	vector<string> modeList;
	string mode;
	modeList.push_back(g_Keyword_HighSpeedMode);
	modeList.push_back(g_Keyword_HighPrecisionMode);
	GetMode();
	if (mode_ == HIGH_SPEED)
		mode = g_Keyword_HighSpeedMode;
	else if (mode_ == HIGH_PRECISION)
		mode = g_Keyword_HighPrecisionMode;

	pAct = new CPropertyAction(this, &MadTweezer::OnMode);
	err = CreateProperty(g_Keyword_Mode, mode.c_str(), MM::String, false, pAct);
	if (err != DEVICE_OK)
		return err;
	err = SetAllowedValues(g_Keyword_Mode, modeList);

	// Direction
	vector<string> directionList;
	directionList.push_back(g_Listword_Clockwise);
	directionList.push_back(g_Listword_CounterClockwise);
	pAct = new CPropertyAction(this, &MadTweezer::OnDirection);
	err = CreateProperty(g_Keyword_Direction, g_Listword_Clockwise, MM::String, false, pAct);
	if (err != DEVICE_OK)
		return err;
	err = SetAllowedValues(g_Keyword_Direction, directionList);

	// Rotation
	sprintf(iToChar, "%f", 0.0);
	pAct = new CPropertyAction(this, &MadTweezer::OnRotation);
	err = CreateProperty(g_Keyword_Rotations, iToChar, MM::Float, false, pAct);
	if (err != DEVICE_OK)
		return err;

	// Velocity
	sprintf(iToChar, "%f", velocity_rad_);
	pAct = new CPropertyAction(this, &MadTweezer::OnVelocity);
	err = CreateProperty(g_Keyword_Velocity, iToChar, MM::Float, false, pAct);
	if (err != DEVICE_OK)
		return err;

	// Steps
	sprintf(iToChar, "%d", 0);
	pAct = new CPropertyAction(this, &MadTweezer::OnSteps);
	err = CreateProperty(g_Keyword_Steps, iToChar, MM::Integer, false, pAct);
	if (err != DEVICE_OK)
		return err;

	// Milliradians
	sprintf(iToChar, "%f", 0.0);
	pAct = new CPropertyAction(this, &MadTweezer::OnMrad);
	err = CreateProperty(g_Keyword_Milliradians, iToChar, MM::Float, false, pAct);
	if (err != DEVICE_OK)
		return err;

	// Stop
	vector<string> stopList;
	stopList.push_back(" ");
	stopList.push_back(g_Keyword_Stop);
	pAct = new CPropertyAction(this, &MadTweezer::OnStop);
	err = CreateProperty(g_Keyword_Stop, " ", MM::String, false, pAct);
	if (err != DEVICE_OK)
		return err;
	err = SetAllowedValues(g_Keyword_Stop, stopList);

	return DEVICE_OK;
}


int MadTweezer::Shutdown() {
	HandleListLock();

	HandleListType device(handle_, MADTWEEZER_TYPE, axis_, 0);
	HandleListRemoveSingleItem(device);
	if (!HandleExistsOnLockedList(handle_))
	{
		MCL_ReleaseHandle(handle_);
	}
	handle_ = 0;
	initialized_ = false;

	HandleListUnlock();

	return DEVICE_OK;
}


int MadTweezer::GetMode() {
	int err, mode;
	err = MCL_MDGetMode(axis_, &mode, handle_);
	if (err != MCL_SUCCESS)
		return err;
	else
	{
		mode_ = mode;
		return DEVICE_OK;
	}
}


int MadTweezer::GetLocation() {
	int err, microSteps;
	double tempLocation;
	double rotation = max_mrad_;

	err = MCL_MDCurrentPositionM(axis_, &microSteps, handle_);
	if (err != MCL_SUCCESS)
	{
		return err;
	}
	if (mode_ == HIGH_SPEED) {
		tempLocation = microSteps * 0.007853981633974483;
		location_mrad_ = fmod(tempLocation, rotation) * 1000.0;
	}
	else if (mode_ == HIGH_PRECISION) {
		tempLocation = microSteps * 0.0009817477042468104;
		location_mrad_ = fmod(tempLocation, rotation) * 1000.0;
	}

	return DEVICE_OK;
}


int MadTweezer::SetMode(int mode) {
	int err;

	err = MCL_MDSetMode(axis_, mode, handle_);
	if (err != MCL_SUCCESS)
		return err;

	mode_ = mode;
	return DEVICE_OK;
}


int MadTweezer::SetLocation(double mrad) {
	int err;
	GetLocation();
	double initLocation = location_mrad_;
	double distance = 0.0;
	double distanceRad;
	if (direction_ == 1 && mrad > initLocation)
	{
		distance = mrad - initLocation;
	}
	else if (direction_ == 1 && mrad < initLocation)
	{
		distance += (max_mrad_ - initLocation);
		distance += (mrad);
	}
	else if (direction_ == -1 && mrad < initLocation)
	{
		distance = mrad - initLocation;
	}
	else if (direction_ == -1 && mrad > initLocation)
	{
		distance -= initLocation;
		distance -= max_mrad_ - mrad;
	}

	distanceRad = distance / 1000.0;
	if ((fabs(distanceRad) <= stepSize_rad_) || (fabs(6.283185 - distanceRad) <= stepSize_rad_)) {
		err = GetLocation();
		if (err != MCL_SUCCESS)
			return err;
		if (location_mrad_ == 0)
			home_ = true;
		else
			home_ = false;

		return DEVICE_OK;
	}

	err = MCL_MDMove(axis_, velocity_rad_, distanceRad, handle_);
	if (err != MCL_SUCCESS)
	{
		return err;
	}

	err = GetLocation();
	if (err != MCL_SUCCESS)
	{
		return err;
	}

	if (location_mrad_ != 0)
		home_ = false;
	else
		home_ = true;

	return DEVICE_OK;
}


int MadTweezer::UpdateVelocity() {
	int err;
	double tempEncoderRes, tempStepSize, tempMaxVel, ignore1, ignore2, tempMinVel;
	int tempUnits;
	char iToChar[25];
	err = MCL_MDAxisInformation(axis_, &tempEncoderRes, &tempStepSize, &tempMaxVel, &ignore1,
		&ignore2, &tempMinVel, &tempUnits, handle_);
	if (err != MCL_SUCCESS)
		return err;
	if (mode_ == HIGH_SPEED) {
		minVelocity_ = MIN_VEL_HIGH_SPEED;
		maxVelocity_ = MAX_VEL_HIGH_SPEED;
	}
	else if (mode_ == HIGH_PRECISION) {
		minVelocity_ = MIN_VEL_HIGH_PRECISION;
		maxVelocity_ = MAX_VEL_HIGH_PRECISION;
	}

	sprintf(iToChar, "%f", maxVelocity_);
	SetProperty(g_Keyword_Velocity, iToChar);

	return DEVICE_OK;
}


int MadTweezer::UpdateLocation() {
	char iToChar[25];

	GetLocation();
	sprintf(iToChar, "%f", location_mrad_);
	SetProperty(g_Keyword_Location, iToChar);

	return DEVICE_OK;
}


int MadTweezer::OnMode(MM::PropertyBase* pProp, MM::ActionType eAct) {
	int err;
	string mode;

	if (eAct == MM::BeforeGet)
	{
		if (mode_ == HIGH_SPEED)
			pProp->Set(g_Keyword_HighSpeedMode);
		else if (mode_ == HIGH_PRECISION)
			pProp->Set(g_Keyword_HighPrecisionMode);
	}
	else if (eAct == MM::AfterSet)
	{

		pProp->Get(mode);
		if (mode.compare(g_Keyword_HighSpeedMode) == 0)
		{
			err = SetMode(HIGH_SPEED);
			if (err != MCL_SUCCESS)
				return err;
			mode_ = HIGH_SPEED;
			UpdateVelocity();
		}
		if (mode.compare(g_Keyword_HighPrecisionMode) == 0)
		{
			err = SetMode(HIGH_PRECISION);
			if (err != MCL_SUCCESS)
				return err;
			mode_ = HIGH_PRECISION;
			UpdateVelocity();
		}
	}

	return DEVICE_OK;
}


int MadTweezer::OnLocation(MM::PropertyBase* pProp, MM::ActionType eAct) {
	int err;
	double location;

	if (eAct == MM::BeforeGet)
	{
		GetLocation();
		pProp->Set(location_mrad_);
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(location);

		if (location < 0 || location > 6238)
		{
			return INVALID_LOCATION;
		}
		err = SetLocation(location);
		if (err != DEVICE_OK)
			return err;
		location_mrad_ = location;
		if (location != 0) {
			home_ = false;
		}
	}

	return DEVICE_OK;
}


int MadTweezer::OnHome(MM::PropertyBase* pProp, MM::ActionType eAct) {
	int err;
	string input;

	if (eAct == MM::BeforeGet)
	{
		if (home_)
			pProp->Set(g_Listword_Yes);
		else
			pProp->Set(g_Listword_No);
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(input);

		if (input.compare(g_Listword_Yes) == 0)
		{
			err = MCL_MDFindHome(axis_, handle_);
			if (err != MCL_SUCCESS)
				return err;
			home_ = true;
			location_mrad_ = 0;
		}
	}
	return DEVICE_OK;
}


int MadTweezer::OnDirection(MM::PropertyBase* pProp, MM::ActionType eAct) {
	string input;

	// 1 for Clockwise, -1 for Counteclockwise
	if (eAct == MM::BeforeGet)
	{
		if (direction_ == 1)
			pProp->Set(g_Listword_Clockwise);
		else if (direction_ == -1)
			pProp->Set(g_Listword_CounterClockwise);
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(input);

		if (input.compare(g_Listword_Clockwise) == 0)
			direction_ = 1;
		else if (input.compare(g_Listword_CounterClockwise) == 0)
			direction_ = -1;
	}
	return DEVICE_OK;
}


int MadTweezer::OnVelocity(MM::PropertyBase* pProp, MM::ActionType eAct) {
	double vel;

	if (eAct == MM::BeforeGet)
		pProp->Set(velocity_rad_);
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(vel);

		if (vel <  minVelocity_ || vel > maxVelocity_) {
			return INVALID_VELOCITY;
		}

		velocity_rad_ = vel;
	}

	return DEVICE_OK;
}


int MadTweezer::OnRotation(MM::PropertyBase* pProp, MM::ActionType eAct) {
	int err;
	double rotations;
	double radToMove;

	if (eAct == MM::AfterSet)
	{
		pProp->Get(rotations);

		radToMove = rotations * 2 * M_PI;
		err = MCL_MDMove(axis_, velocity_rad_, radToMove * direction_, handle_);
		if (err != MCL_SUCCESS)
			return err;
		//MCL_MicroDriveWait(handle_);
		UpdateLocation();
	}

	return DEVICE_OK;
}


int MadTweezer::OnSteps(MM::PropertyBase* pProp, MM::ActionType eAct) {
	int err;
	long steps;

	if (eAct == MM::AfterSet)
	{
		pProp->Get(steps);

		err = MCL_MDMoveM(axis_, velocity_rad_, steps * direction_, handle_);
		if (err != MCL_SUCCESS)
			return err;
		//MCL_MicroDriveWait(handle_);
		UpdateLocation();
	}
	return DEVICE_OK;
}


int MadTweezer::OnMrad(MM::PropertyBase* pProp, MM::ActionType eAct) {
	int err;
	double mrad;
	double rad;

	if (eAct == MM::AfterSet)
	{
		pProp->Get(mrad);

		rad = mrad / 1000.0;
		err = MCL_MDMoveR(axis_, velocity_rad_, rad * direction_, 0, handle_);
		if (err != MCL_SUCCESS)
			return err;
		//MCL_MicroDriveWait(handle_);
		UpdateLocation();
	}
	return DEVICE_OK;
}


int MadTweezer::OnStop(MM::PropertyBase* pProp, MM::ActionType eAct) {
	int err = 0;
	unsigned short* status = 0;
	string input;
	if (eAct == MM::AfterSet)
	{
		pProp->Get(input);

		if (input.compare(g_Keyword_Stop) == 0)
		{
			err = MCL_MDStop(status, handle_);
			if (err != MCL_SUCCESS)
				return err;
		}
		else if (input.compare(" ") == 0)
			return DEVICE_OK;

	}
	return DEVICE_OK;
}