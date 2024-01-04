/*
File:		MicroDriveZStage.cpp
Copyright:	Mad City Labs Inc., 2023
License:	Distributed under the BSD license.
*/
#include "AcquireDevice.h"
#include "MicroDriveZStage.h"
#include "mdutils.h"

#include <math.h>
#include <string.h>

#include <vector>
using namespace std;

MCL_MicroDrive_ZStage::MCL_MicroDrive_ZStage() :
	handle_(0),
	serialNumber_(0),
	pid_(0),
	axis_(0),
	axisBitmap_(0),
	stepSize_mm_(0.0),
	encoderResolution_(0.0),
	maxVelocity_(0.0),
	minVelocity_(0.0),
	velocity_(0.0),
	initialized_(false),
	encoded_(false),
	lastZ_(0.0),
	iterativeMoves_(false),
	imRetry_(0),
	imToleranceUm_(.250),
	movementDistance_(0.0),
	deviceHasTirfModuleAxis_(false),
	axisIsTirfModule_(false),
	hasUnknownTirfModuleAxis_(false),
	tirfModCalibrationMm_(0.0),
	stopCommanded_(false),
	movementType_(0),
	movementThread_(NULL)
{
	InitializeDefaultErrorMessages();

	//MCL error Messages
	SetErrorText(MCL_GENERAL_ERROR, "MCL Error: General Error");
	SetErrorText(MCL_DEV_ERROR, "MCL Error: Error transferring data to device");
	SetErrorText(MCL_DEV_NOT_ATTACHED, "MCL Error: Device not attached");
	SetErrorText(MCL_USAGE_ERROR, "MCL Error: Using a function from library device does not support");
	SetErrorText(MCL_DEV_NOT_READY, "MCL Error: Device not ready");
	SetErrorText(MCL_ARGUMENT_ERROR, "MCL Error: Argument out of range");
	SetErrorText(MCL_INVALID_AXIS, "MCL Error: Device trying to use unsupported axis");
	SetErrorText(MCL_INVALID_HANDLE, "MCL Error: Handle not valid");

 	// Encoders present?
	CPropertyAction* pAct = new CPropertyAction(this, &MCL_MicroDrive_ZStage::OnEncoded);
	CreateProperty(g_Keyword_Encoded, "Yes", MM::String, false, pAct, true);

	threadStartMutex_ = CreateMutex(NULL, FALSE, NULL);
}


MCL_MicroDrive_ZStage::~MCL_MicroDrive_ZStage()
{
	Shutdown();
}


void MCL_MicroDrive_ZStage::GetName(char* name) const
{
	CDeviceUtils::CopyLimitedString(name, g_StageDeviceName);
}


int MCL_MicroDrive_ZStage::Initialize()
{
	int err = DEVICE_OK;

	HandleListLock();
	err = InitDeviceAdapter();
	HandleListUnlock();

	return err;
}


int MCL_MicroDrive_ZStage::InitDeviceAdapter()
{
	if (initialized_)
		return DEVICE_OK;

	// Attempt to acquire a device/axis for this adapter.
	int ret = MCL_SUCCESS;
	ret = AcquireDeviceHandle(STAGE_TYPE, handle_, axis_);
	if (ret != MCL_SUCCESS)
	{
		return ret;
	}

	// Query device information
	serialNumber_ = MCL_GetSerialNumber(handle_);

	ret = MCL_GetProductID(&pid_, handle_);
	if (ret != MCL_SUCCESS)
		return ret;

	ret = MCL_GetAxisInfo(&axisBitmap_, handle_);
	if (ret != MCL_SUCCESS)
		return ret;

	double ignore1, ignore2;
	ret = MCL_MDInformation(&encoderResolution_, &stepSize_mm_, &maxVelocity_, &ignore1, &ignore2, &minVelocity_, handle_);
	if (ret != MCL_SUCCESS)
		return ret;
	velocity_ = maxVelocity_;

	// Check TIRF mod settings.
	ret = MCL_GetTirfModuleCalibration(&tirfModCalibrationMm_, handle_);
	if (ret == MCL_SUCCESS)
	{
		// If we have a calibration we may also have the assigned tirf module axis.
		int tirfAxis = 0;
		ret = MCL_GetTirfModuleAxis(&tirfAxis, handle_);
		if (ret == MCL_SUCCESS)
		{
			// Check if one of the device adapter axes match the tirf mod axis.
			hasUnknownTirfModuleAxis_ = false;
			if (tirfAxis == axis_)
			{
				deviceHasTirfModuleAxis_ = true;
				axisIsTirfModule_ = true;
			}
		}
		else
		{
			// If the tirf mod axis data is not available use a heuristic to determine if 
			// our device adapter axes are a tirf mod axis.
			hasUnknownTirfModuleAxis_ = true;
			deviceHasTirfModuleAxis_ = true;

			if (IsAxisADefaultTirfModuleAxis(pid_, axisBitmap_, axis_))
			{
				axisIsTirfModule_ = true;
			}
		}
	}

	// Create velocity error text.
	char velErrText[50];
	sprintf(velErrText, "Velocity must be between %f and %f", minVelocity_, maxVelocity_);
	SetErrorText(INVALID_VELOCITY, velErrText);

	// Create Stage properties.
	int err = DEVICE_OK;
	err = CreateZStageProperties();
	if (err != DEVICE_OK)
		return err;

	err = UpdateStatus();
	if (err != DEVICE_OK)
		return err;

	initialized_ = true;

	return err;
}


int MCL_MicroDrive_ZStage::Shutdown()
{
	unsigned short status;
	MCL_MDStop(&status, handle_);
	WaitForSingleObject(movementThread_, INFINITE);

	HandleListLock();

	HandleListType device(handle_, STAGE_TYPE, axis_, 0);
	HandleListRemoveSingleItem(device);
	if(!HandleExistsOnLockedList(handle_))
		MCL_ReleaseHandle(handle_);

	handle_ = 0;
	initialized_ = false;

	HandleListUnlock();

	CloseHandle(threadStartMutex_);

	return DEVICE_OK;
}


bool MCL_MicroDrive_ZStage::Busy()
{
	// Check if the thread is running
	long ret = WaitForSingleObject(movementThread_, 0);
	if (ret == WAIT_TIMEOUT)
	{
		return true;
	}

	return false;
}


double MCL_MicroDrive_ZStage::GetStepSize()
{
	return stepSize_mm_;
}


int MCL_MicroDrive_ZStage::SetPositionUm(double z)
{
	return BeginMovementThread(STANDARD_MOVE_TYPE, z / 1000.0);
}

	
int MCL_MicroDrive_ZStage::GetPositionUm(double& z)
{
	int err = DEVICE_OK;

	// Check if the thread is running
	long ret = WaitForSingleObject(movementThread_, 0);
	if (ret == WAIT_TIMEOUT)
	{
		z = lastZ_;
	}
	else
	{
		err = GetPositionMm(z);
	}
	z *= 1000.0;

	return err;
}


int MCL_MicroDrive_ZStage::SetRelativePositionUm(double z)
{
	double absZ;
	int err = ConvertRelativeToAbsoluteMm(z / 1000.0, absZ);
	if (err != MCL_SUCCESS)
		return err;

	err = BeginMovementThread(STANDARD_MOVE_TYPE, absZ);

	return err;
}


int MCL_MicroDrive_ZStage::SetPositionSteps(long z)
{
	return BeginMovementThread(STANDARD_MOVE_TYPE, z* stepSize_mm_);
}


int MCL_MicroDrive_ZStage::GetPositionSteps(long& z)
{
	int err = DEVICE_OK;
	double getZ;

	// Check if the thread is running
	long ret = WaitForSingleObject(movementThread_, 0);
	if (ret == WAIT_TIMEOUT)
		getZ = lastZ_;
	else
		err = GetPositionMm(getZ);

	z = (long) (getZ / stepSize_mm_);

	return err;
}


int MCL_MicroDrive_ZStage::SetOrigin()
{
	long ret = WaitForSingleObject(movementThread_, 0);
	if (ret == WAIT_TIMEOUT)
	{
		return MCL_DEV_NOT_READY;
	}

	return SetOriginSync();
}


int MCL_MicroDrive_ZStage::SetOriginSync()
{
	int err = MCL_SUCCESS;
	if (encoded_ && axis_ < M5AXIS)
	{
		err = MCL_MDResetEncoder(axis_, NULL, handle_);
		if (err != MCL_SUCCESS)
			return err;
	}
	lastZ_ = 0;

	return DEVICE_OK;
}


int MCL_MicroDrive_ZStage::FindEpiSync()
{
	if (axisIsTirfModule_ == false)
		return DEVICE_OK;

	int err = MCL_SUCCESS;
	unsigned short status;
	unsigned short mask = LimitBitMask(pid_, axis_, REVERSE);

	MCL_MDStatus(&status, handle_);

	// Move the stage to its reverse limit.
	while ((status & mask) == mask && !stopCommanded_)
	{
		err = SetRelativePositionMmSync(-.5);
		if (err != DEVICE_OK)
			return err;
		MCL_MDStatus(&status, handle_);
	}

	// Set the orgin at the reverse limit.
	SetOriginSync();

	// Move the calibration distance to find epi.
	err = SetPositionMmSync(tirfModCalibrationMm_);
	if (err != DEVICE_OK)
		return err;

	// Set the orgin at epi.
	SetOriginSync();

	return DEVICE_OK;
}


int MCL_MicroDrive_ZStage::GetLimits(double& /*lower*/, double& /*upper*/)
{
	return DEVICE_UNSUPPORTED_COMMAND;
}


int MCL_MicroDrive_ZStage::IsStageSequenceable(bool& isSequenceable) const
{
	isSequenceable = false;
	return DEVICE_OK;
}


bool MCL_MicroDrive_ZStage::IsContinuousFocusDrive() const
{
	return false;
}


int MCL_MicroDrive_ZStage::OnPositionMm(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	int err;
	double pos;
	if (eAct == MM::BeforeGet)
	{
		double z;
		GetPositionMm(z);
		pProp->Set(z);
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(pos);
		
		double z;
		err = GetPositionMm(z);
		if(err != MCL_SUCCESS)
			return err;
		err = BeginMovementThread(STANDARD_MOVE_TYPE, z);		
		if (err != DEVICE_OK)
			return err;
	}
		
	return DEVICE_OK;
}

int MCL_MicroDrive_ZStage::OnMovemm(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	int err;
	double pos;
	if (eAct == MM::AfterSet)
	{
		pProp->Get(pos);

		err = SetRelativePositionUm(pos * 1000.0);
		if (err != MCL_SUCCESS)
			return err;
	}

	return DEVICE_OK;
}	 
	
int MCL_MicroDrive_ZStage::OnSetOrigin(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	int err;
	string message;

	if (eAct == MM::BeforeGet)
	{
		pProp->Set(g_Listword_No);
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(message);

		if (message.compare(g_Listword_Yes) == 0)
		{
			err = SetOrigin();
			if (err != DEVICE_OK)
			{
				return err;
			}
		}
	} 

	return DEVICE_OK;
}

	
int MCL_MicroDrive_ZStage::OnCalibrate(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	int err;
	string message;

	if (eAct == MM::BeforeGet)
	{
		pProp->Set(g_Listword_No);
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(message);

		if (message.compare(g_Listword_Yes) == 0)
		{
			err = BeginMovementThread(CALIBRATE_TYPE, 0.0);
			if (err != DEVICE_OK)
				return err;
		}
	}

	return DEVICE_OK;
}


int MCL_MicroDrive_ZStage::OnReturnToOrigin(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	int err;
	string message;

	if (eAct == MM::BeforeGet)
	{
		pProp->Set(g_Listword_No);
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(message);

		if (message.compare(g_Listword_Yes) == 0)
		{
			err = BeginMovementThread(RETURN_TO_ORIGIN_TYPE, 0);
			if (err != DEVICE_OK)
				return err;
		}
	}

	return DEVICE_OK;
}


int MCL_MicroDrive_ZStage::OnVelocity(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	double vel;

	if (eAct == MM::BeforeGet)
	{
		pProp->Set(velocity_);
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(vel);

		if (vel <  minVelocity_ || vel > maxVelocity_){
			return INVALID_VELOCITY;
		}
		
		velocity_ = vel;
	}
	
	return DEVICE_OK;
}


int MCL_MicroDrive_ZStage::OnEncoded(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(encoded_ ? g_Listword_Yes : g_Listword_No);
	}
	else if (eAct == MM::AfterSet)
	{
   	string message;
		pProp->Get(message);
		encoded_ = (message.compare(g_Listword_Yes) == 0);
	}

	return DEVICE_OK;
}


int MCL_MicroDrive_ZStage::OnIterativeMove(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
	    pProp->Set(iterativeMoves_ ? g_Listword_Yes : g_Listword_No);
	}
	else if (eAct == MM::AfterSet)
	{
	   	string message;
		pProp->Get(message);
		iterativeMoves_ = (message.compare(g_Listword_Yes) == 0);
	}

	return DEVICE_OK;
}


int MCL_MicroDrive_ZStage::OnImRetry(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set((long)imRetry_);
	}
	else if (eAct == MM::AfterSet)
	{
		long retries = 0;
		pProp->Get(retries);
		imRetry_ = retries;
	}

	return DEVICE_OK;
}


int MCL_MicroDrive_ZStage::OnImToleranceUm(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(imToleranceUm_);
	}
	else if (eAct == MM::AfterSet)
	{
		double tol;
		pProp->Get(tol);
		imToleranceUm_ = tol;
	}

	return DEVICE_OK;
}


int MCL_MicroDrive_ZStage::OnIsTirfModule(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(axisIsTirfModule_ ? g_Listword_Yes : g_Listword_No);
	}
	else if (eAct == MM::AfterSet)
	{
		string message;
		pProp->Get(message);
		axisIsTirfModule_ = (message.compare(g_Listword_Yes) == 0);
	}
	return DEVICE_OK;
}


int MCL_MicroDrive_ZStage::OnFindEpi(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	int err;
	string message;

	if (eAct == MM::BeforeGet)
	{
		pProp->Set(g_Listword_No);
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(message);

		if (message.compare(g_Listword_Yes) == 0)
		{ 
			err = BeginMovementThread(FIND_EPI_TYPE, 0.0);
			if (err != DEVICE_OK)
				return err;
		}
	}
	return DEVICE_OK;
}


int MCL_MicroDrive_ZStage::OnStop(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	int err = DEVICE_OK;
	string input;
	if (eAct == MM::AfterSet)
	{
		pProp->Get(input);

		if (input.compare(g_Keyword_Stop) == 0)
		{
			err = Stop();
		}

	}
	return DEVICE_OK;
}


int MCL_MicroDrive_ZStage::CreateZStageProperties()
{
	int err;
	char iToChar[25];

	vector<string> yesNoList;
	yesNoList.push_back(g_Listword_No);
	yesNoList.push_back(g_Listword_Yes);

	/// Read only properties
		
	// Name property
	err = CreateProperty(MM::g_Keyword_Name, g_StageDeviceName, MM::String, true);
	if (err != DEVICE_OK)
		return err;

	// Description property
	err = CreateProperty(MM::g_Keyword_Description, "Stage Driver", MM::String, true);
	if (err != DEVICE_OK)
		return err;

	// Device handle
	sprintf(iToChar, "%d", handle_);
	err = CreateProperty("Handle", iToChar, MM::String, true);
	if (err != DEVICE_OK)
		return err;

	// Product ID
	sprintf(iToChar, "%hu", pid_);
	err = CreateProperty("Product ID", iToChar, MM::String, true);
	if (err != DEVICE_OK)
		return err;

	// Serial Number
	sprintf(iToChar, "%d", serialNumber_);
	err = CreateProperty("Serial number", iToChar, MM::String, true);
	if (err != DEVICE_OK)
		return err;

	// Maximum velocity
	sprintf(iToChar, "%f", maxVelocity_);
	err = CreateProperty("Maximum velocity (mm/s)", iToChar, MM::Float, true);
	if (err != DEVICE_OK)
		return err;

	// Minimum velocity
	sprintf(iToChar, "%f", minVelocity_);
	err = CreateProperty("Minimum velocity (mm/s)", iToChar, MM::Float, true);
	if (err != DEVICE_OK)
		return err;

	if (deviceHasTirfModuleAxis_)
	{
		sprintf(iToChar, "%f", tirfModCalibrationMm_);
		err = CreateProperty("Distance to epi", iToChar, MM::Float, true);
		if (err != DEVICE_OK)
			return err;
	}

	// Wait Time
	err = CreateProperty(g_Keyword_WaitTime, " ", MM::Float, false);
	if (err != DEVICE_OK)
		return err;

	///// Action properties

	// Change velocity
	sprintf(iToChar, "%f", maxVelocity_);
	CPropertyAction* pAct = new CPropertyAction(this, &MCL_MicroDrive_ZStage::OnVelocity);
	err = CreateProperty("Velocity", iToChar, MM::Float, false, pAct);
	if (err != DEVICE_OK)
		return err;

	// Change position (mm)
	pAct = new CPropertyAction(this, &MCL_MicroDrive_ZStage::OnPositionMm);
	err = CreateProperty(g_Keyword_SetPosZmm, "0", MM::Float, false, pAct);
	if (err != DEVICE_OK)
		return err;

	// Move position relative (mm)
	pAct = new CPropertyAction(this, &MCL_MicroDrive_ZStage::OnMovemm);
	err = CreateProperty(g_Keyword_SetRelativePosZmm, "0", MM::Float, false, pAct);
	if (err != DEVICE_OK)
		return err;

	// Set origin at current position (reset encoders)
	pAct = new CPropertyAction(this, &MCL_MicroDrive_ZStage::OnSetOrigin);
	err = CreateProperty(g_Keyword_SetOriginHere, "No", MM::String, false, pAct);
	if (err != DEVICE_OK)
		return err;
	err = SetAllowedValues(g_Keyword_SetOriginHere, yesNoList);
	if (err != DEVICE_OK)
		return err;

	// Calibrate
	pAct = new CPropertyAction(this, &MCL_MicroDrive_ZStage::OnCalibrate);
	err = CreateProperty(g_Keyword_Calibrate, "No", MM::String, false, pAct);
	if (err != DEVICE_OK)
		return err;
	err = SetAllowedValues(g_Keyword_Calibrate, yesNoList);
	if (err != DEVICE_OK)
		return err;

	// Stop
	vector<string> stopList;
	stopList.push_back(" ");
	stopList.push_back(g_Keyword_Stop);
	pAct = new CPropertyAction(this, &MCL_MicroDrive_ZStage::OnStop);
	err = CreateProperty(g_Keyword_Stop, " ", MM::String, false, pAct);
	if (err != DEVICE_OK)
		return err;
	err = SetAllowedValues(g_Keyword_Stop, stopList);
	if (err != DEVICE_OK)
		return err;

	// Return to origin
	pAct = new CPropertyAction(this, &MCL_MicroDrive_ZStage::OnReturnToOrigin);
	err = CreateProperty(g_Keyword_ReturnToOrigin, "No", MM::String, false, pAct);
	if (err != DEVICE_OK)
		return err;
	err = SetAllowedValues(g_Keyword_ReturnToOrigin, yesNoList);
	if (err != DEVICE_OK)
		return err;

	// Iterative Moves
	pAct = new CPropertyAction(this, &MCL_MicroDrive_ZStage::OnIterativeMove);
	err = CreateProperty(g_Keyword_IterativeMove, "No", MM::String, false, pAct);
	if (err != DEVICE_OK)
		return err;
	err = SetAllowedValues(g_Keyword_IterativeMove, yesNoList);
	if (err != DEVICE_OK)
		return err;

	// Iterative Retries
	pAct = new CPropertyAction(this, &MCL_MicroDrive_ZStage::OnImRetry);
	err = CreateProperty(g_Keyword_ImRetry, "0", MM::Integer, false, pAct);
	if (err != DEVICE_OK)
		return err;

	// Iterative Tolerance
	sprintf(iToChar, "%f", imToleranceUm_);
	pAct = new CPropertyAction(this, &MCL_MicroDrive_ZStage::OnImToleranceUm);
	err = CreateProperty(g_Keyword_ImTolerance, iToChar, MM::Float, false, pAct);
	if (err != DEVICE_OK)
		return err;


	if (deviceHasTirfModuleAxis_)
	{
		// Axis is tirfModule
		pAct = new CPropertyAction(this, &MCL_MicroDrive_ZStage::OnIsTirfModule);
		err = CreateProperty(
				g_Keyword_IsTirfModuleAxis,
				axisIsTirfModule_ ? g_Listword_Yes : g_Listword_No,
				MM::String, 
				hasUnknownTirfModuleAxis_ ? false : true,
				pAct);
		if (err != DEVICE_OK)
			return err;
		err = SetAllowedValues(g_Keyword_IsTirfModuleAxis, yesNoList);
		if (err != DEVICE_OK)
			return err;

		// Find Epi
		pAct = new CPropertyAction(this, &MCL_MicroDrive_ZStage::OnFindEpi);
		err = CreateProperty(g_Keyword_FindEpi, "No", MM::String, false, pAct);
		if (err != DEVICE_OK)
			return err;
		err = SetAllowedValues(g_Keyword_FindEpi, yesNoList);
		if (err != DEVICE_OK)
			return err;
	}

	return DEVICE_OK;
}


int MCL_MicroDrive_ZStage::SetPositionMmSync(double goalZ)
{
	int err, waitTime;
	char iToChar[10];
	int currentRetries = 0;
	bool moveFinished = false;

	if (stopCommanded_)
		return DEVICE_OK;

	//Calculate the absolute position.
	double zCurrent;
	err = GetPositionMm(zCurrent);
	if (err != MCL_SUCCESS)
		return err;

	do 
	{
		double zMove = goalZ - zCurrent;
		int startingMicroSteps = 0;
		int endingMicroSteps = 0;
		err = MCL_MDCurrentPositionM(axis_, &startingMicroSteps, handle_);
		if (err != MCL_SUCCESS)
			return err;

		// Verify we are moving at least a step.
		bool noMovement = fabs(zMove) < stepSize_mm_;
		if (noMovement)
			return MCL_SUCCESS;

		err = MCL_MDMove(axis_, velocity_, zMove, handle_);
		if (err != MCL_SUCCESS)
				return err;

		err = MCL_MicroDriveGetWaitTime(&waitTime, handle_);
		if (err != MCL_SUCCESS)
			return err;
		sprintf(iToChar, "%d", waitTime);
		SetProperty(g_Keyword_WaitTime, iToChar);

		PauseDevice();

		err = MCL_MDCurrentPositionM(axis_, &endingMicroSteps, handle_);
		if (err != MCL_SUCCESS)
			return err;

		lastZ_ += (endingMicroSteps - startingMicroSteps) * stepSize_mm_;

		// Update current position
		err = GetPositionMm(zCurrent);
		if (err != MCL_SUCCESS)
			return err;
		Sleep(50);
		err = GetPositionMm(zCurrent);
		if (err != MCL_SUCCESS)
			return err;

		if(iterativeMoves_ && encoded_ && !stopCommanded_)
		{
			double absDiffUmZ = abs(goalZ - zCurrent) * 1000.0; 
			bool zInTolerance = absDiffUmZ < imToleranceUm_;
			if(zInTolerance)
			{
				moveFinished = true;
			}
			else 
			{
				currentRetries++;
				if(currentRetries <= imRetry_)
				{
					moveFinished = false;
				}
				else
				{
					moveFinished = true;
				}
			}
		}
		else
		{
			moveFinished = true;
		}
	} while( !moveFinished );

	return DEVICE_OK;
}


int MCL_MicroDrive_ZStage::GetPositionMm(double& z)
{
	if (encoded_ && (axis_ < M5AXIS))
	{
		double tempM1, tempM2, tempM3, tempM4;
		int err = MCL_MDReadEncoders(&tempM1, &tempM2, &tempM3, &tempM4, handle_);
		if (err != MCL_SUCCESS)
			return err;
		switch (axis_)
		{
			case M1AXIS:
				z = tempM1;
				break;
			case M2AXIS:
				z = tempM2;
				break;
			case M3AXIS:
				z = tempM3;
				break;
			case M4AXIS:
				z = tempM4;
				break;
		}
	}
	else
	{
		z = lastZ_;
	}
	return DEVICE_OK;
}


int MCL_MicroDrive_ZStage::ConvertRelativeToAbsoluteMm(double relZ, double& absZ)
{
	int err;

	//Calculate the absolute position.
	double zCurrent;
	err = GetPositionMm(zCurrent);
	if (err != MCL_SUCCESS)
		return err;

	absZ = zCurrent + relZ;

	return MCL_SUCCESS;
}


int MCL_MicroDrive_ZStage::SetRelativePositionMmSync(double z)
{
	double absZ;
	int err = ConvertRelativeToAbsoluteMm(z, absZ);
	if (err != MCL_SUCCESS)
		return err;

	err = SetPositionMmSync(absZ);

	return err;
}


int MCL_MicroDrive_ZStage::CalibrateSync()
{
	int err;
	double zPosOrig;
	double zPosLimit;

	err = GetPositionMm(zPosOrig);
	if (err != MCL_SUCCESS)
		return err;

	err = MoveToForwardLimitSync();
	if (err != DEVICE_OK)
		return err;

	err = GetPositionMm(zPosLimit);
	if (err != MCL_SUCCESS)
		return err;

	err = SetOriginSync();
	if (err != DEVICE_OK)
		return err;

	err = SetPositionMmSync((zPosOrig - zPosLimit));
	if (err != DEVICE_OK)
		return err;

	return DEVICE_OK;
}
	
	
int MCL_MicroDrive_ZStage::MoveToForwardLimitSync()
{
	int err;
	unsigned short status = 0;

	err = MCL_MDStatus(&status, handle_);
	if(err != MCL_SUCCESS)	
		return err;

	unsigned short bitMask = LimitBitMask(pid_, axis_, FORWARD);
	while ((status & bitMask) != 0 && !stopCommanded_)
	{ 
		err = SetRelativePositionUm(4000);
		if (err != DEVICE_OK)
			return err;

		err = MCL_MDStatus(&status, handle_);
		if (err != MCL_SUCCESS)	
			return err;
	}
	return DEVICE_OK;
}


int MCL_MicroDrive_ZStage::ReturnToOriginSync()
{
	return SetPositionMmSync(0.0);
}


void MCL_MicroDrive_ZStage::PauseDevice()
{
	MCL_MicroDriveWait(handle_);
}


int MCL_MicroDrive_ZStage::Stop()
{
	int err = MCL_MDStop(NULL, handle_);
	stopCommanded_ = true;
	if (err != MCL_SUCCESS)
		return err;

	return DEVICE_OK;
}


int MCL_MicroDrive_ZStage::BeginMovementThread(int type, double distance)
{
	long ret = WaitForSingleObject(threadStartMutex_, 0);
	if (ret == WAIT_TIMEOUT)
	{
		return MCL_DEV_NOT_READY;
	}

	// Check if the thread is running
	ret = WaitForSingleObject(movementThread_, 0);
	if (ret == WAIT_TIMEOUT)
	{
		ReleaseMutex(threadStartMutex_);
		return MCL_DEV_NOT_READY;
	}

	// Create a new thread for our action
	movementType_ = type;
	movementDistance_ = distance;
	movementThread_ = CreateThread(0, 0, ExecuteMovement, this, 0, 0);

	ReleaseMutex(threadStartMutex_);

	return DEVICE_OK;
}


DWORD WINAPI MCL_MicroDrive_ZStage::ExecuteMovement(LPVOID lpParam) {

	MCL_MicroDrive_ZStage* instance = reinterpret_cast<MCL_MicroDrive_ZStage*>(lpParam);
	instance->stopCommanded_ = false;
	switch (instance->movementType_)
	{
	case STANDARD_MOVE_TYPE:
		instance->SetPositionMmSync(instance->movementDistance_);
		break;
	case CALIBRATE_TYPE:
		instance->CalibrateSync();
		break;
	case HOME_TYPE:
		instance->MoveToForwardLimitSync();
		break;
	case RETURN_TO_ORIGIN_TYPE:
		instance->ReturnToOriginSync();
		break;
	case FIND_EPI_TYPE:
		instance->FindEpiSync();
		break;
	}

	return 0;
}