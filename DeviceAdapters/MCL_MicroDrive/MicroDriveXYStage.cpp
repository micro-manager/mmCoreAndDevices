/*
File:		MicroDriveXYStage.cpp
Copyright:	Mad City Labs Inc., 2023
License:	Distributed under the BSD license.
*/
#include "AcquireDevice.h"
#include "MicroDriveXYStage.h"
#include "mdutils.h"

#include <math.h>
#include <string.h>
#include<windows.h>

#include <vector>
using namespace std;

MicroDriveXYStage::MicroDriveXYStage() :
	handle_(0),
	serialNumber_(0),
	pid_(0),
	axis1_(0),
	axis2_(0),
	axisBitmap_(0),
	stepSize_mm_(0.0),
	encoderResolution_(0.0),
	maxVelocity_(0.0),
	maxVelocityThreeAxis_(0),
	maxVelocityTwoAxis_(0),
	minVelocity_(0.0),
	velocity_(0.0),
	initialized_(false),
	encoded_(false),
	lastX_(0.0),
	lastY_(0.0),
	iterativeMoves_(false),
	imRetry_(0),
	imToleranceUm_(.250),
	movementDistanceX_(0.0),
	movementDistanceY_(0.0),
	movementType_(0),
	deviceHasTirfModuleAxis_(false),
	axis1IsTirfModule_(false),
	axis2IsTirfModule_(false),
	hasUnknownTirfModuleAxis_(false),
	tirfModCalibrationMm_(0.0),
	stopCommanded_(false),
	movementThread_(NULL)
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

	// Encoders present?
	CPropertyAction* pAct = new CPropertyAction(this, &MicroDriveXYStage::OnEncoded);
	CreateProperty(g_Keyword_Encoded, "Yes", MM::String, false, pAct, true);

	threadStartMutex_ = CreateMutex(NULL, FALSE, NULL);
}


MicroDriveXYStage::~MicroDriveXYStage()
{
	Shutdown();
}


bool MicroDriveXYStage::Busy()
{
	// Check if the thread is running
	long ret = WaitForSingleObject(movementThread_, 0);
	if (ret == WAIT_TIMEOUT)
	{
		return true;
	}

	return false;
}


void MicroDriveXYStage::GetName(char* pszName) const
{
   CDeviceUtils::CopyLimitedString(pszName, g_XYStageDeviceName);
}


int MicroDriveXYStage::Initialize()
{
	int err = DEVICE_OK;

	HandleListLock();
	err = InitDeviceAdapter();
	HandleListUnlock();

	return err;
}


int MicroDriveXYStage::InitDeviceAdapter()
{
	if (initialized_)
		return DEVICE_OK;

	// Attempt to acquire a device/axis for this adapter.
	int ret = MCL_SUCCESS;
	ret = AcquireDeviceHandle(XYSTAGE_TYPE, handle_, axis1_);
	axis2_ = axis1_ + 1;
	if (ret != MCL_SUCCESS)
		return ret;

	// Query device information
	serialNumber_ = MCL_GetSerialNumber(handle_);

	ret = MCL_GetProductID(&pid_, handle_);
	if (ret != MCL_SUCCESS)
		return ret;

	ret = MCL_GetAxisInfo(&axisBitmap_, handle_);
	if (ret != MCL_SUCCESS)
		return ret;

	ret = MCL_MDInformation(&encoderResolution_, &stepSize_mm_, &maxVelocity_, &maxVelocityTwoAxis_, &maxVelocityThreeAxis_, &minVelocity_, handle_);
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
			if (tirfAxis == axis1_)
			{
				deviceHasTirfModuleAxis_ = true;
				axis1IsTirfModule_ = true;
			}
			else if (tirfAxis == (axis2_))
			{
				deviceHasTirfModuleAxis_ = true;
				axis2IsTirfModule_ = true;
			}
		}
		else
		{
			// If the tirf mod axis data is not available use a heuristic to determine if 
			// our device adapter axes are a tirf mod axis.
			hasUnknownTirfModuleAxis_ = true;
			deviceHasTirfModuleAxis_ = true;
			if (IsAxisADefaultTirfModuleAxis(pid_, axisBitmap_, axis1_))
			{
				axis1IsTirfModule_ = true;
			}
			if (IsAxisADefaultTirfModuleAxis(pid_, axisBitmap_, axis2_))
			{
				axis2IsTirfModule_ = true;
			}
		}
	}

	// Create velocity error text.
	char velErrText[50];
	snprintf(velErrText, sizeof(velErrText), "Velocity must be between %f and %f", minVelocity_, maxVelocity_);
	SetErrorText(INVALID_VELOCITY, velErrText);

	// Create Stage properties.
	int err = DEVICE_OK;
	err = CreateMicroDriveXYProperties();
	if (err != DEVICE_OK)
		return err;

	err = UpdateStatus();
	if (err != DEVICE_OK)
	{
		return err;
	}

	initialized_ = true;
	
	return err;
}


int MicroDriveXYStage::CreateMicroDriveXYProperties()
{
	int err;
	char iToChar[25];

	vector<string> yesNoList;
    yesNoList.push_back(g_Listword_No);
    yesNoList.push_back(g_Listword_Yes);

	/// Read only properties
	
	// Name property
	err = CreateProperty(MM::g_Keyword_Name, g_XYStageDeviceName, MM::String, true);
	if (err != DEVICE_OK)
		return err;

	// Description property
	err = CreateProperty(MM::g_Keyword_Description, "XY Stage Driver", MM::String, true);
	if (err != DEVICE_OK)
		return err;

	// Device handle
	snprintf(iToChar, sizeof(iToChar), "%d", handle_);
	err = CreateProperty(g_Keyword_Handle, iToChar, MM::String, true);
	if (err != DEVICE_OK)
		return err;

	// Serial Number
	snprintf(iToChar, sizeof(iToChar), "%hu", pid_);
	err = CreateProperty(g_Keyword_Serial_Num, iToChar, MM::String, true);
	if (err != DEVICE_OK)
		return err;

	// Product ID
	snprintf(iToChar, sizeof(iToChar), "%hu", pid_);
	err = CreateProperty(g_Keyword_ProductID, iToChar, MM::String, true);
	if (err != DEVICE_OK)
		return err;

	// Maximum velocity
	snprintf(iToChar, sizeof(iToChar), "%f", maxVelocity_);
	err = CreateProperty(g_Keyword_MaxVelocity, iToChar, MM::Float, true);
	if (err != DEVICE_OK)
		return err;

	// Minumum velocity
	snprintf(iToChar, sizeof(iToChar), "%f", minVelocity_);
	err = CreateProperty(g_Keyword_MinVelocity, iToChar, MM::Float, true);
	if (err != DEVICE_OK)
		return err;

	// Wait Time
	err = CreateProperty(g_Keyword_WaitTime, " ", MM::Float, false);
	if (err != DEVICE_OK)
		return err;

	if (deviceHasTirfModuleAxis_)
	{
		snprintf(iToChar, sizeof(iToChar), "%f", tirfModCalibrationMm_);
		err = CreateProperty(g_Keyword_DistanceToEpi, iToChar, MM::Float, true);
		if (err != DEVICE_OK)
			return err;
	}

	/// Action properties

	// Change velocity
	snprintf(iToChar, sizeof(iToChar), "%f", maxVelocity_);
	CPropertyAction* pAct = new CPropertyAction(this, &MicroDriveXYStage::OnVelocity);
	err = CreateProperty("Velocity", iToChar, MM::Float, false, pAct);
	if (err != DEVICE_OK)
		return err;

	// Change X position (mm)
	pAct = new CPropertyAction(this, &MicroDriveXYStage::OnPositionXmm);
	err = CreateProperty(g_Keyword_SetPosXmm, "0", MM::Float, false, pAct);
	if (err != DEVICE_OK)
		return err;

	// Change Y position (mm)
	pAct = new CPropertyAction(this, &MicroDriveXYStage::OnPositionYmm);
	err = CreateProperty(g_Keyword_SetPosYmm, "0", MM::Float, false, pAct);
	if (err != DEVICE_OK)
		return err;

	// Change X position (mm)
	pAct = new CPropertyAction(this, &MicroDriveXYStage::OnMoveXmm);
	err = CreateProperty(g_Keyword_SetRelativePosXmm, "0", MM::Float, false, pAct);
	if (err != DEVICE_OK)
		return err;

	// Change Y position (mm)
	pAct = new CPropertyAction(this, &MicroDriveXYStage::OnMoveYmm);
	err = CreateProperty(g_Keyword_SetRelativePosYmm, "0", MM::Float, false, pAct);
	if (err != DEVICE_OK)
		return err;

	// Set origin at current position (reset encoders)
	pAct = new CPropertyAction(this, &MicroDriveXYStage::OnSetOriginHere);
	err = CreateProperty(g_Keyword_SetOriginHere, "No", MM::String, false, pAct);
	if (err != DEVICE_OK)
		return err;
    err = SetAllowedValues(g_Keyword_SetOriginHere, yesNoList);
	if (err != DEVICE_OK)
		return err;

	// Calibrate
	pAct = new CPropertyAction(this, &MicroDriveXYStage::OnCalibrate);
	err = CreateProperty(g_Keyword_Calibrate, "No", MM::String, false, pAct);
	if (err != DEVICE_OK)
		return err;
	err = SetAllowedValues(g_Keyword_Calibrate, yesNoList);
	if (err != DEVICE_OK)
		return err;

	//Stop
	vector<string> stopList;
	stopList.push_back(" ");
	stopList.push_back(g_Keyword_Stop);
	pAct = new CPropertyAction(this, &MicroDriveXYStage::OnStop);
	err = CreateProperty(g_Keyword_Stop, " ", MM::String, false, pAct);
	if (err != DEVICE_OK)
		return err;
	err = SetAllowedValues(g_Keyword_Stop, stopList);
	if (err != DEVICE_OK)
		return err;

	// Return to origin
	pAct = new CPropertyAction(this, &MicroDriveXYStage::OnReturnToOrigin);
	err = CreateProperty(g_Keyword_ReturnToOrigin, "No", MM::String, false, pAct);
	if (err != DEVICE_OK)
		return err;
	err = SetAllowedValues(g_Keyword_ReturnToOrigin, yesNoList);
	if (err != DEVICE_OK)
		return err;

	// Iterative Moves
	pAct = new CPropertyAction(this, &MicroDriveXYStage::OnIterativeMove);
	err = CreateProperty(g_Keyword_IterativeMove, "No", MM::String, false, pAct);
	if (err != DEVICE_OK)
		return err;
	err = SetAllowedValues(g_Keyword_IterativeMove, yesNoList);
	if (err != DEVICE_OK)
		return err;

	// Iterative Retries
	pAct = new CPropertyAction(this, &MicroDriveXYStage::OnImRetry);
	err = CreateProperty(g_Keyword_ImRetry, "0", MM::Integer, false, pAct);
	if (err != DEVICE_OK)
		return err;

	// Iterative Tolerance
	snprintf(iToChar, sizeof(iToChar), "%f", imToleranceUm_);
	pAct = new CPropertyAction(this, &MicroDriveXYStage::OnImToleranceUm);
	err = CreateProperty(g_Keyword_ImTolerance, iToChar, MM::Float, false, pAct);
	if (err != DEVICE_OK)
		return err;

	if (deviceHasTirfModuleAxis_)
	{
		// Axis is tirfModule
		pAct = new CPropertyAction(this, &MicroDriveXYStage::OnIsTirfModuleAxis1);
		err = CreateProperty(
				g_Keyword_IsTirfModuleAxis1,
				axis1IsTirfModule_ ? g_Listword_Yes : g_Listword_No,
				MM::String,
				hasUnknownTirfModuleAxis_ ? false : true,
				pAct);
		if (err != DEVICE_OK)
			return err;
		err = SetAllowedValues(g_Keyword_IsTirfModuleAxis1, yesNoList);
		if (err != DEVICE_OK)
			return err;

		pAct = new CPropertyAction(this, &MicroDriveXYStage::OnIsTirfModuleAxis2);
		err = CreateProperty(
				g_Keyword_IsTirfModuleAxis2,
				axis2IsTirfModule_ ? g_Listword_Yes : g_Listword_No,
				MM::String,
				hasUnknownTirfModuleAxis_ ? false : true,
				pAct);
		if (err != DEVICE_OK)
			return err;
		err = SetAllowedValues(g_Keyword_IsTirfModuleAxis2, yesNoList);
		if (err != DEVICE_OK)
			return err;

		// Find Epi
		pAct = new CPropertyAction(this, &MicroDriveXYStage::OnFindEpi);
		err = CreateProperty(g_Keyword_FindEpi, "No", MM::String, false, pAct);
		if (err != DEVICE_OK)
			return err;
		err = SetAllowedValues(g_Keyword_FindEpi, yesNoList);
		if (err != DEVICE_OK)
			return err;
	}

	return DEVICE_OK;
}


int MicroDriveXYStage::Shutdown()
{
	unsigned short status;
	MCL_MDStop(&status, handle_);
	WaitForSingleObject(movementThread_, INFINITE);

	HandleListLock();

	HandleListType device(handle_, XYSTAGE_TYPE, axis1_, axis2_);
	HandleListRemoveSingleItem(device);
	if (!HandleExistsOnLockedList(handle_))
	{
		MCL_ReleaseHandle(handle_);
	}
	handle_ = 0;
	initialized_ = false;

	HandleListUnlock();

	CloseHandle(threadStartMutex_);

	return DEVICE_OK;
}


int MicroDriveXYStage::SetPositionUm(double x, double y)
{
	return BeginMovementThread(STANDARD_MOVE_TYPE, x / 1000.0, y / 1000.0);
}


int MicroDriveXYStage::GetPositionUm(double& x, double& y)
{
	int err = DEVICE_OK;

	// Check if the thread is running
	long ret = WaitForSingleObject(movementThread_, 0);
	if (ret == WAIT_TIMEOUT)
	{
		// If we are moving simply return the last know position.
		x = lastX_;
		y = lastY_;

	}
	else {
		err = GetPositionMm(x, y);
	}

	x *= 1000.0;
	y *= 1000.0;

	return err;
}


int MicroDriveXYStage::SetPositionMmSync(double goalX, double goalY)
{
	int err;
	int currentRetries = 0;
	int waitTime;
	bool moveFinished = false;
	char iToChar[10];

	if (stopCommanded_)
		return DEVICE_OK;

	//Calculate the absolute position.
	double xCurrent, yCurrent;
	err = GetPositionMm(xCurrent, yCurrent);
	if (err != MCL_SUCCESS)
		return err;

	do 
	{
		double xMove = goalX - xCurrent;
		double yMove = goalY - yCurrent;
		int startingMicroStepsX = 0;
		int endingMicroStepsX = 0;
		int startingMicroStepsY = 0;
		int endingMicroStepsY = 0;
		err = MCL_MDCurrentPositionM(axis1_, &startingMicroStepsX, handle_);
		if (err != MCL_SUCCESS)
			return err;
		err = MCL_MDCurrentPositionM(axis2_, &startingMicroStepsY, handle_);
		if (err != MCL_SUCCESS)
			return err;

		bool noXMovement = (fabs(xMove) < stepSize_mm_); 
		bool noYMovement = (fabs(yMove) < stepSize_mm_);
		if (noXMovement && noYMovement)
		{
			///No movement	
			return MCL_SUCCESS;
		}
		else if (noXMovement || XMoveBlocked(xMove))
		{ 
			err = MCL_MDMove(axis2_, velocity_, yMove, handle_);
			if (err != MCL_SUCCESS)
				return err;
		}
		else if (noYMovement || YMoveBlocked(yMove))
		{
			err = MCL_MDMove(axis1_, velocity_, xMove, handle_);
			if (err != MCL_SUCCESS)
				return err;
		}
		else 
		{
			double twoAxisVelocity = min(maxVelocityTwoAxis_, velocity_);
			err = MCL_MDMoveThreeAxes(axis1_, twoAxisVelocity, xMove,
									  axis2_, twoAxisVelocity, yMove,
									  0, 0, 0,
									  handle_);
			if (err != MCL_SUCCESS)
				return err;
		}

		err = MCL_MicroDriveGetWaitTime(&waitTime, handle_);
		if (err != MCL_SUCCESS)
			return err;
		snprintf(iToChar, sizeof(iToChar), "%d", waitTime);
		SetProperty(g_Keyword_WaitTime, iToChar);
	
		PauseDevice();

		err = MCL_MDCurrentPositionM(axis1_, &endingMicroStepsX, handle_);
		if (err != MCL_SUCCESS)
			return err;
		err = MCL_MDCurrentPositionM(axis2_, &endingMicroStepsY, handle_);
		if (err != MCL_SUCCESS)
			return err;
		lastX_ += (endingMicroStepsX - startingMicroStepsX) * stepSize_mm_;
		lastY_ += (endingMicroStepsY - startingMicroStepsY) * stepSize_mm_;

		// Update current position
		err = GetPositionMm(xCurrent, yCurrent);
		if (err != MCL_SUCCESS)
			return err;
		Sleep(50);
		err = GetPositionMm(xCurrent, yCurrent);
		if (err != MCL_SUCCESS)
			return err;

		if(iterativeMoves_ && encoded_ && !stopCommanded_)
		{
			double absDiffUmX = abs(goalX - xCurrent) * 1000.0; 
			double absDiffUmY = abs(goalY - yCurrent) * 1000.0;
			bool xInTolerance = noXMovement || (!noXMovement && absDiffUmX < imToleranceUm_);
			bool yInTolerance = noYMovement || (!noYMovement && absDiffUmY < imToleranceUm_);
			
			if(xInTolerance && yInTolerance)
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


int MicroDriveXYStage::SetRelativePositionUm(double dx, double dy)
{
	double absX, absY;
	int err = ConvertRelativeToAbsoluteMm(dx / 1000, dy / 1000, absX, absY);
	if (err != MCL_SUCCESS)
		return err;

	err = BeginMovementThread(STANDARD_MOVE_TYPE, absX, absY);

	return err;
}


int MicroDriveXYStage::ConvertRelativeToAbsoluteMm(double relX, double relY, double &absX, double &absY)
{
	int err;
	bool mirrorX, mirrorY;
	GetOrientation(mirrorX, mirrorY);

	if (mirrorX)
		relX = -relX;
	if (mirrorY)
		relY = -relY;

	//Calculate the absolute position.
	double xCurrent, yCurrent;
	err = GetPositionMm(xCurrent, yCurrent);
	if (err != MCL_SUCCESS)
		return err;

	absX = xCurrent + relX;
	absY = yCurrent + relY;

	return MCL_SUCCESS;
}


int MicroDriveXYStage::SetRelativePositionMmSync(double x, double y)
{
	double absX, absY;
	int err = ConvertRelativeToAbsoluteMm(x, y, absX, absY);
	if (err != MCL_SUCCESS)
		return err;

	err = SetPositionMmSync(absX, absY);

	return err;
}


int MicroDriveXYStage::GetPositionMm(double& x, double& y)
{
   if (encoded_) {
	   double m1, m2, m3, m4;
	   int err = MCL_MDReadEncoders(&m1, &m2, &m3, &m4, handle_);
	   if (err != MCL_SUCCESS)
		   return err;

	   switch (axis1_)
	   {
		   case M1AXIS:
			   x = m1;
			   break;
		   case M2AXIS:
			   x = m2;
			   break;
		   case M3AXIS:
			   x = m3;
			   break;
		   case M4AXIS:
			   x = m4;
			   break;
		   default:
			   x = lastX_;
			   break;
	   }
	   switch (axis2_)
	   {
		   case M1AXIS:
			   y = m1;
			   break;
		   case M2AXIS:
			   y = m2;
			   break;
		   case M3AXIS:
			   y = m3;
			   break;
		   case M4AXIS:
			   y = m4;
			   break;
		   default:
			   y = lastY_;
			   break;
	   }
   } else {
      x = lastX_;
      y = lastY_;
   }

	return DEVICE_OK;
}


double MicroDriveXYStage::GetStepSize()
{
	return stepSize_mm_;
}


int MicroDriveXYStage::SetPositionSteps(long x, long y)
{
	return BeginMovementThread(STANDARD_MOVE_TYPE, x * stepSize_mm_, y * stepSize_mm_);
}


int MicroDriveXYStage::GetPositionSteps(long& x, long& y)
{
	int err = DEVICE_OK;
	double getX, getY;

	// Check if the thread is running
	long ret = WaitForSingleObject(movementThread_, 0);
	if (ret == WAIT_TIMEOUT)
	{
		// If we are moving simply return the last know position.
		getX = lastX_;
		getY = lastY_;
	}
	else
		err = GetPositionMm(getX, getY);

	x = (long) (getX / stepSize_mm_);
	y = (long) (getY / stepSize_mm_);

	return err;
}


int MicroDriveXYStage::Home()
{
	return BeginMovementThread(HOME_TYPE, 0.0, 0.0);
}


void MicroDriveXYStage::PauseDevice()
{
	MCL_MicroDriveWait(handle_);
}


int MicroDriveXYStage::Stop()
{
	int err = MCL_MDStop(NULL, handle_);
	stopCommanded_ = true;
	if (err != MCL_SUCCESS)
		return err;

	return DEVICE_OK;
}


int MicroDriveXYStage::SetOrigin()
{
	long ret = WaitForSingleObject(movementThread_, 0);
	if (ret == WAIT_TIMEOUT)
	{
		return MCL_DEV_NOT_READY;
	}

	return SetOriginSync();
}


int MicroDriveXYStage::SetOriginSync()
{
	if (encoded_)
	{
		int err = MCL_SUCCESS;
		if (axis1_ < M5AXIS)
		{
			err = MCL_MDResetEncoder(axis1_, NULL, handle_);
			if (err != MCL_SUCCESS)
				return err;
		}
		if (axis2_ < M5AXIS)
		{
			err = MCL_MDResetEncoder(axis2_, NULL, handle_);
			if (err != MCL_SUCCESS)
				return err;
		}
	}
	lastX_ = 0;
	lastY_ = 0;

	return DEVICE_OK;
}


int MicroDriveXYStage::GetLimitsUm(double& /*xMin*/, double& /*xMax*/, double& /*yMin*/, double& /*yMax*/)
{
	return DEVICE_UNSUPPORTED_COMMAND;
}


int MicroDriveXYStage::GetStepLimits(long& /*xMin*/, long& /*xMax*/, long& /*yMin*/, long& /*yMax*/)
{
	return DEVICE_UNSUPPORTED_COMMAND;
}


double MicroDriveXYStage::GetStepSizeXUm()
{
	return stepSize_mm_;
}


double MicroDriveXYStage::GetStepSizeYUm()
{
	return stepSize_mm_;
}


int MicroDriveXYStage::CalibrateSync()
{
	int err;
	double xPosOrig;
	double yPosOrig;
	double xPosLimit;
	double yPosLimit;

	err = GetPositionMm(xPosOrig, yPosOrig);
	if (err != MCL_SUCCESS)
		return err;

	err = MoveToForwardLimitsSync();
	if (err != DEVICE_OK)
		return err;

	err = GetPositionMm(xPosLimit, yPosLimit);
	if (err != MCL_SUCCESS)
		return err;

	err = SetOriginSync();
	if (err != DEVICE_OK)
		return err;

	err = SetPositionMmSync((xPosOrig - xPosLimit), (yPosOrig - yPosLimit));
	if (err != DEVICE_OK)
		return err;

	return DEVICE_OK;
}


bool MicroDriveXYStage::XMoveBlocked(double possNewPos)
{
	unsigned short status = 0;

	MCL_MDStatus(&status, handle_);

	unsigned short revLimitBitMask = LimitBitMask(pid_, axis1_, REVERSE);
	unsigned short fowLimitBitMask = LimitBitMask(pid_, axis1_, FORWARD);

	bool atReverseLimit = ((status & revLimitBitMask) == 0);
	bool atForwardLimit = ((status & fowLimitBitMask) == 0);

	if(atReverseLimit && possNewPos < 0)
		return true;
	else if (atForwardLimit && possNewPos > 0)
		return true;
	return false;
}


bool MicroDriveXYStage::YMoveBlocked(double possNewPos)
{
	unsigned short status = 0;

	MCL_MDStatus(&status, handle_);

	unsigned short revLimitBitMask = LimitBitMask(pid_, axis2_, REVERSE);
	unsigned short fowLimitBitMask = LimitBitMask(pid_, axis2_, FORWARD);

	bool atReverseLimit = ((status & revLimitBitMask) == 0);
	bool atForwardLimit = ((status & fowLimitBitMask) == 0);

	if (atReverseLimit && possNewPos < 0)
		return true;
	else if (atForwardLimit && possNewPos > 0)
		return true;
	return false;
}


void MicroDriveXYStage::GetOrientation(bool& mirrorX, bool& mirrorY) 
{
	char val[MM::MaxStrLength];
	int ret = this->GetProperty(MM::g_Keyword_Transpose_MirrorX, val);

	assert(ret == DEVICE_OK);
	mirrorX = strcmp(val, "1") == 0 ? true : false;

	ret = this->GetProperty(MM::g_Keyword_Transpose_MirrorY, val);
	assert(ret == DEVICE_OK);
	mirrorY = strcmp(val, "1") == 0 ? true : false;
}


int MicroDriveXYStage::MoveToForwardLimitsSync()
{
	int err;
	unsigned short status = 0;

	err = MCL_MDStatus(&status, handle_);
	if(err != MCL_SUCCESS)	
		return err;

	unsigned short axis1ForwardLimitBitMask = LimitBitMask(pid_, axis1_, FORWARD);
	unsigned short axis2ForwardLimitBitMask = LimitBitMask(pid_, axis2_, FORWARD);
	unsigned short bothLimits = axis1ForwardLimitBitMask | axis2ForwardLimitBitMask;

	while (((status & bothLimits) != 0) && !stopCommanded_)
	{ 
		err = SetRelativePositionUm(4000, 4000);
		if (err != DEVICE_OK)
			return err;

		err = MCL_MDStatus(&status, handle_);
		if (err != MCL_SUCCESS)	
			return err;
	}

	return DEVICE_OK;
}


int MicroDriveXYStage::ReturnToOriginSync()
{
	int err;
	double xPos;
	double yPos;

	err = SetPositionMmSync(0.0, 0.0);
	if (err != DEVICE_OK)
		return err;

	///Finish the motion if one axis hit its limit first.
	err = GetPositionMm(xPos, yPos);
	if (err != MCL_SUCCESS)
		return err;

	bool yBlocked = (YMoveBlocked(1) || YMoveBlocked(-1));
	bool xBlocked = (XMoveBlocked(1) || XMoveBlocked(-1));
	
	bool xNotFinished = xPos != 0 && yBlocked;
	bool yNotFinished = yPos != 0 && xBlocked;

	if ((xNotFinished || yNotFinished) && !stopCommanded_)
	{
		err = SetPositionMmSync(0, 0); 
		if (err != DEVICE_OK)
			return err;
	}

	return DEVICE_OK;
}


int MicroDriveXYStage::FindEpiSync()
{
	if ((axis1IsTirfModule_ == false) && (axis2IsTirfModule_ == false))
		return DEVICE_OK;

	int err = MCL_SUCCESS;
	int epiAxis = 0;
	double a1Find = 0.0;
	double a2Find = 0.0;
	double a1Epi = 0.0;
	double a2Epi = 0.0;
	if (axis1IsTirfModule_)
	{
		epiAxis = axis1_;
		a1Find = -.5;
		a1Epi = tirfModCalibrationMm_;
	}
	else if(axis2IsTirfModule_)
	{
		epiAxis = axis2_;
		a2Find = -.5;
		a2Epi = tirfModCalibrationMm_;
	}

	unsigned short status;
	unsigned short mask = LimitBitMask(pid_, epiAxis, REVERSE);

	MCL_MDStatus(&status, handle_);

	// Move the stage to its reverse limit.
	while (((status & mask) == mask) && !stopCommanded_)
	{
		err = SetRelativePositionMmSync(a1Find, a2Find);
		if (err != DEVICE_OK)
			return err;
		MCL_MDStatus(&status, handle_);
	}

	// Set the orgin of the epi axis at the reverse limit.
	if (encoded_ && epiAxis < M5AXIS)
	{
		err = MCL_MDResetEncoder(epiAxis, NULL, handle_);
		if (err != MCL_SUCCESS)
			return err;
	}
	if (epiAxis == axis1_)
		lastX_ = 0;
	else
		lastY_ = 0;

	// Move the calibration distance to find epi.
	err = SetPositionMmSync(a1Epi, a2Epi);
	if (err != DEVICE_OK)
		return err;

	// Set the orgin of the epi axis at epi.
	if (encoded_ && epiAxis < M5AXIS)
	{
		err = MCL_MDResetEncoder(epiAxis, NULL, handle_);
		if (err != MCL_SUCCESS)
			return err;
	}
	if (epiAxis == axis1_)
		lastX_ = 0;
	else
		lastY_ = 0;

	return DEVICE_OK;
}


int MicroDriveXYStage::OnPositionXmm(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	int err;
	double pos;
	if (eAct == MM::BeforeGet)
	{
		double x, y;
		GetPositionMm(x, y);
		pProp->Set(x);
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(pos);

		double x, y;
		err = GetPositionMm(x, y);
		if(err != MCL_SUCCESS)
			return err;
		err = BeginMovementThread(STANDARD_MOVE_TYPE, pos, y);
		if (err != DEVICE_OK)
			return err;
	}

	return DEVICE_OK;
}


int MicroDriveXYStage::OnPositionYmm(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	int err;
	double pos;
	
	if (eAct == MM::BeforeGet)
	{
		double x, y;
		GetPositionMm(x, y);
		pProp->Set(y);
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(pos);

		double x, y;
		err = GetPositionMm(x, y);		
		if(err != MCL_SUCCESS)
			return err;
		err = BeginMovementThread(STANDARD_MOVE_TYPE, x, pos);
		if (err != DEVICE_OK)
			return err;
	}

	return DEVICE_OK;
}


int MicroDriveXYStage::OnMoveXmm(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	int err;
	double pos;
	if (eAct == MM::AfterSet)
	{
		pProp->Get(pos);

		err = SetRelativePositionUm(pos * 1000.0, 0.0);
		if (err != MCL_SUCCESS)
			return err;
	}

	return DEVICE_OK;
}


int MicroDriveXYStage::OnMoveYmm(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	int err;
	double pos;
	if (eAct == MM::AfterSet)
	{
		pProp->Get(pos);
		
		err = SetRelativePositionUm(0.0, pos * 1000.0);
		if (err != MCL_SUCCESS)
			return err;
	}
	return DEVICE_OK;
}


int MicroDriveXYStage::OnSetOriginHere(MM::PropertyBase* pProp, MM::ActionType eAct)
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


int MicroDriveXYStage::OnCalibrate(MM::PropertyBase* pProp, MM::ActionType eAct)
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
			err = BeginMovementThread(CALIBRATE_TYPE, 0.0, 0.0);
			if (err != DEVICE_OK)
				return err;
		}
	}

	return DEVICE_OK;
}


int MicroDriveXYStage::OnReturnToOrigin(MM::PropertyBase* pProp, MM::ActionType eAct)
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
			err = BeginMovementThread(RETURN_TO_ORIGIN_TYPE, 0, 0);
			if (err != DEVICE_OK)
				return err;
		}
	}

	return DEVICE_OK;
}


int MicroDriveXYStage::OnPositionXYmm(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	int err;
	double x, y;
	string input;
	vector<string> tokenInput;
	char* pEnd;

	if (eAct == MM::BeforeGet)
	{
		GetPositionMm(x, y);
		char iToChar[30];
		snprintf(iToChar, sizeof(iToChar), "X = %f Y = %f", x, y);
		pProp->Set(iToChar);
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(input);

		CDeviceUtils::Tokenize(input, tokenInput, "X=");

		x = strtod(tokenInput[0].c_str(), &pEnd);
	    y = strtod(tokenInput[1].c_str(), &pEnd);

		err = BeginMovementThread(STANDARD_MOVE_TYPE, x, y);
		if (err != DEVICE_OK)
			return err;
	}

	return DEVICE_OK;
}


int MicroDriveXYStage::OnVelocity(MM::PropertyBase* pProp, MM::ActionType eAct){

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


int MicroDriveXYStage::OnEncoded(MM::PropertyBase* pProp, MM::ActionType eAct)
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


int MicroDriveXYStage::OnIterativeMove(MM::PropertyBase* pProp, MM::ActionType eAct)
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


int MicroDriveXYStage::OnImRetry(MM::PropertyBase* pProp, MM::ActionType eAct)
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


int MicroDriveXYStage::OnImToleranceUm(MM::PropertyBase* pProp, MM::ActionType eAct)
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


int MicroDriveXYStage::IsXYStageSequenceable(bool& isSequenceable) const
{
	isSequenceable = false;
	return DEVICE_OK;
}


int MicroDriveXYStage::OnIsTirfModuleAxis1(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(axis1IsTirfModule_ ? g_Listword_Yes : g_Listword_No);
	}
	else if (eAct == MM::AfterSet)
	{
		string message;
		pProp->Get(message);
		axis1IsTirfModule_ = (message.compare(g_Listword_Yes) == 0);
	}
	return DEVICE_OK;
}


int MicroDriveXYStage::OnIsTirfModuleAxis2(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(axis2IsTirfModule_ ? g_Listword_Yes : g_Listword_No);
	}
	else if (eAct == MM::AfterSet)
	{
		string message;
		pProp->Get(message);
		axis2IsTirfModule_ = (message.compare(g_Listword_Yes) == 0);
	}
	return DEVICE_OK;
}


int MicroDriveXYStage::OnFindEpi(MM::PropertyBase* pProp, MM::ActionType eAct)
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
			err = BeginMovementThread(FIND_EPI_TYPE, 0.0, 0.0);
			if (err != DEVICE_OK)
				return err;
		}
	}
	return DEVICE_OK;
}


int MicroDriveXYStage::OnStop(MM::PropertyBase* pProp, MM::ActionType eAct) 
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
	return err;
}


int MicroDriveXYStage::BeginMovementThread(int type, double distanceX, double distanceY)
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
	movementDistanceX_ = distanceX;
	movementDistanceY_ = distanceY;
	movementThread_ = CreateThread(0, 0, ExecuteMovement, this, 0, 0);

	ReleaseMutex(threadStartMutex_);

	return DEVICE_OK;

}


DWORD WINAPI MicroDriveXYStage::ExecuteMovement(LPVOID lpParam) {

	MicroDriveXYStage* instance = reinterpret_cast<MicroDriveXYStage*>(lpParam);
	instance->stopCommanded_ = false;
	switch (instance->movementType_)
	{
	case STANDARD_MOVE_TYPE:
		instance->SetPositionMmSync(instance->movementDistanceX_, instance->movementDistanceY_);
		break;
	case CALIBRATE_TYPE:
		instance->CalibrateSync();
		break;
	case HOME_TYPE:
		instance->MoveToForwardLimitsSync();
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