/*
File:		MotorizedMicroMirrorTIRF.cpp
Copyright:	Mad City Labs Inc., 2025
License:	Distributed under the BSD license.
*/

#include "AcquireDevice.h"
#include "MotorizedMicroMirrorTIRF.h"
#include "MCL_MicroDrive.h"
#include "HandleListType.h"
#include "DeviceUtils.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <string.h>

#include <vector>
#include <iostream>

using namespace std;

// MMTState bitmask values.
#define M1_LS1_BITMASK 0x01
#define M1_LS2_BITMASK 0x02
#define M2_LS1_BITMASK 0x04
#define M2_LS2_BITMASK 0x08
#define M3_LS1_BITMASK 0x10
#define M3_LS2_BITMASK 0x20
#define M4_LS1_BITMASK 0x40
#define M4_LS2_BITMASK 0x80  

MotorizedMicroMirrorTIRF::MotorizedMicroMirrorTIRF() :
	handle_(0),
	axis_(0),
	serialNumber_(0),
	pid_(0),
	busy_(false),
	initialized_(false),
	entranceAxis_(0),
	exitAxis_(0),
	focusAxis_(0),
	entranceStepSize_(0.0),
	exitStepSize_(0.0),
	focusStepSize_(0.0),
	foundEntranceLimit_(false),
	foundExitLimit_(false),
	foundFocusLimit_(false),
	foundEpi_(false),
	foundFocus_(false),
	foundTIRFAir_(false),
	entranceLimitStepCount_(0),
	exitLimitStepCount_(0),
	focusLimitStepCount_(0),
	epiStepCount_(0),
	focusStepCount_(0),
	tirfStepCount_(0),
	stepCountFromEpiToTIRF_(0)
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
}


MotorizedMicroMirrorTIRF::~MotorizedMicroMirrorTIRF()
{
	Shutdown();
}

bool MotorizedMicroMirrorTIRF::Busy()
{
	return busy_;
}


void MotorizedMicroMirrorTIRF::GetName(char* pszName) const
{
	CDeviceUtils::CopyLimitedString(pszName, g_DeviceMMMTIRFName);
}

int MotorizedMicroMirrorTIRF::Initialize()
{
	int err = DEVICE_OK;

	HandleListLock();
	err = InitDeviceAdapter();
	HandleListUnlock();

	return err;
}


int MotorizedMicroMirrorTIRF::InitDeviceAdapter()
{
	if (initialized_)
		return DEVICE_OK;

	// Attempt to acquire a device/axis for this adapter.
	int ret = MCL_SUCCESS;
	ret = AcquireDeviceHandle(MOTORIZIED_MICROMIRROR_TIRF_TYPE, handle_, axis_);
	if (ret != MCL_SUCCESS)
		return ret;
	entranceAxis_ = axis_;
	exitAxis_ = axis_ + 1;
	focusAxis_ = axis_ + 2;

	// Query device information
	serialNumber_ = MCL_GetSerialNumber(handle_);

	ret = MCL_GetProductID(&pid_, handle_);
	if (ret != MCL_SUCCESS)
		return ret;

	double er, ss, mv, mv2, mv3, minV;
	int units;
	ret = MCL_MDAxisInformation(entranceAxis_, &er, &ss, &mv, &mv2, &mv3, &minV, &units, handle_);
	if (ret != MCL_SUCCESS)
		return ret;
	entranceStepSize_ = ss * 1000.0 * 1000.0;

	ret = MCL_MDAxisInformation(exitAxis_, &er, &ss, &mv, &mv2, &mv3, &minV, &units, handle_);
	if (ret != MCL_SUCCESS)
		return ret;
	exitStepSize_ = ss * 1000.0 * 1000.0;

	ret = MCL_MDAxisInformation(focusAxis_, &er, &ss, &mv, &mv2, &mv3, &minV, &units, handle_);
	if (ret != MCL_SUCCESS)
		return ret;
	focusStepSize_ = ss * 1000.0 * 1000.0;

	MMTState mmtState;
	ret = MCL_MMTGetState(&mmtState, handle_);	
	if (ret != MCL_SUCCESS)
		return ret;
	foundEntranceLimit_ = (mmtState.limitSwitchesFound & M2_LS2_BITMASK) == M2_LS2_BITMASK;
	foundExitLimit_ = (mmtState.limitSwitchesFound & M3_LS2_BITMASK) == M3_LS2_BITMASK;
	foundFocusLimit_ = (mmtState.limitSwitchesFound & M4_LS2_BITMASK) == M4_LS2_BITMASK;
	foundEpi_ = mmtState.epiFound == 1;
	foundFocus_ = mmtState.focusFound == 1;
	foundTIRFAir_ = mmtState.tirfAIRFound == 1;
	entranceLimitStepCount_ = mmtState.m2LS2Steps;
	exitLimitStepCount_ = mmtState.m3LS2Steps;
	focusLimitStepCount_ = mmtState.m4LS2Steps;
	stepCountFromEpiToTIRF_ = mmtState.epiToTirfSteps;
	epiStepCount_ = mmtState.epiSteps;
	focusStepCount_ = mmtState.focusSteps;
	tirfStepCount_ = mmtState.tirfAIRSteps;

	// Create properties
	int err = DEVICE_OK;

	err = CreateProperties();
	if (err != DEVICE_OK)
		return err;

	err = UpdateStatus();
	if (err != DEVICE_OK)
		return err;

	initialized_ = true;

	return DEVICE_OK;
}


int MotorizedMicroMirrorTIRF::CreateProperties()
{
	vector<string> yesNoList;
	yesNoList.push_back(g_Listword_No);
	yesNoList.push_back(g_Listword_Yes);

	int propertyCount = 32;
	int ii = 0;
	int *propErrors = new int[propertyCount];
	memset(propErrors, 0, sizeof(int) * propertyCount);

	/// Read only properties
	propErrors[ii++] = CreateStringProperty(MM::g_Keyword_Name, g_DeviceMMMTIRFName, true);
	propErrors[ii++] = CreateStringProperty(MM::g_Keyword_Description, "Motorized Micro-Mirror TIRF", true);
	propErrors[ii++] = CreateIntegerProperty(g_Keyword_Handle, handle_, true);
	propErrors[ii++] = CreateIntegerProperty(g_Keyword_ProductID, pid_, true);
	propErrors[ii++] = CreateIntegerProperty(g_Keyword_Serial_Num, serialNumber_, true);
	propErrors[ii++] = CreateIntegerProperty(g_Keyword_EntranceAxis, entranceAxis_, true);
	propErrors[ii++] = CreateIntegerProperty(g_Keyword_ExitAxis, exitAxis_, true);
	propErrors[ii++] = CreateIntegerProperty(g_Keyword_FocusAxis, focusAxis_, true);
	propErrors[ii++] = CreateFloatProperty(g_Keyword_EntranceStepSize, entranceStepSize_, true);
	propErrors[ii++] = CreateFloatProperty(g_Keyword_ExitStepSize, exitStepSize_, true);
	propErrors[ii++] = CreateFloatProperty(g_Keyword_FocusStepSize, focusStepSize_, true);
	propErrors[ii++] = CreateIntegerProperty(g_Keyword_EntranceLimitStepCount, entranceLimitStepCount_, true, new CPropertyAction(this, &MotorizedMicroMirrorTIRF::OnEntranceLimitStepCount));
	propErrors[ii++] = CreateIntegerProperty(g_Keyword_ExitLimitStepCount, exitLimitStepCount_, true, new CPropertyAction(this, &MotorizedMicroMirrorTIRF::OnExitLimitStepCount));
	propErrors[ii++] = CreateIntegerProperty(g_Keyword_FocusLimitStepCount, focusLimitStepCount_, true, new CPropertyAction(this, &MotorizedMicroMirrorTIRF::OnFocusLimitStepCount));

	propErrors[ii++] = CreateIntegerProperty(g_Keyword_LimitBitMap, 0, true, new CPropertyAction(this, &MotorizedMicroMirrorTIRF::OnLimitBitmap));
	propErrors[ii++] = CreateIntegerProperty(g_Keyword_EntranceMicroSteps, 0, true, new CPropertyAction(this, &MotorizedMicroMirrorTIRF::OnEntranceMicroSteps));
	propErrors[ii++] = CreateIntegerProperty(g_Keyword_ExitMicroSteps, 0, true, new CPropertyAction(this, &MotorizedMicroMirrorTIRF::OnExitMicroSteps));
	propErrors[ii++] = CreateIntegerProperty(g_Keyword_FocusMicroSteps, 0, true, new CPropertyAction(this, &MotorizedMicroMirrorTIRF::OnFocusMicroSteps));
	propErrors[ii++] = CreateIntegerProperty(g_Keyword_FoundEntranceLimit, foundEntranceLimit_ ? 1 : 0, true, new CPropertyAction(this, &MotorizedMicroMirrorTIRF::OnFoundEntranceLimit));
	propErrors[ii++] = CreateIntegerProperty(g_Keyword_FoundExitLimit, foundExitLimit_ ? 1 : 0, true, new CPropertyAction(this, &MotorizedMicroMirrorTIRF::OnFoundExitLimit));
	propErrors[ii++] = CreateIntegerProperty(g_Keyword_FoundFocusLimit, foundFocusLimit_ ? 1 : 0, true, new CPropertyAction(this, &MotorizedMicroMirrorTIRF::OnFoundFocusLimit));
	/// Read/Write Properties
	propErrors[ii++] = CreateFloatProperty(g_Keyword_MoveDistanceEntrance, 0.0, false);
	propErrors[ii++] = CreateFloatProperty(g_Keyword_MoveDistanceFocus, 0.0, false);
	propErrors[ii++] = CreateFloatProperty(g_Keyword_MoveDistanceTirf, 0, 0, false);
	propErrors[ii++] = CreateIntegerProperty(g_Keyword_EpiStepCount, epiStepCount_, false, new CPropertyAction(this, &MotorizedMicroMirrorTIRF::OnEpiStepCount));
	propErrors[ii++] = CreateIntegerProperty(g_Keyword_FocusStepCount, focusStepCount_, false, new CPropertyAction(this, &MotorizedMicroMirrorTIRF::OnFocusStepCount));
	propErrors[ii++] = CreateIntegerProperty(g_Keyword_TirfStepCount, tirfStepCount_, false, new CPropertyAction(this, &MotorizedMicroMirrorTIRF::OnTirfStepCount));
	propErrors[ii++] = CreateIntegerProperty(g_Keyword_EpiToTIRFStepCount, stepCountFromEpiToTIRF_, false, new CPropertyAction(this, &MotorizedMicroMirrorTIRF::OnEpiToTIRFStepCount));
	/// Action Properties
	propErrors[ii++] = CreateIntegerProperty(g_Keyword_FoundEpi, foundEpi_ ? 1 : 0, false, new CPropertyAction(this, &MotorizedMicroMirrorTIRF::OnFoundEpi));
	propErrors[ii++] = CreateIntegerProperty(g_Keyword_FoundFocus, foundFocus_ ? 1 : 0, false, new CPropertyAction(this, &MotorizedMicroMirrorTIRF::OnFoundFocus));
	propErrors[ii++] = CreateIntegerProperty(g_Keyword_FoundTIRFAir, foundTIRFAir_ ? 1 : 0, false, new CPropertyAction(this, &MotorizedMicroMirrorTIRF::OnFoundTIRFAir));
	propErrors[ii++] = CreateIntegerProperty(g_Keyword_AxesToMove, 0, false, new CPropertyAction(this, &MotorizedMicroMirrorTIRF::OnMove));
	//32

	for (int jj = 0; jj < propertyCount; jj++)
	{
		if (propErrors[jj] != DEVICE_OK)
			return propErrors[jj];
	}

	delete [] propErrors;


	return DEVICE_OK;
}


int MotorizedMicroMirrorTIRF::Shutdown() {
	HandleListLock();

	HandleListType device(handle_, MOTORIZIED_MICROMIRROR_TIRF_TYPE, axis_, 0);
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

int MotorizedMicroMirrorTIRF::OnLimitBitmap(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		unsigned short limitbitmap = 0;
		int ret = MCL_MDStatus(&limitbitmap, handle_);
		if (ret != MCL_SUCCESS)
			return ret;

		int entranceBits = (limitbitmap & (0x3 << ((entranceAxis_ - 1) * 2))) >> ((entranceAxis_ - 1) * 2);
		int focusBits = (limitbitmap & (0x3 << ((focusAxis_ - 1) * 2))) >> ((focusAxis_ - 1) * 2);
		int exitBits = (limitbitmap & (0x3 << ((exitAxis_ - 1) * 2))) >> ((exitAxis_ - 1) * 2);
		int limitBitmask = entranceBits | (exitBits << 2) | (focusBits << 4);
		pProp->Set((long)limitBitmask);
	}

	return DEVICE_OK;
}

int MotorizedMicroMirrorTIRF::OnEntranceMicroSteps(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		int microSteps = 0;
		int ret = MCL_MDCurrentPositionM(entranceAxis_, &microSteps, handle_);
		if (ret != MCL_SUCCESS)
			return ret;
		pProp->Set((long)microSteps);
	}

	return DEVICE_OK;
}

int MotorizedMicroMirrorTIRF::OnExitMicroSteps(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		int microSteps = 0;
		int ret = MCL_MDCurrentPositionM(exitAxis_, &microSteps, handle_);
		if (ret != MCL_SUCCESS)
			return ret;
		pProp->Set((long)microSteps);
	}

	return DEVICE_OK;
}

int MotorizedMicroMirrorTIRF::OnFocusMicroSteps(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		int microSteps = 0;
		int ret = MCL_MDCurrentPositionM(focusAxis_, &microSteps, handle_);
		if (ret != MCL_SUCCESS)
			return ret;
		pProp->Set((long)microSteps);
	}

	return DEVICE_OK;
}

int MotorizedMicroMirrorTIRF::OnFoundEntranceLimit(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		MMTState mmtState;
		int ret = MCL_MMTGetState(&mmtState, handle_);
		if (ret != MCL_SUCCESS)
			return ret;
		foundEntranceLimit_ = (mmtState.limitSwitchesFound & M2_LS2_BITMASK) == M2_LS2_BITMASK;
		pProp->Set(foundEntranceLimit_ ? 1l : 0l);
	}

	return DEVICE_OK;
}

int MotorizedMicroMirrorTIRF::OnFoundExitLimit(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		MMTState mmtState;
		int ret = MCL_MMTGetState(&mmtState, handle_);
		if (ret != MCL_SUCCESS)
			return ret;
		foundExitLimit_ = (mmtState.limitSwitchesFound & M3_LS2_BITMASK) == M3_LS2_BITMASK;
		pProp->Set(foundExitLimit_ ? 1l : 0l);
	}
	return DEVICE_OK;
}

int MotorizedMicroMirrorTIRF::OnFoundFocusLimit(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		MMTState mmtState;
		int ret = MCL_MMTGetState(&mmtState, handle_);
		if (ret != MCL_SUCCESS)
			return ret;
		foundFocusLimit_ = (mmtState.limitSwitchesFound & M4_LS2_BITMASK) == M4_LS2_BITMASK;
		pProp->Set(foundFocusLimit_ ? 1l : 0l);
	}
	return DEVICE_OK;
}

int MotorizedMicroMirrorTIRF::OnFoundEpi(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		// Reset value.  It may have been updated by loading distances.
		pProp->Set(foundEpi_ ? 1l : 0l);
	}
	else if (eAct == MM::AfterSet)
	{
		long value = -1;
		pProp->Get(value);

		if (value == 1)
		{
			int ret = MCL_SUCCESS;
			int microsteps = 0;
			MMTState mmt;
			ret = MCL_MMTGetState(&mmt, handle_);
			if (ret != MCL_SUCCESS)
				return ret;
			ret = MCL_MDCurrentPositionM(entranceAxis_, &microsteps, handle_);
			if (ret != MCL_SUCCESS)
				return ret;

			epiStepCount_ = microsteps - mmt.m2LS2Steps;
			ret = MCL_MMTSetState(MC_MTS_EPI_FOUND, 1, handle_);
			if (ret != MCL_SUCCESS)
				return ret;
			ret = MCL_MMTSetState(MC_MTS_EPI_STEPS, epiStepCount_, handle_);
			if (ret != MCL_SUCCESS)
				return ret;
			foundEpi_ = true;
		}
		else
		{
			// If the value is invalid, reset the value.
			pProp->Set(foundEpi_ ? 1l : 0l);
		}
	}

	return DEVICE_OK;
}

int MotorizedMicroMirrorTIRF::OnFoundFocus(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		// Reset value.  It may have been updated by loading distances.
		pProp->Set(foundFocus_ ? 1l : 0l);
	}
	else if (eAct == MM::AfterSet)
	{
		long value = -1;
		pProp->Get(value);

		if (value == 1)
		{
			int ret = MCL_SUCCESS;
			int microsteps = 0;

			MMTState mmt;
			ret = MCL_MMTGetState(&mmt, handle_);
			if (ret != MCL_SUCCESS)
				return ret;
			ret = MCL_MDCurrentPositionM(focusAxis_, &microsteps, handle_);
			if (ret != MCL_SUCCESS)
				return ret;

			focusStepCount_ = microsteps - mmt.m4LS2Steps;
			ret = MCL_MMTSetState(MC_MTS_FOCUS_FOUND, 1, handle_);
			if (ret != MCL_SUCCESS)
				return ret;
			ret = MCL_MMTSetState(MC_MTS_FOCUS_STEPS, focusStepCount_, handle_);
			if (ret != MCL_SUCCESS)
				return ret;
			foundFocus_ = true;
		}
		else
		{
			// If the value is invalid, reset the value.
			pProp->Set(foundFocus_ ? 1l : 0l);
		}
	}

	return DEVICE_OK;
}


int MotorizedMicroMirrorTIRF::OnFoundTIRFAir(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		// Reset value.  It may have been updated by loading distances.
		pProp->Set(foundTIRFAir_ ? 1l : 0l);
	}
	else if (eAct == MM::AfterSet)
	{
		long value = -1;
		pProp->Get(value);

		if (value == 1)
		{
			int ret = MCL_SUCCESS;
			int microsteps = 0;
			int microstepsEpi = 0;
			MMTState mmt;
			ret = MCL_MMTGetState(&mmt, handle_);
			if (ret != MCL_SUCCESS)
				return ret;
			ret = MCL_MDCurrentPositionM(exitAxis_, &microsteps, handle_);
			if (ret != MCL_SUCCESS)
				return ret;
			ret = MCL_MDCurrentPositionM(entranceAxis_, &microstepsEpi, handle_);
			if (ret != MCL_SUCCESS)
				return ret;

			tirfStepCount_ = microsteps - mmt.m3LS2Steps;
			stepCountFromEpiToTIRF_ = microstepsEpi - (mmt.m2LS2Steps + mmt.epiSteps);

			ret = MCL_MMTSetState(MC_MTS_TIRF_FOUND, 1, handle_);
			if (ret != MCL_SUCCESS)
				return ret;
			ret = MCL_MMTSetState(MC_MTS_TIRF_STEPS, tirfStepCount_, handle_);
			if (ret != MCL_SUCCESS)
				return ret;
			ret = MCL_MMTSetState(MC_MTS_EPI_TO_TIRF_STEPS, stepCountFromEpiToTIRF_, handle_);
			if (ret != MCL_SUCCESS)
				return ret;
			foundTIRFAir_ = true;
		}
		else
		{
			// If the value is invalid, reset the value.
			pProp->Set(foundTIRFAir_ ? 1l : 0l);
		}
	}

	return DEVICE_OK;
}

int MotorizedMicroMirrorTIRF::OnMove(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::AfterSet)
	{
		long axes = -1;
		pProp->Get(axes);

		double entranceDistanceNm = 0.0;
		double exitDistanceNm = 0.0;
		double focusDistanceNm = 0.0;
		GetProperty(g_Keyword_MoveDistanceEntrance, entranceDistanceNm);
		GetProperty(g_Keyword_MoveDistanceTirf, exitDistanceNm);
		GetProperty(g_Keyword_MoveDistanceFocus, focusDistanceNm);
		double entranceDistanceMm = entranceDistanceNm / 1000.0 / 1000.0;
		double exitDistanceMm = exitDistanceNm / 1000.0 / 1000.0;
		double focusDistanceMm = focusDistanceNm / 1000.0 / 1000.0;

		int ret = MCL_SUCCESS;
		double velocity = 1.0;

		// Zero out the distance arguments of axes we don't want to move;
		entranceDistanceMm = (axes & 0x01) != 0 ? entranceDistanceMm : 0;
		exitDistanceMm = (axes & 0x02) != 0 ? exitDistanceMm : 0;
		focusDistanceMm = (axes & 0x04) != 0 ? focusDistanceMm : 0;
		
		ret = MCL_MDMoveThreeAxes(
			entranceAxis_, velocity, entranceDistanceMm,
			exitAxis_, velocity, exitDistanceMm,
			focusAxis_, velocity, focusDistanceMm,
			handle_);
		if (ret != MCL_SUCCESS)
			return ret;

		ret = MCL_MicroDriveWait(handle_);
		if (ret != MCL_SUCCESS)
			return ret;

		// After each move check to see if the limit steps have updated.
		MMTState mmtState;
		ret = MCL_MMTGetState(&mmtState, handle_);
		if (ret != MCL_SUCCESS)
			return ret;
		foundEntranceLimit_ = (mmtState.limitSwitchesFound & M2_LS2_BITMASK) == M2_LS2_BITMASK;
		foundExitLimit_ = (mmtState.limitSwitchesFound & M3_LS2_BITMASK) == M3_LS2_BITMASK;
		foundFocusLimit_ = (mmtState.limitSwitchesFound & M4_LS2_BITMASK) == M4_LS2_BITMASK;
		entranceLimitStepCount_ = mmtState.m2LS2Steps;
		exitLimitStepCount_ = mmtState.m3LS2Steps;
		focusLimitStepCount_ = mmtState.m4LS2Steps;
	}

	return DEVICE_OK;
}

int MotorizedMicroMirrorTIRF::OnEntranceLimitStepCount(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set((long)entranceLimitStepCount_);
	}
	return DEVICE_OK;
}

int MotorizedMicroMirrorTIRF::OnExitLimitStepCount(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set((long)exitLimitStepCount_);
	}
	return DEVICE_OK;
}

int MotorizedMicroMirrorTIRF::OnFocusLimitStepCount(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set((long)focusLimitStepCount_);
	}
	return DEVICE_OK;
}

int MotorizedMicroMirrorTIRF::OnEpiStepCount(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set((long)epiStepCount_);
	}
	else if (eAct == MM::AfterSet)
	{
		long value = -1;
		pProp->Get(value);

		int ret = MCL_SUCCESS;
		epiStepCount_ = value;
		ret = MCL_MMTSetState(MC_MTS_EPI_FOUND, 1, handle_);
		if (ret != MCL_SUCCESS)
			return ret;
		ret = MCL_MMTSetState(MC_MTS_EPI_STEPS, epiStepCount_, handle_);
		if (ret != MCL_SUCCESS)
			return ret;
		foundEpi_ = 1;
	}
	return DEVICE_OK;
}

int MotorizedMicroMirrorTIRF::OnFocusStepCount(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set((long)focusStepCount_);
	}
	else if (eAct == MM::AfterSet)
	{
		long value = -1;
		pProp->Get(value);

		int ret = MCL_SUCCESS;
		focusStepCount_ = value;
		ret = MCL_MMTSetState(MC_MTS_FOCUS_FOUND, 1, handle_);
		if (ret != MCL_SUCCESS)
			return ret;
		ret = MCL_MMTSetState(MC_MTS_FOCUS_STEPS, focusStepCount_, handle_);
		if (ret != MCL_SUCCESS)
			return ret;
		foundFocus_ = 1;
	}
	return DEVICE_OK;
}

int MotorizedMicroMirrorTIRF::OnTirfStepCount(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set((long)tirfStepCount_);
	}
	else if (eAct == MM::AfterSet)
	{
		long value = -1;
		pProp->Get(value);

		int ret = MCL_SUCCESS;

		tirfStepCount_ = value;
		ret = MCL_MMTSetState(MC_MTS_TIRF_FOUND, 1, handle_);
		if (ret != MCL_SUCCESS)
			return ret;
		ret = MCL_MMTSetState(MC_MTS_TIRF_STEPS, tirfStepCount_, handle_);
		if (ret != MCL_SUCCESS)
			return ret;
		foundTIRFAir_ = true;
	}
	return DEVICE_OK;
}


int MotorizedMicroMirrorTIRF::OnEpiToTIRFStepCount(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set((long)stepCountFromEpiToTIRF_);
	}
	else if (eAct == MM::AfterSet)
	{
		long value = -1;
		pProp->Get(value);

		int ret = MCL_SUCCESS;
		stepCountFromEpiToTIRF_ = value;
		ret = MCL_MMTSetState(MC_MTS_EPI_TO_TIRF_STEPS, stepCountFromEpiToTIRF_, handle_);
		if (ret != MCL_SUCCESS)
			return ret;
	}
	return DEVICE_OK;
}