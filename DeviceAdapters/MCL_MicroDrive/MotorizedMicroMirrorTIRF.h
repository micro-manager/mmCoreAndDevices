/*
File:		MotorizedMicroMirrorTIRF.h
Copyright:	Mad City Labs Inc., 2025
License:	Distributed under the BSD license.
*/
#pragma once

//MCL Headers
#include "MCL_MicroDrive.h"
#include "MicroDrive.h"

// MM headers
#include "MMDevice.h"
#include "ModuleInterface.h"
#include "MMDeviceConstants.h"
#include "DeviceBase.h"

// List headers
#include "handle_list_if.h"

using namespace std;

#define MC_MTS_EPI_FOUND 0
#define MC_MTS_FOCUS_FOUND 1
#define MC_MTS_TIRF_FOUND 2
#define MC_MTS_EPI_STEPS 3
#define MC_MTS_FOCUS_STEPS 4
#define MC_MTS_TIRF_STEPS 5
#define MC_MTS_EPI_TO_TIRF_STEPS 6


class MotorizedMicroMirrorTIRF : public CGenericBase<MotorizedMicroMirrorTIRF>
{
public:
	MotorizedMicroMirrorTIRF();
	~MotorizedMicroMirrorTIRF();

	// Device Interface
	int Initialize();
	int Shutdown();

	bool Busy();
	void GetName(char* pszName) const;

	// Action Interface
	int OnLimitBitmap(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnEntranceMicroSteps(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnExitMicroSteps(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnFocusMicroSteps(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnFoundEntranceLimit(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnFoundExitLimit(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnFoundFocusLimit(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnFoundEpi(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnFoundFocus(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnFoundTIRFAir(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnMove(MM::PropertyBase* pProp, MM::ActionType eAct);

	int OnEntranceLimitStepCount(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnExitLimitStepCount(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnFocusLimitStepCount(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnEpiStepCount(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnFocusStepCount(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnTirfStepCount(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnEpiToTIRFStepCount(MM::PropertyBase* pProp, MM::ActionType eAct);
	

private:
	//Initialization
	int CreateProperties();
	int InitDeviceAdapter();

	//Device Information
	int handle_;
	int axis_;
	int serialNumber_;
	unsigned short pid_;

	//Device State
	bool busy_;
	bool initialized_;

	// Motorized Micro-Mirror State
	int entranceAxis_;
	int exitAxis_;
	int focusAxis_;

	double entranceStepSize_;
	double exitStepSize_;
	double focusStepSize_;

	bool foundEntranceLimit_;
	bool foundExitLimit_;
	bool foundFocusLimit_;
	bool foundEpi_;
	bool foundFocus_;
	bool foundTIRFAir_;

	int entranceLimitStepCount_;
	int exitLimitStepCount_;
	int focusLimitStepCount_;
	int epiStepCount_;
	int focusStepCount_;
	int tirfStepCount_;
	int stepCountFromEpiToTIRF_;
};
