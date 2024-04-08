/*
File:		MicroDriveZStage.h
Copyright:	Mad City Labs Inc., 2023
License:	Distributed under the BSD license.
*/
#pragma once

// MCL headers
#include "MicroDrive.h"
#include "MCL_MicroDrive.h"

// MM headers
#include "MMDevice.h"
#include "DeviceBase.h"
#include "ImgBuffer.h"
#include "ModuleInterface.h"

// List
#include "handle_list_if.h"
#include "HandleListType.h"

#define ERR_UNKNOWN_MODE         102
#define ERR_UNKNOWN_POSITION     103
#define ERR_NOT_VALID_INPUT      104

#define STANDARD_MOVE_TYPE			1
#define CALIBRATE_TYPE				2

class MCL_MicroDrive_ZStage : public CStageBase<MCL_MicroDrive_ZStage>
{
public:

	MCL_MicroDrive_ZStage();
	~MCL_MicroDrive_ZStage();

	// Device Interface
	int Initialize();
	int Shutdown();
	bool Busy();
	void GetName(char* pszName) const;

	// Stage API
	virtual double GetStepSize();
	virtual int SetPositionUm(double pos);
	virtual int GetPositionUm(double& pos);
	virtual int SetRelativePositionUm(double d);
	virtual int SetPositionSteps(long steps);
    virtual int GetPositionSteps(long& steps);
    virtual int SetOrigin();
	virtual int GetLimits(double& lower, double& upper);
	virtual int IsStageSequenceable(bool& isSequenceable) const;
	virtual bool IsContinuousFocusDrive() const;


	int getHandle(){ return handle_;}

	// Action interface
	int OnPositionMm(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnMovemm(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnSetOrigin(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnCalibrate(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnReturnToOrigin(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnVelocity(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnEncoded(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnIterativeMove(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnImRetry(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnImToleranceUm(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnIsTirfModule(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnFindEpi(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnStop(MM::PropertyBase* pProp, MM::ActionType eAct);


private:
	// Initialization
	int CreateZStageProperties();
	int InitDeviceAdapter();

	// Set/Get positions
	int SetPositionMmSync(double z);
	int GetPositionMm(double& z);
	int SetRelativePositionMmSync(double z);
	int ConvertRelativeToAbsoluteMm(double relZ, double &absZ);

	// Calibration & origin methods
	int CalibrateSync();
	int MoveToForwardLimitSync();
	int ReturnToOriginSync();
	int FindEpiSync();
	int SetOriginSync();

	void PauseDevice();
	int Stop();

	// Threading
	int BeginMovementThread(int type, double distance);
	static DWORD WINAPI ExecuteMovement(LPVOID lpParam);

	// Device Information
	int handle_;
	int serialNumber_;
	unsigned short pid_;
	int axis_;
	unsigned char axisBitmap_;
	double stepSize_mm_;
	double encoderResolution_; 
	double maxVelocity_;
	double minVelocity_;
	double velocity_;
	// Device State
	bool initialized_;
	bool encoded_;
	double lastZ_;
	// Iterative Move State
	bool iterativeMoves_;
	int imRetry_;
	double imToleranceUm_;
	// Tirf-Module State
	bool deviceHasTirfModuleAxis_;
	bool axisIsTirfModule_;
	bool hasUnknownTirfModuleAxis_;
	double tirfModCalibrationMm_;
	// Threading
	bool stopCommanded_;
	int movementType_;
	double movementDistance_;
	HANDLE movementThread_;
	HANDLE threadStartMutex_;
};
