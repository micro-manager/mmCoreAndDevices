/*
File:		MadTweezer.h
Copyright:	Mad City Labs Inc., 2023
License:	Distributed under the BSD license.
*/

#ifndef _MADTWEEZER_H_
#define _MADTWEEZER_H_

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


#define ERR_UNKNOWN_MODE        102

#define HIGH_SPEED				1
#define HIGH_PRECISION			3

class MadTweezer : public CGenericBase<MadTweezer>
{
public:
	MadTweezer();
	~MadTweezer();

	// Device Interface
	int Initialize();
	int Shutdown();

	bool Busy();
	void GetName(char* pszName) const;

	// Action Interface
	int OnMode(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnLocation(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnHome(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnDirection(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnVelocity(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnRotation(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnSteps(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnStop(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnMrad(MM::PropertyBase* pProp, MM::ActionType eAct);

	// Rotational Stage API
	int GetLocation();
	int SetLocation(double mrad);
	int GetMode();
	int SetMode(int mode);
	int UpdateVelocity();
	int UpdateLocation();

private:
	//Initialization
	int CreateMadTweezerProperties();
	int InitDeviceAdapter();

	//Device Information
	int handle_;
	int serialNumber_;
	unsigned short pid_;
	int axis_;
	double encoderResolution_;
	double stepSize_rad_;
	double maxVelocity_;
	double minVelocity_;
	double location_mrad_;
	double max_mrad_;
	double velocity_rad_;
	int units_;
	int mode_;
	int direction_;

	//Device State
	bool busy_;
	bool initialized_;
	bool encoded_;
	bool home_;
};









#endif //_MADTWEEZER_H_