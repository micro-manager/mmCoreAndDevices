#pragma once
#include "GenericLaser.h"
using namespace std;

class OxxiusLBX : public GenericLaser 
{
public:
	OxxiusLBX(const char* nameAndSlot);
	~OxxiusLBX();
	
	int Initialize() ;
	// int Shutdown() ;

	int Fire(double deltaT) ;

	// int OnPowerSetPoint(MM::PropertyBase* pProp, MM::ActionType eAct) ;
	int OnCurrentSetPoint(MM::PropertyBase* pProp, MM::ActionType eAct) ;
	int OnEmissionOnOff(MM::PropertyBase* pProp, MM::ActionType eAct) ;
	int OnControlMode(MM::PropertyBase* pProp, MM::ActionType eAct) ;
	int OnAnalogMod(MM::PropertyBase* pProp, MM::ActionType eAct) ;
	int OnDigitalMod(MM::PropertyBase* pProp, MM::ActionType eAct) ;
	int OnState(MM::PropertyBase* pProp, MM::ActionType eAct) ;
	int OnAlarm(MM::PropertyBase* pProp, MM::ActionType eAct) ;
	// int OnFire(MM::PropertyBase* pProp, MM::ActionType eAct) ;

	int OnPowerReadback(MM::PropertyBase* , MM::ActionType ) { return DEVICE_OK; };
	int OnCurrentReadback(MM::PropertyBase* , MM::ActionType ) { return DEVICE_OK; };
	int OnOperatingMode(MM::PropertyBase* , MM::ActionType ) { return DEVICE_OK; };


private:
	bool initialized_;

	//// SPECIFIC TO LBX
	float maxRelPower_;
	float nominalPower_;
	float maxCurrent_;
	unsigned int waveLength;
	string state_;
	string alarm_;
	string controlMode_;
	string analogMod_;
	string digitalMod_;

	int mpa_number;
};
