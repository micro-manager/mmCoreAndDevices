#pragma once
#include "GenericLaser.h"
using namespace std;

class OxxiusLCX : public GenericLaser
{
public:
	OxxiusLCX(const char* nameAndSlot);
	~OxxiusLCX();
	
	int Initialize();
	
	int Fire(double deltaT);
	
	// int OnPowerSetPoint(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnCurrentSetPoint(MM::PropertyBase* pProp, MM::ActionType eAct) { return DEVICE_OK; }; //never called because it's a LCX laser
	int OnEmissionOnOff(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnControlMode(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnAnalogMod(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnDigitalMod(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnState(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnAlarm(MM::PropertyBase* pProp, MM::ActionType eAct);
	// int OnFire(MM::PropertyBase* pProp, MM::ActionType eAct);

	int OnPowerReadback(MM::PropertyBase* pProp, MM::ActionType eAct) { return DEVICE_OK; };
	int OnCurrentReadback(MM::PropertyBase* pProp, MM::ActionType eAct) { return DEVICE_OK; };
	int OnOperatingMode(MM::PropertyBase* pProp, MM::ActionType eAct) { return DEVICE_OK; };

private:
	bool initialized_;
	
	//// SPECIFIC TO LCX
	float maxRelPower_;
	float nominalPower_;
	float maxCurrent_;
	unsigned int waveLength;
	string state_;
	string alarm_;
	string controlMode_;
	string analogMod_;
	string digitalMod_;

	bool link_AOM1;//->LCX linked to a AOM number 1
	bool link_AOM2;//->LCX linked to a AOM number 2
	bool pow_adj; //-> LCX with power adjustment
	int mpa_number; //-> LCX linked to mpa number n
};