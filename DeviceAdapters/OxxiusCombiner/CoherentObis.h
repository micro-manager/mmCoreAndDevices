#pragma once
#include "GenericLaser.h"
using namespace std;

class CoherentObis : public GenericLaser
{
public:
	CoherentObis(const char* nameAndSlot); //OK
	~CoherentObis(); //OK

	int Initialize(); //OK
	
	int Fire(double deltaT); 

	// int OnPowerSetPoint(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnCurrentSetPoint(MM::PropertyBase* pProp, MM::ActionType eAct) { return DEVICE_OK; };//Aucun pour l'instant
	int OnEmissionOnOff(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnControlMode(MM::PropertyBase* pProp, MM::ActionType eAct) { return DEVICE_OK; };//Aucun pour l'instant
	int OnAnalogMod(MM::PropertyBase* pProp, MM::ActionType eAct) { return DEVICE_OK; };//Aucun pour l'instant
	int OnDigitalMod(MM::PropertyBase* pProp, MM::ActionType eAct) { return DEVICE_OK; };//Aucun pour l'instant
	int OnState(MM::PropertyBase* pProp, MM::ActionType eAct) { return DEVICE_OK; };//Aucun pour l'instant
	int OnAlarm(MM::PropertyBase* pProp, MM::ActionType eAct);
	// int OnFire(MM::PropertyBase* pProp, MM::ActionType eAct);

	int OnPowerReadback(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnCurrentReadback(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnOperatingMode(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
	bool initialized_;

	//// SPECIFIC TO CoherentObis
	double maxPower_;
	double minPower_;
	string alarm_;
	double wavelength_;
	string description_;

};
