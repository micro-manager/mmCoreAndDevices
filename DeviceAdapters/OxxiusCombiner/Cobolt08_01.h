#pragma once
#include "GenericLaser.h"
using namespace std;

class Cobolt08_01 : public GenericLaser
{
public:
	Cobolt08_01(const char* nameAndSlot); //OK
	~Cobolt08_01(); //OK

	int Initialize(); //OK
	int Shutdown(); //OK

	int Fire(double deltaT); //OK

	int OnPowerSetPoint(MM::PropertyBase* pProp, MM::ActionType eAct) { return DEVICE_OK; };//Aucun pour l'instant
	int OnCurrentSetPoint(MM::PropertyBase* pProp, MM::ActionType eAct) { return DEVICE_OK; };//Aucun pour l'instant
	int OnEmissionOnOff(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnControlMode(MM::PropertyBase* pProp, MM::ActionType eAct) { return DEVICE_OK; };//Aucun pour l'instant
	int OnAnalogMod(MM::PropertyBase* pProp, MM::ActionType eAct) { return DEVICE_OK; };//Aucun pour l'instant
	int OnDigitalMod(MM::PropertyBase* pProp, MM::ActionType eAct) { return DEVICE_OK; };//Aucun pour l'instant
	int OnState(MM::PropertyBase* pProp, MM::ActionType eAct) { return DEVICE_OK; };//Aucun pour l'instant
	int OnAlarm(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnFire(MM::PropertyBase* pProp, MM::ActionType eAct) { return DEVICE_OK; };//Aucun pour l'instant

	int OnPowerReadback(MM::PropertyBase* pProp, MM::ActionType eAct) { return DEVICE_OK; };//Aucun pour l'instant
	int OnCurrentReadback(MM::PropertyBase* pProp, MM::ActionType eAct) { return DEVICE_OK; };//Aucun pour l'instant
	int OnOperatingMode(MM::PropertyBase* pProp, MM::ActionType eAct) { return DEVICE_OK; };//Aucun pour l'instant

private:
	bool initialized_;

	//// SPECIFIC TO Cobolt08_01
	bool laserOn_;
	//double maxPower_;
	//double minPower_;
	string alarm_;
	//double wavelength_;
	//string description_;

};