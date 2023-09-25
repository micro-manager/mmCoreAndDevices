#pragma once
#include "OxxiusCombinerHub.h"
using namespace std;

#define	POW_UPPER_LIMIT 100.0

class GenericLaser : public CShutterBase<GenericLaser>
{
public:
	virtual int Initialize()=0;
	int Shutdown();

	virtual void GetName(char* pszName) const;
	virtual bool Busy();
	virtual int SetOpen(bool openCommand = true);
	virtual int GetOpen(bool& isOpen);
	virtual int Fire(double deltaT) = 0;

	int OnPowerSetPoint(MM::PropertyBase* pProp, MM::ActionType eAct);
	virtual int OnCurrentSetPoint(MM::PropertyBase* , MM::ActionType ) = 0;
	virtual int OnEmissionOnOff(MM::PropertyBase* , MM::ActionType ) = 0;
	virtual int OnControlMode(MM::PropertyBase* , MM::ActionType ) = 0;
	virtual int OnAnalogMod(MM::PropertyBase* , MM::ActionType ) = 0;
	virtual int OnDigitalMod(MM::PropertyBase* , MM::ActionType ) = 0;
	virtual int OnState(MM::PropertyBase* , MM::ActionType ) = 0;
	virtual int OnAlarm(MM::PropertyBase* , MM::ActionType ) = 0;
	int OnFire(MM::PropertyBase* , MM::ActionType );

	virtual int OnPowerReadback(MM::PropertyBase* , MM::ActionType ) = 0;
	virtual int OnCurrentReadback(MM::PropertyBase* , MM::ActionType ) = 0;
	virtual int OnOperatingMode(MM::PropertyBase* , MM::ActionType ) = 0;

protected: // Common data to all lasers
	string serialNumber_;
	string name_;
	unsigned int slot_;
	OxxiusCombinerHub* parentHub_;
	bool busy_;
	bool laserOn_;

private:
	bool initialized_;
};