///////////////////////////////////////////////////////////////////////////////
// FILE:          Shutter.h
// PROJECT:       Micro-Manager 2.0
// SUBSYSTEM:     DeviceAdapters
//  
//-----------------------------------------------------------------------------
// DESCRIPTION:   SIGMA-KOKI device adapter 2.0
//                
// AUTHOR   :    Hiroki Kibata, Abed Toufik  Release Date :  05/02/2023
//
// COPYRIGHT:     SIGMA KOKI CO.,LTD, Tokyo, 2023
#pragma once

#include "SigmaBase.h"
using namespace std;

extern const char* g_ShutterDeviceName_C2B1;
extern const char* g_ShutterDeviceName_C2B2;
extern const char* g_ShutterDeviceName_C4B1;
extern const char* g_ShutterDeviceName_C4B2;
extern const char* g_ShutterDeviceName_C4B3;
extern const char* g_ShutterDeviceName_C4B4;


class Shutter : public CShutterBase<Shutter>, public SigmaBase
{
public:
	Shutter(const char* name, int channel);
	~Shutter();

	bool Busy();
	void GetName(char* pszName) const;
	int Initialize();
	int Shutdown();

	// Shutter API
	int SetOpen(bool open = true);
	int GetOpen(bool& open);
	int Fire(double deltaT);

	// Action Interface
	// ----------------
	int OnState(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnPort(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnDelay(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnModel(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
	/// <summary>
	/// Controller model
	/// </summary>
	enum ShutterModel
	{
		C2B,	// SSH-C2B
		C4B		// SSH-C4B
	};

	int SetShutterPosition(bool state);
	int GetShutterPosition(bool& state);
	int SetDeviceID(ShutterModel& model);
	int SetCommandSystem();
	int SetShutterActionMode();
	int SetShutterModel(const std::string model);
	int GetShutterName(std::string& name, int index);
	int SetInterruptPacketState();
	int SetControllerDelay(double val);

	// bool initialized_;
	std::string name_;
	std::string modelType_;
	std::string defUser1_;
	std::string defUser2_;
	std::string defUser3_;
	ShutterModel model_;
	int channel_;
	MM::MMTime changedTime_;
};
