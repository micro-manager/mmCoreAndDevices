///////////////////////////////////////////////////////////////////////////////
// FILE:          ArduinoShutter.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   A basic Arduino-based shutter
//                
// AUTHOR:        Kyle M. Douglass, https://kylemdouglass.com
//
// VERSION:       0.0.0
//
// FIRMWARE:      xxx
//                
// COPYRIGHT:     ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland
//                Laboratory of Experimental Biophysics (LEB), 2025
//

#ifndef _ARDUINO_SHUTTER_H_
#define _ARDUINO_SHUTTER_H_

#include "DeviceBase.h"

class ArduinoShutter : public CShutterBase<ArduinoShutter>
{
public:
	ArduinoShutter();
	~ArduinoShutter();

	// MMDevice API
	int Initialize();
	int Shutdown();
	void GetName(char* name) const;
	bool Busy() { return false; };

	// MMShutter API
	int SetOpen(bool open = true);
	int GetOpen(bool& open);
	int Fire(double deltaT);

	// Pre-init action handlers
	int OnPort(MM::PropertyBase* pProp, MM::ActionType eAct);

	// Action handlers
	int OnResponseChange(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnStateChange(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
	// MM API
	bool initialized_;

	// Pre-init device properties
	std::string port_;

	// Controlled device properties
	bool open_;

	// MM Properties
	int GeneratePreInitProperties();
	int GenerateReadOnlyProperties();
	int GenerateControlledProperties();

	// Serial communications
	std::string msg_;
	std::string response_;
	const std::string CMD_TERM_ = "\n";
	const std::string ANS_TERM_ = "\n";

	int PurgeBuffer();
	int QueryDevice(std::string msg);
	int ReceiveMsg();
	int SendMsg(std::string msg);

	// Error codes
	const int ERR_PORT_CHANGE_FORBIDDEN = 101;
};

#endif // _ARDUINO_SHUTTER_H_