///////////////////////////////////////////////////////////////////////////////
//// FILE:    	ELL14.cpp
//// PROJECT:	MicroManage
//// SUBSYSTEM:  DeviceAdapters
////-----------------------------------------------------------------------------
//// DESCRIPTION: Implementation of the Thorlabs'Elliptec Rotation Mount ELL14
//// https://www.thorlabs.de/newgrouppage9.cfm?objectgroup_ID=12829
////           	 
//// AUTHOR: Manon Paillat, 2022
//// developped under the supervision of Florian Ströhl
//// Contact: florian.strohl@uit.no
//

#ifndef _ELL14_H_
#define _ELL14_H_

#include "../MMDevice/MMDevice.h"
#include "../MMDevice/DeviceBase.h"
#include <string>
#include <map>

//////////////////////////////////////////////////////////////////////////////
// Error codes
//
#define ERR_PORT_CHANGE_FORBIDDEN 101
#define ERR_UNEXPECTED_ANSWER 102
#define ERR_WRONG_DEVICE 103

// Device specific status
#define ERR_COMMUNICATION_TIME_OUT 201
#define ERR_MECHANICAL_TIME_OUT 202
#define ERR_COMMAND_ERROR_OR_NOT_SUPPORTED 203
#define ERR_VALUE_OUT_OF_RANGE 204
#define ERR_MODULE_ISOLATED 205
#define ERR_MODULE_OUT_OF_ISOLATION 206
#define ERR_INITIALIZING_ERROR 207
#define ERR_THERMAL_ERROR 208
#define ERR_BUSY 209
#define ERR_SENSOR_ERROR 210
#define ERR_MOTOR_ERROR 211
#define ERR_OUT_OF_RANGE 212
#define ERR_OVER_CURRENT_ERROR 213
#define ERR_UNKNOWN_ERROR 214

enum class rotDirection : char {
	FW = '0',
	BW = '1'
};

class ELL14 : public CGenericBase< ELL14>
{
public:
	ELL14();
	~ELL14();

	// Device API
	// ---------
	int Initialize();
	int Shutdown();
	bool Busy();
	void GetName(char* pszName) const;
	int getID(std::string* id, double* pulsesPerRev);

	// API
	int SetPosition(double pos);
	int GetPosition(double& pos);
	int SetRelativePosition(double d);
	int SetOffset(double pos);
	int GetOffset(double& offset);
	int SetJogStep(double jogStep);
	int GetJogStep(double& jogStep);

	int Home();
	int Jog();
	int SearchFrequencies();

	// Action Interface                                                                                                       	 
	int OnPort(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnChannel(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnPosition(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnHomeValue(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnHome(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnRelativeMove(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnJogStep(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnJog(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnSearchFrequencies(MM::PropertyBase* pProp, MM::ActionType eAct);

	//Convenient functions
	std::string positionFromValue(long val);
	int positionFromHex(std::string pos);
	int getStatusCode(std::string message);
	std::string removeLineFeed(std::string answer);
	bool isStatus(std::string message);
	std::string removeCommandFlag(std::string message);
	int receivePosition(double angle);
	double modulo2pi(double angle);

private:
	bool initialized_;
	std::string port_;
	std::string channel_;
	double pulsesPerRev_;
	double maxReplyTimeMs_;

	double pos_;                	// in °
	double offset_;             	//  "  
	double jogStep_;            	//  "  
	double relativeMove_;       	//  "
	rotDirection homeDir_;
	rotDirection jogDir_;
};

#endif // _ELL14_H_