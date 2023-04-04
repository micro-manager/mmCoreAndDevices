///////////////////////////////////////////////////////////////////////////////
// FILE:          SigmaBase.h
// PROJECT:       Micro-Manager 2.0
// SUBSYSTEM:     DeviceAdapters
//  
//-----------------------------------------------------------------------------
// DESCRIPTION:   SIGMA-KOKI device adapter 2.0
//                
// AUTHOR   :    Hiroki Kibata, Abed Toufik  Release Date :  05/02/2023
//
// COPYRIGHT:     SIGMA KOKI CO.,LTD, Tokyo, 2023
#ifndef _SIGMABASE_H_
#define _SIGMABASE_H_

#include <MMDevice.h>
#include <DeviceBase.h>
#include <ModuleInterface.h>
#include <ImgBuffer.h>
//////////////////////////////////////////////////////////////////////////////
// Error codes
//General
#define ERR_PORT_CHANGE_FORBIDDEN						10001
//ZStage
#define ERR_ZSTEGE_DEVICE_UNRECOGNIZABLE				10201
#define ERR_ZSTAGE_STEPSIZE_FAILED						10202
#define ERR_ZSTAGE_SPEED_FAILED							10203
#define ERR_ZSTAGE_SET_RESOLUTION_CLOSE_CONTROL_FAILED	10204
//XYStage
#define ERR_XYSTEGE_DEVICE_UNRECOGNIZABLE				10101
#define ERR_XYSTAGE_STEPSIZE_FAILED						10102
#define ERR_XYSTAGE_SPEED_FAILED						10103
//Shutter
#define ERR_SHUTTER_DEVICE_UNRECOGNIZABLE				10301
#define ERR_SHUTTER_INTERRUPT_FAILED					10302
#define ERR_SHUTTER_CONTROL_FAILED						10303
#define ERR_SHUTTER_STATE_FAILED						10304
#define ERR_SHUTTER_COMMANDSYSTEM_FAILED				10305
#define ERR_SHUTTER_ACTIONMODE_FAILED					10306
#define ERR_SHUTTER_MODEL_FAILED						10307
#define ERR_SHUTTER_MODELNAME_FAILED					10308
#define ERR_SHUTTER_DELAY_FAILED						10309

//////////////////////////////////////////////////////////////////////////////


class SigmaBase
{
public:
	SigmaBase(MM::Device *device);
	virtual ~SigmaBase();

protected:
	int ClearPort() const;

	int SendCommand(const std::string command);
	int RecieveData(std::string& data);
	int SendRecieve(const std::string command, std::string& data);
	int SendCheckRecievedOK(const std::string command);

	/*
	* [Added] - 2022-03-02, h.kibata@sigma-koki.com
	*/
	std::vector<std::string> split(std::string src, char del);

	bool initialized_;
	std::string port_;
	MM::Device* device_;
	MM::Core* core_;
};

#endif //_TEST_SIGMAKOKI_H_
