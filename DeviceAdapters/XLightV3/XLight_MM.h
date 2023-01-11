///////////////////////////////////////////////////////////////////////////////
// FILE:       XLIGHT3_MM.h
// PROJECT:    Micro-Manager
// SUBSYSTEM:  DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Crestoptics XLight adapter
//                                                                                     
// AUTHOR:        ing. S. Silvestri silvestri.salvatore.ing@gmaiil.com, 29/12/2020
//
//
// COPYRIGHT:     2021, Crestoptics s.r.l.
// LICENSE:       This file is distributed under the BSD license.
//                License text is included with the source distribution.
//
//                This file is distributed in the hope that it will be useful,
//                but WITHOUT ANY WARRANTY; without even the implied warranty
//                of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//                IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//                CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//                INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.
//


#ifndef _XLIGHTV3MM_H_
#define _XLIGHTV3MM_H_


#include "DeviceBase.h"

#include <string>


//////////////////////////////////////////////////////////////////////////////
// Error codes
//

#define ERR_NOT_CONNECTED           10001
#define ERR_COMMAND_CANNOT_EXECUTE  10002
#define ERR_COMMUNICATION           11003
#define ERR_COMMUNICATION_TIMEOUT   11004
#define ERR_COMMAND_EXECUTION_ERROR 11005


#define ERR_UNKNOWN_COMMAND          10006
#define ERR_UNKNOWN_POSITION         10007
#define ERR_HALT_COMMAND             10008

#define ERR_XLIGHT_NOT_FOUND		 11009

typedef enum {UNKNOW_FT=0, EMISSION_FT, DICHROIC_FT, EXCITATION_FT , SPINNING_SLIDER, CAMERA_SLIDER, SPINNING_MOTOR, EMISSION_IT, ILLUMINATION_IT} TDevicelType;

typedef enum {POS_VAL=0, RANGE_VAL, WORK_VAL} TValueType;

typedef enum {GETPOS_CMD=0, GETNUMPOS_CMD, SETPOS_CMD} TCmdType;


/*


#define ERR_UNKNOWN_MODE         102
#define ERR_UNKNOWN_POSITION     103
#define ERR_IN_SEQUENCE          104
#define ERR_SEQUENCE_INACTIVE    105
#define ERR_STAGE_MOVING         106
#define HUB_NOT_AVAILABLE        107

*/

// ==================================================================================================

typedef struct {
	bool Connected;
	bool Working;
	std::string PrefixCMD;
	long Value;
	long MaxValue;
	TDevicelType DeviceType_;
	std::string name_;
	std::string description_;
	long MinValue;

} TDeviceInfo;
// ==================================================================================================

class XLightHub : public HubBase<XLightHub>
{
public:
	XLightHub();
	~XLightHub() {}

	// Device API
	// ---------
	int Initialize();
	int Shutdown() ;
	void GetName(char* pName) const; 
	bool Busy() ;

	// HUB api
	int DetectInstalledDevices();

	int SendCmdString(std::string pcCmdTxt,  unsigned uCmdTmOut=0, unsigned uRetry=1);

	std::string GetInputStr();
	MM::DeviceDetectionStatus DetectDevice(void);
	static MMThreadLock& GetLock() {return lock_;}
	int ExecuteCmd(TCmdType eCmdType, TDeviceInfo* pDeviceInfo,long value=0);
  bool SupportsDeviceDetection(void);

private:

	int GetControllerInfo();
	int IsOnline(TDevicelType DeviceType);
	int GetIntFromAnswer(std::string cmdbase_str, std::string answ_str, bool *ans_present, int *answ_value);
	int GetDeviceValue(TDevicelType DeviceType, TValueType ValueType, int * iValue);
	std::string BuilCommand (TCmdType eCmdType,  TDeviceInfo* pDeviceInfo, int value);
	std::string BuilCommandBase (TCmdType eCmdType, TDeviceInfo* pDeviceInfo);
	int ParseAnswer(std::string pCmd, std::string pAnsw, TCmdType eCmdType , TDeviceInfo* pDeviceInfo);

	bool initialized_;
	bool busy_;
	std::string port_;
	static const int RCV_BUF_LENGTH = 1024;
	char rcvBuf_[RCV_BUF_LENGTH];

	static MMThreadLock lock_;
	int version_;

	int OnPort(MM::PropertyBase* pProp, MM::ActionType eAct);
	void SetPort(const char* port);
	void ClearAllRcvBuf();

};
// ==================================================================================================

class XLightStateDevice : public CStateDeviceBase<XLightStateDevice>
{
public:
	XLightStateDevice();
	~XLightStateDevice();

	// MMDevice API
	// ------------
	int Initialize();
	int Shutdown();

	void GetName(char* pszName) const;
	bool Busy();
	unsigned long GetNumberOfPositions()const;

	// action interface
	// ----------------
	int OnState(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnNumberOfStates(MM::PropertyBase* pProp, MM::ActionType eAct);
	TDeviceInfo DeviceInfo_;

private:

	bool initialized_;
	int setPosition (long lValue);
	int getPosition ();
	int getPositionsNumber ();
};

// ==================================================================================================
class IrisDevice : public CGenericBase<IrisDevice>
{
public:
	IrisDevice();
	~IrisDevice();

	// MMDevice API
	// ------------
	int Initialize();
	int Shutdown();

	void GetName(char* pszName) const;
	bool Busy();

	// action interface
	// ----------------
	int OnSetAperture(MM::PropertyBase* pProp, MM::ActionType eAct);
	void setIrisType(TDevicelType eFilterType);
	TDeviceInfo DeviceInfo_;

private:
	bool initialized_;
	int setaperture (long lValue);


};
// ==================================================================================================


#endif