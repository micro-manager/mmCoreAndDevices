///////////////////////////////////////////////////////////////////////////////
// FILE:          XYStage.h
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
extern const char* g_XYStageDeviceName;


class XYStage : public CXYStageBase<XYStage>, public SigmaBase
{
public:
	XYStage();
	~XYStage();

	//Device API
	//----------
	bool Busy();
	void GetName(char* pszName) const;
	int Initialize();
	int Shutdown();

	// XYStage API
	//------------
	int GetPositionUm(double& x, double& y);
	int GetPositionSteps(long& x, long& y);
	int SetPositionSteps(long x, long y);
	int SetRelativePositionSteps(long x, long y);
	int Home();
	int Stop();
	int SetOrigin();
	int GetLimitsUm(double& xMin, double& xMax, double& yMin, double& yMax);
	int GetStepLimits(long& xMin, long& xMax, long& yMin, long& yMax);
	double GetStepSizeXUm() { return stepSizeXum_; }
	double GetStepSizeYUm() { return stepSizeYum_; }
	int IsXYStageSequenceable(bool& isSequenceable) const { isSequenceable = false; return DEVICE_OK; }

	// Action Interface
	// ----------------
	int OnPort(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnChannelX(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnChannelY(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnStepSizeX(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnStepSizeY(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnSpeedXY(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnFullStepSizeX(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnFullStepSizeY(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnDivisionX(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnDivisionY(MM::PropertyBase* pProp, MM::ActionType eAct);
	// test 
	int onPulseRateX(MM::PropertyBase* pProp, MM::ActionType eAct);
	int onPulseRateY(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
	/// <summary>
	/// Controller model
	/// </summary>
	enum XYStageModel
	{
		SHOT2,	//SHOT-702, 302GS
		SHOT4,	//SHOT-304GS
		FC2,	//FC-Series
		HSC3,	//HSC-103
		SHRC3	//HRSC-203    SHRC changed
	};

	void CreateDivisionPropXY();
	void CreateFullStepPropXY();
	void AddAllowedDivisionPropXY(const char* div);    // Added 5ŒŽ12“ú2022   t.abed
	int SetDeviceModel();   // change from [ SetDeviceModel(XYStageModel& model)] no needed 
	int SetDivision(int channel, int division);
	int SetSpeedXY(int vxy);
	long GetSlowSpeedPulse(long fast);
	int UpdatePosition();
	void AssignmentOtherChannels();
	string ToDriveCommand(long x, long y, bool is_abs);
	int DriveCommandProcess(long x, long y, bool is_abs);
	int GenericCommandProcess(string command_header);
	int PulseSpeedSetting(int val);   // Added 12/4/2022  t.abed

	XYStageModel model_;
	int channelX_;
	int channelY_;
	int channelA_;
	int channelB_;
	long positionXpulse_;
	long positionYpulse_;
	long positionApulse_;
	long positionBpulse_;
	long slow_pulse;// for speed setting
	long fast_pulse;// for speed setting
	int speedXYum_;
	double fullStepSizeXum_;
	double fullStepSizeYum_;
	double stepSizeXum_;
	double stepSizeYum_;
	int divisionX_;
	int divisionY_;
	double answerTimeoutMs_;
	bool isBusyHomeShot4_;//Busy flag for Home in case of Shot4
	string fcModel_;
	int PlsRate1; // pulse rate StageX for HSC controller
	int PlsRate2; // pulse rate STageY for HSC controller

};
