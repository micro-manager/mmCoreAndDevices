///////////////////////////////////////////////////////////////////////////////
// FILE:          ZStage.h
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
extern const char* g_ZStageDeviceName;

class ZStage : public CStageBase<ZStage>, public SigmaBase
{
public:
	ZStage();
	~ZStage();

	// Device API
	// ----------
	bool Busy();
	void GetName(char* pszName) const;
	int Initialize();
	int Shutdown();

	// Stage API
	// ---------
	int GetPositionUm(double& pos);
	int GetPositionSteps(long& steps);
	int SetPositionUm(double pos);
	int SetPositionSteps(long steps) { return DEVICE_OK; };
	int SetOrigin();
	int GetLimits(double& min, double& max);

	int IsStageSequenceable(bool& isSequenceable) const { isSequenceable = false; return DEVICE_OK; }
	bool IsContinuousFocusDrive() const { return false; }

	// Action Interface
	// ----------------
	int OnPort(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnChannel(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnFullStepSize(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnStepSizeZ(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnDivision(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnSpeed(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnControlModel(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnPulseRateZ(MM::PropertyBase* pProp, MM::ActionType eAct);
private:
	/// <summary>
	/// Controller Model
	/// </summary>
	enum ZStageModel
	{
		SHOT1,	//GIP-101, TAF-C01
		SHOT2,	//SHOT-702, 302GS
		SHOT4,	//SHOT-304GS
		FINE1,	//FINE-01
		FINE3,	//FINE-503
		HSC3,	//HSC-103
		SHRC3	//HRSC-203     change to SHRC
	};

	int SetDeviceModel(); // change from [ SetDeviceModel(XYStageModel& model)] no needed 
	void CreateDivisionProp();
	void AddAllowedDivisionPropXY(const char* div);   // Added 5åé12ì˙2022   t.abed
	void CreateChanelProp();
	int SetDivision(int division);
	int SetSpeed(int val);
	int UpdatePositionZ();    // Added 4åé7ì˙2022Å@Å@Å@t.abed
	int DriveCommadProcessZ(double position);// Added 4åé7ì˙2022   t.abed
	long GetSlowSpeedPulse(long pls);

	ZStageModel model_;
	std::string controlMode_;
	std::string channel_;
	int speed_;
	double stepsZ_; // Z stage current position Added 4åé7ì˙2022   t.abed
	double fullstepSizeZum_;
	double stepSizeZum_;
	std::string divisionZ_;
	double answerTimeoutMs_;
	int PlsRateZ;
};

