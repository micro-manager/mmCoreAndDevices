/*
 * Project: ASIStage Device Adapter
 * License/Copyright: BSD 3-clause, see license.txt
 * Maintainers: Brandon Simpson (brandon@asiimaging.com)
 *              Jon Daniels (jon@asiimaging.com)
 */

#pragma once

#include "ASIBase.h"

// CRISP Autofocus (Nico, Nov 2011)
class CRISP : public CAutoFocusBase<CRISP>, public ASIBase {
public:
	CRISP();
	~CRISP();

	// MM Device API
	bool Busy();
	void GetName(char* name) const;

	int Initialize();
	int Shutdown();

	bool SupportsDeviceDetection();
	MM::DeviceDetectionStatus DetectDevice();

	// MM AutoFocus API
	int SetContinuousFocusing(bool state);
	int GetContinuousFocusing(bool& state);
	bool IsContinuousFocusLocked();
	int FullFocus();
	int IncrementalFocus();
	int GetLastFocusScore(double& score) { return GetCurrentFocusScore(score); }
	int GetCurrentFocusScore(double& score);
	int GetOffset(double& offset);
	int SetOffset(double offset);

	// action interface
	int OnPort(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnAxis(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnFocus(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnNA(MM::PropertyBase* pProp, MM::ActionType eAct);
	int GetObjectiveNA(double& objNA);
	int OnLockRange(MM::PropertyBase* pProp, MM::ActionType eAct);
	int GetLockRange(double& lockRange);
	int OnLEDIntensity(MM::PropertyBase* pProp, MM::ActionType eAct);
	int GetLEDIntensity(long& ledIntensity);
	int OnNumAvg(MM::PropertyBase* pProp, MM::ActionType eAct);
	int GetNumAverages(long& numAverages);
	int OnCalGain(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnCalRange(MM::PropertyBase* pProp, MM::ActionType eAct);
	int GetCalRange(double& calRange);
	int OnGainMultiplier(MM::PropertyBase* pProp, MM::ActionType eAct);
	int GetGainMultiplier(long& gainMult);
	int OnFocusCurve(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnFocusCurveData(MM::PropertyBase* pProp, MM::ActionType eAct, long index);
	int OnNumSkips(MM::PropertyBase* pProp, MM::ActionType eAct);
	int GetNumSkips(long& updateRate);
	int OnInFocusRange(MM::PropertyBase* pProp, MM::ActionType eAct);
	int GetInFocusRange(double& inFocusRange);

private:
	int UpdateFocusState();
	int SetFocusState(const std::string& focusState);
	int ForceSetFocusState(const std::string& focusState);

	void LogFirmwareSupport(const bool hasLockQueries, const bool hasExShortcut) const;

	// Serial Communication Helpers
	int GetValue(const std::string& command, double& value);
	int SetCommand(const std::string& command);

	// Properties

	// Software-only
	void CreateWaitAfterLockProperty();
	// Read-only
	void CreateStateProperty();
	void CreateSNRProperty();
	void CreateLockOffsetProperty();
	void CreateSumProperty();
	void CreateDitherErrorProperty();
	void CreateLogAmpAGCProperty();
	// Advanced
	void CreateSetLogAmpAGCProperty();
	void CreateSetLockOffsetProperty();

	std::string axisLetter_ = "Z"; // determined by pre-init property
	std::string focusState_{};

	long waitAfterLock_ = 1000;
	int answerTimeoutMs_ = 1000;

	// cached properties
	long gainMultiplier_ = 0;
	long ledIntensity_ = 0;
	long numAverages_ = 0;
	long numSkips_ = 0; // update rate (milliseconds)
	double calibrationRange_ = 0; // microns
	double inFocusRange_ = 0; // microns
	double lockRange_ = 0; // millimeters
	double objectiveNA_ = 0;

	static constexpr int SIZE_OF_FC_ARRAY = 24;
	std::string focusCurveData_[SIZE_OF_FC_ARRAY]{};
};
