/*
 * Project: ASIStage Device Adapter
 * License/Copyright: BSD 3-clause, see license.txt
 * Maintainers: Brandon Simpson (brandon@asiimaging.com)
 *              Jon Daniels (jon@asiimaging.com)
 */

#ifndef ASICRISP_H
#define ASICRISP_H

#include "ASIBase.h"

// CRISP reflection based autofocusing unit (Nico, Nov 2011)
class CRISP : public CAutoFocusBase<CRISP>, public ASIBase
{
public:
	CRISP();
	~CRISP();

	// MMDevice API
	bool Busy();
	void GetName(char* name) const;

	int Initialize();
	int Shutdown();

	bool SupportsDeviceDetection();
	MM::DeviceDetectionStatus DetectDevice();

	// AutoFocus API
	int SetContinuousFocusing(bool state);
	int GetContinuousFocusing(bool& state);
	bool IsContinuousFocusLocked();
	int FullFocus();
	int IncrementalFocus();
	int GetLastFocusScore(double& score);
	int GetCurrentFocusScore(double& score);
	int GetOffset(double& offset);
	int SetOffset(double offset);

	// action interface
	int OnPort(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnAxis(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnFocus(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnNA(MM::PropertyBase* pProp, MM::ActionType eAct);
	int GetObjectiveNA(double& objNA);
	int OnWaitAfterLock(MM::PropertyBase* pProp, MM::ActionType eAct);
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
	int OnSNR(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnLogAmpAGC(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnNumSkips(MM::PropertyBase* pProp, MM::ActionType eAct);
	int GetNumSkips(long& updateRate);
	int OnInFocusRange(MM::PropertyBase* pProp, MM::ActionType eAct);
	int GetInFocusRange(double& inFocusRange);
	int OnOffset(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnState(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnSetLogAmpAGC(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnSetLockOffset(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
	int GetFocusState(std::string& focusState);
	int SetFocusState(const std::string& focusState);
	int ForceSetFocusState(const std::string& focusState);
	int GetValue(const std::string& command, double& value);
	int SetCommand(const std::string& command);

	// Properties
	void CreateSumProperty();
	void CreateDitherErrorProperty();

	std::string axis_;
	std::string focusState_;
	long waitAfterLock_;
	int answerTimeoutMs_;

	// cached properties
	long gainMultiplier_;
	long ledIntensity_;
	long numAverages_;
	long numSkips_; // update rate (milliseconds)
	double calibrationRange_; // microns
	double inFocusRange_; // microns
	double lockRange_; // millimeters
	double objectiveNA_;

	static constexpr int SIZE_OF_FC_ARRAY = 24;
	std::string focusCurveData_[SIZE_OF_FC_ARRAY];
};

#endif // ASICRISP_H
