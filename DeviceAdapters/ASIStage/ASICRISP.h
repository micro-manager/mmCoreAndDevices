/*
 * Project: ASIStage Device Adapter
 * License/Copyright: BSD 3-clause, see license.txt
 * Maintainers: Brandon Simpson (brandon@asiimaging.com)
 *              Jon Daniels (jon@asiimaging.com)
 */

#ifndef _ASICRISP_H_
#define _ASICRISP_H_

#include "ASIBase.h"

// CRISP reflection based autofocusing unit (Nico, Nov 2011)
class CRISP : public CAutoFocusBase<CRISP>, public ASIBase
{
public:
	CRISP();
	~CRISP();

	// MMDevice API
	bool Busy();
	void GetName(char* pszName) const;

	int Initialize();
	int Shutdown();

	bool SupportsDeviceDetection(void);
	MM::DeviceDetectionStatus DetectDevice(void);

	// AutoFocus API
	virtual int SetContinuousFocusing(bool state);
	virtual int GetContinuousFocusing(bool& state);
	virtual bool IsContinuousFocusLocked();
	virtual int FullFocus();
	virtual int IncrementalFocus();
	virtual int GetLastFocusScore(double& score);
	virtual int GetCurrentFocusScore(double& score);
	virtual int GetOffset(double& offset);
	virtual int SetOffset(double offset);

	// action interface
	// ----------------
	int OnPort(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnAxis(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnFocus(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnNA(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnWaitAfterLock(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnLockRange(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnLEDIntensity(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnNumAvg(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnCalGain(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnGainMultiplier(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnFocusCurve(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnFocusCurveData(MM::PropertyBase* pProp, MM::ActionType eAct, long index);
	int OnSNR(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnDitherError(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnLogAmpAGC(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnNumSkips(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnInFocusRange(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnSum(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnOffset(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnState(MM::PropertyBase* pProp, MM::ActionType eAct);
	// For backwards compatibility with MS2000 firmware < 9.2o
	int OnSumLegacy(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnDitherErrorLegacy(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
	int GetFocusState(std::string& focusState);
	int SetFocusState(std::string focusState);
	int ForceSetFocusState(std::string focusState);
	int GetValue(std::string cmd, float& val);
	int SetCommand(std::string cmd);

	static const int SIZE_OF_FC_ARRAY = 24;
	std::string focusCurveData_[SIZE_OF_FC_ARRAY];
	std::string axis_;
	std::string focusState_;
	long waitAfterLock_;
	int answerTimeoutMs_;
};

#endif // end _ASICRISP_H_
