/*
 * Project: ASIStage Device Adapter
 * License/Copyright: BSD 3-clause, see license.txt
 * Maintainers: Brandon Simpson (brandon@asiimaging.com)
 *              Jon Daniels (jon@asiimaging.com)
 */

#ifndef _ASICRIF_H_
#define _ASICRIF_H_

#include "ASIBase.h"

// CRIF reflection-based autofocusing unit (Nico, May 2007)
class CRIF : public CAutoFocusBase<CRIF>, public ASIBase
{
public:
	CRIF();
	~CRIF();

	// MMDevice API
	bool Busy();
	void GetName(char* pszName) const;

	int Initialize();
	int Shutdown();

	// AutoFocus API
	virtual int SetContinuousFocusing(bool state);
	virtual int GetContinuousFocusing(bool& state);
	virtual bool IsContinuousFocusLocked();
	virtual int FullFocus();
	virtual int IncrementalFocus();
	virtual int GetLastFocusScore(double& score);
	virtual int GetCurrentFocusScore(double& /*score*/) { return DEVICE_UNSUPPORTED_COMMAND; }
	virtual int GetOffset(double& offset);
	virtual int SetOffset(double offset);

	// action interface
	// ----------------
	int OnPort(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnFocus(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnWaitAfterLock(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
	int GetFocusState(std::string& focusState);
	int SetFocusState(std::string focusState);
	int SetPositionUm(double pos);
	int GetPositionUm(double& pos);

	bool justCalibrated_;
	std::string axis_;
	double stepSizeUm_;
	std::string focusState_;
	long waitAfterLock_;
};

#endif // _ASICRIF_H_
