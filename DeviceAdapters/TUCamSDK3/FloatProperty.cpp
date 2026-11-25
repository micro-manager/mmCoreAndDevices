#include "FloatProperty.h"

using namespace std;

TuFloatProperty::TuFloatProperty(const string & MM_name,
                                 TUC_HDEVICE hDevice,
                                 const string & featureName,
                                 double minVal, double maxVal,
                                 CMMTUCam* callback,
                                 bool readOnly, bool needsCallBack)
: MM_name_(MM_name),
  m_hDevice(hDevice),
  featureName_(featureName),
  callback_(callback),
  callBackRegistered_(needsCallBack)
{
	CPropertyAction * pAct = new CPropertyAction (this, &TuFloatProperty::OnFloat);
	callback->CreateProperty(MM_name.c_str(), "1", MM::Float, readOnly, pAct);
	callback->SetPropertyLimits(MM_name.c_str(), minVal, maxVal + 0.001);
}

TuFloatProperty::~TuFloatProperty()
{
}

int TuFloatProperty::OnFloat(MM::PropertyBase * pProp, MM::ActionType eAct)
{
	double dbVal = 0;
	TUC_FEATURE_INFO info = {};
	TucGetFeatureInfoByName(m_hDevice, featureName_.c_str(), &info);

	TUC_FEATURE_FLOAT_INFO floatInfo = {};
	TucGetFeatureFloatInfo(m_hDevice, featureName_.c_str(), &floatInfo);

	if (eAct == MM::BeforeGet)
	{
		callback_->SetPropertyLimits(MM_name_.c_str(), floatInfo.minimum, floatInfo.maximum + 0.001);
		pProp->Set(floatInfo.currentValue);
		callback_->PropertyReadOnly(pProp, featureName_.c_str(), info.accessMode);
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(dbVal);
		if (dbVal < floatInfo.minimum)
			dbVal = floatInfo.minimum;
		else if (dbVal > floatInfo.maximum)
			dbVal = floatInfo.maximum;

		bool living = callback_->OnCapLiving();
		bool needRestart = (info.accessMode != TUC_ACCESS_READ_WRITE);
		if (needRestart)
		{
			callback_->OnCaptureStop();
		}

		TucSetFeatureFloat(m_hDevice, featureName_.c_str(), dbVal);

		if (needRestart)
		{
			callback_->OnCaptureStart();
			if (!living)
			{
				callback_->OnCaptureStop();
			}
		}
	}

	return DEVICE_OK;
}
