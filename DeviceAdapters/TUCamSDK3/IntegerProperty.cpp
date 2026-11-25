#include "IntegerProperty.h"

using namespace std;

TuIntegerProperty::TuIntegerProperty(const string & MM_name,
                                     TUC_HDEVICE hDevice,
                                     const string & featureName,
                                     int64_t minVal, int64_t maxVal,
                                     CMMTUCam* callback,
                                     bool readOnly,
                                     bool needsCallBack)
: MM_name_(MM_name),
  m_hDevice(hDevice),
  featureName_(featureName),
  callback_(callback),
  callBackRegistered_(needsCallBack)
{
	CPropertyAction * pAct = new CPropertyAction(this, &TuIntegerProperty::OnInteger);

	if (0 == MM_name.compare(MM::g_Keyword_Exposure))
	{
		callback->CreateProperty(MM_name.c_str(), "1", MM::Float, readOnly, pAct);
		callback->SetPropertyLimits(MM_name.c_str(), 0, 2147483.647);
	}
	else
	{
		callback->CreateProperty(MM_name.c_str(), "1", MM::Integer, readOnly, pAct);
		callback->SetPropertyLimits(MM_name.c_str(), (double)minVal, (double)maxVal);
	}
}

TuIntegerProperty::~TuIntegerProperty()
{
}

int TuIntegerProperty::OnInteger(MM::PropertyBase * pProp, MM::ActionType eAct)
{
	long lVal = 0;
	TUC_FEATURE_INFO info = {};
	TucGetFeatureInfoByName(m_hDevice, featureName_.c_str(), &info);

	TUC_FEATURE_INT64_INFO intInfo = {};
	TucGetFeatureInt64Info(m_hDevice, featureName_.c_str(), &intInfo);

	if (0 == MM_name_.compare(MM::g_Keyword_Exposure))
	{
		if (eAct == MM::BeforeGet)
		{
			pProp->Set((double)(intInfo.currentValue / 1000.0));
			callback_->PropertyReadOnly(pProp, featureName_.c_str(), info.accessMode);
		}
		else if (eAct == MM::AfterSet)
		{
			double dbVal = 0;
			pProp->Get(dbVal);
			TucSetFeatureInt64(m_hDevice, featureName_.c_str(), (int64_t)(dbVal * 1000.0));
		}
		return DEVICE_OK;
	}

	if (eAct == MM::BeforeGet)
	{
		callback_->SetPropertyLimits(MM_name_.c_str(), (double)intInfo.minimum, (double)intInfo.maximum);
		pProp->Set((long)intInfo.currentValue);
		callback_->PropertyReadOnly(pProp, featureName_.c_str(), info.accessMode);
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(lVal);
		if (lVal < intInfo.minimum)
			lVal = (long)intInfo.minimum;
		else if (lVal > intInfo.maximum)
			lVal = (long)intInfo.maximum;

		int64_t step = (intInfo.increment != 0) ? intInfo.increment : 1;
		int64_t setVal = (lVal / step) * step;

		bool living = callback_->OnCapLiving();
		bool needRestart = (info.accessMode != TUC_ACCESS_READ_WRITE);
		if (needRestart)
		{
			callback_->OnCaptureStop();
		}

		TucSetFeatureInt64(m_hDevice, featureName_.c_str(), setVal);

		if (needRestart)
		{
			callback_->OnCaptureStart();
			if (!living)
			{
				callback_->OnCaptureStop();
			}
		}

		if (0 == MM_name_.compare("Width") || 0 == MM_name_.compare("Height"))
		{
			callback_->UpdateImageBuffer();
		}
	}

	return DEVICE_OK;
}
