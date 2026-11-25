#include "BooleanProperty.h"

using namespace std;

TuBooleanProperty::TuBooleanProperty(const string &MM_name,
                                     TUC_HDEVICE hDevice,
                                     const string & featureName,
                                     CMMTUCam *callback, bool readOnly)
: MM_name_(MM_name),
  m_hDevice(hDevice),
  featureName_(featureName),
  callback_(callback)
{
	initialise(readOnly);
}

void TuBooleanProperty::initialise(bool readOnly)
{
	MM::ActionFunctor * pAct = CreatePropertyAction();
	callback_->CreateProperty(MM_name_.c_str(), g_StatusON, MM::String, readOnly, pAct);

	vector<string> values;
	values.push_back(g_StatusOFF);
	values.push_back(g_StatusON);
	callback_->SetAllowedValues(MM_name_.c_str(), values);
}

MM::ActionFunctor* TuBooleanProperty::CreatePropertyAction()
{
	return new CPropertyAction(this, &TuBooleanProperty::OnBoolean);
}

TuBooleanProperty::~TuBooleanProperty()
{
}

int TuBooleanProperty::OnBoolean(MM::PropertyBase * pProp, MM::ActionType eAct)
{
	TUC_FEATURE_INFO info = {};
	TucGetFeatureInfoByName(m_hDevice, featureName_.c_str(), &info);

	bool cur = false;
	TucGetFeatureBool(m_hDevice, featureName_.c_str(), &cur);

	if (eAct == MM::BeforeGet)
	{
		pProp->Set(cur ? g_StatusON : g_StatusOFF);
		callback_->PropertyReadOnly(pProp, featureName_.c_str(), info.accessMode);
	}
	else if (eAct == MM::AfterSet)
	{
		string val;
		pProp->Get(val);
		if (val.length() != 0)
		{
			bool setVal = (0 == val.compare(g_StatusON));

			bool living = callback_->OnCapLiving();
			bool needRestart = (info.accessMode != TUC_ACCESS_READ_WRITE);
			if (needRestart)
			{
				callback_->OnCaptureStop();
			}

			TucSetFeatureBool(m_hDevice, featureName_.c_str(), setVal);

			if (needRestart)
			{
				callback_->OnCaptureStart();
				if (!living)
				{
					callback_->OnCaptureStop();
				}
			}
		}
	}

	return DEVICE_OK;
}
