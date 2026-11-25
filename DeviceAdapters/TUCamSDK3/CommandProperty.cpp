#include "CommandProperty.h"

using namespace std;

TuCommandProperty::TuCommandProperty(const string & MM_name,
                                     TUC_HDEVICE hDevice,
                                     const string & featureName,
                                     CMMTUCam * callback, bool readOnly)
: MM_name_(MM_name),
  m_hDevice(hDevice),
  featureName_(featureName),
  callback_(callback)
{
  initialise(readOnly);
}

void TuCommandProperty::initialise(bool readOnly)
{
	CPropertyAction *pAct = new CPropertyAction (this, &TuCommandProperty::OnCommand);
	callback_->CreateStringProperty(MM_name_.c_str(), MM_name_.c_str(), readOnly, pAct);
	callback_->AddAllowedValue(MM_name_.c_str(), MM_name_.c_str());
}

MM::ActionFunctor* TuCommandProperty::CreatePropertyAction()
{
  return new CPropertyAction(this, &TuCommandProperty::OnCommand);
}

TuCommandProperty::~TuCommandProperty()
{
}

int TuCommandProperty::OnCommand(MM::PropertyBase * pProp, MM::ActionType eAct)
{
	TUC_FEATURE_INFO info = {};
	TucGetFeatureInfoByName(m_hDevice, featureName_.c_str(), &info);

	if (eAct == MM::BeforeGet)
	{
		pProp->Set(MM_name_.c_str());
		callback_->PropertyReadOnly(pProp, featureName_.c_str(), info.accessMode);
	}
	else if (eAct == MM::AfterSet)
	{
		string val;
		pProp->Get(val);
		if (val.length() != 0)
		{
			TucExecuteFeatureCommand(m_hDevice, featureName_.c_str());
		}
	}

	return DEVICE_OK;
}
