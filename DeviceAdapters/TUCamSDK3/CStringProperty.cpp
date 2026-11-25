#include "CstringProperty.h"
#include "XTDef.h"

using namespace std;

TuCstringProperty::TuCstringProperty(const string & MM_name,
                                     TUC_HDEVICE hDevice,
                                     const string & featureName,
                                     TUC_FEATURE_TYPE featureType,
                                     TUC_FEATURE_REPRESENTATION representation,
                                     const string & initialValue,
                                     CMMTUCam * callback, bool readOnly)
: MM_name_(MM_name),
  m_hDevice(hDevice),
  featureName_(featureName),
  featureType_(featureType),
  representation_(representation),
  initialValue_(initialValue),
  callback_(callback)
{
  initialise(readOnly);
}

void TuCstringProperty::initialise(bool readOnly)
{
	CPropertyAction *pAct = new CPropertyAction (this, &TuCstringProperty::OnStringFun);
	if (TUC_REPRESENTATION_MACADDRESS == representation_ ||
		TUC_REPRESENTATION_HEXNUMBER == representation_ ||
		TUC_REPRESENTATION_IPV4ADDRESS == representation_)
	{
		callback_->CreateStringProperty(MM_name_.c_str(), "Int", readOnly, pAct);
	}
	else
	{
		callback_->CreateStringProperty(MM_name_.c_str(), initialValue_.c_str(), readOnly, pAct);
		if (0 != MM_name_.compare("DeviceUserID"))
		{
			callback_->AddAllowedValue(MM_name_.c_str(), initialValue_.c_str());
		}
	}
}

TuCstringProperty::~TuCstringProperty()
{
}

int TuCstringProperty::OnStringFun(MM::PropertyBase * pProp, MM::ActionType eAct)
{
	TUC_FEATURE_INFO info = {};
	TucGetFeatureInfoByName(m_hDevice, featureName_.c_str(), &info);

	char strBuf[1024] = { 0 };
	uint32_t strSize = sizeof(strBuf);
	int64_t intVal = 0;

	bool isSpecialInt = (TUC_REPRESENTATION_MACADDRESS == representation_ ||
		TUC_REPRESENTATION_HEXNUMBER == representation_ ||
		TUC_REPRESENTATION_IPV4ADDRESS == representation_ ||
		featureType_ == TUC_FEATURE_TYPE_INTEGER);

	if (isSpecialInt && featureType_ == TUC_FEATURE_TYPE_INTEGER)
	{
		TucGetFeatureInt64(m_hDevice, featureName_.c_str(), &intVal);
	}
	else
	{
		TucGetFeatureString(m_hDevice, featureName_.c_str(), strBuf, &strSize);
	}

	if (eAct == MM::BeforeGet)
	{
		CString str;
		if (TUC_REPRESENTATION_HEXNUMBER == representation_)
		{
			str.Format("%X", (unsigned)intVal);
			pProp->Set(str.GetBuffer());
		}
		else if (TUC_REPRESENTATION_IPV4ADDRESS == representation_)
		{
			str = IntToIP(intVal);
			pProp->Set(str.GetBuffer());
		}
		else if (TUC_REPRESENTATION_MACADDRESS == representation_)
		{
			str = IntToMac(intVal);
			pProp->Set(str.GetBuffer());
		}
		else
		{
			pProp->Set(strBuf);
		}

		callback_->PropertyReadOnly(pProp, featureName_.c_str(), info.accessMode);
	}
	else if (eAct == MM::AfterSet)
	{
		string val;
		pProp->Get(val);
		if (TUC_REPRESENTATION_HEXNUMBER == representation_)
		{
			TucSetFeatureInt64(m_hDevice, featureName_.c_str(), StrHexToDec(val.c_str()));
		}
		else if (TUC_REPRESENTATION_IPV4ADDRESS == representation_)
		{
			TucSetFeatureInt64(m_hDevice, featureName_.c_str(), IPToInt(val.c_str()));
		}
		else if (TUC_REPRESENTATION_MACADDRESS == representation_)
		{
			TucSetFeatureInt64(m_hDevice, featureName_.c_str(), MacToInt(val.c_str()));
		}
		else if (0 == MM_name_.compare("DeviceUserID"))
		{
			TucSetFeatureString(m_hDevice, featureName_.c_str(), val.c_str());
		}
		else
		{
			pProp->Set(strBuf);
		}
	}

	return DEVICE_OK;
}
