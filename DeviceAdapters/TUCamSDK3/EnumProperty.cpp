#include "EnumProperty.h"

using namespace std;

TuEnumProperty::TuEnumProperty(const string & MM_name,
                               TUC_HDEVICE hDevice,
                               const string & featureName,
                               CMMTUCam* callback,
                               bool readOnly,
                               bool needsCallBack)
: MM_name_(MM_name),
  m_hDevice(hDevice),
  featureName_(featureName),
  callback_(callback),
  callBackRegistered_(needsCallBack)
{
	initialise(readOnly);
}

bool TuEnumProperty::CollectEnumValues(vector<string>& values)
{
	uint32_t count = 0;
	if (!TUC_STATUS_IS_SUCCESS(TucGetFeatureEnumCount(m_hDevice, featureName_.c_str(), &count)))
		return false;

	for (uint32_t i = 0; i < count; ++i)
	{
		char symbolic[256] = { 0 };
		uint32_t size = sizeof(symbolic);
		if (TUC_STATUS_IS_SUCCESS(TucGetFeatureEnumSymbolic(m_hDevice, featureName_.c_str(), i, symbolic, &size)))
		{
			values.push_back(symbolic);
		}
	}
	return !values.empty();
}

void TuEnumProperty::initialise(bool readOnly)
{
	CPropertyAction * pAct = new CPropertyAction(this, &TuEnumProperty::OnEnum);
	if (0 == MM_name_.compare("BinningSelector"))
	{
		callback_->CreateProperty(MM::g_Keyword_Binning, "", MM::String, readOnly, pAct);
	}
	else
	{
		callback_->CreateProperty(MM_name_.c_str(), "", MM::String, readOnly, pAct);
	}

	vector<string> values;
	CollectEnumValues(values);

	if (0 == MM_name_.compare("BinningSelector"))
	{
		callback_->ClearAllowedValues(MM::g_Keyword_Binning);
		callback_->SetAllowedValues(MM::g_Keyword_Binning, values);
	}
	else
	{
		callback_->ClearAllowedValues(MM_name_.c_str());
		callback_->SetAllowedValues(MM_name_.c_str(), values);
	}
}

void TuEnumProperty::UpdatePixelFormatAllowedValues()
{
	uint32_t count = 0;
	if (!TUC_STATUS_IS_SUCCESS(TucGetFeatureEnumCount(m_hDevice, "PixelFormat", &count)))
		return;

	vector<string> values;
	for (uint32_t i = 0; i < count; ++i)
	{
		char symbolic[256] = { 0 };
		uint32_t size = sizeof(symbolic);
		if (TUC_STATUS_IS_SUCCESS(TucGetFeatureEnumSymbolic(m_hDevice, "PixelFormat", i, symbolic, &size)))
		{
			values.push_back(symbolic);
		}
	}
	callback_->ClearAllowedValues("PixelFormat");
	callback_->SetAllowedValues("PixelFormat", values);
}

TuEnumProperty::~TuEnumProperty()
{
}

int TuEnumProperty::OnEnum(MM::PropertyBase * pProp, MM::ActionType eAct)
{
	TUC_FEATURE_INFO info = {};
	TucGetFeatureInfoByName(m_hDevice, featureName_.c_str(), &info);

	int64_t curVal = 0;
	TucGetFeatureEnum(m_hDevice, featureName_.c_str(), &curVal);

	if (eAct == MM::BeforeGet)
	{
		uint32_t count = 0;
		TucGetFeatureEnumCount(m_hDevice, featureName_.c_str(), &count);
		string curSymbolic;
		TUC_FEATURE_ENUM_ENTRY entry = {};
		for (uint32_t i = 0; i < count; ++i)
		{
			if (!TUC_STATUS_IS_SUCCESS(TucGetFeatureEnumEntryByIndex(m_hDevice, featureName_.c_str(), i, &entry)))
				continue;
		}

		if (TUC_STATUS_IS_SUCCESS(TucGetFeatureEnumEntryByIndex(m_hDevice, featureName_.c_str(), curVal, &entry)))
		{
			curSymbolic = entry.symbolic;
		}

		if (curSymbolic.empty())
		{
			char symbolic[256] = { 0 };
			uint32_t size = sizeof(symbolic);
			if (TUC_STATUS_IS_SUCCESS(TucGetFeatureEnumSymbolic(m_hDevice, featureName_.c_str(), 0, symbolic, &size)))
				curSymbolic = symbolic;
		}

		pProp->Set(curSymbolic.c_str());

		if (0 == MM_name_.compare("BinningSelector"))
		{
			callback_->PropertyChanged(MM::g_Keyword_Binning, curSymbolic.c_str());
		}

		if (0 == MM_name_.compare("SensorOperationMode"))
		{
			UpdatePixelFormatAllowedValues();
		}

		callback_->PropertyReadOnly(pProp, featureName_.c_str(), info.accessMode);
	}
	else if (eAct == MM::AfterSet)
	{
		string val;
		pProp->Get(val);
		if (val.length() != 0)
		{
			uint32_t count = 0;
			int64_t setVal = curVal;
	
			TucGetFeatureEnumCount(m_hDevice, featureName_.c_str(), &count);

			char symbolic[256] = { 0 };
			for (uint32_t i = 0; i < count; ++i)
			{
				uint32_t size = sizeof(symbolic);
				if (TUC_STATUS_IS_SUCCESS(TucGetFeatureEnumSymbolic(m_hDevice, featureName_.c_str(), i, symbolic, &size)))
				{
					if (0 == val.compare(symbolic))
					{
						setVal = i;
						break;
					}
				}
			}

			bool living = callback_->OnCapLiving();
			bool needRestart = (0 == MM_name_.compare("BinningSelector") ||
				info.accessMode != TUC_ACCESS_READ_WRITE);
			if (needRestart)
			{
				callback_->OnCaptureStop();
			}

			TucSetFeatureEnum(m_hDevice, featureName_.c_str(), setVal);

			if (needRestart)
			{
				callback_->OnCaptureStart();
				if (!living)
				{
					callback_->OnCaptureStop();
				}
			}

			if (0 == MM_name_.compare("PixelFormat") || 0 == MM_name_.compare("SensorOperationMode"))
			{
				callback_->UpdateImageBuffer();
			}
		}
	}

	return DEVICE_OK;
}
