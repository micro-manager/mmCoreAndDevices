///////////////////////////////////////////////////////////////////////////////
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// AUTHOR:        Nenad Amodaj, 2018
// COPYRIGHT:     JAI
//
// DISCLAIMER:    This file is provided WITHOUT ANY WARRANTY;
//                without even the implied warranty of MERCHANTABILITY or
//                FITNESS FOR A PARTICULAR PURPOSE.
//
//                IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//                CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//                INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.
//

#include "JAI.h"
#include <PvInterface.h>
#include <PvDevice.h>
#include <PvSystem.h>
#include <PvAcquisitionStateManager.h>
#include <PvPipeline.h>

using namespace std;

int JAICamera::OnBinning(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::AfterSet)
   {
		if (IsCapturing())
			return ERR_NOT_ALLOWED_DURING_CAPTURE;

      long bin = 1;
      pProp->Get(bin);
      PvResult pvr = genParams->SetIntegerValue(g_pv_BinH, (int64_t)bin);
      if (pvr.IsFailure())
         return processPvError(pvr);
      pvr = genParams->SetIntegerValue(g_pv_BinV, (int64_t)bin);
      if (pvr.IsFailure())
         return processPvError(pvr);
      return ResizeImageBuffer();
   }
   else if (eAct == MM::BeforeGet)
   {
      int64_t hbin, vbin;
      PvResult pvr = genParams->GetIntegerValue(g_pv_BinH, hbin);
      if (pvr.IsFailure())
         return processPvError(pvr);
      pvr = genParams->GetIntegerValue(g_pv_BinV, vbin);
      assert(hbin == vbin);
      if (pvr.IsFailure())
         return processPvError(pvr);
      pProp->Set((long)hbin);
   }
   return DEVICE_OK;
}

int JAICamera::OnPixelType(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::AfterSet)
	{
      if(IsCapturing())
         return DEVICE_CAMERA_BUSY_ACQUIRING;

      string pixelType;
      pProp->Get(pixelType);
      if ( pixelType.compare(g_PixelType_32bitRGB) == 0)
      {
			PvResult pvr = genParams->SetEnumValue(g_pv_PixelFormat, g_pv_PixelFormat_BGR8);
			if (!pvr.IsOK())
				return processPvError(pvr);
         pixelSize = 4;
         bitDepth = 8;
      }
      else if ( pixelType.compare(g_PixelType_64bitRGB_10bit) == 0)
      {
			PvResult pvr = genParams->SetEnumValue(g_pv_PixelFormat, g_pv_PixelFormat_BGR10);
			if (!pvr.IsOK())
				return processPvError(pvr);
			pixelSize = 8;
         bitDepth = 10;
		}
      else if ( pixelType.compare(g_PixelType_64bitRGB_12bit) == 0)
      {
			PvResult pvr = genParams->SetEnumValue(g_pv_PixelFormat, g_pv_PixelFormat_BGR12);
			if (!pvr.IsOK())
				return processPvError(pvr);
			pixelSize = 8;
         bitDepth = 12;
		}
		return ResizeImageBuffer();
	}
	else if (eAct == MM::BeforeGet)
	{
		PvString val;
		PvResult pvr = genParams->GetEnumValue(g_pv_PixelFormat, val);
		if (!pvr.IsOK())
			return processPvError(pvr);

		if (strcmp(val.GetAscii(), g_pv_PixelFormat_BGR8) == 0)
			pProp->Set(g_PixelType_32bitRGB);
		else if (strcmp(val.GetAscii(), g_pv_PixelFormat_BGR10) == 0)
			pProp->Set(g_PixelType_64bitRGB_10bit);
		else if (strcmp(val.GetAscii(), g_pv_PixelFormat_BGR12) == 0)
			pProp->Set(g_PixelType_64bitRGB_12bit);
		else
			assert(!"Unsupported pixel type");

	}
	return DEVICE_OK;
}

int JAICamera::OnFrameRate(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::AfterSet)
	{
		double val(1.0);
		pProp->Get(val);
		PvResult pvr = genParams->SetFloatValue("AcquisitionFrameRate", val);
		if (!pvr.IsOK())
			return processPvError(pvr);

		// adjust exposure limits
		double expMinUs, expMaxUs;
		pvr = genParams->GetFloatRange("ExposureTime", expMinUs, expMaxUs);
		if (!pvr.IsOK())
			return processPvError(pvr);
		SetPropertyLimits(MM::g_Keyword_Exposure, expMinUs / 1000, expMaxUs / 1000);
	}
	else if (eAct == MM::BeforeGet)
	{
		double fps;
		PvResult pvr = genParams->GetFloatValue("AcquisitionFrameRate", fps);
		if (!pvr.IsOK())
			return processPvError(pvr);
		pProp->Set(fps);
	}
	return DEVICE_OK;
}

int JAICamera::OnExposure(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::AfterSet)
	{
		double val(0.0);
		pProp->Get(val);
		SetExposure(val);
	}
	else if (eAct == MM::BeforeGet)
	{
		pProp->Set(GetExposure());
	}
	return DEVICE_OK;
}

int JAICamera::OnExposureIsIndividual(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	PvGenEnum* etm = genParams->GetEnum("ExposureTimeMode");
	if (eAct == MM::AfterSet)
	{
		if (IsCapturing())
			return ERR_NOT_ALLOWED_DURING_CAPTURE;
		std::string val;
		pProp->Get(val);
		const bool on = (val == "On");
		PvResult pvr = etm->SetValue(on ? "Individual" : "Common");
		if (!pvr.IsOK())
			return processPvError(pvr);
	}
	else if (eAct == MM::BeforeGet)
	{
		PvString val;
		PvResult pvr = etm->GetValue(val);
		if (!pvr.IsOK())
			return processPvError(pvr);
		const bool on = (std::string(val.GetAscii()) == "Individual");
		pProp->Set(on ? "On" : "Off");
	}
	return DEVICE_OK;
}

int JAICamera::OnSelectorExposure(const std::string& selector, MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::AfterSet)
	{
		if (IsCapturing())
			return ERR_NOT_ALLOWED_DURING_CAPTURE;
		double val{};
		pProp->Get(val);
		return SetSelectorExposure(selector, val);
	}
	else if (eAct == MM::BeforeGet)
	{
		double val{};
		int ret = GetSelectorExposure(selector, val);
		if (ret != DEVICE_OK)
			return ret;
		pProp->Set(val);
	}
	return DEVICE_OK;
}

int JAICamera::GetSelectorExposure(const std::string& selector, double& expMs)
{
	PvGenEnum *ets = genParams->GetEnum("ExposureTimeSelector");
	PvResult pvr = ets->SetValue(selector.c_str());
	if (!pvr.IsOK())
		return processPvError(pvr);

	double expUs{};
	PvResult pvr2 = genParams->GetFloatValue("ExposureTime", expUs);
	expMs = expUs * 1e-3;

	pvr = ets->SetValue(commonExposureSelector_.c_str());
	if (!pvr.IsOK())
		return processPvError(pvr);

	if (!pvr2.IsOK())
		return processPvError(pvr2);
	return DEVICE_OK;
}

int JAICamera::SetSelectorExposure(const std::string& selector, double expMs)
{
	PvGenEnum *ets = genParams->GetEnum("ExposureTimeSelector");
	PvResult pvr = ets->SetValue(selector.c_str());
	if (!pvr.IsOK())
		return processPvError(pvr);

	const double expUs = expMs * 1e3;
	PvResult pvr2 = genParams->SetFloatValue("ExposureTime", expUs);

	pvr = ets->SetValue(commonExposureSelector_.c_str());
	if (!pvr.IsOK())
		return processPvError(pvr);

	if (!pvr2.IsOK())
		return processPvError(pvr2);
	return DEVICE_OK;
}

int JAICamera::GetSelectorExposureMinMax(const std::string& selector, double& eMinMs, double& eMaxMs)
{
	PvGenEnum *ets = genParams->GetEnum("ExposureTimeSelector");
	PvResult pvr = ets->SetValue(selector.c_str());
	if (!pvr.IsOK())
		return processPvError(pvr);

	double eMinUs{}, eMaxUs{};
	PvResult pvr2 = genParams->GetFloatRange("ExposureTime", eMinUs, eMaxUs);
	eMinMs = eMinUs * 1e-3;
	eMaxMs = eMaxUs * 1e-3;

	pvr = ets->SetValue(commonExposureSelector_.c_str());
	if (!pvr.IsOK())
		return processPvError(pvr);

	if (!pvr2.IsOK())
		return processPvError(pvr2);
	return DEVICE_OK;
}

int JAICamera::OnGain(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::AfterSet)
	{
		double val(1.0);
		pProp->Get(val);
		PvResult pvr = genParams->SetFloatValue("Gain", val);
		if (!pvr.IsOK())
			return processPvError(pvr);
	}
	else if (eAct == MM::BeforeGet)
	{
		double gain;
		PvResult pvr = genParams->GetFloatValue("Gain", gain);
		if (!pvr.IsOK())
			return processPvError(pvr);
		pProp->Set(gain);
	}
	return DEVICE_OK;
}

int JAICamera::OnGainIsIndividual(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	PvGenEnum* igm = genParams->GetEnum("IndividualGainMode");
	if (eAct == MM::AfterSet)
	{
		if (IsCapturing())
			return ERR_NOT_ALLOWED_DURING_CAPTURE;
		std::string val;
		pProp->Get(val);
		const bool on = (val == "On");
		PvResult pvr = igm->SetValue(on ? "On" : "Off");
		if (!pvr.IsOK())
			return processPvError(pvr);
	}
	else if (eAct == MM::BeforeGet)
	{
		PvString val;
		PvResult pvr = igm->GetValue(val);
		if (!pvr.IsOK())
			return processPvError(pvr);
		const bool on = (std::string(val.GetAscii()) == "On");
		pProp->Set(on ? "On" : "Off");
	}
	return DEVICE_OK;
}

int JAICamera::OnSelectorGain(const std::string& selector, MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::AfterSet)
	{
		if (IsCapturing())
			return ERR_NOT_ALLOWED_DURING_CAPTURE;
		double val{};
		pProp->Get(val);
		return SetSelectorGain(selector, val);
	}
	else if (eAct == MM::BeforeGet)
	{
		double val{};
		int ret = GetSelectorGain(selector, val);
		if (ret != DEVICE_OK)
			return ret;
		pProp->Set(val);
	}
	return DEVICE_OK;
}

int JAICamera::GetSelectorGain(const std::string& selector, double& gain)
{
	PvGenEnum *gs = genParams->GetEnum("GainSelector");
	PvResult pvr = gs->SetValue(selector.c_str());
	if (!pvr.IsOK())
		return processPvError(pvr);

	PvResult pvr2 = genParams->GetFloatValue("Gain", gain);

	pvr = gs->SetValue(commonGainSelector_.c_str());
	if (!pvr.IsOK())
		return processPvError(pvr);

	if (!pvr2.IsOK())
		return processPvError(pvr2);
	return DEVICE_OK;
}

int JAICamera::SetSelectorGain(const std::string& selector, double gain)
{
	PvGenEnum *gs = genParams->GetEnum("GainSelector");
	PvResult pvr = gs->SetValue(selector.c_str());
	if (!pvr.IsOK())
		return processPvError(pvr);

	PvResult pvr2 = genParams->SetFloatValue("Gain", gain);

	pvr = gs->SetValue(commonGainSelector_.c_str());
	if (!pvr.IsOK())
		return processPvError(pvr);

	if (!pvr2.IsOK())
		return processPvError(pvr2);
	return DEVICE_OK;
}

int JAICamera::GetSelectorGainMinMax(const std::string& selector, double& gMin, double& gMax)
{
	PvGenEnum *gs = genParams->GetEnum("GainSelector");
	PvResult pvr = gs->SetValue(selector.c_str());
	if (!pvr.IsOK())
		return processPvError(pvr);

	PvResult pvr2 = genParams->GetFloatRange("Gain", gMin, gMax);

	pvr = gs->SetValue(commonGainSelector_.c_str());
	if (!pvr.IsOK())
		return processPvError(pvr);

	if (!pvr2.IsOK())
		return processPvError(pvr2);
	return DEVICE_OK;
}

int JAICamera::OnGamma(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::AfterSet)
	{
		double val(1.0);
		pProp->Get(val);
		PvResult pvr = genParams->SetFloatValue("Gamma", val);
		if (!pvr.IsOK())
			return processPvError(pvr);
	}
	else if (eAct == MM::BeforeGet)
	{
		double gamma;
		PvResult pvr = genParams->GetFloatValue("Gamma", gamma);
		if (!pvr.IsOK())
			return processPvError(pvr);
		pProp->Set(gamma);
	}
	return DEVICE_OK;
}

int JAICamera::OnBlackLevel(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::AfterSet)
	{
		if (IsCapturing())
			return ERR_NOT_ALLOWED_DURING_CAPTURE;
		double val{};
		pProp->Get(val);
		PvResult pvr = genParams->SetFloatValue("BlackLevel", val);
		if (!pvr.IsOK())
			return processPvError(pvr);
	}
	else if (eAct == MM::BeforeGet)
	{
		double val{};
		PvResult pvr = genParams->GetFloatValue("BlackLevel", val);
		if (!pvr.IsOK())
			return processPvError(pvr);
		pProp->Set(val);
	}
	return DEVICE_OK;
}

int JAICamera::OnSelectorBlackLevel(const std::string& selector, MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::AfterSet)
	{
		if (IsCapturing())
			return ERR_NOT_ALLOWED_DURING_CAPTURE;
		double val{};
		pProp->Get(val);
		return SetSelectorBlackLevel(selector, val);
	}
	else if (eAct == MM::BeforeGet)
	{
		double val{};
		int ret = GetSelectorBlackLevel(selector, val);
		if (ret != DEVICE_OK)
			return ret;
		pProp->Set(val);
	}
	return DEVICE_OK;
}

int JAICamera::GetSelectorBlackLevel(const std::string& selector, double& level)
{
	PvGenEnum *bls = genParams->GetEnum("BlackLevelSelector");
	PvResult pvr = bls->SetValue(selector.c_str());
	if (!pvr.IsOK())
		return processPvError(pvr);

	pvr = genParams->GetFloatValue("BlackLevel", level);
	if (!pvr.IsOK())
		return processPvError(pvr);
	return DEVICE_OK;
}

int JAICamera::SetSelectorBlackLevel(const std::string& selector, double level)
{
	PvGenEnum *bls = genParams->GetEnum("BlackLevelSelector");
	PvResult pvr = bls->SetValue(selector.c_str());
	if (!pvr.IsOK())
		return processPvError(pvr);

	pvr = genParams->SetFloatValue("BlackLevel", level);
	if (!pvr.IsOK())
		return processPvError(pvr);
	return DEVICE_OK;
}

int JAICamera::GetSelectorBlackLevelMinMax(const std::string& selector, double& minLevel, double& maxLevel)
{
	PvGenEnum *bls = genParams->GetEnum("BlackLevelSelector");
	PvResult pvr = bls->SetValue(selector.c_str());
	if (!pvr.IsOK())
		return processPvError(pvr);

	pvr = genParams->GetFloatRange("BlackLevel", minLevel, maxLevel);
	if (!pvr.IsOK())
		return processPvError(pvr);
	return DEVICE_OK;
}

int JAICamera::OnWhiteBalance(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	const char* pvCmd = "BalanceWhiteAuto";
	if (eAct == MM::AfterSet)
	{
		string val;
		long data;
		pProp->Get(val);
		GetPropertyData(g_WhiteBalance, val.c_str(), data);

		if (data == 1)
		{
			// special handling for "once" option
			// request white balance adjustment on next sequence acquisition
			ostringstream os;
			os << "WB Once set to PENDING status, will apply on the next frame";
			LogMessage(os.str());
			InterlockedExchange(&whiteBalancePending, 1L);
			wbPendingOption = val;
		}
		else
		{
			PvResult pvr = genParams->SetEnumValue(pvCmd, data);
			if (!pvr.IsOK())
				return processPvError(pvr);
			ostringstream os;
			os << "Set " << g_WhiteBalance << " : " << val << " = " << data;
			LogMessage(os.str());
		}
	}
	else if (eAct == MM::BeforeGet)
	{
		if (whiteBalancePending)
		{
			pProp->Set(wbPendingOption.c_str());
		}
		else
		{
			PvString val;
			PvResult pvr = genParams->GetEnumValue(pvCmd, val);
			if (!pvr.IsOK())
				return processPvError(pvr);
			pProp->Set(val.GetAscii());
		}
	}
	return DEVICE_OK;
}

int JAICamera::OnTestPattern(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	const char* pvCmd = "TestPattern";
	if (eAct == MM::AfterSet)
	{
		if (IsCapturing())
			return ERR_NOT_ALLOWED_DURING_CAPTURE;

		string val;
		long data;
		pProp->Get(val);
		GetPropertyData(g_TestPattern, val.c_str(), data);
		PvResult pvr = genParams->SetEnumValue(pvCmd, data);
		if (!pvr.IsOK())
			return processPvError(pvr);
	}
	else if (eAct == MM::BeforeGet)
	{
		PvString val;
		PvResult pvr = genParams->GetEnumValue(pvCmd, val);
		if (!pvr.IsOK())
			return processPvError(pvr);
		pProp->Set(val.GetAscii());
	}
	return DEVICE_OK;
}


int JAICamera::OnTemperature(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		double tempC;
		PvResult pvr = genParams->GetFloatValue("DeviceTemperature", tempC);
		if (!pvr.IsOK())
			return processPvError(pvr);
		pProp->Set(tempC);
	}
	return DEVICE_OK;
}

