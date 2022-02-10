///////////////////////////////////////////////////////////////////////////////
// FILE:          ASIDacXYStage.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   ASI DAC XYStage device adapter
//
// COPYRIGHT:     Applied Scientific Instrumentation, Eugene OR
//
// LICENSE:       This file is distributed under the BSD license.
//
//                This file is distributed in the hope that it will be useful,
//                but WITHOUT ANY WARRANTY; without even the implied warranty
//                of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//                IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//                CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//                INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.
//
// AUTHOR:        Brandon Simpson (brandon@asiimaging.com) 2/2022
//
// BASED ON:      ASIStage.cpp and others
//

#include "ASIDacXYStage.h"
#include "ASITiger.h"
#include "ASIHub.h"
#include "ModuleInterface.h"
#include "DeviceUtils.h"
#include "DeviceBase.h"
#include "MMDevice.h"
#include <iostream>
#include <string>

CDACXYStage::CDACXYStage(const char* name) :
	ASIPeripheralBase< ::CXYStageBase, CDACXYStage >(name),
	axisLetterX_(g_EmptyAxisLetterStr),    // value determined by extended name
	axisLetterY_(g_EmptyAxisLetterStr),    // value determined by extended name
	maxvoltsX_(0.0), // X axis limits
	minvoltsX_(0.0),
	maxvoltsY_(0.0), // Y axis limits
	minvoltsY_(0.0),
	ring_buffer_supported_(false),
	ring_buffer_capacity_(0),
	ttl_trigger_supported_(false),
	ttl_trigger_enabled_(false),
	umToMvX_(1),
	umToMvY_(1)
{
	if (IsExtendedName(name))  // only set up these properties if we have the required information in the name
	{
		axisLetterX_ = GetAxisLetterFromExtName(name);
		CreateProperty(g_AxisLetterXPropertyName, axisLetterX_.c_str(), MM::String, true);
		axisLetterY_ = GetAxisLetterFromExtName(name, 1);
		CreateProperty(g_AxisLetterYPropertyName, axisLetterY_.c_str(), MM::String, true);
	}
	
	// pre-init properties => sets the conversion factor for microns to millivolts on x and y
	CPropertyAction* pAct;

	pAct = new CPropertyAction(this, &CDACXYStage::OnConversionFactorX);
	CreateProperty(g_DACMicronsPerMvXPropertyName, "1", MM::Integer, false, pAct, true);
	SetPropertyLimits(g_DACMicronsPerMvXPropertyName, 1, LONG_MAX);
	UpdateProperty(g_DACMicronsPerMvXPropertyName);

	pAct = new CPropertyAction(this, &CDACXYStage::OnConversionFactorY);
	CreateProperty(g_DACMicronsPerMvYPropertyName, "1", MM::Integer, false, pAct, true);
	SetPropertyLimits(g_DACMicronsPerMvYPropertyName, 1, LONG_MAX);
	UpdateProperty(g_DACMicronsPerMvYPropertyName);
}

int CDACXYStage::Initialize()
{
	// call generic Initialize first, this gets the hub
	RETURN_ON_MM_ERROR(PeripheralInitialize());

	CPropertyAction* pAct;

	// create MM description; this doesn't work during hardware configuration wizard but will work afterwards
	std::ostringstream command;
	command.str("");
	command << g_DacXYStageDeviceDescription << " Xaxis=" << axisLetterX_ << " Yaxis=" << axisLetterY_ << " HexAddr=" << addressString_;
	CreateProperty(MM::g_Keyword_Description, command.str().c_str(), MM::String, true);

	// refresh properties from controller every time - default is not to refresh (speeds things up by not redoing so much serial comm)
	pAct = new CPropertyAction(this, &CDACXYStage::OnRefreshProperties);
	CreateProperty(g_RefreshPropValsPropertyName, g_NoState, MM::String, false, pAct);
	AddAllowedValue(g_RefreshPropValsPropertyName, g_NoState);
	AddAllowedValue(g_RefreshPropValsPropertyName, g_YesState);

	// save settings to controller if requested
	pAct = new CPropertyAction(this, &CDACXYStage::OnSaveCardSettings);
	CreateProperty(g_SaveSettingsPropertyName, g_SaveSettingsOrig, MM::String, false, pAct);
	AddAllowedValue(g_SaveSettingsPropertyName, g_SaveSettingsX);
	AddAllowedValue(g_SaveSettingsPropertyName, g_SaveSettingsY);
	AddAllowedValue(g_SaveSettingsPropertyName, g_SaveSettingsZ);
	AddAllowedValue(g_SaveSettingsPropertyName, g_SaveSettingsZJoystick);
	AddAllowedValue(g_SaveSettingsPropertyName, g_SaveSettingsOrig);
	AddAllowedValue(g_SaveSettingsPropertyName, g_SaveSettingsDone);

	// get voltage limits
	GetMaxVolts(maxvoltsX_, axisLetterX_);
	GetMinVolts(minvoltsX_, axisLetterX_);
	GetMaxVolts(maxvoltsY_, axisLetterY_);
	GetMinVolts(minvoltsY_, axisLetterY_);

	// min and max voltages for both axes as read only property
	command.str("");
	command << maxvoltsX_;
	CreateProperty(g_DACMaxVoltsXPropertyName, command.str().c_str(), MM::Float, true);
	command.str("");
	command << minvoltsX_;
	CreateProperty(g_DACMinVoltsXPropertyName, command.str().c_str(), MM::Float, true);
	command.str("");
	command << maxvoltsY_;
	CreateProperty(g_DACMaxVoltsYPropertyName, command.str().c_str(), MM::Float, true);
	command.str("");
	command << minvoltsY_;
	CreateProperty(g_DACMinVoltsYPropertyName, command.str().c_str(), MM::Float, true);

	// signal DAC properties
	pAct = new CPropertyAction(this, &CDACXYStage::OnDACModeX);
	CreateProperty(g_DACModeXPropertyName, "0", MM::String, false, pAct);
	AddAllowedValue(g_DACModeXPropertyName, g_DACOutputMode_0);
	AddAllowedValue(g_DACModeXPropertyName, g_DACOutputMode_1);
	AddAllowedValue(g_DACModeXPropertyName, g_DACOutputMode_2);
	AddAllowedValue(g_DACModeXPropertyName, g_DACOutputMode_4);
	AddAllowedValue(g_DACModeXPropertyName, g_DACOutputMode_5);
	AddAllowedValue(g_DACModeXPropertyName, g_DACOutputMode_6);
	AddAllowedValue(g_DACModeXPropertyName, g_DACOutputMode_7);
	UpdateProperty(g_DACModeXPropertyName);

	pAct = new CPropertyAction(this, &CDACXYStage::OnDACModeY);
	CreateProperty(g_DACModeYPropertyName, "0", MM::String, false, pAct);
	AddAllowedValue(g_DACModeYPropertyName, g_DACOutputMode_0);
	AddAllowedValue(g_DACModeYPropertyName, g_DACOutputMode_1);
	AddAllowedValue(g_DACModeYPropertyName, g_DACOutputMode_2);
	AddAllowedValue(g_DACModeYPropertyName, g_DACOutputMode_4);
	AddAllowedValue(g_DACModeYPropertyName, g_DACOutputMode_5);
	AddAllowedValue(g_DACModeYPropertyName, g_DACOutputMode_6);
	AddAllowedValue(g_DACModeYPropertyName, g_DACOutputMode_7);
	UpdateProperty(g_DACModeYPropertyName);

	// DAC gate open or closed
	pAct = new CPropertyAction(this, &CDACXYStage::OnDACGateX);
	CreateProperty(g_DACGateXPropertyName, "0", MM::String, false, pAct);
	AddAllowedValue(g_DACGateXPropertyName, g_OpenState);
	AddAllowedValue(g_DACGateXPropertyName, g_ClosedState);
	UpdateProperty(g_DACGateXPropertyName);

	pAct = new CPropertyAction(this, &CDACXYStage::OnDACGateY);
	CreateProperty(g_DACGateYPropertyName, "0", MM::String, false, pAct);
	AddAllowedValue(g_DACGateYPropertyName, g_OpenState);
	AddAllowedValue(g_DACGateYPropertyName, g_ClosedState);
	UpdateProperty(g_DACGateYPropertyName);

	// filter cut-off frequency
	pAct = new CPropertyAction(this, &CDACXYStage::OnCutoffFreqX);
	CreateProperty(g_ScannerCutoffFilterXPropertyName, "0", MM::Float, false, pAct);
	SetPropertyLimits(g_ScannerCutoffFilterXPropertyName, 0.1, 650);
	UpdateProperty(g_ScannerCutoffFilterXPropertyName);

	pAct = new CPropertyAction(this, &CDACXYStage::OnCutoffFreqY);
	CreateProperty(g_ScannerCutoffFilterYPropertyName, "0", MM::Float, false, pAct);
	SetPropertyLimits(g_ScannerCutoffFilterYPropertyName, 0.1, 650);
	UpdateProperty(g_ScannerCutoffFilterYPropertyName);

	// joystick fast speed (JS X=)
	pAct = new CPropertyAction(this, &CDACXYStage::OnJoystickFastSpeed);
	CreateProperty(g_JoystickFastSpeedPropertyName, "100", MM::Float, false, pAct);
	SetPropertyLimits(g_JoystickFastSpeedPropertyName, 0, 100);
	UpdateProperty(g_JoystickFastSpeedPropertyName);

	// joystick slow speed (JS Y=)
	pAct = new CPropertyAction(this, &CDACXYStage::OnJoystickSlowSpeed);
	CreateProperty(g_JoystickSlowSpeedPropertyName, "10", MM::Float, false, pAct);
	SetPropertyLimits(g_JoystickSlowSpeedPropertyName, 0, 100);
	UpdateProperty(g_JoystickSlowSpeedPropertyName);

	// joystick mirror (changes joystick fast/slow speeds to negative)
	pAct = new CPropertyAction(this, &CDACXYStage::OnJoystickMirror);
	CreateProperty(g_JoystickMirrorPropertyName, g_NoState, MM::String, false, pAct);
	AddAllowedValue(g_JoystickMirrorPropertyName, g_NoState);
	AddAllowedValue(g_JoystickMirrorPropertyName, g_YesState);
	UpdateProperty(g_JoystickMirrorPropertyName);

	// joystick rotate (interchanges X and Y axes, useful if camera is rotated
	pAct = new CPropertyAction(this, &CDACXYStage::OnJoystickRotate);
	CreateProperty(g_JoystickRotatePropertyName, g_NoState, MM::String, false, pAct);
	AddAllowedValue(g_JoystickRotatePropertyName, g_NoState);
	AddAllowedValue(g_JoystickRotatePropertyName, g_YesState);
	UpdateProperty(g_JoystickRotatePropertyName);

	// joystick enable/disable
	pAct = new CPropertyAction(this, &CDACXYStage::OnJoystickEnableDisable);
	CreateProperty(g_JoystickEnabledPropertyName, g_YesState, MM::String, false, pAct);
	AddAllowedValue(g_JoystickEnabledPropertyName, g_NoState);
	AddAllowedValue(g_JoystickEnabledPropertyName, g_YesState);
	UpdateProperty(g_JoystickEnabledPropertyName);

	// get build info so we can add optional properties
	build_info_type build;
	RETURN_ON_MM_ERROR(hub_->GetBuildInfo(addressChar_, build));

	// add single-axis properties if supported
	// Normally we check for version, but this card is always going to have v3.39 and above
	if (build.vAxesProps[0] & BIT5)
	{
		pAct = new CPropertyAction(this, &CDACXYStage::OnSAAmplitudeX);
		CreateProperty(g_SAAmplitudeXDACPropertyName, "0", MM::Float, false, pAct);
		SetPropertyLimits(g_SAAmplitudeXDACPropertyName, 0, (maxvoltsX_ - minvoltsX_) * 1000);
		UpdateProperty(g_SAAmplitudeXDACPropertyName);

		pAct = new CPropertyAction(this, &CDACXYStage::OnSAAmplitudeY);
		CreateProperty(g_SAAmplitudeYDACPropertyName, "0", MM::Float, false, pAct);
		SetPropertyLimits(g_SAAmplitudeYDACPropertyName, 0, (maxvoltsY_ - minvoltsY_) * 1000);
		UpdateProperty(g_SAAmplitudeYDACPropertyName);

		pAct = new CPropertyAction(this, &CDACXYStage::OnSAOffsetX);
		CreateProperty(g_SAOffsetDACXPropertyName, "0", MM::Float, false, pAct);
		SetPropertyLimits(g_SAOffsetDACXPropertyName, minvoltsX_ * 1000, maxvoltsX_ * 1000);
		UpdateProperty(g_SAOffsetDACXPropertyName);

		pAct = new CPropertyAction(this, &CDACXYStage::OnSAOffsetY);
		CreateProperty(g_SAOffsetDACYPropertyName, "0", MM::Float, false, pAct);
		SetPropertyLimits(g_SAOffsetDACYPropertyName, minvoltsY_ * 1000, maxvoltsY_ * 1000);
		UpdateProperty(g_SAOffsetDACYPropertyName);

		pAct = new CPropertyAction(this, &CDACXYStage::OnSAPeriodX);
		CreateProperty(g_SAPeriodXPropertyName, "0", MM::Integer, false, pAct);
		UpdateProperty(g_SAPeriodXPropertyName);

		pAct = new CPropertyAction(this, &CDACXYStage::OnSAPeriodY);
		CreateProperty(g_SAPeriodYPropertyName, "0", MM::Integer, false, pAct);
		UpdateProperty(g_SAPeriodYPropertyName);

		pAct = new CPropertyAction(this, &CDACXYStage::OnSAModeX);
		CreateProperty(g_SAModeXPropertyName, g_SAMode_0, MM::String, false, pAct);
		AddAllowedValue(g_SAModeXPropertyName, g_SAMode_0);
		AddAllowedValue(g_SAModeXPropertyName, g_SAMode_1);
		AddAllowedValue(g_SAModeXPropertyName, g_SAMode_2);
		AddAllowedValue(g_SAModeXPropertyName, g_SAMode_3);
		UpdateProperty(g_SAModeXPropertyName);

		pAct = new CPropertyAction(this, &CDACXYStage::OnSAModeY);
		CreateProperty(g_SAModeYPropertyName, g_SAMode_0, MM::String, false, pAct);
		AddAllowedValue(g_SAModeYPropertyName, g_SAMode_0);
		AddAllowedValue(g_SAModeYPropertyName, g_SAMode_1);
		AddAllowedValue(g_SAModeYPropertyName, g_SAMode_2);
		AddAllowedValue(g_SAModeYPropertyName, g_SAMode_3);
		UpdateProperty(g_SAModeYPropertyName);

		pAct = new CPropertyAction(this, &CDACXYStage::OnSAPatternX);
		CreateProperty(g_SAPatternXPropertyName, g_SAPattern_0, MM::String, false, pAct);
		AddAllowedValue(g_SAPatternXPropertyName, g_SAPattern_0);
		AddAllowedValue(g_SAPatternXPropertyName, g_SAPattern_1);
		AddAllowedValue(g_SAPatternXPropertyName, g_SAPattern_2);
		AddAllowedValue(g_SAPatternXPropertyName, g_SAPattern_3);
		UpdateProperty(g_SAPatternXPropertyName);

		pAct = new CPropertyAction(this, &CDACXYStage::OnSAPatternY);
		CreateProperty(g_SAPatternYPropertyName, g_SAPattern_0, MM::String, false, pAct);
		AddAllowedValue(g_SAPatternYPropertyName, g_SAPattern_0);
		AddAllowedValue(g_SAPatternYPropertyName, g_SAPattern_1);
		AddAllowedValue(g_SAPatternYPropertyName, g_SAPattern_2);
		AddAllowedValue(g_SAPatternYPropertyName, g_SAPattern_3);
		UpdateProperty(g_SAPatternYPropertyName);

		// generates a set of additional advanced properties that are rarely used
		pAct = new CPropertyAction(this, &CDACXYStage::OnSAAdvancedX);
		CreateProperty(g_AdvancedSAPropertiesXPropertyName, g_NoState, MM::String, false, pAct);
		AddAllowedValue(g_AdvancedSAPropertiesXPropertyName, g_NoState);
		AddAllowedValue(g_AdvancedSAPropertiesXPropertyName, g_YesState);
		UpdateProperty(g_AdvancedSAPropertiesXPropertyName);

		// generates a set of additional advanced properties that are rarely used
		pAct = new CPropertyAction(this, &CDACXYStage::OnSAAdvancedY);
		CreateProperty(g_AdvancedSAPropertiesYPropertyName, g_NoState, MM::String, false, pAct);
		AddAllowedValue(g_AdvancedSAPropertiesYPropertyName, g_NoState);
		AddAllowedValue(g_AdvancedSAPropertiesYPropertyName, g_YesState);
		UpdateProperty(g_AdvancedSAPropertiesYPropertyName);
	}

	// add ring buffer properties if supported
	if (build.vAxesProps[0] & BIT1)
	{
		// get the number of ring buffer positions from the BU X output
		std::string rb_define = hub_->GetDefineString(build, "RING BUFFER");

		ring_buffer_capacity_ = 0;
		if (rb_define.size() > 12)
		{
			ring_buffer_capacity_ = atol(rb_define.substr(11).c_str());
		}

		if (ring_buffer_capacity_ != 0)
		{
			ring_buffer_supported_ = true;

			pAct = new CPropertyAction(this, &CDACXYStage::OnRBMode);
			CreateProperty(g_RB_ModePropertyName, g_RB_OnePoint_1, MM::String, false, pAct);
			AddAllowedValue(g_RB_ModePropertyName, g_RB_OnePoint_1);
			AddAllowedValue(g_RB_ModePropertyName, g_RB_PlayOnce_2);
			AddAllowedValue(g_RB_ModePropertyName, g_RB_PlayRepeat_3);
			UpdateProperty(g_RB_ModePropertyName);

			pAct = new CPropertyAction(this, &CDACXYStage::OnRBDelayBetweenPoints);
			CreateProperty(g_RB_DelayPropertyName, "0", MM::Integer, false, pAct);
			UpdateProperty(g_RB_DelayPropertyName);

			// "do it" property to do TTL trigger via serial
			pAct = new CPropertyAction(this, &CDACXYStage::OnRBTrigger);
			CreateProperty(g_RB_TriggerPropertyName, g_IdleState, MM::String, false, pAct);
			AddAllowedValue(g_RB_TriggerPropertyName, g_IdleState, 0);
			AddAllowedValue(g_RB_TriggerPropertyName, g_DoItState, 1);
			AddAllowedValue(g_RB_TriggerPropertyName, g_DoneState, 2);
			UpdateProperty(g_RB_TriggerPropertyName);

			pAct = new CPropertyAction(this, &CDACXYStage::OnRBRunning);
			CreateProperty(g_RB_AutoplayRunningPropertyName, g_NoState, MM::String, false, pAct);
			AddAllowedValue(g_RB_AutoplayRunningPropertyName, g_NoState);
			AddAllowedValue(g_RB_AutoplayRunningPropertyName, g_YesState);
			UpdateProperty(g_RB_AutoplayRunningPropertyName);

			pAct = new CPropertyAction(this, &CDACXYStage::OnUseSequence);
			CreateProperty(g_UseSequencePropertyName, g_NoState, MM::String, false, pAct);
			AddAllowedValue(g_UseSequencePropertyName, g_NoState);
			AddAllowedValue(g_UseSequencePropertyName, g_YesState);
			ttl_trigger_enabled_ = false;
		}

		if ((hub_->IsDefinePresent(build, "IN0_INT")) && ring_buffer_supported_)
		{
			ttl_trigger_supported_ = true;
			
			// TTL In Mode
			pAct = new CPropertyAction(this, &CDACXYStage::OnTTLin);
			CreateProperty(g_TTLinName, "0", MM::Integer, false, pAct);
			SetPropertyLimits(g_TTLinName, 0, 30);
			UpdateProperty(g_TTLinName);

			// TTL Out Mode
			pAct = new CPropertyAction(this, &CDACXYStage::OnTTLout);
			CreateProperty(g_TTLoutName, "0", MM::Integer, false, pAct);
			SetPropertyLimits(g_TTLoutName, 0, 30);
			UpdateProperty(g_TTLoutName);
		}
	}

	initialized_ = true;
	return DEVICE_OK;
}

/////////////// XYStage API ///////////////

int CDACXYStage::Stop()
{
	// note this stops the card which usually is synonymous with the stage, \ stops all stages
	std::ostringstream command;
	command.str("");
	command << addressChar_ << "HALT";
	RETURN_ON_MM_ERROR(hub_->QueryCommand(command.str()));
	return DEVICE_OK;
}

int CDACXYStage::GetPositionSteps(long& x, long& y)
{
	std::ostringstream command;
	command.str("");
	command << "W " << axisLetterX_;
	RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), ":A"));
	double tmp;
	RETURN_ON_MM_ERROR(hub_->ParseAnswerAfterPosition2(tmp));
	x = (long)(tmp * umToMvX_);
	command.str("");
	command << "W " << axisLetterY_;
	RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), ":A"));
	RETURN_ON_MM_ERROR(hub_->ParseAnswerAfterPosition2(tmp));
	y = (long)(tmp * umToMvY_);
	return DEVICE_OK;
}

int CDACXYStage::SetPositionSteps(long x, long y)
{
	std::ostringstream command;
	command.str("");
	command << "M " << axisLetterX_ << "=" << (x / umToMvX_) << " " << axisLetterY_ << "=" << (y / umToMvY_);
	return hub_->QueryCommandVerify(command.str(), ":A");
}

int CDACXYStage::SetRelativePositionSteps(long x, long y)
{
	std::ostringstream command;
	command.str("");
	if ((x == 0) && (y != 0))
	{
		command << "R " << axisLetterY_ << "=" << (y / umToMvY_);
	}
	else if ((x != 0) && (y == 0))
	{
		command << "R " << axisLetterX_ << "=" << (x / umToMvX_);
	}
	else
	{
		command << "R " << axisLetterX_ << "=" << (x / umToMvX_) << " " << axisLetterY_ << "=" << (y / umToMvY_);
	}
	return hub_->QueryCommandVerify(command.str(), ":A");
}

// FIXME: is this correct?
int CDACXYStage::GetStepLimits(long& xMin, long& xMax, long& yMin, long& yMax)
{
	double minVoltsX = 0.0;
	double minVoltsY = 0.0;
	double maxVoltsX = 0.0;
	double maxVoltsY = 0.0;
	// get limits in volts
	GetMinVolts(minVoltsX, axisLetterX_);
	GetMinVolts(minVoltsY, axisLetterY_);
	GetMaxVolts(maxVoltsX, axisLetterX_);
	GetMaxVolts(maxVoltsY, axisLetterY_);
	// convert to millivolts
	xMin = (long)(minVoltsX * 1000);
	yMin = (long)(minVoltsY * 1000);
	xMax = (long)(maxVoltsX * 1000);
	yMax = (long)(maxVoltsY * 1000);
	// convert to microns
	xMin *= umToMvX_;
	yMin *= umToMvY_;
	xMax *= umToMvX_;
	yMax *= umToMvY_;
	return DEVICE_OK;
}

int CDACXYStage::SetOrigin()
{
	std::ostringstream command;
	command.str("");
	command << "H " << axisLetterX_ << "=0 " << axisLetterY_ << "=0";
	return hub_->QueryCommandVerify(command.str(), ":A");
}

int CDACXYStage::SetXOrigin()
{
	std::ostringstream command; 
	command.str("");
	command << "H " << axisLetterX_ << "=0";;
	return hub_->QueryCommandVerify(command.str(), ":A");
}

int CDACXYStage::SetYOrigin()
{
	std::ostringstream command;
	command.str("");
	command << "H " << axisLetterY_ << "=0";;
	return hub_->QueryCommandVerify(command.str(), ":A");
}

int CDACXYStage::Home()
{
	std::ostringstream command;
	command.str("");
	command << "! " << axisLetterX_ << " " << axisLetterY_;
	return hub_->QueryCommandVerify(command.str(), ":A");
}

int CDACXYStage::SetHome()
{
	std::ostringstream command;
	command.str("");
	command << "HM " << axisLetterX_ << "+" << " " << axisLetterY_ << "+";
	return hub_->QueryCommandVerify(command.str(), ":A");
}

int CDACXYStage::Move(double vx, double vy)
{
	std::ostringstream command;
	command.str("");
	command << "VE " << axisLetterX_ << "=" << vx << " " << axisLetterY_ << "=" << vy;
	return hub_->QueryCommandVerify(command.str(), ":A");
}

/////////////// DAC API ///////////////

int CDACXYStage::SetGateOpen(bool open, std::string axisLetter)
{
	std::ostringstream command;
	if (open)
	{
		command.str("");
		command << "MC " << axisLetter << "+";
		RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), ":A"));
	}
	else
	{
		command.str("");
		command << "MC " << axisLetter << "-";
		RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), ":A"));
	}
	return DEVICE_OK;
}

int CDACXYStage::GetGateOpen(bool& open, std::string axisLetter)
{
	std::ostringstream command;
	long tmp;
	command.str("");
	command << "MC " << axisLetter << "?";

	open = false; // in case of error we return that the gate is closed
	// "MC <Axis>?" replies look like ":A 1"
	RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), ":A"));
	RETURN_ON_MM_ERROR(hub_->ParseAnswerAfterPosition(3, tmp));

	if (tmp == 1)
	{
		open = true;
	}
	return DEVICE_OK;
}

/////////////// DAC Helpers ///////////////

int CDACXYStage::GetMaxVolts(double& volts, std::string axisLetter)
{
	std::ostringstream command;
	command.str("");
	command << "SU " << axisLetter << "?";
	RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), ":A"));
	RETURN_ON_MM_ERROR(hub_->ParseAnswerAfterEquals(volts));
	return DEVICE_OK;
}

int CDACXYStage::GetMinVolts(double& volts, std::string axisLetter)
{
	std::ostringstream command;
	command.str("");
	command << "SL " << axisLetter << "?";
	RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), ":A"));
	RETURN_ON_MM_ERROR(hub_->ParseAnswerAfterEquals(volts));
	return DEVICE_OK;
}

///////////////////
// action handlers

// redoes the joystick settings so they can be saved using SS Z
int CDACXYStage::OnSaveJoystickSettings()
{
	long tmp;
	std::string tmpstr;
	std::ostringstream command;
	std::ostringstream response;
	command.str("");
	response.str("");
	command << "J " << axisLetterX_ << "?";
	response << ":A " << axisLetterX_ << "=";
	RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), response.str()));
	RETURN_ON_MM_ERROR(hub_->ParseAnswerAfterEquals(tmp));
	tmp += 100;
	command.str("");
	command << "J " << axisLetterX_ << "=" << tmp;
	RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), ":A"));
	command.str("");
	response.str("");
	command << "J " << axisLetterY_ << "?";
	response << ":A " << axisLetterY_ << "=";
	RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), response.str()));
	RETURN_ON_MM_ERROR(hub_->ParseAnswerAfterEquals(tmp));
	tmp += 100;
	command.str("");
	command << "J " << axisLetterY_ << "=" << tmp;
	RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), ":A"));
	return DEVICE_OK;
}

int CDACXYStage::OnRefreshProperties(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	std::string tmpstr;
	if (eAct == MM::AfterSet)
	{
		pProp->Get(tmpstr);
		if (tmpstr.compare(g_YesState) == 0)
		{
			refreshProps_ = true;
		}
		else
		{
			refreshProps_ = false;
		}
	}
	return DEVICE_OK;
}

int CDACXYStage::OnSaveCardSettings(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	std::string tmpstr;
	std::ostringstream command;
	command.str("");
	if (eAct == MM::AfterSet)
	{
		command.str("");
		command << addressChar_ << "SS ";
		pProp->Get(tmpstr);
		if (tmpstr.compare(g_SaveSettingsOrig) == 0)
		{
			return DEVICE_OK;
		}
		if (tmpstr.compare(g_SaveSettingsDone) == 0)
		{
			return DEVICE_OK;
		}
		if (tmpstr.compare(g_SaveSettingsX) == 0)
		{
			command << 'X';
		}
		else if (tmpstr.compare(g_SaveSettingsY) == 0)
		{
			command << 'X';
		}
		else if (tmpstr.compare(g_SaveSettingsZ) == 0)
		{
			command << 'Z';
		}
		else if (tmpstr.compare(g_SaveSettingsZJoystick) == 0)
		{
			command << 'Z';
			// do save joystick settings first
			RETURN_ON_MM_ERROR(OnSaveJoystickSettings());
		}
		RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), ":A", (long)200));  // note added 200ms delay
		pProp->Set(g_SaveSettingsDone);
		command.str("");
		command << g_SaveSettingsDone;
		RETURN_ON_MM_ERROR(hub_->UpdateSharedProperties(addressChar_, pProp->GetName(), command.str()));
	}
	return DEVICE_OK;
}

/////////////// DAC PROPERTIES ///////////////

// will need to restart the controller for settings to take effect
int CDACXYStage::OnDACModeGeneric(MM::PropertyBase* pProp, MM::ActionType eAct, std::string axisLetter)
{
	std::ostringstream command;
	command.str("");
	long tmp = 0;
	if (eAct == MM::BeforeGet)
	{
		if (!refreshProps_ && initialized_)
		{
			return DEVICE_OK;
		}
		std::ostringstream response;
		response.str("");
		command << "PR " << axisLetter << "?"; // On SIGNAL_DAC this is on PR(system flag) instead of PM because I needed upper and lower limits to change too 
		response << ":A " << axisLetter << "="; // Reply for PR command is ":A H=6 <CR><LF>"
		RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), response.str()));
		RETURN_ON_MM_ERROR(hub_->ParseAnswerAfterEquals(tmp));

		bool success = 0;
		switch (tmp)
		{
		case 1: success = pProp->Set(g_DACOutputMode_1); break;
		case 0: success = pProp->Set(g_DACOutputMode_0); break;
		case 2: success = pProp->Set(g_DACOutputMode_2);  break;
		case 4: success = pProp->Set(g_DACOutputMode_4);  break;
		case 5: success = pProp->Set(g_DACOutputMode_5);  break;
		case 6: success = pProp->Set(g_DACOutputMode_6);  break;
		case 7: success = pProp->Set(g_DACOutputMode_7);  break;
		default: success = 0; break;
		}
		if (!success)
		{
			return DEVICE_INVALID_PROPERTY_VALUE;
		}
	}
	else if (eAct == MM::AfterSet)
	{
		string tmpstr;
		pProp->Get(tmpstr);
		if (tmpstr.compare(g_DACOutputMode_0) == 0)
		{
			tmp = 0;
		}
		else if (tmpstr.compare(g_DACOutputMode_1) == 0)
		{
			tmp = 1;
		}
		else if (tmpstr.compare(g_DACOutputMode_2) == 0)
		{
			tmp = 2;
		}
		else if (tmpstr.compare(g_DACOutputMode_4) == 0)
		{
			tmp = 4;
		}
		else if (tmpstr.compare(g_DACOutputMode_5) == 0)
		{
			tmp = 5;
		}
		else if (tmpstr.compare(g_DACOutputMode_6) == 0)
		{
			tmp = 6;
		}
		else if (tmpstr.compare(g_DACOutputMode_7) == 0)
		{
			tmp = 7;
		}
		else
		{
			return DEVICE_INVALID_PROPERTY_VALUE;
		}
		command << "PR " << axisLetter << "=" << tmp;
		RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), ":A"));
	}
	return DEVICE_OK;
}

int CDACXYStage::OnDACGateGeneric(MM::PropertyBase* pProp, MM::ActionType eAct, std::string axisLetter)
{
	bool tmp;
	if (eAct == MM::BeforeGet)
	{
		if (!refreshProps_ && initialized_)
		{
			return DEVICE_OK;
		}
		RETURN_ON_MM_ERROR(GetGateOpen(tmp, axisLetter));

		if (tmp) // true
		{
			pProp->Set(g_OpenState);
		}
		else
		{
			pProp->Set(g_ClosedState);
		}
		return DEVICE_OK;
	}
	else if (eAct == MM::AfterSet)
	{
		string tmpstr;
		pProp->Get(tmpstr);
		if (tmpstr.compare(g_OpenState) == 0)
		{
			tmp = true;
		}
		else
		{
			tmp = false;
		}
		RETURN_ON_MM_ERROR(SetGateOpen(tmp, axisLetter));
	}
	return DEVICE_OK;
}

int CDACXYStage::OnCutoffFreqGeneric(MM::PropertyBase* pProp, MM::ActionType eAct, std::string axisLetter)
{
	std::ostringstream command;
	std::ostringstream response;
	command.str("");
	response.str("");
	double tmp = 0;
	if (eAct == MM::BeforeGet)
	{
		if (!refreshProps_ && initialized_)
		{
			return DEVICE_OK;
		}
		command << "B " << axisLetter << "?";
		response << ":" << axisLetter << "=";
		RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), response.str()));
		RETURN_ON_MM_ERROR(hub_->ParseAnswerAfterEquals(tmp));
		if (!pProp->Set(tmp))
		{
			return DEVICE_INVALID_PROPERTY_VALUE;
		}
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(tmp);
		command << "B " << axisLetter << "=" << tmp;
		RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), ":A"));
	}
	return DEVICE_OK;
}

/////////////// RING BUFFER ///////////////

int CDACXYStage::OnRBMode(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	std::ostringstream command;
	std::ostringstream response;
	command.str("");
	response.str("");
	long tmp;
	if (eAct == MM::BeforeGet)
	{
		if (!refreshProps_ && initialized_)
		{
			return DEVICE_OK;
		}
		command << addressChar_ << "RM F?";
		response << ":A F=";
		RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), response.str()));
		RETURN_ON_MM_ERROR(hub_->ParseAnswerAfterEquals(tmp));
		if (tmp >= 128)
		{
			tmp -= 128;  // remove the "running now" code if present
		}
		bool success;
		switch (tmp)
		{
		case 1: success = pProp->Set(g_RB_OnePoint_1); break;
		case 2: success = pProp->Set(g_RB_PlayOnce_2); break;
		case 3: success = pProp->Set(g_RB_PlayRepeat_3); break;
		default: success = false;
		}
		if (!success)
		{
			return DEVICE_INVALID_PROPERTY_VALUE;
		}
	}
	else if (eAct == MM::AfterSet)
	{
		if (hub_->UpdatingSharedProperties())
		{
			return DEVICE_OK;
		}
		std::string tmpstr;
		pProp->Get(tmpstr);
		if (tmpstr.compare(g_RB_OnePoint_1) == 0)
		{
			tmp = 1;
		}
		else if (tmpstr.compare(g_RB_PlayOnce_2) == 0)
		{
			tmp = 2;
		}
		else if (tmpstr.compare(g_RB_PlayRepeat_3) == 0)
		{
			tmp = 3;
		}
		else
		{
			return DEVICE_INVALID_PROPERTY_VALUE;
		}
		command << addressChar_ << "RM F=" << tmp;
		RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), ":A"));
		RETURN_ON_MM_ERROR(hub_->UpdateSharedProperties(addressChar_, pProp->GetName(), tmpstr.c_str()));
	}
	return DEVICE_OK;
}

int CDACXYStage::OnRBTrigger(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	std::ostringstream command;
	command.str("");
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(g_IdleState);
	}
	else if (eAct == MM::AfterSet)
	{
		if (hub_->UpdatingSharedProperties())
		{
			return DEVICE_OK;
		}
		std::string tmpstr;
		pProp->Get(tmpstr);
		if (tmpstr.compare(g_DoItState) == 0)
		{
			command << addressChar_ << "RM";
			RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), ":A"));
			pProp->Set(g_DoneState);
		}
	}
	return DEVICE_OK;
}

int CDACXYStage::OnRBRunning(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	std::ostringstream command;
	std::ostringstream response;
	command.str("");
	response.str("");
	long tmp = 0;
	static bool justSet;
	if (eAct == MM::BeforeGet)
	{
		if (!refreshProps_ && initialized_ && !justSet)
		{
			return DEVICE_OK;
		}
		command << addressChar_ << "RM F?";
		response << ":A F=";
		RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), response.str()));
		RETURN_ON_MM_ERROR(hub_->ParseAnswerAfterEquals(tmp));
		bool success;
		if (tmp >= 128)
		{
			success = pProp->Set(g_YesState);
		}
		else
		{
			success = pProp->Set(g_NoState);
		}
		if (!success)
		{
			return DEVICE_INVALID_PROPERTY_VALUE;
		}
		justSet = false;
	}
	else if (eAct == MM::AfterSet)
	{
		justSet = true;
		return OnRBRunning(pProp, MM::BeforeGet);
		// TODO determine how to handle this with shared properties since ring buffer is per-card and not per-axis
		// the reason this property exists (and why it's not a read-only property) are a bit hazy as of mid-2017
	}
	return DEVICE_OK;
}

int CDACXYStage::OnRBDelayBetweenPoints(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	std::ostringstream command;
	command.str("");
	long tmp = 0;
	if (eAct == MM::BeforeGet)
	{
		if (!refreshProps_ && initialized_)
		{
			return DEVICE_OK;
		}
		command << addressChar_ << "RT Z?";
		RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), ":A Z="));
		RETURN_ON_MM_ERROR(hub_->ParseAnswerAfterEquals(tmp));
		if (!pProp->Set(tmp))
		{
			return DEVICE_INVALID_PROPERTY_VALUE;
		}
	}
	else if (eAct == MM::AfterSet)
	{
		if (hub_->UpdatingSharedProperties())
		{
			return DEVICE_OK;
		}
		pProp->Get(tmp);
		command << addressChar_ << "RT Z=" << tmp;
		RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), ":A"));
		command.str(""); command << tmp;
		RETURN_ON_MM_ERROR(hub_->UpdateSharedProperties(addressChar_, pProp->GetName(), command.str()));
	}
	return DEVICE_OK;
}

/////////////// Joystick ///////////////

// ASI controller mirrors by having negative speed, but here we have separate property for mirroring
//   and for speed (which is strictly positive)... that makes this code a bit odd
int CDACXYStage::OnJoystickFastSpeed(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	std::ostringstream command;
	command.str("");
	double tmp = 0;
	if (eAct == MM::BeforeGet)
	{
		if (!refreshProps_ && initialized_)
		{
			return DEVICE_OK;
		}
		command << addressChar_ << "JS X?";
		RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), ":A X="));
		RETURN_ON_MM_ERROR(hub_->ParseAnswerAfterEquals(tmp));
		tmp = abs(tmp);
		if (!pProp->Set(tmp))
		{
			return DEVICE_INVALID_PROPERTY_VALUE;
		}
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(tmp);
		char joystickMirror[MM::MaxStrLength];
		RETURN_ON_MM_ERROR(GetProperty(g_JoystickMirrorPropertyName, joystickMirror));
		command.str("");
		if (strcmp(joystickMirror, g_YesState) == 0)
		{
			command << addressChar_ << "JS X=-" << tmp;
		}
		else
		{
			command << addressChar_ << "JS X=" << tmp;
		}
		RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), ":A"));
	}
	return DEVICE_OK;
}

// ASI controller mirrors by having negative speed, but here we have separate property for mirroring
//   and for speed (which is strictly positive)... that makes this code a bit odd
int CDACXYStage::OnJoystickSlowSpeed(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	std::ostringstream command;
	command.str("");
	double tmp = 0;
	if (eAct == MM::BeforeGet)
	{
		if (!refreshProps_ && initialized_)
		{
			return DEVICE_OK;
		}
		command << addressChar_ << "JS Y?";
		RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), ":A Y="));
		RETURN_ON_MM_ERROR(hub_->ParseAnswerAfterEquals(tmp));
		if (!pProp->Set(tmp))
		{
			return DEVICE_INVALID_PROPERTY_VALUE;
		}
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(tmp);
		char joystickMirror[MM::MaxStrLength];
		RETURN_ON_MM_ERROR(GetProperty(g_JoystickMirrorPropertyName, joystickMirror));
		command.str("");
		if (strcmp(joystickMirror, g_YesState) == 0)
		{
			command << addressChar_ << "JS Y=-" << tmp;
		}
		else
		{
			command << addressChar_ << "JS Y=" << tmp;
		}
		RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), ":A"));
	}
	return DEVICE_OK;
}

// ASI controller mirrors by having negative speed, but here we have separate property for mirroring
//   and for speed (which is strictly positive)... that makes this code a bit odd
int CDACXYStage::OnJoystickMirror(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	std::ostringstream command;
	command.str("");
	double tmp = 0;
	if (eAct == MM::BeforeGet)
	{
		if (!refreshProps_ && initialized_)
		{
			return DEVICE_OK;
		}
		command << addressChar_ << "JS X?";  // query only the fast setting to see if already mirrored
		RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), ":A X="));
		RETURN_ON_MM_ERROR(hub_->ParseAnswerAfterEquals(tmp));
		bool success = 0;
		if (tmp < 0) // speed negative <=> mirrored
		{
			success = pProp->Set(g_YesState);
		}
		else
		{
			success = pProp->Set(g_NoState);
		}
		if (!success)
		{
			return DEVICE_INVALID_PROPERTY_VALUE;
		}
	}
	else if (eAct == MM::AfterSet)
	{
		std::string tmpstr;
		pProp->Get(tmpstr);
		double joystickFast = 0.0;
		RETURN_ON_MM_ERROR(GetProperty(g_JoystickFastSpeedPropertyName, joystickFast));
		double joystickSlow = 0.0;
		RETURN_ON_MM_ERROR(GetProperty(g_JoystickSlowSpeedPropertyName, joystickSlow));
		command.str("");
		if (tmpstr.compare(g_YesState) == 0)
		{
			command << addressChar_ << "JS X=-" << joystickFast << " Y=-" << joystickSlow;
		}
		else
		{
			command << addressChar_ << "JS X=" << joystickFast << " Y=" << joystickSlow;
		}
		RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), ":A"));
	}
	return DEVICE_OK;
}

// interchanges axes for X and Y on the joystick
int CDACXYStage::OnJoystickRotate(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	std::ostringstream command;
	std::ostringstream response;
	command.str("");
	response.str("");
	double tmp = 0;
	if (eAct == MM::BeforeGet)
	{
		if (!refreshProps_ && initialized_)
		{
			return DEVICE_OK;
		}
		command << "J " << axisLetterX_ << "?";  // only look at X axis for joystick
		response << ":A " << axisLetterX_ << "=";
		RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), response.str()));
		RETURN_ON_MM_ERROR(hub_->ParseAnswerAfterEquals(tmp));
		bool success = 0;
		if (tmp == 3) // if set to be Y joystick direction then we are rotated, otherwise assume not rotated
		{
			success = pProp->Set(g_YesState);
		}
		else
		{
			success = pProp->Set(g_NoState);
		}
		if (!success)
		{
			return DEVICE_INVALID_PROPERTY_VALUE;
		}
	}
	else if (eAct == MM::AfterSet)
	{
		// ideally would call OnJoystickEnableDisable but don't know how to get the appropriate pProp
		std::string tmpstr;
		pProp->Get(tmpstr);
		char joystickEnabled[MM::MaxStrLength];
		RETURN_ON_MM_ERROR(GetProperty(g_JoystickEnabledPropertyName, joystickEnabled));
		if (strcmp(joystickEnabled, g_YesState) == 0)
		{
			if (tmpstr.compare(g_YesState) == 0)
			{
				command << "J " << axisLetterX_ << "=3" << " " << axisLetterY_ << "=2";  // rotated
			}
			else
			{
				command << "J " << axisLetterX_ << "=2" << " " << axisLetterY_ << "=3";
			}
		}
		else  // No = disabled
		{
			command << "J " << axisLetterX_ << "=0" << " " << axisLetterY_ << "=0";
		}
		RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), ":A"));
	}
	return DEVICE_OK;
}

int CDACXYStage::OnJoystickEnableDisable(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	std::ostringstream command;
	std::ostringstream response;
	command.str("");
	response.str("");
	long tmp = 0;
	if (eAct == MM::BeforeGet)
	{
		if (!refreshProps_ && initialized_)
		{
			return DEVICE_OK;
		}
		command << "J " << axisLetterX_ << "?";
		response << ":A " << axisLetterX_ << "=";
		RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), response.str()));
		RETURN_ON_MM_ERROR(hub_->ParseAnswerAfterEquals(tmp));
		bool success = 0;
		if (tmp) // treat anything nozero as enabled when reading
		{
			success = pProp->Set(g_YesState);
		}
		else
		{
			success = pProp->Set(g_NoState);
		}
		if (!success)
		{
			return DEVICE_INVALID_PROPERTY_VALUE;
		}
	}
	else if (eAct == MM::AfterSet)
	{
		std::string tmpstr;
		pProp->Get(tmpstr);
		if (tmpstr.compare(g_YesState) == 0)
		{
			char joystickRotate[MM::MaxStrLength];
			RETURN_ON_MM_ERROR(GetProperty(g_JoystickRotatePropertyName, joystickRotate));
			if (strcmp(joystickRotate, g_YesState) == 0)
			{
				command << "J " << axisLetterX_ << "=3" << " " << axisLetterY_ << "=2";  // rotated
			}
			else
			{
				command << "J " << axisLetterX_ << "=2" << " " << axisLetterY_ << "=3";
			}
		}
		else  // No = disabled
		{
			command << "J " << axisLetterX_ << "=0" << " " << axisLetterY_ << "=0";
		}
		RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), ":A"));
	}
	return DEVICE_OK;
}

/////////////// Single Axis Functions ///////////////

int CDACXYStage::OnSAAmplitudeGeneric(MM::PropertyBase* pProp, MM::ActionType eAct, std::string axisLetter)
{
	std::ostringstream command;
	std::ostringstream response;
	command.str("");
	response.str("");
	double tmp = 0;
	if (eAct == MM::BeforeGet)
	{
		if (!refreshProps_ && initialized_)
		{
			return DEVICE_OK;
		}
		command << "SAA " << axisLetter << "?";
		response << ":A " << axisLetter << "=";
		RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), response.str()));
		RETURN_ON_MM_ERROR(hub_->ParseAnswerAfterEquals(tmp));
		if (!pProp->Set(tmp))
		{
			return DEVICE_INVALID_PROPERTY_VALUE;
		}
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(tmp);
		command << "SAA " << axisLetter << "=" << tmp;
		RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), ":A"));
	}
	return DEVICE_OK;
}

int CDACXYStage::OnSAOffsetGeneric(MM::PropertyBase* pProp, MM::ActionType eAct, std::string axisLetter)
{
	std::ostringstream command;
	std::ostringstream response;
	command.str("");
	response.str("");
	double tmp = 0;
	if (eAct == MM::BeforeGet)
	{
		if (!refreshProps_ && initialized_)
		{
			return DEVICE_OK;
		}
		command << "SAO " << axisLetter << "?";
		response << ":A " << axisLetter << "=";
		RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), response.str()));
		RETURN_ON_MM_ERROR(hub_->ParseAnswerAfterEquals(tmp));
		if (!pProp->Set(tmp))
		{
			return DEVICE_INVALID_PROPERTY_VALUE;
		}
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(tmp);
		command << "SAO " << axisLetter << "=" << tmp;
		RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), ":A"));
	}
	return DEVICE_OK;
}

int CDACXYStage::OnSAPeriodGeneric(MM::PropertyBase* pProp, MM::ActionType eAct, std::string axisLetter)
{
	std::ostringstream command;
	std::ostringstream response;
	command.str("");
	response.str("");
	long tmp = 0;
	if (eAct == MM::BeforeGet)
	{
		if (!refreshProps_ && initialized_)
		{
			return DEVICE_OK;
		}
		command << "SAF " << axisLetter << "?";
		response << ":A " << axisLetter << "=";
		RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), response.str()));
		RETURN_ON_MM_ERROR(hub_->ParseAnswerAfterEquals(tmp));
		if (!pProp->Set(tmp))
		{
			return DEVICE_INVALID_PROPERTY_VALUE;
		}
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(tmp);
		command << "SAF " << axisLetter << "=" << tmp;
		RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), ":A"));
	}
	return DEVICE_OK;
}

int CDACXYStage::OnSAModeGeneric(MM::PropertyBase* pProp, MM::ActionType eAct, std::string axisLetter)
{
	static bool justSet = false;
	std::ostringstream command;
	std::ostringstream response;
	command.str("");
	response.str("");
	long tmp = 0;
	if (eAct == MM::BeforeGet)
	{
		if (!refreshProps_ && initialized_ && !justSet)
		{
			return DEVICE_OK;
		}
		command << "SAM " << axisLetter << "?";
		response << ":A " << axisLetter << "=";
		RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), response.str()));
		RETURN_ON_MM_ERROR(hub_->ParseAnswerAfterEquals(tmp));
		bool success;
		switch (tmp)
		{
		case 0: success = pProp->Set(g_SAMode_0); break;
		case 1: success = pProp->Set(g_SAMode_1); break;
		case 2: success = pProp->Set(g_SAMode_2); break;
		case 3: success = pProp->Set(g_SAMode_3); break;
		default:success = 0;                      break;
		}
		if (!success)
		{
			return DEVICE_INVALID_PROPERTY_VALUE;
		}
		justSet = false;
	}
	else if (eAct == MM::AfterSet)
	{
		std::string tmpstr;
		pProp->Get(tmpstr);
		if (tmpstr.compare(g_SAMode_0) == 0)
		{
			tmp = 0;
		}
		else if (tmpstr.compare(g_SAMode_1) == 0)
		{
			tmp = 1;
		}
		else if (tmpstr.compare(g_SAMode_2) == 0)
		{
			tmp = 2;
		}
		else if (tmpstr.compare(g_SAMode_3) == 0)
		{
			tmp = 3;
		}
		else
		{
			return DEVICE_INVALID_PROPERTY_VALUE;
		}
		command << "SAM " << axisLetter << "=" << tmp;
		RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), ":A"));
		// get the updated value right away
		justSet = true;
		return OnSAModeGeneric(pProp, MM::BeforeGet, axisLetter);
	}
	return DEVICE_OK;
}

int CDACXYStage::OnSAPatternGeneric(MM::PropertyBase* pProp, MM::ActionType eAct, std::string axisLetter)
{
	std::ostringstream command;
	std::ostringstream response;
	command.str("");
	response.str("");
	long tmp = 0;
	if (eAct == MM::BeforeGet)
	{
		if (!refreshProps_ && initialized_)
		{
			return DEVICE_OK;
		}
		command << "SAP " << axisLetter << "?";
		response << ":A " << axisLetter << "=";
		RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), response.str()));
		RETURN_ON_MM_ERROR(hub_->ParseAnswerAfterEquals(tmp));
		bool success;
		tmp = tmp & ((long)(BIT2 | BIT1 | BIT0));  // zero all but the lowest 3 bits
		switch (tmp)
		{
		case 0: success = pProp->Set(g_SAPattern_0); break;
		case 1: success = pProp->Set(g_SAPattern_1); break;
		case 2: success = pProp->Set(g_SAPattern_2); break;
		case 3: success = pProp->Set(g_SAPattern_3); break;
		default:success = 0;                         break;
		}
		if (!success)
		{
			return DEVICE_INVALID_PROPERTY_VALUE;
		}
	}
	else if (eAct == MM::AfterSet)
	{
		std::string tmpstr;
		pProp->Get(tmpstr);
		if (tmpstr.compare(g_SAPattern_0) == 0)
		{
			tmp = 0;
		}
		else if (tmpstr.compare(g_SAPattern_1) == 0)
		{
			tmp = 1;
		}
		else if (tmpstr.compare(g_SAPattern_2) == 0)
		{
			tmp = 2;
		}
		else if (tmpstr.compare(g_SAPattern_3) == 0)
		{
			tmp = 3;
		}
		else
		{
			return DEVICE_INVALID_PROPERTY_VALUE;
		}
		// have to get current settings and then modify bits 0-2 from there
		command << "SAP " << axisLetter << "?";
		response << ":A " << axisLetter << "=";
		RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), response.str()));
		long current;
		RETURN_ON_MM_ERROR(hub_->ParseAnswerAfterEquals(current));
		current = current & (~(long)(BIT2 | BIT1 | BIT0));  // set lowest 3 bits to zero
		tmp += current;
		command.str("");
		command << "SAP " << axisLetter << "=" << tmp;
		RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), ":A"));
	}
	return DEVICE_OK;
}

int CDACXYStage::OnSAClkSrcGeneric(MM::PropertyBase* pProp, MM::ActionType eAct, std::string axisLetter)
{
	std::ostringstream command;
	std::ostringstream response;
	command.str("");
	response.str("");
	long tmp = 0;
	if (eAct == MM::BeforeGet)
	{
		if (!refreshProps_ && initialized_)
		{
			return DEVICE_OK;
		}
		command << "SAP " << axisLetter << "?";
		response << ":A " << axisLetter << "=";
		RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), response.str()));
		RETURN_ON_MM_ERROR(hub_->ParseAnswerAfterEquals(tmp));
		bool success;
		tmp = tmp & ((long)(BIT7));  // zero all but bit 7
		switch (tmp)
		{
		case 0: success = pProp->Set(g_SAClkSrc_0);    break;
		case BIT7: success = pProp->Set(g_SAClkSrc_1); break;
		default: success = 0;                          break;
		}
		if (!success)
		{
			return DEVICE_INVALID_PROPERTY_VALUE;
		}
	}
	else if (eAct == MM::AfterSet)
	{
		std::string tmpstr;
		pProp->Get(tmpstr);
		if (tmpstr.compare(g_SAClkSrc_0) == 0)
		{
			tmp = 0;
		}
		else if (tmpstr.compare(g_SAClkSrc_1) == 0)
		{
			tmp = BIT7;
		}
		else
		{
			return DEVICE_INVALID_PROPERTY_VALUE;
		}
		// have to get current settings and then modify bit 7 from there
		command << "SAP " << axisLetter << "?";
		response << ":A " << axisLetter << "=";
		RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), response.str()));
		long current;
		RETURN_ON_MM_ERROR(hub_->ParseAnswerAfterEquals(current));
		current = current & (~(long)(BIT7));  // clear bit 7
		tmp += current;
		command.str("");
		command << "SAP " << axisLetter << "=" << tmp;
		RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), ":A"));
	}
	return DEVICE_OK;
}

int CDACXYStage::OnSAClkPolGeneric(MM::PropertyBase* pProp, MM::ActionType eAct, std::string axisLetter)
{
	std::ostringstream command;
	std::ostringstream response;
	command.str("");
	response.str("");
	long tmp = 0;
	if (eAct == MM::BeforeGet)
	{
		if (!refreshProps_ && initialized_)
		{
			return DEVICE_OK;
		}
		command << "SAP " << axisLetter << "?";
		response << ":A " << axisLetter << "=";
		RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), response.str()));
		RETURN_ON_MM_ERROR(hub_->ParseAnswerAfterEquals(tmp));
		bool success;
		tmp = tmp & ((long)(BIT6));  // zero all but bit 6
		switch (tmp)
		{
		case 0: success = pProp->Set(g_SAClkPol_0);    break;
		case BIT6: success = pProp->Set(g_SAClkPol_1); break;
		default: success = 0;                          break;
		}
		if (!success)
		{
			return DEVICE_INVALID_PROPERTY_VALUE;
		}
	}
	else if (eAct == MM::AfterSet)
	{
		std::string tmpstr;
		pProp->Get(tmpstr);
		if (tmpstr.compare(g_SAClkPol_0) == 0)
		{
			tmp = 0;
		}
		else if (tmpstr.compare(g_SAClkPol_1) == 0)
		{
			tmp = BIT6;
		}
		else
		{
			return DEVICE_INVALID_PROPERTY_VALUE;
		}
		// have to get current settings and then modify bit 6 from there
		command << "SAP " << axisLetter << "?";
		response << ":A " << axisLetter << "=";
		RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), response.str()));
		long current;
		RETURN_ON_MM_ERROR(hub_->ParseAnswerAfterEquals(current));
		current = current & (~(long)(BIT6));  // clear bit 6
		tmp += current;
		command.str("");
		command << "SAP " << axisLetter << "=" << tmp;
		RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), ":A"));
	}
	return DEVICE_OK;
}

// get every single time
int CDACXYStage::OnSAPatternByteGeneric(MM::PropertyBase* pProp, MM::ActionType eAct, std::string axisLetter)
{
	std::ostringstream command;
	std::ostringstream response;
	command.str("");
	response.str("");
	long tmp = 0;
	if (eAct == MM::BeforeGet)
	{
		command << "SAP " << axisLetter << "?";
		response << ":A " << axisLetter << "=";
		RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), response.str()));
		RETURN_ON_MM_ERROR(hub_->ParseAnswerAfterEquals(tmp));
		if (!pProp->Set(tmp))
		{
			return DEVICE_INVALID_PROPERTY_VALUE;
		}
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(tmp);
		command << "SAP " << axisLetter << "=" << tmp;
		RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), ":A"));
	}
	return DEVICE_OK;
}

int CDACXYStage::OnSATTLOutGeneric(MM::PropertyBase* pProp, MM::ActionType eAct, std::string axisLetter)
{
	std::ostringstream command;
	std::ostringstream response;
	command.str("");
	response.str("");
	long tmp = 0;
	if (eAct == MM::BeforeGet)
	{
		if (!refreshProps_ && initialized_)
		{
			return DEVICE_OK;
		}
		command << "SAP " << axisLetter << "?";
		response << ":A " << axisLetter << "=";
		RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), response.str()));
		RETURN_ON_MM_ERROR(hub_->ParseAnswerAfterEquals(tmp));
		bool success;
		tmp = tmp & ((long)(BIT5));  // zero all but bit 5
		switch (tmp)
		{
		case 0: success = pProp->Set(g_SATTLOut_0);    break;
		case BIT5: success = pProp->Set(g_SATTLOut_1); break;
		default: success = 0;                          break;
		}
		if (!success)
		{
			return DEVICE_INVALID_PROPERTY_VALUE;
		}
	}
	else if (eAct == MM::AfterSet)
	{
		std::string tmpstr;
		pProp->Get(tmpstr);
		if (tmpstr.compare(g_SATTLOut_0) == 0)
		{
			tmp = 0;
		}
		else if (tmpstr.compare(g_SATTLOut_1) == 0)
		{
			tmp = BIT5;
		}
		else
		{
			return DEVICE_INVALID_PROPERTY_VALUE;
		}
		// have to get current settings and then modify bit 5 from there
		command << "SAP " << axisLetter << "?";
		response << ":A " << axisLetter << "=";
		RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), response.str()));
		long current;
		RETURN_ON_MM_ERROR(hub_->ParseAnswerAfterEquals(current));
		current = current & (~(long)(BIT5));  // clear bit 5
		tmp += current;
		command.str("");
		command << "SAP " << axisLetter << "=" << tmp;
		RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), ":A"));
	}
	return DEVICE_OK;
}

int CDACXYStage::OnSATTLPolGeneric(MM::PropertyBase* pProp, MM::ActionType eAct, std::string axisLetter)
{
	std::ostringstream command;
	std::ostringstream response;
	command.str("");
	response.str("");
	long tmp = 0;
	if (eAct == MM::BeforeGet)
	{
		if (!refreshProps_ && initialized_)
		{
			return DEVICE_OK;
		}
		command << "SAP " << axisLetter << "?";
		response << ":A " << axisLetter << "=";
		RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), response.str()));
		RETURN_ON_MM_ERROR(hub_->ParseAnswerAfterEquals(tmp));
		bool success;
		tmp = tmp & ((long)(BIT4));  // zero all but bit 4
		switch (tmp)
		{
		case 0: success = pProp->Set(g_SATTLPol_0);    break;
		case BIT4: success = pProp->Set(g_SATTLPol_1); break;
		default: success = 0;                          break;
		}
		if (!success)
		{
			return DEVICE_INVALID_PROPERTY_VALUE;
		}
	}
	else if (eAct == MM::AfterSet)
	{
		string tmpstr;
		pProp->Get(tmpstr);
		if (tmpstr.compare(g_SATTLPol_0) == 0)
		{
			tmp = 0;
		}
		else if (tmpstr.compare(g_SATTLPol_1) == 0)
		{
			tmp = BIT4;
		}
		else
		{
			return DEVICE_INVALID_PROPERTY_VALUE;
		}
		// have to get current settings and then modify bit 4 from there
		command << "SAP " << axisLetter << "?";
		response << ":A " << axisLetter << "=";
		RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), response.str()));
		long current;
		RETURN_ON_MM_ERROR(hub_->ParseAnswerAfterEquals(current));
		current = current & (~(long)(BIT4));  // clear bit 4
		tmp += current;
		command.str("");
		command << "SAP " << axisLetter << "=" << tmp;
		RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), ":A"));
	}
	return DEVICE_OK;
}

// special property, when set to "yes" it creates a set of little-used properties that can be manipulated thereafter
int CDACXYStage::OnSAAdvancedX(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		return DEVICE_OK; // do nothing
	}
	else if (eAct == MM::AfterSet)
	{
		std::string tmpstr;
		pProp->Get(tmpstr);
		if (tmpstr.compare(g_YesState) == 0)
		{
			CPropertyAction* pAct;

			pAct = new CPropertyAction(this, &CDACXYStage::OnSAClkSrcX);
			CreateProperty(g_SAClkSrcXPropertyName, g_SAClkSrc_0, MM::String, false, pAct);
			AddAllowedValue(g_SAClkSrcXPropertyName, g_SAClkSrc_0);
			AddAllowedValue(g_SAClkSrcXPropertyName, g_SAClkSrc_1);
			UpdateProperty(g_SAClkSrcXPropertyName);

			pAct = new CPropertyAction(this, &CDACXYStage::OnSAClkPolX);
			CreateProperty(g_SAClkPolXPropertyName, g_SAClkPol_0, MM::String, false, pAct);
			AddAllowedValue(g_SAClkPolXPropertyName, g_SAClkPol_0);
			AddAllowedValue(g_SAClkPolXPropertyName, g_SAClkPol_1);
			UpdateProperty(g_SAClkPolXPropertyName);

			pAct = new CPropertyAction(this, &CDACXYStage::OnSATTLOutX);
			CreateProperty(g_SATTLOutXPropertyName, g_SATTLOut_0, MM::String, false, pAct);
			AddAllowedValue(g_SATTLOutXPropertyName, g_SATTLOut_0);
			AddAllowedValue(g_SATTLOutXPropertyName, g_SATTLOut_1);
			UpdateProperty(g_SATTLOutXPropertyName);

			pAct = new CPropertyAction(this, &CDACXYStage::OnSATTLPolX);
			CreateProperty(g_SATTLPolXPropertyName, g_SATTLPol_0, MM::String, false, pAct);
			AddAllowedValue(g_SATTLPolXPropertyName, g_SATTLPol_0);
			AddAllowedValue(g_SATTLPolXPropertyName, g_SATTLPol_1);
			UpdateProperty(g_SATTLPolXPropertyName);

			pAct = new CPropertyAction(this, &CDACXYStage::OnSAPatternByteX);
			CreateProperty(g_SAPatternModeXPropertyName, "0", MM::Integer, false, pAct);
			SetPropertyLimits(g_SAPatternModeXPropertyName, 0, 255);
			UpdateProperty(g_SAPatternModeXPropertyName);
		}
	}
	return DEVICE_OK;
}

// special property, when set to "yes" it creates a set of little-used properties that can be manipulated thereafter
int CDACXYStage::OnSAAdvancedY(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		return DEVICE_OK; // do nothing
	}
	else if (eAct == MM::AfterSet)
	{
		std::string tmpstr;
		pProp->Get(tmpstr);
		if (tmpstr.compare(g_YesState) == 0)
		{
			CPropertyAction* pAct;

			pAct = new CPropertyAction(this, &CDACXYStage::OnSAClkSrcY);
			CreateProperty(g_SAClkSrcYPropertyName, g_SAClkSrc_0, MM::String, false, pAct);
			AddAllowedValue(g_SAClkSrcYPropertyName, g_SAClkSrc_0);
			AddAllowedValue(g_SAClkSrcYPropertyName, g_SAClkSrc_1);
			UpdateProperty(g_SAClkSrcYPropertyName);

			pAct = new CPropertyAction(this, &CDACXYStage::OnSAClkPolY);
			CreateProperty(g_SAClkPolYPropertyName, g_SAClkPol_0, MM::String, false, pAct);
			AddAllowedValue(g_SAClkPolYPropertyName, g_SAClkPol_0);
			AddAllowedValue(g_SAClkPolYPropertyName, g_SAClkPol_1);
			UpdateProperty(g_SAClkPolYPropertyName);

			pAct = new CPropertyAction(this, &CDACXYStage::OnSATTLOutY);
			CreateProperty(g_SATTLOutYPropertyName, g_SATTLOut_0, MM::String, false, pAct);
			AddAllowedValue(g_SATTLOutYPropertyName, g_SATTLOut_0);
			AddAllowedValue(g_SATTLOutYPropertyName, g_SATTLOut_1);
			UpdateProperty(g_SATTLOutYPropertyName);

			pAct = new CPropertyAction(this, &CDACXYStage::OnSATTLPolY);
			CreateProperty(g_SATTLPolYPropertyName, g_SATTLPol_0, MM::String, false, pAct);
			AddAllowedValue(g_SATTLPolYPropertyName, g_SATTLPol_0);
			AddAllowedValue(g_SATTLPolYPropertyName, g_SATTLPol_1);
			UpdateProperty(g_SATTLPolYPropertyName);

			pAct = new CPropertyAction(this, &CDACXYStage::OnSAPatternByteY);
			CreateProperty(g_SAPatternModeYPropertyName, "0", MM::Integer, false, pAct);
			SetPropertyLimits(g_SAPatternModeYPropertyName, 0, 255);
			UpdateProperty(g_SAPatternModeYPropertyName);
		}
	}
	return DEVICE_OK;
}

/////////////// Sequencing ///////////////

int CDACXYStage::OnUseSequence(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	std::ostringstream command;
	command.str("");
	if (eAct == MM::BeforeGet)
	{
		if (ttl_trigger_enabled_)
		{
			pProp->Set(g_YesState);
		}
		else
		{
			pProp->Set(g_NoState);
		}
	}
	else if (eAct == MM::AfterSet)
	{
		std::string tmpstr;
		pProp->Get(tmpstr);
		ttl_trigger_enabled_ = (ttl_trigger_supported_ && (tmpstr.compare(g_YesState) == 0));
		return OnUseSequence(pProp, MM::BeforeGet);  // refresh value
	}
	return DEVICE_OK;
}

// enables TTL triggering; doesn't actually start anything going on controller
int CDACXYStage::StartXYStageSequence()
{
	std::ostringstream command;
	command.str("");
	if (!ttl_trigger_supported_)
	{
		return DEVICE_UNSUPPORTED_COMMAND;
	}
	// ensure that ringbuffer pointer points to first entry
	// for now leave the axis_byte unchanged (hopefully default)
	// for now leave mode (RM F) unchanged; would normally be set to 1 and is done in OnRBMode = property "RingBufferMode"
	command << addressChar_ << "RM Z=0";
	RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), ":A"));

	command.str("");
	command << addressChar_ << "TTL X=1";  // switch on TTL triggering
	RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), ":A"));
	return DEVICE_OK;
}

// disables TTL triggering; doesn't actually stop anything already happening on controller
int CDACXYStage::StopXYStageSequence()
{
	std::ostringstream command;
	command.str("");
	if (!ttl_trigger_supported_)
	{
		return DEVICE_UNSUPPORTED_COMMAND;
	}
	command << addressChar_ << "TTL X=0";  // switch off TTL triggering
	RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), ":A"));
	return DEVICE_OK;
}

int CDACXYStage::ClearXYStageSequence()
{
	std::ostringstream command;
	command.str("");
	if (!ttl_trigger_supported_)
	{
		return DEVICE_UNSUPPORTED_COMMAND;
	}
	sequenceX_.clear();
	sequenceY_.clear();
	command << addressChar_ << "RM X=0";  // clear ring buffer
	RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), ":A"));
	return DEVICE_OK;
}

int CDACXYStage::AddToXYStageSequence(double positionX, double positionY)
{
	if (!ttl_trigger_supported_)
	{
		return DEVICE_UNSUPPORTED_COMMAND;
	}
	sequenceX_.push_back(positionX);
	sequenceY_.push_back(positionY);
	return DEVICE_OK;
}

// TODO: unit conversion is needed
int CDACXYStage::SendXYStageSequence()
{
	std::ostringstream command;
	command.str("");
	if (!ttl_trigger_supported_)
	{
		return DEVICE_UNSUPPORTED_COMMAND;
	}
	command << addressChar_ << "RM X=0"; // clear ring buffer
	RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), ":A"));
	for (unsigned i = 0; i < sequenceX_.size(); i++)  // send new points
	{
		command.str("");
		command << "LD " << axisLetterX_ << "=" << sequenceX_[i] << " " << axisLetterY_ << "=" << sequenceY_[i];
		RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), ":A"));
	}
	return DEVICE_OK;
}

/////////////// TTL ///////////////

int CDACXYStage::OnTTLin(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	std::ostringstream command;
	std::ostringstream response;
	command.str("");
	response.str("");
	double tmp = 0;
	if (eAct == MM::BeforeGet)
	{
		if (!refreshProps_ && initialized_)
		{
			return DEVICE_OK;
		}
		command << addressChar_ << "TTL X?";
		response << ":A X=";
		RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), response.str()));
		RETURN_ON_MM_ERROR(hub_->ParseAnswerAfterEquals(tmp));
		if (!pProp->Set(tmp))
		{
			return DEVICE_INVALID_PROPERTY_VALUE;
		}
	}
	else if (eAct == MM::AfterSet)
	{
		if (hub_->UpdatingSharedProperties())
		{
			return DEVICE_OK;
		}
		pProp->Get(tmp);
		command << addressChar_ << "TTL X=" << tmp;
		RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), ":A"));
		command.str(""); command << tmp;
		RETURN_ON_MM_ERROR(hub_->UpdateSharedProperties(addressChar_, pProp->GetName(), command.str()));
	}
	return DEVICE_OK;
}

int CDACXYStage::OnTTLout(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	std::ostringstream command;
	std::ostringstream response;
	command.str("");
	response.str("");
	double tmp = 0;
	if (eAct == MM::BeforeGet)
	{
		if (!refreshProps_ && initialized_)
		{
			return DEVICE_OK;
		}
		command << addressChar_ << "TTL Y?";
		response << ":A Y=";
		RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), response.str()));
		RETURN_ON_MM_ERROR(hub_->ParseAnswerAfterEquals(tmp));
		if (!pProp->Set(tmp))
		{
			return DEVICE_INVALID_PROPERTY_VALUE;
		}
	}
	else if (eAct == MM::AfterSet)
	{
		if (hub_->UpdatingSharedProperties())
		{
			return DEVICE_OK;
		}
		pProp->Get(tmp);
		command << addressChar_ << "TTL Y=" << tmp;
		RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), ":A"));
		command.str(""); command << tmp;
		RETURN_ON_MM_ERROR(hub_->UpdateSharedProperties(addressChar_, pProp->GetName(), command.str()));
	}
	return DEVICE_OK;
}

/////////////// Pre-init Properties ///////////////

int CDACXYStage::OnConversionFactorX(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		// do nothing
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(umToMvX_);
	}
	return DEVICE_OK;
}

int CDACXYStage::OnConversionFactorY(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		// do nothing
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(umToMvY_);
	}
	return DEVICE_OK;
}
