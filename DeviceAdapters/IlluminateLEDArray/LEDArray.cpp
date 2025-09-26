//////////////////////////////////////////////////////////////////////////////
// FILE:          LedArray.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Adapter for illuminate LED controller firmware
//                Needs accompanying firmware to be installed on the LED Array:
//                https://github.com/zfphil/illuminate
//
// COPYRIGHT:     Regents of the University of California
// LICENSE:       LGPL
//
// AUTHOR:        Henry Pinkard, hbp@berkeley.edu, 12/13/2016
// AUTHOR:        Zack Phillips, zkphil@berkeley.edu, 3/1/2019
// AUTHOR:        Zack Phillips, zack@zackphillips.com 8/9/2022
//
//////////////////////////////////////////////////////////////////////////////

#include "LEDArray.h"
#include "ModuleInterface.h"
#include <sstream>
#include <cstdio>
#include <cstring>
#include <string>
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include <algorithm>
#include <math.h>

#ifdef WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif


int LedArray::Initialize()
{
	if (initialized)
		return DEVICE_OK;

	// Clear COM port
	PurgeComPort(port_str.c_str());

	// Reset the LED array
	Reset();

	// Set the device to machine-readable mode
	SetMachineMode(true);

	// Get Device Parameters
	GetDeviceParameters();
	
	// Check version. Round to nearest hundredth.
	interface_version = round(100 * interface_version) / 100;
	if ((interface_version < round(100 * MIN_INTERFACE_VERSION) / 100) || (interface_version > round(100 * MAX_INTERFACE_VERSION) / 100))
	{
		SetErrorText(ILLUMINATE_GENERAL_ERROR_CODE, "Invalid interface version.");
		return ILLUMINATE_GENERAL_ERROR_CODE;
	}

	// Sync Current Parameters
	SyncState();

	// Model Name
	int ret = CreateProperty(g_Keyword_ModelName, "", MM::String, true);
	assert(DEVICE_OK == ret);

	// Manual Command Interface
	CPropertyAction* pCommand = new CPropertyAction(this, &LedArray::OnCommand);
	ret = CreateProperty(g_Keyword_Command, "", MM::String, false, pCommand);
	assert(DEVICE_OK == ret);

	// Most Recent Serial Response
	ret = CreateProperty(g_Keyword_Response, "", MM::String, false);
	assert(DEVICE_OK == ret);

	// Reset state
	CPropertyAction* pActreset = new CPropertyAction(this, &LedArray::OnReset);
	ret = CreateProperty(g_Keyword_Reset, "Reset", MM::String, false, pActreset);
	assert(DEVICE_OK == ret);
	AddAllowedValue(g_Keyword_Reset, "");
	AddAllowedValue(g_Keyword_Reset, "Reset");

	// Annulus Start
	CPropertyAction* pActAnnulusStartNa = new CPropertyAction(this, &LedArray::OnAnnulusStart);
	ret = CreateProperty(g_Keyword_AnnulusStartNa, std::to_string((long double)annulus_width_na).c_str(), MM::Float, false, pActAnnulusStartNa);
	assert(DEVICE_OK == ret);

	// Annulus Width
	CPropertyAction* pActAnnulusWidthNa = new CPropertyAction(this, &LedArray::OnAnnulusWidth);
	ret = CreateProperty(g_Keyword_AnnulusWidthNa, std::to_string((long double)annulus_width_na).c_str(), MM::Float, false, pActAnnulusWidthNa);
	assert(DEVICE_OK == ret);

	// LED indices illumination:
	CPropertyAction* pActled = new CPropertyAction(this, &LedArray::OnManualLedList);
	ret = CreateProperty(g_Keyword_LedList, "0.1.2.3.4", MM::String, false, pActled);
	assert(DEVICE_OK == ret);

	// Brightness
	CPropertyAction* pActbr = new CPropertyAction(this, &LedArray::OnBrightness);
	ret = CreateProperty(g_Keyword_Brightness, std::to_string((long long)brightness).c_str(), MM::Integer, false, pActbr);
	assert(DEVICE_OK == ret);
	SetPropertyLimits(g_Keyword_Brightness, 0, 255);

	if (color_channel_count > 1)
	{
		// Red Color Slider
		CPropertyAction* pActr = new CPropertyAction(this, &LedArray::OnColorBalanceRed);
		ret = CreateProperty(g_Keyword_ColorBalanceRed, std::to_string((long long)color_r).c_str(), MM::Integer, false, pActr);
		assert(DEVICE_OK == ret);
		SetPropertyLimits(g_Keyword_ColorBalanceRed, 0, 255);

		// Green Color Slider
		CPropertyAction* pActg = new CPropertyAction(this, &LedArray::OnColorBalanceGreen);
		ret = CreateProperty(g_Keyword_ColorBalanceGreen, std::to_string((long long)color_g).c_str(), MM::Integer, false, pActg);
		assert(DEVICE_OK == ret);
		SetPropertyLimits(g_Keyword_ColorBalanceGreen, 0, 255);

		// Blue Color Slider
		CPropertyAction* pActb = new CPropertyAction(this, &LedArray::OnColorBalanceBlue);
		ret = CreateProperty(g_Keyword_ColorBalanceBlue, std::to_string((long long)color_b).c_str(), MM::Integer, false, pActb);
		assert(DEVICE_OK == ret);
		SetPropertyLimits(g_Keyword_ColorBalanceBlue, 0, 255);
	}


	// Global Shutter Property
	CPropertyAction* pActGlobalShutter = new CPropertyAction(this, &LedArray::OnGlobalShutter);
	ret = CreateProperty(g_Keyword_GlobalShutter, std::to_string((long double)shutter_open).c_str(), MM::Integer, false, pActGlobalShutter);
	AddAllowedValue(g_Keyword_GlobalShutter, "0");
	AddAllowedValue(g_Keyword_GlobalShutter, "1");
	assert(DEVICE_OK == ret);

	// NA Property
	CPropertyAction* pActNa = new CPropertyAction(this, &LedArray::OnNumericalAperture);
	ret = CreateProperty(g_Keyword_NumericalAperture, std::to_string((long double)numerical_aperture).c_str(), MM::Float, false, pActNa);
	assert(DEVICE_OK == ret);

	// Inner NA Property
	CPropertyAction* pActNai = new CPropertyAction(this, &LedArray::OnInnerNumericalAperture);
	ret = CreateProperty(g_Keyword_InnerNumericalAperture, std::to_string((long double)inner_numerical_aperture).c_str(), MM::Float, false, pActNai);
	assert(DEVICE_OK == ret);

	// LED Array Distance
	CPropertyAction* pActap2 = new CPropertyAction(this, &LedArray::OnDistance);
	ret = CreateProperty(g_Keyword_ArrayDistance, std::to_string((long double)array_distance_z).c_str(), MM::Float, false, pActap2);
	assert(DEVICE_OK == ret);

	// LED Count
	ret = CreateProperty(g_Keyword_LedCount, std::to_string((long long)led_count).c_str(), MM::Integer, true);
	assert(DEVICE_OK == ret);

	// Color channel count
	ret = CreateProperty(g_Keyword_ColorChannelCount, std::to_string((long long)color_channel_count).c_str(), MM::Integer, true);
	assert(DEVICE_OK == ret);

	// Interface version
	ret = CreateProperty(g_Keyword_InterfaceVersion, std::to_string((long double)interface_version).c_str(), MM::Float, true);
	assert(DEVICE_OK == ret);

	// Part number
	ret = CreateProperty(g_Keyword_PartNumber, std::to_string((long long)part_number).c_str(), MM::Integer, true);
	assert(DEVICE_OK == ret);

	// Serial number
	ret = CreateProperty(g_Keyword_SerialNumber, std::to_string((long long)serial_number).c_str(), MM::Integer, true);
	assert(DEVICE_OK == ret);

	// Device MAC address
	CreateProperty(g_Keyword_MacAddress, mac_address_str.c_str(), MM::String, true);
	assert(DEVICE_OK == ret);

	// Trigger input count
	ret = CreateProperty(g_Keyword_TriggerInputCount, std::to_string((long long)trigger_input_count).c_str(), MM::Integer, true);
	assert(DEVICE_OK == ret);
	
	// Trigger input modes
	if (trigger_input_count > 0)
	{
		CPropertyAction* pActTriggerInput0Mode = new CPropertyAction(this, &LedArray::OnTriggerInput0Mode);
		ret = CreateProperty(g_Keyword_TriggerInput0Mode, g_TriggerModeDisabled, MM::String, false, pActTriggerInput0Mode);
		assert(DEVICE_OK == ret);

		AddAllowedValue(g_Keyword_TriggerInput0Mode, g_TriggerModeDisabled);
		AddAllowedValue(g_Keyword_TriggerInput0Mode, g_TriggerModeEveryPattternChange);
		AddAllowedValue(g_Keyword_TriggerInput0Mode, g_TriggerModeEveryPatternCycle);
		AddAllowedValue(g_Keyword_TriggerInput0Mode, g_TriggerModeOnceAtStart);

		CPropertyAction* pActTriggerInput0Polarity = new CPropertyAction(this, &LedArray::OnTriggerInput0Polarity);
		ret = CreateProperty(g_Keyword_TriggerInput0Polarity, g_TriggerModeDisabled, MM::String, false, pActTriggerInput0Polarity);
		assert(DEVICE_OK == ret);

		if (trigger_input_count > 1)
		{
			CPropertyAction* pActTriggerInput1Mode = new CPropertyAction(this, &LedArray::OnTriggerInput1Mode);
			ret = CreateProperty(g_Keyword_TriggerInput1Mode, g_TriggerModeDisabled, MM::String, false, pActTriggerInput1Mode);
			assert(DEVICE_OK == ret);

			AddAllowedValue(g_Keyword_TriggerInput1Mode, g_TriggerModeDisabled);
			AddAllowedValue(g_Keyword_TriggerInput1Mode, g_TriggerModeEveryPattternChange);
			AddAllowedValue(g_Keyword_TriggerInput1Mode, g_TriggerModeEveryPatternCycle);
			AddAllowedValue(g_Keyword_TriggerInput1Mode, g_TriggerModeOnceAtStart);

			CPropertyAction* pActTriggerInput1Polarity = new CPropertyAction(this, &LedArray::OnTriggerInput1Polarity);
			ret = CreateProperty(g_Keyword_TriggerInput1Polarity, g_TriggerModeDisabled, MM::Integer, false, pActTriggerInput1Polarity);
			assert(DEVICE_OK == ret);

		}
	}

	// Trigger output count
	ret = CreateProperty(g_Keyword_TriggerOutputCount, std::to_string((long long)trigger_output_count).c_str(), MM::Integer, true);
	assert(DEVICE_OK == ret);

	// Trigger output modes
	if (trigger_output_count > 0)
	{
		// Trigger Output Modes
		CPropertyAction* pActTriggerOutput0Mode = new CPropertyAction(this, &LedArray::OnTriggerOutput0Mode);
		ret = CreateProperty(g_Keyword_TriggerOutput0Mode, g_TriggerModeDisabled, MM::String, false, pActTriggerOutput0Mode);
		assert(DEVICE_OK == ret);

		AddAllowedValue(g_Keyword_TriggerOutput0Mode, g_TriggerModeDisabled);
		AddAllowedValue(g_Keyword_TriggerOutput0Mode, g_TriggerModeEveryPattternChange);
		AddAllowedValue(g_Keyword_TriggerOutput0Mode, g_TriggerModeEveryPatternCycle);
		AddAllowedValue(g_Keyword_TriggerOutput0Mode, g_TriggerModeOnceAtStart);

		CPropertyAction* pActTriggerOutput0Polarity = new CPropertyAction(this, &LedArray::OnTriggerOutput0Polarity);
		ret = CreateProperty(g_Keyword_TriggerOutput0Polarity, g_TriggerModeDisabled, MM::String, false, pActTriggerOutput0Polarity);
		assert(DEVICE_OK == ret);

		// Trigger Output Pulse Widths
		CPropertyAction* pActTriggerOutput0PulseWidth = new CPropertyAction(this, &LedArray::OnTriggerOutput0PulseWidth);
		ret = CreateProperty(g_Keyword_TriggerOutput0PulseWidth, std::to_string(trigger_output0_pulse_width_s).c_str(), MM::Float, false, pActTriggerOutput0PulseWidth);
		assert(DEVICE_OK == ret);

		// Trigger Output Start Delays
		CPropertyAction* pActTriggerOutput0StartDelay = new CPropertyAction(this, &LedArray::OnTriggerOutput0StartDelay);
		ret = CreateProperty(g_Keyword_TriggerOutput0StartDelay, std::to_string(trigger_output0_start_delay_s).c_str(), MM::Float, false, pActTriggerOutput0StartDelay);
		assert(DEVICE_OK == ret);

		if (trigger_output_count > 1)
		{

			CPropertyAction* pActTriggerOutput1Mode = new CPropertyAction(this, &LedArray::OnTriggerOutput1Mode);
			ret = CreateProperty(g_Keyword_TriggerOutput1Mode, g_TriggerModeDisabled, MM::String, false, pActTriggerOutput1Mode);
			assert(DEVICE_OK == ret);
			 
			AddAllowedValue(g_Keyword_TriggerOutput1Mode, g_TriggerModeDisabled);
			AddAllowedValue(g_Keyword_TriggerOutput1Mode, g_TriggerModeEveryPattternChange);
			AddAllowedValue(g_Keyword_TriggerOutput1Mode, g_TriggerModeEveryPatternCycle);
			AddAllowedValue(g_Keyword_TriggerOutput1Mode, g_TriggerModeOnceAtStart);

			CPropertyAction* pActTriggerOutput1Polarity = new CPropertyAction(this, &LedArray::OnTriggerOutput1Polarity);
			ret = CreateProperty(g_Keyword_TriggerOutput0Polarity, g_TriggerModeDisabled, MM::String, false, pActTriggerOutput1Polarity);
			assert(DEVICE_OK == ret);
			
			CPropertyAction* pActTriggerOutput1PulseWidth = new CPropertyAction(this, &LedArray::OnTriggerOutput1PulseWidth);
			ret = CreateProperty(g_Keyword_TriggerOutput1PulseWidth, std::to_string(trigger_output1_pulse_width_s).c_str(), MM::Float, false, pActTriggerOutput1PulseWidth);
			assert(DEVICE_OK == ret);
			
			CPropertyAction* pActTriggerOutput1StartDelay = new CPropertyAction(this, &LedArray::OnTriggerOutput1StartDelay);
			ret = CreateProperty(g_Keyword_TriggerOutput1StartDelay, std::to_string(trigger_output1_start_delay_s).c_str(), MM::Float, false, pActTriggerOutput1StartDelay);
			assert(DEVICE_OK == ret);
		}
	}

	CPropertyAction* pActTriggerInputTimeout = new CPropertyAction(this, &LedArray::OnTriggerInputTimeout);
	ret = CreateProperty(g_Keyword_TriggerInputTimeout, std::to_string(int(trigger_input_timeout_s)).c_str(), MM::Float, false, pActTriggerInputTimeout);
	assert(DEVICE_OK == ret);

	// Bit depth
	ret = CreateProperty(g_Keyword_BitDepth, std::to_string((long long)bit_depth).c_str(), MM::Integer, true);
	assert(DEVICE_OK == ret);

	// Illumination Pattern allowed values
	CPropertyAction* pActpat = new CPropertyAction(this, &LedArray::OnPattern);
	CreateProperty(g_Keyword_Pattern, g_Pattern_Clear, MM::String, false, pActpat);
	AddAllowedValue(g_Keyword_Pattern, g_Pattern_Clear);
	AddAllowedValue(g_Keyword_Pattern, g_Pattern_Brightfield);
	AddAllowedValue(g_Keyword_Pattern, g_Pattern_Darkfield);
	AddAllowedValue(g_Keyword_Pattern, g_Pattern_DpcTop);
	AddAllowedValue(g_Keyword_Pattern, g_Pattern_DpcBottom);
	AddAllowedValue(g_Keyword_Pattern, g_Pattern_DpcLeft);
	AddAllowedValue(g_Keyword_Pattern, g_Pattern_DpcRight);
	AddAllowedValue(g_Keyword_Pattern, g_Pattern_ColorDpc);
	AddAllowedValue(g_Keyword_Pattern, g_Pattern_ColorDarkfield);
	AddAllowedValue(g_Keyword_Pattern, g_Pattern_ManualLedIndices);
	AddAllowedValue(g_Keyword_Pattern, g_Pattern_Annulus);
	AddAllowedValue(g_Keyword_Pattern, g_Pattern_HalfAnnulusTop);
	AddAllowedValue(g_Keyword_Pattern, g_Pattern_HalfAnnulusBottom);
	AddAllowedValue(g_Keyword_Pattern, g_Pattern_HalfAnnulusLeft);
	AddAllowedValue(g_Keyword_Pattern, g_Pattern_HalfAnnulusRight);
	AddAllowedValue(g_Keyword_Pattern, g_Pattern_CenterLed);
	AddAllowedValue(g_Keyword_Pattern, g_Pattern_Dpc0);
	AddAllowedValue(g_Keyword_Pattern, g_Pattern_Dpc30);
	AddAllowedValue(g_Keyword_Pattern, g_Pattern_Dpc60);
	AddAllowedValue(g_Keyword_Pattern, g_Pattern_Dpc90);
	AddAllowedValue(g_Keyword_Pattern, g_Pattern_Dpc120);
	AddAllowedValue(g_Keyword_Pattern, g_Pattern_Dpc150);
	AddAllowedValue(g_Keyword_Pattern, g_Pattern_Dpc180);
	AddAllowedValue(g_Keyword_Pattern, g_Pattern_Dpc210);
	AddAllowedValue(g_Keyword_Pattern, g_Pattern_Dpc240);
	AddAllowedValue(g_Keyword_Pattern, g_Pattern_Dpc270);
	AddAllowedValue(g_Keyword_Pattern, g_Pattern_Dpc300);
	AddAllowedValue(g_Keyword_Pattern, g_Pattern_Dpc330);
	SetProperty(g_Keyword_Pattern, g_Pattern_Clear);

	// Sequence allowed values
	CPropertyAction* pActseq = new CPropertyAction(this, &LedArray::OnSequence);
	CreateProperty(g_Keyword_Sequence, g_SequenceModeNotRunning, MM::String, false, pActseq);
	AddAllowedValue(g_Keyword_Sequence, g_SequenceModeNotRunning);
	AddAllowedValue(g_Keyword_Sequence, g_SequenceModeAllLeds);
	AddAllowedValue(g_Keyword_Sequence, g_SequenceModeBrightfieldOnly);
	AddAllowedValue(g_Keyword_Sequence, g_SequenceModeDarkfieldOnly);
	AddAllowedValue(g_Keyword_Sequence, g_SequenceModeDpc90);
	SetProperty(g_Keyword_Sequence, g_SequenceModeNotRunning);

	// Sequence Delay
	CPropertyAction* pActseqdelay = new CPropertyAction(this, &LedArray::OnSequenceDelay);
	ret = CreateProperty(g_Keyword_SequenceDelay, std::to_string(int(sequence_delay_s)).c_str(), MM::Float, false, pActseqdelay);
	assert(DEVICE_OK == ret);

	// Sequence Run Count
	CPropertyAction* pActseqRunCount = new CPropertyAction(this, &LedArray::OnSequenceRunCount);
	ret = CreateProperty(g_Keyword_SequenceRunCount, std::to_string(sequence_run_count).c_str(), MM::Integer, false, pActseqRunCount);
	assert(DEVICE_OK == ret);

	// Check status
	ret = UpdateStatus();
	if (ret != DEVICE_OK)
		return ret;
	initialized = true;

	return DEVICE_OK;
}

int LedArray::SendCommand(const char * command, bool get_response)
{
	// Purge COM port
	PurgeComPort(port_str.c_str());

	// Convert command to std::string
	std::string command_str(command);

	// Send command to device
	command_str += "\n";
	WriteToComPort(port_str.c_str(), &((unsigned char *)command_str.c_str())[0], (unsigned int)command_str.length());

	// Get/check response if desired
	if (get_response)
	{
		// Impose a small delay to prevent overloading buffer
		Sleep(SERIAL_DELAY_MS);

		// Get answer
		GetSerialAnswer(port_str.c_str(), "-==-\n", serial_response_str);

		// Set property
		SetProperty(g_Keyword_Response, serial_response_str.c_str());

		// Search for error
		std::string error_flag("ERROR");
		if (serial_response_str.find(error_flag) == 0)
		{
			SetErrorText(ILLUMINATE_GENERAL_ERROR_CODE, serial_response_str.c_str());
			return ILLUMINATE_GENERAL_ERROR_CODE;
		}
		else
			return DEVICE_OK;
	}
	else
		return DEVICE_OK;
}
int LedArray::SyncState()
{

	int response = DEVICE_OK;

	// Get current NA
	response = SendCommand("na", true);
	if (response != DEVICE_OK)
		return response;
	std::string na_str("NA.");
	numerical_aperture = (float)atoi(serial_response_str.substr(serial_response_str.find(na_str) + na_str.length(), serial_response_str.length() - na_str.length()).c_str()) / 100.0;

	// Get current inner NA
	response = SendCommand("nai", true);
	if (response != DEVICE_OK)
		return response;
	std::string nai_str("NAI.");
	inner_numerical_aperture = (float)atoi(serial_response_str.substr(serial_response_str.find(nai_str) + nai_str.length(), serial_response_str.length() - nai_str.length()).c_str()) / 100.0;

	// Get current array distance
	response = SendCommand("sad", true);
	if (response != DEVICE_OK)
		return response;
	std::string sad_str("DZ.");
	array_distance_z = (float)atoi(serial_response_str.substr(serial_response_str.find(sad_str) + sad_str.length(), serial_response_str.length() - sad_str.length()).c_str());

	// Get Current Color
	if ((color_channel_count > 1))
	{

		// We only support three colors for now
		assert((color_channel_count == 3));

		// Get color intensities
		response = SendCommand("sc", true);
		if (response != DEVICE_OK)
			return response;

		// Vector of string to save tokens 
		std::vector <std::string> color_values;

		// stringstream class check1 
		std::stringstream check1(serial_response_str);
		std::string intermediate;

		// Tokenizing w.r.t. space ' ' 
		while (getline(check1, intermediate, '.'))
			color_values.push_back(intermediate);

		if (color_values.size() > 0)
		{
			// Remove first value (the SC)
			color_values.erase(color_values.begin());

			// Assign current color values
			color_r = strtoul((color_values.at(0).c_str()), NULL, false);
			color_g = strtoul((color_values.at(1).c_str()), NULL, false);
			color_b = strtoul((color_values.at(2).c_str()), NULL, false);

			// Red:
			response = SetProperty(g_Keyword_ColorBalanceRed, std::to_string((long long)color_r).c_str());
			if (response != DEVICE_OK)
				return response;

			// Green:
			response = SetProperty(g_Keyword_ColorBalanceGreen, std::to_string((long long)color_g).c_str());
			if (response != DEVICE_OK)
				return response;

			// Blue:
			response = SetProperty(g_Keyword_ColorBalanceBlue, std::to_string((long long)color_b).c_str());
			if (response != DEVICE_OK)
				return response;

		}
	}

	// Get current brightness
	response = SendCommand("sb", true);
	if (response != DEVICE_OK)
		return response;
	std::string brightness_str("SB.");
	brightness = (long)atoi(serial_response_str.substr(serial_response_str.find(brightness_str) + brightness_str.length(), serial_response_str.length() - brightness_str.length()).c_str());

	// Set brightness:
	SetProperty(g_Keyword_Brightness, std::to_string((long long)brightness).c_str());

	// Set Numerical Apertures:
	SetProperty(g_Keyword_NumericalAperture, std::to_string((long double)numerical_aperture).c_str());
	SetProperty(g_Keyword_InnerNumericalAperture, std::to_string((long double)inner_numerical_aperture).c_str());
	SetProperty(g_Keyword_AnnulusStartNa, std::to_string((long double)annulus_start_na).c_str());
	SetProperty(g_Keyword_AnnulusWidthNa, std::to_string((long double)annulus_width_na).c_str());

	// Set Array Distance
	SetProperty(g_Keyword_ArrayDistance, std::to_string((long double)array_distance_z).c_str());

	return DEVICE_OK;
}

int LedArray::SetMachineMode(bool mode)
{
	// Check that we have a controller:
	PurgeComPort(port_str.c_str());

	if (mode)
	{
		// Send command to device
		unsigned char myString[] = "machine\n";
		WriteToComPort(port_str.c_str(), &myString[0], 8);
	}
	else {
		// Send command to device
		unsigned char myString[] = "human\n";
		WriteToComPort(port_str.c_str(), &myString[0], 7);
	}

	std::string answer;
	GetSerialAnswer(port_str.c_str(), "-==-", answer);

	// Set property
	SetProperty(g_Keyword_Response, answer.c_str());

	return DEVICE_OK;
}

int LedArray::GetDeviceParameters()
{
	// Send command to device
	int response = SendCommand("pprops", true);
	if (response != DEVICE_OK)
		return response;

	// Set property
	SetProperty(g_Keyword_Response, serial_response_str.c_str());

	// Set Properties based on JSON output
	rapidjson::Document d;
	d.Parse(serial_response_str.c_str());

	// Parse json if it is valid
	if (d.IsObject())
	{

		// Parse LED count
		if (d.HasMember("led_count"))
		{
			led_count = (long)d["led_count"].GetUint64();
			SetProperty(g_Keyword_LedCount, std::to_string((long long)led_count).c_str());
		}

		// Parse trigger input count
		if (d.HasMember("trigger_input_count"))
		{
			trigger_input_count = (int)d["trigger_input_count"].GetUint64();
			SetProperty(g_Keyword_TriggerInputCount, std::to_string((long long)trigger_input_count).c_str());
		}

		// Parse trigger output count
		if (d.HasMember("trigger_output_count"))
		{
			trigger_output_count = (int)d["trigger_output_count"].GetUint64();
			SetProperty(g_Keyword_TriggerOutputCount, std::to_string((long long)trigger_output_count).c_str());
		}

		// Parse part number
		if (d.HasMember("part_number"))
		{
			part_number = (int)d["part_number"].GetUint64();
			SetProperty(g_Keyword_PartNumber, std::to_string((long long)part_number).c_str());
		}

		// Parse serial number
		if (d.HasMember("serial_number"))
		{
			serial_number = (int)d["serial_number"].GetUint64();
			SetProperty(g_Keyword_SerialNumber, std::to_string((long long)serial_number).c_str());
		}

		// Parse bit depth
		if (d.HasMember("bit_depth"))
		{
			bit_depth = (int)d["bit_depth"].GetUint64();
			SetProperty(g_Keyword_BitDepth, std::to_string((long long)bit_depth).c_str());
		}

		// Parse color channel count
		if (d.HasMember("color_channel_count"))
		{
			color_channel_count = (int)d["color_channel_count"].GetUint64();
			SetProperty(g_Keyword_ColorChannelCount, std::to_string((long long)color_channel_count).c_str());
		}

		// Parse interface version
		if (d.HasMember("interface_version"))
		{
			interface_version = d["interface_version"].GetFloat();
			SetProperty(g_Keyword_InterfaceVersion, std::to_string((long double)interface_version).c_str());
		}

		// Parse mac address
		if (d.HasMember("mac_address"))
		{
			mac_address_str = std::string(d["mac_address"].GetString());
			SetProperty(g_Keyword_MacAddress, mac_address_str.c_str());
		}

		// Parse device name
		if (d.HasMember("device_name"))
		{
			device_name_str = std::string(d["device_name"].GetString());
			SetProperty(g_Keyword_ModelName, device_name_str.c_str());
		}

		// Parse color channel mean wavelengths
		if (d.HasMember("color_channel_center_wavelengths_nm"))
		{
			// Parse infrared channel
			if (d["color_channel_center_wavelengths_nm"].HasMember("ir"))
				CreateProperty(g_Keyword_MeanWavelengthInfrared, std::to_string((long double)d["color_channel_center_wavelengths_nm"]["ir"].GetDouble()).c_str(), MM::String, true);

			// Parse red channel
			if (d["color_channel_center_wavelengths_nm"].HasMember("r"))
				CreateProperty(g_Keyword_MeanWavelengthRed, std::to_string((long double)d["color_channel_center_wavelengths_nm"]["r"].GetDouble()).c_str(), MM::String, true);

			// Parse green channel
			if (d["color_channel_center_wavelengths_nm"].HasMember("g"))
				CreateProperty(g_Keyword_MeanWavelengthGreen, std::to_string((long double)d["color_channel_center_wavelengths_nm"]["g"].GetDouble()).c_str(), MM::String, true);

			// Parse blue channel
			if (d["color_channel_center_wavelengths_nm"].HasMember("b"))
				CreateProperty(g_Keyword_MeanWavelengthBlue, std::to_string((long double)d["color_channel_center_wavelengths_nm"]["b"].GetDouble()).c_str(), MM::String, true);
		
			// Parse uv channel
			if (d["color_channel_center_wavelengths_nm"].HasMember("uv"))
				CreateProperty(g_Keyword_MeanWavelengthUV, std::to_string((long double)d["color_channel_center_wavelengths_nm"]["uv"].GetDouble()).c_str(), MM::String, true);

		}

		// Parse color channel fwhm wavelengths
		if (d.HasMember("color_channel_fwhm_wavelengths_nm"))
		{
			// Parse infrared channel
			if (d["color_channel_fwhm_wavelengths_nm"].HasMember("ir"))
				CreateProperty(g_Keyword_FwhmWavelengthInfrared, std::to_string((long double)d["color_channel_fwhm_wavelengths_nm"]["ir"].GetDouble()).c_str(), MM::String, true);

			// Parse red channel
			if (d["color_channel_fwhm_wavelengths_nm"].HasMember("r"))
				CreateProperty(g_Keyword_FwhmWavelengthRed, std::to_string((long double)d["color_channel_fwhm_wavelengths_nm"]["r"].GetDouble()).c_str(), MM::String, true);

			// Parse green channel
			if (d["color_channel_fwhm_wavelengths_nm"].HasMember("g"))
				CreateProperty(g_Keyword_FwhmWavelengthGreen, std::to_string((long double)d["color_channel_fwhm_wavelengths_nm"]["g"].GetDouble()).c_str(), MM::String, true);

			// Parse blue channel
			if (d["color_channel_fwhm_wavelengths_nm"].HasMember("b"))
				CreateProperty(g_Keyword_FwhmWavelengthBlue, std::to_string((long double)d["color_channel_fwhm_wavelengths_nm"]["b"].GetDouble()).c_str(), MM::String, true);

			// Parse uv channel
			if (d["color_channel_fwhm_wavelengths_nm"].HasMember("uv"))
				CreateProperty(g_Keyword_FwhmWavelengthUV, std::to_string((long double)d["color_channel_fwhm_wavelengths_nm"]["uv"].GetDouble()).c_str(), MM::String, true);

		}

		// Return
		return DEVICE_OK;
	}
	else
		return ILLUMINATE_GENERAL_ERROR_CODE;

}

///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////
MODULE_API void InitializeModuleData()
{
	RegisterDevice(g_Keyword_DeviceAdapterName, MM::ShutterDevice, "Illuminate LED Array");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
	if (deviceName == 0)
		return 0;

	// decide which device class to create based on the deviceName parameter
	if (strcmp(deviceName, g_Keyword_DeviceAdapterName) == 0)
	{
		return new LedArray;
	}

	// ...supplied name not recognized
	return 0;
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
	delete pDevice;
}

///////////////////////////////////////////////////////////////////////////////
// LedArray implementation
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~

LedArray::LedArray(void)
{
	return_code = 0;
	initialized = false;
	port_available = false;
	bit_depth = 0;
	brightness = 0;
	color_channel_count = 0;
	serial_number = 0;
	part_number = 0;
	trigger_input_count = 0;
	trigger_output_count = 0;
	led_count = 0;
	array_distance_z = 0;
	numerical_aperture = 0.25;
	inner_numerical_aperture = 0.0;
	annulus_start_na = numerical_aperture;
	annulus_width_na = 0.2;
	interface_version = 0.0;
	color_r = 0;
	color_g = 0;
	color_b = 0;
	led_positions_cartesian = NULL;
	trigger_output0_pulse_width_s = 0.001;
	trigger_output1_pulse_width_s = 0.001;
	trigger_output0_start_delay_s = 0.0;
	trigger_output1_start_delay_s = 0.0;
	trigger_input0_invert_polarity = true;
	trigger_input1_invert_polarity = false;
	trigger_output0_invert_polarity = false;
	trigger_output1_invert_polarity = false;
	trigger_input0_mode = TriggerMode::Disabled;
	trigger_input1_mode = TriggerMode::Disabled;
	trigger_output0_mode = TriggerMode::Disabled;
	trigger_output1_mode = TriggerMode::Disabled;
	trigger_input_timeout_s = 30.0;
	sequence_delay_s = 0.0;
	sequence_run_count = 1;

	// Initialize default error messages
	InitializeDefaultErrorMessages();

	////pre initialization property: port name
	CPropertyAction* pAct = new CPropertyAction(this, &LedArray::OnPort);
	CreateProperty(MM::g_Keyword_Port, "Undefined", MM::String, false, pAct, true);
}

LedArray::~LedArray()
{
	Shutdown();
}

int LedArray::Shutdown()
{
	if (initialized)
	{
		initialized = false;
	}
	return DEVICE_OK;
}

bool LedArray::Busy()
{
	return false;
}

void LedArray::GetName(char* name) const
{
	CDeviceUtils::CopyLimitedString(name, g_Keyword_DeviceAdapterName);
}

int LedArray::SetBrightness(long new_brightness)
{
	// Initialize Command
	std::string command("sb.");

	// Append Red
	command += std::to_string((long long)new_brightness);

	// Send Command
	return SendCommand(command.c_str(), true);
}

int LedArray::UpdateColor(long redint, long greenint, long blueint)
{
	// Initialize Command
	std::string command("sc.");

	// Append Red
	command += std::to_string((long long)redint);
	command += std::string(".");

	// Append Green
	command += std::to_string((long long)greenint);
	command += std::string(".");

	// Append Blue
	command += std::to_string((long long)blueint);

	// Send Command
	return SendCommand(command.c_str(), true);
}

int LedArray::Reset()
{
	// Send reset command
	return SendCommand("reset", false);
}

int LedArray::SetArrayDistance(double distMM)
{
	// Set Numerical Aperture
	std::string command("sad.");
	command += std::to_string((long long)(distMM));

	// Send Command
	return SendCommand(command.c_str(), true);
}


int LedArray::SetNumericalAperture(double new_numerical_aperture)
{
	// Set Numerical Aperture
	std::string command("na.");
	command += std::to_string((long long)(new_numerical_aperture * 100.0));

	// Send Command
	return SendCommand(command.c_str(), true);
}

int LedArray::SetInnerNumericalAperture(double new_inner_numerical_aperture)
{
	// Set Numerical Aperture
	std::string command("nai.");
	command += std::to_string((long long)(new_inner_numerical_aperture * 100.0));

	// Send Command
	return SendCommand(command.c_str(), true);
}

int LedArray::SetOpen(bool open)
{
	// Set Global Shutter
	shutter_open = int(open);
	std::string command("gs.");
	command += std::to_string((int)(open));

	// Send Command
	return SendCommand(command.c_str(), true);
}

int LedArray::GetOpen(bool& open)
{	
	// Return cached global shutter setting
	open = shutter_open;
	return DEVICE_OK;
}

int LedArray::UpdatePattern() {
	//master function that gets called to send commands to LED array.
	//Waits on response from serial port so that call blocks until pattern
	//is corectly shown
	
	int response = ILLUMINATE_GENERAL_ERROR_CODE;
	if (pattern_str == g_Pattern_Brightfield)
		response = SendCommand("bf", true);
	else if (pattern_str == g_Pattern_Darkfield)
		response = SendCommand("df", true);
	else if (pattern_str == g_Pattern_DpcTop)
		response = SendCommand("dpc.t", true);
	else if (pattern_str == g_Pattern_DpcBottom)
		response = SendCommand("dpc.b", true);
	else if (pattern_str == g_Pattern_DpcLeft)
		response = SendCommand("dpc.l", true);
	else if (pattern_str == g_Pattern_DpcRight)
		response = SendCommand("dpc.r", true);
	else if (pattern_str == g_Pattern_Dpc0)
		response = SendCommand("dpc.0", true);
	else if (pattern_str == g_Pattern_Dpc30)
		response = SendCommand("dpc.30", true);
	else if (pattern_str == g_Pattern_Dpc60)
		response = SendCommand("dpc.60", true);
	else if (pattern_str == g_Pattern_Dpc90)
		response = SendCommand("dpc.90", true);
	else if (pattern_str == g_Pattern_Dpc120)
		response = SendCommand("dpc.120", true);
	else if (pattern_str == g_Pattern_Dpc150)
		response = SendCommand("dpc.150", true);
	else if (pattern_str == g_Pattern_Dpc180)
		response = SendCommand("dpc.180", true);
	else if (pattern_str == g_Pattern_Dpc210)
		response = SendCommand("dpc.210", true);
	else if (pattern_str == g_Pattern_Dpc240)
		response = SendCommand("dpc.240", true);
	else if (pattern_str == g_Pattern_Dpc270)
		response = SendCommand("dpc.270", true);
	else if (pattern_str == g_Pattern_Dpc300)
		response = SendCommand("dpc.300", true);
	else if (pattern_str == g_Pattern_Dpc330)
		response = SendCommand("dpc.330", true);
	else if (pattern_str == g_Pattern_ColorDpc)
		response = SendCommand("cdpc", true);
	else if (pattern_str == g_Pattern_ColorDarkfield)
		response = SendCommand("cdf", true);
	else if (pattern_str == g_Pattern_Annulus)
		response = SendCommand((std::string("an.") + std::to_string((long long)(annulus_start_na * 100.0)) + std::string(".") + std::to_string((long long)(annulus_width_na * 100.0))).c_str(), true);
	else if (pattern_str == g_Pattern_HalfAnnulusTop)
		response = SendCommand((std::string("ha.t.") + std::to_string((long long)(annulus_start_na * 100.0)) + std::string(".") + std::to_string((long long)(annulus_width_na * 100.0))).c_str(), true);
	else if (pattern_str == g_Pattern_HalfAnnulusBottom)
		response = SendCommand((std::string("ha.b.") + std::to_string((long long)(annulus_start_na * 100.0)) + std::string(".") + std::to_string((long long)(annulus_width_na * 100.0))).c_str(), true);
	else if (pattern_str == g_Pattern_HalfAnnulusLeft)
		response = SendCommand((std::string("ha.l.") + std::to_string((long long)(annulus_start_na * 100.0)) + std::string(".") + std::to_string((long long)(annulus_width_na * 100.0))).c_str(), true);
	else if (pattern_str == g_Pattern_HalfAnnulusRight)
		response = SendCommand((std::string("ha.r.") + std::to_string((long long)(annulus_start_na * 100.0)) + std::string(".") + std::to_string((long long)(annulus_width_na * 100.0))).c_str(), true);
	else if (pattern_str == g_Pattern_CenterLed)
		response = SendCommand("l.0", true);
	else if (pattern_str == g_Pattern_Clear)
		response = SendCommand("x", true);
	else if (pattern_str == g_Pattern_ManualLedIndices)
		response = DrawLedList(led_indices_str.c_str());
	else {
		SetErrorText(ILLUMINATE_GENERAL_ERROR_CODE, "Invalid pattern type.");
		return ILLUMINATE_GENERAL_ERROR_CODE;
	}
	return response;
}

int LedArray::DrawLedList(const char* led_list_char)
{
	// Generate command
	std::string command("l.");

	// Replace commas with periods
	std::string led_list_string(led_list_char);
	std::replace(led_list_string.begin(), led_list_string.end(), ',', '.');

	// Append LED list
	command += led_list_string;

	// Send Command
	return SendCommand(command.c_str(), true);
}

///////////////////////////////////////////////////////////////////////////////
// Action handlers
///////////////////////////////////////////////////////////////////////////////

int LedArray::OnPort(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	if (pAct == MM::BeforeGet)
	{
		pProp->Set(port_str.c_str());
	}
	else if (pAct == MM::AfterSet)
	{
		pProp->Get(port_str);
		port_available = true;
	}
	return DEVICE_OK;
}

int LedArray::OnReset(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	if (pAct == MM::BeforeGet)
	{
		pProp->Set("");
	}
	else if (pAct == MM::AfterSet)
	{
		return Reset();
	}
	return DEVICE_OK;
}

int LedArray::OnPattern(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	if (pAct == MM::BeforeGet)
	{
		pProp->Set(pattern_str.c_str());
	}
	else if (pAct == MM::AfterSet)
	{
		pProp->Get(pattern_str);
		// Set pattern to clear if Manual and list of LEDs is empty
		if ((pattern_str == g_Pattern_ManualLedIndices) && (led_indices_str.size() == 0))
			pattern_str = g_Pattern_Clear;

		return_code = UpdatePattern();
		if (return_code != DEVICE_OK)
			return return_code;
	}
	return DEVICE_OK;
}

int LedArray::OnSequence(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	if (pAct == MM::BeforeGet)
	{
		pProp->Set(sequence_str.c_str());
		return DEVICE_OK;
	}
	else if (pAct == MM::AfterSet)
	{
		pProp->Get(sequence_str);

		int response = ILLUMINATE_GENERAL_ERROR_CODE;
		if (sequence_str == g_SequenceModeNotRunning)
			response = DEVICE_OK;
		else if (sequence_str == g_SequenceModeAllLeds)
		{
			std::string command = "scf." + std::to_string(SecondsToMilliseconds(sequence_delay_s)) + "." + std::to_string(sequence_run_count);
			response = SendCommand(command.c_str(), true);
		}
		else if (sequence_str == g_SequenceModeBrightfieldOnly)
		{
			std::string command = "scb." + std::to_string(SecondsToMilliseconds(sequence_delay_s)) + "." + std::to_string(sequence_run_count);
			response = SendCommand(command.c_str(), true);
		}
		else if (sequence_str == g_SequenceModeDarkfieldOnly)
		{
			std::string command = "scd." + std::to_string(SecondsToMilliseconds(sequence_delay_s)) + "." + std::to_string(sequence_run_count);
			response = SendCommand(command.c_str(), true);
		}
		else if (sequence_str == g_SequenceModeDpc90)
		{
			std::string command = "rdpc." + std::to_string(SecondsToMilliseconds(sequence_delay_s)) + "." + std::to_string(sequence_run_count);
			response = SendCommand(command.c_str(), true);
		}
		else {
			SetErrorText(ILLUMINATE_GENERAL_ERROR_CODE, (std::string("Invalid sequence type: ") + std::string(sequence_str)).c_str());
			return ILLUMINATE_GENERAL_ERROR_CODE;
		}

		// Set param back, to indicate successful kick-off. We can't tell when it's done so we'll just leave it blank.
		sequence_str = g_SequenceModeNotRunning;

		//SetProperty(g_Keyword_Sequence, g_SequenceModeNotRunning);
		return response;
	}
}

int LedArray::OnManualLedList(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	if (pAct == MM::BeforeGet) {
		pProp->Set(led_indices_str.c_str());
	}
	else if (pAct == MM::AfterSet)
	{
		pProp->Get(led_indices_str);
		pattern_str = std::string(g_Pattern_ManualLedIndices);
		return_code = UpdatePattern();
		if (return_code != DEVICE_OK)
			return return_code;
	}

	return DEVICE_OK;
}


int LedArray::OnCommand(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	if (pAct == MM::BeforeGet)
	{
		pProp->Set(serial_command_str.c_str());
	}
	else if (pAct == MM::AfterSet)
	{
		pProp->Get(serial_command_str);
		return_code = SendCommand(serial_command_str.c_str(), true);
		if (return_code != DEVICE_OK)
			return return_code;
	}

	// Return
	return DEVICE_OK;
}

int LedArray::OnAnnulusWidth(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	if (pAct == MM::BeforeGet)
	{
		pProp->Set(annulus_width_na);
	}
	else if (pAct == MM::AfterSet)
	{
		pProp->Get(annulus_width_na);
		return_code = UpdatePattern();
		if (return_code != DEVICE_OK)
			return return_code;
	}
	return DEVICE_OK;
}

int LedArray::OnAnnulusStart(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	if (pAct == MM::BeforeGet)
	{
		pProp->Set(annulus_start_na);
	}
	else if (pAct == MM::AfterSet)
	{
		pProp->Get(annulus_start_na);
		return_code = UpdatePattern();
		if (return_code != DEVICE_OK)
			return return_code;

	}
	return DEVICE_OK;
}


int LedArray::OnBrightness(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	if (pAct == MM::BeforeGet)
	{
		pProp->Set(brightness);
	}
	else if (pAct == MM::AfterSet)
	{
		pProp->Get(brightness);
		return_code = SetBrightness(brightness);
		if (return_code != DEVICE_OK)
			return return_code;
		return_code = UpdatePattern();
		if (return_code != DEVICE_OK)
			return return_code;
	}
	return DEVICE_OK;
}

int LedArray::OnColorBalanceRed(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	if (pAct == MM::BeforeGet)
	{
		pProp->Set(color_r);
	}
	else if (pAct == MM::AfterSet)
	{
		pProp->Get(color_r);
		return_code = UpdateColor(color_r, color_g, color_b);
		if (return_code != DEVICE_OK)
			return return_code;
		return_code = UpdatePattern();
		if (return_code != DEVICE_OK)
			return return_code;
	}
	return DEVICE_OK;
}

int LedArray::OnColorBalanceBlue(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	if (pAct == MM::BeforeGet)
	{
		pProp->Set(color_b);
	}
	else if (pAct == MM::AfterSet)
	{
		pProp->Get(color_b);
		return_code = UpdateColor(color_r, color_g, color_b);
		if (return_code != DEVICE_OK)
			return return_code;
		return_code = UpdatePattern();
		if (return_code != DEVICE_OK)
			return return_code;
	}
	return DEVICE_OK;
}

int LedArray::OnColorBalanceGreen(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	if (pAct == MM::BeforeGet)
	{
		pProp->Set(color_g);
	}
	else if (pAct == MM::AfterSet)
	{
		return_code = UpdateColor(color_r, color_g, color_b);
		if (return_code != DEVICE_OK)
			return return_code;
		return_code = UpdatePattern();
		if (return_code != DEVICE_OK)
			return return_code;
		pProp->Get(color_g);
	}
	return DEVICE_OK;
}

int LedArray::OnDistance(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	if (pAct == MM::BeforeGet)
	{
		pProp->Set(array_distance_z);
	}
	else if (pAct == MM::AfterSet)
	{
		pProp->Get(array_distance_z);
		return_code = SetArrayDistance(array_distance_z);
		if (return_code != DEVICE_OK)
			return return_code;
		return_code = UpdatePattern();
		if (return_code != DEVICE_OK)
			return return_code;
	}
	return DEVICE_OK;
}

int LedArray::OnGlobalShutter(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	if (pAct == MM::BeforeGet)
	{
		pProp->Set(shutter_open);
	}
	else if (pAct == MM::AfterSet)
	{
		return_code = SetOpen(bool(shutter_open));
		if (return_code != DEVICE_OK)
			return return_code;
		pProp->Get(shutter_open);
	}
	return DEVICE_OK;
}

int LedArray::OnNumericalAperture(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	if (pAct == MM::BeforeGet)
	{
		pProp->Set(numerical_aperture);
	}
	else if (pAct == MM::AfterSet)
	{
		pProp->Get(numerical_aperture);
		return_code = SetNumericalAperture(numerical_aperture);
		if (return_code != DEVICE_OK)
			return return_code;
		return_code = UpdatePattern();
		if (return_code != DEVICE_OK)
			return return_code;
	}
	return DEVICE_OK;
}

int LedArray::OnInnerNumericalAperture(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	if (pAct == MM::BeforeGet)
	{
		pProp->Set(inner_numerical_aperture);
	}
	else if (pAct == MM::AfterSet)
	{
		pProp->Get(inner_numerical_aperture);
		return_code = SetInnerNumericalAperture(inner_numerical_aperture);
		if (return_code != DEVICE_OK)
			return return_code;
		return_code = UpdatePattern();
		if (return_code != DEVICE_OK)
			return return_code;
	}
	return DEVICE_OK;
}

LedArray::TriggerMode LedArray::StringToTriggerMode(const std::string& modeStr)
{
	if (modeStr == "Disabled") return Disabled;
	if (modeStr == "Every Pattern Change") return EveryPatternChange;
	if (modeStr == "Every Pattern Cycle") return EveryPatternCycle;
	if (modeStr == "Start of First Pattern") return OnceAtStart;

	throw std::invalid_argument("Invalid TriggerMode string: " + modeStr);
}

std::string LedArray::TriggerModeToString(TriggerMode mode)
{
	switch (mode)
	{
	case Disabled: return std::string("Disabled");
	case EveryPatternChange: return std::string("Every Pattern Change");
	case EveryPatternCycle: return std::string("Every Pattern Cycle");
	case OnceAtStart: return std::string("Start of First Pattern");
	default: throw std::invalid_argument("Invalid TriggerMode integer");
	}
}

long LedArray::SecondsToMicroseconds(double seconds)
{
	return long(seconds * 1000000.0);
}

long LedArray::SecondsToMilliseconds(double seconds)
{
	return long(seconds * 1000.0);
}

double LedArray::MicrosecondsToSeconds(long microseconds)
{
	return double(microseconds) / 1000000.0;
}


int LedArray::OnTriggerInput0Mode(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	if (pAct == MM::BeforeGet) {
		pProp->Set(TriggerModeToString(trigger_input0_mode).c_str());
	}
	else if (pAct == MM::AfterSet) {
		std::string mode;
		pProp->Get(mode);
		trigger_input0_mode = StringToTriggerMode(mode);
		std::string command = "tim.0." + std::to_string(static_cast<int>(trigger_input0_mode));
		return SendCommand(command.c_str(), true);
	}
	return DEVICE_OK;
}

int LedArray::OnTriggerInput1Mode(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	if (pAct == MM::BeforeGet) {
		pProp->Set(TriggerModeToString(trigger_input1_mode).c_str());
	}
	else if (pAct == MM::AfterSet) {
		std::string mode;
		pProp->Get(mode);
		trigger_input1_mode = StringToTriggerMode(mode);
		std::string command = "tim.1." + std::to_string(static_cast<int>(trigger_input1_mode));
		return SendCommand(command.c_str(), true);
	}
	return DEVICE_OK;
}

int LedArray::OnTriggerInput0Polarity(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	if (pAct == MM::BeforeGet) {
		pProp->Set(std::to_string(int(trigger_input0_invert_polarity)).c_str());
	}
	else if (pAct == MM::AfterSet) {
		std::string polarity_str;
		pProp->Get(polarity_str);
		trigger_input0_invert_polarity = bool(std::stoi(polarity_str));
		std::string command = "tipol.0." + std::to_string(int(!trigger_input0_invert_polarity));
		return SendCommand(command.c_str(), true);
	}
	return DEVICE_OK;
}

int LedArray::OnTriggerInput1Polarity(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	if (pAct == MM::BeforeGet) {
		pProp->Set(std::to_string(int(trigger_input1_invert_polarity)).c_str());
	}
	else if (pAct == MM::AfterSet) {
		std::string polarity_str;
		pProp->Get(polarity_str);
		trigger_input1_invert_polarity = bool(std::stoi(polarity_str));
		std::string command = "tipol.1." + std::to_string(int(!trigger_input1_invert_polarity));
		return SendCommand(command.c_str(), true);
	}
	return DEVICE_OK;
}

int LedArray::OnTriggerOutput0Mode(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	if (pAct == MM::BeforeGet) {
		pProp->Set(TriggerModeToString(trigger_output0_mode).c_str());
	}
	else if (pAct == MM::AfterSet) {
		std::string mode;
		pProp->Get(mode);
		trigger_output0_mode = StringToTriggerMode(mode);
		std::string command = "topol.0." + std::to_string(static_cast<int>(trigger_output0_mode));
		return SendCommand(command.c_str(), true);
	}
	return DEVICE_OK;
}

int LedArray::OnTriggerOutput1Mode(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	if (pAct == MM::BeforeGet) {
		pProp->Set(TriggerModeToString(trigger_output1_mode).c_str());
	}
	else if (pAct == MM::AfterSet) {
		std::string mode;
		pProp->Get(mode);
		trigger_output1_mode = StringToTriggerMode(mode);
		std::string command = "tom.1." + std::to_string(static_cast<int>(trigger_output1_mode));
		return SendCommand(command.c_str(), true);
	}
	return DEVICE_OK;
}

int LedArray::OnTriggerOutput0Polarity(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	if (pAct == MM::BeforeGet) {
		pProp->Set(std::to_string(int(trigger_output0_invert_polarity)).c_str());
	}
	else if (pAct == MM::AfterSet) {
		std::string polarity_str;
		pProp->Get(polarity_str);
		trigger_output0_invert_polarity = bool(std::stoi(polarity_str));
		std::string command = "topol.0." + std::to_string(int(!trigger_output0_invert_polarity));
		return SendCommand(command.c_str(), true);
	}
	return DEVICE_OK;
}

int LedArray::OnTriggerOutput1Polarity(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	if (pAct == MM::BeforeGet) {
		pProp->Set(std::to_string(int(trigger_output1_invert_polarity)).c_str());
	}
	else if (pAct == MM::AfterSet) {
		std::string polarity_str;
		pProp->Get(polarity_str);
		trigger_output1_invert_polarity = bool(std::stoi(polarity_str));
		std::string command = "topol.1." + std::to_string(int(!trigger_output1_invert_polarity));
		return SendCommand(command.c_str(), true);
	}
	return DEVICE_OK;
}

int LedArray::OnTriggerOutput0PulseWidth(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	if (pAct == MM::BeforeGet) {
		pProp->Set(trigger_output0_pulse_width_s);
	}
	else if (pAct == MM::AfterSet) {
		pProp->Get(trigger_output0_pulse_width_s);
		std::string command = "topw.0." + std::to_string(SecondsToMicroseconds(trigger_output0_pulse_width_s));
		return SendCommand(command.c_str(), true);
	}
	return DEVICE_OK;
}

int LedArray::OnTriggerOutput1PulseWidth(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	if (pAct == MM::BeforeGet) {
		pProp->Set(trigger_output1_pulse_width_s);
	}
	else if (pAct == MM::AfterSet) {
		pProp->Get(trigger_output1_pulse_width_s);
		std::string command = "topw.1." + std::to_string(SecondsToMicroseconds(trigger_output1_pulse_width_s));
		return SendCommand(command.c_str(), true);
	}
	return DEVICE_OK;
}

int LedArray::OnTriggerOutput0StartDelay(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	if (pAct == MM::BeforeGet) {
		pProp->Set(trigger_output0_start_delay_s);
	}
	else if (pAct == MM::AfterSet) {
		pProp->Get(trigger_output0_start_delay_s);
		std::string command = "tosd.0." + std::to_string(SecondsToMicroseconds(trigger_output0_start_delay_s));
		return SendCommand(command.c_str(), true);
	}
	return DEVICE_OK;
}

int LedArray::OnTriggerOutput1StartDelay(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	if (pAct == MM::BeforeGet) {
		pProp->Set(trigger_output1_start_delay_s);
	}
	else if (pAct == MM::AfterSet) {
		pProp->Get(trigger_output1_start_delay_s);
		std::string command = "tosd.1." + std::to_string(SecondsToMicroseconds(trigger_output1_start_delay_s));
		return SendCommand(command.c_str(), true);
	}
	return DEVICE_OK;
}

int LedArray::OnTriggerInputTimeout(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	if (pAct == MM::BeforeGet) {
		pProp->Set(trigger_input_timeout_s);
	}
	else if (pAct == MM::AfterSet) {
		pProp->Get(trigger_input_timeout_s);
		std::string command = "tit." + std::to_string(int(trigger_input_timeout_s));
		return SendCommand(command.c_str(), true);
	}
	return DEVICE_OK;
}

int LedArray::OnSequenceDelay(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	if (pAct == MM::BeforeGet) {
		pProp->Set(sequence_delay_s);
	}
	else if (pAct == MM::AfterSet) {
		pProp->Get(sequence_delay_s);
	}
	return DEVICE_OK;
}

int LedArray::OnSequenceRunCount(MM::PropertyBase* pProp, MM::ActionType pAct)
{
	if (pAct == MM::BeforeGet) {
		pProp->Set(double(sequence_run_count));
	}
	else if (pAct == MM::AfterSet) {
		pProp->Get(sequence_run_count);
	}
	return DEVICE_OK;
}
