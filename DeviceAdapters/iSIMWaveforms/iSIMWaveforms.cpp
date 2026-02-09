///////////////////////////////////////////////////////////////////////////////
// FILE:          iSIMWaveforms.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Generates analog waveforms for an iSIM microscope.
//
// AUTHOR:        Kyle M. Douglass, https://kylemdouglass.com
//
// VERSION:       0.0.0
//
// FIRMWARE:      xxx
//                
// COPYRIGHT:     ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland
//                Laboratory of Experimental Biophysics (LEB), 2026
//

#include "iSIMWaveforms.h"
#include "MockDAQAdapter.h"
#include "NIDAQmxAdapter.h"
#include "ModuleInterface.h"

#include <cmath>
#include <stdexcept>

using namespace std;

const char* g_DeviceName = "iSIM Waveforms";
const char* g_DeviceDescription = "Generates analog waveforms for an iSIM microscope";
const char* g_Undefined = "Undefined";
const char* g_PhysicalCamera = "Physical Camera";

// Semantic channel property name constants
const char* const iSIMWaveforms::PROP_GALVO_CHANNEL = "Galvo Waveform Channel";
const char* const iSIMWaveforms::PROP_CAMERA_CHANNEL = "Camera Trigger Channel";
const char* const iSIMWaveforms::PROP_AOTF_BLANKING_CHANNEL = "AOTF Blanking Channel";
const char* const iSIMWaveforms::PROP_AOTF_MOD_IN_1 = "AOTF MOD IN Channel 1";
const char* const iSIMWaveforms::PROP_AOTF_MOD_IN_2 = "AOTF MOD IN Channel 2";
const char* const iSIMWaveforms::PROP_AOTF_MOD_IN_3 = "AOTF MOD IN Channel 3";
const char* const iSIMWaveforms::PROP_AOTF_MOD_IN_4 = "AOTF MOD IN Channel 4";

///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////

/**
 * List all supported hardware devices here
 */
MODULE_API void InitializeModuleData()
{
   RegisterDevice(g_DeviceName, MM::CameraDevice, "iSIM Waveforms");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
   if (deviceName == 0)
      return 0;

   // decide which device class to create based on the deviceName parameter
   if (strcmp(deviceName, g_DeviceName) == 0)
   {
      // create the test device
      return new iSIMWaveforms();
   }

   // ...supplied name not recognized
   return 0;
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
   delete pDevice;
}

///////////////////////////////////////////////////////////////////////////////
// iSIMWaveforms implementation
// ~~~~~~~~~~~~~~~~~~~~~~~

/**
* iSIMWaveforms constructor.
* Setup default all variables and create device properties required to exist
* before intialization. In this case, no such properties were required. All
* properties will be created in the Initialize() method.
*
* As a general guideline Micro-Manager devices do not access hardware in the
* the constructor. We should do as little as possible in the constructor and
* perform most of the initialization in the Initialize() method.
*/
iSIMWaveforms::iSIMWaveforms() :
	initialized_(false),
	deviceName_(""),
	aotfBlankingVoltage_(0.0),
	samplingRateHz_(50000.0),
	frameIntervalMs_(50.0),
	readoutTimeMs_(22.94),
	parkingFraction_(0.8),
	exposurePpV_(0.46),
	galvoOffsetV_(-0.075),
	cameraPulseVoltage_(5.0),
	triggerSource_("/Dev1/PFI3"),
	counterChannel_("Dev1/ctr1"),
	clockSource_("/Dev1/Ctr1InternalOutput"),
	waveformRunning_(false)
{
	InitializeDefaultErrorMessages();

	// Register custom error messages
	SetErrorText(ERR_INSUFFICIENT_AO_CHANNELS,
		"The selected device must have at least 4 analog output channels. "
		"Please select a device with more AO channels.");
	SetErrorText(ERR_DUPLICATE_CHANNEL_MAPPING,
		"A hardware channel can only be mapped to one semantic channel. "
		"Please ensure each AO channel is used only once.");
	SetErrorText(ERR_NO_MOD_IN_CONFIGURED,
		"At least one AOTF MOD IN channel must be configured. "
		"Please select a hardware channel for at least one MOD IN.");
	SetErrorText(ERR_REQUIRED_CHANNEL_NOT_SET,
		"Galvo Waveform, Camera Trigger, and AOTF Blanking channels are required. "
		"Please select a hardware channel for each.");
	SetErrorText(ERR_FRAME_INTERVAL_TOO_SHORT,
		"Frame interval must be greater than readout time.");
	SetErrorText(ERR_GALVO_VOLTAGE_OUT_OF_RANGE,
		"Galvo waveform voltage exceeds configured voltage limits.");
	SetErrorText(ERR_NO_MOD_IN_ENABLED,
		"At least one AOTF MOD IN channel must be enabled for waveform generation.");
	SetErrorText(ERR_WAVEFORM_GENERATION_FAILED,
		"Failed to generate waveforms. Check DAQ configuration.");
	SetErrorText(ERR_NO_PHYSICAL_CAMERA,
		"No physical camera selected. Please select a camera in the Physical Camera property.");
	SetErrorText(ERR_INVALID_CAMERA,
		"The selected physical camera is invalid or not loaded.");

	// Create DAQ adapter for device discovery
	// Toggle between Mock and NIDAQmx by commenting/uncommenting:
	auto mockDaq = std::make_unique<MockDAQAdapter>();
	mockDaq->setLogger([this](const std::string& msg) {
		LogMessage(msg, false);
	});
	daq_ = std::move(mockDaq);
	//daq_ = std::make_unique<NIDAQmxAdapter>();

	// Pre-init property: Device
	std::vector<std::string> devices = daq_->getDeviceNames();
	CPropertyAction* pAct = new CPropertyAction(this, &iSIMWaveforms::OnDevice);
	std::string defaultDevice = devices.empty() ? "" : devices[0];
	deviceName_ = defaultDevice;
	CreateStringProperty("Device", defaultDevice.c_str(), false, pAct, true);
	for (const auto& dev : devices)
		AddAllowedValue("Device", dev.c_str());

	// Discover channels for default device
	availableChannels_ = daq_->getAnalogOutputChannels(deviceName_);

	// Initialize semantic channel defaults and create pre-init properties
	InitializeChannelDefaults();
	CreateChannelPreInitProperties();

	// Pre-init property: Counter Channel
	pAct = new CPropertyAction(this, &iSIMWaveforms::OnCounterChannel);
	CreateStringProperty("Counter Channel", counterChannel_.c_str(), false, pAct, true);

	// Pre-init property: Clock Source
	pAct = new CPropertyAction(this, &iSIMWaveforms::OnClockSource);
	CreateStringProperty("Clock Source", clockSource_.c_str(), false, pAct, true);
}

/**
* iSIMWaveforms destructor.
* If this device used as intended within the Micro-Manager system,
* Shutdown() will be always called before the destructor. But in any case
* we need to make sure that all resources are properly released even if
* Shutdown() was not called.
*/
iSIMWaveforms::~iSIMWaveforms()
{
   if (initialized_)
      Shutdown();
}

/////////////////////////////////////////////
// Helper methods for channel configuration
/////////////////////////////////////////////

std::vector<const char*> iSIMWaveforms::GetSemanticChannelNames()
{
	return {
		PROP_GALVO_CHANNEL,
		PROP_CAMERA_CHANNEL,
		PROP_AOTF_BLANKING_CHANNEL,
		PROP_AOTF_MOD_IN_1,
		PROP_AOTF_MOD_IN_2,
		PROP_AOTF_MOD_IN_3,
		PROP_AOTF_MOD_IN_4
	};
}

std::vector<const char*> iSIMWaveforms::GetModInChannelNames()
{
	return {
		PROP_AOTF_MOD_IN_1,
		PROP_AOTF_MOD_IN_2,
		PROP_AOTF_MOD_IN_3,
		PROP_AOTF_MOD_IN_4
	};
}

void iSIMWaveforms::InitializeChannelDefaults()
{
	for (const auto& channel : GetSemanticChannelNames())
	{
		channelMapping_[channel] = "None";
		minVoltage_[channel] = -10.0;
		maxVoltage_[channel] = 10.0;
	}
}

void iSIMWaveforms::CreateChannelPreInitProperties()
{
	for (const auto& semanticChannel : GetSemanticChannelNames())
	{
		// Channel selection property (pre-init)
		CPropertyAction* pAct = new CPropertyAction(this, &iSIMWaveforms::OnChannelMapping);
		CreateStringProperty(semanticChannel, "None", false, pAct, true);
		AddAllowedValue(semanticChannel, "None");
		for (const auto& hwChannel : availableChannels_)
			AddAllowedValue(semanticChannel, hwChannel.c_str());

		// Min voltage property (pre-init)
		std::string minVoltProp = std::string(semanticChannel) + " Min Voltage";
		CPropertyAction* pActMin = new CPropertyAction(this, &iSIMWaveforms::OnMinVoltage);
		CreateFloatProperty(minVoltProp.c_str(), -10.0, false, pActMin, true);

		// Max voltage property (pre-init)
		std::string maxVoltProp = std::string(semanticChannel) + " Max Voltage";
		CPropertyAction* pActMax = new CPropertyAction(this, &iSIMWaveforms::OnMaxVoltage);
		CreateFloatProperty(maxVoltProp.c_str(), 10.0, false, pActMax, true);
	}
}

void iSIMWaveforms::UpdateChannelAllowedValues()
{
	for (const auto& semanticChannel : GetSemanticChannelNames())
	{
		ClearAllowedValues(semanticChannel);
		AddAllowedValue(semanticChannel, "None");
		for (const auto& hwChannel : availableChannels_)
			AddAllowedValue(semanticChannel, hwChannel.c_str());

		// Reset mapping if current value is no longer valid
		std::string currentMapping = channelMapping_[semanticChannel];
		if (currentMapping != "None")
		{
			bool found = false;
			for (const auto& hwChannel : availableChannels_)
			{
				if (hwChannel == currentMapping)
				{
					found = true;
					break;
				}
			}
			if (!found)
			{
				channelMapping_[semanticChannel] = "None";
				SetProperty(semanticChannel, "None");
			}
		}
	}
}

int iSIMWaveforms::ValidateMinimumChannels()
{
	const size_t minimumChannels = 4;

	if (availableChannels_.size() < minimumChannels)
	{
		LogMessage("Device has " + std::to_string(availableChannels_.size()) +
			" AO channels, but at least " + std::to_string(minimumChannels) +
			" are required", false);
		return ERR_INSUFFICIENT_AO_CHANNELS;
	}
	return DEVICE_OK;
}

int iSIMWaveforms::ValidateRequiredChannels()
{
	const std::vector<const char*> requiredChannels = {
		PROP_GALVO_CHANNEL,
		PROP_CAMERA_CHANNEL,
		PROP_AOTF_BLANKING_CHANNEL
	};

	for (const auto& channel : requiredChannels)
	{
		if (channelMapping_[channel] == "None")
		{
			LogMessage(std::string("Required channel not set: ") + channel, false);
			return ERR_REQUIRED_CHANNEL_NOT_SET;
		}
	}
	return DEVICE_OK;
}

int iSIMWaveforms::ValidateChannelMappings()
{
	std::set<std::string> usedChannels;

	for (const auto& semantic : GetSemanticChannelNames())
	{
		const std::string& hwChannel = channelMapping_[semantic];
		if (hwChannel != "None")
		{
			if (usedChannels.find(hwChannel) != usedChannels.end())
			{
				LogMessage("Duplicate channel mapping: " + hwChannel +
					" is assigned to multiple semantic channels", false);
				return ERR_DUPLICATE_CHANNEL_MAPPING;
			}
			usedChannels.insert(hwChannel);
		}
	}
	return DEVICE_OK;
}

int iSIMWaveforms::ValidateModInConfiguration()
{
	bool hasModIn = false;

	for (const auto& modIn : GetModInChannelNames())
	{
		if (channelMapping_[modIn] != "None")
		{
			hasModIn = true;
			break;
		}
	}

	if (!hasModIn)
	{
		LogMessage("No MOD IN channel configured", false);
		return ERR_NO_MOD_IN_CONFIGURED;
	}

	return DEVICE_OK;
}

std::vector<std::string> iSIMWaveforms::GetEnabledModInChannels() const
{
	std::vector<std::string> result;
	for (const auto& modIn : GetModInChannelNames())
	{
		// Check if channel is configured (mapped to hardware) and enabled
		auto mappingIt = channelMapping_.find(modIn);
		auto enabledIt = modInEnabled_.find(modIn);
		if (mappingIt != channelMapping_.end() && mappingIt->second != "None" &&
			enabledIt != modInEnabled_.end() && enabledIt->second)
		{
			result.push_back(modIn);
		}
	}
	return result;
}

int iSIMWaveforms::GetNumEnabledModInChannels() const
{
	return static_cast<int>(GetEnabledModInChannels().size());
}

int iSIMWaveforms::ValidateWaveformParameters() const
{
	// Frame interval must be greater than readout time
	if (frameIntervalMs_ <= readoutTimeMs_)
	{
		return ERR_FRAME_INTERVAL_TOO_SHORT;
	}

	// Calculate derived voltage parameters to check galvo range
	double exposureTimeMs = frameIntervalMs_ - readoutTimeMs_;
	double parkingTimeMs = readoutTimeMs_ * parkingFraction_;
	double rampTimeMs = frameIntervalMs_ - parkingTimeMs;
	double waveformPpV = exposurePpV_ * (rampTimeMs / exposureTimeMs);
	double waveformAmplitudeV = waveformPpV / 2.0;
	double waveformHighV = galvoOffsetV_ + waveformAmplitudeV;
	double waveformLowV = galvoOffsetV_ - waveformAmplitudeV;

	// Check galvo voltage is within configured limits
	double galvoMinV = minVoltage_.at(PROP_GALVO_CHANNEL);
	double galvoMaxV = maxVoltage_.at(PROP_GALVO_CHANNEL);
	if (waveformHighV > galvoMaxV || waveformLowV < galvoMinV)
	{
		return ERR_GALVO_VOLTAGE_OUT_OF_RANGE;
	}

	return DEVICE_OK;
}

void iSIMWaveforms::ComputeWaveformParameters(WaveformParams& params) const
{
	// Derived timing (milliseconds)
	params.waveformIntervalMs = 2.0 * frameIntervalMs_;
	params.exposureTimeMs = frameIntervalMs_ - readoutTimeMs_;
	params.parkingTimeMs = readoutTimeMs_ * parkingFraction_;
	params.rampTimeMs = frameIntervalMs_ - params.parkingTimeMs;
	params.waveformOffsetMs = params.parkingTimeMs + (readoutTimeMs_ - params.parkingTimeMs) / 2.0;

	// Sample counts - ensure even number of waveform samples
	size_t rawWaveformSamples = static_cast<size_t>(
		std::round((params.waveformIntervalMs / 1000.0) * samplingRateHz_));
	params.numWaveformSamples = 2 * (rawWaveformSamples / 2);  // Make even
	params.numCounterSamples = params.numWaveformSamples / 2;
	params.numReadoutSamples = static_cast<size_t>(
		std::round((readoutTimeMs_ / 1000.0) * samplingRateHz_));

	// Derived voltages
	params.waveformPpV = exposurePpV_ * (params.rampTimeMs / params.exposureTimeMs);
	params.waveformAmplitudeV = params.waveformPpV / 2.0;
	params.waveformHighV = galvoOffsetV_ + params.waveformAmplitudeV;
	params.waveformLowV = galvoOffsetV_ - params.waveformAmplitudeV;

	// Sample indices
	params.parkingTimeSamples = static_cast<size_t>(
		std::round((params.parkingTimeMs / 1000.0) * samplingRateHz_));
	params.rampTimeSamples = params.numCounterSamples - params.parkingTimeSamples;
	params.waveformOffsetSamples = static_cast<size_t>(
		std::round((params.waveformOffsetMs / 1000.0) * samplingRateHz_));
}

std::vector<double> iSIMWaveforms::ConstructCameraWaveform(const WaveformParams& params) const
{
	std::vector<double> waveform(params.numWaveformSamples, 0.0);

	// Camera trigger: 5% duty cycle pulse at the start of each frame
	size_t pulseWidth = params.numWaveformSamples / 20;  // 5% of waveform period

	// First pulse at t=0
	for (size_t i = 0; i < pulseWidth && i < params.numWaveformSamples; ++i)
	{
		waveform[i] = cameraPulseVoltage_;
	}

	// Second pulse at t=frameInterval (start of second frame)
	size_t secondPulseStart = params.numCounterSamples;
	for (size_t i = 0; i < pulseWidth && (secondPulseStart + i) < params.numWaveformSamples; ++i)
	{
		waveform[secondPulseStart + i] = cameraPulseVoltage_;
	}

	return waveform;
}

std::vector<double> iSIMWaveforms::ConstructGalvoWaveform(const WaveformParams& params) const
{
	std::vector<double> waveform(params.numWaveformSamples, 0.0);

	// First frame: ramp up then park at high
	// Ramp from low to high
	for (size_t i = 0; i < params.rampTimeSamples; ++i)
	{
		double t = static_cast<double>(i) / static_cast<double>(params.rampTimeSamples);
		waveform[i] = params.waveformLowV + t * params.waveformPpV;
	}
	// Park at high
	for (size_t i = params.rampTimeSamples; i < params.numCounterSamples; ++i)
	{
		waveform[i] = params.waveformHighV;
	}

	// Second frame: ramp down then park at low
	size_t frameOffset = params.numCounterSamples;
	// Ramp from high to low
	for (size_t i = 0; i < params.rampTimeSamples; ++i)
	{
		double t = static_cast<double>(i) / static_cast<double>(params.rampTimeSamples);
		waveform[frameOffset + i] = params.waveformHighV - t * params.waveformPpV;
	}
	// Park at low
	for (size_t i = params.rampTimeSamples; i < params.numCounterSamples; ++i)
	{
		waveform[frameOffset + i] = params.waveformLowV;
	}

	// Apply waveform offset by rolling the array
	if (params.waveformOffsetSamples > 0 && params.waveformOffsetSamples < params.numWaveformSamples)
	{
		std::vector<double> shifted(params.numWaveformSamples);
		for (size_t i = 0; i < params.numWaveformSamples; ++i)
		{
			size_t newIdx = (i + params.waveformOffsetSamples) % params.numWaveformSamples;
			shifted[newIdx] = waveform[i];
		}
		waveform = std::move(shifted);
	}

	return waveform;
}

std::vector<double> iSIMWaveforms::ConstructBlankingWaveform(const WaveformParams& params) const
{
	std::vector<double> waveform(params.numWaveformSamples, 0.0);

	// Blanking is high during exposure period (after readout) of each frame
	// Frame 1: from numReadoutSamples to numCounterSamples
	for (size_t i = params.numReadoutSamples; i < params.numCounterSamples; ++i)
	{
		waveform[i] = aotfBlankingVoltage_;
	}

	// Frame 2: from numCounterSamples + numReadoutSamples to end
	size_t frame2Start = params.numCounterSamples + params.numReadoutSamples;
	for (size_t i = frame2Start; i < params.numWaveformSamples; ++i)
	{
		waveform[i] = aotfBlankingVoltage_;
	}

	return waveform;
}

std::vector<double> iSIMWaveforms::ConstructModInWaveform(
	const std::string& semanticChannel,
	const WaveformParams& params) const
{
	std::vector<double> waveform(params.numWaveformSamples, 0.0);

	// Get voltage for this MOD IN channel
	auto voltageIt = modInVoltage_.find(semanticChannel);
	if (voltageIt == modInVoltage_.end())
	{
		return waveform;  // Return zeros if channel not found
	}
	double voltage = voltageIt->second;

	// Normal mode: all enabled channels output simultaneously on every frame
	// Output during exposure period (after readout) of each frame

	// Frame 1: from numReadoutSamples to numCounterSamples
	for (size_t i = params.numReadoutSamples; i < params.numCounterSamples; ++i)
	{
		waveform[i] = voltage;
	}

	// Frame 2: from numCounterSamples + numReadoutSamples to end
	size_t frame2Start = params.numCounterSamples + params.numReadoutSamples;
	for (size_t i = frame2Start; i < params.numWaveformSamples; ++i)
	{
		waveform[i] = voltage;
	}

	return waveform;
}

void iSIMWaveforms::ConfigureDAQChannels()
{
	// Clear any existing configuration
	daq_->clearTasks();

	// Add channels in order: Camera, Galvo, Blanking, then enabled MOD INs

	// Camera channel
	std::string cameraHw = channelMapping_.at(PROP_CAMERA_CHANNEL);
	double cameraMinV = minVoltage_.at(PROP_CAMERA_CHANNEL);
	double cameraMaxV = maxVoltage_.at(PROP_CAMERA_CHANNEL);
	daq_->addAnalogOutputChannel(cameraHw, cameraMinV, cameraMaxV);

	// Galvo channel
	std::string galvoHw = channelMapping_.at(PROP_GALVO_CHANNEL);
	double galvoMinV = minVoltage_.at(PROP_GALVO_CHANNEL);
	double galvoMaxV = maxVoltage_.at(PROP_GALVO_CHANNEL);
	daq_->addAnalogOutputChannel(galvoHw, galvoMinV, galvoMaxV);

	// Blanking channel
	std::string blankingHw = channelMapping_.at(PROP_AOTF_BLANKING_CHANNEL);
	double blankingMinV = minVoltage_.at(PROP_AOTF_BLANKING_CHANNEL);
	double blankingMaxV = maxVoltage_.at(PROP_AOTF_BLANKING_CHANNEL);
	daq_->addAnalogOutputChannel(blankingHw, blankingMinV, blankingMaxV);

	// Enabled MOD IN channels (in order: 1, 2, 3, 4)
	for (const auto& modInChannel : GetEnabledModInChannels())
	{
		std::string modInHw = channelMapping_.at(modInChannel);
		double modInMinV = minVoltage_.at(modInChannel);
		double modInMaxV = maxVoltage_.at(modInChannel);
		daq_->addAnalogOutputChannel(modInHw, modInMinV, modInMaxV);
	}
}

int iSIMWaveforms::CreatePostInitProperties()
{
	int nRet;

	// Sampling Rate property
	CPropertyAction* pActRate = new CPropertyAction(this, &iSIMWaveforms::OnSamplingRate);
	nRet = CreateFloatProperty("Sampling Rate (Hz)", samplingRateHz_, false, pActRate);
	if (nRet != DEVICE_OK)
		return nRet;
	SetPropertyLimits("Sampling Rate (Hz)", 1000.0, 1000000.0);

	// AOTF Blanking Voltage slider (always created since AOTF Blanking is required)
	CPropertyAction* pAct = new CPropertyAction(this, &iSIMWaveforms::OnAOTFBlankingVoltage);
	double minV = minVoltage_[PROP_AOTF_BLANKING_CHANNEL];
	double maxV = maxVoltage_[PROP_AOTF_BLANKING_CHANNEL];
	aotfBlankingVoltage_ = minV;

	nRet = CreateFloatProperty("AOTF Blanking Voltage", aotfBlankingVoltage_, false, pAct);
	if (nRet != DEVICE_OK)
		return nRet;
	SetPropertyLimits("AOTF Blanking Voltage", minV, maxV);

	// MOD IN properties (only for configured channels)
	const std::vector<std::pair<const char*, int>> modInChannels = {
		{PROP_AOTF_MOD_IN_1, 1},
		{PROP_AOTF_MOD_IN_2, 2},
		{PROP_AOTF_MOD_IN_3, 3},
		{PROP_AOTF_MOD_IN_4, 4}
	};

	for (const auto& modIn : modInChannels)
	{
		const char* semanticName = modIn.first;
		int channelNum = modIn.second;

		if (channelMapping_[semanticName] != "None")
		{
			// Initialize state
			modInEnabled_[semanticName] = false;
			modInVoltage_[semanticName] = minVoltage_[semanticName];

			// Enabled property
			std::string enabledPropName = "AOTF MOD IN " + std::to_string(channelNum) + " Enabled";
			CPropertyAction* pActEnabled = new CPropertyAction(this, &iSIMWaveforms::OnModInEnabled);
			nRet = CreateStringProperty(enabledPropName.c_str(), "No", false, pActEnabled);
			if (nRet != DEVICE_OK)
				return nRet;
			AddAllowedValue(enabledPropName.c_str(), "No");
			AddAllowedValue(enabledPropName.c_str(), "Yes");

			// Voltage property
			std::string voltagePropName = "AOTF MOD IN " + std::to_string(channelNum) + " Voltage";
			CPropertyAction* pActVoltage = new CPropertyAction(this, &iSIMWaveforms::OnModInVoltage);
			double modMinV = minVoltage_[semanticName];
			double modMaxV = maxVoltage_[semanticName];

			nRet = CreateFloatProperty(voltagePropName.c_str(), modMinV, false, pActVoltage);
			if (nRet != DEVICE_OK)
				return nRet;
			SetPropertyLimits(voltagePropName.c_str(), modMinV, modMaxV);
		}
	}

	// Waveform timing properties
	CPropertyAction* pActParking = new CPropertyAction(this, &iSIMWaveforms::OnParkingFraction);
	nRet = CreateFloatProperty("Parking Fraction", parkingFraction_, false, pActParking);
	if (nRet != DEVICE_OK)
		return nRet;
	SetPropertyLimits("Parking Fraction", 0.0, 1.0);

	CPropertyAction* pActExposureV = new CPropertyAction(this, &iSIMWaveforms::OnExposureVoltage);
	nRet = CreateFloatProperty("Exposure Voltage (Vpp)", exposurePpV_, false, pActExposureV);
	if (nRet != DEVICE_OK)
		return nRet;
	// Limits based on galvo voltage range
	double galvoRange = maxVoltage_[PROP_GALVO_CHANNEL] - minVoltage_[PROP_GALVO_CHANNEL];
	SetPropertyLimits("Exposure Voltage (Vpp)", 0.0, galvoRange);

	CPropertyAction* pActGalvoOffset = new CPropertyAction(this, &iSIMWaveforms::OnGalvoOffset);
	nRet = CreateFloatProperty("Galvo Offset (V)", galvoOffsetV_, false, pActGalvoOffset);
	if (nRet != DEVICE_OK)
		return nRet;
	SetPropertyLimits("Galvo Offset (V)", minVoltage_[PROP_GALVO_CHANNEL], maxVoltage_[PROP_GALVO_CHANNEL]);

	CPropertyAction* pActTrigger = new CPropertyAction(this, &iSIMWaveforms::OnTriggerSource);
	nRet = CreateStringProperty("Trigger Source", triggerSource_.c_str(), false, pActTrigger);
	if (nRet != DEVICE_OK)
		return nRet;

	// Binning property (required by MMCore for camera devices)
	CPropertyAction* pActBinning = new CPropertyAction(this, &iSIMWaveforms::OnBinning);
	nRet = CreateIntegerProperty(MM::g_Keyword_Binning, 1, false, pActBinning);
	if (nRet != DEVICE_OK)
		return nRet;

	return DEVICE_OK;
}

/**
* Obtains device name.
* Required by the MM::Device API.
*/
void iSIMWaveforms::GetName(char* name) const
{
   // We just return the name we use for referring to this
   // device adapter.
   CDeviceUtils::CopyLimitedString(name, g_DeviceName);
}

/**
* Intializes the hardware.
* Typically we access and initialize hardware at this point.
* Device properties are typically created here as well.
* Required by the MM::Device API.
*/
int iSIMWaveforms::Initialize()
{
	if (initialized_)
		return DEVICE_OK;

	// Set read-only properties
	int nRet = CreateStringProperty(MM::g_Keyword_Name, g_DeviceName, true);
	if (DEVICE_OK != nRet)
		return nRet;

	nRet = CreateStringProperty(
		MM::g_Keyword_Description,
		g_DeviceDescription,
		true
	);
	if (DEVICE_OK != nRet)
		return nRet;

	// Enumerate loaded camera devices (excluding self)
	availableCameras_.clear();
	availableCameras_.push_back(g_Undefined);
	char deviceName[MM::MaxStrLength];
	unsigned int deviceIterator = 0;
	for (;;)
	{
		GetLoadedDeviceOfType(MM::CameraDevice, deviceName, deviceIterator++);
		if (strlen(deviceName) > 0)
		{
			MM::Device* camera = GetDevice(deviceName);
			if (camera && (this != camera))
				availableCameras_.push_back(std::string(deviceName));
		}
		else
			break;
	}

	// Create Physical Camera property
	CPropertyAction* pActCamera = new CPropertyAction(this, &iSIMWaveforms::OnPhysicalCamera);
	nRet = CreateStringProperty(g_PhysicalCamera, g_Undefined, false, pActCamera);
	if (nRet != DEVICE_OK)
		return nRet;
	for (const auto& cam : availableCameras_)
		AddAllowedValue(g_PhysicalCamera, cam.c_str());

	// Refresh available AO channels for the selected device
	availableChannels_ = daq_->getAnalogOutputChannels(deviceName_);

	// Validate channel configuration
	nRet = ValidateMinimumChannels();
	if (nRet != DEVICE_OK)
		return nRet;

	nRet = ValidateRequiredChannels();
	if (nRet != DEVICE_OK)
		return nRet;

	nRet = ValidateChannelMappings();
	if (nRet != DEVICE_OK)
		return nRet;

	nRet = ValidateModInConfiguration();
	if (nRet != DEVICE_OK)
		return nRet;

	// Create post-init voltage control properties
	nRet = CreatePostInitProperties();
	if (nRet != DEVICE_OK)
		return nRet;

	// Synchronize all properties
	nRet = UpdateStatus();
	if (nRet != DEVICE_OK)
		return nRet;

	initialized_ = true;
	return DEVICE_OK;
}

/**
* Shuts down (unloads) the device.
* Ideally this method will completely unload the device and release all resources.
* Shutdown() may be called multiple times in a row.
* Required by the MM::Device API.
*/
int iSIMWaveforms::Shutdown()
{
   daq_.reset();
   initialized_ = false;
   return DEVICE_OK;
}

/////////////////////////////////////////////
// Property Generators
/////////////////////////////////////////////

/////////////////////////////////////////////
// Action handlers
/////////////////////////////////////////////

int iSIMWaveforms::OnDevice(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(deviceName_.c_str());
	}
	else if (eAct == MM::AfterSet)
	{
		std::string newDevice;
		pProp->Get(newDevice);

		if (newDevice != deviceName_)
		{
			deviceName_ = newDevice;

			// Rediscover channels for new device
			availableChannels_ = daq_->getAnalogOutputChannels(deviceName_);

			// Update allowed values for all semantic channel properties
			UpdateChannelAllowedValues();
		}
	}
	return DEVICE_OK;
}

int iSIMWaveforms::OnPhysicalCamera(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(physicalCameraName_.c_str());
	}
	else if (eAct == MM::AfterSet)
	{
		std::string cameraName;
		pProp->Get(cameraName);

		if (cameraName != physicalCameraName_)
		{
			physicalCameraName_ = cameraName;

			// Sync timing parameters from the new camera
			if (physicalCameraName_ != g_Undefined)
			{
				MM::Camera* camera = GetPhysicalCamera();
				if (camera)
				{
					// Sync frame interval from camera's current exposure
					frameIntervalMs_ = camera->GetExposure();
					QueryReadoutTime();

					// TODO: How to handle an invalid exposure time that causes frame interval < readout time?
					BuildAndWriteWaveforms();
				}
			}
		}
	}
	return DEVICE_OK;
}

MM::Camera* iSIMWaveforms::GetPhysicalCamera() const
{
	if (physicalCameraName_.empty() || physicalCameraName_ == g_Undefined)
		return nullptr;

	return static_cast<MM::Camera*>(GetDevice(physicalCameraName_.c_str()));
}

int iSIMWaveforms::OnChannelMapping(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	std::string propName = pProp->GetName();

	if (eAct == MM::BeforeGet)
	{
		pProp->Set(channelMapping_[propName].c_str());
	}
	else if (eAct == MM::AfterSet)
	{
		std::string value;
		pProp->Get(value);
		channelMapping_[propName] = value;
	}
	return DEVICE_OK;
}

int iSIMWaveforms::OnMinVoltage(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	// Extract semantic channel name from property name (remove " Min Voltage" suffix)
	std::string propName = pProp->GetName();
	std::string semanticChannel = propName.substr(0, propName.length() - 12); // " Min Voltage" = 12 chars

	if (eAct == MM::BeforeGet)
	{
		pProp->Set(minVoltage_[semanticChannel]);
	}
	else if (eAct == MM::AfterSet)
	{
		double value;
		pProp->Get(value);
		minVoltage_[semanticChannel] = value;
	}
	return DEVICE_OK;
}

int iSIMWaveforms::OnMaxVoltage(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	// Extract semantic channel name from property name (remove " Max Voltage" suffix)
	std::string propName = pProp->GetName();
	std::string semanticChannel = propName.substr(0, propName.length() - 12); // " Max Voltage" = 12 chars

	if (eAct == MM::BeforeGet)
	{
		pProp->Set(maxVoltage_[semanticChannel]);
	}
	else if (eAct == MM::AfterSet)
	{
		double value;
		pProp->Get(value);
		maxVoltage_[semanticChannel] = value;
	}
	return DEVICE_OK;
}

int iSIMWaveforms::OnAOTFBlankingVoltage(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(aotfBlankingVoltage_);
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(aotfBlankingVoltage_);
		// Rebuild waveforms if initialized and camera is selected
		if (initialized_ && GetPhysicalCamera())
		{
			bool wasRunning = waveformRunning_;
			if (wasRunning)
				StopWaveformOutput();
			BuildAndWriteWaveforms();
			if (wasRunning)
				StartWaveformOutput();
		}
	}
	return DEVICE_OK;
}

int iSIMWaveforms::OnModInEnabled(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	// Extract channel number from property name (e.g., "AOTF MOD IN 1 Enabled")
	std::string propName = pProp->GetName();
	char channelChar = propName[12]; // "AOTF MOD IN X" - X is at position 12
	int channelNum = channelChar - '0';

	const char* semanticKey = nullptr;
	switch (channelNum)
	{
	case 1: semanticKey = PROP_AOTF_MOD_IN_1; break;
	case 2: semanticKey = PROP_AOTF_MOD_IN_2; break;
	case 3: semanticKey = PROP_AOTF_MOD_IN_3; break;
	case 4: semanticKey = PROP_AOTF_MOD_IN_4; break;
	default: return DEVICE_ERR;
	}

	if (eAct == MM::BeforeGet)
	{
		pProp->Set(modInEnabled_[semanticKey] ? "Yes" : "No");
	}
	else if (eAct == MM::AfterSet)
	{
		std::string value;
		pProp->Get(value);
		modInEnabled_[semanticKey] = (value == "Yes");
		// Rebuild waveforms if initialized and camera is selected
		if (initialized_ && GetPhysicalCamera())
		{
			bool wasRunning = waveformRunning_;
			if (wasRunning)
				StopWaveformOutput();
			BuildAndWriteWaveforms();
			if (wasRunning)
				StartWaveformOutput();
		}
	}
	return DEVICE_OK;
}

int iSIMWaveforms::OnModInVoltage(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	// Extract channel number from property name (e.g., "AOTF MOD IN 1 Voltage")
	std::string propName = pProp->GetName();
	char channelChar = propName[12]; // "AOTF MOD IN X" - X is at position 12
	int channelNum = channelChar - '0';

	const char* semanticKey = nullptr;
	switch (channelNum)
	{
	case 1: semanticKey = PROP_AOTF_MOD_IN_1; break;
	case 2: semanticKey = PROP_AOTF_MOD_IN_2; break;
	case 3: semanticKey = PROP_AOTF_MOD_IN_3; break;
	case 4: semanticKey = PROP_AOTF_MOD_IN_4; break;
	default: return DEVICE_ERR;
	}

	if (eAct == MM::BeforeGet)
	{
		pProp->Set(modInVoltage_[semanticKey]);
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(modInVoltage_[semanticKey]);
		// Rebuild waveforms if initialized and camera is selected
		if (initialized_ && GetPhysicalCamera())
		{
			bool wasRunning = waveformRunning_;
			if (wasRunning)
				StopWaveformOutput();
			BuildAndWriteWaveforms();
			if (wasRunning)
				StartWaveformOutput();
		}
	}
	return DEVICE_OK;
}

int iSIMWaveforms::OnSamplingRate(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(samplingRateHz_);
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(samplingRateHz_);
		// Rebuild waveforms if initialized and camera is selected
		if (initialized_ && GetPhysicalCamera())
		{
			bool wasRunning = waveformRunning_;
			if (wasRunning)
				StopWaveformOutput();
			BuildAndWriteWaveforms();
			if (wasRunning)
				StartWaveformOutput();
		}
	}
	return DEVICE_OK;
}

int iSIMWaveforms::OnParkingFraction(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(parkingFraction_);
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(parkingFraction_);
		// Rebuild waveforms if initialized and camera is selected
		if (initialized_ && GetPhysicalCamera())
		{
			bool wasRunning = waveformRunning_;
			if (wasRunning)
				StopWaveformOutput();
			BuildAndWriteWaveforms();
			if (wasRunning)
				StartWaveformOutput();
		}
	}
	return DEVICE_OK;
}

int iSIMWaveforms::OnExposureVoltage(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(exposurePpV_);
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(exposurePpV_);
		// Rebuild waveforms if initialized and camera is selected
		if (initialized_ && GetPhysicalCamera())
		{
			bool wasRunning = waveformRunning_;
			if (wasRunning)
				StopWaveformOutput();
			BuildAndWriteWaveforms();
			if (wasRunning)
				StartWaveformOutput();
		}
	}
	return DEVICE_OK;
}

int iSIMWaveforms::OnGalvoOffset(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(galvoOffsetV_);
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(galvoOffsetV_);
		// Rebuild waveforms if initialized and camera is selected
		if (initialized_ && GetPhysicalCamera())
		{
			bool wasRunning = waveformRunning_;
			if (wasRunning)
				StopWaveformOutput();
			BuildAndWriteWaveforms();
			if (wasRunning)
				StartWaveformOutput();
		}
	}
	return DEVICE_OK;
}

int iSIMWaveforms::OnTriggerSource(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(triggerSource_.c_str());
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(triggerSource_);
	}
	return DEVICE_OK;
}

int iSIMWaveforms::OnCounterChannel(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(counterChannel_.c_str());
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(counterChannel_);
	}
	return DEVICE_OK;
}

int iSIMWaveforms::OnClockSource(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(clockSource_.c_str());
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(clockSource_);
	}
	return DEVICE_OK;
}

int iSIMWaveforms::OnBinning(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set((long)GetBinning());
	}
	else if (eAct == MM::AfterSet)
	{
		long binning;
		pProp->Get(binning);
		int ret = SetBinning((int)binning);
		if (ret != DEVICE_OK)
			return ret;
	}
	return DEVICE_OK;
}

/////////////////////////////////////////////
// Camera wrapper helpers
/////////////////////////////////////////////

int iSIMWaveforms::QueryReadoutTime()
{
	MM::Camera* camera = GetPhysicalCamera();
	if (!camera)
		return ERR_NO_PHYSICAL_CAMERA;

	// Try to get Timing-ReadoutTimeNs property from camera
	char value[MM::MaxStrLength];
	int ret = camera->GetProperty("Timing-ReadoutTimeNs", value);
	if (ret == DEVICE_OK)
	{
		double readoutTimeNs = atof(value);
		readoutTimeMs_ = readoutTimeNs / 1e6;  // Convert ns to ms
	}
	else
	{
		// Log warning - using default readout time
		LogMessage("Warning: Could not read 'Timing-ReadoutTimeNs' from camera. "
		           "Using default readout time: " + std::to_string(readoutTimeMs_) + " ms", false);
	}

	return DEVICE_OK;
}

int iSIMWaveforms::BuildAndWriteWaveforms()
{
	// Validate parameters
	int ret = ValidateWaveformParameters();
	if (ret != DEVICE_OK)
		return ret;

	// Compute waveform parameters
	WaveformParams params;
	ComputeWaveformParameters(params);

	// Construct individual waveforms
	auto cameraWave = ConstructCameraWaveform(params);
	auto galvoWave = ConstructGalvoWaveform(params);
	auto blankingWave = ConstructBlankingWaveform(params);

	// Build row-major interleaved data
	size_t numChannels = 3 + GetNumEnabledModInChannels();
	size_t samplesPerChannel = params.numWaveformSamples;
	std::vector<double> data(numChannels * samplesPerChannel);

	// Copy waveforms to data array (row-major order)
	std::copy(cameraWave.begin(), cameraWave.end(), data.begin());
	std::copy(galvoWave.begin(), galvoWave.end(), data.begin() + samplesPerChannel);
	std::copy(blankingWave.begin(), blankingWave.end(), data.begin() + 2 * samplesPerChannel);

	size_t channelIdx = 3;
	for (const auto& modIn : GetEnabledModInChannels())
	{
		auto modInWave = ConstructModInWaveform(modIn, params);
		std::copy(modInWave.begin(), modInWave.end(), data.begin() + channelIdx * samplesPerChannel);
		++channelIdx;
	}

	// Configure DAQ and write waveforms
	try
	{
		ConfigureDAQChannels();
		daq_->configureTiming(samplingRateHz_, samplesPerChannel, triggerSource_,
		                      counterChannel_, clockSource_);
		daq_->writeAnalogOutput(data, numChannels, samplesPerChannel);
	}
	catch (const std::runtime_error& e)
	{
		LogMessage(std::string("DAQ error: ") + e.what(), false);
		return ERR_WAVEFORM_GENERATION_FAILED;
	}

	return DEVICE_OK;
}

int iSIMWaveforms::StartWaveformOutput()
{
	try
	{
		daq_->start();
	}
	catch (const std::runtime_error& e)
	{
		LogMessage(std::string("DAQ start error: ") + e.what(), false);
		return ERR_WAVEFORM_GENERATION_FAILED;
	}
	waveformRunning_ = true;
	return DEVICE_OK;
}

int iSIMWaveforms::StopWaveformOutput()
{
	try
	{
		daq_->stop();
	}
	catch (const std::runtime_error& e)
	{
		LogMessage(std::string("DAQ stop error: ") + e.what(), false);
	}
	waveformRunning_ = false;
	return DEVICE_OK;
}

/////////////////////////////////////////////
// Camera API - Pass-through methods
/////////////////////////////////////////////

const unsigned char* iSIMWaveforms::GetImageBuffer()
{
	MM::Camera* camera = GetPhysicalCamera();
	return camera ? camera->GetImageBuffer() : nullptr;
}

const unsigned char* iSIMWaveforms::GetImageBuffer(unsigned channelNr)
{
	MM::Camera* camera = GetPhysicalCamera();
	return camera ? camera->GetImageBuffer(channelNr) : nullptr;
}

unsigned iSIMWaveforms::GetImageWidth() const
{
	MM::Camera* camera = GetPhysicalCamera();
	return camera ? camera->GetImageWidth() : 0;
}

unsigned iSIMWaveforms::GetImageHeight() const
{
	MM::Camera* camera = GetPhysicalCamera();
	return camera ? camera->GetImageHeight() : 0;
}

unsigned iSIMWaveforms::GetImageBytesPerPixel() const
{
	MM::Camera* camera = GetPhysicalCamera();
	return camera ? camera->GetImageBytesPerPixel() : 0;
}

unsigned iSIMWaveforms::GetBitDepth() const
{
	MM::Camera* camera = GetPhysicalCamera();
	return camera ? camera->GetBitDepth() : 0;
}

long iSIMWaveforms::GetImageBufferSize() const
{
	MM::Camera* camera = GetPhysicalCamera();
	return camera ? camera->GetImageBufferSize() : 0;
}

double iSIMWaveforms::GetExposure() const
{
	MM::Camera* camera = GetPhysicalCamera();
	return camera ? camera->GetExposure() : 0.0;
}

void iSIMWaveforms::SetExposure(double exp)
{
	// Store frame interval (MM exposure = frame interval in rolling shutter mode)
	frameIntervalMs_ = exp;

	// Pass through to physical camera
	MM::Camera* camera = GetPhysicalCamera();
	if (camera)
		camera->SetExposure(exp);

	// Query updated readout time from camera
	QueryReadoutTime();

	// Build waveforms with new timing (always rebuild when exposure changes)
	// If currently running, stop first, then rebuild and restart
	bool wasRunning = waveformRunning_;
	if (wasRunning)
		StopWaveformOutput();

	int ret = BuildAndWriteWaveforms();
	if (ret != DEVICE_OK)
	{
		LogMessage("SetExposure: Failed to build waveforms (error " +
		           std::to_string(ret) + ")", false);
		return;
	}

	if (wasRunning)
		StartWaveformOutput();
}

int iSIMWaveforms::SetROI(unsigned x, unsigned y, unsigned xSize, unsigned ySize)
{
	MM::Camera* camera = GetPhysicalCamera();
	if (!camera)
		return ERR_NO_PHYSICAL_CAMERA;
	return camera->SetROI(x, y, xSize, ySize);
}

int iSIMWaveforms::GetROI(unsigned& x, unsigned& y, unsigned& xSize, unsigned& ySize)
{
	MM::Camera* camera = GetPhysicalCamera();
	if (!camera)
		return ERR_NO_PHYSICAL_CAMERA;
	return camera->GetROI(x, y, xSize, ySize);
}

int iSIMWaveforms::ClearROI()
{
	MM::Camera* camera = GetPhysicalCamera();
	if (!camera)
		return ERR_NO_PHYSICAL_CAMERA;
	return camera->ClearROI();
}

int iSIMWaveforms::GetBinning() const
{
	MM::Camera* camera = GetPhysicalCamera();
	return camera ? camera->GetBinning() : 1;
}

int iSIMWaveforms::SetBinning(int binSize)
{
	MM::Camera* camera = GetPhysicalCamera();
	if (!camera)
		return ERR_NO_PHYSICAL_CAMERA;
	return camera->SetBinning(binSize);
}

int iSIMWaveforms::IsExposureSequenceable(bool& isSequenceable) const
{
	isSequenceable = false;
	return DEVICE_OK;
}

unsigned iSIMWaveforms::GetNumberOfComponents() const
{
	MM::Camera* camera = GetPhysicalCamera();
	return camera ? camera->GetNumberOfComponents() : 1;
}

unsigned iSIMWaveforms::GetNumberOfChannels() const
{
	MM::Camera* camera = GetPhysicalCamera();
	return camera ? camera->GetNumberOfChannels() : 1;
}

int iSIMWaveforms::GetChannelName(unsigned channel, char* name)
{
	MM::Camera* camera = GetPhysicalCamera();
	if (!camera)
	{
		CDeviceUtils::CopyLimitedString(name, "");
		return DEVICE_OK;
	}
	return camera->GetChannelName(channel, name);
}

bool iSIMWaveforms::IsCapturing()
{
	MM::Camera* camera = GetPhysicalCamera();
	return camera ? camera->IsCapturing() : false;
}

/////////////////////////////////////////////
// Camera API - Acquisition methods
/////////////////////////////////////////////

int iSIMWaveforms::SnapImage()
{
	MM::Camera* camera = GetPhysicalCamera();
	if (!camera)
		return ERR_NO_PHYSICAL_CAMERA;

	// Waveforms already built in SetExposure(), just start output
	int ret = StartWaveformOutput();
	if (ret != DEVICE_OK)
		return ret;

	// Call physical camera's SnapImage (blocks until exposure done)
	ret = camera->SnapImage();

	// Stop DAQ waveforms
	StopWaveformOutput();

	return ret;
}

int iSIMWaveforms::PrepareSequenceAcqusition()
{
	MM::Camera* camera = GetPhysicalCamera();
	if (!camera)
		return ERR_NO_PHYSICAL_CAMERA;

	return camera->PrepareSequenceAcqusition();
}

int iSIMWaveforms::StartSequenceAcquisition(double interval)
{
	MM::Camera* camera = GetPhysicalCamera();
	if (!camera)
		return ERR_NO_PHYSICAL_CAMERA;

	// Waveforms already built in SetExposure(), just start output
	int ret = StartWaveformOutput();
	if (ret != DEVICE_OK)
		return ret;

	// Start camera sequence acquisition
	ret = camera->StartSequenceAcquisition(interval);
	if (ret != DEVICE_OK)
	{
		StopWaveformOutput();
		return ret;
	}

	return DEVICE_OK;
}

int iSIMWaveforms::StartSequenceAcquisition(long numImages, double interval_ms, bool stopOnOverflow)
{
	MM::Camera* camera = GetPhysicalCamera();
	if (!camera)
		return ERR_NO_PHYSICAL_CAMERA;

	// Waveforms already built in SetExposure(), just start output
	int ret = StartWaveformOutput();
	if (ret != DEVICE_OK)
		return ret;

	// Start camera sequence acquisition
	ret = camera->StartSequenceAcquisition(numImages, interval_ms, stopOnOverflow);
	if (ret != DEVICE_OK)
	{
		StopWaveformOutput();
		return ret;
	}

	return DEVICE_OK;
}

int iSIMWaveforms::StopSequenceAcquisition()
{
	MM::Camera* camera = GetPhysicalCamera();

	// Stop camera first
	int ret = DEVICE_OK;
	if (camera)
		ret = camera->StopSequenceAcquisition();

	// Then stop waveforms
	StopWaveformOutput();

	return ret;
}
