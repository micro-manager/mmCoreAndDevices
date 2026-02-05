///////////////////////////////////////////////////////////////////////////////
// FILE:          NIDAQWaveforms.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Generates analog waveforms on a NI-DAQ.
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

#include "NIDAQWaveforms.h"
#include "MockDAQAdapter.h"
#include "NIDAQmxAdapter.h"
#include "ModuleInterface.h"

using namespace std;

const char* g_DeviceName = "NIDAQ Waveforms";
const char* g_DeviceDescription = "Generates analog waveforms on a NI-DAQ device";

// Semantic channel property name constants
const char* const NIDAQWaveforms::PROP_GALVO_CHANNEL = "Galvo Waveform Channel";
const char* const NIDAQWaveforms::PROP_CAMERA_CHANNEL = "Camera Trigger Channel";
const char* const NIDAQWaveforms::PROP_AOTF_BLANKING_CHANNEL = "AOTF Blanking Channel";
const char* const NIDAQWaveforms::PROP_AOTF_MOD_IN_1 = "AOTF MOD IN Channel 1";
const char* const NIDAQWaveforms::PROP_AOTF_MOD_IN_2 = "AOTF MOD IN Channel 2";
const char* const NIDAQWaveforms::PROP_AOTF_MOD_IN_3 = "AOTF MOD IN Channel 3";
const char* const NIDAQWaveforms::PROP_AOTF_MOD_IN_4 = "AOTF MOD IN Channel 4";

///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////

/**
 * List all supported hardware devices here
 */
MODULE_API void InitializeModuleData()
{
   RegisterDevice(g_DeviceName, MM::GenericDevice, "NIDAQ Waveforms");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
   if (deviceName == 0)
      return 0;

   // decide which device class to create based on the deviceName parameter
   if (strcmp(deviceName, g_DeviceName) == 0)
   {
      // create the test device
      return new NIDAQWaveforms();
   }

   // ...supplied name not recognized
   return 0;
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
   delete pDevice;
}

///////////////////////////////////////////////////////////////////////////////
// NIDAQWaveforms implementation
// ~~~~~~~~~~~~~~~~~~~~~~~

/**
* NIDAQWaveforms constructor.
* Setup default all variables and create device properties required to exist
* before intialization. In this case, no such properties were required. All
* properties will be created in the Initialize() method.
*
* As a general guideline Micro-Manager devices do not access hardware in the
* the constructor. We should do as little as possible in the constructor and
* perform most of the initialization in the Initialize() method.
*/
NIDAQWaveforms::NIDAQWaveforms() :
	initialized_(false),
	deviceName_(""),
	aotfBlankingVoltage_(0.0),
	samplingRateHz_(50000.0)
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

	// Create DAQ adapter for device discovery
	// Toggle between Mock and NIDAQmx by commenting/uncommenting:
	daq_ = std::make_unique<MockDAQAdapter>();
	// daq_ = std::make_unique<NIDAQmxAdapter>();

	// Pre-init property: Device
	std::vector<std::string> devices = daq_->getDeviceNames();
	CPropertyAction* pAct = new CPropertyAction(this, &NIDAQWaveforms::OnDevice);
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
}

/**
* NIDAQWaveforms destructor.
* If this device used as intended within the Micro-Manager system,
* Shutdown() will be always called before the destructor. But in any case
* we need to make sure that all resources are properly released even if
* Shutdown() was not called.
*/
NIDAQWaveforms::~NIDAQWaveforms()
{
   if (initialized_)
      Shutdown();
}

/////////////////////////////////////////////
// Helper methods for channel configuration
/////////////////////////////////////////////

std::vector<const char*> NIDAQWaveforms::GetSemanticChannelNames()
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

std::vector<const char*> NIDAQWaveforms::GetModInChannelNames()
{
	return {
		PROP_AOTF_MOD_IN_1,
		PROP_AOTF_MOD_IN_2,
		PROP_AOTF_MOD_IN_3,
		PROP_AOTF_MOD_IN_4
	};
}

void NIDAQWaveforms::InitializeChannelDefaults()
{
	for (const auto& channel : GetSemanticChannelNames())
	{
		channelMapping_[channel] = "None";
		minVoltage_[channel] = -10.0;
		maxVoltage_[channel] = 10.0;
	}
}

void NIDAQWaveforms::CreateChannelPreInitProperties()
{
	for (const auto& semanticChannel : GetSemanticChannelNames())
	{
		// Channel selection property (pre-init)
		CPropertyAction* pAct = new CPropertyAction(this, &NIDAQWaveforms::OnChannelMapping);
		CreateStringProperty(semanticChannel, "None", false, pAct, true);
		AddAllowedValue(semanticChannel, "None");
		for (const auto& hwChannel : availableChannels_)
			AddAllowedValue(semanticChannel, hwChannel.c_str());

		// Min voltage property (pre-init)
		std::string minVoltProp = std::string(semanticChannel) + " Min Voltage";
		CPropertyAction* pActMin = new CPropertyAction(this, &NIDAQWaveforms::OnMinVoltage);
		CreateFloatProperty(minVoltProp.c_str(), -10.0, false, pActMin, true);

		// Max voltage property (pre-init)
		std::string maxVoltProp = std::string(semanticChannel) + " Max Voltage";
		CPropertyAction* pActMax = new CPropertyAction(this, &NIDAQWaveforms::OnMaxVoltage);
		CreateFloatProperty(maxVoltProp.c_str(), 10.0, false, pActMax, true);
	}
}

void NIDAQWaveforms::UpdateChannelAllowedValues()
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

int NIDAQWaveforms::ValidateMinimumChannels()
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

int NIDAQWaveforms::ValidateRequiredChannels()
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

int NIDAQWaveforms::ValidateChannelMappings()
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

int NIDAQWaveforms::ValidateModInConfiguration()
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

int NIDAQWaveforms::CreatePostInitProperties()
{
	int nRet;

	// Sampling Rate property
	CPropertyAction* pActRate = new CPropertyAction(this, &NIDAQWaveforms::OnSamplingRate);
	nRet = CreateFloatProperty("Sampling Rate (Hz)", samplingRateHz_, false, pActRate);
	if (nRet != DEVICE_OK)
		return nRet;
	SetPropertyLimits("Sampling Rate (Hz)", 1000.0, 1000000.0);

	// AOTF Blanking Voltage slider (always created since AOTF Blanking is required)
	CPropertyAction* pAct = new CPropertyAction(this, &NIDAQWaveforms::OnAOTFBlankingVoltage);
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
			CPropertyAction* pActEnabled = new CPropertyAction(this, &NIDAQWaveforms::OnModInEnabled);
			nRet = CreateStringProperty(enabledPropName.c_str(), "No", false, pActEnabled);
			if (nRet != DEVICE_OK)
				return nRet;
			AddAllowedValue(enabledPropName.c_str(), "No");
			AddAllowedValue(enabledPropName.c_str(), "Yes");

			// Voltage property
			std::string voltagePropName = "AOTF MOD IN " + std::to_string(channelNum) + " Voltage";
			CPropertyAction* pActVoltage = new CPropertyAction(this, &NIDAQWaveforms::OnModInVoltage);
			double modMinV = minVoltage_[semanticName];
			double modMaxV = maxVoltage_[semanticName];

			nRet = CreateFloatProperty(voltagePropName.c_str(), modMinV, false, pActVoltage);
			if (nRet != DEVICE_OK)
				return nRet;
			SetPropertyLimits(voltagePropName.c_str(), modMinV, modMaxV);
		}
	}

	return DEVICE_OK;
}

/**
* Obtains device name.
* Required by the MM::Device API.
*/
void NIDAQWaveforms::GetName(char* name) const
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
int NIDAQWaveforms::Initialize()
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
int NIDAQWaveforms::Shutdown()
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

int NIDAQWaveforms::OnDevice(MM::PropertyBase* pProp, MM::ActionType eAct)
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

int NIDAQWaveforms::OnChannelMapping(MM::PropertyBase* pProp, MM::ActionType eAct)
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

int NIDAQWaveforms::OnMinVoltage(MM::PropertyBase* pProp, MM::ActionType eAct)
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

int NIDAQWaveforms::OnMaxVoltage(MM::PropertyBase* pProp, MM::ActionType eAct)
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

int NIDAQWaveforms::OnAOTFBlankingVoltage(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(aotfBlankingVoltage_);
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(aotfBlankingVoltage_);
		// TODO: Apply voltage to hardware via daq_
	}
	return DEVICE_OK;
}

int NIDAQWaveforms::OnModInEnabled(MM::PropertyBase* pProp, MM::ActionType eAct)
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
		// TODO: Apply enable state to hardware
	}
	return DEVICE_OK;
}

int NIDAQWaveforms::OnModInVoltage(MM::PropertyBase* pProp, MM::ActionType eAct)
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
		// TODO: Apply voltage to hardware via daq_
	}
	return DEVICE_OK;
}

int NIDAQWaveforms::OnSamplingRate(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(samplingRateHz_);
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(samplingRateHz_);
	}
	return DEVICE_OK;
}
