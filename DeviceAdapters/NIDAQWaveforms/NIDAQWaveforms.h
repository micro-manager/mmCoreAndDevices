///////////////////////////////////////////////////////////////////////////////
// FILE:          NIDAQWaveforms.h
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

#pragma once

#include "DeviceBase.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

// Error codes
#define ERR_DUPLICATE_CHANNEL_MAPPING    101
#define ERR_NO_MOD_IN_CONFIGURED         102
#define ERR_REQUIRED_CHANNEL_NOT_SET     103

class IDAQDevice;

class NIDAQWaveforms : public CGenericBase<NIDAQWaveforms>
{
public:
	NIDAQWaveforms();
	~NIDAQWaveforms();

	// MMDevice API
	int Initialize();
	int Shutdown();
	void GetName(char* name) const;
	bool Busy() { return false; };

	// Semantic channel property name constants
	static const char* const PROP_GALVO_CHANNEL;
	static const char* const PROP_CAMERA_CHANNEL;
	static const char* const PROP_AOTF_BLANKING_CHANNEL;
	static const char* const PROP_AOTF_MOD_IN_1;
	static const char* const PROP_AOTF_MOD_IN_2;
	static const char* const PROP_AOTF_MOD_IN_3;
	static const char* const PROP_AOTF_MOD_IN_4;

	// Action handlers
	int OnDevice(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnChannelMapping(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnMinVoltage(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnMaxVoltage(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnAOTFBlankingVoltage(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnModInEnabled(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnModInVoltage(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
	// Helper methods
	static std::vector<const char*> GetSemanticChannelNames();
	static std::vector<const char*> GetModInChannelNames();
	void InitializeChannelDefaults();
	void CreateChannelPreInitProperties();
	void UpdateChannelAllowedValues();
	int ValidateRequiredChannels();
	int ValidateChannelMappings();
	int ValidateModInConfiguration();
	int CreatePostInitProperties();

	// State
	bool initialized_;
	std::string deviceName_;
	std::vector<std::string> availableChannels_;
	std::unique_ptr<IDAQDevice> daq_;

	// Channel mappings (semantic name -> hardware channel)
	std::map<std::string, std::string> channelMapping_;

	// Voltage ranges per semantic channel
	std::map<std::string, double> minVoltage_;
	std::map<std::string, double> maxVoltage_;

	// Post-init runtime state
	double aotfBlankingVoltage_;
	std::map<std::string, bool> modInEnabled_;
	std::map<std::string, double> modInVoltage_;
};
