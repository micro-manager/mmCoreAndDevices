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
#define ERR_INSUFFICIENT_AO_CHANNELS     101
#define ERR_DUPLICATE_CHANNEL_MAPPING    102
#define ERR_NO_MOD_IN_CONFIGURED         103
#define ERR_REQUIRED_CHANNEL_NOT_SET     104
#define ERR_FRAME_INTERVAL_TOO_SHORT     105
#define ERR_GALVO_VOLTAGE_OUT_OF_RANGE   106
#define ERR_NO_MOD_IN_ENABLED            107
#define ERR_WAVEFORM_GENERATION_FAILED   108
#define ERR_NO_PHYSICAL_CAMERA           109
#define ERR_INVALID_CAMERA               110

class IDAQDevice;

// Waveform parameters computed from user settings
struct WaveformParams
{
	// Derived timing (milliseconds)
	double waveformIntervalMs;      // = 2 * frameIntervalMs
	double exposureTimeMs;          // = frameIntervalMs - readoutTimeMs
	double parkingTimeMs;           // = readoutTimeMs * parkingFraction
	double rampTimeMs;              // = frameIntervalMs - parkingTimeMs
	double waveformOffsetMs;        // = parkingTimeMs + (readoutTimeMs - parkingTimeMs) / 2.0

	// Sample counts (normal mode - no interleaving)
	size_t numWaveformSamples;      // Samples in one galvo period (2 frames, must be even)
	size_t numCounterSamples;       // = numWaveformSamples / 2 (samples per frame)
	size_t numReadoutSamples;       // Samples in readout period

	// Derived voltages
	double waveformPpV;             // Total galvo peak-to-peak
	double waveformAmplitudeV;      // = waveformPpV / 2
	double waveformHighV;           // = galvoOffsetV + waveformAmplitudeV
	double waveformLowV;            // = galvoOffsetV - waveformAmplitudeV

	// Sample indices
	size_t parkingTimeSamples;
	size_t rampTimeSamples;
	size_t waveformOffsetSamples;
};

class NIDAQWaveforms : public CCameraBase<NIDAQWaveforms>
{
public:
	NIDAQWaveforms();
	~NIDAQWaveforms();

	// MMDevice API
	int Initialize();
	int Shutdown();
	void GetName(char* name) const;
	bool Busy() { return false; }

	// Camera API - Image acquisition
	int SnapImage();
	const unsigned char* GetImageBuffer();
	const unsigned char* GetImageBuffer(unsigned channelNr);

	// Camera API - Image properties
	unsigned GetImageWidth() const;
	unsigned GetImageHeight() const;
	unsigned GetImageBytesPerPixel() const;
	unsigned GetBitDepth() const;
	long GetImageBufferSize() const;

	// Camera API - Exposure
	double GetExposure() const;
	void SetExposure(double exp);

	// Camera API - ROI
	int SetROI(unsigned x, unsigned y, unsigned xSize, unsigned ySize);
	int GetROI(unsigned& x, unsigned& y, unsigned& xSize, unsigned& ySize);
	int ClearROI();

	// Camera API - Binning
	int GetBinning() const;
	int SetBinning(int binSize);

	// Camera API - Sequence acquisition
	int PrepareSequenceAcqusition();  // Note: MM API typo preserved
	int StartSequenceAcquisition(double interval);
	int StartSequenceAcquisition(long numImages, double interval_ms, bool stopOnOverflow);
	int StopSequenceAcquisition();
	bool IsCapturing();

	// Camera API - Additional methods
	int IsExposureSequenceable(bool& isSequenceable) const;
	unsigned GetNumberOfComponents() const;
	unsigned GetNumberOfChannels() const;
	int GetChannelName(unsigned channel, char* name);

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
	int OnPhysicalCamera(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnChannelMapping(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnMinVoltage(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnMaxVoltage(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnAOTFBlankingVoltage(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnModInEnabled(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnModInVoltage(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnSamplingRate(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnParkingFraction(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnExposureVoltage(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnGalvoOffset(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnTriggerSource(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnCounterChannel(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnClockSource(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnBinning(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
	// Helper methods
	static std::vector<const char*> GetSemanticChannelNames();
	static std::vector<const char*> GetModInChannelNames();
	void InitializeChannelDefaults();
	void CreateChannelPreInitProperties();
	void UpdateChannelAllowedValues();
	int ValidateMinimumChannels();
	int ValidateRequiredChannels();
	int ValidateChannelMappings();
	int ValidateModInConfiguration();
	int CreatePostInitProperties();

	// Waveform construction
	void ComputeWaveformParameters(WaveformParams& params) const;
	std::vector<double> ConstructCameraWaveform(const WaveformParams& params) const;
	std::vector<double> ConstructGalvoWaveform(const WaveformParams& params) const;
	std::vector<double> ConstructBlankingWaveform(const WaveformParams& params) const;
	std::vector<double> ConstructModInWaveform(const std::string& semanticChannel,
	                                            const WaveformParams& params) const;

	// DAQ configuration helpers
	void ConfigureDAQChannels();
	std::vector<std::string> GetEnabledModInChannels() const;
	int GetNumEnabledModInChannels() const;
	int ValidateWaveformParameters() const;

	// Camera wrapper helpers
	MM::Camera* GetPhysicalCamera() const;
	int QueryReadoutTime();
	int BuildAndWriteWaveforms();
	int StartWaveformOutput();
	int StopWaveformOutput();

	// State
	bool initialized_;
	std::string deviceName_;
	std::vector<std::string> availableChannels_;
	std::unique_ptr<IDAQDevice> daq_;

	// Camera wrapper state
	std::string physicalCameraName_;
	std::vector<std::string> availableCameras_;

	// Channel mappings (semantic name -> hardware channel)
	std::map<std::string, std::string> channelMapping_;

	// Voltage ranges per semantic channel
	std::map<std::string, double> minVoltage_;
	std::map<std::string, double> maxVoltage_;

	// Post-init runtime state
	double aotfBlankingVoltage_;
	std::map<std::string, bool> modInEnabled_;
	std::map<std::string, double> modInVoltage_;
	double samplingRateHz_;

	// Waveform timing parameters
	double frameIntervalMs_;        // Set via camera exposure time when in rolling shutter mode
	// TODO: Make the readout time property name of the physical camera settable or provide a way to specify a default value in case the camera does not have the property
	double readoutTimeMs_;          // Read from camera property
	double parkingFraction_;        // Set via property
	double exposurePpV_;            // Set via property
	double galvoOffsetV_;           // Set via property

	// Camera trigger parameters
	double cameraPulseVoltage_;

	// Trigger and counter configuration
	std::string triggerSource_;     // Set via property
	std::string counterChannel_;    // e.g., "Dev1/ctr1"
	std::string clockSource_;       // e.g., "/Dev1/Ctr1InternalOutput"

	// Waveform state
	bool waveformRunning_;
};
