//////////////////////////////////////////////////////////////////////////////
// FILE:          LedArray.h
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
//
//////////////////////////////////////////////////////////////////////////////


#ifndef _ILLUMINATE_H_
#define _ILLUMINATE_H_

#include "MMDevice.h"
#include "DeviceBase.h"
#include <string>
#include <map>

//////////////////////////////////////////////////////////////////////////////
// Error codes
//////////////////////////////////////////////////////////////////////////////
#define ERR_UNKNOWN_POSITION 101
#define ERR_INITIALIZE_FAILED 102
#define ERR_WRITE_FAILED 103
#define ERR_CLOSE_FAILED 104
#define ERR_BOARD_NOT_FOUND 105
#define ERR_PORT_OPEN_FAILED 106
#define ERR_COMMUNICATION 107
#define ERR_NO_PORT_SET 108
#define ERR_VERSION_MISMATCH 109
#define COMMAND_TERMINATOR '\n'
#define SERIAL_DELAY_MS 30
const double MAX_INTERFACE_VERSION = 10.0;
const double MIN_INTERFACE_VERSION = 2.30;

const char* g_Keyword_DeviceAdapterName = "IlluminateLedArray";

const char * g_Keyword_ColorBalanceRed = "ColorBalanceRed";      // Global intensity with a maximum of 255
const char * g_Keyword_ColorBalanceGreen = "ColorBalanceGreen";  // Global intensity with a maximum of 255
const char * g_Keyword_ColorBalanceBlue = "ColorBalanceBlue";    // Global intensity with a maximum of 255
const char * g_Keyword_Brightness = "Brightness";
const char * g_Keyword_NumericalAperture = "NumericalAperture";
const char * g_Keyword_InnerNumericalAperture = "InnerNumericalAperture";
const char * g_Keyword_AnnulusStartNa = "AnnulusStartNa";
const char * g_Keyword_ArrayDistance = "ArrayDistanceFromSample";
const char * g_Keyword_Pattern = "IlluminationPattern";
const char * g_Keyword_AnnulusWidthNa = "AnnulusWidthNa";
const char * g_Keyword_LedList = "ManualLedList";
const char * g_Keyword_Reset = "Reset";

const char * g_Keyword_ModelName = "ModelName";
const char * g_Keyword_MeanWavelengthInfrared = "ColorMeanWavelengthInfraredNm";
const char * g_Keyword_MeanWavelengthRed = "ColorMeanWavelengthRedNm";
const char * g_Keyword_MeanWavelengthGreen = "ColorMeanWavelengthGreenNm";
const char * g_Keyword_MeanWavelengthBlue = "ColorMeanWavelengthBlueNm";
const char * g_Keyword_MeanWavelengthUV = "ColorMeanWavelengthUvNm";
const char * g_Keyword_FwhmWavelengthInfrared = "ColorMeanWavelengthInfraredNm";
const char * g_Keyword_FwhmWavelengthRed = "ColorFwhmWavelengthRedNm";
const char * g_Keyword_FwhmWavelengthGreen = "ColorFwhmWavelengthGreenNm";
const char * g_Keyword_FwhmWavelengthBlue = "ColorFwhmWavelengthBlueNm";
const char * g_Keyword_FwhmWavelengthUV = "ColorFwhmWavelengthUvNm";
const char * g_Keyword_LedCount = "LedCount";
const char * g_Keyword_PartNumber = "PartNumber";
const char * g_Keyword_SerialNumber = "SerialNumber";
const char * g_Keyword_MacAddress = "MacAddress";
const char * g_Keyword_BitDepth = "BitDepth";
const char * g_Keyword_InterfaceVersion = "InterfaceVersion";
const char * g_Keyword_ColorChannelCount = "ColorChannelCount";
const char * g_Keyword_LedPositions = "LedPositionsCartesian";
const char * g_Keyword_GlobalShutter = "GlobalShutter";
const char * g_Keyword_Sequence = "Sequence";
const char * g_Keyword_SequenceDelay = "SequenceDelaySeconds";
const char* g_Keyword_SequenceRunCount = "SequenceRunCount";

// Trigger Settings
const char* g_Keyword_TriggerInputCount = "TriggerInputCount";
const char* g_Keyword_TriggerOutputCount = "TriggerOutputCount";
const char* g_Keyword_TriggerInput0Mode = "TriggerInput0Mode";
const char* g_Keyword_TriggerInput1Mode = "TriggerInput1Mode";
const char* g_Keyword_TriggerOutput0Mode = "TriggerOutput0Mode";
const char* g_Keyword_TriggerOutput1Mode = "TriggerOutput1Mode";
const char* g_Keyword_TriggerOutput0PulseWidth = "TriggerOutput0PulseWidth";
const char* g_Keyword_TriggerOutput1PulseWidth = "TriggerOutput1PulseWidth";
const char* g_Keyword_TriggerOutput0StartDelay = "TriggerOutput0StartDelaySeconds";
const char* g_Keyword_TriggerOutput1StartDelay = "TriggerOutput1StartDelaySeconds";
const char* g_Keyword_TriggerInput0Polarity = "TriggerInput0Polarity";
const char* g_Keyword_TriggerInput1Polarity = "TriggerInput1Polarity";
const char* g_Keyword_TriggerOutput0Polarity = "TriggerOutput0Polarity";
const char* g_Keyword_TriggerOutput1Polarity = "TriggerOutput1Polarity";
const char* g_Keyword_TriggerInputTimeout = "TriggerInputTimeoutSeconds";

// Low-Level Serial IO
const char * g_Keyword_Response = "SerialResponse";
const char * g_Keyword_Command = "SerialCommand";

// LED pattern labels
const char * g_Pattern_Brightfield = "Brightfield";
const char * g_Pattern_Darkfield = "Darkfield";
const char * g_Pattern_DpcTop = "DPC Top";
const char * g_Pattern_DpcBottom = "DPC Bottom";
const char * g_Pattern_DpcLeft = "DPC Left";
const char * g_Pattern_DpcRight = "DPC Right";
const char* g_Pattern_Dpc0 = "DPC 000 Deg.";
const char* g_Pattern_Dpc30 = "DPC 030 Deg.";
const char* g_Pattern_Dpc60 = "DPC 060 Deg.";
const char* g_Pattern_Dpc90 = "DPC 090 Deg.";
const char* g_Pattern_Dpc120 = "DPC 120 Deg.";
const char* g_Pattern_Dpc150 = "DPC 150 Deg.";
const char* g_Pattern_Dpc180 = "DPC 180 Deg.";
const char* g_Pattern_Dpc210 = "DPC 210 Deg.";
const char* g_Pattern_Dpc240 = "DPC 240 Deg.";
const char* g_Pattern_Dpc270 = "DPC 270 Deg.";
const char* g_Pattern_Dpc300 = "DPC 300 Deg.";
const char* g_Pattern_Dpc330 = "DPC 330 Deg.";
const char * g_Pattern_ColorDpc = "Color DPC";
const char * g_Pattern_ColorDarkfield = "Color Darkfield";
const char * g_Pattern_ManualLedIndices = "Manual LED Indicies";
const char * g_Pattern_Annulus = "Annulus";
const char * g_Pattern_HalfAnnulusTop = "Half Annulus Top";
const char * g_Pattern_HalfAnnulusBottom = "Half Annulus Bottom";
const char * g_Pattern_HalfAnnulusLeft = "Half Annulus Left";
const char * g_Pattern_HalfAnnulusRight = "Half Annulus Right";
const char * g_Pattern_CenterLed = "Center LED";
const char * g_Pattern_Clear = "Clear";

// Trigger Modes
const char* g_TriggerModeDisabled = "Disabled";
const char* g_TriggerModeEveryPattternChange = "Every Pattern Change";
const char* g_TriggerModeEveryPatternCycle = "Every Pattern Cycle";
const char* g_TriggerModeOnceAtStart = "Start of First Pattern";

// Sequences
const char* g_SequenceModeNotRunning = "";
const char* g_SequenceModeAllLeds = "Individual LEDs (All)";
const char* g_SequenceModeBrightfieldOnly = "Individual LEDs (Brightfield)";
const char* g_SequenceModeDarkfieldOnly = "Individual LEDs (Darkfield)";
const char* g_SequenceModeDpc90 = "DPC Patterns";

// Error code
const int ILLUMINATE_GENERAL_ERROR_CODE = 02142025;

class LedArray : public CShutterBase<LedArray>
{
public:
	LedArray(void);
	virtual ~LedArray(void);
  
	// MMDevice API
	// ------------
	int Initialize();
	int Shutdown();

	bool Busy();
	void GetName(char *) const;

	// Shutter API
	int SetOpen(bool open = true);
	int GetOpen(bool& open);
	int Fire(double) { return DEVICE_UNSUPPORTED_COMMAND; }

	// Trigger Modes
	enum TriggerMode
	{
		Disabled = 0,
		EveryPatternChange = 1,
		EveryPatternCycle = -1,
		OnceAtStart = -2
	};


	// action interface
	// ----------------
	int OnPort(MM::PropertyBase* pPropt, MM::ActionType eAct);
	int OnPattern(MM::PropertyBase* pPropt, MM::ActionType eAct);
	int OnColorBalanceRed( MM::PropertyBase* pPropt, MM::ActionType eAct);
	int OnColorBalanceGreen(MM::PropertyBase* pPropt, MM::ActionType eAct);
	int OnColorBalanceBlue(MM::PropertyBase* pPropt, MM::ActionType eAct);
	int OnNumericalAperture(MM::PropertyBase* pPropt, MM::ActionType eAct);
	int OnInnerNumericalAperture(MM::PropertyBase* pPropt, MM::ActionType eAct);
	int OnDistance(MM::PropertyBase* pPropt, MM::ActionType eAct);
	int OnAnnulusWidth(MM::PropertyBase* pProp, MM::ActionType pAct);
	int OnAnnulusStart(MM::PropertyBase* pProp, MM::ActionType pAct);
	int OnManualLedList(MM::PropertyBase* pPropt, MM::ActionType eAct);
	int OnReset(MM::PropertyBase* pPropt, MM::ActionType eAct);
	int OnCommand(MM::PropertyBase* pPropt, MM::ActionType eAct);
	int OnBrightness(MM::PropertyBase* pPropt, MM::ActionType eAct);
	int OnGlobalShutter(MM::PropertyBase* pPropt, MM::ActionType eAct);
	int OnTriggerInput0Mode(MM::PropertyBase* pPropt, MM::ActionType eAct);
	int OnTriggerInput1Mode(MM::PropertyBase* pPropt, MM::ActionType eAct);
	int OnTriggerInput0Polarity(MM::PropertyBase* pPropt, MM::ActionType eAct);
	int OnTriggerInput1Polarity(MM::PropertyBase* pPropt, MM::ActionType eAct);
	int OnTriggerOutput0Mode(MM::PropertyBase* pPropt, MM::ActionType eAct);
	int OnTriggerOutput1Mode(MM::PropertyBase* pPropt, MM::ActionType eAct);
	int OnTriggerOutput0Polarity(MM::PropertyBase* pPropt, MM::ActionType eAct);
	int OnTriggerOutput1Polarity(MM::PropertyBase* pPropt, MM::ActionType eAct);
	int OnTriggerOutput0PulseWidth(MM::PropertyBase* pPropt, MM::ActionType eAct);
	int OnTriggerOutput1PulseWidth(MM::PropertyBase* pPropt, MM::ActionType eAct);
	int OnTriggerOutput0StartDelay(MM::PropertyBase* pPropt, MM::ActionType eAct);
	int OnTriggerOutput1StartDelay(MM::PropertyBase* pPropt, MM::ActionType eAct);
	int OnTriggerInputTimeout(MM::PropertyBase* pPropt, MM::ActionType eAct);
	int OnSequence(MM::PropertyBase* pPropt, MM::ActionType eAct);
	int OnSequenceRunCount(MM::PropertyBase* pProp, MM::ActionType pAct);
	int OnSequenceDelay(MM::PropertyBase* pProp, MM::ActionType pAct);
	TriggerMode StringToTriggerMode(const std::string&);
	std::string TriggerModeToString(TriggerMode);

	long SecondsToMicroseconds(double);
	double MicrosecondsToSeconds(long);
	long SecondsToMilliseconds(double);
private:


	bool initialized;
	bool port_available = false;
	long is_shutter_open = false;
	int part_number;
	int serial_number;
	int return_code;
	double numerical_aperture;
	double inner_numerical_aperture;
	double annulus_start_na;
	double annulus_width_na;
	double array_distance_z;
	double interface_version;
	long led_count;
	int bit_depth;
	int color_channel_count;
	int trigger_input_count;
	int trigger_output_count;
	TriggerMode trigger_input0_mode;
	TriggerMode trigger_input1_mode;
	TriggerMode trigger_output0_mode;
	TriggerMode trigger_output1_mode;
	double trigger_output0_pulse_width_s;
	double trigger_output1_pulse_width_s;
	double trigger_output0_start_delay_s;
	double trigger_output1_start_delay_s;
	bool trigger_input0_invert_polarity;
	bool trigger_input1_invert_polarity;
	bool trigger_output0_invert_polarity;
	bool trigger_output1_invert_polarity;
	double trigger_input_timeout_s;
	long color_r;
	long color_g;
	long color_b;
	long brightness;
	double** led_positions_cartesian;
	long shutter_open = 1;
	double sequence_delay_s;
	long sequence_run_count;

	// Strings
	std::string device_name_str;
	std::string mac_address_str;
	std::string port_str;
	std::string pattern_str;
	std::string sequence_str;
	std::string led_indices_str;
	std::string serial_command_str;
	std::string serial_response_str;

	// Action functions with LEDs:
	int UpdateColor(long redint, long greenint, long blueint);
	int SetBrightness(long brightness);
	int DrawLedList(const char * led_list_char);
	int SetNumericalAperture(double numa);
	int SetInnerNumericalAperture(double numa);
	int SetArrayDistance(double dist);
	int UpdatePattern();
	int Reset();
	int GetDeviceParameters();
	int SetMachineMode(bool mode);
	int SendCommand(const char * command, bool get_response);
	int SyncState();

};

#endif 
