///////////////////////////////////////////////////////////////////////////////
// FILE:          XYStage.cpp
// PROJECT:       Micro-Manager 2.0
// SUBSYSTEM:     DeviceAdapters
//  
//-----------------------------------------------------------------------------
// DESCRIPTION:   SIGMA-KOKI device adapter 2.0
//                
// AUTHOR   :    Hiroki Kibata, Abed Toufik  Release Date :  05/02/2023
//
// COPYRIGHT:     SIGMA KOKI CO.,LTD, Tokyo, 2023
#include "XYStage.h"
#include <string>
#include <iostream>
#include <vector>
#include <sstream>

using namespace std;
// XY_Device Adapter (Registartion from SigmaBase)
const char* g_XYStageDeviceName = "XYStage";
// cont variable to be used  proprietes
const char* g_ChannelX = "Channel_X";
const char* g_ChannelY = "Channel_Y";
const char* g_XYStageSpeed = "Speed (micron/sec)";
const char* g_XStepSize = "StepSize_X (micron)";
const char* g_YStepSize = "StepSize_Y (micron)";
const char* g_FullStepStageX = "Full Step Stage X (micron)";
const char* g_FullStepStageY = "Full Step Stage Y (micron)";
const char* g_DivisionX = "Division Stage X";
const char* g_DivisionY = "Division Stage Y";
const char* g_DivisionInfo = "Division Information";
const char* g_XYStageController = "Name of Controller";
const char* g_FCModel = "Controller Model";
const char* g_FCModelSet = "FC-types";
const char* g_FCResolution = "Controller Resolution (micron)";
const char* g_PulseRateInfo = "Please set the correct PULSE_RATE for each axis in memory switches";

const int MaxSpeedum = 5000;  //  micromter unit
const int MinSpeedum = 10;    //  micromter unit

/// <summary>
/// Constructor
/// </summary>
XYStage::XYStage() :
	SigmaBase(this),
	model_(SHOT2),
	channelX_(1),
	channelY_(2),
	channelA_(3),
	channelB_(4),
	speedXYum_(3000),
	fullStepSizeXum_(2.0),
	fullStepSizeYum_(2.0),
	stepSizeXum_(0.1),
	stepSizeYum_(0.1),
	divisionX_(20),
	divisionY_(20),
	answerTimeoutMs_(500),
	positionXpulse_(0),
	positionYpulse_(0),
	positionApulse_(0),
	positionBpulse_(0),
	isBusyHomeShot4_(false),
	fcModel_("FC-***"),
	slow_pulse (0),
	fast_pulse(0),
	PlsRate1(0),
	PlsRate2(0)
{
	InitializeDefaultErrorMessages();

	// Create pre-initialization properties
	SetErrorText(ERR_XYSTEGE_DEVICE_UNRECOGNIZABLE, "Connected device can not recognized.");
	SetErrorText(ERR_XYSTAGE_STEPSIZE_FAILED, "Failed to resolution setting.");
	SetErrorText(ERR_XYSTAGE_SPEED_FAILED, "Failed to speed setting.");

	// Name
	CreateProperty(MM::g_Keyword_Name, g_XYStageDeviceName, MM::String, true);

	// Description
	CreateProperty(MM::g_Keyword_Description, "SIGMA-KOKI XYStage adapter", MM::String, true);

	// Port
	CPropertyAction* pAct = new CPropertyAction(this, &XYStage::OnPort);
	CreateProperty(MM::g_Keyword_Port, "Undefined", MM::String, false, pAct, true);
}

/// <summary>
/// Destructor
/// </summary>
XYStage::~XYStage()
{
	Shutdown();
}

/// <summary>
/// get name 
/// <param name="Name"></param>
void XYStage::GetName(char* Name) const
{
	CDeviceUtils::CopyLimitedString(Name, g_XYStageDeviceName);
}

/// <summary>
/// Shutdown 
/// <returns></returns>
int XYStage::Shutdown()
{
	if (initialized_)
	{
		initialized_ = false;
	}
	return DEVICE_OK;
}

/// <summary>
/// busy status
/// <returns></returns>
bool XYStage::Busy()
{
	//  Home command on Y channel when using SHOT-304
	if (model_ == SHOT4)
	{
		if (isBusyHomeShot4_)
		{
			int ret = SendCheckRecievedOK("H:" + to_string(channelY_));
			if (ret != DEVICE_OK) return ret;
			isBusyHomeShot4_ = false;
			Busy();
		}
	}

	//First Clear serial port from previous stuff
	int ret = ClearPort();
	if (ret != DEVICE_OK)
		return false;

	// Send command
	string answer = "";
	ret = SendRecieve("!:", answer);
	if (ret != DEVICE_OK)
		return false;

	// Receive check (busy or ready)
	if (model_ == XYStage::HSC3||model_==SHRC3)
	{
		string::size_type index = answer.find("1");
		if (index == string::npos)
		{
			return false;	//ready
		}
		else
		{
			return true;	//busy
		}
	}
	else if (answer.substr(0, 1).compare("B") == 0)
	{
		return true;		//busy
	}
	
	return false;			//ready
}

/// <summary>
/// Set device model
/// <returns></returns>
int XYStage::SetDeviceModel()
{
	//First Clear serial port from previous stuff
	int ret = ClearPort();
	if (ret != DEVICE_OK)
		return ret;


	// Send status command and recieve data
	string data = "";
	ret = SendRecieve("Q:", data);
	if (ret != DEVICE_OK)
		return ret;

	const int SHOT2_STATUS_STRING_LENGTH = 27;
	const int SHOT4_STATUS_STRING_LENGTH = 49;
	const int FC2_STATUS_STRING_LENGTH = 32;

	// setting
	if (data.length() == SHOT2_STATUS_STRING_LENGTH)
	{
		model_ = XYStageModel::SHOT2;		// SHOT-302GS, SHOT-702
		g_DivisionInfo = "Set Division Of Shot-2Axis";
		g_XYStageController = "SHOT-302GS/SHOT-702";
	}
	else if (data.length() == SHOT4_STATUS_STRING_LENGTH)
	{
		model_ = XYStageModel::SHOT4;		// SHOT-304GS
		g_DivisionInfo = "Set Division Of Shot-304";
		g_XYStageController = "SHOT-304GS";
	}
	else if (data.length() == FC2_STATUS_STRING_LENGTH)
	{
		model_ = XYStageModel::FC2;
		g_DivisionInfo = "No Division Setting For FC Series";
		g_XYStageController = "FC Series Controller";

		// Get product name
		string name = "unknown";
		ret = SendRecieve("*IDN?", name);
		if (ret != DEVICE_OK) { return ret; }

		// ','-separated array
		vector<string> product_info = split(name, ',');
	
		string resolution_axis1;
		string resolution_axis2;
		
		const int FC_MODEL_INDEX = 0;
		const int FC_AXIS_COUNT = 1;

		fcModel_ = product_info[FC_MODEL_INDEX] + " " + product_info[FC_AXIS_COUNT];
		
		// Get resolution
		// Key words are extracted to determine old and current models.
		// "0"	:old model		...FC/SC-101G, FC-401G, FC-501G
		// "1"	:current model	...FC-111, 411, 511, 611, 911, 114, 414, 514 
		string key_word = product_info[FC_AXIS_COUNT].substr(4, 1);
		if (key_word == "1")
		{
			/*
			* return unit :nanometer
			*/
			ret = SendRecieve("RESO:"+ to_string(channelX_), resolution_axis1);
			if (ret != DEVICE_OK) { return ret; }
			ret = SendRecieve("RESO:"+ to_string(channelY_), resolution_axis2);
			if (ret != DEVICE_OK) { return ret; }

			stepSizeXum_ = stod(resolution_axis1)/double(1000);
			stepSizeYum_ = stod(resolution_axis2)/double(1000);
		}
		else
		{
			/*
			* return unit :micrometer
			*/
			ret = SendRecieve("RES:"+ to_string(channelX_), resolution_axis1);
			if (ret != DEVICE_OK) { return ret; }
			ret = SendRecieve("RES:"+ to_string(channelY_), resolution_axis2);
			if (ret != DEVICE_OK) { return ret; }

			stepSizeXum_ = stod(resolution_axis1);
			stepSizeYum_ = stod(resolution_axis2);
		}
		// Freeing up memory for vectors
		vector<string>().swap(product_info);
	}
	else
	{
		//Get product name
		
		string name = "unknown";
		ret = SendRecieve("?:N", name);
		if (ret != DEVICE_OK) { return ret; }

		// remove white spaces 
		name.erase(std::remove_if(name.begin(), name.end(), isspace), name.end());

		if (strcmp(name.c_str(), "HSC-103" ) == 0)
		{
			model_ = XYStageModel::HSC3;		// HSC-103
			g_DivisionInfo = "No division setting for HSC controller | fixed to 40";
			g_XYStageController = "HSC-103 Controller";
		}
		else if (strcmp(name.c_str(), "SHRC-203") == 0)
		{
			model_ = XYStageModel::SHRC3;	// SHRC-203
			g_DivisionInfo = "Set Division Of SHRC-203";
			g_XYStageController = "SHRC-203";
			string mode = "";
			//MODE:HOST (to be confirmed)
			//FMT:HIT
			ret = SendRecieve("?:FMT",mode);
			if (ret != DEVICE_OK) { return ret; }

			if (mode != "HIT")
			{
				SendCheckRecievedOK("FMT:HIT");
				if (ret != DEVICE_OK) { return ret; }
			}

		}
	}

	return DEVICE_OK;
}

/// <summary>
/// Initialize
/// <returns></returns>
int XYStage::Initialize()
{
	if (initialized_){ return DEVICE_OK; }
	core_ = GetCoreCallback();

	this->LogMessage("XYStage::Initialize\n", true);

	//Set device model 
	int ret = SetDeviceModel();
	if (ret != DEVICE_OK) { return ERR_XYSTEGE_DEVICE_UNRECOGNIZABLE; }

	// Create propiretes XY-Channel
	CPropertyAction* pAct = new CPropertyAction(this, &XYStage::OnChannelX);
	ret = CreateProperty(g_ChannelX, "1", MM::Integer, false, pAct);
	if (ret != DEVICE_OK)
		return ret;
	pAct = new CPropertyAction(this, &XYStage::OnChannelY);
	ret = CreateProperty(g_ChannelY, "2", MM::Integer, false, pAct);
	if (ret != DEVICE_OK)
		return ret;

	if (model_ != FC2)
	{
		// stages full step X Y 
		// Valid for all controllers except FC Series
		pAct = new CPropertyAction(this, &XYStage::OnFullStepSizeX);
		ret = CreateProperty(g_FullStepStageX, "2", MM::String, false, pAct);
		if (ret != DEVICE_OK)
			return ret;
		pAct = new CPropertyAction(this, &XYStage::OnFullStepSizeY);
		ret = CreateProperty(g_FullStepStageY, "2", MM::String, false, pAct);
		if (ret != DEVICE_OK)
			return ret;
	}

	bool ifdisabled = true; // enable/disable Division Setting in proprietes Available on (SHOT Series) not available on (HSC & FC Series)  

	// Channels and full step creation 
	switch (model_)
	{
	case XYStage::SHOT2:
		for (int i = 1; i <= 2; i++)
		{
			AddAllowedValue(g_ChannelX, to_string(i).c_str());
			AddAllowedValue(g_ChannelY, to_string(i).c_str());
		}
		CreateFullStepPropXY();
		ifdisabled = false;
		break;
	case XYStage::FC2:
		for (int i = 1; i <= 2; i++)
		{
			AddAllowedValue(g_ChannelX, to_string(i).c_str());
			AddAllowedValue(g_ChannelY, to_string(i).c_str());
		}
		// Read Only Prop {{ controller model, Resolution, stepsizeX and stepsizeY }}
		CreateProperty(g_FCModel, fcModel_.c_str(), MM::String, true);
		CreateProperty(g_FCResolution, to_string(stepSizeXum_).c_str(), MM::String, true);
		break;
	case XYStage::SHOT4:
		for (int i = 1; i <= 4; i++)
		{
			AddAllowedValue(g_ChannelX, to_string(i).c_str());
			AddAllowedValue(g_ChannelY, to_string(i).c_str());
		}
		CreateFullStepPropXY();
		ifdisabled = false;
		break;
	case XYStage::HSC3:
		for (int i = 1; i <= 3; i++)
		{
			AddAllowedValue(g_ChannelX, to_string(i).c_str());
			AddAllowedValue(g_ChannelY, to_string(i).c_str());
		}
		CreateFullStepPropXY();
		ifdisabled = true;
		
	
		// Pulse Rate Information 
		CreateProperty("Stage Pulse Rate Info", g_PulseRateInfo, MM::String, true);
		// Set pulse rate 1
		pAct = new CPropertyAction(this, &XYStage::onPulseRateX);
		ret = CreateProperty("StageX Pulse Rate", to_string(PlsRate1).c_str(), MM::String, true, pAct);
		if (ret != DEVICE_OK)
			return ret;
		// Set pulse rate 2
		pAct = new CPropertyAction(this, &XYStage::onPulseRateY);
		ret = CreateProperty("StageY Pulse Rate", to_string(PlsRate2).c_str(), MM::String, true, pAct);
		if (ret != DEVICE_OK)
			return ret;

		break;
	case XYStage::SHRC3:

		for (int i = 1; i <= 3; i++)
		{
			AddAllowedValue(g_ChannelX, to_string(i).c_str());
			AddAllowedValue(g_ChannelY, to_string(i).c_str());
		}
		CreateFullStepPropXY();
		ifdisabled = false;
		/*
		* TODO: To be implemented once specifications are finalised.
		*/
		break;
	default:
		break;
	}

	
	// Axis Division X and Y for all controllers except FC Serie.
	// No division for FC Series
	// No step Size for FC Series
	if (model_ != FC2)
	{
		pAct = new CPropertyAction(this, &XYStage::OnDivisionX);
		ret = CreateProperty(g_DivisionX, "20", MM::String, ifdisabled, pAct);
		if (ret != DEVICE_OK)
			return ret;

		pAct = new CPropertyAction(this, &XYStage::OnDivisionY);
		ret = CreateProperty(g_DivisionY, "20", MM::String, ifdisabled, pAct);
		if (ret != DEVICE_OK)
			return ret;
		CreateDivisionPropXY();
	}
	// Step Size X proprietes (Read only )
	pAct = new CPropertyAction(this, &XYStage::OnStepSizeX);
	ret = CreateProperty(g_XStepSize, to_string(stepSizeXum_).c_str(), MM::String, true, pAct);
	if (ret != DEVICE_OK)
		return ret;
	// Step Size y proprietes (Read only )
	pAct = new CPropertyAction(this, &XYStage::OnStepSizeY);
	ret = CreateProperty(g_YStepSize, to_string(stepSizeYum_).c_str(), MM::String, true, pAct);
	if (ret != DEVICE_OK)
		return ret;
	

	// Division Information read only 
	CreateProperty("XY Division Information", g_DivisionInfo, MM::String, true);

	// controller name 
	CreateProperty("Name of Controller(XY)", g_XYStageController, MM::String, true);

	// XY-Speed (micron/sec) 
	pAct = new CPropertyAction(this, &XYStage::OnSpeedXY);
	ret = CreateProperty(g_XYStageSpeed, "3000", MM::Integer, false, pAct);
	if (ret != DEVICE_OK)
		return ret;
	SetPropertyLimits(g_XYStageSpeed, MinSpeedum, MaxSpeedum);

	ret = UpdateStatus();
	if (ret != DEVICE_OK)
		return ret;
	initialized_ = true;

	return DEVICE_OK;
}

/// <summary>
/// Mechanical origin return
/// <returns></returns>
int XYStage::Home() 
{
	int ret = GenericCommandProcess("H:");
	if (ret != DEVICE_OK) return ret;

	return DEVICE_OK;
}

/// Setting the XY-stage resolution(move per step [micron]).   Deleted, no needed 
/// NOTE: to be confirmed .
//int XYStage::SetResolution()
//{
//	// Set division of X stage with corresponding division X
//	int ret = SetDivision(channelX_, divisionX_);
//	if (ret != DEVICE_OK)
//		return ret;
//	// Set division of Y stage with corresponding division Y
//	ret = SetDivision(channelY_, divisionY_);
//	if (ret != DEVICE_OK)
//		return ret;
//
//	return DEVICE_OK;
//}

// Setting the XY-stage division by axis(channel). 
/// NOTE: Shot controllers only. (+ SHRC Controller )
int XYStage::SetDivision(int channel, int division)
{
	//First Clear serial port from previous stuff
	int ret = ClearPort();
	if (ret != DEVICE_OK)
		return ret;

	// Send command Åc the  division number
	if (model_ == SHRC3)   // hit mode SHRC controller
	{
		if (channel == 1) { ret = SendCheckRecievedOK("S:" + to_string(division) + ",,"); }
		else if (channel == 2 ) { ret = SendCheckRecievedOK("S:," +  to_string(division) + ","); }
		else if (channel == 3) { ret = SendCheckRecievedOK("S:,," + to_string(division) ); }
		if (ret != DEVICE_OK)
			return ret;
	}
	else {/// Other Shot controller
		ret = SendCheckRecievedOK("S:" + to_string(channel) + to_string(division));
		if (ret != DEVICE_OK)
			return ret;
	}
	return DEVICE_OK;
}

/// <summary>
/// Update position all controller except FC Series
/// </summary>
/// <returns></returns>
int XYStage::UpdatePosition()
{
	// First Clear serial port from previous stuff
	int ret = ClearPort();
	if (ret != DEVICE_OK)
		return ret;

	// Send status command 'Q:' & Recieve position data
	string data = "";
	ret = SendRecieve("Q:", data);
	if (ret != DEVICE_OK)
	return ret;

	// Remove spaces in data strings
	data.erase(std::remove_if(data.begin(), data.end(), isspace), data.end());

	// ','-separated array
	vector<string> status = split(data, ',');

	// Set the position status for each axis 
	switch (model_)
	{
	case XYStage::SHOT2:
	case XYStage::FC2:
		positionXpulse_ = atol(status[channelX_ - 1].c_str());
		positionYpulse_ = atol(status[channelY_ - 1].c_str());
		positionApulse_ = 0;
		positionBpulse_ = 0;
		break;
	case XYStage::SHOT4:
		positionXpulse_ = atol(status[channelX_ - 1].c_str());
		positionYpulse_ = atol(status[channelY_ - 1].c_str());
		positionApulse_ = atol(status[channelA_ - 1].c_str());
		positionBpulse_ = atol(status[channelB_ - 1].c_str());
		break;
	case XYStage::HSC3:
	case XYStage::SHRC3:
		positionXpulse_ = atol(status[channelX_ - 1].c_str());
		positionYpulse_ = atol(status[channelY_ - 1].c_str());
		positionApulse_ = atol(status[channelA_ - 1].c_str());
		positionBpulse_ = 0;
		break;
	default:
		positionXpulse_ = 0;
		positionYpulse_ = 0;
		positionApulse_ = 0;
		positionBpulse_ = 0;
		break;
	}

	// Freeing up memory for vectors
	vector<string>().swap(status);
	
	return DEVICE_OK;
}

/// <summary>
/// Update position only fc controllers 
/// Unit -> true is micromter, otherwise is pulse
/// </summary>
/// <returns></returns>



/// <summary>
/// Assignment of other channels 'A' and 'B'
/// </summary>
void XYStage::AssignmentOtherChannels() 
{
	switch (model_)
	{
	case XYStage::SHOT2:
	case XYStage::FC2:
		channelA_ = 3;
		channelB_ = 4;
		break;
	case XYStage::SHOT4:
		for (int i = 1; i <= 4; i++)
		{
			if (i != channelX_)
			{
				if (i != channelY_)
				{
					channelA_ = i;
					break;
				}
			}
		}
		for (int j = 1; j <= 4; j++)
		{
			if (j != channelX_)
			{
				if (j != channelY_)
				{
					if (j != channelA_)
					{
						channelB_ = j;
						break;
					}
				}
			}
		}
		break;
	case XYStage::HSC3:
	case XYStage::SHRC3:
		for (int i = 1; i <= 3; i++)
		{
			if (i != channelX_)
			{
				if (i != channelY_)
				{
					channelA_ = i;
					break;
				}
			}
		}
		channelB_ = 4;
		break;
	default:
		channelA_ = 3;
		channelB_ = 4;
		break;
	}
	core_->LogMessage(device_, ("X:" + to_string(channelX_) +
								",Y:" + to_string(channelY_) +
								",A:" + to_string(channelA_) +
								",B:" + to_string(channelB_) + "\n").c_str(), false);
}

/// <summary>
/// Create drive command; @Test added, 2022-03-03, h.kibata@sigma-koki.com
/// </summary>
/// <param name="x">X-position[pulse]</param>   // X-position [Micrometer] fc series 
/// <param name="y">Y-position[pulse]</param>   // Y-position [Micrometer] fc series 
/// <param name="is_abs">drive mode, true->absolue/false->relative</param>
/// <returns>command string</returns>
string XYStage::ToDriveCommand(long x, long y, bool is_abs)
{
	// Update status
	if (UpdatePosition() != DEVICE_OK)
	{
		return "";
	}
	
	int axis_count = 2;					//	Axis count
	string pos_header;				    //	Number of Pulses 
	string comma;						//	Comma in case of Hit mode
	string command;						//	Commands
	long* src_pos = new long[4];
	src_pos[0] = x;
	src_pos[1] = y;

	// Drive mode
	if (is_abs)
	{
		//Absolue
		command = "A:";
		src_pos[2] = positionApulse_;
		src_pos[3] = positionBpulse_;
	}
	else
	{
		//Relative
		command = "M:";
		src_pos[2] = 0;
		src_pos[3] = 0;
	}

	switch (model_)
	{
	case XYStage::SHOT2:case XYStage::FC2:
		axis_count = 2;
		pos_header = "P";
		comma = "";
		command += "W";
		break;
	case XYStage::SHOT4:
		axis_count = 4;
		pos_header = "P";
		comma = "";
		command += "W";
		break;
	case XYStage::HSC3:case XYStage::SHRC3:
		axis_count = 3;
		pos_header = "";
		comma = ",";
		break;
	default:
		axis_count = 2;
		pos_header = "P";
		comma = "";
		command += "W";
		break;
	}

	string* sign = new string[axis_count];							// Sign array
	string* buffer_command = new string[axis_count];				// Buffer command array
	long dst_pos[4] = {0, 0, 0, 0};									// Destination position array
	int channel[4] = {channelX_, channelY_, channelA_, channelB_};	// Channel array
	
	// combined
	for (int i = 0; i < axis_count; i++)
	{
		for (int j = 0; j < axis_count; j++)
		{
			if (i == channel[j] - 1) // Get Amount Sign in all axis
			{
				if (src_pos[j] >= 0)
				{ 
					sign[i] = "+";
				}
				else
				{
					sign[i] = "-";
				}
				dst_pos[i] = src_pos[j];
			}
		}
		if (i == axis_count - 1)
		{
			buffer_command[i] = sign[i] + pos_header + to_string(abs(dst_pos[i]));  // Build and create command SHOT mode
		}
		else
		{
			buffer_command[i] = sign[i] + pos_header + to_string(abs(dst_pos[i])) + comma; // Build and create command HIT mode
		}
		command += buffer_command[i];
	}

	// retrieve 
	delete[] src_pos;
	delete[] sign;
	delete[] buffer_command;

	return command;
}

/// <summary>
/// Drive command processing (generic)
/// </summary>
/// <param name="x">X-position[pulse]</param>
/// <param name="y">Y-position[pulse]</param>
/// <param name="is_abs">drive mode, true->absolue/false->relative</param>
/// <returns></returns>
int XYStage::DriveCommandProcess(long x, long y, bool is_abs)
{
	// First Clear serial port
	int ret = ClearPort();
	if (ret != DEVICE_OK)
		return ret;

	// create drive command
	string command = ToDriveCommand(x, y, is_abs);
	if (command.empty())
	{
		return DEVICE_UNSUPPORTED_COMMAND;
	}

	// Send command
	switch (model_)
	{
	case XYStage::SHOT2:
	case XYStage::SHOT4:
		ret = SendCheckRecievedOK(command);
		if (ret != DEVICE_OK) return ret;
		ret = SendCheckRecievedOK("G:");
		if (ret != DEVICE_OK) return ret;
		break;
	case XYStage::FC2:
		ret = SendCommand(command);
		if (ret != DEVICE_OK) return ret;
		ret = SendCommand("G");
		if (ret != DEVICE_OK) return ret;
		break;
	case XYStage::HSC3:case XYStage::SHRC3:
		ret = SendCheckRecievedOK(command);
		if (ret != DEVICE_OK) return ret;
		break;
	
		/*
		* TODO: To be implemented once specifications are finalised.
		*/
	default:
		return DEVICE_UNSUPPORTED_COMMAND;
	}

	return DEVICE_OK;
}

/// <summary>
/// Generic command 'H:', 'L:', 'R:' processing
/// </summary>
/// <param name="cmd_header">command header string</param>
/// <returns></returns>
int XYStage::GenericCommandProcess(string command_header)
{
	// First Clear serial port from previous stuff
	int ret = ClearPort();
	if (ret != DEVICE_OK)
		return ret;

	string retour;
	// Send Stop Command
	switch (model_)
	{
	case XYStage::SHOT2:
		ret = SendCheckRecievedOK(command_header + "W");
		if (ret != DEVICE_OK) return ret;
		break;
	case XYStage::SHOT4:
		if (command_header == "H:")
		{
			ret = SendCheckRecievedOK(command_header + to_string(channelX_));
			if (ret != DEVICE_OK) return ret;
			isBusyHomeShot4_= true;
		}
		else
		{
			ret = SendCheckRecievedOK(command_header + to_string(channelX_));
			if (ret != DEVICE_OK) return ret;
		
			ret = SendCheckRecievedOK(command_header + to_string(channelY_));
			if (ret != DEVICE_OK) return ret;
		}
		break;
	case XYStage::FC2:
		ret = SendCommand(command_header + "W");
		if (ret != DEVICE_OK) return ret;
		break;
	case XYStage::HSC3:case XYStage::SHRC3:
		for (int i = 1; i <= 3; i++)
		{
			if (i == channelX_)
			{
				command_header += "1";
			}
			else if (i == channelY_)
			{
				command_header += "1";
			}
			else
			{
				command_header += "0";
			}
			if (i != 3)
			{
				command_header += ",";
			}
		}
		
		// Send Recieve data check OK 
		ret = SendCheckRecievedOK(command_header);
		if (ret != DEVICE_OK) return ret;
		break;		
	default:
		return DEVICE_UNSUPPORTED_COMMAND;
	}	
	return DEVICE_OK;
}

/// <summary>
/// Pulse Speed setting 
/// </summary>
/// <param name="val"></param>
/// <returns></returns>
int XYStage::PulseSpeedSetting(int val)
{
	// Setting the XY-speed pulse
	fast_pulse = (long)(val / stepSizeXum_);
	slow_pulse = 0;
	switch (model_)
	{
	case XYStage::SHOT2:case XYStage::SHOT4:case XYStage::HSC3:case XYStage::SHRC3:
		//Speed(F) [PPS]
		if (fast_pulse < 1 || fast_pulse > (long)(MaxSpeedum/stepSizeXum_)) { return ERR_XYSTAGE_SPEED_FAILED; }
		//Speed(S) [PPS]
		slow_pulse = GetSlowSpeedPulse(fast_pulse);
		break;
	case XYStage::FC2:
		if (fast_pulse < 1 || fast_pulse > (long)(MaxSpeedum / stepSizeXum_)) { return ERR_XYSTAGE_SPEED_FAILED; }
		break;
	
	default:
		break;
	}

	return DEVICE_OK;
}

/// </summary>
/// Stop stage
/// <returns></returns>
int XYStage::Stop()
{
	int ret = GenericCommandProcess("L:");
	if (ret != DEVICE_OK)
		return ret;
	return DEVICE_OK;
}

/// <summary>
/// Logical origin
/// <returns></returns>
int XYStage::SetOrigin()
{
	int ret = GenericCommandProcess("R:");
	if (ret != DEVICE_OK) 
		return ret;

	return DEVICE_OK;
}

/// <summary>
/// Positioning by steps || Absolute Move (pulse)
/// <returns></returns>
int XYStage::SetPositionSteps(long x, long y)
{
	int ret = DriveCommandProcess(x, y, true);
	if (ret != DEVICE_OK)
		return ret;

	return DEVICE_OK;
}

/// <summary>
/// Positioning by steps || Relative Move (pulse)
/// <returns></returns>
int XYStage::SetRelativePositionSteps(long x, long y)
{
	int ret = DriveCommandProcess(x, y, false);
	if (ret != DEVICE_OK)
		return ret;

	return DEVICE_OK;
}

/// <summary>
/// Get position per micrometer
/// </summary>
/// <param name="x">X-position[um]</param>
/// <param name="y">Y-position[um]</param>
/// <returns></returns>
int XYStage::GetPositionUm(double& x, double& y)
{

	int ret = UpdatePosition();
	if (ret != DEVICE_OK)
		return ret;

	// retrieve x and y in Um
	x = (double)(positionXpulse_ * stepSizeXum_);
	y = (double)(positionYpulse_ * stepSizeYum_);

	return DEVICE_OK;
}

/// <summary>
/// Get position steps
/// </summary>
/// <param name="x">X-position[pulse]</param>
/// <param name="y">Y-position[pulse]</param>
/// <returns></returns>
int XYStage::GetPositionSteps(long& x, long& y)
{
	
	int ret = UpdatePosition();
	if (ret != DEVICE_OK)
		return ret;
	
	x = positionXpulse_;
	y = positionYpulse_;

	return DEVICE_OK;
}

/// <summary>
/// Set speed
/// </summary>
/// <param name="val">speed value</param>
/// <returns></returns>
int XYStage::SetSpeedXY(int val)
{
	//First Clear serial port from previous stuff
	int ret = ClearPort();
	if (ret != DEVICE_OK)
		return ret;

	// Set slow_pulse and fast_pulse
	ret = PulseSpeedSetting(val);
	if (ret != DEVICE_OK)
		return ret;

	// Create command and send
	// NOTE: Acceleration and Deceleration time fixed to 100 msec.
	string cmd; string cmd_x; string cmd_y;
	switch (model_)
	{
	case XYStage::SHOT2:
		cmd = "D:WS" + to_string(slow_pulse) + "F" + to_string(fast_pulse) + "R100S" + to_string(slow_pulse) + "F" + to_string(fast_pulse) + "R100";
		SendCheckRecievedOK(cmd);
		break;
	case XYStage::SHOT4:
		cmd_x = "D:" + to_string(channelX_) + "S" + to_string(slow_pulse) + "F" + to_string(fast_pulse) + "R100";
		SendCheckRecievedOK(cmd_x);
		cmd_y = "D:" + to_string(channelY_) + "S" + to_string(slow_pulse) + "F" + to_string(fast_pulse) + "R100";
		SendCheckRecievedOK(cmd_y);
		break;
	case XYStage::FC2:
		cmd = "D:WF" + to_string(fast_pulse) + "F" + to_string(fast_pulse);
		SendCommand(cmd);
		break;
	case XYStage::HSC3:case XYStage::SHRC3:
		cmd = "D:" + to_string(channelX_) + "," + to_string(slow_pulse) + "," + to_string(fast_pulse) + ",100";
		SendCheckRecievedOK(cmd);
		cmd = "D:" + to_string(channelY_) + "," + to_string(slow_pulse) + "," + to_string(fast_pulse) + ",100";
		SendCheckRecievedOK(cmd);
		break;
	default:
		break;
	}

	// Send command (Setting of acc and dec time of FC series.)
	if (model_== FC2)
	{
		// X-axis
		string acc_x;
		acc_x = "ACC:" + to_string(channelX_) + " 100";
		ret = SendCommand(acc_x);
		if (ret != DEVICE_OK)
			return ret;
		// Y-axis
		string acc_y;
		acc_y = "ACC:" + to_string(channelY_) + " 100";
		ret = SendCommand(acc_y);
		if (ret != DEVICE_OK)
			return ret;
	}
	return DEVICE_OK;
}

/// <summary>
/// Get slow speed
/// </summary>
/// <param name="fast">'Fast'-speed[pulse]</param>
/// <returns></returns>
long XYStage::GetSlowSpeedPulse(long fast)
{
	long slow = 0;
	if (fast <= 50000) { slow = fast / 2; }
	else { slow = 15000; }
	return slow;
}

/// <summary>
/// Get Limit <not supported>
/// <returns></returns>
int XYStage::GetLimitsUm(double& /*xMin*/, double& /*xMax*/, double& /*yMin*/, double& /*yMax*/)
{
	return DEVICE_UNSUPPORTED_COMMAND;
}

/// <summary>
/// Get Limit step <not supported>
/// <returns></returns>
int XYStage::GetStepLimits(long& /*xMin*/, long& /*xMax*/, long& /*yMin*/, long& /*yMax*/)
{
	return DEVICE_UNSUPPORTED_COMMAND;
}




///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Action handlers
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int XYStage::OnPort(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(port_.c_str());
	}
	else if (eAct == MM::AfterSet)
	{
		if (initialized_)
		{
			// revert
			pProp->Set(port_.c_str());
			return ERR_PORT_CHANGE_FORBIDDEN;
		}
		pProp->Get(port_);
	}

	return DEVICE_OK;
}


int XYStage::OnChannelX(MM::PropertyBase* pProp, MM::ActionType eAct)
{	
	if (eAct == MM::BeforeGet)
	{
		pProp->Set((long)channelX_);
	}
	else if (eAct == MM::AfterSet)
	{
		long n;
		pProp->Get(n);
		channelX_ = int(n);
		AssignmentOtherChannels();
	}
	
	return DEVICE_OK;
}

int XYStage::OnChannelY(MM::PropertyBase* pProp, MM::ActionType eAct)
{	
	if (eAct == MM::BeforeGet)
	{
		pProp->Set((long)channelY_);
	}
	else if (eAct == MM::AfterSet)
	{
		long n;
		pProp->Get(n);
		channelY_ = int(n);
		AssignmentOtherChannels();
	}

	return DEVICE_OK;
}

// NOTE: Step size is calculated depends full step stage and division.
int XYStage::OnStepSizeX(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		if (model_ == SHOT2 ||model_== SHOT4||model_ == SHRC3)
		{
			int ret = SetDivision(channelX_, divisionX_);
			if (ret != DEVICE_OK)
				return ERR_XYSTAGE_STEPSIZE_FAILED;
		}
	
		pProp->Set(stepSizeXum_);
	}

	return DEVICE_OK;
}

// NOTE: Because the resolution is fixed (0.1 micron), Action-interface does not create now.
int XYStage::OnStepSizeY(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		if (model_ == SHOT2 ||model_ == SHOT4 || model_ == SHRC3)
		{
			int ret = SetDivision(channelY_, divisionY_);
			if (ret != DEVICE_OK)
				return ERR_XYSTAGE_STEPSIZE_FAILED;
		}
	
		pProp->Set(stepSizeYum_);
	}

	return DEVICE_OK;
}


// Set speed for all controllers 
int XYStage::OnSpeedXY(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set((long)speedXYum_);

		int ret = SetSpeedXY(speedXYum_);
		if (ret != DEVICE_OK)
			return ERR_XYSTAGE_SPEED_FAILED;
	}
	else if (eAct == MM::AfterSet)
	{
		long n;
		pProp->Get(n);
		speedXYum_ = (int)n;

		int ret = SetSpeedXY(speedXYum_);
		if (ret != DEVICE_OK)
			return ERR_XYSTAGE_SPEED_FAILED;
	}

	return DEVICE_OK;
}

// Get Set proprietes full step X (depend on stages categories)
int XYStage::OnFullStepSizeX(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(fullStepSizeXum_);
	}
	else if (eAct == MM::AfterSet)
	{
		long n;
		pProp->Get(n);
		fullStepSizeXum_ = n;
	}
	
	return DEVICE_OK;
}
// Get Set proprietes full step Y (depend on stages categories)
int XYStage::OnFullStepSizeY(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(fullStepSizeYum_);
	}
	else if (eAct == MM::AfterSet)
	{
		long n;
		pProp->Get(n);
		fullStepSizeYum_ = n;
	}

	return DEVICE_OK;
}

// Get Set proprietes Division X
int XYStage::OnDivisionX(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		string x = "00";
		if (model_ == SHOT2 || model_ == SHOT4||model_==SHRC3)
		{
			x = to_string(divisionX_);
			int ret = SetDivision(channelX_, divisionX_);
			if (ret != DEVICE_OK)
				return ERR_XYSTAGE_STEPSIZE_FAILED;
			stepSizeXum_ = fullStepSizeXum_ / double(divisionX_);
		}
		if (model_ == HSC3)
		{
			divisionX_ = 40;
			x = "40";
			//stepSizeXum_ = fullStepSizeXum_ / double(divisionX_); 
			stepSizeXum_ = 0.01;
		}
		pProp->Set(x.c_str());
	}
	else if (eAct == MM::AfterSet)
	{
		string n;
		pProp->Get(n);
		if (model_ == SHOT2 || model_ == SHOT4 || model_ == SHRC3)
		{
			divisionX_ = stoi(n);
			int ret = SetDivision(channelX_, divisionX_);
			if (ret != DEVICE_OK)
				return ERR_XYSTAGE_STEPSIZE_FAILED;
			stepSizeXum_ = fullStepSizeXum_ / double(divisionX_);
		}
		if (model_ == HSC3)
		{
			divisionX_ = 40;
			//stepSizeXum_ = fullStepSizeXum_ / double(divisionX_);
			stepSizeXum_ = 0.01;
		}
	}
	
	return DEVICE_OK;
}

// Get Set proprietes Division y
int XYStage::OnDivisionY(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{   
		string y = "00";
		if (model_ == SHOT2 || model_ == SHOT4 || model_ == SHRC3)
		{
			y = to_string(divisionY_);
			
			int ret = SetDivision(channelY_, divisionY_);
			if (ret != DEVICE_OK)
				return ERR_XYSTAGE_STEPSIZE_FAILED;

			stepSizeYum_ = fullStepSizeYum_ / double(divisionY_);
		}
		if (model_ == HSC3)
		{
			divisionY_ = 40;
			y = "40";
			//stepSizeYum_ = fullStepSizeYum_ / double(divisionY_);
			stepSizeYum_ = 0.01;
		}
		pProp->Set(y.c_str());

	}
	else if (eAct == MM::AfterSet)
	{
		string n;
		pProp->Get(n);
		if (model_ == SHOT2 || model_ == SHOT4 || model_ == SHRC3)
		{
			divisionY_ = stoi(n);
			int ret = SetDivision(channelY_, divisionY_);
			if (ret != DEVICE_OK)
				return ERR_XYSTAGE_STEPSIZE_FAILED;

			stepSizeYum_ = fullStepSizeYum_ / double(divisionY_);
		}
		if (model_ == HSC3)
		{
			divisionY_ = 40;
			//stepSizeYum_ = fullStepSizeYum_ / double(divisionY_);
			stepSizeYum_ = 0.01;
		}
	}
	
	return DEVICE_OK;
}


// Pulse Rate Setting for HSC-103 only 
int XYStage::onPulseRateX(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		/*Pulse Rate Calculclation in HSC-103 controller 
		PULSE RATE = (FULL_STEP / DIVISION)* 1000 /0.1*/
		const int UM_TO_NM_RATE = 1000;
		const double NM_TO_PULSERATE_RATE = 0.1;
		if (model_ == HSC3)
		{
			PlsRate1 = ((fullStepSizeXum_/40) * UM_TO_NM_RATE) / NM_TO_PULSERATE_RATE;
		}
		pProp->Set(long(PlsRate1));
	}
	
	return DEVICE_OK;
}

int XYStage::onPulseRateY(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		/* Pulse Rate Calculclation in HSC-103 controller 
		 PULSE RATE = (FULL_STEP / DIVISION)* 1000 /0.1*/
		const int UM_TO_NM_RATE = 1000;
		const double NM_TO_PULSERATE_RATE = 0.1;
		if (model_ == HSC3)
		{
			PlsRate2 = ((fullStepSizeYum_ / 40) * UM_TO_NM_RATE) / NM_TO_PULSERATE_RATE;
		}
		pProp->Set(long(PlsRate2));
	}
	return DEVICE_OK;
}

/// <summary>
/// Create Division proprietes values
/// </summary>
void XYStage::CreateDivisionPropXY()
{
	ClearAllowedValues(g_DivisionX);
	ClearAllowedValues(g_DivisionY);

	switch (model_)
	{
	case XYStage::SHOT2:case XYStage::SHOT4:case XYStage::SHRC3:case XYStage::HSC3:
		/*
		* HACK: Where possible, we would like to make the code a bit smarter.
		*/
		// Division X 
		AddAllowedDivisionPropXY(g_DivisionX);
		// Division Y
		AddAllowedDivisionPropXY(g_DivisionY);
		break;
	case XYStage::FC2:
		/*
		* NOTE: Fixed for each FC series model.
		*/
		break;
	default:
		break;
	}
}

/// <summary>
/// Creation of full step proprietes values.
/// </summary>
void XYStage::CreateFullStepPropXY()
{
	// Full Steps X
	AddAllowedValue(g_FullStepStageX, "2");
	AddAllowedValue(g_FullStepStageX, "4");
	AddAllowedValue(g_FullStepStageX, "20");
	// Full Steps Y
	AddAllowedValue(g_FullStepStageY, "2");
	AddAllowedValue(g_FullStepStageY, "4");
	AddAllowedValue(g_FullStepStageY, "20");
}

/// <summary>
/// Add division prop for X and Y (reduce size and avoid writing for x and y proprietes one by one) 
/// </summary>
/// <param name="div"></param>
void XYStage::AddAllowedDivisionPropXY(const char* div)
{

	switch (model_)
	{
	case XYStage::SHOT2:case XYStage::SHOT4:
		// Division X and Y
		AddAllowedValue(div, "1");
		AddAllowedValue(div, "2");
		AddAllowedValue(div, "4");
		AddAllowedValue(div, "5");
		AddAllowedValue(div, "8");
		AddAllowedValue(div, "10");
		AddAllowedValue(div, "20");
		AddAllowedValue(div, "25");
		AddAllowedValue(div, "40");
		AddAllowedValue(div, "50");
		AddAllowedValue(div, "80");
		AddAllowedValue(div, "100");
		AddAllowedValue(div, "125");
		AddAllowedValue(div, "200");
		AddAllowedValue(div, "250");
		break;
	case XYStage::FC2:
		break;
	case XYStage::HSC3:
		/*
		* NOTE: Fixed in 40 divisions.
		*/
		AddAllowedValue(div, "40");

		break;
	case XYStage::SHRC3:
		for (int i = 1; i <= 6; i++)    // Add from 1 to 6
		{
			AddAllowedValue(div, to_string(i).c_str());
		}
		AddAllowedValue(div, "8");
		AddAllowedValue(div, "12");
		AddAllowedValue(div, "25");
		for (int i = 1; i < 60; i++) // Add from 10 to 60
		{
			i += 9;
			AddAllowedValue(div, to_string(i).c_str());
		}
		AddAllowedValue(div, "80");
		AddAllowedValue(div, "120");
		AddAllowedValue(div, "125");
		AddAllowedValue(div, "250");
		for (int i = 1; i < 400; i++) // add from 100 to 400
		{
			i += 99;
			AddAllowedValue(div, to_string(i).c_str());
		}
		for (int i = 600; i <= 1000; i++) // add from 600 to 1000
		{
			int k = i;
			AddAllowedValue(div, to_string(k).c_str());
			i = i + 199;
		}
		for (int i = 1000; i <= 8000; i++) // add from 1000 to 8000
		{
			int l = i;
			AddAllowedValue(div, to_string(l).c_str());
			i = i + i - 1;
		}

		break;
	default:
		break;
	}


}
