/*
 * Project: ASIStage Device Adapter
 * License/Copyright: BSD 3-clause, see license.txt
 * Maintainers: Brandon Simpson (brandon@asiimaging.com)
 *              Jon Daniels (jon@asiimaging.com)
 */

#include "ASIXYStage.h"

XYStage::XYStage() :
	ASIBase(this, "2H"),
	stepSizeXUm_(0.0),
	stepSizeYUm_(0.0),
	maxSpeed_(7.5),
	ASISerialUnit_(10.0),
	motorOn_(true),
	joyStickSpeedFast_(60),
	joyStickSpeedSlow_(5),
	joyStickMirror_(false),
	nrMoveRepetitions_(0),
	answerTimeoutMs_(1000),
	serialOnlySendChanged_(true),
	manualSerialAnswer_(""),
	advancedPropsEnabled_(false),
	axisletterX_("X"),
	axisletterY_("Y")
{
	InitializeDefaultErrorMessages();

	// create pre-initialization properties
	// ------------------------------------

	// Name
	CreateProperty(MM::g_Keyword_Name, g_XYStageDeviceName, MM::String, true);

	// Description
	CreateProperty(MM::g_Keyword_Description, g_XYStageDeviceDescription, MM::String, true);

	// Port
	CPropertyAction* pAct = new CPropertyAction(this, &XYStage::OnPort);
	CreateProperty(MM::g_Keyword_Port, "Undefined", MM::String, false, pAct, true);
	stopSignal_ = false;

	CreateProperty("AxisLetterX", axisletterX_.c_str(), MM::String, true);
	CreateProperty("AxisLetterY", axisletterY_.c_str(), MM::String, true);
}

XYStage::~XYStage()
{
	Shutdown();
}

void XYStage::GetName(char* Name) const
{
	CDeviceUtils::CopyLimitedString(Name, g_XYStageDeviceName);
}

bool XYStage::SupportsDeviceDetection(void)
{
	return true;
}

MM::DeviceDetectionStatus XYStage::DetectDevice(void)
{
	return ASICheckSerialPort(*this, *GetCoreCallback(), port_, answerTimeoutMs_);
}

int XYStage::Initialize()
{
	core_ = GetCoreCallback();

	// empty the Rx serial buffer before sending command
	ClearPort();

	CPropertyAction* pAct = new CPropertyAction(this, &XYStage::OnVersion);
	CreateProperty("Version", "", MM::String, true, pAct);

	// check status first (test for communication protocol)
	int ret = CheckDeviceStatus();
	if (ret != DEVICE_OK)
	{
		return ret;
	}

	pAct = new CPropertyAction(this, &XYStage::OnCompileDate);
	CreateProperty("CompileDate", "", MM::String, true, pAct);
	UpdateProperty("CompileDate");

	// get the date of the firmware
	char compile_date[MM::MaxStrLength];
	if (GetProperty("CompileDate", compile_date) == DEVICE_OK)
	{
		compileDay_ = ExtractCompileDay(compile_date);
	}

	// if really old firmware then don't get build name
	// build name is really just for diagnostic purposes anyway
	// I think it was present before 2010 but this is easy way
	if (compileDay_ >= ConvertDay(2010, 1, 1))
	{
		pAct = new CPropertyAction(this, &XYStage::OnBuildName);
		CreateProperty("BuildName", "", MM::String, true, pAct);
		UpdateProperty("BuildName");
	}

	// Most ASIStages have the origin in the top right corner, the following reverses direction of the X-axis:
	ret = SetAxisDirection();
	if (ret != DEVICE_OK)
	{
		return ret;
	}

	// set stage step size and resolution
	/**
	 * NOTE: ASI returns numbers in 10th of microns with an extra decimal place
	 * To convert into steps, we multiply by 10 (variable ASISerialUnit_) making the step size 0.01 microns
	 */
	stepSizeXUm_ = 0.01;
	stepSizeYUm_ = 0.01;

	// Step size
	pAct = new CPropertyAction(this, &XYStage::OnStepSizeX);
	CreateProperty("StepSizeX_um", "0.0", MM::Float, true, pAct);
	pAct = new CPropertyAction(this, &XYStage::OnStepSizeY);
	CreateProperty("StepSizeY_um", "0.0", MM::Float, true, pAct);

	// Wait cycles
	if (hasCommand("WT X?"))
	{
		pAct = new CPropertyAction(this, &XYStage::OnWait);
		CreateProperty("Wait_Cycles", "5", MM::Integer, false, pAct);
		// SetPropertyLimits("Wait_Cycles", 0, 255);  // don't artificially restrict range
	}

	// Speed (sets both x and y)
	if (hasCommand("S " + axisletterX_ + "?"))
	{
		pAct = new CPropertyAction(this, &XYStage::OnSpeed);
		CreateProperty("Speed-S", "1", MM::Float, false, pAct);
		// Maximum Speed that can be set in Speed-S property
		char max_speed[MM::MaxStrLength];
		GetMaxSpeed(max_speed);
		CreateProperty("Maximum Speed (Do Not Change)", max_speed, MM::Float, true);
	}

	// Backlash (sets both x and y)
	if (hasCommand("B " + axisletterX_ + "?"))
	{
		pAct = new CPropertyAction(this, &XYStage::OnBacklash);
		CreateProperty("Backlash-B", "0", MM::Float, false, pAct);
	}
	
	// Error (sets both x and y)
	if (hasCommand("E " + axisletterX_ + "?"))
	{
		pAct = new CPropertyAction(this, &XYStage::OnError);
		CreateProperty("Error-E(nm)", "0", MM::Float, false, pAct);
	}
	
	// acceleration (sets both x and y)
	if (hasCommand("AC " + axisletterX_ + "?"))
	{
		pAct = new CPropertyAction(this, &XYStage::OnAcceleration);
		CreateProperty("Acceleration-AC(ms)", "0", MM::Integer, false, pAct);
	}
	
	// Finish Error (sets both x and y)
	if (hasCommand("PC " + axisletterX_ + "?"))
	{
		pAct = new CPropertyAction(this, &XYStage::OnFinishError);
		CreateProperty("FinishError-PCROS(nm)", "0", MM::Float, false, pAct);
	}
	
	// OverShoot (sets both x and y)
	if (hasCommand("OS " + axisletterX_ + "?"))
	{
		pAct = new CPropertyAction(this, &XYStage::OnOverShoot);
		CreateProperty("OverShoot(um)", "0", MM::Float, false, pAct);
	}
	
	// MotorCtrl (works on both x and y)
	pAct = new CPropertyAction(this, &XYStage::OnMotorCtrl);
	CreateProperty("MotorOnOff", "On", MM::String, false, pAct);
	AddAllowedValue("MotorOnOff", "On");
	AddAllowedValue("MotorOnOff", "Off");
	
	// JoyStick MirrorsX
	// TODO: the following properties should only appear in controllers version 8 and higher
	pAct = new CPropertyAction(this, &XYStage::OnJSMirror);
	CreateProperty("JoyStick Reverse", "Off", MM::String, false, pAct);
	AddAllowedValue("JoyStick Reverse", "On");
	AddAllowedValue("JoyStick Reverse", "Off");

	pAct = new CPropertyAction(this, &XYStage::OnJSFastSpeed);
	CreateProperty("JoyStick Fast Speed", "100", MM::Integer, false, pAct);
	SetPropertyLimits("JoyStick Fast Speed", 1, 100);

	pAct = new CPropertyAction(this, &XYStage::OnJSSlowSpeed);
	CreateProperty("JoyStick Slow Speed", "100", MM::Integer, false, pAct);
	SetPropertyLimits("JoyStick Slow Speed", 1, 100);
	
	// property to allow sending arbitrary serial commands and receiving response
	pAct = new CPropertyAction(this, &XYStage::OnSerialCommand);
	CreateProperty("SerialCommand", "", MM::String, false, pAct);

	// this is only changed programmatically, never by user
	// contains last response to the OnSerialCommand action
	pAct = new CPropertyAction(this, &XYStage::OnSerialResponse);
	CreateProperty("SerialResponse", "", MM::String, true, pAct);

	// disable sending serial commands unless changed (by default this is enabled)
	pAct = new CPropertyAction(this, &XYStage::OnSerialCommandOnlySendChanged);
	CreateProperty("OnlySendSerialCommandOnChange", "Yes", MM::String, false, pAct);
	AddAllowedValue("OnlySendSerialCommandOnChange", "Yes");
	AddAllowedValue("OnlySendSerialCommandOnChange", "No");

	// generates a set of additional advanced properties that are rarely used
	pAct = new CPropertyAction(this, &XYStage::OnAdvancedProperties);
	CreateProperty("EnableAdvancedProperties", "No", MM::String, false, pAct);
	AddAllowedValue("EnableAdvancedProperties", "No");
	AddAllowedValue("EnableAdvancedProperties", "Yes");
	
	if (hasCommand("VE " + axisletterX_ + "=0")) {
		char orig_speed[MM::MaxStrLength];
		ret = GetProperty("Speed-S", orig_speed);
		double mspeed;
		if (ret != DEVICE_OK)
		{
			mspeed = 8;
		}
		else
		{
			mspeed = atof(orig_speed);
		}

		pAct = new CPropertyAction(this, &XYStage::OnVectorX);
		CreateProperty("VectorMoveX-VE(mm/s)", "0", MM::Float, false, pAct);
		SetPropertyLimits("VectorMoveX-VE(mm/s)", mspeed * -1, mspeed);
		UpdateProperty("VectorMoveX-VE(mm/s)");

		pAct = new CPropertyAction(this, &XYStage::OnVectorY);
		CreateProperty("VectorMoveY-VE(mm/s)", "0", MM::Float, false, pAct);
		SetPropertyLimits("VectorMoveY-VE(mm/s)", mspeed * -1, mspeed);
		UpdateProperty("VectorMoveY-VE(mm/s)");
	}
	
	initialized_ = true;
	return DEVICE_OK;
}

int XYStage::Shutdown()
{
	if (initialized_)
	{
		initialized_ = false;
	}
	return DEVICE_OK;
}

bool XYStage::Busy()
{
	// empty the Rx serial buffer before sending command
	ClearPort();

	const char* command = "/";
	std::string answer;

	// query the device
	int ret = QueryCommand(command, answer);
	if (ret != DEVICE_OK)
	{
		return false;
	}

	if (answer.length() >= 1)
	{
		if (answer.substr(0, 1) == "B")
		{
			return true;
		}
		else if (answer.substr(0, 1) == "N")
		{
			return false;
		}
		else
		{
			return false;
		}
	}
	return false;
}

int XYStage::SetPositionSteps(long x, long y)
{
	// empty the Rx serial buffer before sending command
	ClearPort();

	std::ostringstream command;
	command << std::fixed << "M " << axisletterX_ << "=" << x / ASISerialUnit_ << " " << axisletterY_ << "=" << y / ASISerialUnit_; // steps are 10th of micros
	std::string answer;

	// query the device
	int ret = QueryCommand(command.str().c_str(), answer);
	if (ret != DEVICE_OK)
	{
		return ret;
	}

	if ((answer.substr(0, 2).compare(":A") == 0) || (answer.substr(1, 2).compare(":A") == 0))
	{
		return DEVICE_OK;
	}
	// deal with error later
	else if (answer.substr(0, 2).compare(":N") == 0 && answer.length() > 2)
	{
		int errNo = atoi(answer.substr(4).c_str());
		return ERR_OFFSET + errNo;
	}
	return ERR_UNRECOGNIZED_ANSWER;
}

int XYStage::SetRelativePositionSteps(long x, long y)
{
	// empty the Rx serial buffer before sending command
	ClearPort();

	std::ostringstream command;
	if ((x == 0) && (y != 0))
	{
		command << std::fixed << "R " << axisletterY_ << "=" << y / ASISerialUnit_;
	}
	else if ((x != 0) && (y == 0))
	{
		command << std::fixed << "R " << axisletterX_ << "=" << x / ASISerialUnit_;
	}
	else
	{
		command << std::fixed << "R " << axisletterX_ << "=" << x / ASISerialUnit_ << " " << axisletterY_ << "=" << y / ASISerialUnit_; // in 10th of microns
	}

	std::string answer;
	// query the device
	int ret = QueryCommand(command.str().c_str(), answer);
	if (ret != DEVICE_OK)
	{
		return ret;
	}

	if ((answer.substr(0, 2).compare(":A") == 0) || (answer.substr(1, 2).compare(":A") == 0))
	{
		return DEVICE_OK;
	}
	// deal with error later
	else if (answer.substr(0, 2).compare(":N") == 0 && answer.length() > 2)
	{
		int errNo = atoi(answer.substr(4).c_str());
		return ERR_OFFSET + errNo;
	}
	return ERR_UNRECOGNIZED_ANSWER;
}

int XYStage::GetPositionSteps(long& x, long& y)
{
	// empty the Rx serial buffer before sending command
	ClearPort();

	std::ostringstream command;
	command << "W " << axisletterX_ << " " << axisletterY_;
	std::string answer;

	// query the device
	int ret = QueryCommand(command.str().c_str(), answer);
	if (ret != DEVICE_OK)
	{
		return ret;
	}

	if (answer.length() > 2 && answer.substr(0, 2).compare(":N") == 0)
	{
		int errNo = atoi(answer.substr(2).c_str());
		return ERR_OFFSET + errNo;
	}
	else if (answer.length() > 0)
	{
		float xx;
		float yy;
		char head[64];
		char iBuf[256];
		strcpy(iBuf, answer.c_str());
		sscanf(iBuf, "%s %f %f\r\n", head, &xx, &yy);
		x = (long)(xx * ASISerialUnit_);
		y = (long)(yy * ASISerialUnit_);

		return DEVICE_OK;
	}
	return ERR_UNRECOGNIZED_ANSWER;
}

int XYStage::SetOrigin()
{
	std::string answer;
	std::string cmd;
	// query the device
	cmd = "H " + axisletterX_ + "=0 " + axisletterY_ + "=0";
	int ret = QueryCommand(cmd.c_str(), answer); // use command HERE, zero (z) zero all x,y,z
	if (ret != DEVICE_OK)
	{
		return ret;
	}

	if (answer.substr(0, 2).compare(":A") == 0 || answer.substr(1, 2).compare(":A") == 0)
	{
		return DEVICE_OK;
	}
	else if (answer.substr(0, 2).compare(":N") == 0 && answer.length() > 2)
	{
		int errNo = atoi(answer.substr(2, 4).c_str());
		return ERR_OFFSET + errNo;
	}
	return ERR_UNRECOGNIZED_ANSWER;
}

void XYStage::Wait()
{
	// empty the Rx serial buffer before sending command
	ClearPort();

	// if (stopSignal_) return DEVICE_OK;
	bool busy = true;
	const char* command = "/";
	std::string answer = "";

	// query the device
	QueryCommand(command, answer);
	// if (ret != DEVICE_OK)
	//     return ret;

	// block/wait for acknowledge, or until we time out;
	if (answer.substr(0, 1) == "B")
	{
		busy = true;
	}
	else if (answer.substr(0, 1) == "N")
	{
		busy = false;
	}
	else
	{
		busy = true;
	}

	// if (stopSignal_) return DEVICE_OK;

	int intervalMs = 100;
	int totaltime = 0;
	while (busy)
	{
		// if (stopSignal_) return DEVICE_OK;
		// Sleep(intervalMs);
		totaltime += intervalMs;

		// query the device
		QueryCommand(command, answer);
		// if (ret != DEVICE_OK)
		//     return ret;

		if (answer.substr(0, 1) == "B")
		{
			busy = true;
		}
		else if (answer.substr(0, 1) == "N")
		{
			busy = false;
		}
		else
		{
			busy = true;
		}

		if (!busy)
		{
			break;
		}
		// if (totaltime > timeout ) break;
	}
	// return DEVICE_OK;
}

int XYStage::Home()
{
	// empty the Rx serial buffer before sending command
	ClearPort();

	std::string answer;
	std::string cmd = "! " + axisletterX_ + " " + axisletterY_;
	// query the device
	int ret = QueryCommand(cmd.c_str(), answer); // use command HOME
	if (ret != DEVICE_OK)
	{
		return ret;
	}

	if (answer.substr(0, 2).compare(":A") == 0 || answer.substr(1, 2).compare(":A") == 0)
	{
		// do nothing
	}
	else if (answer.substr(0, 2).compare(":N") == 0 && answer.length() > 2)
	{
		int errNo = atoi(answer.substr(2, 4).c_str());
		return ERR_OFFSET + errNo;
	}
	return DEVICE_OK;
}

int XYStage::Calibrate() {

	if (stopSignal_)
	{
		return DEVICE_OK;
	}

	double x1, y1;
	int ret = GetPositionUm(x1, y1);
	if (ret != DEVICE_OK)
	{
		return ret;
	}

	Wait();
	// ret = Wait();
	// if (ret != DEVICE_OK)
	//     return ret;

	if (stopSignal_)
	{
		return DEVICE_OK;
	}

	// do home command
	std::string answer;
	std::string cmd = "! " + axisletterX_ + " " + axisletterY_;
	// query the device
	ret = QueryCommand(cmd.c_str(), answer); // use command HOME
	if (ret != DEVICE_OK)
		return ret;

	if (answer.substr(0, 2).compare(":A") == 0 || answer.substr(1, 2).compare(":A") == 0)
	{
		// do nothing
	}
	else if (answer.substr(0, 2).compare(":N") == 0 && answer.length() > 2)
	{
		int errNo = atoi(answer.substr(2, 4).c_str());
		return ERR_OFFSET + errNo;
	}
	return DEVICE_OK;
}

int XYStage::Calibrate1() {
	int ret = Calibrate();
	stopSignal_ = false;
	return ret;
}

int XYStage::Stop()
{
	// empty the Rx serial buffer before sending command
	ClearPort();

	stopSignal_ = true;
	std::string answer;

	// query the device
	int ret = QueryCommand("HALT", answer);  // use command HALT "\"
	if (ret != DEVICE_OK)
	{
		return ret;
	}

	if (answer.substr(0, 2).compare(":A") == 0 || answer.substr(1, 2).compare(":A") == 0)
	{
		return DEVICE_OK;
	}
	else if (answer.substr(0, 2).compare(":N") == 0 && answer.length() > 2)
	{
		int errNo = atoi(answer.substr(2, 4).c_str());
		if (errNo == -21)
		{
			return DEVICE_OK;
		}
		else
		{
			return errNo; // ERR_OFFSET + errNo;
		}
	}

	return DEVICE_OK;
}

int XYStage::GetLimitsUm(double& /*xMin*/, double& /*xMax*/, double& /*yMin*/, double& /*yMax*/)
{
	return DEVICE_UNSUPPORTED_COMMAND;
}

int XYStage::GetStepLimits(long& /*xMin*/, long& /*xMax*/, long& /*yMin*/, long& /*yMax*/)
{
	return DEVICE_UNSUPPORTED_COMMAND;
}

bool XYStage::hasCommand(std::string command)
{
	std::string answer;

	// query the device
	int ret = QueryCommand(command.c_str(), answer);
	if (ret != DEVICE_OK)
	{
		return false;
	}

	if (answer.substr(0, 2).compare(":A") == 0)
	{
		return true;
	}
	if (answer.substr(0, 4).compare(":N-1") == 0)
	{
		return false;
	}

	// if we do not get an answer, or any other answer, this is probably OK
	return true;
}


/////////////////////////////////////////////////////////////////////////////////
//// Action handlers
/////////////////////////////////////////////////////////////////////////////////

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

int XYStage::OnStepSizeX(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(stepSizeXUm_);
	}
	return DEVICE_OK;
}

int XYStage::OnStepSizeY(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(stepSizeYUm_);
	}
	return DEVICE_OK;
}

// This sets how often the stage will approach the same position (0 = 1!!)
int XYStage::OnNrMoveRepetitions(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		// some controllers will return this, the current ones do not, so cache
		pProp->Set(nrMoveRepetitions_);
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(nrMoveRepetitions_);
		if (nrMoveRepetitions_ < 0)
		{
			nrMoveRepetitions_ = 0;
		}
		std::ostringstream command;
		command << "CCA Y=" << nrMoveRepetitions_;
		std::string answer;
		// some controller do not answer, so do not check answer
		int ret = SendCommand(command.str().c_str());
		if (ret != DEVICE_OK)
		{
			return ret;
		}
		/*
		// block/wait for acknowledge, or until we time out;
		string answer;
		ret = GetSerialAnswer(port_.c_str(), "\r\n", answer);
		if (ret != DEVICE_OK)
		   return ret;

		if (answer.substr(0,2).compare(":A") == 0)
		   return DEVICE_OK;
		else if (answer.substr(0, 2).compare(":N") == 0 && answer.length() > 2)
		{
		   int errNo = atoi(answer.substr(3).c_str());
		   return ERR_OFFSET + errNo;
		}
		return ERR_UNRECOGNIZED_ANSWER;
		*/
	}
	return DEVICE_OK;
}

// This sets the number of waitcycles
int XYStage::OnWait(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		// To simplify our life we only read out waitcycles for the X axis, but set for both
		std::ostringstream command;
		command << "WT " + axisletterX_ + "?";
		std::string answer;
		// query command
		int ret = QueryCommand(command.str().c_str(), answer);
		if (ret != DEVICE_OK)
		{
			return ret;
		}

		if (answer.substr(0, 2).compare(":X") == 0)
		{
			long waitCycles = 0;
			const int code = ParseResponseAfterPosition(answer, 3, waitCycles);
			pProp->Set(waitCycles);
			return code;
		}
		// deal with error later
		else if (answer.substr(0, 2).compare(":N") == 0 && answer.length() > 2)
		{
			int errNo = atoi(answer.substr(3).c_str());
			return ERR_OFFSET + errNo;
		}
		return ERR_UNRECOGNIZED_ANSWER;
	}
	else if (eAct == MM::AfterSet)
	{
		long waitCycles;
		pProp->Get(waitCycles);

		// enforce positive
		if (waitCycles < 0)
		{
			waitCycles = 0;
		}

		// if firmware date is 2009+  then use msec/int definition of WaitCycles
		// would be better to parse firmware (8.4 and earlier used unsigned char)
		// and that transition occurred ~2008 but not sure exactly when
		if (compileDay_ >= ConvertDay(2009, 1, 1))
		{
			// don't enforce upper limit
		}
		else  // enforce limit for 2008 and earlier firmware or
		{     // if getting compile date wasn't successful
			if (waitCycles > 255)
			{
				waitCycles = 255;
			}
		}

		std::ostringstream command;
		command << "WT " << axisletterX_ << "=" << waitCycles << " " << axisletterY_ << "=" << waitCycles;
		std::string answer;
		// query command
		int ret = QueryCommand(command.str().c_str(), answer);
		if (ret != DEVICE_OK)
		{
			return ret;
		}
		return ResponseStartsWithColonA(answer);
	}
	return DEVICE_OK;
}

int XYStage::OnBacklash(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		// To simplify our life we only read out waitcycles for the X axis, but set for both
		std::ostringstream command;
		command << "B " << axisletterX_ << "?";
		std::string answer;
		// query command
		int ret = QueryCommand(command.str().c_str(), answer);
		if (ret != DEVICE_OK)
		{
			return ret;
		}

		if (answer.substr(0, 2).compare(":X") == 0)
		{
			double speed = 0.0;
			const int code = ParseResponseAfterPosition(answer, 3, 8, speed);
			pProp->Set(speed);
			return code;
		}
		// deal with error later
		else if (answer.substr(0, 2).compare(":N") == 0 && answer.length() > 2)
		{
			int errNo = atoi(answer.substr(3).c_str());
			return ERR_OFFSET + errNo;
		}
		return ERR_UNRECOGNIZED_ANSWER;
	}
	else if (eAct == MM::AfterSet)
	{
		double backlash;
		pProp->Get(backlash);
		if (backlash < 0.0)
		{
			backlash = 0.0;
		}
		std::ostringstream command;
		command << "B " << axisletterX_ << "=" << backlash << " " << axisletterY_ << "=" << backlash;
		std::string answer;

		// query command
		int ret = QueryCommand(command.str().c_str(), answer);
		if (ret != DEVICE_OK)
		{
			return ret;
		}
		return ResponseStartsWithColonA(answer);
	}
	return DEVICE_OK;
}

int XYStage::OnFinishError(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		// To simplify our life we only read out waitcycles for the X axis, but set for both
		std::ostringstream command;
		command << "PC " << axisletterX_ << "?";
		std::string answer;
		// query command
		int ret = QueryCommand(command.str().c_str(), answer);
		if (ret != DEVICE_OK)
		{
			return ret;
		}

		if (answer.substr(0, 2).compare(":X") == 0)
		{
			double finishError = 0.0;
			const int code = ParseResponseAfterPosition(answer, 3, 8, finishError);
			pProp->Set(1000000 * finishError);
			return code;
		}
		if (answer.substr(0, 2).compare(":A") == 0)
		{
			// Answer is of the form :A X=0.00003
			double finishError = 0.0;
			const int code = ParseResponseAfterPosition(answer, 5, 8, finishError);
			pProp->Set(1000000 * finishError);
			return code;
		}
		// deal with error later
		else if (answer.substr(0, 2).compare(":N") == 0 && answer.length() > 2)
		{
			int errNo = atoi(answer.substr(3).c_str());
			return ERR_OFFSET + errNo;
		}
		return ERR_UNRECOGNIZED_ANSWER;
	}
	else if (eAct == MM::AfterSet)
	{
		double error;
		pProp->Get(error);
		if (error < 0.0)
		{
			error = 0.0;
		}
		error = error / 1000000;
		std::ostringstream command;
		command << "PC " << axisletterX_ << "=" << error << " " << axisletterY_ << "=" << error;
		std::string answer;

		// query command
		int ret = QueryCommand(command.str().c_str(), answer);
		if (ret != DEVICE_OK)
		{
			return ret;
		}
		return ResponseStartsWithColonA(answer);
	}
	return DEVICE_OK;
}

int XYStage::OnAcceleration(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		// To simplify our life we only read out acceleration for the X axis, but set for both
		std::ostringstream command;
		command << "AC " << axisletterX_ << "?";
		std::string answer;

		// query command
		int ret = QueryCommand(command.str().c_str(), answer);
		if (ret != DEVICE_OK)
		{
			return ret;
		}

		if (answer.substr(0, 2).compare(":X") == 0)
		{
			double speed = 0.0;
			const int code = ParseResponseAfterPosition(answer, 3, 8, speed);
			pProp->Set(speed);
			return code;
		}
		// deal with error later
		else if (answer.substr(0, 2).compare(":N") == 0 && answer.length() > 2)
		{
			int errNo = atoi(answer.substr(3).c_str());
			return ERR_OFFSET + errNo;
		}
		return ERR_UNRECOGNIZED_ANSWER;
	}
	else if (eAct == MM::AfterSet)
	{
		double accel;
		pProp->Get(accel);
		if (accel < 0.0)
		{
			accel = 0.0;
		}
		std::ostringstream command;
		command << "AC " << axisletterX_ << "=" << accel << " " << axisletterY_ << "=" << accel;
		std::string answer;

		// query command
		int ret = QueryCommand(command.str().c_str(), answer);
		if (ret != DEVICE_OK)
		{
			return ret;
		}
		return ResponseStartsWithColonA(answer);
	}
	return DEVICE_OK;
}

int XYStage::OnOverShoot(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		// To simplify our life we only read out waitcycles for the X axis, but set for both
		std::ostringstream command;
		command << "OS " << axisletterX_ << "?";
		std::string answer;

		// query command
		int ret = QueryCommand(command.str().c_str(), answer);
		if (ret != DEVICE_OK)
		{
			return ret;
		}

		if (answer.substr(0, 2).compare(":A") == 0)
		{
			double overshoot = 0.0;
			const int code = ParseResponseAfterPosition(answer, 5, 8, overshoot);
			pProp->Set(overshoot * 1000.0);
			return code;
		}
		// deal with error later
		else if (answer.substr(0, 2).compare(":N") == 0 && answer.length() > 2)
		{
			int errNo = atoi(answer.substr(3).c_str());
			return ERR_OFFSET + errNo;
		}
		return ERR_UNRECOGNIZED_ANSWER;
	}
	else if (eAct == MM::AfterSet)
	{
		double overShoot;
		pProp->Get(overShoot);
		if (overShoot < 0.0)
		{
			overShoot = 0.0;
		}
		overShoot = overShoot / 1000.0;
		std::ostringstream command;
		command << std::fixed << "OS " << axisletterX_ << "=" << overShoot << " " << axisletterY_ << "=" << overShoot;
		std::string answer;

		// query the device
		int ret = QueryCommand(command.str().c_str(), answer);
		if (ret != DEVICE_OK)
		{
			return ret;
		}
		return ResponseStartsWithColonA(answer);
	}
	return DEVICE_OK;
}

int XYStage::OnError(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		// To simplify our life we only read out waitcycles for the X axis, but set for both
		std::ostringstream command;
		command << "E " << axisletterX_ << "?";
		std::string answer;
		// query command
		int ret = QueryCommand(command.str().c_str(), answer);
		if (ret != DEVICE_OK)
		{
			return ret;
		}

		if (answer.substr(0, 2).compare(":X") == 0)
		{
			double error = 0.0;
			const int code = ParseResponseAfterPosition(answer, 3, 8, error);
			pProp->Set(error * 1000000.0);
			return code;
		}
		// deal with error later
		else if (answer.substr(0, 2).compare(":N") == 0 && answer.length() > 2)
		{
			int errNo = atoi(answer.substr(3).c_str());
			return ERR_OFFSET + errNo;
		}
		return ERR_UNRECOGNIZED_ANSWER;
	}
	else if (eAct == MM::AfterSet)
	{
		double error;
		pProp->Get(error);
		if (error < 0.0)
		{
			error = 0.0;
		}
		error = error / 1000000.0;
		std::ostringstream command;
		command << std::fixed << "E " << axisletterX_ << "=" << error << " " << axisletterY_ << "=" << error;
		std::string answer;

		// query the device
		int ret = QueryCommand(command.str().c_str(), answer);
		if (ret != DEVICE_OK)
		{
			return ret;
		}
		return ResponseStartsWithColonA(answer);
	}
	return DEVICE_OK;
}

int XYStage::GetMaxSpeed(char* maxSpeedStr)
{
	double origMaxSpeed = maxSpeed_;
	char orig_speed[MM::MaxStrLength];
	int ret = GetProperty("Speed-S", orig_speed);
	if (ret != DEVICE_OK)
	{
		return ret;
	}
	maxSpeed_ = 10001;
	SetProperty("Speed-S", "10000");
	ret = GetProperty("Speed-S", maxSpeedStr);
	maxSpeed_ = atof(maxSpeedStr);
	if (maxSpeed_ <= 0.1)
	{
		maxSpeed_ = origMaxSpeed;  // restore default if something went wrong in which case atof returns 0.0
	}
	if (ret != DEVICE_OK)
	{
		return ret;
	}
	ret = SetProperty("Speed-S", orig_speed);
	if (ret != DEVICE_OK)
	{
		return ret;
	}
	return DEVICE_OK;
}

int XYStage::OnSpeed(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		// To simplify our life we only read out waitcycles for the X axis, but set for both
		std::ostringstream command;
		command << "S " << axisletterX_ << "?";
		std::string answer;
		// query command
		int ret = QueryCommand(command.str().c_str(), answer);
		if (ret != DEVICE_OK)
		{
			return ret;
		}

		if (answer.substr(0, 2).compare(":A") == 0)
		{
			double speed = 0.0;
			const int code = ParseResponseAfterPosition(answer, 5, speed);
			pProp->Set(speed);
			return code;
		}
		// deal with error later
		else if (answer.substr(0, 2).compare(":N") == 0 && answer.length() > 2)
		{
			int errNo = atoi(answer.substr(3).c_str());
			return ERR_OFFSET + errNo;
		}
		return ERR_UNRECOGNIZED_ANSWER;
	}
	else if (eAct == MM::AfterSet)
	{
		double speed;
		pProp->Get(speed);
		if (speed < 0.0)
		{
			speed = 0.0;
		}
		// Note, max speed may differ depending on pitch screw
		else if (speed > maxSpeed_)
		{
			speed = maxSpeed_;
		}
		std::ostringstream command;
		command << std::fixed << "S " << axisletterX_ << "=" << speed << " " << axisletterY_ << "=" << speed;
		std::string answer;
		// query the device
		int ret = QueryCommand(command.str().c_str(), answer);
		if (ret != DEVICE_OK)
		{
			return ret;
		}
		return ResponseStartsWithColonA(answer);
	}
	return DEVICE_OK;
}

int XYStage::OnMotorCtrl(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		// The controller can not report whether or not the motors are on.  Cache the value
		if (motorOn_)
		{
			pProp->Set("On");
		}
		else
		{
			pProp->Set("Off");
		}
		return DEVICE_OK;
	}
	else if (eAct == MM::AfterSet)
	{
		std::string motorOn;
		std::string value;
		pProp->Get(motorOn);
		if (motorOn == "On")
		{
			motorOn_ = true;
			value = "+";
		}
		else
		{
			motorOn_ = false;
			value = "-";
		}
		std::ostringstream command;
		command << "MC " << axisletterX_ << value << " " << axisletterY_ << value;
		std::string answer;

		// query command
		int ret = QueryCommand(command.str().c_str(), answer);
		if (ret != DEVICE_OK)
		{
			return ret;
		}
		return ResponseStartsWithColonA(answer);
	}
	return DEVICE_OK;
}

int XYStage::OnJSMirror(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		// TODO: read from device, at least on initialization
		if (joyStickMirror_)
		{
			pProp->Set("On");
		}
		else
		{
			pProp->Set("Off");
		}
		return DEVICE_OK;
	}
	else if (eAct == MM::AfterSet) {
		std::string mirror;
		std::string value;
		pProp->Get(mirror);
		if (mirror == "On")
		{
			if (joyStickMirror_)
			{
				return DEVICE_OK;
			}
			joyStickMirror_ = true;
			value = "-";
		}
		else
		{
			if (!joyStickMirror_)
			{
				return DEVICE_OK;
			}
			joyStickMirror_ = false;
			value = "";
		}
		std::ostringstream command;
		command << "JS X=" << value << joyStickSpeedFast_ << " Y=" << value << joyStickSpeedSlow_; //X and Y psuedo axis not real axis names
		std::string answer;

		// query command
		int ret = QueryCommand(command.str().c_str(), answer);
		if (ret != DEVICE_OK)
		{
			return ret;
		}

		if (answer.substr(0, 2).compare(":A") == 0)
		{
			return DEVICE_OK;
		}
		else if (answer.substr(answer.length(), 1) == "A")
		{
			return DEVICE_OK;
		}
		else if (answer.substr(0, 2).compare(":N") == 0 && answer.length() > 2)
		{
			int errNo = atoi(answer.substr(3).c_str());
			return ERR_OFFSET + errNo;
		}
		else
			return ERR_UNRECOGNIZED_ANSWER;
	}
	return DEVICE_OK;
}

int XYStage::OnJSSwapXY(MM::PropertyBase* /* pProp*/, MM::ActionType /* eAct*/)
{
	return DEVICE_NOT_SUPPORTED;
}

int XYStage::OnJSFastSpeed(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		// TODO: read from device, at least on initialization
		pProp->Set((long)joyStickSpeedFast_);
		return DEVICE_OK;
	}
	else if (eAct == MM::AfterSet)
	{
		long speed;
		pProp->Get(speed);
		joyStickSpeedFast_ = (int)speed;

		std::string value = "";
		if (joyStickMirror_)
		{
			value = "-";
		}

		std::ostringstream command;
		command << "JS X=" << value << joyStickSpeedFast_ << " Y=" << value << joyStickSpeedSlow_;
		std::string answer;
		// query command
		int ret = QueryCommand(command.str().c_str(), answer);
		if (ret != DEVICE_OK)
		{
			return ret;
		}

		if (answer.substr(0, 2).compare(":A") == 0)
		{
			return DEVICE_OK;
		}
		else if (answer.substr(answer.length(), 1) == "A")
		{
			return DEVICE_OK;
		}
		else if (answer.substr(0, 2).compare(":N") == 0 && answer.length() > 2)
		{
			int errNo = atoi(answer.substr(3).c_str());
			return ERR_OFFSET + errNo;
		}
		else
		{
			return ERR_UNRECOGNIZED_ANSWER;
		}
	}
	return DEVICE_OK;
}

int XYStage::OnJSSlowSpeed(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		// TODO: read from device, at least on initialization
		pProp->Set((long)joyStickSpeedSlow_);
		return DEVICE_OK;
	}
	else if (eAct == MM::AfterSet)
	{
		long speed;
		pProp->Get(speed);
		joyStickSpeedSlow_ = (int)speed;

		std::string value = "";
		if (joyStickMirror_)
		{
			value = "-";
		}

		std::ostringstream command;
		command << "JS X=" << value << joyStickSpeedFast_ << " Y=" << value << joyStickSpeedSlow_;
		std::string answer;

		// query command
		int ret = QueryCommand(command.str().c_str(), answer);
		if (ret != DEVICE_OK)
		{
			return ret;
		}

		if (answer.substr(0, 2).compare(":A") == 0)
		{
			return DEVICE_OK;
		}
		else if (answer.substr(answer.length(), 1) == "A")
		{
			return DEVICE_OK;
		}
		else if (answer.substr(0, 2).compare(":N") == 0 && answer.length() > 2)
		{
			int errNo = atoi(answer.substr(3).c_str());
			return ERR_OFFSET + errNo;
		}
		else
		{
			return ERR_UNRECOGNIZED_ANSWER;
		}
	}
	return DEVICE_OK;
}

// use the peculiar fact that the info command is the only Tiger command
// that begins with the letter I.  So isolate for the actual command
// (stripping card address and leading whitespace) and then see if the
// first character is an "I" (not case sensitive)
bool isINFOCommand(const std::string command)
{
	bool ret = false;
	try
	{
		ret = toupper(command.at(command.find_first_not_of(" 0123456789"))) == 'I';
	}
	catch (...)
	{
	}
	return ret;
}

int XYStage::OnSerialCommand(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		// do nothing
	}
	else if (eAct == MM::AfterSet)
	{
		static std::string last_command_via_property;
		std::string tmpstr;
		pProp->Get(tmpstr);
		tmpstr = UnescapeControlCharacters(tmpstr);
		// only send the command if it has been updated, or if the feature has been set to "no"/false then always send
		if (!serialOnlySendChanged_ || (tmpstr.compare(last_command_via_property) != 0))
		{
			// prevent executing the INFO command
			if (isINFOCommand(tmpstr))
			{
				return ERR_INFO_COMMAND_NOT_SUPPORTED;
			}

			last_command_via_property = tmpstr;
			int ret = QueryCommand(tmpstr.c_str(), manualSerialAnswer_);
			if (ret != DEVICE_OK)
			{
				return ret;
			}
		}
	}
	return DEVICE_OK;
}

int XYStage::OnSerialResponse(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet || eAct == MM::AfterSet)
	{
		// always read
		if (!pProp->Set(EscapeControlCharacters(manualSerialAnswer_).c_str()))
		{
			return DEVICE_INVALID_PROPERTY_VALUE;
		}
	}
	return DEVICE_OK;
}

int XYStage::OnSerialCommandOnlySendChanged(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	std::string tmpstr;
	if (eAct == MM::AfterSet) {
		pProp->Get(tmpstr);
		if (tmpstr.compare("Yes") == 0)
		{
			serialOnlySendChanged_ = true;
		}
		else
		{
			serialOnlySendChanged_ = false;
		}
	}
	return DEVICE_OK;
}

// special property, when set to "yes" it creates a set of little-used properties that can be manipulated thereafter
// these parameters exposed with some hurdle to user: KP, KI, KD, AA
int XYStage::OnAdvancedProperties(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		return DEVICE_OK; // do nothing
	}
	else if (eAct == MM::AfterSet)
	{
		std::string tmpstr;
		pProp->Get(tmpstr);
		if ((tmpstr.compare("Yes") == 0) && !advancedPropsEnabled_) // after creating advanced properties once no need to repeat
		{
			CPropertyAction* pAct;
			advancedPropsEnabled_ = true;

			// overshoot (OS)  // in Nico's original

			// servo integral term (KI)
			if (hasCommand("KI " + axisletterX_ + "?"))
			{
				pAct = new CPropertyAction(this, &XYStage::OnKIntegral);
				CreateProperty("ServoIntegral-KI", "0", MM::Integer, false, pAct);
			}

			// servo proportional term (KP)
			if (hasCommand("KP " + axisletterX_ + "?"))
			{
				pAct = new CPropertyAction(this, &XYStage::OnKProportional);
				CreateProperty("ServoProportional-KP", "0", MM::Integer, false, pAct);
			}

			// servo derivative term (KD)
			if (hasCommand("KD " + axisletterX_ + "?"))
			{
				pAct = new CPropertyAction(this, &XYStage::OnKDerivative);
				CreateProperty("ServoIntegral-KD", "0", MM::Integer, false, pAct);
			}

			// Align calibration/setting for pot in drive electronics (AA)
			if (hasCommand("AA " + axisletterX_ + "?"))
			{
				pAct = new CPropertyAction(this, &XYStage::OnAAlign);
				CreateProperty("MotorAlign-AA", "0", MM::Integer, false, pAct);
			}

			// Autozero drive electronics (AZ)  // omitting for now, need to do for each axis (see Tiger)
			// number of extra move repetitions  // in Nico's original
		}
	}
	return DEVICE_OK;
}

int XYStage::OnKIntegral(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	std::ostringstream command;
	std::ostringstream response;
	command.str("");
	response.str("");
	long tmp = 0;
	if (eAct == MM::BeforeGet)
	{
		command << "KI " << axisletterX_ << "?";
		std::string answer;
		int ret = QueryCommand(command.str().c_str(), answer);
		if (ret != DEVICE_OK)
		{
			return ret;
		}

		if (answer.substr(0, 2).compare(":A") == 0)
		{
			ParseResponseAfterPosition(answer, 5, tmp);
			if (!pProp->Set(tmp))
			{
				return DEVICE_INVALID_PROPERTY_VALUE;
			}
			else
			{
				return DEVICE_OK;
			}
		}
		// deal with error later
		else if (answer.substr(0, 2).compare(":N") == 0 && answer.length() > 2)
		{
			int errNo = atoi(answer.substr(3).c_str());
			return ERR_OFFSET + errNo;
		}
		return ERR_UNRECOGNIZED_ANSWER;
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(tmp);
		command << "KI " << axisletterX_ << "=" << tmp << " " << axisletterY_ << "=" << tmp;
		std::string answer;
		int ret = QueryCommand(command.str().c_str(), answer);
		if (ret != DEVICE_OK)
		{
			return ret;
		}
		return ResponseStartsWithColonA(answer);
	}
	return DEVICE_OK;
}

int XYStage::OnKProportional(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	std::ostringstream command;
	std::ostringstream response;
	command.str("");
	response.str("");

	long tmp = 0;
	if (eAct == MM::BeforeGet)
	{
		command << "KP " << axisletterX_ << "?";
		std::string answer;
		int ret = QueryCommand(command.str().c_str(), answer);
		if (ret != DEVICE_OK)
		{
			return ret;
		}
		if (answer.substr(0, 2).compare(":A") == 0)
		{
			ParseResponseAfterPosition(answer, 5, tmp);
			if (!pProp->Set(tmp))
			{
				return DEVICE_INVALID_PROPERTY_VALUE;
			}
			else
			{
				return DEVICE_OK;
			}
		}
		// deal with error later
		else if (answer.substr(0, 2).compare(":N") == 0 && answer.length() > 2)
		{
			int errNo = atoi(answer.substr(3).c_str());
			return ERR_OFFSET + errNo;
		}
		return ERR_UNRECOGNIZED_ANSWER;
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(tmp);
		command << "KP " << axisletterX_ << "=" << tmp << " " << axisletterY_ << "=" << tmp;
		std::string answer;
		int ret = QueryCommand(command.str().c_str(), answer);
		if (ret != DEVICE_OK)
		{
			return ret;
		}
		return ResponseStartsWithColonA(answer);
	}
	return DEVICE_OK;
}

int XYStage::OnKDerivative(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	std::ostringstream command;
	std::ostringstream response;
	command.str("");
	response.str("");

	long tmp = 0;
	if (eAct == MM::BeforeGet)
	{
		command << "KD " << axisletterX_ << "?";
		std::string answer;
		int ret = QueryCommand(command.str().c_str(), answer);
		if (ret != DEVICE_OK)
		{
			return ret;
		}
		if (answer.substr(0, 2).compare(":A") == 0)
		{
			ParseResponseAfterPosition(answer, 5, tmp);
			if (!pProp->Set(tmp))
			{
				return DEVICE_INVALID_PROPERTY_VALUE;
			}
			else
			{
				return DEVICE_OK;
			}
		}
		// deal with error later
		else if (answer.substr(0, 2).compare(":N") == 0 && answer.length() > 2)
		{
			int errNo = atoi(answer.substr(3).c_str());
			return ERR_OFFSET + errNo;
		}
		return ERR_UNRECOGNIZED_ANSWER;
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(tmp);
		command << "KD " << axisletterX_ << "=" << tmp << " " << axisletterY_ << "=" << tmp;
		std::string answer;
		int ret = QueryCommand(command.str().c_str(), answer);
		if (ret != DEVICE_OK)
		{
			return ret;
		}
		return ResponseStartsWithColonA(answer);
	}
	return DEVICE_OK;
}

int XYStage::OnAAlign(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	std::ostringstream command;
	std::ostringstream response;
	command.str("");
	response.str("");

	long tmp = 0;
	if (eAct == MM::BeforeGet)
	{
		command << "AA " << axisletterX_ << "?";
		std::string answer;
		int ret = QueryCommand(command.str().c_str(), answer);
		if (ret != DEVICE_OK)
		{
			return ret;
		}
		if (answer.substr(0, 2).compare(":A") == 0)
		{
			ParseResponseAfterPosition(answer, 5, tmp);
			if (!pProp->Set(tmp))
			{
				return DEVICE_INVALID_PROPERTY_VALUE;
			}
			else
			{
				return DEVICE_OK;
			}
		}
		// deal with error later
		else if (answer.substr(0, 2).compare(":N") == 0 && answer.length() > 2)
		{
			int errNo = atoi(answer.substr(3).c_str());
			return ERR_OFFSET + errNo;
		}
		return ERR_UNRECOGNIZED_ANSWER;
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(tmp);
		command << "AA " << axisletterX_ << "=" << tmp << " " << axisletterY_ << "=" << tmp;
		std::string answer;
		int ret = QueryCommand(command.str().c_str(), answer);
		if (ret != DEVICE_OK)
		{
			return ret;
		}
		return ResponseStartsWithColonA(answer);
	}
	return DEVICE_OK;
}

int XYStage::GetPositionStepsSingle(char /*axis*/, long& /*steps*/)
{
	return ERR_UNRECOGNIZED_ANSWER;
}

int XYStage::SetAxisDirection()
{
	std::ostringstream command;
	command << "UM " << axisletterX_ << "=-10000 " << axisletterY_ << "=10000";
	std::string answer = "";
	// query command
	int ret = QueryCommand(command.str().c_str(), answer);
	if (ret != DEVICE_OK)
	{
		return ret;
	}
	return ResponseStartsWithColonA(answer);
	/*
	//ASI XY Stage positive limit is top-right, however micromanager convention is top-left
	//so we reverse X axis direction
	XYStage::SetProperty(MM::g_Keyword_Transpose_MirrorX,"1" );
	XYStage::SetProperty(MM::g_Keyword_Transpose_MirrorY,"0" );
	return DEVICE_OK;
	*/
}

// based on similar function in FreeSerialPort.cpp
std::string XYStage::EscapeControlCharacters(const std::string v)
{
	std::ostringstream mess;
	mess.str("");
	for (std::string::const_iterator ii = v.begin(); ii != v.end(); ++ii)
	{
		if (*ii > 31)
		{
			mess << *ii;
		}
		else if (*ii == 13)
		{
			mess << "\\r";
		}
		else if (*ii == 10)
		{
			mess << "\\n";
		}
		else if (*ii == 9)
		{
			mess << "\\t";
		}
		else
		{
			mess << "\\" << (unsigned int)(*ii);
		}
	}
	return mess.str();
}

// based on similar function in FreeSerialPort.cpp
std::string XYStage::UnescapeControlCharacters(const std::string v0)
{
	// the string input from the GUI can contain escaped control characters, currently these are always preceded with \ (0x5C)
	// and always assumed to be decimal or C style, not hex

	std::string detokenized;
	std::string v = v0;

	for (std::string::iterator jj = v.begin(); jj != v.end(); ++jj)
	{
		bool breakNow = false;
		if ('\\' == *jj)
		{
			// the next 1 to 3 characters might be converted into a control character
			++jj;
			if (v.end() == jj)
			{
				// there was an escape at the very end of the input string so output it literally
				detokenized.push_back('\\');
				break;
			}
			const std::string::iterator nextAfterEscape = jj;
			std::string thisControlCharacter;
			// take any decimal digits immediately after the escape character and convert to a control character
			while (0x2F < *jj && *jj < 0x3A)
			{
				thisControlCharacter.push_back(*jj++);
				if (v.end() == jj)
				{
					breakNow = true;
					break;
				}
			}
			int code = -1;
			if (0 < thisControlCharacter.length())
			{
				std::istringstream tmp(thisControlCharacter);
				tmp >> code;
			}
			// otherwise, if we are still at the first character after the escape,
			// possibly treat the next character like a 'C' control character
			if (nextAfterEscape == jj)
			{
				switch (*jj)
				{
				case 'r':
					++jj;
					code = 13; // CR \r
					break;
				case 'n':
					++jj;
					code = 10; // LF \n
					break;
				case 't':
					++jj;
					code = 9; // TAB \t
					break;
				case '\\':
					++jj;
					code = '\\';
					break;
				default:
					code = '\\'; // the '\' wasn't really an escape character....
					break;
				}
				if (v.end() == jj)
				{
					breakNow = true;
				}
			}
			if (-1 < code)
			{
				detokenized.push_back((char)code);
			}
		}
		if (breakNow)
		{
			break;
		}
		detokenized.push_back(*jj);
	}
	return detokenized;
}

int XYStage::OnVectorGeneric(MM::PropertyBase* pProp, MM::ActionType eAct, std::string axisLetter) {
	if (eAct == MM::BeforeGet)
	{
		std::ostringstream command;
		command << "VE " + axisLetter + "?";
		std::string answer;
		// query command
		int ret = QueryCommand(command.str().c_str(), answer);
		if (ret != DEVICE_OK)
		{
			return ret;
		}

		// if (answer.substr(0,2).compare(":" + axisLetter) == 0)
		if (answer.substr(0, 5).compare(":A " + axisLetter + "=") == 0)
		{
			double speed = 0.0;
			const int code = ParseResponseAfterPosition(answer, 6, 13, speed);
			pProp->Set(speed);
			return code;
		}
		// deal with error later
		else if (answer.substr(0, 2).compare(":N") == 0 && answer.length() > 2)
		{
			int errNo = atoi(answer.substr(3).c_str());
			return ERR_OFFSET + errNo;
		}
		return ERR_UNRECOGNIZED_ANSWER;
	}
	else if (eAct == MM::AfterSet)
	{
		double vector;
		pProp->Get(vector);

		std::ostringstream command;
		command << "VE " << axisLetter << "=" << vector;
		std::string answer;
		// query command
		int ret = QueryCommand(command.str().c_str(), answer);
		if (ret != DEVICE_OK)
		{
			return ret;
		}
		return ResponseStartsWithColonA(answer);
	}
	return DEVICE_OK;
}
