/*
 * Project: ASIStage Device Adapter
 * License/Copyright: BSD 3-clause, see license.txt
 * Maintainers: Brandon Simpson (brandon@asiimaging.com)
 *              Jon Daniels (jon@asiimaging.com)
 */

#include "ASILED.h"

LED::LED() :
	ASIBase(this, ""), // LX-4000 Prefix Unknown
	open_(false),
	intensity_(20),
	name_("LED"),
	answerTimeoutMs_(1000),
	channel_(0),
	channelAxisChar_('X'),
	hasDLED_(false)
{
	InitializeDefaultErrorMessages();

	// create pre-initialization properties
	// ------------------------------------

	// Name
	CreateProperty(MM::g_Keyword_Name, g_LEDDeviceName, MM::String, true);

	// Description
	CreateProperty(MM::g_Keyword_Description, g_LEDDeviceDescription, MM::String, true);

	// Port
	CPropertyAction* pAct = new CPropertyAction(this, &LED::OnPort);
	CreateProperty(MM::g_Keyword_Port, "Undefined", MM::String, false, pAct, true);

	// MS2000 supports multiple channels now
	pAct = new CPropertyAction(this, &LED::OnChannel);
	CreateProperty("Channel", "0", MM::Integer, false, pAct, true);
	AddAllowedValue("Channel", "0");
	AddAllowedValue("Channel", "1");
	AddAllowedValue("Channel", "2");
	AddAllowedValue("Channel", "3");
}

LED::~LED()
{
	Shutdown();
}

void LED::GetName(char* Name) const
{
	CDeviceUtils::CopyLimitedString(Name, g_LEDDeviceName);
}


bool LED::SupportsDeviceDetection(void)
{
	return true;
}

MM::DeviceDetectionStatus LED::DetectDevice(void)
{
	return ASICheckSerialPort(*this, *GetCoreCallback(), port_, answerTimeoutMs_);
}

int LED::Initialize()
{
	core_ = GetCoreCallback();

	// empty the Rx serial buffer before sending command
	ClearPort();

	// check status first (test for communication protocol)
	int ret = CheckDeviceStatus();
	if (ret != DEVICE_OK)
		return ret;

	CPropertyAction* pAct = new CPropertyAction(this, &LED::OnVersion);
	CreateProperty("Version", "", MM::String, true, pAct);

	pAct = new CPropertyAction(this, &LED::OnCompileDate);
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
		pAct = new CPropertyAction(this, &LED::OnBuildName);
		CreateProperty("BuildName", "", MM::String, true, pAct);
		UpdateProperty("BuildName");
	}

	// not needed SetOpen and GetOpen do the same job. 
	/*
	CPropertyAction* pAct = new CPropertyAction (this, &LED::OnState);
	CreateProperty(MM::g_Keyword_State, g_Closed, MM::String, false, pAct);
	AddAllowedValue(MM::g_Keyword_State, g_Closed);
	AddAllowedValue(MM::g_Keyword_State, g_Open);
	*/
	pAct = new CPropertyAction(this, &LED::OnIntensity);
	CreateProperty("Intensity", "20", MM::Integer, false, pAct);
	SetPropertyLimits("Intensity", 0, 100);

	// Check if LED is DLED
	std::string answer;
	ret = QueryCommand("BU X", answer);
	if (ret != DEVICE_OK)
	{
		return ret;
	}

	std::istringstream is(answer);
	std::string token;
	while (getline(is, token, '\r'))
	{
		std::string dled = "DLED";
		if (0 == token.compare(0, dled.size(), dled))
		{
			hasDLED_ = true;
		}
	}

	ret = CurrentIntensity(&intensity_);
	if (ret != DEVICE_OK)
	{
		return ret;
	}
	if (hasDLED_ || (channel_ != 0))
	{
		// we can figure out if LED is ON or OFF from intensity_ value

		if (intensity_ > 0)
		{
			open_ = true;
		}
		else
		{
			open_ = false;
		}
	}
	else
	{
		// however if it isn't DLED, then we need to check TTL Y 
		ret = IsOpen(&open_);
		if (ret != DEVICE_OK)
		{
			return ret;
		}
	}

	initialized_ = true;
	return DEVICE_OK;
}

int LED::Shutdown()
{
	initialized_ = false;
	return DEVICE_OK;
}

// the LED should be a whole lot faster than our serial communication so always respond false
bool LED::Busy()
{
	return false;
}

// Shutter API
// All communication with the LED takes place in this function
int LED::SetOpen(bool open)
{
	// empty the Rx serial buffer before sending command
	ClearPort();

	std::ostringstream command;

	if ((!hasDLED_) & (channel_ == 0)) {
		//On Old Regulator LED , we turn the TTL mode itself on and off to reduce flicker
		if (open)
		{
			if (intensity_ == 100)
			{
				command << "TTL Y=1";
			}
			else
			{
				command << std::fixed << "TTL Y=9 " << intensity_;
			}
		}
		else
		{
			command << "TTL Y=0";
		}
	}
	else
	{
		// if DLED or other full on and off LEDs, use LED command 
		if (open)
		{
			command << "LED " << channelAxisChar_ << "=" << intensity_;
		}
		else
		{
			command << "LED " << channelAxisChar_ << "=0";
		}
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
		open_ = open;
		return DEVICE_OK;
	}
	// deal with error later
	else if (answer.substr(0, 2).compare(":N") == 0 && answer.length() > 2)
	{
		int errNo = atoi(answer.substr(4).c_str());
		return ERR_OFFSET + errNo;
	}

	open_ = open;
	return DEVICE_OK;
}

/**
 * GetOpen returns a cached value. If ASI ever gives another control to the TTL out
 * other than the serial interface, this will need to be changed to a call to IsOpen
 */
int LED::GetOpen(bool& open)
{
	open = open_;
	return DEVICE_OK;
}

int LED::IsOpen(bool* open)
{
	*open = true;

	// empty the Rx serial buffer before sending command
	ClearPort();
	std::ostringstream command;
	if ((!hasDLED_) & (channel_ == 0)) {

		command << "TTL Y?";

		std::string answer;
		// query the device
		int ret = QueryCommand(command.str().c_str(), answer);
		if (ret != DEVICE_OK)
		{
			return ret;
		}

		if ((answer.substr(0, 2).compare(":A") == 0) || (answer.substr(1, 2).compare(":A") == 0))
		{
			if (answer.substr(2, 1) == "0")
			{
				*open = false;
			}
		}
		else if (answer.substr(0, 2).compare(":N") == 0 && answer.length() > 2)
		{
			int errNo = atoi(answer.substr(4).c_str());
			return ERR_OFFSET + errNo;
		}
		return DEVICE_OK;
	}
	else
	{
		/*
		// Query the LED command
		command << "LED " << channelAxisChar_ << "?";

		string answer;
		// query the device
		int ret = QueryCommand(command.str().c_str(), answer);
		if (ret != DEVICE_OK)
			return ret;

		// Command "LED X?" return "X=0 :A"
		if (answer.substr(0, 1)[0]==channelAxisChar_) {
			if (answer.substr(2, 1) == "0")
				*open = false;
		}
		else if (answer.substr(0, 2).compare(":N") == 0 && answer.length() > 2)
		{
			int errNo = atoi(answer.substr(4).c_str());
			return ERR_OFFSET + errNo;
		}
		return DEVICE_OK;
		*/

		// figure out if shutter is open or close from led x? value
		int ret;
		long curr_intensity;

		ret = CurrentIntensity(&curr_intensity);

		if (ret != DEVICE_OK)
		{
			return ret;
		}

		if (curr_intensity > 0)
		{
			*open = true;
		}
		else
		{
			*open = false;
		}

		return ret;
	}
}

int LED::CurrentIntensity(long* intensity)
{
	*intensity = 1;

	// empty the Rx serial buffer before sending command
	ClearPort();

	std::ostringstream command;
	command << "LED " << channelAxisChar_ << "?";

	std::string answer;
	// query the device
	int ret = QueryCommand(command.str().c_str(), answer);
	if (ret != DEVICE_OK)
	{
		return ret;
	}

	std::istringstream is(answer);
	std::string tok;
	std::string tok2;
	is >> tok;
	is >> tok2;
	if ((tok2.substr(0, 2).compare(":A") == 0) || (tok2.substr(1, 2).compare(":A") == 0))
	{
		*intensity = atoi(tok.substr(2).c_str());
	}
	else if (tok.substr(0, 2).compare(":N") == 0 && tok.length() > 2)
	{
		int errNo = atoi(tok.substr(4).c_str());
		return ERR_OFFSET + errNo;
	}
	return DEVICE_OK;
}

int LED::Fire(double)
{
	return DEVICE_OK;
}

/////////////////////////////////////////////////////////////////////////////////
//// Action handlers
/////////////////////////////////////////////////////////////////////////////////

int LED::OnIntensity(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(intensity_);
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(intensity_);
		// We could check that 0 < intensity_ < 101, but the system should guarantee that
		/* if (intensity_ < 100 & open_)
		{
			ClearPort();

			ostringstream command;
			command << "LED " << channelAxisChar_ <<"=";
			command << intensity_;

			string answer;
			// query the device
			int ret = QueryCommand(command.str().c_str(), answer);
			if (ret != DEVICE_OK)
				return ret;

			std::istringstream is(answer);
			std::string tok;
			is >> tok;
			if (tok.substr(0, 2).compare(":N") == 0 && tok.length() > 2)
			{
				int errNo = atoi(tok.substr(4).c_str());
				return ERR_OFFSET + errNo;
			}

		}
		*/
		if (open_)
		{
			return SetOpen(open_);
		}
	}
	return DEVICE_OK;
}

int LED::OnPort(MM::PropertyBase* pProp, MM::ActionType eAct)
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

int LED::OnChannel(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(channel_);
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(channel_);

		// select axis character to use
		switch (channel_)
		{
		case 1:
			channelAxisChar_ = 'Y';
			break;
		case 2:
			channelAxisChar_ = 'Z';
			break;
		case 3:
			channelAxisChar_ = 'F';
			break;
		case 4:
			channelAxisChar_ = 'T';
			break;
		case 5:
			channelAxisChar_ = 'R';
			break;
		case 0:
		default:
			channelAxisChar_ = 'X';
			break;
		}
	}
	return DEVICE_OK;
}

/*
// not needed SetOpen and GetOpen does the same job
int LED::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
	  std::string state = g_Open;
	  if (!open_)
		 state = g_Closed;
	  pProp->Set(state.c_str());
   }
   else if (eAct == MM::AfterSet)
   {
	  bool open = true;
	  std::string state;
	  pProp->Get(state);
	  if (state == g_Closed)
		 open = false;
	  return SetOpen(open);
   }

   return DEVICE_OK;
}
*/
