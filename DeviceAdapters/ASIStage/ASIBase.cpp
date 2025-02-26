/*
 * Project: ASIStage Device Adapter
 * License/Copyright: BSD 3-clause, see license.txt
 * Maintainers: Brandon Simpson (brandon@asiimaging.com)
 *              Jon Daniels (jon@asiimaging.com)
 */

#include "ASIBase.h"

ASIBase::ASIBase(MM::Device* device, const char* prefix) :
	oldstage_(false),
	initialized_(false),
	core_(nullptr),
	device_(device),
	port_("Undefined"),
	version_("Undefined"),
	buildName_("Undefined"),
	compileDate_("Undefined"),
	oldstagePrefix_(prefix),
	versionData_(VersionData()),
	compileDay_(0)
{
}

ASIBase::~ASIBase()
{
}

// Communication "clear buffer" utility function:
int ASIBase::ClearPort()
{
	// Clear contents of serial port
	const int bufSize = 255;
	unsigned char clear[bufSize];
	unsigned long read = bufSize;
	int ret;
	while ((int)read == bufSize)
	{
		ret = core_->ReadFromSerial(device_, port_.c_str(), clear, bufSize, read);
		if (ret != DEVICE_OK)
		{
			return ret;
		}
	}
	return DEVICE_OK;
}

// Communication "send" utility function:
int ASIBase::SendCommand(const char* command) const
{
	std::string base_command = "";
	if (oldstage_)
	{
		base_command += oldstagePrefix_;
	}
	base_command += command;
	// send command
	return core_->SetSerialCommand(device_, port_.c_str(), base_command.c_str(), "\r");
}

// Communication "send & receive" utility function:
int ASIBase::QueryCommand(const char* command, std::string& answer) const
{
	const char* terminator;

	// send command
	int ret = SendCommand(command);
	if (ret != DEVICE_OK)
	{
		return ret;
	}
	// block/wait for acknowledge (or until we time out)
	if (oldstage_)
	{
		terminator = "\r\n\3";
	}
	else
	{
		terminator = "\r\n";
	}

	const size_t BUFSIZE = 2048;
	char buf[BUFSIZE] = { '\0' };
	ret = core_->GetSerialAnswer(device_, port_.c_str(), BUFSIZE, buf, terminator);
	answer = buf;

	return ret;
}

// Communication "send, receive, and look for acknowledgement" utility function:
int ASIBase::QueryCommandACK(const char* command) const
{
	std::string answer;
	int ret = QueryCommand(command, answer);
	if (ret != DEVICE_OK)
	{
		return ret;
	}
	// the controller only acknowledges receipt of the command
	if (answer.substr(0, 2) != ":A")
	{
		return ERR_UNRECOGNIZED_ANSWER;
	}
	return DEVICE_OK;
}

// Communication "test device type" utility function:
// Set the value of oldstage_ to true for LX-4000, false for MS-2000.
int ASIBase::CheckDeviceStatus()
{
	// send status command (test for new protocol)
	oldstage_ = false;
	std::string answer;
	int ret = QueryCommand("/", answer);
	if (ret != DEVICE_OK && !oldstagePrefix_.empty())
	{
		// send status command (test for older LX-4000 protocol)
		oldstage_ = true;
		ret = QueryCommand("/", answer);
	}
	return ret;
}

unsigned int ASIBase::ConvertDay(int year, int month, int day) const
{
	return day + 31 * (month - 1) + 372 * (year - 2000);
}

unsigned int ASIBase::ExtractCompileDay(const char* compile_date) const
{
	const char* months = "anebarprayunulugepctovec";
	const size_t compile_date_len = strlen(compile_date);
	if (compile_date_len < 11)
	{
		return 0;
	}
	int year = 0;
	int month = 0;
	int day = 0;
	if (compile_date_len >= 11
		&& compile_date[7] == '2'  // must be 20xx for sanity checking
		&& compile_date[8] == '0'
		&& compile_date[9] <= '9'
		&& compile_date[9] >= '0'
		&& compile_date[10] <= '9'
		&& compile_date[10] >= '0')
	{
		year = 2000 + 10 * (compile_date[9] - '0') + (compile_date[10] - '0');
		// look for the year based on the last two characters of the abbreviated month name
		month = 1;
		for (int i = 0; i < 12; i++)
		{
			if (compile_date[1] == months[2 * i] && compile_date[2] == months[2 * i + 1])
			{
				month = i + 1;
			}
		}
		day = 10 * (compile_date[4] - '0') + (compile_date[5] - '0');
		if (day < 1 || day > 31)
		{
			day = 1;
		}
		return ConvertDay(year, month, day);
	}
	return 0;
}

VersionData ASIBase::ExtractVersionData(const std::string& version) const
{	
	// Version command response examples:
	// Example A) ":A Version: USB-9.2p \r\n"
	// Example B) ":A Version: USB-9.50 \r\n"
	const size_t startIndex = version.find("-");
	if (startIndex == std::string::npos)
	{
		return VersionData(); // error => default data
	}

	// shortVersion => "9.2m \r\n"
	const std::string shortVersion = version.substr(startIndex + 1);

	// find the index of the dot that separates major and minor version
	const size_t dotIndex = shortVersion.find(".");
	if (dotIndex == std::string::npos)
	{
		return VersionData(); // error => default data
	}
	
	// use substr for major versions with more than 1 digit, ##.## for example
	const int major = std::stoi(shortVersion.substr(0, dotIndex));

	// minor version and revision will only ever be 1 character,
	// at these specific locations after the dot in the response
	const int minor = std::stoi(shortVersion.substr(dotIndex + 1, 1));
	const char revision = shortVersion.at(dotIndex + 2);
	return VersionData(major, minor, revision);
}

int ASIBase::GetVersion(std::string& version)
{
   std::string answer;
   int ret = QueryCommand("V", answer);
   if (ret != DEVICE_OK)
   {
      return ret;
   }
   if (answer.substr(0, 2).compare(":A") == 0)
   {
		version = answer.substr(3);
		return DEVICE_OK;
   }
   return ERR_UNRECOGNIZED_ANSWER;
}

// Get the version of this controller
int ASIBase::OnVersion(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
      pProp->Set(version_.c_str());
	}
	return DEVICE_OK;
}

int ASIBase::GetBuildName(std::string& buildName)
{
	std::string answer;
	int ret = QueryCommand("BU", answer);
	if (ret != DEVICE_OK)
	{
		return ret;
	}
	buildName = answer;
	return DEVICE_OK;
}

// Get the build name of this controller
int ASIBase::OnBuildName(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(buildName_.c_str());
	}
	return DEVICE_OK;
}

int ASIBase::GetCompileDate(std::string& buildName)
{
	std::string answer;
	int ret = QueryCommand("CD", answer);
	if (ret != DEVICE_OK)
	{
		return ret;
	}
	buildName = answer;
	return DEVICE_OK;
}

// Get the compile date of this controller
int ASIBase::OnCompileDate(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(compileDate_.c_str());
	}
	return DEVICE_OK;
}

// long version
int ASIBase::ParseResponseAfterPosition(const std::string& answer, const unsigned int position, long& value) const
{
	// specify position as 3 to parse skipping the first 3 characters, e.g. for ":A 45.1"
	if (position >= answer.length())
	{
		return ERR_UNRECOGNIZED_ANSWER;
	}
	value = atol(answer.substr(position).c_str());
	return DEVICE_OK;
}

// double version
int ASIBase::ParseResponseAfterPosition(const std::string& answer, const unsigned int position, double& value) const
{
	// specify position as 3 to parse skipping the first 3 characters, e.g. for ":A 45.1"
	if (position >= answer.length())
	{
		return ERR_UNRECOGNIZED_ANSWER;
	}
	value = atof(answer.substr(position).c_str());
	return DEVICE_OK;
}

// double version + count for substr
int ASIBase::ParseResponseAfterPosition(const std::string& answer, const unsigned int position, const unsigned int count, double& value) const
{
	// specify position as 3 to parse skipping the first 3 characters, e.g. for ":A 45.1"
	if (position >= answer.length())
	{
		return ERR_UNRECOGNIZED_ANSWER;
	}
	value = atof(answer.substr(position, count).c_str());
	return DEVICE_OK;
}

// Returns DEVICE_OK if the response string starts with ":A" otherwise it returns the error code.
int ASIBase::ResponseStartsWithColonA(const std::string& answer) const
{
	if (answer.length() < 2)
	{
		return ERR_UNRECOGNIZED_ANSWER;
	}
	if (answer.substr(0, 2).compare(":A") == 0)
	{
		return DEVICE_OK;
	}
	else if (answer.substr(0, 2).compare(":N") == 0 && answer.length() > 2)
	{
		const int errorNumber = atoi(answer.substr(3).c_str());
		return ERR_OFFSET + errorNumber;
	}
	return ERR_UNRECOGNIZED_ANSWER;
}
