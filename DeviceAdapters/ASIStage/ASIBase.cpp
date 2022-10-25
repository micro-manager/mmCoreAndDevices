/*
 * Project: ASIStage Device Adapter
 * License/Copyright: BSD 3-clause, see license.txt
 * Maintainers: Brandon Simpson (brandon@asiimaging.com)
 *              Jon Daniels (jon@asiimaging.com)
 */

#include "ASIBase.h"

ASIBase::ASIBase(MM::Device* device, const char* prefix) :
	oldstage_(false),
	core_(0),
	compileDay_(0),
	initialized_(false),
	device_(device),
	oldstagePrefix_(prefix),
	port_("Undefined")
{
	versionData_ = VersionData();
}

ASIBase::~ASIBase()
{
}

// Communication "clear buffer" utility function:
int ASIBase::ClearPort(void)
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
	int ret;

	if (oldstage_)
	{
		base_command += oldstagePrefix_;
	}
	base_command += command;
	// send command
	ret = core_->SetSerialCommand(device_, port_.c_str(), base_command.c_str(), "\r");
	return ret;
}

// Communication "send & receive" utility function:
int ASIBase::QueryCommand(const char* command, std::string& answer) const
{
	const char* terminator;
	int ret;

	// send command
	ret = SendCommand(command);
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
int ASIBase::QueryCommandACK(const char* command)
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
int ASIBase::CheckDeviceStatus(void)
{
	const char* command = "/"; // check STATUS
	std::string answer;
	int ret;

	// send status command (test for new protocol)
	oldstage_ = false;
	ret = QueryCommand(command, answer);
	if (ret != DEVICE_OK && !oldstagePrefix_.empty())
	{
		// send status command (test for older LX-4000 protocol)
		oldstage_ = true;
		ret = QueryCommand(command, answer);
	}
	return ret;
}

unsigned int ASIBase::ConvertDay(int year, int month, int day)
{
	return day + 31 * (month - 1) + 372 * (year - 2000);
}

unsigned int ASIBase::ExtractCompileDay(const char* compile_date)
{
	const char* months = "anebarprayunulugepctovec";
	if (strlen(compile_date) < 11)
	{
		return 0;
	}
	int year = 0;
	int month = 0;
	int day = 0;
	if (strlen(compile_date) >= 11
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

VersionData ASIBase::ExtractVersionData(const std::string &version) const
{	
	// Version response example: ":A Version: USB-9.2m \r\n"
	size_t startIndex = version.find("-");
	if (startIndex == std::string::npos)
	{
		return VersionData(); // error => default data
	}
	std::string shortVersion = version.substr(startIndex+1);
	// shortVersion => "9.2m \r\n"

	// extract revision letter
	int revIndex = 0;
	char revision = '-';
	for (int i = 0; i < shortVersion.size(); i++)
	{
		char c = shortVersion[i];
		if (std::isalpha(c))
		{
			revIndex = i; // index
			revision = c; // char
			break;
		}
	}

	// find the index of the dot to separate major and minor
	size_t dotIndex = shortVersion.find(".");
	if (dotIndex == std::string::npos)
	{
		return VersionData(); // error => default data
	}
	
	size_t charsToCopy = revIndex - (dotIndex + 1);
	// shortVersion => "9.2m \r\n"
	//                   ^ ^
	//            dotIndex revIndex
	
	// convert substrings to integers
	int major = std::stoi(shortVersion.substr(0, dotIndex)); // use index as chars to copy
	int minor = std::stoi(shortVersion.substr(dotIndex + 1, charsToCopy));
	return VersionData(major, minor, revision);
}

// Get the version of this controller
int ASIBase::OnVersion(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		std::ostringstream command;
		command << "V";
		std::string answer;
		// query the device
		int ret = QueryCommand(command.str().c_str(), answer);
		if (ret != DEVICE_OK)
		{
			return ret;
		}
		if (answer.substr(0, 2).compare(":A") == 0)
		{
			pProp->Set(answer.substr(3).c_str());
			return DEVICE_OK;
		}
		// deal with error later
		else if (answer.substr(0, 2).compare(":N") == 0 && answer.length() > 2)
		{
			int errNo = atoi(answer.substr(3).c_str());
			return ERR_OFFSET + errNo;
		}
		return ERR_UNRECOGNIZED_ANSWER;
	}

	return DEVICE_OK;
}

// Get the build name of this controller
int ASIBase::OnBuildName(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		if (initialized_)
		{
			return DEVICE_OK;
		}
		std::ostringstream command;
		command << "BU";
		std::string answer;
		// query the device
		int ret = QueryCommand(command.str().c_str(), answer);
		if (ret != DEVICE_OK)
		{
			return ret;
		}
		pProp->Set(answer.c_str());
	}
	return DEVICE_OK;
}

// Get the compile date of this controller
int ASIBase::OnCompileDate(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		if (initialized_)
		{
			return DEVICE_OK;
		}
		std::ostringstream command;
		command << "CD";
		std::string answer;
		// query the device
		int ret = QueryCommand(command.str().c_str(), answer);
		if (ret != DEVICE_OK)
		{
			return ret;
		}
		pProp->Set(answer.c_str());
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
