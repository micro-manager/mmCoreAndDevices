/*
 * Project: ASIStage Device Adapter
 * License/Copyright: BSD 3-clause, see license.txt
 * Maintainers: Brandon Simpson (brandon@asiimaging.com)
 *              Jon Daniels (jon@asiimaging.com)
 */

#include "ASIBase.h"

ASIBase::ASIBase(MM::Device* device, const char* prefix) :
	core_(nullptr),
	device_(device),
	port_("Undefined"),
	initialized_(false),
	oldstage_(false),
	versionData_(VersionData()),
	version_("Undefined"),
	buildName_("Undefined"),
	compileDate_("Undefined"),
	oldstagePrefix_(prefix),
	commandPrefix_(""),
	serialTerm_("\r\n")
{
}

ASIBase::~ASIBase()
{
}

// Clear contents of serial port
int ASIBase::ClearPort()
{
	constexpr size_t bufferSize = 255;
	unsigned char clear[bufferSize];
	unsigned long read = bufferSize;
	int ret;
	while (read == bufferSize)
	{
		ret = core_->ReadFromSerial(device_, port_.c_str(), clear, bufferSize, read);
		if (ret != DEVICE_OK)
		{
			return ret;
		}
	}
	return DEVICE_OK;
}

// Communication "send" utility function:
int ASIBase::SendCommand(const char* command) const {
	const std::string cmd = commandPrefix_ + command;
	return core_->SetSerialCommand(device_, port_.c_str(), cmd.c_str(), "\r");
}

// Communication "send & receive" utility function:
int ASIBase::QueryCommand(const char* command, std::string& answer) const {
	// send command
	int ret = SendCommand(command);
	if (ret != DEVICE_OK) {
		return ret;
	}
	// block/wait for acknowledge (or until we time out)
	char buf[SERIAL_RXBUFFER_SIZE] = { '\0' };
	ret = core_->GetSerialAnswer(device_, port_.c_str(), SERIAL_RXBUFFER_SIZE, buf, serialTerm_.c_str());
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
	if (answer.compare(0, 2, ":A") != 0)
	{
		return ERR_UNRECOGNIZED_ANSWER;
	}
	return DEVICE_OK;
}

// Communication "test device type" utility function:
// Set the value of oldstage_ to true for LX-4000 and false for MS-2000.
// This determines commandPrefix_ and serialTerm_ as well.
int ASIBase::CheckDeviceStatus() {
	// send status command (test for new protocol)
	std::string answer;
	int ret = QueryCommand("/", answer);
	if (!oldstagePrefix_.empty() && ret != DEVICE_OK) {
		// send status command (test for older LX-4000 protocol)
		oldstage_ = true;
		serialTerm_ = "\r\n\3";
		commandPrefix_ = oldstagePrefix_;
		ret = QueryCommand("/", answer);
	} else {
		// standard configuration for the MS-2000
		oldstage_ = false;
		serialTerm_ = "\r\n";
		commandPrefix_.clear();
	}
	return ret;
}

VersionData ASIBase::ParseVersionString(const std::string& version) const
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

int ASIBase::GetVersion(std::string& version) const
{
   std::string answer;
   int ret = QueryCommand("V", answer);
   if (ret != DEVICE_OK)
   {
      return ret;
   }
   if (answer.compare(0, 2, ":A") == 0)
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

int ASIBase::GetBuildName(std::string& buildName) const
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

int ASIBase::GetCompileDate(std::string& compileDate) const
{
	std::string answer;
	int ret = QueryCommand("CD", answer);
	if (ret != DEVICE_OK)
	{
		return ret;
	}
	compileDate = answer;
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
	if (answer.compare(0, 2, ":A") == 0)
	{
		return DEVICE_OK;
	}
	else if (answer.length() > 2 && answer.compare(0, 2, ":N") == 0)
	{
		const int errorNumber = atoi(answer.substr(3).c_str());
		return ERR_OFFSET + errorNumber;
	}
	return ERR_UNRECOGNIZED_ANSWER;
}
