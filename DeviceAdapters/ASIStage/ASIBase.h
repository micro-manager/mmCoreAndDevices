/*
 * Project: ASIStage Device Adapter
 * License/Copyright: BSD 3-clause, see license.txt
 * Maintainers: Brandon Simpson (brandon@asiimaging.com)
 *              Jon Daniels (jon@asiimaging.com)
 */

#ifndef ASIBASE_H
#define ASIBASE_H

#include "MMDevice.h"
#include "DeviceBase.h"
#include "ASIStage.h"
#include <regex>
#include <string>


// The ASIStage device adapter does not implement a hub device, each device could be connected to a different MS2000, 
// that's why the build name, compile date, and version are queried for each device.


// This data structure is used to store MS-2000 version data.
// The format for firmware versions changed to match the Tiger controller after 9.2p
// Old version format: ":A Version: USB-9.2p \r\n" (revision is a character: 'a'-'z')
// New version format: ":A Version: USB-9.50 \r\n" (revision is a digit character: '0'-'9')
class Version {
public:
	Version() : major_(0), minor_(0), rev_(0) { }
	explicit Version(unsigned int major, unsigned int minor, unsigned int rev)
		: major_(major), minor_(minor), rev_(rev) { }

	// Return true if the controller firmware version is at least the specified version.
	bool IsVersionAtLeast(unsigned int major, unsigned int minor, unsigned int rev) const {
		// Avoid comparing 'rev' across pre-9.50 (char) and 9.50+ (digit) formats.
		// Comparing 'rev' relies on ASCII, which doesn't reflect logical version order.
		// This scenario should never be encountered under the current versioning scheme.
		if (major_ > major) {
			return true;
		}
		if (major_ < major) {
			return false;
		}
		if (minor_ > minor) {
			return true;
		}
		if (minor_ < minor) {
			return false;
		}
		return rev_ >= rev;
	}

	// Return the version data parsed from a std::string.
	static Version ParseString(const std::string& version) {
		// Version response examples:
		// Example A: ":A Version: USB-9.2p \r\n"
		// Example B: ":A Version: USB-9.50 \r\n"
		const size_t dashIndex = version.find("-");
		if (dashIndex == std::string::npos) {
			return Version(); // error => default data
		}

		// short version => "9.2m \r\n"
		const std::string ver = version.substr(dashIndex + 1);

		// find the index of the dot that separates major and minor version
		const size_t dotIndex = ver.find(".");
		if (dotIndex == std::string::npos) {
			return Version(); // error => default data
		}

		// use substr for major versions with more than 1 digit, ##.## for example
		// minor version and revision will only ever be 1 character,
		// at these specific locations after the dot in the response
		const unsigned int major = std::stoi(ver.substr(0, dotIndex));
		const unsigned int minor = std::stoi(ver.substr(dotIndex + 1, 1));
		const unsigned int revision = static_cast<unsigned int>(ver.at(dotIndex + 2));
		return Version(major, minor, revision);
	}

	bool operator>=(const Version& other) const {
		return IsVersionAtLeast(other.major_, other.minor_, other.rev_);
	}

private:
	unsigned int major_;
	unsigned int minor_;
	unsigned int rev_;
};

// Note: concrete device classes deriving ASIBase must set core_ in Initialize()
class ASIBase {
public:
	ASIBase(MM::Device* device, const char* prefix);
	virtual ~ASIBase();

	int ClearPort();
	int CheckDeviceStatus();
	int SendCommand(const char* command) const;
	int QueryCommandACK(const char* command) const;
	int QueryCommand(const char* command, std::string& answer) const;

	int ParseResponseAfterPosition(const std::string& answer, unsigned int position, long& value) const;
	int ParseResponseAfterPosition(const std::string& answer, unsigned int position, double& value) const;
	int ParseResponseAfterPosition(const std::string& answer, unsigned int position, unsigned int count, double& value) const;
	int ResponseStartsWithColonA(const std::string& answer) const;

protected:
	int OnVersion(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnBuildName(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnCompileDate(MM::PropertyBase* pProp, MM::ActionType eAct);
	int GetVersion(std::string& version) const;
	int GetBuildName(std::string& buildName) const;
	int GetCompileDate(std::string& compileDate) const;

	static constexpr size_t SERIAL_RXBUFFER_SIZE = 2048;

	MM::Core* core_;
	MM::Device* device_;
	std::string port_;

	bool initialized_;
	bool oldstage_;

	Version version_;
	std::string firmwareVersion_;
	std::string firmwareBuild_;
	std::string firmwareDate_;

	// Stage-specific configuration
	std::string oldstagePrefix_; // "1H" or "2H" for LX-4000 stages, empty string for MS-2000 stages
	std::string commandPrefix_; // set to oldstagePrefix_ if oldstage_ is true, otherwise empty string
	std::string serialTerm_; // changes if oldstage_ is true
};

#endif // ASIBASE_H
