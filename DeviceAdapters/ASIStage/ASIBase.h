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
#include <string>


// The ASIStage device adapter does not implement a hub device, each device could be connected to a different MS2000, 
// that's why the build name, compile date, and version are queried for each device.


// This data structure is used to store MS2000 version data.
// The format for firmware versions changed match the Tiger controller after 9.2p
// Old version format: ":A Version: USB-9.2p \r\n" (revision is a character: 'a'-'z')
// New version format: ":A Version: USB-9.50 \r\n" (revision is a digit character: '0'-'9')
class VersionData
{
public:
	VersionData() : major_(0), minor_(0), rev_('-') { }
	explicit VersionData(int major, int minor, char rev)
		: major_(major), minor_(minor), rev_(rev) { }

	// Return true if the controller firmware version is at least the specified version.
	bool IsVersionAtLeast(int major, int minor, char rev) const
	{
		// Avoid comparing 'rev' across pre-9.50 (char) and 9.50+ (digit) formats.
		// Comparing 'rev' relies on ASCII, which doesn't reflect logical version order.
		// This scenario should never be encountered under the current versioning scheme.
		if (major_ > major)
		{
			return true;
		}
		if (major_ < major)
		{
			return false;
		}
		if (minor_ > minor)
		{
			return true;
		}
		if (minor_ < minor)
		{
			return false;
		}
		return rev_ >= rev;
	}

private:
	int major_;
	int minor_;
	char rev_;
};

// Note: concrete device classes deriving ASIBase must set core_ in Initialize()
class ASIBase
{
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
	VersionData ParseVersionString(const std::string& version) const;

protected:
	int OnVersion(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnBuildName(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnCompileDate(MM::PropertyBase* pProp, MM::ActionType eAct);
	int GetVersion(std::string& version) const;
	int GetBuildName(std::string& buildName) const;
	int GetCompileDate(std::string& compileDate) const;

	bool oldstage_;
	bool initialized_;
	MM::Core* core_;
	MM::Device* device_;
	std::string port_;
	std::string version_;
	std::string buildName_;
	std::string compileDate_;
	std::string oldstagePrefix_; // "1H" or "2H" for LX-4000 stages
	VersionData versionData_;
};

#endif // ASIBASE_H
