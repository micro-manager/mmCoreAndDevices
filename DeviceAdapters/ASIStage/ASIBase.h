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
// Old version format: ":A Version: USB-9.2p \r\n"
// New version format: ":A Version: USB-9.50 \r\n"
class VersionData
{
public:
	VersionData() : major_(0), minor_(0), rev_('-') { }
	VersionData(int major, int minor, char rev) : major_(major), minor_(minor), rev_(rev) { }
	void setMajor(int major) { major_ = major; }
	void setMinor(int minor) { minor_ = minor; }
	void setRev(char rev) { rev_ = rev; }
	int getMajor() const { return major_; }
	int getMinor() const { return minor_; }
	char getRev() const { return rev_; }

	// Return true if the controller firmware version is at least the specified version.
	bool isVersionAtLeast(int major, int minor, char rev) const
	{
		// Note: avoid comparing the old character revision numbers
		// with the new numeric revision numbers by returning early.
		// Character revisions only exist prior to firmware version 9.50.
		if (major_ >= major)
		{
			return true;
		}
		if (minor_ >= minor)
		{
			return true;
		}
		if (rev_ >= rev)
		{
			return true;
		}
		return false;
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
	unsigned int ConvertDay(int year, int month, int day) const;
	unsigned int ExtractCompileDay(const char* compile_date) const;

	int ParseResponseAfterPosition(const std::string& answer, unsigned int position, long& value) const;
	int ParseResponseAfterPosition(const std::string& answer, unsigned int position, double& value) const;
	int ParseResponseAfterPosition(const std::string& answer, unsigned int position, unsigned int count, double& value) const;
	int ResponseStartsWithColonA(const std::string& answer) const;
	VersionData ExtractVersionData(const std::string& version) const;

protected:
	int OnVersion(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnBuildName(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnCompileDate(MM::PropertyBase* pProp, MM::ActionType eAct);
	int GetVersion(std::string& version);
	int GetBuildName(std::string& buildName);
	int GetCompileDate(std::string& compileDate);

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
	unsigned int compileDay_;
	// "days" since Jan 1 2000 since the firmware was compiled according to:
	// (compile day + 31*(compile month-1) + 12*31*(compile year-2000))
};

#endif // ASIBASE_H
