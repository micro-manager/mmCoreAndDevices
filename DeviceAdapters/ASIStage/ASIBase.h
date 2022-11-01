/*
 * Project: ASIStage Device Adapter
 * License/Copyright: BSD 3-clause, see license.txt
 * Maintainers: Brandon Simpson (brandon@asiimaging.com)
 *              Jon Daniels (jon@asiimaging.com)
 */

#ifndef _ASIBASE_H_
#define _ASIBASE_H_

#include "MMDevice.h"
#include "DeviceBase.h"
#include "ASIStage.h"
#include <string>

// The ASIStage device adapter does not implement a hub device, each device could be connected to a different MS2000, 
// that's why the build name, compile date, and version are queried every time.

// This data structure is used to store MS2000 version data.
class VersionData
{
public:
	VersionData() : major_(0), minor_(0), rev_('-') { }
	VersionData(int major, int minor, char rev) : major_(major), minor_(minor), rev_(rev) { }
	void setMajor(int major) { major_ = major;  }
	void setMinor(int minor) { minor_ = minor; }
	void setRev(char rev) { rev_ = rev; }
	int getMajor() const { return major_; }
	int getMinor() const { return minor_; }
	char getRev() const { return rev_; }
	bool isVersionAtLeast(int major, int minor, char rev) const
	{
		if (major_ < major)
		{
			return false;
		}
		if (minor_ < minor)
		{
			return false;
		}
		if (rev_ < rev)
		{
			return false;
		}
		return true;
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

	int ClearPort(void);
	int CheckDeviceStatus(void);
	int SendCommand(const char* command) const;
	int QueryCommandACK(const char* command);
	int QueryCommand(const char* command, std::string& answer) const;
	unsigned int ConvertDay(int year, int month, int day);
	unsigned int ExtractCompileDay(const char* compile_date);

	int ParseResponseAfterPosition(const std::string& answer, const unsigned int position, long& value) const;
	int ParseResponseAfterPosition(const std::string& answer, const unsigned int position, double& value) const;
	int ParseResponseAfterPosition(const std::string& answer, const unsigned int position, const unsigned int count, double& value) const;
	int ResponseStartsWithColonA(const std::string& answer) const;
	VersionData ExtractVersionData(const std::string& version) const;

protected:
	int OnVersion(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnBuildName(MM::PropertyBase* pProp, MM::ActionType eAct);
	int OnCompileDate(MM::PropertyBase* pProp, MM::ActionType eAct);

	bool oldstage_;
	MM::Core* core_;
	bool initialized_;
	MM::Device* device_;
	std::string oldstagePrefix_;
	std::string port_;
	VersionData versionData_;
	unsigned int compileDay_; // "days" since Jan 1 2000 since the firmware was compiled according to (compile day + 31*(compile month-1) + 12*31*(compile year-2000))
};

#endif // end _ASIBASE_H_
