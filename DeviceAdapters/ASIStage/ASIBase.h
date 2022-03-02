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
	unsigned int compileDay_; // "days" since Jan 1 2000 since the firmware was compiled according to (compile day + 31*(compile month-1) + 12*31*(compile year-2000))
};

#endif // _ASIBASE_H_
