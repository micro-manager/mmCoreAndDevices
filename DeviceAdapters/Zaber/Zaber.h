///////////////////////////////////////////////////////////////////////////////
// FILE:          Zaber.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Zaber Controller Driver
//
// AUTHOR:        David Goosen, Athabasca Witschi, Martin Zak (contact@zaber.com)
//
// COPYRIGHT:     Zaber Technologies Inc., 2014
//
// LICENSE:       This file is distributed under the BSD license.
//                License text is included with the source distribution.
//
//                This file is distributed in the hope that it will be useful,
//                but WITHOUT ANY WARRANTY; without even the implied warranty
//                of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//                IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//                CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//                INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.

#ifndef _ZABER_H_
#define _ZABER_H_

#include <MMDevice.h>
#include <DeviceBase.h>
#include <ModuleInterface.h>
#include <sstream>
#include <string>
#undef VERSION
#include <zaber/motion/microscopy.h>
#include <zaber/motion/ascii.h>

#include "ConnectionManager.h"

namespace zmlbase = zaber::motion;
namespace zml = zaber::motion::ascii;
namespace zmlmi = zaber::motion::microscopy;

//////////////////////////////////////////////////////////////////////////////
// Various constants: error codes, error messages
//////////////////////////////////////////////////////////////////////////////

#define ERR_PORT_CHANGE_FORBIDDEN    10002
#define ERR_DRIVER_DISABLED          10004
#define ERR_MOVEMENT_FAILED          10016
#define ERR_COMMAND_REJECTED         10032
#define	ERR_NO_REFERENCE_POS         10064
#define	ERR_SETTING_FAILED           10128
#define ERR_LAMP_DISCONNECTED        10512
#define ERR_LAMP_OVERHEATED          11024
#define ERR_PERIPHERAL_DISCONNECTED  12048
#define ERR_PERIPHERAL_UNSUPPORTED   14096
#define ERR_FIRMWARE_UNSUPPORTED     18192

// N.B. Concrete device classes deriving ZaberBase must set core_ in
// Initialize().
class ZaberBase
{
public:
	ZaberBase(MM::Device *device);
	virtual ~ZaberBase();

	static void setErrorMessages(std::function<void(int, const char*)> setter);

protected:
	int Command(long device, long axis, const std::string command, zml::Response& reply);
	int Command(long device, long axis, const std::string command);
	template<typename TReturn> int GetSetting(long device, long axis, std::string setting, TReturn& data);
	int GetSettings(long device, long axis, std::string setting, std::vector<double>& data);
	int SetSetting(long device, long axis, std::string setting, double data, int decimalPlaces = -1);
	bool IsBusy(long device);
	int Stop(long device, long lockstepGroup = 0);
	int GetLimits(long device, long axis, long& min, long& max);
	int SendMoveCommand(long device, long axis, std::string type, long data, bool lockstep = false);
	int SendAndPollUntilIdle(long device, long axis, std::string command);
	int GetRotaryIndexedDeviceInfo(long device, long axis, long& numIndices, long& currentIndex);
	int GetFirmwareVersion(long device, double& version);
	int ActivatePeripheralsIfNeeded(long device);
	int handleException(std::function<void()> wrapped);
	void ensureConnected();
	virtual void onNewConnection();
	void resetConnection();

	bool initialized_;
	std::string port_;
	MM::Device *device_;
	MM::Core *core_;
	std::shared_ptr<zml::Connection> connection_;

private:
	static ConnectionManager connections;
};

#endif //_ZABER_H_
