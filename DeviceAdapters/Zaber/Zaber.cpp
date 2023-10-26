///////////////////////////////////////////////////////////////////////////////
// FILE:          Zaber.cpp
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
//

#ifdef WIN32
#pragma warning(disable: 4355)
#endif

#include "Zaber.h"
#include "XYStage.h"
#include "Stage.h"
#include "FilterWheel.h"
#include "FilterCubeTurret.h"
#include "Illuminator.h"
#include "ObjectiveChanger.h"

#include <algorithm>

using namespace std;

const char* g_Msg_PORT_CHANGE_FORBIDDEN = "The port cannot be changed once the device is initialized.";
const char* g_Msg_DRIVER_DISABLED = "The driver has disabled itself due to overheating.";
const char* g_Msg_MOVEMENT_FAILED = "The movement has failed.";
const char* g_Msg_COMMAND_REJECTED = "The device rejected the command.";
const char* g_Msg_NO_REFERENCE_POS = "The device has not had a reference position established.";
const char* g_Msg_SETTING_FAILED = "The property could not be set. Is the value in the valid range?";
const char* g_Msg_LAMP_DISCONNECTED = "Some of the illuminator lamps are disconnected.";
const char* g_Msg_LAMP_OVERHEATED = "Some of the illuminator lamps are overheated.";
const char* g_Msg_PERIPHERAL_DISCONNECTED = "A peripheral has been disconnected; please reconnect it or set its peripheral ID to zero and then restart the driver.";
const char* g_Msg_PERIPHERAL_UNSUPPORTED = "Controller firmware does not support one of the connected peripherals; please update the firmware.";
const char* g_Msg_FIRMWARE_UNSUPPORTED = "Firmware is not suported; please update the firmware.";


//////////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
//////////////////////////////////////////////////////////////////////////////////
MODULE_API void InitializeModuleData()
{
	RegisterDevice(g_XYStageName, MM::XYStageDevice, g_XYStageDescription);
	RegisterDevice(g_StageName, MM::StageDevice, g_StageDescription);
	RegisterDevice(g_FilterWheelName, MM::StateDevice, g_FilterWheelDescription);
	RegisterDevice(g_FilterTurretName, MM::StateDevice, g_FilterTurretDescription);
	RegisterDevice(g_IlluminatorName, MM::ShutterDevice, g_IlluminatorDescription);
	RegisterDevice(g_ObjectiveChangerName, MM::StateDevice, g_ObjectiveChangerDescription);
}


MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
	if (strcmp(deviceName, g_XYStageName) == 0)
	{
		return new XYStage();
	}
	else if (strcmp(deviceName, g_StageName) == 0)
	{
		return new Stage();
	}
	else if (strcmp(deviceName, g_FilterWheelName) == 0)
	{
		return new FilterWheel();
	}
	else if (strcmp(deviceName, g_FilterTurretName) == 0)
	{
		return new FilterCubeTurret();
	}
	else if (strcmp(deviceName, g_IlluminatorName) == 0)
	{
		return new Illuminator();
	}
	else if (strcmp(deviceName, g_ObjectiveChangerName) == 0)
	{
		return new ObjectiveChanger();
	}
	else
	{
		return 0;
	}
}


MODULE_API void DeleteDevice(MM::Device* pDevice)
{
	delete pDevice;
}


///////////////////////////////////////////////////////////////////////////////
// ZaberBase (convenience parent class)
///////////////////////////////////////////////////////////////////////////////

ConnectionManager ZaberBase::connections;

ZaberBase::ZaberBase(MM::Device *device) :
	initialized_(false),
	port_("Undefined"),
	device_(device),
	core_(0)
{
}


ZaberBase::~ZaberBase()
{
	resetConnection();
}


int ZaberBase::Command(long device, long axis, const string command, zml::Response& reply)
{
	core_->LogMessage(device_, "ZaberBase::Command\n", true);

	return handleException([&]() {
		ensureConnected();
		reply = connection_->genericCommand(command, static_cast<int>(device), static_cast<int>(axis));
	});
}


void ZaberBase::ensureConnected() {
	if (!connection_) {
		core_->LogMessage(device_, "ZaberBase::ensureConnected\n", true);
		connection_ = ZaberBase::connections.getConnection(port_);
		connection_->enableAlerts();
		onNewConnection();
	}
}

void ZaberBase::onNewConnection() {
}

void ZaberBase::resetConnection() {
	try
	{
		// the connection destructor can throw in the rarest occasions
		connection_ = nullptr;
	}
	catch (const zmlbase::MotionLibException e) 
	{
	}
}

int ZaberBase::Command(long device, long axis, const std::string command) {
	zml::Response reply;
	return Command(device, axis, command, reply);
}


template<typename TReturn> int ZaberBase::GetSetting(long device, long axis, string setting, TReturn& data)
{
	core_->LogMessage(device_, "ZaberBase::GetSetting()\n", true);

	zml::Response resp;
	int ret = Command(device, axis, "get " + setting, resp);
	if (ret != DEVICE_OK)
	{
		return ret;
	}

	stringstream(resp.getData()) >> data;
	return DEVICE_OK;
}

int ZaberBase::GetSettings(long device, long axis, std::string setting, std::vector<double>& data)
{
	core_->LogMessage(device_, "ZaberBase::GetSettings()\n", true);

	data.clear();
	zml::Response resp;
	int ret = Command(device, axis, "get " + setting, resp);
	if (ret != DEVICE_OK)
	{
		if (ret == DEVICE_UNSUPPORTED_COMMAND) {
			return DEVICE_OK;
		}
		return ret;
	}

	std::vector<string> strData;
	CDeviceUtils::Tokenize(resp.getData(), strData, " ");
	for (const auto& token : strData) {
		if (token == "NA") {
			data.push_back(NAN);
			continue;
		}

		double numData;
		stringstream(token) >> numData;
		data.push_back(numData);
	}

	return DEVICE_OK;
}

int ZaberBase::SetSetting(long device, long axis, string setting, double data, int decimalPlaces)
{
	core_->LogMessage(device_, "ZaberBase::SetSetting(double)\n", true);

	ostringstream cmd;
	cmd.precision(decimalPlaces > 0 ? decimalPlaces : 0);
	cmd << "set " << setting << " " << fixed << data;
	zml::Response resp;

	int ret = Command(device, axis, cmd.str(), resp);
	if (ret == ERR_COMMAND_REJECTED)
	{
		return ERR_SETTING_FAILED;
	}
	return ret;
}


bool ZaberBase::IsBusy(long device)
{
	core_->LogMessage(device_, "ZaberBase::IsBusy\n", true);

	zml::Response resp;

	int ret = Command(device, 0, "", resp);
	if (ret != DEVICE_OK)
	{
		ostringstream os;
		os << "ZaberBase::IsBusy failed, error code: " << ret;
		core_->LogMessage(device_, os.str().c_str(), false);
		return false;
	}

	return resp.getStatus() != "IDLE";
}


int ZaberBase::Stop(long device, long lockstepGroup)
{
	core_->LogMessage(device_, "ZaberBase::Stop\n", true);

	ostringstream cmd;
	if (lockstepGroup > 0)
	{
		cmd << "lockstep " << lockstepGroup << " stop";
	}
	else
	{
		cmd << "stop";
	}

	return Command(device, 0, cmd.str());
}


int ZaberBase::GetLimits(long device, long axis, long& min, long& max)
{
	core_->LogMessage(device_, "ZaberBase::GetLimits\n", true);

	int ret = GetSetting(device, axis, "limit.min", min);
	if (ret != DEVICE_OK)
	{
		return ret;
	}

	return GetSetting(device, axis, "limit.max", max);
}


int ZaberBase::SendMoveCommand(long device, long axis, std::string type, long data, bool lockstep)
{
	core_->LogMessage(device_, "ZaberBase::SendMoveCommand\n", true);

	ostringstream cmd;
	if (lockstep)
	{
		cmd << "lockstep " << axis << " move " << type << " " << data;
		axis = 0;
	}
	else
	{
		cmd << "move " << type << " " << data;
	}

	return Command(device, axis, cmd.str());
}


int ZaberBase::SendAndPollUntilIdle(long device, long axis, string command)
{
	core_->LogMessage(device_, "ZaberBase::SendAndPollUntilIdle\n", true);
	return handleException([&]()
	{
		ensureConnected();
		connection_->genericCommand(command, static_cast<int>(device), static_cast<int>(axis));
		auto zmlDevice = connection_->getDevice(device);
		if (axis == 0) {
			zmlDevice.getAllAxes().waitUntilIdle();
		}
		else
		{
			zmlDevice.getAxis(axis).waitUntilIdle();
		}
	});
}


int ZaberBase::GetRotaryIndexedDeviceInfo(long device, long axis, long& numIndices, long& currentIndex)
{
	core_->LogMessage(device_, "ZaberBase::GetRotaryIndexedDeviceInfo\n", true);

   // Get the size of a full circle in microsteps.
   long cycleSize = -1;
   int ret = GetSetting(device, axis, "limit.cycle.dist", cycleSize);
   if (ret != DEVICE_OK)
   {
      core_->LogMessage(device_, "Attempt to detect rotary cycle distance failed.\n", true);
      return ret;
   }

   if ((cycleSize < 1) || (cycleSize > 1000000000))
   {
      core_->LogMessage(device_, "Device cycle distance is out of range or was not returned.\n", true);
      return DEVICE_SERIAL_INVALID_RESPONSE;
   }


   // Get the size of a filter increment in microsteps.
   long indexSize = -1;
   ret = GetSetting(device, axis, "motion.index.dist", indexSize);
   if (ret != DEVICE_OK)
   {
      core_->LogMessage(device_, "Attempt to detect index spacing failed.\n", true);
      return ret;
   }

   if ((indexSize < 1) || (indexSize > 1000000000) || (indexSize > cycleSize))
   {
      core_->LogMessage(device_, "Device index distance is out of range or was not returned.\n", true);
      return DEVICE_SERIAL_INVALID_RESPONSE;
   }

   numIndices = cycleSize / indexSize;

   long index = -1;
   ret = GetSetting(device, axis, "motion.index.num", index);
   if (ret != DEVICE_OK)
   {
      core_->LogMessage(device_, "Attempt to detect current index position failed.\n", true);
      return ret;
   }

   if ((index < 0) || (index > 1000000000))
   {
      core_->LogMessage(device_, "Device current index is out of range or was not returned.\n", true);
      return DEVICE_SERIAL_INVALID_RESPONSE;
   }

   currentIndex = index;

   return ret;
}


// Attempts to get the device firmware version in major.minor floating
// point representation (ie 7.01). This ignores reply flags but does
// return error codes relating to the flags. If the command is rejected
// or the version can otherwise not be determined, the version will
// be 0.00. The build number is not included.
int ZaberBase::GetFirmwareVersion(long device, double& version)
{
	core_->LogMessage(device_, "ZaberBase::GetFirmwareVersion\n", true);
	return GetSetting(device, 0, "version", version);
}


int ZaberBase::ActivatePeripheralsIfNeeded(long device)
{
	core_->LogMessage(device_, "ZaberBase::ActivatePeripheralsIfNeeded\n", true);

	zml::Response reply;
	int ret = Command(device, 0, "warnings", reply);
	if (ret != DEVICE_OK)
	{
		core_->LogMessage(device_, "Could not get device warning flags.\n", true);
		return ret;
	}

	if (reply.getData().find("FZ") == string::npos) {
		return DEVICE_OK;
	}

	core_->LogMessage(device_, "An axis needs activation.\n", false);
	ret = Command(device, 0, "activate");
	if (ret != DEVICE_OK)
	{
		core_->LogMessage(device_, "Activating a peripheral failed.\n", false);
	}

	return DEVICE_OK;
}

int ZaberBase::handleException(std::function<void()> wrapped) {
	try
	{
		wrapped();
		return DEVICE_OK;
	}
	catch (const zmlbase::ConnectionFailedException e) {
		core_->LogMessage(device_, e.what(), true);
		resetConnection();
		return DEVICE_NOT_CONNECTED;
	}
	catch (const zmlbase::ConnectionClosedException e) {
		core_->LogMessage(device_, e.what(), true);
		resetConnection();
		return DEVICE_NOT_CONNECTED;
	}
	catch (const zmlbase::CommandFailedException e) {
		core_->LogMessage(device_, e.what(), true);
		auto reason = e.getDetails().getResponseData();
		if (reason == "BADCOMMAND") {
			return DEVICE_UNSUPPORTED_COMMAND;
		}
		else if (reason == "DRIVERDISABLED") {
			return ERR_DRIVER_DISABLED;
		}
		else if (reason == "INACTIVE") {
			return ERR_PERIPHERAL_DISCONNECTED;
		}
		else if (reason == "NOTSUPPORTED") {
			return ERR_PERIPHERAL_UNSUPPORTED;
		}
		return ERR_COMMAND_REJECTED;
	}
	catch (const zmlbase::RequestTimeoutException e) {
		core_->LogMessage(device_, e.what(), true);
		return DEVICE_NOT_CONNECTED;
	}
	catch (const zmlbase::MovementFailedException e) {
		core_->LogMessage(device_, e.what(), true);
		return ERR_MOVEMENT_FAILED;
	}
	catch (const zmlbase::MotionLibException e) {
		core_->LogMessage(device_, e.what(), true);
		return DEVICE_ERR;
	}
}


void ZaberBase::setErrorMessages(std::function<void(int, const char*)> setter) {
	setter(ERR_PORT_CHANGE_FORBIDDEN, g_Msg_PORT_CHANGE_FORBIDDEN);
	setter(ERR_DRIVER_DISABLED, g_Msg_DRIVER_DISABLED);
	setter(ERR_COMMAND_REJECTED, g_Msg_COMMAND_REJECTED);
	setter(ERR_MOVEMENT_FAILED, g_Msg_MOVEMENT_FAILED);
	setter(ERR_NO_REFERENCE_POS, g_Msg_NO_REFERENCE_POS);
	setter(ERR_SETTING_FAILED, g_Msg_SETTING_FAILED);
	setter(ERR_PERIPHERAL_DISCONNECTED, g_Msg_PERIPHERAL_DISCONNECTED);
	setter(ERR_PERIPHERAL_UNSUPPORTED, g_Msg_PERIPHERAL_UNSUPPORTED);
	setter(ERR_LAMP_DISCONNECTED, g_Msg_LAMP_DISCONNECTED);
	setter(ERR_LAMP_OVERHEATED, g_Msg_LAMP_OVERHEATED);
	setter(ERR_FIRMWARE_UNSUPPORTED, g_Msg_FIRMWARE_UNSUPPORTED);
}
