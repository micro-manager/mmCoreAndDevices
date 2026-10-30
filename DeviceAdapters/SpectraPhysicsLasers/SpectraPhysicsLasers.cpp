// Micro-Manager Device Adapter for Spectra-Physics Ultrafast Laser Systems
// Author: Gabriel Selzer
//
// Copyright 2026 Board of Regents of the University of Wisconsin System
//
// This file is distributed under the BSD license. License text is included
// with the source distribution.
//
// This file is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE.
//
// IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.

#include "SpectraPhysicsLasers.h"

#include "ModuleInterface.h"

#include <chrono>
#include <cstring>
#include <system_error>

const char* g_DeviceNameHub = "Spectra-Physics Laser";
const char* g_DeviceNameShutter = "Spectra-Physics Main Shutter";
const char* g_DeviceNameInsight1040 = "Spectra-Physics Insight 1040nm Shutter";

const char* g_On = "On";
const char* g_Off = "Off";
const char* g_Standby = "Maintain laser emission";
const char* g_Hibernate = "Turn off laser";

MODULE_API void InitializeModuleData()
{
    RegisterDevice(g_DeviceNameHub, MM::HubDevice, "Spectra-Physics Ultrafast Laser System");
    RegisterDevice(g_DeviceNameShutter, MM::ShutterDevice, "Spectra-Physics Ultrafast Laser System Main Shutter");
    RegisterDevice(g_DeviceNameInsight1040, MM::ShutterDevice, "Spectra-Physics InSight Laser System 1040nm Shutter");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
    if (deviceName == 0)
        return 0;

    if (std::strcmp(deviceName, g_DeviceNameHub) == 0)
    {
        return new SpectraPhysicsHub();
    }

    if (std::strcmp(deviceName, g_DeviceNameShutter) == 0)
    {
        return new SpectraPhysicsMain();
    }

    if (std::strcmp(deviceName, g_DeviceNameInsight1040) == 0)
    {
        return new SpectraPhysicsInsight1040();
    }
    return 0;
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
    delete pDevice;
}

////////////////////////////////////////////////////////////////////////////////////
// HUB DEVICE
////////////////////////////////////////////////////////////////////////////////////

SpectraPhysicsHub::SpectraPhysicsHub() :
    port_("Undefined"),
    onClose_(g_Hibernate),
    initialized_(false),
    watchdog_(*this)
{
    InitializeDefaultErrorMessages();
    SetErrorText(ERR_PORT_CHANGE_FORBIDDEN, "Cannot change port after initialization");
    SetErrorText(ERR_PUMP_LASER_NOT_WARM, "Cannot turn on the pump laser before it's warmed up");
    SetErrorText(ERR_WAVELENGTH_CHANGING, "Cannot turn on the pump laser before the wavelength is stable");
    SetErrorText(ERR_NO_HUB, "Cannot obtain the SpectraPhysics MMCore Hub device");
    SetErrorText(ERR_INVALID_MODEL, "This Spectra-Physics Laser cannot perform the requested functionality");
    SetErrorText(ERR_WATCHDOG_START_FAILED, "Cannot start the watchdog thread");

    // COM port property
	CPropertyAction* pActPort = new CPropertyAction (this, &SpectraPhysicsHub::OnPort);
	CreateStringProperty(MM::g_Keyword_Port, "Undefined", false, pActPort, true);

}

SpectraPhysicsHub::~SpectraPhysicsHub()
{
    Shutdown();
}

void SpectraPhysicsHub::GetName(char* name) const
{
    CDeviceUtils::CopyLimitedString(name, g_DeviceNameHub);
}

int SpectraPhysicsHub::Initialize()
{
    int ret{};

    // Clean COM port
    ret = PurgeComPort(port_.c_str());
    if (ret != DEVICE_OK)
        return ret;

    // Sanity check the device
    std::string id{};
    ret = ExecuteCommand("*IDN?", id);
    if (ret != 0)
        return ret;
    // NOTE: the "Spectra Physics"/"Spectra-Physics" vendor prefix varies (space vs. hyphen,
    // comma vs. comma-space) across models and firmware versions, so match on the model name only.
    if (id.find("InSight") != std::string::npos) {
        // TODO: If we want to support DeepSee commands, we could enable them iff "Insight DeepSee" is
        // present in this string.
        LogMessage("Controlling device with id \"" + id + "\"");
        model_ = INSIGHT;
    }
    else if (id.find("MaiTai") != std::string::npos) {
        // TODO: MaiTai's id string doesn't differ for DeepSee, according to the manuals...
        LogMessage("Controlling device with id \"" + id + "\"");
        model_ = MAITAI;
    }
    else {
        // *Waves hand* this is not the device you're looking for...
        return DEVICE_NOT_CONNECTED;
    }

    // Configure on close property
    std::string watchdogProp = "When Shutting Down";
	CPropertyAction* pActWatchdog = new CPropertyAction (this, &SpectraPhysicsHub::OnWatchdog);
	ret = CreateStringProperty(watchdogProp.c_str(), g_Hibernate, false, pActWatchdog);
    if (ret != 0)
        return ret;
    ret = AddAllowedValue(watchdogProp.c_str(), g_Hibernate);
    if (ret != 0)
        return ret;
    ret = AddAllowedValue(watchdogProp.c_str(), g_Standby);
    if (ret != 0)
        return ret;
    // (and initialize the watchdog timer)
	ret = ExecuteCommand("TIM:WATC 3");
	if (ret != 0)
		return ret;

    // Configure pump laser property
	CPropertyAction* pActPumpLaser = new CPropertyAction(this, &SpectraPhysicsHub::OnPumpLaser);
    ret = CreateStringProperty("Laser", g_Off, false, pActPumpLaser);
    if (ret != 0)
        return ret;
    ret = AddAllowedValue("Laser", g_Off);
    if (ret != 0)
        return ret;
    ret = AddAllowedValue("Laser", g_On);
    if (ret != 0)
        return ret;

    // Configure humidity property (read-only)
	CPropertyAction* pActHumidity = new CPropertyAction(this, &SpectraPhysicsHub::OnHumidity);
    ret = CreateFloatProperty("Relative Humidity (%)", 0.0, true, pActHumidity);
    if (ret != 0)
        return ret;

    // Configure warmup percentage property (read-only)
	CPropertyAction* pActWarmup = new CPropertyAction(this, &SpectraPhysicsHub::OnWarmup);
    ret = CreateIntegerProperty("Warmup Percentage (%)", 0, true, pActWarmup);
    if (ret != 0)
        return ret;

    // Configure diode1 current property (read-only)
	CPropertyAction* pActDiode1Current = new CPropertyAction(this, &SpectraPhysicsHub::OnDiode1Current);
    ret = CreateFloatProperty("Diode 1 Current (A)", 0, true, pActDiode1Current);
    if (ret != 0)
        return ret;

    // Configure diode1 temperature property (read-only)
	CPropertyAction* pActDiode1Temp = new CPropertyAction(this, &SpectraPhysicsHub::OnDiode1Temp);
    ret = CreateFloatProperty("Diode 1 Temperature (C)", 0, true, pActDiode1Temp);
    if (ret != 0)
        return ret;

    // Configure output power property (read-only)
	CPropertyAction* pActPower = new CPropertyAction(this, &SpectraPhysicsHub::OnPower);
    ret = CreateFloatProperty("Laser Power (W)", 0, true, pActPower);
    if (ret != 0)
        return ret;

    // Configure history buffer property (read-only)
	CPropertyAction* pActHistoryBuffer = new CPropertyAction(this, &SpectraPhysicsHub::OnHistoryBuffer);
    ret = CreateStringProperty("Status Code Buffer", "", true, pActHistoryBuffer);
    if (ret != 0)
        return ret;

    // Configure emission flag property (read-only)
	CPropertyAction* pActEmission = new CPropertyAction(this, &SpectraPhysicsHub::OnEmission);
    ret = CreateIntegerProperty("Emission", 0, true, pActEmission);
    if (ret != 0)
        return ret;
    ret = AddAllowedValue("Emission", "0");
    if (ret != 0)
        return ret;
    ret = AddAllowedValue("Emission", "1");
    if (ret != 0)
        return ret;

    // Configure pusling flag property (read-only)
	CPropertyAction* pActPulsing = new CPropertyAction(this, &SpectraPhysicsHub::OnPulsing);
    ret = CreateIntegerProperty("Pulsing", 0, true, pActPulsing);
    if (ret != 0)
        return ret;
    ret = AddAllowedValue("Pulsing", "0");
    if (ret != 0)
        return ret;
    ret = AddAllowedValue("Pulsing", "1");
    if (ret != 0)
        return ret;

    // Configure servo-on flag property (read-only)
	CPropertyAction* pActServoOn = new CPropertyAction(this, &SpectraPhysicsHub::OnServoOn);
    ret = CreateIntegerProperty("Servo On", 0, true, pActServoOn);
    if (ret != 0)
        return ret;
    ret = AddAllowedValue("Servo On", "0");
    if (ret != 0)
        return ret;
    ret = AddAllowedValue("Servo On", "1");
    if (ret != 0)
        return ret;

    // Configure Warning flag property (read-only)
	CPropertyAction* pActWarning = new CPropertyAction(this, &SpectraPhysicsHub::OnWarning);
    ret = CreateIntegerProperty("Warning", 0, true, pActWarning);
    if (ret != 0)
        return ret;
    ret = AddAllowedValue("Warning", "0");
    if (ret != 0)
        return ret;
    ret = AddAllowedValue("Warning", "1");
    if (ret != 0)
        return ret;

    // Configure Fault flag property (read-only)
	CPropertyAction* pActFault = new CPropertyAction(this, &SpectraPhysicsHub::OnFault);
    ret = CreateIntegerProperty("Fault", 0, true, pActFault);
    if (ret != 0)
        return ret;
    ret = AddAllowedValue("Fault", "0");
    if (ret != 0)
        return ret;
    ret = AddAllowedValue("Fault", "1");
    if (ret != 0)
        return ret;

    // InSight-specific parameters
    if (model_ == INSIGHT) {
		// Configure diode2 current property (read-only)
		CPropertyAction* pActDiode2Current = new CPropertyAction(this, &SpectraPhysicsHub::OnDiode2Current);
		ret = CreateFloatProperty("Diode 2 Current (A)", 0, true, pActDiode2Current);
		if (ret != 0)
			return ret;

		// Configure diode2 temperature property (read-only)
		CPropertyAction* pActDiode2Temp = new CPropertyAction(this, &SpectraPhysicsHub::OnDiode2Temp);
		ret = CreateFloatProperty("Diode 2 Temperature (C)", 0, true, pActDiode2Temp);
		if (ret != 0)
			return ret;

		// Configure diode1 hours property (read-only)
		CPropertyAction* pActDiode1Hours = new CPropertyAction(this, &SpectraPhysicsHub::OnDiode1Hours);
		ret = CreateFloatProperty("Diode 1 Accumulated Hours", 0, true, pActDiode1Hours);
		if (ret != 0)
			return ret;

		// Configure diode2 hours property (read-only)
		CPropertyAction* pActDiode2Hours = new CPropertyAction(this, &SpectraPhysicsHub::OnDiode2Hours);
		ret = CreateFloatProperty("Diode 2 Accumulated Hours", 0, true, pActDiode2Hours);
		if (ret != 0)
			return ret;

		// Configure user interlock flag property (read-only)
		CPropertyAction* pActUserInterlock = new CPropertyAction(this, &SpectraPhysicsHub::OnUserInterlock);
		ret = CreateIntegerProperty("User Interlock", 0, true, pActUserInterlock);
		if (ret != 0)
			return ret;
		ret = AddAllowedValue("User Interlock", "0");
		if (ret != 0)
			return ret;
		ret = AddAllowedValue("User Interlock", "1");
		if (ret != 0)
			return ret;

		// Configure Keyswitch interlock flag property (read-only)
		CPropertyAction* pActKeyswitchInterlock = new CPropertyAction(this, &SpectraPhysicsHub::OnKeyswitchInterlock);
		ret = CreateIntegerProperty("Keyswitch Interlock", 0, true, pActKeyswitchInterlock);
		if (ret != 0)
			return ret;
		ret = AddAllowedValue("Keyswitch Interlock", "0");
		if (ret != 0)
			return ret;
		ret = AddAllowedValue("Keyswitch Interlock", "1");
		if (ret != 0)
			return ret;

		// Configure Power supply interlock flag property (read-only)
		CPropertyAction* pActPowerSupplyInterlock = new CPropertyAction(this, &SpectraPhysicsHub::OnPowerSupplyInterlock);
		ret = CreateIntegerProperty("Power supply Interlock", 0, true, pActPowerSupplyInterlock);
		if (ret != 0)
			return ret;
		ret = AddAllowedValue("Power supply Interlock", "0");
		if (ret != 0)
			return ret;
		ret = AddAllowedValue("Power supply Interlock", "1");
		if (ret != 0)
			return ret;

		// Configure Internal interlock flag property (read-only)
		CPropertyAction* pActInternalInterlock = new CPropertyAction(this, &SpectraPhysicsHub::OnInternalInterlock);
		ret = CreateIntegerProperty("Internal Interlock", 0, true, pActInternalInterlock);
		if (ret != 0)
			return ret;
		ret = AddAllowedValue("Internal Interlock", "0");
		if (ret != 0)
			return ret;
		ret = AddAllowedValue("Internal Interlock", "1");
		if (ret != 0)
			return ret;

		// Configure laser state property (read-only).
        // Only the InSight's *STB? response carries the 7-bit Laser State field
		CPropertyAction* pActLaserState = new CPropertyAction(this, &SpectraPhysicsHub::OnLaserState);
		ret = CreateStringProperty("Laser State", "", true, pActLaserState);
		if (ret != 0)
			return ret;
    }

    initialized_ = true;

	// Start the watchdog thread. We Start it after setting initialized_ so Shutdown cleans it up.
    LogMessage("Starting watchdog thread...");
	ret = watchdog_.Start();
	if (ret != DEVICE_OK)
		return ret;
    LogMessage("Started watchdog thread");

    return DEVICE_OK;
}

int SpectraPhysicsHub::Shutdown()
{
    if (initialized_)
    {
		watchdog_.Stop();

        if (onClose_ == g_Standby) {
            // Disable the watchdog timer
            int	ret = ExecuteCommand("TIM:WATC 0");
            if (ret != 0)
                return ret;
        }
        else {
            // Turn off the laser
            int ret = SetProperty("Laser", g_Off);
            if (ret != 0)
                return ret;
            // TODO: Should we wait?
        }

        initialized_ = false;
    }
    return DEVICE_OK;
}

bool SpectraPhysicsHub::Busy()
{
    if (model_ == MAITAI) {
        bool isBusy{};
        int ret = StatusBit(6, isBusy);
        if (ret != 0) {
            return false;
        }
        return isBusy;
    }
    else if (model_ == INSIGHT)
    {
		if (lastCommand_ == "ON")
		{
			int state{};
			int ret = LaserState(state);
			if (ret != 0)
				return false;
			return state != 50;
		}
		else if (lastCommand_.rfind("WAV ", 0) == 0) {
			int state{};
			int ret = LaserState(state);
			if (ret != 0)
				return false;
			return state != 25;
		}
		else if (lastCommand_ == "SHUT 1")
		{
			// From the manual: "it is normal for the system to return 0 for approximately 1 second
			// after issuing the SHUTter 1 command"
			std::chrono::duration elapsed = std::chrono::steady_clock::now() - lastCommandTime_;
			if (elapsed < std::chrono::seconds(1))
				return true;
			// Now return busy until the Main Status Bit is True
			bool open;
			int ret = StatusBit(2, open);
			if (ret != 0)
				return false;
			return !open;
		}
		else if (lastCommand_ == "SHUT 0")
		{
			// NOTE: The manual doesn't explicitly suggest that we need to wait for 1 second here.
			// But that could be an oversight in the manual.

			// Return busy until the Main Shutter Status Bit is False
			bool open;
			int ret = StatusBit(2, open);
			if (ret != 0)
				return false;
			return open;
		}
		else if (lastCommand_ == "IRSHUT 1")
		{
			// From the manual: "it is normal for the system to return 0 for approximately 1 second
			// after issuing the IRSHUTter 1 command"
			std::chrono::duration elapsed = std::chrono::steady_clock::now() - lastCommandTime_;
			if (elapsed < std::chrono::seconds(1))
				return true;
			// Now return busy until the IRShutter Status Bit is True
			bool open;
			int ret = StatusBit(3, open);
			if (ret != 0)
				return false;
			return !open;
		}
		else if (lastCommand_ == "IRSHUT 0")
		{
			// From the manual: "it is normal for the system to return 1 for approximately 1 second
			// after issuing the IRSHUTter 0 command."
			std::chrono::duration elapsed = std::chrono::steady_clock::now() - lastCommandTime_;
			if (elapsed < std::chrono::seconds(1))
				return true;
			// Now return busy until the IRShutter Status Bit is False
			bool open;
			int ret = StatusBit(3, open);
			if (ret != 0)
				return false;
			return open;
		}
    }
    return false;
}

int SpectraPhysicsHub::DetectInstalledDevices()
{
    MM::Device *laser = CreateDevice(g_DeviceNameShutter);
    if (laser)
        AddInstalledDevice(laser);

    // TODO: Is there a way to detect whether the 1040nm option is installed?
    if (model_ == INSIGHT) {
        MM::Device* laser1040 = CreateDevice(g_DeviceNameInsight1040);
        if (laser1040)
            AddInstalledDevice(laser1040);
    }
    return DEVICE_OK;
}

int SpectraPhysicsHub::OnPort(MM::PropertyBase * pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(port_.c_str());
	}
	else if (eAct == MM::AfterSet)
	{
		if (initialized_)
		{
			// revert
			pProp->Set(port_.c_str());
			return ERR_PORT_CHANGE_FORBIDDEN;
		}
		pProp->Get(port_);
	}

	return DEVICE_OK;
}

int SpectraPhysicsHub::OnWatchdog(MM::PropertyBase * pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		pProp->Set(onClose_.c_str());
	}
	else if (eAct == MM::AfterSet)
	{
		pProp->Get(onClose_);
	}

	return DEVICE_OK;
}

int SpectraPhysicsHub::OnWarmup(MM::PropertyBase * pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
        std::string warmupPct;
        int ret = ExecuteCommand("READ:PCTW?", warmupPct);
        if (ret != 0)
            return ret;
		pProp->Set(warmupPct.c_str());
	}
	return DEVICE_OK;
}

int SpectraPhysicsHub::OnHumidity(MM::PropertyBase * pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
        std::string humidityPct;
        int ret = ExecuteCommand("READ:HUM?", humidityPct);
        if (ret != 0)
            return ret;
		pProp->Set(humidityPct.c_str());
	}
	return DEVICE_OK;
}

int SpectraPhysicsHub::OnDiode1Current(MM::PropertyBase * pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
        std::string diode1Current;
        int ret = ExecuteCommand("READ:PLAS:DIOD1:CURR?", diode1Current);
        if (ret != 0)
            return ret;
		pProp->Set(diode1Current.c_str());
	}
	return DEVICE_OK;
}

int SpectraPhysicsHub::OnDiode2Current(MM::PropertyBase * pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
        std::string diode2Current;
        int ret = ExecuteCommand("READ:PLAS:DIOD2:CURR?", diode2Current);
        if (ret != 0)
            return ret;
		pProp->Set(diode2Current.c_str());
	}
	return DEVICE_OK;
}

int SpectraPhysicsHub::OnDiode1Temp(MM::PropertyBase * pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
        std::string diode1Temperature;
        int ret = ExecuteCommand("READ:PLAS:DIOD1:TEMP?", diode1Temperature);
        if (ret != 0)
            return ret;
		pProp->Set(diode1Temperature.c_str());
	}
	return DEVICE_OK;
}

int SpectraPhysicsHub::OnDiode2Temp(MM::PropertyBase * pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
        std::string diode2Temperature;
        int ret = ExecuteCommand("READ:PLAS:DIOD2:TEMP?", diode2Temperature);
        if (ret != 0)
            return ret;
		pProp->Set(diode2Temperature.c_str());
	}
	return DEVICE_OK;
}

int SpectraPhysicsHub::OnDiode1Hours(MM::PropertyBase * pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
        std::string diode1Hours;
        int ret = ExecuteCommand("READ:PLAS:DIOD1:HOURS?", diode1Hours);
        if (ret != 0)
            return ret;
		pProp->Set(diode1Hours.c_str());
	}
	return DEVICE_OK;
}

int SpectraPhysicsHub::OnDiode2Hours(MM::PropertyBase * pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
        std::string diode2Hours;
        int ret = ExecuteCommand("READ:PLAS:DIOD2:HOURS?", diode2Hours);
        if (ret != 0)
            return ret;
		pProp->Set(diode2Hours.c_str());
	}
	return DEVICE_OK;
}

int SpectraPhysicsHub::OnPower(MM::PropertyBase * pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
        std::string power;
        int ret = ExecuteCommand("READ:POW?", power);
        if (ret != 0)
            return ret;
		pProp->Set(power.c_str());
	}
	return DEVICE_OK;
}

int SpectraPhysicsHub::OnHistoryBuffer(MM::PropertyBase * pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
        std::string power;
        // NOTE that the manual sometimes uses READ:HIS?. It seems that is a typo.
        // This comes from previous laser APIs, which had both raw bytes (READ:HIS?) and
        // ASCII (READ:AHIS?) versions.
        int ret = ExecuteCommand("READ:AHIS?", power);
        if (ret != 0)
            return ret;
		pProp->Set(power.c_str());
	}
	return DEVICE_OK;
}

int SpectraPhysicsHub::OnLaserState(MM::PropertyBase * pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
        int state{};
        int ret = LaserState(state);
        if (ret != 0)
            return ret;
        std::string strState{};
        if (state < 25) {
            strState = "Initializing";
        }
        else if (state == 25) {
            strState = "Ready";
        }
        else if (25 < state && state < 50) {
            strState = "Turning on and/or optimizing";
        }
        else if (state == 50) {
            strState = "Running";
        }
        else if (50 < state && state < 60) {
            strState = "Moving to Align mode";
        }
        else if (state == 60) {
            strState = "Aligning";
        }
        else if (60 < state && state < 70) {
            strState = "Exiting Align mode";
        }
        else if (70 <= state) {
            strState = "Unknown state";
        }
        strState += " (" + std::to_string(state) + ")";
		pProp->Set(strState.c_str());
	}
	return DEVICE_OK;
}

int SpectraPhysicsHub::OnEmission(MM::PropertyBase * pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
        bool isEmitting{};
        int ret = StatusBit(0, isEmitting);
        if (ret != 0) {
            return ret;
        }
		pProp->Set(isEmitting ? "1" : "0");
	}
	return DEVICE_OK;
}

int SpectraPhysicsHub::OnPulsing(MM::PropertyBase * pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
        bool isPulsing{};
        int ret = StatusBit(1, isPulsing);
        if (ret != 0) {
            return ret;
        }
		pProp->Set(isPulsing ? "1" : "0");
	}
	return DEVICE_OK;
}

int SpectraPhysicsHub::OnServoOn(MM::PropertyBase * pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
        bool isServoOn{};
        int ret = StatusBit(5, isServoOn);
        if (ret != 0) {
            return ret;
        }
		pProp->Set(isServoOn ? "1" : "0");
	}
	return DEVICE_OK;
}

int SpectraPhysicsHub::OnUserInterlock(MM::PropertyBase * pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
        bool isUserInterlock{};
        int ret = StatusBit(9, isUserInterlock);
        if (ret != 0) {
            return ret;
        }
		pProp->Set(isUserInterlock ? "1" : "0");
	}
	return DEVICE_OK;
}

int SpectraPhysicsHub::OnKeyswitchInterlock(MM::PropertyBase * pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
        bool isKeyswitch{};
        int ret = StatusBit(10, isKeyswitch);
        if (ret != 0) {
            return ret;
        }
		pProp->Set(isKeyswitch ? "1" : "0");
	}
	return DEVICE_OK;
}

int SpectraPhysicsHub::OnPowerSupplyInterlock(MM::PropertyBase * pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
        bool isPowerSupply{};
        int ret = StatusBit(11, isPowerSupply);
        if (ret != 0) {
            return ret;
        }
		pProp->Set(isPowerSupply ? "1" : "0");
	}
	return DEVICE_OK;
}

int SpectraPhysicsHub::OnInternalInterlock(MM::PropertyBase * pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
        bool isInternal{};
        int ret = StatusBit(12, isInternal);
        if (ret != 0) {
            return ret;
        }
		pProp->Set(isInternal ? "1" : "0");
	}
	return DEVICE_OK;
}

int SpectraPhysicsHub::OnWarning(MM::PropertyBase * pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
        bool isWarning{};
        int ret{};
        if (model_ == INSIGHT) {
            ret = StatusBit(14, isWarning);
        }
        else if (model_ == MAITAI) {
            ret = StatusBit(3, isWarning);
        }
        if (ret != 0) {
            return ret;
        }
		pProp->Set(isWarning ? "1" : "0");
	}
	return DEVICE_OK;
}

int SpectraPhysicsHub::OnFault(MM::PropertyBase * pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
        bool isFault{};
        int ret{};
        if (model_ == INSIGHT) {
            ret = StatusBit(15, isFault);
        }
        else if (model_ == MAITAI) {
            ret = StatusBit(4, isFault);
        }
        if (ret != 0) {
            return ret;
        }
		pProp->Set(isFault ? "1" : "0");
	}
	return DEVICE_OK;
}

int SpectraPhysicsHub::OnPumpLaser(MM::PropertyBase * pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
		// Bit 0 identifies energized state, 1 means energized
        bool isEnergized;
		int ret = StatusBit(0, isEnergized);
        if (ret != 0)
            return ret;
        pProp->Set(isEnergized ? g_On : g_Off);
	}
	else if (eAct == MM::AfterSet)
	{
        std::string cmd;
        pProp->Get(cmd);
        if (cmd == g_On)
        {
            // Ensure that the laser is warmed up - otherwise the command will be ignored
            std::string warmupStr{};
            int ret{};
			ret = ExecuteCommand("READ:PCTW?", warmupStr);
			if (ret != 0)
				return ret;

            int warmupPct{};
            try {
                warmupPct = stoi(warmupStr);
            }
            catch (std::exception&) {
                return DEVICE_ERR;
            }
			if (warmupPct < 100)
				return ERR_PUMP_LASER_NOT_WARM;
            // On the InSight, we can also check the laser state for value 25, indicating
            // "READY to turn on (i.e. the laser is fully warmed up".
            if (model_ == INSIGHT) {
                int state{};
                ret = LaserState(state);
                if (ret != 0)
                    return ret;
                if (state != 25)
                    return ERR_WAVELENGTH_CHANGING;
            }
            // Passed checks, turning on
            return ExecuteCommand("ON");
		}
        else
        {
            return ExecuteCommand("OFF");
        }
	}

	return DEVICE_OK;
}

/**
 * This function not only sends the command but also reads back an answer from the device
 */
int SpectraPhysicsHub::ExecuteCommand(const std::string& cmd, std::string& answer)
{
    std::lock_guard<std::mutex> guard(serialMutex_);
	int ret = SendCommand(cmd);
	if (ret != DEVICE_OK)
		return ret;

    // Answers always end with a line feed.
	return GetSerialAnswer(port_.c_str(), "\n", answer);
}

/**
 * This function sends the command to the laser
 */
int SpectraPhysicsHub::ExecuteCommand(const std::string& cmd)
{
    std::lock_guard<std::mutex> guard(serialMutex_);
    return SendCommand(cmd);
}

/**
 * This function actually sends the command to the laser.
 * It should not be called directly - call ExecuteCommand instead!
 */
int SpectraPhysicsHub::SendCommand(const std::string& cmd)
{
	// Send command
	LogMessage("Sending command " + cmd, true);
    int ret = SendSerialCommand(port_.c_str(), cmd.c_str(), "\r");
    if (ret != 0)
        return ret;
    // Update state for Busy checks
    lastCommand_ = cmd;
	lastCommandTime_ = std::chrono::steady_clock::now();

    return DEVICE_OK;
}

int SpectraPhysicsHub::StatusBit(unsigned int bitNumber, bool& bit)
{
    // Returns an integer value that corresponds to a 32-bit binary number
    std::string status_int;
    int ret = ExecuteCommand("*STB?", status_int);
    if (ret != 0)
        return ret;

    unsigned int status_byte{};
    try {
        status_byte = stoul(status_int);
    }
    catch (std::exception&) {
        return DEVICE_ERR;
    }
    unsigned int bit_mask = 1u << bitNumber;
    bit = (status_byte & bit_mask) != 0;
    return DEVICE_OK;
}

int SpectraPhysicsHub::LaserState(int& state)
{
    if (model_ != INSIGHT) {
        // Only the InSight has the Laser State field...
        return ERR_INVALID_MODEL;
    }
    std::string status_int;
    int ret = ExecuteCommand("*STB?", status_int);
    if (ret != 0)
        return ret;

    unsigned int status_byte{};
    try {
        status_byte = stoul(status_int);
    }
    catch (std::exception&) {
        return DEVICE_ERR;
    }
    unsigned int bit_mask = 0x007F0000;
    state = (status_byte & bit_mask) >> 16;
    return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////////
// MAIN SHUTTER
////////////////////////////////////////////////////////////////////////////////////

SpectraPhysicsMain::SpectraPhysicsMain() :
    initialized_(false),
	parent_(nullptr)
{
    SetErrorText(ERR_PUMP_LASER_TURNING_ON, "Cannot open the shutter before the laser is on");
}

SpectraPhysicsMain::~SpectraPhysicsMain()
{
    Shutdown();
}

int SpectraPhysicsMain::Initialize() {
    MM::Hub* hub = GetParentHub();
    if (!hub)
        return ERR_NO_HUB;
    parent_ = dynamic_cast<SpectraPhysicsHub*>(hub);
    if (!parent_)
        return ERR_NO_HUB;

    // Configure target wavelength property
    std::string wave_min_str{}, wave_max_str{};
    int ret{};
    ret = parent_->ExecuteCommand("WAV:min?", wave_min_str);
    if (ret != 0)
        return ret;
    ret = parent_->ExecuteCommand("WAV:max?", wave_max_str);
    if (ret != 0)
        return ret;
	CPropertyAction* pActTargetWavelength = new CPropertyAction(this, &SpectraPhysicsMain::OnTargetWavelength);
    ret = CreateIntegerProperty("Target Wavelength (nm)", 800, false, pActTargetWavelength);
    if (ret != 0)
        return ret;
    int wave_min{}, wave_max{};
    try {
        wave_min = stoi(wave_min_str);
        wave_max = stoi(wave_max_str);
    }
    catch (std::exception&) {
        return DEVICE_ERR;
    }
    ret = SetPropertyLimits("Target Wavelength (nm)", wave_min, wave_max);
    if (ret != 0)
        return ret;

    // Configure actual wavelength property
	CPropertyAction* pActActualWavelength = new CPropertyAction(this, &SpectraPhysicsMain::OnActualWavelength);
    ret = CreateIntegerProperty("Actual Wavelength (nm)", 800, true, pActActualWavelength);
    if (ret != 0)
        return ret;

    // Configure state property
	CPropertyAction* pActState = new CPropertyAction(this, &SpectraPhysicsMain::OnState);
    ret = CreateIntegerProperty(MM::g_Keyword_State, 0, false, pActState);
    if (ret != 0)
        return ret;
    ret = AddAllowedValue(MM::g_Keyword_State, "0");
    if (ret != 0)
        return ret;
    ret = AddAllowedValue(MM::g_Keyword_State, "1");
    if (ret != 0)
        return ret;

    initialized_ = true;
    return DEVICE_OK;
}

int SpectraPhysicsMain::Shutdown()
{
    if (initialized_)
    {
        if (parent_->onClose_ != g_Standby) {
			int ret = SetOpen(false);
            if (ret != 0)
                return ret;
            // TODO: Should we wait?
        }
        initialized_ = false;
    }
    return DEVICE_OK;
}

void SpectraPhysicsMain::GetName(char* name) const
{
    CDeviceUtils::CopyLimitedString(name, g_DeviceNameShutter);
}

bool SpectraPhysicsMain::Busy()
{
    // TODO: We could probably be more granular about this.
    return parent_->Busy();
}


int SpectraPhysicsMain::SetOpen(bool open)
{
    if (open) {
        if (parent_->model_ == INSIGHT) {
			int ret{}, state{};
			ret = parent_->LaserState(state);
			if (ret != 0)
				return ret;
			if (state != 50)
				return ERR_PUMP_LASER_TURNING_ON;
        }
        return parent_->ExecuteCommand("SHUT 1");
    }
    else
        return parent_->ExecuteCommand("SHUT 0");
}

int SpectraPhysicsMain::GetOpen(bool& open)
{
    // Bit 2 identifies the main shutter, 1 means it is open
    return parent_->StatusBit(2, open);
}

int SpectraPhysicsMain::Fire(double deltaT)
{
   return DEVICE_UNSUPPORTED_COMMAND;
}

int SpectraPhysicsMain::OnTargetWavelength(MM::PropertyBase * pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
        std::string wavelength;
        int ret = parent_->ExecuteCommand("WAV?", wavelength);
        if (ret != 0)
            return ret;
		pProp->Set(wavelength.c_str());
	}
	else if (eAct == MM::AfterSet)
	{
        std::string cmd;
        pProp->Get(cmd);
        cmd = "WAV " + cmd;
        return parent_->ExecuteCommand(cmd);
	}

	return DEVICE_OK;
}

int SpectraPhysicsMain::OnActualWavelength(MM::PropertyBase * pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
        std::string wavelength;
        int ret = parent_->ExecuteCommand("READ:WAV?", wavelength);
        if (ret != 0)
            return ret;
		pProp->Set(wavelength.c_str());
	}

	return DEVICE_OK;
}

int SpectraPhysicsMain::OnState(MM::PropertyBase * pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
        bool isOpen{};
        int ret = GetOpen(isOpen);
        if (ret != 0)
            return ret;
		pProp->Set(isOpen? "1" : "0");
	}
	else if (eAct == MM::AfterSet)
	{
        std::string cmd;
        pProp->Get(cmd);
        int ret = SetOpen(cmd == "1");
        if (ret != 0)
            return ret;
        GetCoreCallback()->OnShutterOpenChanged(this, cmd == "1");
	}

	return DEVICE_OK;
}



////////////////////////////////////////////////////////////////////////////////////
// 1040nm SHUTTER
////////////////////////////////////////////////////////////////////////////////////

SpectraPhysicsInsight1040::SpectraPhysicsInsight1040() :
    initialized_(false),
	parent_(nullptr)
{
    SetErrorText(ERR_PUMP_LASER_TURNING_ON, "Cannot open the shutter before the laser is on");
}

SpectraPhysicsInsight1040::~SpectraPhysicsInsight1040()
{
    Shutdown();
}

int SpectraPhysicsInsight1040::Initialize() {
    MM::Hub* hub = GetParentHub();
    if (!hub)
        return ERR_NO_HUB;
    parent_ = dynamic_cast<SpectraPhysicsHub*>(hub);
    if (!parent_)
        return ERR_NO_HUB;

    // Configure state property
    int ret{};
	CPropertyAction* pActState = new CPropertyAction(this, &SpectraPhysicsInsight1040::OnState);
    ret = CreateIntegerProperty(MM::g_Keyword_State, 0, false, pActState);
    if (ret != 0)
        return ret;
    ret = AddAllowedValue(MM::g_Keyword_State, "0");
    if (ret != 0)
        return ret;
    ret = AddAllowedValue(MM::g_Keyword_State, "1");
    if (ret != 0)
        return ret;

	initialized_ = true;
    return DEVICE_OK;
}

int SpectraPhysicsInsight1040::Shutdown()
{
    if (initialized_)
    {
        if (parent_->onClose_ != g_Standby) {
			int ret = SetOpen(false);
            if (ret != 0)
                return ret;
            // TODO: Should we wait?
        }
        initialized_ = false;
    }
    return DEVICE_OK;
}

void SpectraPhysicsInsight1040::GetName(char* name) const
{
    CDeviceUtils::CopyLimitedString(name, g_DeviceNameInsight1040);
}

bool SpectraPhysicsInsight1040::Busy()
{
    // TODO: We could probably be more granular about this.
    return parent_->Busy();
}

int SpectraPhysicsInsight1040::SetOpen(bool open)
{
    if (open) {
        int ret{}, state{};
		ret = parent_->LaserState(state);
        if (ret != 0)
            return ret;
		if (state != 50)
			return ERR_PUMP_LASER_TURNING_ON;
        return parent_->ExecuteCommand("IRSHUT 1");
    }
    else
        return parent_->ExecuteCommand("IRSHUT 0");
}

int SpectraPhysicsInsight1040::GetOpen(bool& open)
{
    // Bit 3 identifies the IR shutter, 1 means it is open
    return parent_->StatusBit(3, open);
}

int SpectraPhysicsInsight1040::Fire(double deltaT)
{
    return DEVICE_UNSUPPORTED_COMMAND;
}

int SpectraPhysicsInsight1040::OnState(MM::PropertyBase * pProp, MM::ActionType eAct)
{
	if (eAct == MM::BeforeGet)
	{
        bool isOpen{};
        int ret = GetOpen(isOpen);
        if (ret != 0)
            return ret;
		pProp->Set(isOpen? "1" : "0");
	}
	else if (eAct == MM::AfterSet)
	{
        std::string cmd;
        pProp->Get(cmd);
        int ret = SetOpen(cmd == "1");
        if (ret != 0)
            return ret;
        GetCoreCallback()->OnShutterOpenChanged(this, cmd == "1");
	}

	return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////////
// WATCHDOG THREAD
////////////////////////////////////////////////////////////////////////////////////

Watchdog::Watchdog(SpectraPhysicsHub& device) :
    device_(device),
    stop_(false),
    interval_(std::chrono::seconds(1))
{
}

Watchdog::~Watchdog() {
    Stop();
}

int Watchdog::Start()
{
    if (thread_.joinable())
        return DEVICE_OK; // Already running

    {
        std::lock_guard<std::mutex> lock(stopMutex_);
        stop_ = false;
    }
    try
    {
        thread_ = std::thread(&Watchdog::Run, this);
    }
    catch (const std::system_error&)
    {
        return ERR_WATCHDOG_START_FAILED;
    }
    return DEVICE_OK;
}

/*
 * This function tells the thread to cease sending keepalive commands
 * and blocks until the last keepalive command has been sent.
 * It is safe to call this function if the thread was never started or has already been stopped.
 */
void Watchdog::Stop()
{
    {
        std::lock_guard<std::mutex> lock(stopMutex_);
        stop_ = true;
    }
    timerCV_.notify_all();
    if (thread_.joinable())
        thread_.join();
}

void Watchdog::Run() {
    while (true)
    {
        {
            std::unique_lock<std::mutex> lock(stopMutex_);
            // This call blocks until:
            // (1) stop_ is true (Stop() sets this and notifies the CV to wake us early)
            // (2) The duration interval_ elapses
            // The predicate handles spurious wakeups by re-sleeping if stop_ is still false.
            timerCV_.wait_for(lock, interval_, [this] { return stop_; });
            if (stop_)
            {
                device_.LogMessage("Watchdog: stop requested, exiting", true);
                break;
            }
        }
        // Send a benign serial command to the laser.
		device_.LogMessage("Watchdog: pinging laser", true);
		std::string status;
		device_.ExecuteCommand("*STB?", status);
    }
}
