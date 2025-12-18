///////////////////////////////////////////////////////////////////////////////
// FILE:          EvidentHubWin.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Evident IX85 microscope hub implementation
//
// COPYRIGHT:     University of California, San Francisco, 2025
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
// AUTHOR:        Nico Stuurman, 2025

#include "EvidentHubWin.h"
#include "ModuleInterface.h"
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <chrono>

using namespace EvidentIX85Win;

const char* g_HubDeviceName = "IX85Win-Hub";

// Device names
extern const char* g_FocusDeviceName;
extern const char* g_NosepieceDeviceName;
extern const char* g_MagnificationDeviceName;
extern const char* g_LightPathDeviceName;
extern const char* g_CondenserTurretDeviceName;
extern const char* g_DIAShutterDeviceName;
extern const char* g_EPIShutter1DeviceName;
extern const char* g_EPIShutter2DeviceName;
extern const char* g_MirrorUnit1DeviceName;
extern const char* g_MirrorUnit2DeviceName;
extern const char* g_PolarizerDeviceName;
extern const char* g_DICPrismDeviceName;
extern const char* g_EPINDDeviceName;
extern const char* g_CorrectionCollarDeviceName;
extern const char* g_AutofocusDeviceName;
extern const char* g_OffsetLensDeviceName;
extern const char* g_ZDCVirtualOffsetDeviceName;
extern const char* g_ObjectiveSetupDeviceName;

// Property names
const char* g_PropPort = "SerialPort";
const char* g_PropAnswerTimeout = "AnswerTimeout";
const char* g_PropDLLPath = "SDK_DLL_Path";
extern const char* g_Keyword_Magnification;

// Hand Switch (MCZ) property names
const char* g_PropHandSwitchJog = "HandSwitch-FocusJog";
const char* g_PropHandSwitchSwitches = "HandSwitch-Switches";
const char* g_PropHandSwitchCondenser = "HandSwitch-CondenserSwitch";
const char* g_PropHandSwitchIndicators = "HandSwitch-Indicators";

// Hand Switch property values
const char* g_Disabled = "Disabled";
const char* g_Enabled = "Enabled";
const char* g_IndicatorNormal = "Normal";
const char* g_IndicatorDark = "Dark";

EvidentHubWin::EvidentHubWin() :
    initialized_(false),
    port_(""),
    dllPath_("IX5_Library\\msl_pm_ix85.dll"),
    answerTimeoutMs_(ANSWER_TIMEOUT_MS),
    dllHandle_(NULL),
    interfaceHandle_(nullptr),
    pfnInitialize_(nullptr),
    pfnEnumInterface_(nullptr),
    pfnGetInterfaceInfo_(nullptr),
    pfnGetPortName_(nullptr),
    pfnOpenInterface_(nullptr),
    pfnCloseInterface_(nullptr),
    pfnSendCommand_(nullptr),
    pfnRegisterCallback_(nullptr),
    responseReady_(false),
    rememberedDIABrightness_(255)
{
    InitializeDefaultErrorMessages();

    // Custom error messages
    SetErrorText(ERR_COMMAND_TIMEOUT, "Command timeout - no response from microscope");
    SetErrorText(ERR_NEGATIVE_ACK, "Microscope returned error (negative acknowledgement)");
    SetErrorText(ERR_INVALID_RESPONSE, "Invalid response from microscope");
    SetErrorText(ERR_NOT_IN_REMOTE_MODE, "Microscope not in remote mode");
    SetErrorText(ERR_DEVICE_NOT_AVAILABLE, "Device not available on this microscope");
    SetErrorText(ERR_POSITION_UNKNOWN, "Device position is unknown");
    SetErrorText(ERR_PORT_NOT_SET, "Serial port not set");
    SetErrorText(ERR_PORT_CHANGE_FORBIDDEN, "Cannot change serial port after initialization");

    // SDK-specific error messages
    SetErrorText(EvidentSDK::SDK_ERR_DLL_NOT_FOUND, "Evident SDK DLL not found - check DLL path");
    SetErrorText(EvidentSDK::SDK_ERR_DLL_INIT_FAILED, "SDK initialization failed");
    SetErrorText(EvidentSDK::SDK_ERR_FUNCTION_NOT_FOUND, "SDK function not found in DLL");
    SetErrorText(EvidentSDK::SDK_ERR_NO_INTERFACE, "No SDK interface found for selected port");
    SetErrorText(EvidentSDK::SDK_ERR_OPEN_FAILED, "Failed to open SDK interface");
    SetErrorText(EvidentSDK::SDK_ERR_SEND_FAILED, "SDK command send failed");
    SetErrorText(EvidentSDK::SDK_ERR_CALLBACK_FAILED, "SDK callback registration failed");

    // Pre-initialization properties
    CPropertyAction* pAct = new CPropertyAction(this, &EvidentHubWin::OnSerialPort);
    CreateProperty(g_PropPort, "Undefined", MM::String, false, pAct, true);

    // Enumerate available COM ports on Windows
    // This allows pre-initialization port selection without requiring
    // MMCore SerialManager ports to be loaded first.
    AddAllowedValue(g_PropPort, "Undefined");

    for (int i = 1; i <= 64; i++)  // Scan COM1-COM64
    {
        std::ostringstream oss;
        oss << "COM" << i;
        std::string portName = oss.str();
        std::string portPath = "\\\\.\\" + portName;

        HANDLE hCom = CreateFile(portPath.c_str(),
                                 GENERIC_READ | GENERIC_WRITE,
                                 0,                          // Exclusive access
                                 NULL,                       // Default security
                                 OPEN_EXISTING,
                                 FILE_ATTRIBUTE_NORMAL,
                                 NULL);

        if (hCom != INVALID_HANDLE_VALUE)
        {
            AddAllowedValue(g_PropPort, portName.c_str());
            CloseHandle(hCom);
        }
    }

    pAct = new CPropertyAction(this, &EvidentHubWin::OnDLLPath);
    CreateProperty(g_PropDLLPath, dllPath_.c_str(), MM::String, false, pAct, true);

    pAct = new CPropertyAction(this, &EvidentHubWin::OnAnswerTimeout);
    CreateProperty(g_PropAnswerTimeout, "4000", MM::Integer, false, pAct, true);
}

EvidentHubWin::~EvidentHubWin()
{
    Shutdown();
}

void EvidentHubWin::GetName(char* pszName) const
{
    CDeviceUtils::CopyLimitedString(pszName, g_HubDeviceName);
}

bool EvidentHubWin::Busy()
{
    return false;  // Hub itself is never busy
}

int EvidentHubWin::Initialize()
{
    if (initialized_)
        return DEVICE_OK;

    usedDevices_.clear();

    // Start worker thread for command queue
    workerRunning_ = true;
    workerThread_ = std::thread(&EvidentHubWin::CommandWorkerThread, this);

    // Load Evident SDK DLL
    int ret = LoadSDK();
    if (ret != DEVICE_OK)
        return ret;

    // Enumerate and open SDK interface
    ret = EnumerateAndOpenInterface();
    if (ret != DEVICE_OK)
    {
        UnloadSDK();
        return ret;
    }

    // Switch to remote mode
    ret = SetRemoteMode();
    if (ret != DEVICE_OK)
        return ret;

    // Exit Setting mode (SDK enters it automatically after login)
    ret = ExitSettingMode();
    if (ret != DEVICE_OK)
        return ret;

    ret = GetUnit(unit_);
    if (ret != DEVICE_OK)
        return ret;

    LogMessage(("Microscope Version: " + version_).c_str(), false);
    LogMessage(("Microscope Unit: " + unit_).c_str(), false);

    // Detect available devices
    ret = DoDeviceDetection();
    if (ret != DEVICE_OK)
       return ret;

    // Initialize MCU indicators if MCU is present
    if (model_.IsDevicePresent(DeviceType_ManualControl))
    {
        // Initialize nosepiece indicator (I1)
        if (model_.IsDevicePresent(DeviceType_Nosepiece))
        {
            long pos = model_.GetPosition(DeviceType_Nosepiece);
            // Position will be 0 if unknown (not yet queried), display as unknown
            UpdateNosepieceIndicator(pos == 0 ? -1 : static_cast<int>(pos));
        }
        else
        {
            // No nosepiece, display "---"
            UpdateNosepieceIndicator(-1);
        }

        // Initialize mirror unit indicator (I2)
        if (model_.IsDevicePresent(DeviceType_MirrorUnit1))
        {
            long pos = model_.GetPosition(DeviceType_MirrorUnit1);
            // Position will be 0 if unknown (not yet queried), display as unknown
            UpdateMirrorUnitIndicator(pos == 0 ? -1 : static_cast<int>(pos), false);
        }
        else
        {
            // No mirror unit, display "---"
            UpdateMirrorUnitIndicator(-1, false); 
        }

        // Enable encoder E1 for nosepiece control if nosepiece is present
        if (model_.IsDevicePresent(DeviceType_Nosepiece))
        {
            std::string cmd = BuildCommand(CMD_ENCODER1, 1);  // Enable encoder
            std::string response;
            ret = ExecuteCommand(cmd, response);
            if (ret == DEVICE_OK)
            {
                // Verify response is "E1 0"
                std::vector<std::string> params = ParseParameters(response);
                if (params.size() > 0 && params[0] == "0")
                {
                    LogMessage("Encoder E1 enabled for nosepiece control", false);
                }
                else
                {
                    LogMessage(("Unexpected response to E1 enable: " + response).c_str(), false);
                }
            }
            else
            {
                LogMessage("Failed to enable encoder E1", false);
            }
        }

        // Enable encoder E2 for mirror unit control if mirror unit is present
        if (model_.IsDevicePresent(DeviceType_MirrorUnit1))
        {
            std::string cmd = BuildCommand(CMD_ENCODER2, 1);  // Enable encoder
            std::string response;
            ret = ExecuteCommand(cmd, response);
            if (ret == DEVICE_OK)
            {
                // Verify response is "E2 0"
                std::vector<std::string> params = ParseParameters(response);
                if (params.size() > 0 && params[0] == "0")
                {
                    LogMessage("Encoder E2 enabled for mirror unit control", false);
                }
                else
                {
                    LogMessage(("Unexpected response to E2 enable: " + response).c_str(), false);
                }
            }
            else
            {
                LogMessage("Failed to enable encoder E2", false);
            }
        }

        // Enable encoder E3 for DIA brightness control if DIA shutter is present
        if (model_.IsDevicePresent(DeviceType_DIAShutter))
        {
            std::string cmd = BuildCommand(CMD_ENCODER3, 1);  // Enable encoder
            std::string response;
            ret = ExecuteCommand(cmd, response);
            if (ret == DEVICE_OK)
            {
                // Verify response is "E3 0"
                std::vector<std::string> params = ParseParameters(response);
                if (params.size() > 0 && params[0] == "0")
                {
                    LogMessage("Encoder E3 enabled for DIA brightness control", false);
                }
                else
                {
                    LogMessage(("Unexpected response to E3 enable: " + response).c_str(), false);
                }
            }
            else
            {
                LogMessage("Failed to enable encoder E3", false);
            }
        }

        // Enable jog (focus) control if MCU is present
        if (model_.IsDevicePresent(DeviceType_ManualControl))
        {
            std::string cmd = BuildCommand(CMD_JOG, 1);  // Enable jog control
            std::string response;
            ret = ExecuteCommand(cmd, response);
            if (ret == DEVICE_OK)
            {
                LogMessage("Jog (focus) control enabled on MCU", false);
            }
            else
            {
                LogMessage("Failed to enable jog control", false);
            }
        }

        // Enable MCU switches (S2) if MCU is present
        if (model_.IsDevicePresent(DeviceType_ManualControl))
        {
            std::string cmd = BuildCommand(CMD_MCZ_SWITCH, 1);  // Enable switches
            std::string response;
            ret = ExecuteCommand(cmd, response);
            if (ret == DEVICE_OK)
            {
                LogMessage("MCU switches (S2) enabled", false);
            }
            else
            {
                LogMessage("Failed to enable MCU switches", false);
            }
        }

        // Enable condenser switch (S1) if condenser is present
        if (model_.IsDevicePresent(DeviceType_CondenserTurret))
        {
            std::string cmd = BuildCommand(CMD_CONDENSER_SWITCH, 1);  // Enable switches
            std::string response;
            ret = ExecuteCommand(cmd, response);
            if (ret == DEVICE_OK)
            {
                LogMessage("Condenser switches (S1) enabled", false);
            }
            else
            {
                LogMessage("Failed to enable condenser switches", false);
            }
        }

        // Enable indicators (I) with normal intensity
        if (model_.IsDevicePresent(DeviceType_ManualControl))
        {
            std::string cmd = BuildCommand(CMD_INDICATOR_CONTROL, 1);  // 1 = Normal intensity
            std::string response;
            ret = ExecuteCommand(cmd, response);
            if (ret == DEVICE_OK)
            {
                LogMessage("MCU indicators enabled (Normal)", false);
            }
            else
            {
                LogMessage("Failed to enable MCU indicators", false);
            }
        }

        // Enable objective dial request notification (NROB) if nosepiece is present
        if (model_.IsDevicePresent(DeviceType_Nosepiece))
        {
            std::string cmd = BuildCommand(CMD_NOSEPIECE_REQUEST_NOTIFY, 1);
            std::string response;
            ret = ExecuteCommand(cmd, response);
            if (ret == DEVICE_OK)
            {
                LogMessage("Objective dial notification (NROB) enabled", false);
            }
            else
            {
                LogMessage("Failed to enable objective dial notification", false);
            }
        }

        // Enable mirror dial request notification (NRMU) if mirror unit is present
        if (model_.IsDevicePresent(DeviceType_MirrorUnit1))
        {
            std::string cmd = BuildCommand(CMD_MIRROR_REQUEST_NOTIFY, 1);
            std::string response;
            ret = ExecuteCommand(cmd, response);
            if (ret == DEVICE_OK)
            {
                LogMessage("Mirror dial notification (NRMU) enabled", false);
            }
            else
            {
                LogMessage("Failed to enable mirror dial notification", false);
            }
        }

        // Initialize light path indicator (I4)
        if (model_.IsDevicePresent(DeviceType_LightPath))
        {
            long pos = model_.GetPosition(DeviceType_LightPath);
            // Position will be 0 if unknown (not yet queried), display as unknown (all off)
            UpdateLightPathIndicator(pos == 0 ? -1 : static_cast<int>(pos), false);
        }
        else
        {
            // No light path, display all off
            UpdateLightPathIndicator(-1, false);
        }

        // Initialize EPI shutter 1 indicator (I5)
        if (model_.IsDevicePresent(DeviceType_EPIShutter1))
        {
            long state = model_.GetPosition(DeviceType_EPIShutter1);
            // Position will be 0 if unknown (not yet queried), display as closed (I5 1)
            UpdateEPIShutter1Indicator(state == 0 ? 0 : static_cast<int>(state), false);
        }
        else
        {
            // No EPI shutter 1, display as closed
            UpdateEPIShutter1Indicator(0, false);
        }

        // Create Hand Switch control properties
        CPropertyAction* pAct = new CPropertyAction(this, &EvidentHubWin::OnHandSwitchJog);
        CreateProperty(g_PropHandSwitchJog, g_Enabled, MM::String, false, pAct);
        AddAllowedValue(g_PropHandSwitchJog, g_Disabled);
        AddAllowedValue(g_PropHandSwitchJog, g_Enabled);

        pAct = new CPropertyAction(this, &EvidentHubWin::OnHandSwitchSwitches);
        CreateProperty(g_PropHandSwitchSwitches, g_Enabled, MM::String, false, pAct);
        AddAllowedValue(g_PropHandSwitchSwitches, g_Disabled);
        AddAllowedValue(g_PropHandSwitchSwitches, g_Enabled);

        if (model_.IsDevicePresent(DeviceType_CondenserTurret))
        {
            pAct = new CPropertyAction(this, &EvidentHubWin::OnHandSwitchCondenser);
            CreateProperty(g_PropHandSwitchCondenser, g_Enabled, MM::String, false, pAct);
            AddAllowedValue(g_PropHandSwitchCondenser, g_Disabled);
            AddAllowedValue(g_PropHandSwitchCondenser, g_Enabled);
        }

        pAct = new CPropertyAction(this, &EvidentHubWin::OnHandSwitchIndicators);
        CreateProperty(g_PropHandSwitchIndicators, g_IndicatorNormal, MM::String, false, pAct);
        AddAllowedValue(g_PropHandSwitchIndicators, g_Disabled);
        AddAllowedValue(g_PropHandSwitchIndicators, g_IndicatorNormal);
        AddAllowedValue(g_PropHandSwitchIndicators, g_IndicatorDark);
    }

    initialized_ = true;
    return DEVICE_OK;
}

int EvidentHubWin::Shutdown()
{
    if (initialized_)
    {
        // Disable all active notifications BEFORE stopping monitoring thread
        for (auto deviceType : availableDevices_)
        {
            // Disable notifications for devices that support them
            switch (deviceType)
            {
                case DeviceType_Focus:
                    EnableNotification(CMD_FOCUS_NOTIFY, false);
                    break;
                case DeviceType_Nosepiece:
                    EnableNotification(CMD_NOSEPIECE_NOTIFY, false);
                    break;
                case DeviceType_Magnification:
                    EnableNotification(CMD_MAGNIFICATION_NOTIFY, false);
                    break;
                // Add more as needed
                default:
                    break;
            }
        }

        // Disable encoder E1 if MCU is present
        if (model_.IsDevicePresent(DeviceType_ManualControl))
        {
            std::string cmd = BuildCommand(CMD_ENCODER1, 0);  // Disable encoder
            std::string response;
            int ret = ExecuteCommand(cmd, response);
            if (ret == DEVICE_OK)
            {
                LogMessage("Encoder E1 disabled", true);
            }

            else
            {
                LogMessage("Failed to disable encoder E1", true);
            }
        }

        // Disable encoder E2 if MCU is present
        if (model_.IsDevicePresent(DeviceType_ManualControl))
        {
            std::string cmd = BuildCommand(CMD_ENCODER2, 0);  // Disable encoder
            std::string response;
            int ret = ExecuteCommand(cmd, response);
            if (ret == DEVICE_OK)
            {
                LogMessage("Encoder E2 disabled", true);
            }
            else
            {
                LogMessage("Failed to disable encoder E2", true);
            }
        }

        // Disable encoder E3 if MCU is present
        if (model_.IsDevicePresent(DeviceType_ManualControl))
        {
            std::string cmd = BuildCommand(CMD_ENCODER3, 0);  // Disable encoder
            std::string response;
            int ret = ExecuteCommand(cmd, response);
            if (ret == DEVICE_OK)
            {
                LogMessage("Encoder E3 disabled", true);
            }
            else
            {
                LogMessage("Failed to disable encoder E3", true);
            }
        }

        // Disable jog control if MCU is present
        if (model_.IsDevicePresent(DeviceType_ManualControl))
        {
            std::string cmd = BuildCommand(CMD_JOG, 0);  // Disable jog control
            std::string response;
            int ret = ExecuteCommand(cmd, response);
            if (ret == DEVICE_OK)
            {
                LogMessage("Jog (focus) control disabled", true);
            }
            else
            {
                LogMessage("Failed to disable jog control", true);
            }
        }

        // Disable MCU switches (S2) if MCU is present
        if (model_.IsDevicePresent(DeviceType_ManualControl))
        {
            std::string cmd = BuildCommand(CMD_MCZ_SWITCH, 0);  // Disable switches
            std::string response;
            int ret = ExecuteCommand(cmd, response);
            if (ret == DEVICE_OK)
            {
                LogMessage("MCU switches (S2) disabled", true);
            }
            else
            {
                LogMessage("Failed to disable MCU switches", true);
            }
        }

        // Disable condenser switches (S1) if condenser is present
        if (model_.IsDevicePresent(DeviceType_CondenserTurret))
        {
            std::string cmd = BuildCommand(CMD_CONDENSER_SWITCH, 0);  // Disable switches
            std::string response;
            int ret = ExecuteCommand(cmd, response);
            if (ret == DEVICE_OK)
            {
                LogMessage("Condenser switches (S1) disabled", true);
            }
            else
            {
                LogMessage("Failed to disable condenser switches", true);
            }
        }

        // Disable indicators (I)
        if (model_.IsDevicePresent(DeviceType_ManualControl))
        {
            std::string cmd = BuildCommand(CMD_INDICATOR_CONTROL, 0);  // 0 = Disable
            std::string response;
            int ret = ExecuteCommand(cmd, response);
            if (ret == DEVICE_OK)
            {
                LogMessage("MCU indicators disabled", true);
            }
            else
            {
                LogMessage("Failed to disable MCU indicators", true);
            }
        }

        // Switch back to local mode
        std::string cmd = BuildCommand(CMD_LOGIN, 0);  // 0 = Local mode
        std::string response;
        ExecuteCommand(cmd, response);

        // Stop worker thread
        workerRunning_ = false;
        queueCV_.notify_one();

        // Wait for worker thread to finish
        if (workerThread_.joinable())
        {
            workerThread_.join();
        }

        // Clear any pending commands
        {
            std::lock_guard<std::mutex> lock(queueMutex_);
            while (!commandQueue_.empty())
            {
               auto& task = commandQueue_.front();
               // Set exception on unfulfilled promises to prevent blocking
               try
               {
                  task.responsePromise.set_exception(
                     std::make_exception_ptr(std::runtime_error("Hub shutting down")));
               }
               catch (...)
               {
                  // Promise might already be fulfilled, ignore
               }
               commandQueue_.pop();
            }
        }

        // Close SDK interface and unload DLL
        if (interfaceHandle_ != nullptr)
        {
            if (pfnCloseInterface_ != nullptr)
            {
                pfnCloseInterface_(interfaceHandle_);
            }
            interfaceHandle_ = nullptr;
        }
        UnloadSDK();

        initialized_ = false;
    }
    return DEVICE_OK;
}

int EvidentHubWin::OnSerialPort(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(port_.c_str());
    }
    else if (eAct == MM::AfterSet)
    {
        if (initialized_)
        {
            pProp->Set(port_.c_str());
            return ERR_PORT_CHANGE_FORBIDDEN;
        }
        pProp->Get(port_);
    }
    return DEVICE_OK;
}

int EvidentHubWin::OnDLLPath(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(dllPath_.c_str());
    }
    else if (eAct == MM::AfterSet)
    {
        if (initialized_)
        {
            pProp->Set(dllPath_.c_str());
            return ERR_PORT_CHANGE_FORBIDDEN;
        }
        pProp->Get(dllPath_);
    }
    return DEVICE_OK;
}

int EvidentHubWin::OnAnswerTimeout(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(answerTimeoutMs_);
    }
    else if (eAct == MM::AfterSet)
    {
        pProp->Get(answerTimeoutMs_);
    }
    return DEVICE_OK;
}

int EvidentHubWin::OnHandSwitchJog(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::AfterSet)
    {
        std::string value;
        pProp->Get(value);
        int enable = (value == g_Enabled) ? 1 : 0;
        std::string cmd = BuildCommand(CMD_JOG, enable);
        std::string response;
        int ret = ExecuteCommand(cmd, response);
        if (ret != DEVICE_OK)
            return ret;
        if (!IsPositiveAck(response, CMD_JOG))
            return ERR_INVALID_RESPONSE;
    }
    return DEVICE_OK;
}

int EvidentHubWin::OnHandSwitchSwitches(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::AfterSet)
    {
        std::string value;
        pProp->Get(value);
        int enable = (value == g_Enabled) ? 1 : 0;
        std::string cmd = BuildCommand(CMD_MCZ_SWITCH, enable);
        std::string response;
        int ret = ExecuteCommand(cmd, response);
        if (ret != DEVICE_OK)
            return ret;
        if (!IsPositiveAck(response, CMD_MCZ_SWITCH))
            return ERR_INVALID_RESPONSE;
    }
    return DEVICE_OK;
}

int EvidentHubWin::OnHandSwitchCondenser(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::AfterSet)
    {
        std::string value;
        pProp->Get(value);
        int enable = (value == g_Enabled) ? 1 : 0;
        std::string cmd = BuildCommand(CMD_CONDENSER_SWITCH, enable);
        std::string response;
        int ret = ExecuteCommand(cmd, response);
        if (ret != DEVICE_OK)
            return ret;
        if (!IsPositiveAck(response, CMD_CONDENSER_SWITCH))
            return ERR_INVALID_RESPONSE;
    }
    return DEVICE_OK;
}

int EvidentHubWin::OnHandSwitchIndicators(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::AfterSet)
    {
        std::string value;
        pProp->Get(value);
        int mode = 0;  // Disabled
        if (value == g_IndicatorNormal)
            mode = 1;
        else if (value == g_IndicatorDark)
            mode = 2;
        std::string cmd = BuildCommand(CMD_INDICATOR_CONTROL, mode);
        std::string response;
        int ret = ExecuteCommand(cmd, response);
        if (ret != DEVICE_OK)
            return ret;
        if (!IsPositiveAck(response, CMD_INDICATOR_CONTROL))
            return ERR_INVALID_RESPONSE;
    }
    return DEVICE_OK;
}

int EvidentHubWin::SetRemoteMode()
{
    std::string cmd = BuildCommand(CMD_LOGIN, 1);  // 1 = Remote mode
    std::string response;
    int ret = ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
        return ret;

    if (!IsPositiveAck(response, CMD_LOGIN))
        return ERR_NOT_IN_REMOTE_MODE;

    return DEVICE_OK;
}

int EvidentHubWin::SetSettingMode(bool enable)
{
    std::string cmd = BuildCommand(CMD_OPERATION_MODE, enable ? 1 : 0);
    std::string response;
    int ret = ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
        return ret;

    if (!IsPositiveAck(response, CMD_OPERATION_MODE))
    {
        LogMessage(("SetSettingMode failed, response: " + response).c_str(), false);
        return ERR_INVALID_RESPONSE;
    }

    LogMessage(enable ? "Entered Setting mode" : "Exited Setting mode", false);
    return DEVICE_OK;
}

int EvidentHubWin::EnterSettingMode()
{
    return SetSettingMode(true);
}

int EvidentHubWin::ExitSettingMode()
{
    return SetSettingMode(false);
}

int EvidentHubWin::GetUnit(std::string& unit)
{
    std::string cmd = BuildQuery(CMD_UNIT);
    std::string response;
    int ret = ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
        return ret;

    // Parse response: "U IX5,..."
    std::vector<std::string> params = ParseParameters(response);
    if (params.size() > 0)
    {
        unit = params[0];
        return DEVICE_OK;
    }

    return ERR_INVALID_RESPONSE;
}

int EvidentHubWin::GetUnitDirect(std::string& unit)
{
    std::string cmd = BuildQuery(CMD_UNIT);
    std::string response;
    int ret = SendCommand(cmd);
    if (ret != DEVICE_OK)
        return ret;
    ret = GetSerialAnswer(port_.c_str(), TERMINATOR, response);
    if (ret != DEVICE_OK)
       return ret;

    // Parse response: "U IX5,..."
    std::vector<std::string> params = ParseParameters(response);
    if (params.size() > 0)
    {
        unit = params[0];
        return DEVICE_OK;
    }

    return ERR_INVALID_RESPONSE;
}

int EvidentHubWin::ExecuteCommandInternal(const std::string& command, std::string& response)
{
   // Worker thread provides serialization, no mutex needed here

   // Retry logic for empty responses (device not ready)
   const int MAX_RETRIES = 20;
   const int RETRY_DELAY_MS = 50;  // 50ms delay between retries

   for (int attempt = 0; attempt < MAX_RETRIES; attempt++)
   {
      // Reset response state BEFORE sending command to avoid race condition
      // where callback fires before GetResponse can set responseReady_ = false
      {
         std::lock_guard<std::mutex> responseLock(responseMutex_);
         responseReady_ = false;
         pendingResponse_.clear();
      }

      int ret = SendCommand(command);
      if (ret != DEVICE_OK)
      {
         return ret;
      }

      // Extract expected response tag from command
      // First strip any trailing whitespace/terminators from the command
      std::string cleanCommand = command;
      size_t end = cleanCommand.find_last_not_of(" \t\r\n");
      if (end != std::string::npos)
      {
         cleanCommand = cleanCommand.substr(0, end + 1);
      }
      std::string expectedTag = ExtractTag(cleanCommand);

      ret = GetResponse(response, answerTimeoutMs_);
      if (ret != DEVICE_OK)
      {
         return ret;
      }

      // Verify response tag matches command tag
      std::string responseTag = ExtractTag(response);

      if (responseTag == expectedTag)
      {
         if (!IsPositiveAck(response, expectedTag.c_str()))
         {
            // Check if device is busy (error code 70)
            if (IsDeviceBusyError(response) && attempt < MAX_RETRIES - 1)
            {
               LogMessage(("Device " + expectedTag + " busy(attempt " +
                  std::to_string(attempt + 1) + "), retrying...").c_str(), true);
               CDeviceUtils::SleepMs(RETRY_DELAY_MS);
               continue;  // Retry
            }
         }
      }
      else // if (responseTag != expectedTag)
      {
         // Received wrong response - this can happen if responses arrive out of order
         LogMessage(("Warning: Expected response for '" + expectedTag +
            "' but received '" + responseTag + "' (" + response +
            "). Discarding and waiting for correct response.").c_str(), false);

         // Wait for the correct response (with remaining timeout)
         ret = GetResponse(response, answerTimeoutMs_);
         if (ret != DEVICE_OK)
         {
            return ret;
         }

         // Check again
         responseTag = ExtractTag(response);
         if (responseTag != expectedTag)
         {
            LogMessage(("Error: Still did not receive expected response for '" +
               expectedTag + "', got '" + responseTag + "' instead.").c_str(), false);
            return ERR_INVALID_RESPONSE;
         }
      }

      return DEVICE_OK;
   }
   return DEVICE_OK;
}

std::future<std::pair<int, std::string>> EvidentHubWin::ExecuteCommandAsync(const std::string& command)
{
   CommandTask task(command);
   auto future = task.responsePromise.get_future();

   {
      std::lock_guard<std::mutex> lock(queueMutex_);
      commandQueue_.push(std::move(task));
   }
   queueCV_.notify_one();

   return future;
}

int EvidentHubWin::ExecuteCommand(const std::string& command, std::string& response)
{
   // Submit to queue and wait for completion
   auto future = ExecuteCommandAsync(command);

   // Block until response ready
   auto result = future.get();

   // Extract return code and response
   int ret = result.first;
   response = result.second;

   return ret;
}

void EvidentHubWin::CommandWorkerThread()
{
   while (workerRunning_)
   {
      CommandTask task("");

      // Wait for command in queue
      {
         std::unique_lock<std::mutex> lock(queueMutex_);
         queueCV_.wait(lock, [this]
         {
            return !commandQueue_.empty() || !workerRunning_;
         });

         if (!workerRunning_ && commandQueue_.empty())
         {
            break;  // Shutdown signal
         }

         if (commandQueue_.empty())
         {
            continue;
         }

         // Move task out of queue
         task = std::move(commandQueue_.front());
         commandQueue_.pop();
      }

      // Execute command (outside queue lock to allow new submissions)
      std::string response;

      try
      {
         int ret = ExecuteCommandInternal(task.command, response);
         task.responsePromise.set_value(std::make_pair(ret, response));
      }
      catch (...)
      {
         task.responsePromise.set_exception(std::current_exception());
      }
   }
}

int EvidentHubWin::SendCommand(const std::string& command)
{
    // std::string cmdString = command +  TERMINATOR;
    LogMessage(("Sending: " + command).c_str(), true);

    // Verify SDK is initialized
    if (interfaceHandle_ == nullptr || pfnSendCommand_ == nullptr)
    {
        LogMessage("SDK not initialized or interface not open", false);
        return DEVICE_ERR;
    }

    // Initialize SDK command structure (use member to stay valid for async callback)
    // commandMutex_ must be held by caller to protect pendingCommand_
    EvidentSDK::InitCommand(pendingCommand_);
    EvidentSDK::SetCommandString(pendingCommand_, command);

    // Send command via SDK
    if (!pfnSendCommand_(interfaceHandle_, &pendingCommand_))
    {
       LogMessage("SDK SendCommand failed, retrying...", false);
       CDeviceUtils::SleepMs(50);
       if (!pfnSendCommand_(interfaceHandle_, &pendingCommand_))
       {
          LogMessage("SDK SendCommand failed again.");
          // go nuclear, unload and reload interface
          if (interfaceHandle_ != nullptr)
          {
              if (pfnCloseInterface_ != nullptr)
              {
                  pfnCloseInterface_(interfaceHandle_);
              }
              interfaceHandle_ = nullptr;
          }
          this->EnumerateAndOpenInterface();
          if (!pfnSendCommand_(interfaceHandle_, &pendingCommand_))
          {
             return EvidentSDK::SDK_ERR_SEND_FAILED;
          }
       }
    }

    return DEVICE_OK;
}

int EvidentHubWin::GetResponse(std::string& response, long timeoutMs)
{
    if (timeoutMs < 0)
        timeoutMs = answerTimeoutMs_;

    // Wait for the callback to provide a response (responseReady_ already reset by ExecuteCommand)
    std::unique_lock<std::mutex> lock(responseMutex_);

    if (responseCV_.wait_for(lock, std::chrono::milliseconds(timeoutMs),
        [this] { return responseReady_; }))
    {
        response = pendingResponse_;
        LogMessage(("Received: " + response).c_str(), true);
        return DEVICE_OK;
    }

    return ERR_COMMAND_TIMEOUT;
}

int EvidentHubWin::DoDeviceDetection()
{
    availableDevices_.clear();
    detectedDevicesByName_.clear();

    // Use U command to detect device presence
    // The U command returns a comma-separated list of unit codes
    std::string cmd = BuildQuery(CMD_UNIT);
    std::string response;
    int ret = ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
    {
        LogMessage("Failed to query unit codes", false);
        return ret;
    }

    // Parse response: "U IX5,FRM,REA,LWUCDA,..."
    std::vector<std::string> unitCodes = ParseParameters(response);

    // Log all detected unit codes
    std::ostringstream unitLog;
    unitLog << "Unit codes detected: ";
    for (size_t i = 0; i < unitCodes.size(); i++)
    {
        if (i > 0) unitLog << ", ";
        unitLog << unitCodes[i];
    }
    LogMessage(unitLog.str().c_str(), false);

    // Helper lambda to check if a unit code is present
    auto hasUnit = [&unitCodes](const std::string& code) -> bool {
        for (const auto& unit : unitCodes)
        {
            if (unit == code) return true;
        }
        return false;
    };

    // IX5 - Focus, Light Path, Correction Collar, Magnification
    if (hasUnit("IX5"))
    {
        LogMessage("Detected IX5 unit (Focus, LightPath, CorrectionCollar, Magnification)");

        // Focus
        model_.SetDevicePresent(DeviceType_Focus, true);
        availableDevices_.push_back(DeviceType_Focus);
        detectedDevicesByName_.push_back(g_FocusDeviceName);
        QueryFocus();

        // Light Path
        model_.SetDevicePresent(DeviceType_LightPath, true);
        availableDevices_.push_back(DeviceType_LightPath);
        detectedDevicesByName_.push_back(g_LightPathDeviceName);
        QueryLightPath();

        // Correction Collar
        model_.SetDevicePresent(DeviceType_CorrectionCollar, true);
        availableDevices_.push_back(DeviceType_CorrectionCollar);
        detectedDevicesByName_.push_back(g_CorrectionCollarDeviceName);
        model_.SetPosition(DeviceType_CorrectionCollar, 0);

        // Magnification
        model_.SetDevicePresent(DeviceType_Magnification, true);
        availableDevices_.push_back(DeviceType_Magnification);
        detectedDevicesByName_.push_back(g_MagnificationDeviceName);
        QueryMagnification();
    }

    // REA - Nosepiece
    if (hasUnit("REA"))
    {
        LogMessage("Detected REA unit (Nosepiece)");
        model_.SetDevicePresent(DeviceType_Nosepiece, true);
        availableDevices_.push_back(DeviceType_Nosepiece);
        detectedDevicesByName_.push_back(g_NosepieceDeviceName);
        QueryNosepiece();
        QueryObjectiveInfo();
    }

    // LWUCDA - Condenser Turret, DIA Aperture, Polarizer, DIA Shutter
    if (hasUnit("LWUCDA"))
    {
        LogMessage("Detected LWUCDA unit (CondenserTurret, Polarizer, DIAShutter)");

        // Polarizer
        model_.SetDevicePresent(DeviceType_Polarizer, true);
        availableDevices_.push_back(DeviceType_Polarizer);
        detectedDevicesByName_.push_back(g_PolarizerDeviceName);
        QueryPolarizer();

        // Condenser Turret
        model_.SetDevicePresent(DeviceType_CondenserTurret, true);
        availableDevices_.push_back(DeviceType_CondenserTurret);
        detectedDevicesByName_.push_back(g_CondenserTurretDeviceName);
        QueryCondenserTurret();

        // DIA Shutter
        model_.SetDevicePresent(DeviceType_DIAShutter, true);
        availableDevices_.push_back(DeviceType_DIAShutter);
        detectedDevicesByName_.push_back(g_DIAShutterDeviceName);
        QueryDIAShutter();
    }

    // DICTA - DIC Prism and Retardation
    if (hasUnit("DICTA"))
    {
        LogMessage("Detected DICTA unit (DICPrism)");
        model_.SetDevicePresent(DeviceType_DICPrism, true);
        availableDevices_.push_back(DeviceType_DICPrism);
        detectedDevicesByName_.push_back(g_DICPrismDeviceName);
        QueryDICPrism();
    }

    // RFACA.1 - Mirror Unit 1 and EPI Shutter 1
    if (hasUnit("RFACA.1"))
    {
        LogMessage("Detected RFACA.1 unit (MirrorUnit1, EPIShutter1)");

        // Mirror Unit 1
        model_.SetDevicePresent(DeviceType_MirrorUnit1, true);
        availableDevices_.push_back(DeviceType_MirrorUnit1);
        detectedDevicesByName_.push_back(g_MirrorUnit1DeviceName);
        QueryMirrorUnit1();

        // EPI Shutter 1
        model_.SetDevicePresent(DeviceType_EPIShutter1, true);
        availableDevices_.push_back(DeviceType_EPIShutter1);
        detectedDevicesByName_.push_back(g_EPIShutter1DeviceName);
        QueryEPIShutter1();
    }

    // RFACA.2 - Mirror Unit 2 and EPI Shutter 2
    if (hasUnit("RFACA.2"))
    {
        LogMessage("Detected RFACA.2 unit (MirrorUnit2, EPIShutter2)");

        // Mirror Unit 2
        model_.SetDevicePresent(DeviceType_MirrorUnit2, true);
        availableDevices_.push_back(DeviceType_MirrorUnit2);
        detectedDevicesByName_.push_back(g_MirrorUnit2DeviceName);
        QueryMirrorUnit2();

        // EPI Shutter 2
        model_.SetDevicePresent(DeviceType_EPIShutter2, true);
        availableDevices_.push_back(DeviceType_EPIShutter2);
        detectedDevicesByName_.push_back(g_EPIShutter2DeviceName);
        QueryEPIShutter2();
    }

    // MCZ - Manual Control Unit (Hand Switch)
    // Note: Not added to availableDevices_ - properties are added to Hub device instead
    if (hasUnit("MCZ"))
    {
        LogMessage("Detected MCZ unit (ManualControl/HandSwitch)");
        model_.SetDevicePresent(DeviceType_ManualControl, true);
    }

    // ZDC - Autofocus and Offset Lens
    if (hasUnit("ZDC"))
    {
        LogMessage("Detected ZDC unit (Autofocus, OffsetLens)");

        // Autofocus (includes virtual offset
        model_.SetDevicePresent(DeviceType_Autofocus, true);
        availableDevices_.push_back(DeviceType_Autofocus);
        detectedDevicesByName_.push_back(g_AutofocusDeviceName);
        model_.SetDevicePresent(DeviceType_ZDCVirtualOffset, true);
        availableDevices_.push_back(DeviceType_ZDCVirtualOffset);
        detectedDevicesByName_.push_back(g_ZDCVirtualOffsetDeviceName);

        // Offset Lens
        model_.SetDevicePresent(DeviceType_OffsetLens, true);
        availableDevices_.push_back(DeviceType_OffsetLens);
        detectedDevicesByName_.push_back(g_OffsetLensDeviceName);

        // Query initial offset lens position
        QueryOffsetLens();
    }

    // TODO: U-AW - EPI? (needs clarification)
    // TODO: FRM - unknown devices

    // Objective Setup - Always available (utility device for configuration)
    LogMessage("Adding ObjectiveSetup utility device");
    detectedDevicesByName_.push_back(g_ObjectiveSetupDeviceName);

    std::ostringstream msg;
    msg << "Discovered " << availableDevices_.size() << " devices";
    LogMessage(msg.str().c_str(), false);

    return DEVICE_OK;
}

MM::DeviceDetectionStatus EvidentHubWin::DetectDevice(void)
{
   
   if (initialized_)
      return MM::CanCommunicate;

   // our property port_ should have been set to one of the valid ports


   // all conditions must be satisfied...
   MM::DeviceDetectionStatus result = MM::Misconfigured;
   char answerTO[MM::MaxStrLength];
   
   try
   {
      std::string portLowerCase = port_;
      for( std::string::iterator its = portLowerCase.begin(); its != portLowerCase.end(); ++its)
      {
         *its = (char)tolower(*its);
      }
      if( 0< portLowerCase.length() &&  0 != portLowerCase.compare("undefined")  && 0 != portLowerCase.compare("unknown") )
      {
         result = MM::CanNotCommunicate;
         // record current port settings
         GetCoreCallback()->GetDeviceProperty(port_.c_str(), "AnswerTimeout", answerTO);

         // device specific default communication parameters
         GetCoreCallback()->SetDeviceProperty(port_.c_str(), MM::g_Keyword_BaudRate, "115200" );
         GetCoreCallback()->SetDeviceProperty(port_.c_str(), MM::g_Keyword_StopBits, "1");
         GetCoreCallback()->SetDeviceProperty(port_.c_str(), MM::g_Keyword_Parity, "Even");
         GetCoreCallback()->SetDeviceProperty(port_.c_str(), "Verbose", "0");
         GetCoreCallback()->SetDeviceProperty(port_.c_str(), "AnswerTimeout", "5000.0");
         GetCoreCallback()->SetDeviceProperty(port_.c_str(), "DelayBetweenCharsMs", "0");
         MM::Device* pS = GetCoreCallback()->GetDevice(this, port_.c_str());
         pS->Initialize();
         std::string unit;
         int ret = GetUnitDirect(unit);
         if (ret != DEVICE_OK || unit != "IX5")
         {
            pS->Shutdown();
            // always restore the AnswerTimeout to the default
            GetCoreCallback()->SetDeviceProperty(port_.c_str(), "AnswerTimeout", answerTO);
            return result;
         }
         result = MM::CanCommunicate;
         pS->Shutdown();
         // always restore the AnswerTimeout to the default
         GetCoreCallback()->SetDeviceProperty(port_.c_str(), "AnswerTimeout", answerTO);

      }
   }
   catch(...)
   {
      LogMessage("Exception in DetectDevice!",false);
   }
   return result;

}

int EvidentHubWin::DetectInstalledDevices()
{
    for (size_t i=0; i < detectedDevicesByName_.size(); i++)
    {
       MM::Device* pDev = ::CreateDevice(detectedDevicesByName_[i].c_str());
       if (pDev)
          AddInstalledDevice(pDev);
    }

    return DEVICE_OK;
}

bool EvidentHubWin::IsDevicePresent(EvidentIX85Win::DeviceType type) const
{
    return model_.IsDevicePresent(type);
}

std::string EvidentHubWin::GetDeviceVersion(EvidentIX85Win::DeviceType type) const
{
    return model_.GetDeviceVersion(type);
}

int EvidentHubWin::EnableNotification(const char* cmd, bool enable)
{
    std::string command = BuildCommand(cmd, enable ? 1 : 0);
    std::string response;

    int ret = ExecuteCommand(command, response);
    if (ret != DEVICE_OK)
        return ret;

    if (!IsPositiveAck(response, cmd))
        return ERR_NEGATIVE_ACK;

    return DEVICE_OK;
}

// Device query implementations
int EvidentHubWin::QueryFocus()
{
    std::string cmd = BuildQuery(CMD_FOCUS_POSITION);
    std::string response;
    int ret = ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
        return ret;

    if (IsUnknown(response))
        return ERR_DEVICE_NOT_AVAILABLE;

    std::vector<std::string> params = ParseParameters(response);
    if (params.size() > 0 && params[0] != "X")
    {
        long pos = ParseLongParameter(params[0]);
        model_.SetPosition(DeviceType_Focus, pos);
        model_.SetLimits(DeviceType_Focus, FOCUS_MIN_POS, FOCUS_MAX_POS);
        return DEVICE_OK;
    }

    return ERR_DEVICE_NOT_AVAILABLE;
}

int EvidentHubWin::QueryNosepiece()
{
    std::string cmd = BuildQuery(CMD_NOSEPIECE);
    std::string response;
    int ret = ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
        return ret;

    if (IsUnknown(response))
        return ERR_DEVICE_NOT_AVAILABLE;

    std::vector<std::string> params = ParseParameters(response);
    if (params.size() > 0 && params[0] != "X")
    {
        int pos = ParseIntParameter(params[0]);
        model_.SetPosition(DeviceType_Nosepiece, pos);
        model_.SetNumPositions(DeviceType_Nosepiece, NOSEPIECE_MAX_POS);
        model_.SetLimits(DeviceType_Nosepiece, NOSEPIECE_MIN_POS, NOSEPIECE_MAX_POS);
        return DEVICE_OK;
    }

    return ERR_DEVICE_NOT_AVAILABLE;
}

int EvidentHubWin::QueryObjectiveInfo()
{
    objectiveInfo_.clear();
    objectiveInfo_.resize(NOSEPIECE_MAX_POS);  // 6 positions

    for (int pos = 1; pos <= NOSEPIECE_MAX_POS; pos++)
    {
        std::string cmd = BuildCommand(CMD_GET_OBJECTIVE, pos);
        std::string response;
        int ret = ExecuteCommand(cmd, response);
        if (ret != DEVICE_OK)
        {
            LogMessage(("Failed to query objective info for position " + std::to_string(pos)).c_str(), false);
            continue;
        }

        std::vector<std::string> params = ParseParameters(response);
        if (params.size() < 11)
        {
            LogMessage(("Incomplete GOB response for position " + std::to_string(pos)).c_str(), false);
            continue;
        }

        EvidentIX85Win::ObjectiveInfo& info = objectiveInfo_[pos - 1];

        // p1 is position (already known), p2 is name
        info.name = params[1];

        // p3: NA (0.00-2.00, N = indefinite)
        if (params[2] != "N")
            info.na = std::stod(params[2]);
        else
            info.na = -1.0;

        // p4: Magnification (0-200, N = indefinite)
        if (params[3] != "N")
            info.magnification = std::stoi(params[3]);
        else
            info.magnification = -1;

        // p5: Medium (1-5, N = indefinite)
        if (params[4] != "N")
            info.medium = std::stoi(params[4]);
        else
            info.medium = -1;

        // p6 is always 0, skip

        // p7: AS min (0-120, N/U = indefinite/unknown)
        if (params[6] != "N" && params[6] != "U")
            info.asMin = std::stoi(params[6]);
        else
            info.asMin = -1;

        // p8: AS max (0-120, N/U = indefinite/unknown)
        if (params[7] != "N" && params[7] != "U")
            info.asMax = std::stoi(params[7]);
        else
            info.asMax = -1;

        // p9: WD (0.01-25.00, N = indefinite)
        if (params[8] != "N")
            info.wd = std::stod(params[8]);
        else
            info.wd = -1.0;

        // p10: ZDC OneShot compatibility (0-3)
        info.zdcOneShotCompat = std::stoi(params[9]);

        // p11: ZDC Continuous compatibility (0-3)
        info.zdcContinuousCompat = std::stoi(params[10]);

        std::ostringstream msg;
        msg << "Objective " << pos << ": " << info.name
            << ", NA=" << info.na
            << ", Mag=" << info.magnification
            << "x, WD=" << info.wd << "mm";
        LogMessage(msg.str().c_str(), false);
    }

    return DEVICE_OK;
}

int EvidentHubWin::QueryMagnification()
{
    std::string cmd = BuildQuery(CMD_MAGNIFICATION);
    std::string response;
    int ret = ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
        return ret;

    if (IsUnknown(response))
        return ERR_DEVICE_NOT_AVAILABLE;

    std::vector<std::string> params = ParseParameters(response);
    if (params.size() > 0 && params[0] != "X")
    {
        int pos = ParseIntParameter(params[0]);
        model_.SetPosition(DeviceType_Magnification, pos);
        model_.SetNumPositions(DeviceType_Magnification, MAGNIFICATION_MAX_POS);
        model_.SetLimits(DeviceType_Magnification, MAGNIFICATION_MIN_POS, MAGNIFICATION_MAX_POS);
        return DEVICE_OK;
    }

    return ERR_DEVICE_NOT_AVAILABLE;
}

int EvidentHubWin::QueryLightPath()
{
    std::string cmd = BuildQuery(CMD_LIGHT_PATH);
    std::string response;
    int ret = ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
        return ret;

    if (IsUnknown(response))
        return ERR_DEVICE_NOT_AVAILABLE;

    std::vector<std::string> params = ParseParameters(response);
    if (params.size() > 0 && params[0] != "X")
    {
        int pos = ParseIntParameter(params[0]);
        model_.SetPosition(DeviceType_LightPath, pos);
        model_.SetNumPositions(DeviceType_LightPath, 4);
        return DEVICE_OK;
    }

    return ERR_DEVICE_NOT_AVAILABLE;
}

int EvidentHubWin::QueryCondenserTurret()
{
    std::string cmd = BuildQuery(CMD_CONDENSER_TURRET);
    std::string response;
    int ret = ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
        return ret;

    if (IsUnknown(response))
        return ERR_DEVICE_NOT_AVAILABLE;

    std::vector<std::string> params = ParseParameters(response);
    if (params.size() > 0 && params[0] != "X")
    {
        int pos = ParseIntParameter(params[0]);
        model_.SetPosition(DeviceType_CondenserTurret, pos);
        model_.SetNumPositions(DeviceType_CondenserTurret, CONDENSER_TURRET_MAX_POS);
        return DEVICE_OK;
    }

    return ERR_DEVICE_NOT_AVAILABLE;
}

int EvidentHubWin::QueryDIAAperture()
{
    std::string cmd = BuildQuery(CMD_DIA_APERTURE);
    std::string response;
    int ret = ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
        return ret;

    if (IsUnknown(response))
        return ERR_DEVICE_NOT_AVAILABLE;

    std::vector<std::string> params = ParseParameters(response);
    if (params.size() > 0 && params[0] != "X")
    {
        int pos = ParseIntParameter(params[0]);
        model_.SetPosition(DeviceType_DIAAperture, pos);
        return DEVICE_OK;
    }

    return ERR_DEVICE_NOT_AVAILABLE;
}

int EvidentHubWin::QueryDIAShutter()
{
    std::string cmd = BuildQuery(CMD_DIA_SHUTTER);
    std::string response;
    int ret = ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
        return ret;

    if (IsUnknown(response))
        return ERR_DEVICE_NOT_AVAILABLE;

    std::vector<std::string> params = ParseParameters(response);
    if (params.size() > 0 && params[0] != "X")
    {
        int state = ParseIntParameter(params[0]);
        model_.SetPosition(DeviceType_DIAShutter, state);
        return DEVICE_OK;
    }

    return ERR_DEVICE_NOT_AVAILABLE;
}

int EvidentHubWin::QueryPolarizer()
{
    std::string cmd = BuildQuery(CMD_POLARIZER);
    std::string response;
    int ret = ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
        return ret;

    // Note: This function is now only called after V7 confirms condenser unit is present
    // May still return "X" on first query due to firmware bug, but that's okay -
    // we'll just set position to 0 (Out) and device will work correctly
    std::vector<std::string> params = ParseParameters(response);
    if (params.size() > 0)
    {
        int pos = ParseIntParameter(params[0]);
        if (pos >= 0)
        {
            model_.SetPosition(DeviceType_Polarizer, pos);
        }
        else
        {
            // Firmware returned "X", set default position (0 = Out)
            model_.SetPosition(DeviceType_Polarizer, 0);
        }
        model_.SetNumPositions(DeviceType_Polarizer, POLARIZER_MAX_POS);
        return DEVICE_OK;
    }

    return DEVICE_OK;  // Device present (confirmed by V7), just couldn't get position
}

int EvidentHubWin::QueryDICPrism()
{
    std::string cmd = BuildQuery(CMD_DIC_PRISM);
    std::string response;
    int ret = ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
        return ret;

    // Note: This function is now only called after V8 confirms DIC unit is present
    // May still return "X" on first query due to firmware bug, but that's okay -
    // we'll just set position to 0 and device will work correctly
    std::vector<std::string> params = ParseParameters(response);
    if (params.size() > 0)
    {
        int pos = ParseIntParameter(params[0]);
        if (pos >= 0)
        {
            model_.SetPosition(DeviceType_DICPrism, pos);
        }
        else
        {
            // Firmware returned "X", set default position (0)
            model_.SetPosition(DeviceType_DICPrism, 0);
        }
        model_.SetNumPositions(DeviceType_DICPrism, DIC_PRISM_MAX_POS);
        return DEVICE_OK;
    }

    return DEVICE_OK;  // Device present (confirmed by V8), just couldn't get position
}

int EvidentHubWin::QueryDICRetardation()
{
    std::string cmd = BuildQuery(CMD_DIC_RETARDATION);
    std::string response;
    int ret = ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
        return ret;

    if (IsUnknown(response))
        return ERR_DEVICE_NOT_AVAILABLE;

    std::vector<std::string> params = ParseParameters(response);
    if (params.size() > 0 && params[0] != "X")
    {
        int pos = ParseIntParameter(params[0]);
        model_.SetPosition(DeviceType_DICRetardation, pos);
        return DEVICE_OK;
    }

    return ERR_DEVICE_NOT_AVAILABLE;
}

int EvidentHubWin::QueryEPIShutter1()
{
    std::string cmd = BuildQuery(CMD_EPI_SHUTTER1);
    std::string response;
    int ret = ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
        return ret;

    if (IsUnknown(response))
        return ERR_DEVICE_NOT_AVAILABLE;

    std::vector<std::string> params = ParseParameters(response);
    if (params.size() > 0 && params[0] != "X")
    {
        int state = ParseIntParameter(params[0]);
        model_.SetPosition(DeviceType_EPIShutter1, state);
        return DEVICE_OK;
    }

    return ERR_DEVICE_NOT_AVAILABLE;
}

int EvidentHubWin::QueryEPIShutter2()
{
    std::string cmd = BuildQuery(CMD_EPI_SHUTTER2);
    std::string response;
    int ret = ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
        return ret;

    if (IsUnknown(response))
        return ERR_DEVICE_NOT_AVAILABLE;

    std::vector<std::string> params = ParseParameters(response);
    if (params.size() > 0 && params[0] != "X")
    {
        int state = ParseIntParameter(params[0]);
        model_.SetPosition(DeviceType_EPIShutter2, state);
        return DEVICE_OK;
    }

    return ERR_DEVICE_NOT_AVAILABLE;
}

int EvidentHubWin::QueryMirrorUnit1()
{
    std::string cmd = BuildQuery(CMD_MIRROR_UNIT1);
    std::string response;
    int ret = ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
        return ret;

    if (IsUnknown(response))
        return ERR_DEVICE_NOT_AVAILABLE;

    std::vector<std::string> params = ParseParameters(response);
    if (params.size() > 0 && params[0] != "X")
    {
        int pos = ParseIntParameter(params[0]);
        model_.SetPosition(DeviceType_MirrorUnit1, pos);
        model_.SetNumPositions(DeviceType_MirrorUnit1, MIRROR_UNIT_MAX_POS);
        return DEVICE_OK;
    }

    return ERR_DEVICE_NOT_AVAILABLE;
}

int EvidentHubWin::QueryMirrorUnit2()
{
    std::string cmd = BuildQuery(CMD_MIRROR_UNIT2);
    std::string response;
    int ret = ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
        return ret;

    if (IsUnknown(response))
        return ERR_DEVICE_NOT_AVAILABLE;

    std::vector<std::string> params = ParseParameters(response);
    if (params.size() > 0 && params[0] != "X")
    {
        int pos = ParseIntParameter(params[0]);
        model_.SetPosition(DeviceType_MirrorUnit2, pos);
        model_.SetNumPositions(DeviceType_MirrorUnit2, MIRROR_UNIT_MAX_POS);
        return DEVICE_OK;
    }

    return ERR_DEVICE_NOT_AVAILABLE;
}

int EvidentHubWin::QueryEPIND()
{
    std::string cmd = BuildQuery(CMD_EPI_ND);
    std::string response;
    int ret = ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
        return ret;

    if (IsUnknown(response))
        return ERR_DEVICE_NOT_AVAILABLE;

    std::vector<std::string> params = ParseParameters(response);
    if (params.size() > 0 && params[0] != "X")
    {
        int pos = ParseIntParameter(params[0]);
        model_.SetPosition(DeviceType_EPIND, pos);
        model_.SetNumPositions(DeviceType_EPIND, EPIND_MAX_POS);
        return DEVICE_OK;
    }

    return ERR_DEVICE_NOT_AVAILABLE;
}

int EvidentHubWin::QueryRightPort()
{
    std::string cmd = BuildQuery(CMD_RIGHT_PORT);
    std::string response;
    int ret = ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
        return ret;

    if (IsUnknown(response))
        return ERR_DEVICE_NOT_AVAILABLE;

    std::vector<std::string> params = ParseParameters(response);
    if (params.size() > 0 && params[0] != "X")
    {
        int pos = ParseIntParameter(params[0]);
        model_.SetPosition(DeviceType_RightPort, pos);
        return DEVICE_OK;
    }

    return ERR_DEVICE_NOT_AVAILABLE;
}

int EvidentHubWin::QueryCorrectionCollar()
{
    std::string cmd = BuildQuery(CMD_CORRECTION_COLLAR);
    std::string response;
    int ret = ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
        return ret;

    if (IsUnknown(response))
        return ERR_DEVICE_NOT_AVAILABLE;

    std::vector<std::string> params = ParseParameters(response);
    if (params.size() > 0 && params[0] != "X")
    {
        int pos = ParseIntParameter(params[0]);
        model_.SetPosition(DeviceType_CorrectionCollar, pos);
        return DEVICE_OK;
    }

    return ERR_DEVICE_NOT_AVAILABLE;
}

int EvidentHubWin::QueryOffsetLens()
{
    std::string cmd = BuildQuery(CMD_OFFSET_LENS_POSITION);
    std::string response;
    int ret = ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
        return ret;

    if (IsUnknown(response))
        return ERR_DEVICE_NOT_AVAILABLE;

    std::vector<std::string> params = ParseParameters(response);
    if (params.size() > 0 && params[0] != "X")
    {
        int pos = ParseIntParameter(params[0]);
        model_.SetPosition(DeviceType_OffsetLens, pos);
        return DEVICE_OK;
    }

    return ERR_DEVICE_NOT_AVAILABLE;
}

int EvidentHubWin::UpdateNosepieceIndicator(int position)
{
    // Check if MCU is present
    if (!model_.IsDevicePresent(DeviceType_ManualControl))
        return DEVICE_OK;  // Not an error, MCU just not present

    std::string cmd;

    // Position -1 means unknown, display "---" (three dashes)
    if (position == -1 || position < 1 || position > 9)
    {
        // Three dashes: 0x01,0x01,0x01
        cmd = "I1 010101";
    }
    else
    {
        // Single digit display - get 7-segment code
        int code = Get7SegmentCode(position);

        // Format as hex string (2 digits)
        std::ostringstream oss;
        oss << "I1 " << std::hex << std::uppercase << std::setfill('0') << std::setw(2) << code;
        cmd = oss.str();
    }

    // Send command without waiting for response (to avoid deadlock when called from monitoring thread)
    // The "I1 +" response will be consumed by the monitoring thread as a pseudo-notification
    int ret = SendCommand(cmd);
    if (ret != DEVICE_OK)
    {
        LogMessage(("Failed to send nosepiece indicator command: " + cmd).c_str());
        return ret;
    }

    LogMessage(("Sent nosepiece indicator command: " + cmd).c_str(), true);
    return DEVICE_OK;
}

int EvidentHubWin::UpdateMirrorUnitIndicator(int position, bool async)
{
    // Check if MCU is present
    if (!model_.IsDevicePresent(DeviceType_ManualControl))
        return DEVICE_OK;  // Not an error, MCU just not present

    std::string cmd;

    // Position -1 means unknown, display "---" (three dashes)
    if (position == -1 || position < 1 || position > 9)
    {
        // Three dashes: 0x01,0x01,0x01
        cmd = "I2 010101";
    }
    else
    {
        // Single digit display - get 7-segment code
        int code = Get7SegmentCode(position);

        // Format as hex string (2 digits)
        std::ostringstream oss;
        oss << "I2 " << std::hex << std::uppercase << std::setfill('0') << std::setw(2) << code;
        cmd = oss.str();
    }

    if (async)
    {
       auto future = ExecuteCommandAsync(cmd);
    }
    else
    {
       std::string response;
       int ret = ExecuteCommand(cmd, response);
       if (ret != DEVICE_OK)
          return ret;

       // Got a response, check if it's positive
       if (!IsPositiveAck(response, "I2"))
       {
          return ERR_NEGATIVE_ACK;
       }
    }

    LogMessage(("Sent mirror unit indicator command async: " + cmd).c_str(), true);
    return DEVICE_OK;
}

int EvidentHubWin::UpdateLightPathIndicator(int position, bool async)
{
    // Check if MCU is present
    if (!model_.IsDevicePresent(DeviceType_ManualControl))
        return DEVICE_OK;  // Not an error, MCU just not present

    std::string cmd;

    // Map LightPath position (1-4) to I4 indicator value
    // Position 1 (Left Port) -> I4 1 (camera)
    // Position 2 (Binocular 50/50) -> I4 2 (50:50)
    // Position 3 (Binocular 100%) -> I4 4 (eyepiece)
    // Position 4 (Right Port) -> I4 0 (all off)
    // Unknown -> I4 0 (all off)

    int i4Value;
    if (position == 1)
        i4Value = 1;  // Left Port -> camera
    else if (position == 2)
        i4Value = 2;  // 50:50
    else if (position == 3)
        i4Value = 4;  // Binocular 100% -> eyepiece
    else
        i4Value = 0;  // Right Port or unknown -> all off

    std::ostringstream oss;
    oss << "I4 " << i4Value;
    cmd = oss.str();

    if (async)
    {
       auto future = ExecuteCommandAsync(cmd);
       LogMessage(("Sent Light Path indicator command async " + cmd).c_str(), true);
    }
    else
    {
       std::string response;
       int ret = ExecuteCommand(cmd, response);
       if (ret != DEVICE_OK)
          return ret;

       // Got a response, check if it's positive
       if (!IsPositiveAck(response, "I4"))
       {
          return ERR_NEGATIVE_ACK;
       }

       LogMessage(("Sent light path indicator command: " + cmd).c_str(), true);
    }
    return DEVICE_OK;
}

int EvidentHubWin::UpdateEPIShutter1Indicator(int state, bool async)
{
    // Check if MCU is present
    if (!model_.IsDevicePresent(DeviceType_ManualControl))
        return DEVICE_OK;  // Not an error, MCU just not present

    std::string cmd;

    // Map EPI Shutter state to I5 indicator value
    // State 0 (Closed) -> I5 1
    // State 1 (Open) -> I5 2
    // Unknown or other -> I5 1 (default to closed)

    int i5Value;
    if (state == 1)
        i5Value = 2;  // Open -> 2
    else
        i5Value = 1;  // Closed or unknown -> 1

    std::ostringstream oss;
    oss << "I5 " << i5Value;
    cmd = oss.str();

    if (async)
    {
       auto future = ExecuteCommandAsync(cmd);
       // we discard the future, since we do not want to block
       LogMessage(("Sent EPI shutter indicator command async: " + cmd).c_str(), true);
    }
    else {
       std::string response;
       int ret = ExecuteCommand(cmd, response);
       if (ret != DEVICE_OK)
          return ret;

       // Got a response, check if it's positive
       if (!IsPositiveAck(response, CMD_EPI_SHUTTER1))
       {
          return ERR_NEGATIVE_ACK;
       }

       LogMessage(("Sent EPI shutter indicator command: " + cmd).c_str(), true);
    }
    return DEVICE_OK;
}

void EvidentHubWin::NotifyMeasuredZOffsetChanged(long offsetSteps)
{
    // Notify EvidentAutofocus device if it's in use
    auto afIt = usedDevices_.find(EvidentIX85Win::DeviceType_Autofocus);
    if (afIt != usedDevices_.end() && afIt->second != nullptr)
    {
        EvidentAutofocus* afDevice = dynamic_cast<EvidentAutofocus*>(afIt->second);
        if (afDevice)
        {
            afDevice->UpdateMeasuredZOffset(offsetSteps);
        }
    }

    auto zfIt = usedDevices_.find(EvidentIX85Win::DeviceType_ZDCVirtualOffset);
    if (zfIt != usedDevices_.end() && zfIt->second != nullptr)
    {
        EvidentZDCVirtualOffset* zdcDevice = dynamic_cast<EvidentZDCVirtualOffset*>(zfIt->second);
        if (zdcDevice)
        {
            zdcDevice->UpdateMeasuredZOffset(offsetSteps);
        }
    }
}

int EvidentHubWin::SetFocusPositionSteps(long steps)
{
    // Check if we're already at the target position
    long currentPos = GetModel()->GetPosition(DeviceType_Focus);
    bool alreadyAtTarget = IsAtTargetPosition(currentPos, steps, FOCUS_POSITION_TOLERANCE);

    // Set target position BEFORE sending command so notifications can check against it
    GetModel()->SetTargetPosition(DeviceType_Focus, steps);
    GetModel()->SetBusy(DeviceType_Focus, true);

    std::string cmd = BuildCommand(CMD_FOCUS_GOTO, static_cast<int>(steps));
    std::string response;
    int ret = ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
    {
        // Command failed - clear busy state
        GetModel()->SetBusy(DeviceType_Focus, false);
        return ret;
    }

    if (!IsPositiveAck(response, CMD_FOCUS_GOTO))
    {
        // Command rejected - clear busy state
        GetModel()->SetBusy(DeviceType_Focus, false);

        // Check for specific error: position out of range
        if (IsPositionOutOfRangeError(response))
        {
            return ERR_POSITION_OUT_OF_RANGE;
        }
        return ERR_NEGATIVE_ACK;
    }

    // If we're already at the target, firmware won't send notifications, so clear busy immediately
    if (alreadyAtTarget)
    {
        GetModel()->SetBusy(DeviceType_Focus, false);
    }

    // Command accepted - if not already at target, busy will be cleared by notification when target reached
    return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// Notification Processing
///////////////////////////////////////////////////////////////////////////////

void EvidentHubWin::ProcessNotification(const std::string& message)
{
   bool isCommandCompletion = false;
    std::string tag = ExtractTag(message);
    std::vector<std::string> params = ParseParameters(message);

    // Update model based on notification
    if (tag == CMD_FOCUS_NOTIFY && params.size() > 0)
    {
        long pos = ParseLongParameter(params[0]);
        if (pos >= 0)
        {
            model_.SetPosition(DeviceType_Focus, pos);

            // Notify core callback of stage position change
            auto it = usedDevices_.find(DeviceType_Focus);
            if (it != usedDevices_.end() && it->second != nullptr)
            {
                // Convert from 10nm units to micrometers
                double positionUm = pos * FOCUS_STEP_SIZE_UM;
                GetCoreCallback()->OnStagePositionChanged(it->second, positionUm);
            }

            // Check if we've reached the target position (with tolerance for mechanical settling)
            long targetPos = model_.GetTargetPosition(DeviceType_Focus);
            if (IsAtTargetPosition(pos, targetPos, FOCUS_POSITION_TOLERANCE))
            {
                model_.SetBusy(DeviceType_Focus, false);
                LogMessage("Focus reached target position", true);
            }
        }
    }
    else if (tag == CMD_NOSEPIECE_NOTIFY && params.size() > 0)
    {
        int pos = ParseIntParameter(params[0]);
        if (pos >= 0)
        {
            model_.SetPosition(DeviceType_Nosepiece, pos);

            // Note: MCU indicator I1 is updated automatically by SDK when using OBSEQ

            // Check if we've reached the target position
            long targetPos = model_.GetTargetPosition(DeviceType_Nosepiece);
            bool isExpectedChange = (targetPos >= 0 && pos == targetPos);

            std::ostringstream msg;
            msg << "Nosepiece notification: pos=" << pos << ", targetPos=" << targetPos << ", isExpectedChange=" << isExpectedChange;
            LogMessage(msg.str().c_str(), true);

            if (isExpectedChange)
            {
                LogMessage("Nosepiece reached target position, clearing busy flag", true);
                model_.SetBusy(DeviceType_Nosepiece, false);
            }

            // Only notify core of property changes for UNSOLICITED changes (manual controls)
            // For commanded changes, the core already knows about the change
            if (!isExpectedChange)
            {
                LogMessage("Unsolicited nosepiece change detected, notifying core", true);
                auto it = usedDevices_.find(DeviceType_Nosepiece);
                if (it != usedDevices_.end() && it->second != nullptr)
                {
                    // Convert from 1-based position to 0-based state value
                    int stateValue = pos - 1;
                    GetCoreCallback()->OnPropertyChanged(it->second, MM::g_Keyword_State,
                        CDeviceUtils::ConvertToString(stateValue));

                    // This should works since the nosepiece is a state device
                    // it would be safer to test the type first
                    char label[MM::MaxStrLength];
                    int ret = ((MM::State*) it->second)->GetPositionLabel(stateValue, label);
                    if (ret == DEVICE_OK)
                       GetCoreCallback()->OnPropertyChanged(it->second, MM::g_Keyword_Label, label);
                }
            }

        }
    }
    else if (tag == CMD_MAGNIFICATION_NOTIFY && params.size() > 0)
    {
        int pos = ParseIntParameter(params[0]);
        if (pos >= 0)
        {
            model_.SetPosition(DeviceType_Magnification, pos);

            std::ostringstream msg;
            msg << "Magnification notification: pos=" << pos;
            LogMessage(msg.str().c_str(), true);

            auto it = usedDevices_.find(DeviceType_Magnification);
            if (it != usedDevices_.end() && it->second != nullptr)
            {
                // Map position (1-based) to magnification value
                const double magnifications[3] = {1.0, 1.6, 2.0};
                if (pos >= 1 && pos <= 3)
                {
                    double magValue = magnifications[pos - 1];
                    std::string logMsg = std::string("Magnification changed to ") + CDeviceUtils::ConvertToString(magValue);
                    LogMessage(logMsg.c_str(), true);

                    GetCoreCallback()->OnPropertyChanged(it->second, g_Keyword_Magnification,
                        CDeviceUtils::ConvertToString(magValue));
                    // Notify core that magnification has changed
                    GetCoreCallback()->OnMagnifierChanged(it->second);
                }
                else
                {
                    std::ostringstream errMsg;
                    errMsg << "Magnification position " << pos << " out of range (expected 1-3)";
                    LogMessage(errMsg.str().c_str(), false);
                }
            }
            else
            {
                LogMessage("Magnification device not registered, cannot notify core", true);
            }
        }
    }
    else if (tag == CMD_OFFSET_LENS_NOTIFY && params.size() > 0)
    {
        // Handle offset lens position change notification
        long pos = ParseLongParameter(params[0]);
        if (pos >= OFFSET_LENS_MIN_POS && pos <= OFFSET_LENS_MAX_POS)
        {
            model_.SetPosition(DeviceType_OffsetLens, pos);

            // Notify core callback of stage position change
            auto it = usedDevices_.find(DeviceType_OffsetLens);
            if (it != usedDevices_.end() && it->second != nullptr)
            {
                // Convert from steps to micrometers
                double positionUm = pos * OFFSET_LENS_STEP_SIZE_UM;
                GetCoreCallback()->OnStagePositionChanged(it->second, positionUm);
            }
        }
    }
    else if (tag == CMD_AF_STATUS && params.size() > 0)
    {
        // Handle AF status notification (NAFST)
        if (params[0] != "X")
        {
            int status = ParseIntParameter(params[0]);

            // Update AF status in the autofocus device
            auto it = usedDevices_.find(DeviceType_Autofocus);
            if (it != usedDevices_.end() && it->second != nullptr)
            {
                EvidentAutofocus* afDevice = dynamic_cast<EvidentAutofocus*>(it->second);
                if (afDevice != nullptr)
                {
                    afDevice->UpdateAFStatus(status);
                }
            }
        }
    }
    else if (tag == CMD_ENCODER1 && params.size() > 0)
    {
        // Encoder 1 controls nosepiece position
        int delta = ParseIntParameter(params[0]);
        if (delta == -1 || delta == 1)
        {
            // Get current nosepiece position
            long currentPos = model_.GetPosition(DeviceType_Nosepiece);

            // If position is unknown (0), we can't handle encoder input
            if (currentPos == 0)
            {
                LogMessage("E1 encoder input ignored - nosepiece position unknown", true);
                return;
            }

            // Calculate new position with wrapping
            int newPos = static_cast<int>(currentPos) + delta;
            int maxPos = NOSEPIECE_MAX_POS;  // 6

            if (newPos < 1)
                newPos = maxPos;  // Wrap from 1 to max
            else if (newPos > maxPos)
                newPos = 1;  // Wrap from max to 1

            // Send OB command to change nosepiece position
            std::string cmd = BuildCommand(CMD_NOSEPIECE, newPos);
            std::ostringstream msg;
            msg << "E1 encoder: moving nosepiece from " << currentPos << " to " << newPos;
            LogMessage(msg.str().c_str(), true);

            // Send the command (fire-and-forget, the NOB notification will update the model)
            auto future = ExecuteCommandAsync(cmd);
            /*
            int ret = SendCommand(cmd);
            if (ret != DEVICE_OK)
            {
                LogMessage("Failed to send nosepiece command from encoder", false);
            }
            */
        }
    }
    else if (tag == CMD_ENCODER2 && params.size() > 0)
    {
        // Encoder 2 controls MirrorUnit1 position
        int delta = ParseIntParameter(params[0]);
        if (delta == -1 || delta == 1)
        {
            // Get current mirror unit position
            long currentPos = model_.GetPosition(DeviceType_MirrorUnit1);

            // If position is unknown (0), we can't handle encoder input
            if (currentPos == 0)
            {
                LogMessage("E2 encoder input ignored - mirror unit position unknown", true);
                return;
            }

            // Calculate new position with wrapping
            int newPos = static_cast<int>(currentPos) + delta;
            int maxPos = MIRROR_UNIT_MAX_POS;  // 6

            if (newPos < 1)
                newPos = maxPos;  // Wrap from 1 to max
            else if (newPos > maxPos)
                newPos = 1;  // Wrap from max to 1

            // Send MU1 command to change mirror unit position
            std::string cmd = BuildCommand(CMD_MIRROR_UNIT1, newPos);
            std::ostringstream msg;
            msg << "E2 encoder: moving mirror unit from " << currentPos << " to " << newPos;
            LogMessage(msg.str().c_str(), true);

            // Response to the command can be read only after we exit this function, so do not 
            // wait for it here, but trust it will be successful.
            auto future = ExecuteCommandAsync(cmd);
             // Optimistically update model and indicator (mirror unit has no position change notifications)
             model_.SetPosition(DeviceType_MirrorUnit1, newPos);
             UpdateMirrorUnitIndicator(newPos, true);

             // Notify core callback of State property change
             auto it = usedDevices_.find(DeviceType_MirrorUnit1);
             if (it != usedDevices_.end() && it->second != nullptr)
             {
                 // Convert from 1-based position to 0-based state value
                 int stateValue = newPos - 1;
                 GetCoreCallback()->OnPropertyChanged(it->second, MM::g_Keyword_State,
                     CDeviceUtils::ConvertToString(stateValue));

                 // This should works since the MirrorUnit is a state device
                 // it would be safer to test the type first
                 char label[MM::MaxStrLength];
                 int ret = ((MM::State*) it->second)->GetPositionLabel(stateValue, label);
                 if (ret == DEVICE_OK)
                    GetCoreCallback()->OnPropertyChanged(it->second, MM::g_Keyword_Label, label);
             }
        }
    }
    else if (tag == CMD_ENCODER3 && params.size() > 0)
    {
        // Encoder 3 controls DIA brightness
        int delta = ParseIntParameter(params[0]);
        // Fast turning can produce delta values from -9 to 9 (excluding 0)
        if (delta >= -9 && delta <= 9 && delta != 0)
        {
            // Get current remembered brightness (not actual brightness)
            int currentRemembered = rememberedDIABrightness_;

            // Calculate new remembered brightness (each encoder step changes brightness by 3)
            int newRemembered = currentRemembered + (delta * 3);

            // Clamp to valid range 0-255
            if (newRemembered < 0)
                newRemembered = 0;
            else if (newRemembered > 255)
                newRemembered = 255;

            // Always update remembered brightness
            rememberedDIABrightness_ = newRemembered;

            std::ostringstream msg;
            msg << "E3 encoder: changing DIA remembered brightness from " << currentRemembered << " to " << newRemembered;
            LogMessage(msg.str().c_str(), true);

            // Only send DIL command if logical shutter is open (actual brightness > 0)
            long currentActual = model_.GetPosition(DeviceType_DIABrightness);
            if (currentActual > 0)
            {
                // Shutter is open: update actual lamp brightness
                std::string cmd = BuildCommand(CMD_DIA_ILLUMINATION, newRemembered);
                int ret = SendCommand(cmd);
                if (ret == DEVICE_OK)
                {
                    // Update model with new brightness
                    model_.SetPosition(DeviceType_DIABrightness, newRemembered);
                }
                else
                {
                    LogMessage("Failed to send DIA brightness command from encoder", false);
                }
            }
            // If shutter is closed, don't send DIL, don't update model

            // Always update Brightness property callback with remembered value
            auto it = usedDevices_.find(DeviceType_DIAShutter);
            if (it != usedDevices_.end() && it->second != nullptr)
            {
                GetCoreCallback()->OnPropertyChanged(it->second, "Brightness",
                    CDeviceUtils::ConvertToString(newRemembered));
            }
        }
    }
    else if (tag == CMD_DIA_ILLUMINATION_NOTIFY && params.size() > 0)
    {
        // Handle DIA illumination (brightness) change notification
        int brightness = ParseIntParameter(params[0]);
        if (brightness >= 0 && brightness <= 255)
        {
            model_.SetPosition(DeviceType_DIABrightness, brightness);

            // Only update I3 indicator and Brightness property if shutter is open (brightness > 0)
            // When closed (brightness = 0), user wants to continue seeing remembered brightness
            if (brightness > 0)
            {
                // Shutter is open: update remembered brightness and property
                rememberedDIABrightness_ = brightness;

                // Notify core callback of Brightness property change
                auto it = usedDevices_.find(DeviceType_DIAShutter);
                if (it != usedDevices_.end() && it->second != nullptr)
                {
                    GetCoreCallback()->OnPropertyChanged(it->second, "Brightness",
                        CDeviceUtils::ConvertToString(brightness));
                }
            }
            // If brightness = 0 (closed), don't update I3 or Brightness property
            // They should continue showing the remembered brightness value
        }
    }
    else if (tag == CMD_MCZ_SWITCH && params.size() > 0)
    {
        // Handle MCU switch press notifications
        // Parameter is a hex bitmask (0-7F) indicating which switch(es) were pressed
        // Bit 0 (0x01): Switch 1 - Light Path cycle
        // Bit 1 (0x02): Switch 2 - EPI Shutter toggle
        // Bit 2 (0x04): Switch 3 - DIA on/off toggle
        // Bits 3-6: Reserved for future use

        // Parse hex parameter
        int switchMask = 0;
        std::istringstream iss(params[0]);
        iss >> std::hex >> switchMask;

        if (switchMask < 0 || switchMask > 0x7F)
        {
            LogMessage(("Invalid S2 switch mask: " + params[0]).c_str(), false);
            return;
        }

        std::ostringstream msg;
        msg << "MCU switch pressed: 0x" << std::hex << std::uppercase << switchMask;
        LogMessage(msg.str().c_str(), true);

        // Switch 1 (Bit 0): Light Path cycling
        if (switchMask & 0x01)
        {
            if (model_.IsDevicePresent(DeviceType_LightPath))
            {
                long currentPos = model_.GetPosition(DeviceType_LightPath);
                // Cycle: 3 (Eyepiece) → 2 (50/50) → 1 (Left Port) → 4 (Right Port) → 3 (Eyepiece)
                int newPos;
                if (currentPos == 3)      // Eyepiece → 50/50
                    newPos = 2;
                else if (currentPos == 2) // 50/50 → Left Port
                    newPos = 1;
                else if (currentPos == 1) // Left Port → Right Port
                    newPos = 4;
                else                      // Right Port or unknown → Eyepiece
                    newPos = 3;

                std::string cmd = BuildCommand(CMD_LIGHT_PATH, newPos);
                auto future = ExecuteCommandAsync(cmd);
                model_.SetPosition(DeviceType_LightPath, newPos);
                UpdateLightPathIndicator(newPos, true);

                auto it = usedDevices_.find(DeviceType_LightPath);
                if (it != usedDevices_.end() && it->second != nullptr)
                {
                    int stateValue = newPos - 1;
                    GetCoreCallback()->OnPropertyChanged(it->second, MM::g_Keyword_State,
                        CDeviceUtils::ConvertToString(stateValue));

                    // This should works since the LightPath is a state device
                    // it would be safer to test the type first
                    char label[MM::MaxStrLength];
                    int ret = ((MM::State*) it->second)->GetPositionLabel(stateValue, label);
                    if (ret == DEVICE_OK)
                       GetCoreCallback()->OnPropertyChanged(it->second, MM::g_Keyword_Label, label);
                }
                LogMessage(("Switch 1: Light path changed to position " + std::to_string(newPos)).c_str(), true);
            }
        }

        // Switch 2 (Bit 1): EPI Shutter toggle
        if (switchMask & 0x02)
        {
            auto it = usedDevices_.find(DeviceType_EPIShutter1);
            if (it != usedDevices_.end() && it->second != nullptr)
            {
                // Get current state from model
                long currentState = model_.GetPosition(DeviceType_EPIShutter1);
                int newState = (currentState == 0) ? 1 : 0;

                // Send toggle command using SendCommand (not ExecuteCommand) to avoid deadlock
                std::string cmd = BuildCommand(CMD_EPI_SHUTTER1, newState);
                std::string response;
                auto future = ExecuteCommandAsync(cmd);
                model_.SetPosition(DeviceType_EPIShutter1, newState);
                UpdateEPIShutter1Indicator(newState, true);

                // we can not easily check if we succeeded since the above function are asynchronous
                // assume all is good...
                GetCoreCallback()->OnPropertyChanged(it->second, MM::g_Keyword_State,
                    CDeviceUtils::ConvertToString(static_cast<long>(newState)));
                GetCoreCallback()->OnShutterOpenChanged(it->second, newState == 1);
            }
        }

        // Switch 3 (Bit 2): DIA on/off toggle (with brightness memory)
        if (switchMask & 0x04)
        {
            auto it = usedDevices_.find(DeviceType_DIAShutter);
            if (it != usedDevices_.end() && it->second != nullptr)
            {
                // Get current brightness from model to determine if logical shutter is open
                long currentBrightness = model_.GetPosition(DeviceType_DIABrightness);
                int newBrightness;

                if (currentBrightness > 0)
                {
                    // Currently on: remember brightness and turn off
                    rememberedDIABrightness_ = static_cast<int>(currentBrightness);
                    newBrightness = 0;
                }
                else
                {
                    // Currently off: restore remembered brightness
                    newBrightness = rememberedDIABrightness_;
                }

                // Send command using SendCommand (not ExecuteCommand) to avoid deadlock
                std::string cmd = BuildCommand(CMD_DIA_ILLUMINATION, newBrightness);
                int ret = SendCommand(cmd);
                if (ret == DEVICE_OK)
                {
                    model_.SetPosition(DeviceType_DIABrightness, newBrightness);

                    // Only update State property (logical shutter), NOT Brightness property
                    // This keeps the Brightness property at its remembered value
                    GetCoreCallback()->OnPropertyChanged(it->second, MM::g_Keyword_State,
                        CDeviceUtils::ConvertToString(newBrightness > 0 ? 1L : 0L));
                    GetCoreCallback()->OnShutterOpenChanged(it->second, newBrightness > 0);

                    LogMessage(("Switch 3: DIA toggled to " + std::string(newBrightness > 0 ? "on" : "off")).c_str(), true);
                }
                else
                {
                    LogMessage("Failed to toggle DIA from switch", false);
                }
            }
        }
    }
    else if (tag == CMD_NOSEPIECE_REQUEST_NOTIFY && params.size() > 0)
    {
        // Handle objective dial request from MCZ
        // When user turns the objective dial, we need to:
        // 1. Disable S2 and JG
        // 2. Execute the objective switch
        // 3. Re-enable S2 and JG
        int requestedPos = ParseIntParameter(params[0]);
        if (requestedPos >= 1 && requestedPos <= NOSEPIECE_MAX_POS)
        {
            std::ostringstream msg;
            msg << "Objective dial request: position " << requestedPos;
            LogMessage(msg.str().c_str(), true);

            // Note: Responses come through notification callback when called from here,
            // so we use SendCommand with small delays instead of ExecuteCommand

            // Disable MCZ control
            std::string cmd = BuildCommand(CMD_MCZ_SWITCH, 0);
            SendCommand(cmd);
            Sleep(30);

            // Disable jog control
            cmd = BuildCommand(CMD_JOG, 0);
            SendCommand(cmd);
            Sleep(30);

            // Execute objective switch using OBSEQ command (required when responding to NROB)
            // Note: Don't update model/indicator here - let NOB notification handle that
            cmd = BuildCommand(CMD_NOSEPIECE_SEQ, requestedPos);
            SendCommand(cmd);
            LogMessage(("Objective dial: sent OBSEQ " + std::to_string(requestedPos)).c_str(), true);
            Sleep(30);

            // Re-enable MCZ control
            cmd = BuildCommand(CMD_MCZ_SWITCH, 1);
            SendCommand(cmd);
            Sleep(30);

            // Re-enable jog control
            cmd = BuildCommand(CMD_JOG, 1);
            SendCommand(cmd);
        }
    }
    else if (tag == CMD_MIRROR_REQUEST_NOTIFY && params.size() > 0)
    {
        // Handle mirror dial request from MCZ
        // When user turns the mirror dial, we need to:
        // 1. Disable S2
        // 2. Execute the mirror switch
        // 3. Re-enable S2
        int requestedPos = ParseIntParameter(params[0]);
        if (requestedPos >= 1 && requestedPos <= MIRROR_UNIT_MAX_POS)
        {
            std::ostringstream msg;
            msg << "Mirror dial request: position " << requestedPos;
            LogMessage(msg.str().c_str(), true);

            std::string response;

            // Disable MCZ control
            std::string cmd = BuildCommand(CMD_MCZ_SWITCH, 0);
            ExecuteCommandAsync(cmd);

            // Execute mirror switch
            cmd = BuildCommand(CMD_MIRROR_UNIT1, requestedPos);
            auto future = ExecuteCommandAsync(cmd);
            // Update model and indicator
            model_.SetPosition(DeviceType_MirrorUnit1, requestedPos);
            UpdateMirrorUnitIndicator(requestedPos, true);

            // Notify core callback
            auto it = usedDevices_.find(DeviceType_MirrorUnit1);
            if (it != usedDevices_.end() && it->second != nullptr)
            {
               int stateValue = requestedPos - 1;
               GetCoreCallback()->OnPropertyChanged(it->second, MM::g_Keyword_State,
                   CDeviceUtils::ConvertToString(stateValue));

               char label[MM::MaxStrLength];
               int ret = ((MM::State*) it->second)->GetPositionLabel(stateValue, label);
               if (ret == DEVICE_OK)
                   GetCoreCallback()->OnPropertyChanged(it->second, MM::g_Keyword_Label, label);
            }

            LogMessage(("Mirror dial: Async switched to position " + std::to_string(requestedPos)).c_str(), true);

            // Re-enable MCZ control
            cmd = BuildCommand(CMD_MCZ_SWITCH, 1);
            ExecuteCommandAsync(cmd);
        }
    }
    else if (tag == CMD_EPI_SHUTTER1 && params.size() > 0 && params[0] == "+")
    {
       // this was not a notification, but a command completion:
       LogMessage("Command completion for EPI Shutter 1 received", true);
       isCommandCompletion = true;
    }
    else if (tag == CMD_EPI_SHUTTER2 && params.size() > 0 && params[0] == "+")
    {
       // this was not a notification, but a command completion:
       LogMessage("Command completion for EPI Shutter 2 received", true);
       isCommandCompletion = true;
    }
    else if (tag == CMD_MIRROR_UNIT1 && params.size() > 0 && params[0] == "+")
    {
       // this was not a notification, but a command completion:
       LogMessage("Command completion for EPI Shutter 2 received", true);
       isCommandCompletion = true;
    }
    else if (tag == CMD_MIRROR_UNIT2 && params.size() > 0 && params[0] == "+")
    {
       // this was not a notification, but a command completion:
       LogMessage("Command completion for EPI Shutter 2 received", true);
       isCommandCompletion = true;
    }
    else {
       // Add more notification handlers as needed
       LogMessage(("Unhandled notification: " + message).c_str(), true);
    }

    if (isCommandCompletion)
    {
       // Signal waiting thread with the response
       {
           std::lock_guard<std::mutex> lock(responseMutex_);
           pendingResponse_ = message;
           responseReady_ = true;
       }
       responseCV_.notify_one();
    }
}

///////////////////////////////////////////////////////////////////////////////
// SDK DLL Management
///////////////////////////////////////////////////////////////////////////////

int EvidentHubWin::LoadSDK()
{
    LogMessage("Loading Evident SDK DLL...", false);

    // Build DLL path - check if user configured a custom path
    std::string dllFullPath;
    if (!dllPath_.empty())
    {
        dllFullPath = dllPath_;
        LogMessage(("Using custom DLL path: " + dllFullPath).c_str(), false);
    }
    else
    {
        // Default: try to find DLL in standard locations
        // First try relative to the adapter DLL location
        dllFullPath = "msl_pm_ix85.dll";
        LogMessage("Using default DLL path: msl_pm_ix85.dll", false);
    }

    // Load the DLL
    dllHandle_ = LoadLibraryExA(dllFullPath.c_str(), NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (dllHandle_ == NULL)
    {
        DWORD err = GetLastError();
        std::ostringstream msg;
        msg << "Failed to load SDK DLL: " << dllFullPath << " (error " << err << ")";
        LogMessage(msg.str().c_str(), false);
        return EvidentSDK::SDK_ERR_DLL_NOT_FOUND;
    }

    LogMessage("SDK DLL loaded successfully", false);

    // Get function pointers
    pfnInitialize_ = (EvidentSDK::fn_MSL_PM_Initialize)GetProcAddress(dllHandle_, "MSL_PM_Initialize");
    pfnEnumInterface_ = (EvidentSDK::fn_MSL_PM_EnumInterface)GetProcAddress(dllHandle_, "MSL_PM_EnumInterface");
    pfnGetInterfaceInfo_ = (EvidentSDK::fn_MSL_PM_GetInterfaceInfo)GetProcAddress(dllHandle_, "MSL_PM_GetInterfaceInfo");
    pfnGetPortName_ = (EvidentSDK::fn_MSL_PM_GetPortName)GetProcAddress(dllHandle_, "MSL_PM_GetPortName");
    pfnOpenInterface_ = (EvidentSDK::fn_MSL_PM_OpenInterface)GetProcAddress(dllHandle_, "MSL_PM_OpenInterface");
    pfnCloseInterface_ = (EvidentSDK::fn_MSL_PM_CloseInterface)GetProcAddress(dllHandle_, "MSL_PM_CloseInterface");
    pfnSendCommand_ = (EvidentSDK::fn_MSL_PM_SendCommand)GetProcAddress(dllHandle_, "MSL_PM_SendCommand");
    pfnRegisterCallback_ = (EvidentSDK::fn_MSL_PM_RegisterCallback)GetProcAddress(dllHandle_, "MSL_PM_RegisterCallback");

    // Verify all function pointers were found
    if (!pfnInitialize_ || !pfnEnumInterface_ || !pfnGetInterfaceInfo_ ||
        !pfnGetPortName_ || !pfnOpenInterface_ || !pfnCloseInterface_ ||
        !pfnSendCommand_ || !pfnRegisterCallback_)
    {
        LogMessage("Failed to find one or more SDK functions in DLL", false);
        UnloadSDK();
        return EvidentSDK::SDK_ERR_FUNCTION_NOT_FOUND;
    }

    LogMessage("All SDK function pointers resolved", false);

    // Initialize the SDK
    int result = pfnInitialize_();
    if (result != 0)
    {
        std::ostringstream msg;
        msg << "SDK initialization failed with code: " << result;
        LogMessage(msg.str().c_str(), false);
        UnloadSDK();
        return EvidentSDK::SDK_ERR_DLL_INIT_FAILED;
    }

    LogMessage("SDK initialized successfully", false);
    return DEVICE_OK;
}

int EvidentHubWin::UnloadSDK()
{
    LogMessage("Unloading Evident SDK...", false);

    // Clear function pointers
    pfnInitialize_ = nullptr;
    pfnEnumInterface_ = nullptr;
    pfnGetInterfaceInfo_ = nullptr;
    pfnGetPortName_ = nullptr;
    pfnOpenInterface_ = nullptr;
    pfnCloseInterface_ = nullptr;
    pfnSendCommand_ = nullptr;
    pfnRegisterCallback_ = nullptr;

    // Unload DLL
    if (dllHandle_ != NULL)
    {
        FreeLibrary(dllHandle_);
        dllHandle_ = NULL;
        LogMessage("SDK DLL unloaded", false);
    }

    return DEVICE_OK;
}

int EvidentHubWin::EnumerateAndOpenInterface()
{
    LogMessage("Enumerating SDK interfaces...", false);

    // Enumerate available interfaces
    int numInterfaces = pfnEnumInterface_();
    if (numInterfaces <= 0)
    {
        LogMessage("No SDK interfaces found", false);
        return EvidentSDK::SDK_ERR_NO_INTERFACE;
    }

    std::ostringstream msg;
    msg << "Found " << numInterfaces << " interface(s)";
    LogMessage(msg.str().c_str(), false);

    // Try to find and open the interface
    void* pInterface = nullptr;
    std::string selectedPort;

    for (int i = 0; i < numInterfaces; i++)
    {
        void* pTempInterface = nullptr;
        pfnGetInterfaceInfo_(i, &pTempInterface);
        if (pTempInterface == nullptr)
        {
            continue;
        }

        // Get port name
        wchar_t portName[256] = {0};
        if (!pfnGetPortName_(pTempInterface, portName))
        {
            continue;
        }

        // Convert wide string to narrow string
        char narrowPort[256];
        size_t converted = 0;
        wcstombs_s(&converted, narrowPort, sizeof(narrowPort), portName, _TRUNCATE);
        std::string portStr(narrowPort);

        std::ostringstream ifaceMsg;
        ifaceMsg << "Interface " << i << ": " << portStr;
        LogMessage(ifaceMsg.str().c_str(), false);

        // If user specified a port, match it; otherwise use first available
        if (port_.empty() || port_ == "Undefined" || portStr.find(port_) != std::string::npos)
        {
            pInterface = pTempInterface;
            selectedPort = portStr;
            break;
        }
    }

    if (pInterface == nullptr)
    {
        LogMessage("No matching SDK interface found for port", false);
        return EvidentSDK::SDK_ERR_NO_INTERFACE;
    }

    // Open the interface
    if (!pfnOpenInterface_(pInterface))
    {
        LogMessage("Failed to open SDK interface", false);
        return EvidentSDK::SDK_ERR_OPEN_FAILED;
    }

    interfaceHandle_ = pInterface;
    port_ = selectedPort;

    std::ostringstream openMsg;
    openMsg << "Opened SDK interface for port: " << selectedPort;
    LogMessage(openMsg.str().c_str(), false);

    // Register callbacks
    if (!pfnRegisterCallback_(interfaceHandle_,
                              CommandCallbackStatic,
                              NotifyCallbackStatic,
                              ErrorCallbackStatic,
                              this))
    {
        LogMessage("Failed to register SDK callbacks", false);
        pfnCloseInterface_(interfaceHandle_);
        interfaceHandle_ = nullptr;
        return EvidentSDK::SDK_ERR_CALLBACK_FAILED;
    }

    LogMessage("SDK callbacks registered successfully", false);
    return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// SDK Callback Handlers
///////////////////////////////////////////////////////////////////////////////

int CALLBACK EvidentHubWin::CommandCallbackStatic(ULONG /* MsgId */, ULONG /* wParam */, ULONG /* lParam */,
                                                    PVOID pv, PVOID pContext, PVOID /* pCaller */)
{
    EvidentHubWin* pHub = static_cast<EvidentHubWin*>(pContext);
    if (pHub != nullptr)
    {
        EvidentSDK::MDK_MSL_CMD* pCmd = static_cast<EvidentSDK::MDK_MSL_CMD*>(pv);
        return pHub->OnCommandComplete(pCmd);
    }
    return 0;
}

int CALLBACK EvidentHubWin::NotifyCallbackStatic(ULONG /* MsgId */, ULONG /* wParam */, ULONG /* lParam */,
                                                   PVOID pv, PVOID pContext, PVOID /* pCaller */)
{
    EvidentHubWin* pHub = static_cast<EvidentHubWin*>(pContext);
    if (pHub != nullptr && pv != nullptr)
    {
        // For notifications, pv is a null-terminated string, not MDK_MSL_CMD*
        const char* notificationStr = static_cast<const char*>(pv);
        return pHub->OnNotification(notificationStr);
    }
    return 0;
}

int CALLBACK EvidentHubWin::ErrorCallbackStatic(ULONG /* MsgId */, ULONG /* wParam */, ULONG /* lParam */,
                                                  PVOID pv, PVOID pContext, PVOID /* pCaller */)
{
    EvidentHubWin* pHub = static_cast<EvidentHubWin*>(pContext);
    if (pHub != nullptr)
    {
        EvidentSDK::MDK_MSL_CMD* pCmd = static_cast<EvidentSDK::MDK_MSL_CMD*>(pv);
        return pHub->OnError(pCmd);
    }
    return 0;
}

int EvidentHubWin::OnCommandComplete(EvidentSDK::MDK_MSL_CMD* pCmd)
{
    if (pCmd == nullptr)
        return 0;

    // Extract response from the command structure
    std::string response = EvidentSDK::GetResponseString(*pCmd);
    if (response == "")
       return 0;

    // Signal waiting thread with the response
    {
        std::lock_guard<std::mutex> lock(responseMutex_);
        pendingResponse_ = response;
        responseReady_ = true;
    }
    responseCV_.notify_one();

    return 0;
}

int EvidentHubWin::OnNotification(const char* notificationStr)
{
    if (notificationStr == nullptr || notificationStr[0] == '\0')
        return 0;

    std::string notification(notificationStr);
    LogMessage(("Notification received: " + notification).c_str(), true);

    // Process notification
    ProcessNotification(notification);

    return 0;
}

int EvidentHubWin::OnError(EvidentSDK::MDK_MSL_CMD* pCmd)
{
    if (pCmd == nullptr)
        return 0;

    std::ostringstream msg;
    msg << "SDK Error callback - Result: " << pCmd->m_Result << " Status: " << pCmd->m_Status;
    LogMessage(msg.str().c_str(), false);

    // Signal waiting thread with error
    {
        std::lock_guard<std::mutex> lock(responseMutex_);
        pendingResponse_ = "";  // Empty response indicates error
        responseReady_ = true;
    }
    responseCV_.notify_one();

    return 0;
}
