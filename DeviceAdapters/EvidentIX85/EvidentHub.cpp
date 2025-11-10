///////////////////////////////////////////////////////////////////////////////
// FILE:          EvidentHub.cpp
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

#include "EvidentHub.h"
#include "ModuleInterface.h"
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <chrono>

using namespace EvidentIX85;

const char* g_HubDeviceName = "EvidentIX85-Hub";

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

// Property names
const char* g_PropPort = "Port";
const char* g_PropAnswerTimeout = "AnswerTimeout";
extern const char* g_Keyword_Magnification;

EvidentHub::EvidentHub() :
    initialized_(false),
    port_(""),
    answerTimeoutMs_(ANSWER_TIMEOUT_MS),
    stopMonitoring_(false),
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
    SetErrorText(ERR_MONITOR_THREAD_FAILED, "Failed to start monitoring thread");
    SetErrorText(ERR_PORT_NOT_SET, "Serial port not set");
    SetErrorText(ERR_PORT_CHANGE_FORBIDDEN, "Cannot change serial port after initialization");

    // Pre-initialization properties
    CPropertyAction* pAct = new CPropertyAction(this, &EvidentHub::OnPort);
    CreateProperty(g_PropPort, "Undefined", MM::String, false, pAct, true);

    pAct = new CPropertyAction(this, &EvidentHub::OnAnswerTimeout);
    CreateProperty(g_PropAnswerTimeout, "2000", MM::Integer, false, pAct, true);
}

EvidentHub::~EvidentHub()
{
    Shutdown();
}

void EvidentHub::GetName(char* pszName) const
{
    CDeviceUtils::CopyLimitedString(pszName, g_HubDeviceName);
}

bool EvidentHub::Busy()
{
    return false;  // Hub itself is never busy
}

int EvidentHub::Initialize()
{
    if (initialized_)
        return DEVICE_OK;

    usedDevices_.clear();

    // Verify port is set
    if (port_.empty() || port_ == "Undefined")
        return ERR_PORT_NOT_SET;

    // Clear port buffers
    int ret = ClearPort();
    if (ret != DEVICE_OK)
        return ret;

    // Start monitoring thread
    StartMonitoring();

    // Switch to remote mode
    ret = SetRemoteMode();
    if (ret != DEVICE_OK)
        return ret;

    // Get version and unit info
    ret = GetVersion(version_);
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
            UpdateMirrorUnitIndicator(pos == 0 ? -1 : static_cast<int>(pos));
        }
        else
        {
            // No mirror unit, display "---"
            UpdateMirrorUnitIndicator(-1);
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

        // Initialize light path indicator (I4)
        if (model_.IsDevicePresent(DeviceType_LightPath))
        {
            long pos = model_.GetPosition(DeviceType_LightPath);
            // Position will be 0 if unknown (not yet queried), display as unknown (all off)
            UpdateLightPathIndicator(pos == 0 ? -1 : static_cast<int>(pos));
        }
        else
        {
            // No light path, display all off
            UpdateLightPathIndicator(-1);
        }

        // Initialize EPI shutter 1 indicator (I5)
        if (model_.IsDevicePresent(DeviceType_EPIShutter1))
        {
            long state = model_.GetPosition(DeviceType_EPIShutter1);
            // Position will be 0 if unknown (not yet queried), display as closed (I5 1)
            UpdateEPIShutter1Indicator(state == 0 ? 0 : static_cast<int>(state));
        }
        else
        {
            // No EPI shutter 1, display as closed
            UpdateEPIShutter1Indicator(0);
        }
    }

    initialized_ = true;
    return DEVICE_OK;
}

int EvidentHub::Shutdown()
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

        // Switch back to local mode
        std::string cmd = BuildCommand(CMD_LOGIN, 0);  // 0 = Local mode
        std::string response;
        ExecuteCommand(cmd, response);

        // Now stop the monitoring thread (after all commands sent)
        StopMonitoring();

        initialized_ = false;
    }
    return DEVICE_OK;
}

int EvidentHub::OnPort(MM::PropertyBase* pProp, MM::ActionType eAct)
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

int EvidentHub::OnAnswerTimeout(MM::PropertyBase* pProp, MM::ActionType eAct)
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

int EvidentHub::ClearPort()
{
    const unsigned int bufSize = 255;
    unsigned char clear[bufSize];
    unsigned long read = bufSize;
    int ret;

    while (read == bufSize)
    {
        ret = ReadFromComPort(port_.c_str(), clear, bufSize, read);
        if (ret != DEVICE_OK)
            return ret;
    }
    return DEVICE_OK;
}

int EvidentHub::SetRemoteMode()
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

int EvidentHub::GetVersion(std::string& version)
{
    std::string cmd = BuildCommand(CMD_VERSION, 1); // 1 = Firmware version
    std::string response;
    int ret = ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
        return ret;

    if (IsValidAnswer(response, CMD_VERSION))
    {
        // Response format: "V +" - version command doesn't return version number
        // Version is embedded in the response or needs separate query
        version = response.substr(2);
        return DEVICE_OK;
    }

    return ERR_INVALID_RESPONSE;
}

int EvidentHub::GetUnit(std::string& unit)
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

int EvidentHub::ExecuteCommand(const std::string& command, std::string& response)
{
    std::lock_guard<std::mutex> lock(commandMutex_);

    int ret = SendCommand(command);
    if (ret != DEVICE_OK)
        return ret;

    ret = GetResponse(response, answerTimeoutMs_);
    return ret;
}

int EvidentHub::SendCommand(const std::string& command)
{
    LogMessage(("Sending: " + command).c_str(), true);
    int ret = SendSerialCommand(port_.c_str(), command.c_str(), TERMINATOR);
    if (ret != DEVICE_OK)
        return ret;

    return DEVICE_OK;
}

int EvidentHub::GetResponse(std::string& response, long timeoutMs)
{
    if (timeoutMs < 0)
        timeoutMs = answerTimeoutMs_;

    // Wait for the monitoring thread to provide a response
    std::unique_lock<std::mutex> lock(responseMutex_);
    responseReady_ = false;

    if (responseCV_.wait_for(lock, std::chrono::milliseconds(timeoutMs),
        [this] { return responseReady_; }))
    {
        response = pendingResponse_;
        LogMessage(("Received: " + response).c_str(), true);
        return DEVICE_OK;
    }

    return ERR_COMMAND_TIMEOUT;
}

int EvidentHub::DoDeviceDetection()
{
    availableDevices_.clear();
    detectedDevicesByName_.clear();

    // Use V command to detect device presence
    // This avoids firmware bugs with individual device queries
    std::string version;

    // V2 - Nosepiece
    if (QueryDevicePresenceByVersion(V_NOSEPIECE, version) == DEVICE_OK)
    {
        LogMessage(("Detected Nosepiece (V2): " + version).c_str());
        model_.SetDevicePresent(DeviceType_Nosepiece, true);
        model_.SetDeviceVersion(DeviceType_Nosepiece, version);
        availableDevices_.push_back(DeviceType_Nosepiece);
        detectedDevicesByName_.push_back(g_NosepieceDeviceName);
        // Query actual position/numPositions
        QueryNosepiece();
    }

    // V5 - Focus
    if (QueryDevicePresenceByVersion(V_FOCUS, version) == DEVICE_OK)
    {
        LogMessage(("Detected Focus (V5): " + version).c_str());
        model_.SetDevicePresent(DeviceType_Focus, true);
        model_.SetDeviceVersion(DeviceType_Focus, version);
        availableDevices_.push_back(DeviceType_Focus);
        detectedDevicesByName_.push_back(g_FocusDeviceName);
        // Query actual position/limits
        QueryFocus();
    }

    // V6 - Light Path
    if (QueryDevicePresenceByVersion(V_LIGHTPATH, version) == DEVICE_OK)
    {
        LogMessage(("Detected LightPath (V6): " + version).c_str());
        model_.SetDevicePresent(DeviceType_LightPath, true);
        model_.SetDeviceVersion(DeviceType_LightPath, version);
        availableDevices_.push_back(DeviceType_LightPath);
        detectedDevicesByName_.push_back(g_LightPathDeviceName);
        // Query actual position
        QueryLightPath();
    }

    // V7 - Condenser Unit (IX3-LWUCDA): Contains Polarizer, CondenserTurret, DIAShutter, DIAAperture
    if (QueryDevicePresenceByVersion(V_CONDENSER_UNIT, version) == DEVICE_OK)
    {
        LogMessage(("Detected Condenser Unit (V7): " + version).c_str());

        // Polarizer
        model_.SetDevicePresent(DeviceType_Polarizer, true);
        model_.SetDeviceVersion(DeviceType_Polarizer, version);
        availableDevices_.push_back(DeviceType_Polarizer);
        detectedDevicesByName_.push_back(g_PolarizerDeviceName);
        QueryPolarizer();

        // Condenser Turret
        model_.SetDevicePresent(DeviceType_CondenserTurret, true);
        model_.SetDeviceVersion(DeviceType_CondenserTurret, version);
        availableDevices_.push_back(DeviceType_CondenserTurret);
        detectedDevicesByName_.push_back(g_CondenserTurretDeviceName);
        QueryCondenserTurret();

        // DIA Shutter
        model_.SetDevicePresent(DeviceType_DIAShutter, true);
        model_.SetDeviceVersion(DeviceType_DIAShutter, version);
        availableDevices_.push_back(DeviceType_DIAShutter);
        detectedDevicesByName_.push_back(g_DIAShutterDeviceName);
        QueryDIAShutter();
    }

    // V8 - DIC Unit (IX5-DICTA): Contains DICPrism, DICRetardation
    if (QueryDevicePresenceByVersion(V_DIC_UNIT, version) == DEVICE_OK)
    {
        LogMessage(("Detected DIC Unit (V8): " + version).c_str());

        // DIC Prism
        model_.SetDevicePresent(DeviceType_DICPrism, true);
        model_.SetDeviceVersion(DeviceType_DICPrism, version);
        availableDevices_.push_back(DeviceType_DICPrism);
        detectedDevicesByName_.push_back(g_DICPrismDeviceName);
        QueryDICPrism();
    }

    // V9 - Mirror Unit 1
    if (QueryDevicePresenceByVersion(V_MIRROR_UNIT1, version) == DEVICE_OK)
    {
        LogMessage(("Detected MirrorUnit1 (V9): " + version).c_str());
        model_.SetDevicePresent(DeviceType_MirrorUnit1, true);
        model_.SetDeviceVersion(DeviceType_MirrorUnit1, version);
        availableDevices_.push_back(DeviceType_MirrorUnit1);
        detectedDevicesByName_.push_back(g_MirrorUnit1DeviceName);
        QueryMirrorUnit1();
    }

    // V10 - EPI Shutter 1
    if (QueryDevicePresenceByVersion(V_EPI_SHUTTER1, version) == DEVICE_OK)
    {
        LogMessage(("Detected EPIShutter1 (V10): " + version).c_str());
        model_.SetDevicePresent(DeviceType_EPIShutter1, true);
        model_.SetDeviceVersion(DeviceType_EPIShutter1, version);
        availableDevices_.push_back(DeviceType_EPIShutter1);
        detectedDevicesByName_.push_back(g_EPIShutter1DeviceName);
        QueryEPIShutter1();
    }

    // V11 - Mirror Unit 2
    if (QueryDevicePresenceByVersion(V_MIRROR_UNIT2, version) == DEVICE_OK)
    {
        LogMessage(("Detected MirrorUnit2 (V11): " + version).c_str());
        model_.SetDevicePresent(DeviceType_MirrorUnit2, true);
        model_.SetDeviceVersion(DeviceType_MirrorUnit2, version);
        availableDevices_.push_back(DeviceType_MirrorUnit2);
        detectedDevicesByName_.push_back(g_MirrorUnit2DeviceName);
        QueryMirrorUnit2();
    }

    // V12 - EPI Shutter 2
    if (QueryDevicePresenceByVersion(V_EPI_SHUTTER2, version) == DEVICE_OK)
    {
        LogMessage(("Detected EPIShutter2 (V12): " + version).c_str());
        model_.SetDevicePresent(DeviceType_EPIShutter2, true);
        model_.SetDeviceVersion(DeviceType_EPIShutter2, version);
        availableDevices_.push_back(DeviceType_EPIShutter2);
        detectedDevicesByName_.push_back(g_EPIShutter2DeviceName);
        QueryEPIShutter2();
    }

    // V14 - EPI ND Filter
    if (QueryDevicePresenceByVersion(V_EPIND, version) == DEVICE_OK)
    {
        LogMessage(("Detected EPIND (V14): " + version).c_str());
        model_.SetDevicePresent(DeviceType_EPIND, true);
        model_.SetDeviceVersion(DeviceType_EPIND, version);
        availableDevices_.push_back(DeviceType_EPIND);
        detectedDevicesByName_.push_back(g_EPINDDeviceName);
        QueryEPIND();
    }

    // V13 - Manual Control Unit (MCU)
    if (QueryDevicePresenceByVersion(V_MANUAL_CONTROL, version) == DEVICE_OK)
    {
        LogMessage(("Detected Manual Control Unit (V13): " + version).c_str());
        model_.SetDevicePresent(DeviceType_ManualControl, true);
        model_.SetDeviceVersion(DeviceType_ManualControl, version);
        // Note: MCU is not added to availableDevices/detectedDevicesByName
        // as it's not a standalone MM device but provides indicator feedback
    }

    // Keep legacy query methods for devices without clear V mapping

    // Magnification (CA command) - V mapping unclear, keep existing query
    if (QueryMagnification() == DEVICE_OK)
    {
        LogMessage("Detected Magnification (CA)");
        availableDevices_.push_back(DeviceType_Magnification);
        detectedDevicesByName_.push_back(g_MagnificationDeviceName);
        model_.SetDevicePresent(DeviceType_Magnification, true);
    }

    // Correction Collar - V3 or V4, keep existing query
    if (QueryCorrectionCollar() == DEVICE_OK)
    {
        LogMessage("Detected CorrectionCollar (CC)");
        availableDevices_.push_back(DeviceType_CorrectionCollar);
        detectedDevicesByName_.push_back(g_CorrectionCollarDeviceName);
        model_.SetDevicePresent(DeviceType_CorrectionCollar, true);
    }

    std::ostringstream msg;
    msg << "Discovered " << availableDevices_.size() << " devices";
    LogMessage(msg.str().c_str(), false);

    return DEVICE_OK;
}

int EvidentHub::DetectInstalledDevices()
{
    for (size_t i=0; i < detectedDevicesByName_.size(); i++)
    {
       MM::Device* pDev = ::CreateDevice(detectedDevicesByName_[i].c_str());
       if (pDev)
          AddInstalledDevice(pDev);
    }

    return DEVICE_OK;
}

int EvidentHub::QueryDevicePresenceByVersion(int unitNumber, std::string& version)
{
    std::string cmd = BuildCommand(CMD_VERSION, unitNumber);
    std::string response;
    int ret = ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
        return ret;

    // Check for error response indicating device not present
    if (IsNegativeAck(response, CMD_VERSION))
        return ERR_DEVICE_NOT_AVAILABLE;

    // Parse version string from response
    std::vector<std::string> params = ParseParameters(response);
    if (params.size() > 0 && params[0] != "X")
    {
        version = params[0];
        return DEVICE_OK;
    }

    return ERR_DEVICE_NOT_AVAILABLE;
}

bool EvidentHub::IsDevicePresent(EvidentIX85::DeviceType type) const
{
    return model_.IsDevicePresent(type);
}

std::string EvidentHub::GetDeviceVersion(EvidentIX85::DeviceType type) const
{
    return model_.GetDeviceVersion(type);
}

int EvidentHub::EnableNotification(const char* cmd, bool enable)
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
int EvidentHub::QueryFocus()
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

int EvidentHub::QueryNosepiece()
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

int EvidentHub::QueryMagnification()
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

int EvidentHub::QueryLightPath()
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

int EvidentHub::QueryCondenserTurret()
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

int EvidentHub::QueryDIAAperture()
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

int EvidentHub::QueryDIAShutter()
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

int EvidentHub::QueryPolarizer()
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

int EvidentHub::QueryDICPrism()
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

int EvidentHub::QueryDICRetardation()
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

int EvidentHub::QueryEPIShutter1()
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

int EvidentHub::QueryEPIShutter2()
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

int EvidentHub::QueryMirrorUnit1()
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

int EvidentHub::QueryMirrorUnit2()
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

int EvidentHub::QueryEPIND()
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

int EvidentHub::QueryRightPort()
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

int EvidentHub::QueryCorrectionCollar()
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

int EvidentHub::UpdateNosepieceIndicator(int position)
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

int EvidentHub::UpdateMirrorUnitIndicator(int position)
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

    // Send command without waiting for response (to avoid deadlock when called from monitoring thread)
    // The "I2 +" response will be consumed by the monitoring thread as a pseudo-notification
    int ret = SendCommand(cmd);
    if (ret != DEVICE_OK)
    {
        LogMessage(("Failed to send mirror unit indicator command: " + cmd).c_str());
        return ret;
    }

    LogMessage(("Sent mirror unit indicator command: " + cmd).c_str(), true);
    return DEVICE_OK;
}

int EvidentHub::UpdateLightPathIndicator(int position)
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

    // Send command without waiting for response (to avoid deadlock when called from monitoring thread)
    // The "I4 +" response will be consumed by the monitoring thread as a pseudo-notification
    int ret = SendCommand(cmd);
    if (ret != DEVICE_OK)
    {
        LogMessage(("Failed to send light path indicator command: " + cmd).c_str());
        return ret;
    }

    LogMessage(("Sent light path indicator command: " + cmd).c_str(), true);
    return DEVICE_OK;
}

int EvidentHub::UpdateEPIShutter1Indicator(int state)
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

    // Send command without waiting for response (to avoid deadlock when called from monitoring thread)
    // The "I5 +" response will be consumed by the monitoring thread as a pseudo-notification
    int ret = SendCommand(cmd);
    if (ret != DEVICE_OK)
    {
        LogMessage(("Failed to send EPI shutter indicator command: " + cmd).c_str());
        return ret;
    }

    LogMessage(("Sent EPI shutter indicator command: " + cmd).c_str(), true);
    return DEVICE_OK;
}

int EvidentHub::UpdateDIABrightnessIndicator(int brightness)
{
    // Check if MCU is present
    if (!model_.IsDevicePresent(DeviceType_ManualControl))
        return DEVICE_OK;  // Not an error, MCU just not present

    // Map brightness (0-255) to I3 indicator LED pattern (hex values)
    // I3 accepts hex bitmask values: 1, 3, 7, F, 1F
    // 0 brightness -> no LEDs (I3 0)
    // 1-51 -> 1 LED (I3 1)
    // 52-102 -> 2 LEDs (I3 3)
    // 103-153 -> 3 LEDs (I3 7)
    // 154-204 -> 4 LEDs (I3 F)
    // 205-255 -> 5 LEDs (I3 1F)

    std::string i3Value;
    if (brightness == 0)
        i3Value = "0";
    else if (brightness <= 51)
        i3Value = "1";
    else if (brightness <= 102)
        i3Value = "3";
    else if (brightness <= 153)
        i3Value = "7";
    else if (brightness <= 204)
        i3Value = "F";
    else
        i3Value = "1F";

    std::string cmd = "I3 " + i3Value;

    // Send command without waiting for response
    int ret = SendCommand(cmd);
    if (ret != DEVICE_OK)
    {
        LogMessage(("Failed to send DIA brightness indicator command: " + cmd).c_str());
        return ret;
    }

    LogMessage(("Sent DIA brightness indicator command: " + cmd).c_str(), true);
    return DEVICE_OK;
}

// Monitoring thread
void EvidentHub::StartMonitoring()
{
    if (monitorThread_.joinable())
        return;  // Already running

    stopMonitoring_ = false;
    monitorThread_ = std::thread(&EvidentHub::MonitorThreadFunc, this);

    LogMessage("Monitoring thread started", true);
}

void EvidentHub::StopMonitoring()
{
    if (!monitorThread_.joinable())
        return;  // Not running

    stopMonitoring_ = true;

    // Wake up any waiting command threads
    {
        std::lock_guard<std::mutex> lock(responseMutex_);
        responseReady_ = false;
    }
    responseCV_.notify_all();

    monitorThread_.join();

    LogMessage("Monitoring thread stopped", true);
}

void EvidentHub::MonitorThreadFunc()
{
    // This thread is the SOLE reader from the serial port
    // It routes messages to either:
    // - Command thread (via condition variable) for responses
    // - Processes directly for notifications

    LogMessage("Monitor thread function started", true);

    std::string buffer;

    while (!stopMonitoring_.load())
    {
        // Read one byte at a time from serial port
        unsigned char byte;
        unsigned long read;
        int ret = ReadFromComPort(port_.c_str(), &byte, 1, read);

        if (ret != DEVICE_OK || read == 0)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // Build message byte by byte
        if (byte == '\n')
        {
            // End of line - remove trailing \r if present
            if (!buffer.empty() && buffer.back() == '\r')
                buffer.pop_back();

            if (!buffer.empty())
            {
                // Determine if this is a notification or command response
                if (IsNotificationTag(buffer))
                {
                    // This is a notification - process it
                    LogMessage(("Notification: " + buffer).c_str(), true);
                    ProcessNotification(buffer);
                }
                else
                {
                    // This is a command response - pass to waiting command thread
                    LogMessage(("Response (from monitor): " + buffer).c_str(), true);
                    {
                        std::lock_guard<std::mutex> lock(responseMutex_);
                        pendingResponse_ = buffer;
                        responseReady_ = true;
                    }
                    responseCV_.notify_one();
                }

                buffer.clear();
            }
        }
        else
        {
            buffer += static_cast<char>(byte);
        }
    }

    LogMessage("Monitor thread function exiting", true);
}

bool EvidentHub::IsNotificationTag(const std::string& message) const
{
    // Extract tag from the message
    std::string tag = ExtractTag(message);

    // Special case: I1, I2, I3, I4, and I5 (indicator) responses must always be consumed by monitoring thread
    // This prevents them from being sent to command threads and causing sync issues
    if (tag == CMD_INDICATOR1 || tag == CMD_INDICATOR2 || tag == CMD_INDICATOR3 ||
        tag == CMD_INDICATOR4 || tag == CMD_INDICATOR5)
        return true;

    // Special case: E1 (encoder) messages with delta values (-1, 1) are notifications
    // but "E1 0" and acknowledgments "E1 +" are command responses
    if (tag == CMD_ENCODER1)
    {
        std::vector<std::string> params = ParseParameters(message);
        if (params.size() > 0)
        {
            // Reject acknowledgments first (before ParseIntParameter which returns -1 for "+")
            if (params[0] == "+" || params[0] == "!")
                return false;

            int value = ParseIntParameter(params[0]);
            // -1 and 1 are encoder turn notifications
            if (value == -1 || value == 1)
                return true;
        }
        // "E1 0" or other values are command responses
        return false;
    }

    // Special case: E2 (encoder) messages with delta values (-1, 1) are notifications
    // but "E2 0" and acknowledgments "E2 +" are command responses
    if (tag == CMD_ENCODER2)
    {
        std::vector<std::string> params = ParseParameters(message);
        if (params.size() > 0)
        {
            // Reject acknowledgments first (before ParseIntParameter which returns -1 for "+")
            if (params[0] == "+" || params[0] == "!")
                return false;

            int value = ParseIntParameter(params[0]);
            // -1 and 1 are encoder turn notifications
            if (value == -1 || value == 1)
                return true;
        }
        // "E2 0" or other values are command responses
        return false;
    }

    // Special case: E3 (encoder) messages with delta values (-9 to 9) are notifications
    // but "E3 0" and acknowledgments "E3 +" are command responses
    if (tag == CMD_ENCODER3)
    {
        std::vector<std::string> params = ParseParameters(message);
        if (params.size() > 0)
        {
            // Reject acknowledgments first (before ParseIntParameter which returns -1 for "+")
            if (params[0] == "+" || params[0] == "!")
                return false;

            int value = ParseIntParameter(params[0]);
            // Fast turning can produce delta values from -9 to 9 (excluding 0)
            if (value >= -9 && value <= 9 && value != 0)
                return true;
        }
        // "E3 0" or other values are command responses
        return false;
    }

    // Special case: S2 (MCU switches) messages with hex bitmask are notifications
    // but acknowledgments "S2 +" are command responses
    if (tag == CMD_MCZ_SWITCH)
    {
        std::vector<std::string> params = ParseParameters(message);
        if (params.size() > 0)
        {
            // Only reject acknowledgments
            if (params[0] == "+" || params[0] == "!")
                return false;

            // All hex bitmask values (0-7F) are notifications
            return true;
        }
        return false;
    }

    // Check if it's a known notification tag
    bool isNotifyTag = (tag == CMD_FOCUS_NOTIFY ||
                        tag == CMD_NOSEPIECE_NOTIFY ||
                        tag == CMD_MAGNIFICATION_NOTIFY ||
                        tag == CMD_CONDENSER_TURRET_NOTIFY ||
                        tag == CMD_DIA_APERTURE_NOTIFY ||
                        tag == CMD_DIA_ILLUMINATION_NOTIFY ||
                        tag == CMD_POLARIZER_NOTIFY ||
                        tag == CMD_DIC_RETARDATION_NOTIFY ||
                        tag == CMD_DIC_LOCALIZED_NOTIFY ||
                        tag == CMD_MIRROR_UNIT_NOTIFY1 ||
                        tag == CMD_MIRROR_UNIT_NOTIFY2 ||
                        tag == CMD_RIGHT_PORT_NOTIFY ||
                        tag == CMD_OFFSET_LENS_NOTIFY);

    if (!isNotifyTag)
        return false;

    // It's a notify tag, but is it an acknowledgment or actual notification?
    // Acknowledgments: "NCA +", "NFP !"
    // Notifications: "NCA 1", "NFP 3110"
    // Only treat as notification if it's NOT an acknowledgment
    return !IsPositiveAck(message, tag.c_str()) && !IsNegativeAck(message, tag.c_str());
}

void EvidentHub::ProcessNotification(const std::string& message)
{
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

            // Update MCU indicator I1 with new nosepiece position
            UpdateNosepieceIndicator(pos);

            // Notify core callback of State property change
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

            // Check if we've reached the target position
            long targetPos = model_.GetTargetPosition(DeviceType_Nosepiece);
            if (targetPos >= 0 && pos == targetPos)
            {
                model_.SetBusy(DeviceType_Nosepiece, false);
            }
        }
    }
    else if (tag == CMD_MAGNIFICATION_NOTIFY && params.size() > 0)
    {
        int pos = ParseIntParameter(params[0]);
        if (pos >= 0)
        {
            model_.SetPosition(DeviceType_Magnification, pos);
            auto it = usedDevices_.find(DeviceType_Magnification);
            if (it != usedDevices_.end() && it->second != nullptr)
            {
                // Map position (1-based) to magnification value
                const double magnifications[3] = {1.0, 1.6, 2.0};
                if (pos >= 1 && pos <= 3)
                {
                    GetCoreCallback()->OnPropertyChanged(it->second, g_Keyword_Magnification,
                        CDeviceUtils::ConvertToString(magnifications[pos - 1]));
                    // Notify core that magnification has changed
                    GetCoreCallback()->OnMagnifierChanged(it->second);
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
            int ret = SendCommand(cmd);
            if (ret != DEVICE_OK)
            {
                LogMessage("Failed to send nosepiece command from encoder", false);
            }
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

            // Use SendCommand (not ExecuteCommand) to avoid deadlock when called from monitoring thread
            // The "MU1 +" response will be consumed by monitoring thread as a command response
            int ret = SendCommand(cmd);
            if (ret == DEVICE_OK)
            {
                // Optimistically update model and indicator (mirror unit has no position change notifications)
                model_.SetPosition(DeviceType_MirrorUnit1, newPos);
                UpdateMirrorUnitIndicator(newPos);

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
                    ret = ((MM::State*) it->second)->GetPositionLabel(stateValue, label);
                    if (ret == DEVICE_OK)
                       GetCoreCallback()->OnPropertyChanged(it->second, MM::g_Keyword_Label, label);
                }
            }
            else
            {
                LogMessage("Failed to send mirror unit command from encoder", false);
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

            // Always update I3 indicator to match remembered brightness
            UpdateDIABrightnessIndicator(newRemembered);

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
                // Shutter is open: update remembered brightness, I3 indicator, and property
                rememberedDIABrightness_ = brightness;
                UpdateDIABrightnessIndicator(brightness);

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
                int ret = SendCommand(cmd);
                if (ret == DEVICE_OK)
                {
                    model_.SetPosition(DeviceType_LightPath, newPos);
                    UpdateLightPathIndicator(newPos);

                    auto it = usedDevices_.find(DeviceType_LightPath);
                    if (it != usedDevices_.end() && it->second != nullptr)
                    {
                        int stateValue = newPos - 1;
                        GetCoreCallback()->OnPropertyChanged(it->second, MM::g_Keyword_State,
                            CDeviceUtils::ConvertToString(stateValue));

                        // This should works since the LightPath is a state device
                        // it would be safer to test the type first
                        char label[MM::MaxStrLength];
                        ret = ((MM::State*) it->second)->GetPositionLabel(stateValue, label);
                        if (ret == DEVICE_OK)
                           GetCoreCallback()->OnPropertyChanged(it->second, MM::g_Keyword_Label, label);
                    }
                    LogMessage(("Switch 1: Light path changed to position " + std::to_string(newPos)).c_str(), true);
                }
                else
                {
                    LogMessage("Failed to change light path from switch", false);
                }
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
                int ret = SendCommand(cmd);
                if (ret == DEVICE_OK)
                {
                    model_.SetPosition(DeviceType_EPIShutter1, newState);
                    UpdateEPIShutter1Indicator(newState);

                    GetCoreCallback()->OnPropertyChanged(it->second, MM::g_Keyword_State,
                        CDeviceUtils::ConvertToString(static_cast<long>(newState)));
                    GetCoreCallback()->OnShutterOpenChanged(it->second, newState == 1);

                    LogMessage(("Switch 2: EPI shutter toggled to " + std::string(newState ? "open" : "closed")).c_str(), true);
                }
                else
                {
                    LogMessage("Failed to toggle EPI shutter from switch", false);
                }
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

                    // Always update I3 indicator with remembered brightness (not 0 when closing)
                    // User wants to see the remembered brightness value, not that lamp is off
                    UpdateDIABrightnessIndicator(rememberedDIABrightness_);

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
    // Add more notification handlers as needed
}
