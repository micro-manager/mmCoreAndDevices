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
extern const char* g_MirrorUnit1DeviceName;
extern const char* g_PolarizerDeviceName;
extern const char* g_DICPrismDeviceName;
extern const char* g_EPINDDeviceName;
extern const char* g_CorrectionCollarDeviceName;

// Property names
const char* g_PropPort = "Port";
const char* g_PropAnswerTimeout = "AnswerTimeout";

EvidentHub::EvidentHub() :
    initialized_(false),
    port_(""),
    answerTimeoutMs_(ANSWER_TIMEOUT_MS),
    stopMonitoring_(false)
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

    // Start monitoring thread
    StartMonitoring();

    initialized_ = true;
    return DEVICE_OK;
}

int EvidentHub::Shutdown()
{
    if (initialized_)
    {
        StopMonitoring();

        // Disable all active notifications
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

        std::string cmd = BuildCommand(CMD_LOGIN, 0);  // 0 = Local mode
        std::string response;
        ExecuteCommand(cmd, response);

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

    response.clear();
    char c;
    std::string line;
    MM::MMTime startTime = GetCurrentMMTime();

    while ((GetCurrentMMTime() - startTime) < MM::MMTime::fromMs(timeoutMs))
    {
        unsigned long read;
        int ret = ReadFromComPort(port_.c_str(), (unsigned char*)&c, 1, read);
        if (ret != DEVICE_OK)
            return ret;

        if (read == 1)
        {
            if (c == '\n')  // End of line
            {
                // Remove trailing \r if present
                if (!line.empty() && line.back() == '\r')
                    line.pop_back();

                if (!line.empty())
                {
                    response = line;
                    LogMessage(("Received: " + response).c_str(), true);
                    return DEVICE_OK;
                }
            }
            else
            {
                line += c;
            }
        }

        CDeviceUtils::SleepMs(1);
    }

    return ERR_COMMAND_TIMEOUT;
}

int EvidentHub::DoDeviceDetection()
{
    availableDevices_.clear();
    detectedDevicesByName_.clear();

    // Query each possible device to see if it's present
    // Start with essential devices

    if (QueryFocus() == DEVICE_OK)
    {
        availableDevices_.push_back(DeviceType_Focus);
        detectedDevicesByName_.push_back(g_FocusDeviceName);
        model_.SetDevicePresent(DeviceType_Focus, true);
    }

    if (QueryNosepiece() == DEVICE_OK)
    {
        availableDevices_.push_back(DeviceType_Nosepiece);
        detectedDevicesByName_.push_back(g_NosepieceDeviceName);
        model_.SetDevicePresent(DeviceType_Nosepiece, true);
    }

    if (QueryMagnification() == DEVICE_OK)
    {
        availableDevices_.push_back(DeviceType_Magnification);
        detectedDevicesByName_.push_back(g_MagnificationDeviceName);
        model_.SetDevicePresent(DeviceType_Magnification, true);
    }

    if (QueryLightPath() == DEVICE_OK)
    {
        availableDevices_.push_back(DeviceType_LightPath);
        detectedDevicesByName_.push_back(g_LightPathDeviceName);
        model_.SetDevicePresent(DeviceType_LightPath, true);
    }

    if (QueryCondenserTurret() == DEVICE_OK)
    {
        availableDevices_.push_back(DeviceType_CondenserTurret);
        detectedDevicesByName_.push_back(g_CondenserTurretDeviceName);
        model_.SetDevicePresent(DeviceType_CondenserTurret, true);
    }

    if (QueryDIAAperture() == DEVICE_OK)
    {
        availableDevices_.push_back(DeviceType_DIAAperture);
        detectedDevicesByName_.push_back(g_DIAShutterDeviceName);
        model_.SetDevicePresent(DeviceType_DIAAperture, true);
    }

    if (QueryPolarizer() == DEVICE_OK)
    {
        availableDevices_.push_back(DeviceType_Polarizer);
        detectedDevicesByName_.push_back(g_PolarizerDeviceName);
        model_.SetDevicePresent(DeviceType_Polarizer, true);
    }

    if (QueryDICPrism() == DEVICE_OK)
    {
        availableDevices_.push_back(DeviceType_DICPrism);
        detectedDevicesByName_.push_back(g_DICPrismDeviceName);
        model_.SetDevicePresent(DeviceType_DICPrism, true);
    }

    if (QueryEPIShutter1() == DEVICE_OK)
    {
        availableDevices_.push_back(DeviceType_EPIShutter1);
        detectedDevicesByName_.push_back(g_EPIShutter1DeviceName);
        model_.SetDevicePresent(DeviceType_EPIShutter1, true);
    }

    if (QueryMirrorUnit1() == DEVICE_OK)
    {
        availableDevices_.push_back(DeviceType_MirrorUnit1);
        detectedDevicesByName_.push_back(g_MirrorUnit1DeviceName);
        model_.SetDevicePresent(DeviceType_MirrorUnit1, true);
    }

    if (QueryEPIND() == DEVICE_OK)
    {
        availableDevices_.push_back(DeviceType_EPIND);
        detectedDevicesByName_.push_back(g_EPINDDeviceName);
        model_.SetDevicePresent(DeviceType_EPIND, true);
    }

    if (QueryCorrectionCollar() == DEVICE_OK)
    {
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

bool EvidentHub::IsDevicePresent(EvidentIX85::DeviceType type) const
{
    return model_.IsDevicePresent(type);
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

    if (IsUnknown(response))
        return ERR_DEVICE_NOT_AVAILABLE;

    std::vector<std::string> params = ParseParameters(response);
    if (params.size() > 0 && params[0] != "X")
    {
        int pos = ParseIntParameter(params[0]);
        model_.SetPosition(DeviceType_Polarizer, pos);
        return DEVICE_OK;
    }

    return ERR_DEVICE_NOT_AVAILABLE;
}

int EvidentHub::QueryDICPrism()
{
    std::string cmd = BuildQuery(CMD_DIC_PRISM);
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
        model_.SetPosition(DeviceType_DICPrism, pos);
        return DEVICE_OK;
    }

    return ERR_DEVICE_NOT_AVAILABLE;
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
    monitorThread_.join();

    LogMessage("Monitoring thread stopped", true);
}

void EvidentHub::MonitorThreadFunc()
{
    // This function runs in a separate thread and monitors for
    // active notifications from the microscope

    LogMessage("Monitor thread function started", true);

    const int pollIntervalMs = 10;
    std::string buffer;

    while (!stopMonitoring_.load())
    {
        // Try to read available data without blocking command execution
        // Only lock briefly to check for data
        unsigned char byte;
        unsigned long read;

        {
            std::lock_guard<std::mutex> lock(commandMutex_);
            int ret = ReadFromComPort(port_.c_str(), &byte, 1, read);
            if (ret != DEVICE_OK || read == 0)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(pollIntervalMs));
                continue;
            }
        }

        // Process received byte
        if (byte == '\n')
        {
            // End of line - remove trailing \r if present
            if (!buffer.empty() && buffer.back() == '\r')
                buffer.pop_back();

            if (!buffer.empty())
            {
                // Parse notification
                std::string tag = ExtractTag(buffer);
                std::vector<std::string> params = ParseParameters(buffer);

                LogMessage(("Notification: " + buffer).c_str(), true);

                // Update model based on notification
                if (tag == CMD_FOCUS_NOTIFY && params.size() > 0)
                {
                    long pos = ParseLongParameter(params[0]);
                    if (pos >= 0)
                    {
                        model_.SetPosition(DeviceType_Focus, pos);
                        // Check if we've reached the target position
                        long targetPos = model_.GetTargetPosition(DeviceType_Focus);
                        if (targetPos >= 0 && pos == targetPos)
                        {
                            model_.SetBusy(DeviceType_Focus, false);
                        }
                    }
                }
                else if (tag == CMD_NOSEPIECE_NOTIFY && params.size() > 0)
                {
                    int pos = ParseIntParameter(params[0]);
                    if (pos >= 0)
                    {
                        model_.SetPosition(DeviceType_Nosepiece, pos);
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
                       const MM::Device* pDev = usedDevices_.find(DeviceType_Magnification)->second;
                       if (pDev != 0) {
                          GetCoreCallback()->OnPropertyChanged(pDev, "State", CDeviceUtils::ConvertToString(pos));
                       }
                    }
                }
                // Add more notification handlers as needed

                buffer.clear();
            }
        }
        else
        {
            buffer += static_cast<char>(byte);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    LogMessage("Monitor thread function exiting", true);
}
