///////////////////////////////////////////////////////////////////////////////
// FILE:          EvidentIX85.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Evident IX85 microscope device implementations
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

#include "EvidentIX85.h"
#include "ModuleInterface.h"
#include <sstream>

using namespace EvidentIX85;

// External hub name
extern const char* g_HubDeviceName;

// Device names
const char* g_FocusDeviceName = "IX85-Focus";
const char* g_NosepieceDeviceName = "IX85-Nosepiece";
const char* g_MagnificationDeviceName = "IX85-Magnification";
const char* g_LightPathDeviceName = "IX85-LightPath";
const char* g_CondenserTurretDeviceName = "IX85-CondenserTurret";
const char* g_DIAShutterDeviceName = "IX85-DIAShutter";
const char* g_EPIShutter1DeviceName = "IX85-EPIShutter1";
const char* g_EPIShutter2DeviceName = "IX85-EPIShutter2";
const char* g_MirrorUnit1DeviceName = "IX85-MirrorUnit1";
const char* g_MirrorUnit2DeviceName = "IX85-MirrorUnit2";
const char* g_PolarizerDeviceName = "IX85-Polarizer";
const char* g_DICPrismDeviceName = "IX85-DICPrism";
const char* g_EPINDDeviceName = "IX85-EPIND";
const char* g_CorrectionCollarDeviceName = "IX85-CorrectionCollar";

// Property Names
const char* g_Keyword_Magnification = "Magnification";

///////////////////////////////////////////////////////////////////////////////
// MODULE_API - Exported MMDevice interface
///////////////////////////////////////////////////////////////////////////////

MODULE_API void InitializeModuleData()
{
    RegisterDevice(g_HubDeviceName, MM::HubDevice, "Evident IX85 Hub");
    RegisterDevice(g_FocusDeviceName, MM::StageDevice, "Evident IX85 Focus Drive");
    RegisterDevice(g_NosepieceDeviceName, MM::StateDevice, "Evident IX85 Nosepiece");
    RegisterDevice(g_MagnificationDeviceName, MM::MagnifierDevice, "Evident IX85 Magnification Changer");
    RegisterDevice(g_LightPathDeviceName, MM::StateDevice, "Evident IX85 Light Path");
    RegisterDevice(g_CondenserTurretDeviceName, MM::StateDevice, "Evident IX85 Condenser Turret");
    RegisterDevice(g_DIAShutterDeviceName, MM::ShutterDevice, "Evident IX85 DIA Shutter");
    RegisterDevice(g_EPIShutter1DeviceName, MM::ShutterDevice, "Evident IX85 EPI Shutter 1");
    RegisterDevice(g_EPIShutter2DeviceName, MM::ShutterDevice, "Evident IX85 EPI Shutter 2");
    RegisterDevice(g_MirrorUnit1DeviceName, MM::StateDevice, "Evident IX85 Mirror Unit 1");
    RegisterDevice(g_MirrorUnit2DeviceName, MM::StateDevice, "Evident IX85 Mirror Unit 2");
    RegisterDevice(g_PolarizerDeviceName, MM::StateDevice, "Evident IX85 Polarizer");
    RegisterDevice(g_DICPrismDeviceName, MM::StateDevice, "Evident IX85 DIC Prism");
    RegisterDevice(g_EPINDDeviceName, MM::StateDevice, "Evident IX85 EPI ND Filter");
    RegisterDevice(g_CorrectionCollarDeviceName, MM::GenericDevice, "Evident IX85 Correction Collar");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
    if (deviceName == nullptr)
        return nullptr;

    if (strcmp(deviceName, g_HubDeviceName) == 0)
        return new EvidentHub();
    else if (strcmp(deviceName, g_FocusDeviceName) == 0)
        return new EvidentFocus();
    else if (strcmp(deviceName, g_NosepieceDeviceName) == 0)
        return new EvidentNosepiece();
    else if (strcmp(deviceName, g_MagnificationDeviceName) == 0)
        return new EvidentMagnification();
    else if (strcmp(deviceName, g_LightPathDeviceName) == 0)
        return new EvidentLightPath();
    else if (strcmp(deviceName, g_CondenserTurretDeviceName) == 0)
        return new EvidentCondenserTurret();
    else if (strcmp(deviceName, g_DIAShutterDeviceName) == 0)
        return new EvidentDIAShutter();
    else if (strcmp(deviceName, g_EPIShutter1DeviceName) == 0)
        return new EvidentEPIShutter1();
    else if (strcmp(deviceName, g_EPIShutter2DeviceName) == 0)
        return new EvidentEPIShutter2();
    else if (strcmp(deviceName, g_MirrorUnit1DeviceName) == 0)
        return new EvidentMirrorUnit1();
    else if (strcmp(deviceName, g_MirrorUnit2DeviceName) == 0)
        return new EvidentMirrorUnit2();
    else if (strcmp(deviceName, g_PolarizerDeviceName) == 0)
        return new EvidentPolarizer();
    else if (strcmp(deviceName, g_DICPrismDeviceName) == 0)
        return new EvidentDICPrism();
    else if (strcmp(deviceName, g_EPINDDeviceName) == 0)
        return new EvidentEPIND();
    else if (strcmp(deviceName, g_CorrectionCollarDeviceName) == 0)
        return new EvidentCorrectionCollar();

    return nullptr;
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
    delete pDevice;
}

///////////////////////////////////////////////////////////////////////////////
// EvidentFocus - Focus Drive Implementation
///////////////////////////////////////////////////////////////////////////////

EvidentFocus::EvidentFocus() :
    initialized_(false),
    name_(g_FocusDeviceName),
    stepSizeUm_(FOCUS_STEP_SIZE_UM)
{
    InitializeDefaultErrorMessages();
    SetErrorText(ERR_DEVICE_NOT_AVAILABLE, "Focus drive not available on this microscope");

    // Parent ID for hub
    CreateHubIDProperty();
}

EvidentFocus::~EvidentFocus()
{
    Shutdown();
}

void EvidentFocus::GetName(char* pszName) const
{
    CDeviceUtils::CopyLimitedString(pszName, name_.c_str());
}

int EvidentFocus::Initialize()
{
    if (initialized_)
        return DEVICE_OK;

    EvidentHub* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    if (!hub->IsDevicePresent(DeviceType_Focus))
        return ERR_DEVICE_NOT_AVAILABLE;

    // Create properties
    CPropertyAction* pAct = new CPropertyAction(this, &EvidentFocus::OnPosition);
    int ret = CreateProperty(MM::g_Keyword_Position, "0.0", MM::Float, false, pAct);
    if (ret != DEVICE_OK)
        return ret;

    pAct = new CPropertyAction(this, &EvidentFocus::OnSpeed);
    ret = CreateProperty("Speed (um/s)", "30.0", MM::Float, false, pAct);
    if (ret != DEVICE_OK)
        return ret;

    // Add firmware version as read-only property
    std::string version = hub->GetDeviceVersion(DeviceType_Focus);
    if (!version.empty())
    {
        ret = CreateProperty("Firmware Version", version.c_str(), MM::String, true);
        if (ret != DEVICE_OK)
            return ret;
    }

    // Enable active notifications
    ret = EnableNotifications(true);
    if (ret != DEVICE_OK)
        return ret;

    initialized_ = true;
    return DEVICE_OK;
}

int EvidentFocus::Shutdown()
{
    if (initialized_)
    {
        EnableNotifications(false);
        initialized_ = false;
    }
    return DEVICE_OK;
}

bool EvidentFocus::Busy()
{
    EvidentHub* hub = GetHub();
    if (!hub)
        return false;

    return hub->GetModel()->IsBusy(DeviceType_Focus);
}

int EvidentFocus::SetPositionUm(double pos)
{
    EvidentHub* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    // Convert μm to 10nm units
    long steps = static_cast<long>(pos / stepSizeUm_);

    // Clamp to limits
    if (steps < FOCUS_MIN_POS) steps = FOCUS_MIN_POS;
    if (steps > FOCUS_MAX_POS) steps = FOCUS_MAX_POS;

    // Check if we're already at the target position
    long currentPos = hub->GetModel()->GetPosition(DeviceType_Focus);
    bool alreadyAtTarget = IsAtTargetPosition(currentPos, steps, FOCUS_POSITION_TOLERANCE);

    // Set target position BEFORE sending command so notifications can check against it
    hub->GetModel()->SetTargetPosition(DeviceType_Focus, steps);
    hub->GetModel()->SetBusy(DeviceType_Focus, true);

    std::string cmd = BuildCommand(CMD_FOCUS_GOTO, static_cast<int>(steps));
    std::string response;
    int ret = hub->ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
    {
        // Command failed - clear busy state
        hub->GetModel()->SetBusy(DeviceType_Focus, false);
        return ret;
    }

    if (!IsPositiveAck(response, CMD_FOCUS_GOTO))
    {
        // Command rejected - clear busy state
        hub->GetModel()->SetBusy(DeviceType_Focus, false);
        return ERR_NEGATIVE_ACK;
    }

    // If we're already at the target, firmware won't send notifications, so clear busy immediately
    if (alreadyAtTarget)
    {
        hub->GetModel()->SetBusy(DeviceType_Focus, false);
    }

    // Command accepted - if not already at target, busy will be cleared by notification when target reached
    return DEVICE_OK;
}

int EvidentFocus::GetPositionUm(double& pos)
{
    EvidentHub* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    long steps = hub->GetModel()->GetPosition(DeviceType_Focus);
    if (steps < 0)  // Unknown
        return ERR_POSITION_UNKNOWN;

    pos = steps * stepSizeUm_;
    return DEVICE_OK;
}

int EvidentFocus::SetPositionSteps(long steps)
{
    return SetPositionUm(steps * stepSizeUm_);
}

int EvidentFocus::GetPositionSteps(long& steps)
{
    EvidentHub* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    steps = hub->GetModel()->GetPosition(DeviceType_Focus);
    if (steps < 0)
        return ERR_POSITION_UNKNOWN;

    return DEVICE_OK;
}

int EvidentFocus::SetOrigin()
{
    // Not supported by IX85
    return DEVICE_UNSUPPORTED_COMMAND;
}

int EvidentFocus::GetLimits(double& lower, double& upper)
{
    lower = FOCUS_MIN_POS * stepSizeUm_;
    upper = FOCUS_MAX_POS * stepSizeUm_;
    return DEVICE_OK;
}

int EvidentFocus::OnPosition(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        double pos;
        int ret = GetPositionUm(pos);
        if (ret != DEVICE_OK)
            return ret;
        pProp->Set(pos);
    }
    else if (eAct == MM::AfterSet)
    {
        double pos;
        pProp->Get(pos);
        int ret = SetPositionUm(pos);
        if (ret != DEVICE_OK)
            return ret;
    }
    return DEVICE_OK;
}

int EvidentFocus::OnSpeed(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        // Query current speed
        EvidentHub* hub = GetHub();
        if (!hub)
            return DEVICE_ERR;

        std::string cmd = BuildQuery(CMD_FOCUS_SPEED);
        std::string response;
        int ret = hub->ExecuteCommand(cmd, response);
        if (ret != DEVICE_OK)
            return ret;

        std::vector<std::string> params = ParseParameters(response);
        if (params.size() >= 2)
        {
            long highSpeed = ParseLongParameter(params[1]);  // High speed in 10nm/s
            double speedUmPerSec = (highSpeed * stepSizeUm_);
            pProp->Set(speedUmPerSec);
        }
    }
    else if (eAct == MM::AfterSet)
    {
        double speedUmPerSec;
        pProp->Get(speedUmPerSec);

        // Convert to 10nm/s
        long highSpeed = static_cast<long>(speedUmPerSec / stepSizeUm_);
        long initialSpeed = highSpeed / 10;  // Default: 10% of high speed
        long acceleration = highSpeed * 5;   // Default acceleration

        std::string cmd = BuildCommand(CMD_FOCUS_SPEED,
                                       static_cast<int>(initialSpeed),
                                       static_cast<int>(highSpeed),
                                       static_cast<int>(acceleration));

        EvidentHub* hub = GetHub();
        if (!hub)
            return DEVICE_ERR;

        std::string response;
        int ret = hub->ExecuteCommand(cmd, response);
        if (ret != DEVICE_OK)
            return ret;

        if (!IsPositiveAck(response, CMD_FOCUS_SPEED))
            return ERR_NEGATIVE_ACK;
    }
    return DEVICE_OK;
}

EvidentHub* EvidentFocus::GetHub()
{
    MM::Hub* hub = GetParentHub();
    if (!hub)
        return nullptr;
    return dynamic_cast<EvidentHub*>(hub);
}

int EvidentFocus::EnableNotifications(bool enable)
{
    EvidentHub* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    return hub->EnableNotification(CMD_FOCUS_NOTIFY, enable);
}

///////////////////////////////////////////////////////////////////////////////
// EvidentNosepiece - Nosepiece Implementation
///////////////////////////////////////////////////////////////////////////////

EvidentNosepiece::EvidentNosepiece() :
    initialized_(false),
    name_(g_NosepieceDeviceName),
    numPos_(NOSEPIECE_MAX_POS),
    safeNosepieceChange_(true)
{
    InitializeDefaultErrorMessages();
    SetErrorText(ERR_DEVICE_NOT_AVAILABLE, "Nosepiece not available on this microscope");

    CreateHubIDProperty();
}

EvidentNosepiece::~EvidentNosepiece()
{
    Shutdown();
}

void EvidentNosepiece::GetName(char* pszName) const
{
    CDeviceUtils::CopyLimitedString(pszName, name_.c_str());
}

int EvidentNosepiece::Initialize()
{
    if (initialized_)
        return DEVICE_OK;

    EvidentHub* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    if (!hub->IsDevicePresent(DeviceType_Nosepiece))
        return ERR_DEVICE_NOT_AVAILABLE;

    numPos_ = hub->GetModel()->GetNumPositions(DeviceType_Nosepiece);

    // Create properties
    CPropertyAction* pAct = new CPropertyAction(this, &EvidentNosepiece::OnState);
    int ret = CreateProperty(MM::g_Keyword_State, "0", MM::Integer, false, pAct);
    if (ret != DEVICE_OK)
        return ret;

    SetPropertyLimits(MM::g_Keyword_State, 0, numPos_ - 1);

    // Create label property
    pAct = new CPropertyAction(this, &CStateDeviceBase::OnLabel);
    ret = CreateProperty(MM::g_Keyword_Label, "", MM::String, false, pAct);
    if (ret != DEVICE_OK)
        return ret;

    // Define labels
    for (unsigned int i = 0; i < numPos_; i++)
    {
        std::ostringstream label;
        label << "Position-" << (i + 1);
        SetPositionLabel(i, label.str().c_str());
    }

    // Create SafeNosepieceChange property
    pAct = new CPropertyAction(this, &EvidentNosepiece::OnSafeChange);
    ret = CreateProperty("SafeNosepieceChange", "Enabled", MM::String, false, pAct);
    if (ret != DEVICE_OK)
        return ret;
    AddAllowedValue("SafeNosepieceChange", "Disabled");
    AddAllowedValue("SafeNosepieceChange", "Enabled");

    // Add firmware version as read-only property
    std::string version = hub->GetDeviceVersion(DeviceType_Nosepiece);
    if (!version.empty())
    {
        ret = CreateProperty("Firmware Version", version.c_str(), MM::String, true);
        if (ret != DEVICE_OK)
            return ret;
    }

    // Enable notifications
    ret = EnableNotifications(true);
    if (ret != DEVICE_OK)
        return ret;

    initialized_ = true;
    return DEVICE_OK;
}

int EvidentNosepiece::Shutdown()
{
    if (initialized_)
    {
        EnableNotifications(false);
        initialized_ = false;
    }
    return DEVICE_OK;
}

bool EvidentNosepiece::Busy()
{
    EvidentHub* hub = GetHub();
    if (!hub)
        return false;

    return hub->GetModel()->IsBusy(DeviceType_Nosepiece);
}

unsigned long EvidentNosepiece::GetNumberOfPositions() const
{
    return numPos_;
}

int EvidentNosepiece::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        EvidentHub* hub = GetHub();
        if (!hub)
            return DEVICE_ERR;

        long pos = hub->GetModel()->GetPosition(DeviceType_Nosepiece);
        if (pos < 0)
            return ERR_POSITION_UNKNOWN;

        // Convert from 1-based to 0-based
        pProp->Set(pos - 1);
    }
    else if (eAct == MM::AfterSet)
    {
        long pos;
        pProp->Get(pos);

        EvidentHub* hub = GetHub();
        if (!hub)
            return DEVICE_ERR;

        // Use safe nosepiece change if enabled
        if (safeNosepieceChange_)
        {
            return SafeNosepieceChange(pos + 1);  // Convert 0-based to 1-based
        }

        // Direct nosepiece change (original behavior)
        // Set target position BEFORE sending command so notifications can check against it
        // Convert from 0-based to 1-based for the microscope
        hub->GetModel()->SetTargetPosition(DeviceType_Nosepiece, pos + 1);
        hub->GetModel()->SetBusy(DeviceType_Nosepiece, true);

        std::string cmd = BuildCommand(CMD_NOSEPIECE, static_cast<int>(pos + 1));
        std::string response;
        int ret = hub->ExecuteCommand(cmd, response);
        if (ret != DEVICE_OK)
        {
            // Command failed - clear busy state
            hub->GetModel()->SetBusy(DeviceType_Nosepiece, false);
            return ret;
        }

        if (!IsPositiveAck(response, CMD_NOSEPIECE))
        {
            // Command rejected - clear busy state
            hub->GetModel()->SetBusy(DeviceType_Nosepiece, false);
            return ERR_NEGATIVE_ACK;
        }

        // Command accepted - busy state already set, will be cleared by notification when target reached
    }
    return DEVICE_OK;
}

EvidentHub* EvidentNosepiece::GetHub()
{
    MM::Hub* hub = GetParentHub();
    if (!hub)
        return nullptr;
    return dynamic_cast<EvidentHub*>(hub);
}

int EvidentNosepiece::EnableNotifications(bool enable)
{
    EvidentHub* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    return hub->EnableNotification(CMD_NOSEPIECE_NOTIFY, enable);
}

int EvidentNosepiece::OnSafeChange(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(safeNosepieceChange_ ? "Enabled" : "Disabled");
    }
    else if (eAct == MM::AfterSet)
    {
        std::string value;
        pProp->Get(value);
        safeNosepieceChange_ = (value == "Enabled");
    }
    return DEVICE_OK;
}

int EvidentNosepiece::SafeNosepieceChange(long targetPosition)
{
    EvidentHub* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    // Check if Focus device is available
    if (!hub->IsDevicePresent(DeviceType_Focus))
    {
        // No focus device - just do a regular nosepiece change
        LogMessage("Focus device not available, skipping safe nosepiece change");
        hub->GetModel()->SetTargetPosition(DeviceType_Nosepiece, targetPosition);
        hub->GetModel()->SetBusy(DeviceType_Nosepiece, true);

        std::string cmd = BuildCommand(CMD_NOSEPIECE, static_cast<int>(targetPosition));
        std::string response;
        int ret = hub->ExecuteCommand(cmd, response);
        if (ret != DEVICE_OK)
        {
            hub->GetModel()->SetBusy(DeviceType_Nosepiece, false);
            return ret;
        }
        if (!IsPositiveAck(response, CMD_NOSEPIECE))
        {
            hub->GetModel()->SetBusy(DeviceType_Nosepiece, false);
            return ERR_NEGATIVE_ACK;
        }
        return DEVICE_OK;
    }

    // Get current focus position
    long originalFocusPos = hub->GetModel()->GetPosition(DeviceType_Focus);
    if (originalFocusPos < 0)
    {
        LogMessage("Focus position unknown, cannot perform safe nosepiece change");
        return ERR_POSITION_UNKNOWN;
    }

    LogMessage("Safe nosepiece change: Moving focus to zero");

    // Check if focus is already at zero
    bool alreadyAtZero = IsAtTargetPosition(originalFocusPos, 0, FOCUS_POSITION_TOLERANCE);

    // Move focus to zero
    hub->GetModel()->SetTargetPosition(DeviceType_Focus, 0);
    hub->GetModel()->SetBusy(DeviceType_Focus, true);

    std::string cmd = BuildCommand(CMD_FOCUS_GOTO, 0);
    std::string response;
    int ret = hub->ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
    {
        hub->GetModel()->SetBusy(DeviceType_Focus, false);
        return ret;
    }
    if (!IsPositiveAck(response, CMD_FOCUS_GOTO))
    {
        hub->GetModel()->SetBusy(DeviceType_Focus, false);
        return ERR_NEGATIVE_ACK;
    }

    // If already at zero, firmware won't send notifications, so clear busy immediately
    if (alreadyAtZero)
    {
        hub->GetModel()->SetBusy(DeviceType_Focus, false);
    }

    // Wait for focus to reach zero (with timeout)
    int focusWaitCount = 0;
    const int maxWaitIterations = 100;  // 10 seconds max
    while (hub->GetModel()->IsBusy(DeviceType_Focus) && focusWaitCount < maxWaitIterations)
    {
        CDeviceUtils::SleepMs(100);
        focusWaitCount++;
    }

    if (focusWaitCount >= maxWaitIterations)
    {
        LogMessage("Timeout waiting for focus to reach zero");
        return ERR_COMMAND_TIMEOUT;
    }

    LogMessage("Safe nosepiece change: Changing nosepiece position");

    // Change nosepiece position
    hub->GetModel()->SetTargetPosition(DeviceType_Nosepiece, targetPosition);
    hub->GetModel()->SetBusy(DeviceType_Nosepiece, true);

    cmd = BuildCommand(CMD_NOSEPIECE, static_cast<int>(targetPosition));
    ret = hub->ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
    {
        hub->GetModel()->SetBusy(DeviceType_Nosepiece, false);
        // Try to restore focus position even if nosepiece change failed
        hub->GetModel()->SetTargetPosition(DeviceType_Focus, originalFocusPos);
        hub->GetModel()->SetBusy(DeviceType_Focus, true);
        std::string focusCmd = BuildCommand(CMD_FOCUS_GOTO, static_cast<int>(originalFocusPos));
        std::string focusResponse;
        hub->ExecuteCommand(focusCmd, focusResponse);
        return ret;
    }

    if (!IsPositiveAck(response, CMD_NOSEPIECE))
    {
        hub->GetModel()->SetBusy(DeviceType_Nosepiece, false);
        // Try to restore focus position even if nosepiece change failed
        hub->GetModel()->SetTargetPosition(DeviceType_Focus, originalFocusPos);
        hub->GetModel()->SetBusy(DeviceType_Focus, true);
        std::string focusCmd = BuildCommand(CMD_FOCUS_GOTO, static_cast<int>(originalFocusPos));
        std::string focusResponse;
        hub->ExecuteCommand(focusCmd, focusResponse);
        return ERR_NEGATIVE_ACK;
    }

    // Wait for nosepiece to complete (with timeout)
    int nosepieceWaitCount = 0;
    while (hub->GetModel()->IsBusy(DeviceType_Nosepiece) && nosepieceWaitCount < maxWaitIterations)
    {
        CDeviceUtils::SleepMs(100);
        nosepieceWaitCount++;
    }

    if (nosepieceWaitCount >= maxWaitIterations)
    {
        LogMessage("Timeout waiting for nosepiece to complete");
        return ERR_COMMAND_TIMEOUT;
    }

    LogMessage("Safe nosepiece change: Restoring focus position");

    // Check if we're already at the target position (originalFocusPos)
    long currentFocusPos = hub->GetModel()->GetPosition(DeviceType_Focus);
    bool alreadyAtTarget = IsAtTargetPosition(currentFocusPos, originalFocusPos, FOCUS_POSITION_TOLERANCE);

    // Restore original focus position
    hub->GetModel()->SetTargetPosition(DeviceType_Focus, originalFocusPos);
    hub->GetModel()->SetBusy(DeviceType_Focus, true);

    cmd = BuildCommand(CMD_FOCUS_GOTO, static_cast<int>(originalFocusPos));
    ret = hub->ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
    {
        hub->GetModel()->SetBusy(DeviceType_Focus, false);
        return ret;
    }

    if (!IsPositiveAck(response, CMD_FOCUS_GOTO))
    {
        hub->GetModel()->SetBusy(DeviceType_Focus, false);
        return ERR_NEGATIVE_ACK;
    }

    // If already at target, firmware won't send notifications, so clear busy immediately
    if (alreadyAtTarget)
    {
        hub->GetModel()->SetBusy(DeviceType_Focus, false);
    }

    LogMessage("Safe nosepiece change completed successfully");
    return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// EvidentMagnification - Magnification Changer Implementation
///////////////////////////////////////////////////////////////////////////////

// Static magnification values
const double EvidentMagnification::magnifications_[3] = {1.0, 1.6, 2.0};

EvidentMagnification::EvidentMagnification() :
    initialized_(false),
    name_(g_MagnificationDeviceName),
    numPos_(MAGNIFICATION_MAX_POS)
{
    InitializeDefaultErrorMessages();
    SetErrorText(ERR_DEVICE_NOT_AVAILABLE, "Magnification changer not available on this microscope");

    CreateHubIDProperty();
}

EvidentMagnification::~EvidentMagnification()
{
    Shutdown();
}

void EvidentMagnification::GetName(char* pszName) const
{
    CDeviceUtils::CopyLimitedString(pszName, name_.c_str());
}

int EvidentMagnification::Initialize()
{
    if (initialized_)
        return DEVICE_OK;

    EvidentHub* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    if (!hub->IsDevicePresent(DeviceType_Magnification))
        return ERR_DEVICE_NOT_AVAILABLE;

    numPos_ = hub->GetModel()->GetNumPositions(DeviceType_Magnification);

    // Create magnification property (read-only)
    CPropertyAction* pAct = new CPropertyAction(this, &EvidentMagnification::OnMagnification);
    int ret = CreateProperty(g_Keyword_Magnification, "1.0", MM::Float, true, pAct);
    if (ret != DEVICE_OK)
        return ret;

    // Set allowed values
    for (unsigned int i = 0; i < numPos_; i++)
    {
        std::ostringstream os;
        os << magnifications_[i];
        AddAllowedValue(g_Keyword_Magnification, os.str().c_str());
    }

    hub->RegisterDeviceAsUsed(DeviceType_Magnification, this);

    // Enable notifications
    ret = EnableNotifications(true);
    if (ret != DEVICE_OK)
        return ret;

    initialized_ = true;
    return DEVICE_OK;
}

int EvidentMagnification::Shutdown()
{
    if (initialized_)
    {
        EnableNotifications(false);
        GetHub()->UnRegisterDeviceAsUsed(DeviceType_Magnification);
        initialized_ = false;
    }
    return DEVICE_OK;
}

bool EvidentMagnification::Busy()
{
    EvidentHub* hub = GetHub();
    if (!hub)
        return false;

    return hub->GetModel()->IsBusy(DeviceType_Magnification);
}

double EvidentMagnification::GetMagnification()
{
    EvidentHub* hub = GetHub();
    if (!hub)
        return 1.0;

    long pos = hub->GetModel()->GetPosition(DeviceType_Magnification);
    if (pos < 1 || pos > static_cast<long>(numPos_))
        return 1.0;

    // pos is 1-based, array is 0-based
    return magnifications_[pos - 1];
}

int EvidentMagnification::OnMagnification(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        double mag = GetMagnification();
        pProp->Set(mag);
    }
    else if (eAct == MM::AfterSet)
    {
        // Read-only - nothing to do
    }
    return DEVICE_OK;
}

EvidentHub* EvidentMagnification::GetHub()
{
    MM::Hub* hub = GetParentHub();
    if (!hub)
        return nullptr;
    return dynamic_cast<EvidentHub*>(hub);
}

int EvidentMagnification::EnableNotifications(bool enable)
{
    EvidentHub* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    return hub->EnableNotification(CMD_MAGNIFICATION_NOTIFY, enable);
}

///////////////////////////////////////////////////////////////////////////////
// EvidentLightPath - Light Path Selector Implementation
///////////////////////////////////////////////////////////////////////////////

EvidentLightPath::EvidentLightPath() :
    initialized_(false),
    name_(g_LightPathDeviceName)
{
    InitializeDefaultErrorMessages();
    SetErrorText(ERR_DEVICE_NOT_AVAILABLE, "Light path selector not available on this microscope");

    CreateHubIDProperty();
}

EvidentLightPath::~EvidentLightPath()
{
    Shutdown();
}

void EvidentLightPath::GetName(char* pszName) const
{
    CDeviceUtils::CopyLimitedString(pszName, name_.c_str());
}

int EvidentLightPath::Initialize()
{
    if (initialized_)
        return DEVICE_OK;

    EvidentHub* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    if (!hub->IsDevicePresent(DeviceType_LightPath))
        return ERR_DEVICE_NOT_AVAILABLE;

    // Create properties
    CPropertyAction* pAct = new CPropertyAction(this, &EvidentLightPath::OnState);
    int ret = CreateProperty(MM::g_Keyword_State, "0", MM::Integer, false, pAct);
    if (ret != DEVICE_OK)
        return ret;

    SetPropertyLimits(MM::g_Keyword_State, 0, 3);

    // Create label property
    pAct = new CPropertyAction(this, &CStateDeviceBase::OnLabel);
    ret = CreateProperty(MM::g_Keyword_Label, "", MM::String, false, pAct);
    if (ret != DEVICE_OK)
        return ret;

    // Define labels for light path positions
    SetPositionLabel(0, "Left Port");       // LIGHT_PATH_LEFT_PORT = 1
    SetPositionLabel(1, "Binocular 50/50"); // LIGHT_PATH_BI_50_50 = 2
    SetPositionLabel(2, "Binocular 100%");  // LIGHT_PATH_BI_100 = 3
    SetPositionLabel(3, "Right Port");      // LIGHT_PATH_RIGHT_PORT = 4

    // Add firmware version as read-only property
    std::string version = hub->GetDeviceVersion(DeviceType_LightPath);
    if (!version.empty())
    {
        ret = CreateProperty("Firmware Version", version.c_str(), MM::String, true);
        if (ret != DEVICE_OK)
            return ret;
    }

    initialized_ = true;
    return DEVICE_OK;
}

int EvidentLightPath::Shutdown()
{
    if (initialized_)
    {
        initialized_ = false;
    }
    return DEVICE_OK;
}

bool EvidentLightPath::Busy()
{
    return false;  // Light path changes are instantaneous
}

unsigned long EvidentLightPath::GetNumberOfPositions() const
{
    return 4;
}

int EvidentLightPath::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        EvidentHub* hub = GetHub();
        if (!hub)
            return DEVICE_ERR;

        // Query current position
        std::string cmd = BuildQuery(CMD_LIGHT_PATH);
        std::string response;
        int ret = hub->ExecuteCommand(cmd, response);
        if (ret != DEVICE_OK)
            return ret;

        std::vector<std::string> params = ParseParameters(response);
        if (params.size() > 0)
        {
            int pos = ParseIntParameter(params[0]);
            if (pos >= 1 && pos <= 4)
            {
                // Convert from 1-based to 0-based
                pProp->Set(static_cast<long>(pos - 1));
            }
        }
    }
    else if (eAct == MM::AfterSet)
    {
        long pos;
        pProp->Get(pos);

        EvidentHub* hub = GetHub();
        if (!hub)
            return DEVICE_ERR;

        // Convert from 0-based to 1-based for the microscope
        std::string cmd = BuildCommand(CMD_LIGHT_PATH, static_cast<int>(pos + 1));
        std::string response;
        int ret = hub->ExecuteCommand(cmd, response);
        if (ret != DEVICE_OK)
            return ret;

        if (!IsPositiveAck(response, CMD_LIGHT_PATH))
            return ERR_NEGATIVE_ACK;
    }
    return DEVICE_OK;
}

EvidentHub* EvidentLightPath::GetHub()
{
    MM::Hub* hub = GetParentHub();
    if (!hub)
        return nullptr;
    return dynamic_cast<EvidentHub*>(hub);
}

///////////////////////////////////////////////////////////////////////////////
// EvidentCondenserTurret - Condenser Turret Implementation
///////////////////////////////////////////////////////////////////////////////

EvidentCondenserTurret::EvidentCondenserTurret() :
    initialized_(false),
    name_(g_CondenserTurretDeviceName),
    numPos_(6)
{
    InitializeDefaultErrorMessages();
    SetErrorText(ERR_DEVICE_NOT_AVAILABLE, "Condenser turret not available on this microscope");

    CreateHubIDProperty();
}

EvidentCondenserTurret::~EvidentCondenserTurret()
{
    Shutdown();
}

void EvidentCondenserTurret::GetName(char* pszName) const
{
    CDeviceUtils::CopyLimitedString(pszName, name_.c_str());
}

int EvidentCondenserTurret::Initialize()
{
    if (initialized_)
        return DEVICE_OK;

    EvidentHub* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    if (!hub->IsDevicePresent(DeviceType_CondenserTurret))
        return ERR_DEVICE_NOT_AVAILABLE;

    numPos_ = hub->GetModel()->GetNumPositions(DeviceType_CondenserTurret);

    // Create properties
    CPropertyAction* pAct = new CPropertyAction(this, &EvidentCondenserTurret::OnState);
    int ret = CreateProperty(MM::g_Keyword_State, "0", MM::Integer, false, pAct);
    if (ret != DEVICE_OK)
        return ret;

    SetPropertyLimits(MM::g_Keyword_State, 0, numPos_ - 1);

    // Create label property
    pAct = new CPropertyAction(this, &CStateDeviceBase::OnLabel);
    ret = CreateProperty(MM::g_Keyword_Label, "", MM::String, false, pAct);
    if (ret != DEVICE_OK)
        return ret;

    // Define labels
    for (unsigned int i = 0; i < numPos_; i++)
    {
        std::ostringstream label;
        label << "Position-" << (i + 1);
        SetPositionLabel(i, label.str().c_str());
    }

    // Add firmware version as read-only property
    std::string version = hub->GetDeviceVersion(DeviceType_CondenserTurret);
    if (!version.empty())
    {
        ret = CreateProperty("Firmware Version", version.c_str(), MM::String, true);
        if (ret != DEVICE_OK)
            return ret;
    }

    // Enable notifications
    ret = EnableNotifications(true);
    if (ret != DEVICE_OK)
        return ret;

    initialized_ = true;
    return DEVICE_OK;
}

int EvidentCondenserTurret::Shutdown()
{
    if (initialized_)
    {
        EnableNotifications(false);
        initialized_ = false;
    }
    return DEVICE_OK;
}

bool EvidentCondenserTurret::Busy()
{
    EvidentHub* hub = GetHub();
    if (!hub)
        return false;

    return hub->GetModel()->IsBusy(DeviceType_CondenserTurret);
}

unsigned long EvidentCondenserTurret::GetNumberOfPositions() const
{
    return numPos_;
}

int EvidentCondenserTurret::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        EvidentHub* hub = GetHub();
        if (!hub)
            return DEVICE_ERR;

        long pos = hub->GetModel()->GetPosition(DeviceType_CondenserTurret);
        if (pos < 0)
            return ERR_POSITION_UNKNOWN;

        // Convert from 1-based to 0-based
        pProp->Set(pos - 1);
    }
    else if (eAct == MM::AfterSet)
    {
        long pos;
        pProp->Get(pos);

        EvidentHub* hub = GetHub();
        if (!hub)
            return DEVICE_ERR;

        // Set busy before sending command
        hub->GetModel()->SetBusy(DeviceType_CondenserTurret, true);

        // Convert from 0-based to 1-based for the microscope
        std::string cmd = BuildCommand(CMD_CONDENSER_TURRET, static_cast<int>(pos + 1));
        std::string response;
        int ret = hub->ExecuteCommand(cmd, response);
        if (ret != DEVICE_OK)
        {
            hub->GetModel()->SetBusy(DeviceType_CondenserTurret, false);
            return ret;
        }

        if (!IsPositiveAck(response, CMD_CONDENSER_TURRET))
        {
            hub->GetModel()->SetBusy(DeviceType_CondenserTurret, false);
            return ERR_NEGATIVE_ACK;
        }

        // CondenserTurret does not send notifications (NTR) when movement completes.
        // The positive ack ("TR +") is only returned after movement completes,
        // so we can clear busy immediately and update position.
        hub->GetModel()->SetPosition(DeviceType_CondenserTurret, pos + 1);
        hub->GetModel()->SetBusy(DeviceType_CondenserTurret, false);
    }
    return DEVICE_OK;
}

EvidentHub* EvidentCondenserTurret::GetHub()
{
    MM::Hub* hub = GetParentHub();
    if (!hub)
        return nullptr;
    return dynamic_cast<EvidentHub*>(hub);
}

int EvidentCondenserTurret::EnableNotifications(bool enable)
{
    EvidentHub* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    // Use TR (command tag) not NTR (notification tag) to enable notifications
    return hub->EnableNotification(CMD_CONDENSER_TURRET, enable);
}

///////////////////////////////////////////////////////////////////////////////
// EvidentDIAShutter - DIA (Transmitted Light) Shutter Implementation
///////////////////////////////////////////////////////////////////////////////

EvidentDIAShutter::EvidentDIAShutter() :
    initialized_(false),
    name_(g_DIAShutterDeviceName)
{
    InitializeDefaultErrorMessages();
    SetErrorText(ERR_DEVICE_NOT_AVAILABLE, "DIA shutter not available on this microscope");

    CreateHubIDProperty();
}

EvidentDIAShutter::~EvidentDIAShutter()
{
    Shutdown();
}

void EvidentDIAShutter::GetName(char* pszName) const
{
    CDeviceUtils::CopyLimitedString(pszName, name_.c_str());
}

int EvidentDIAShutter::Initialize()
{
    if (initialized_)
        return DEVICE_OK;

    EvidentHub* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    if (!hub->IsDevicePresent(DeviceType_DIAShutter))
        return ERR_DEVICE_NOT_AVAILABLE;

    // Create state property
    CPropertyAction* pAct = new CPropertyAction(this, &EvidentDIAShutter::OnState);
    int ret = CreateProperty(MM::g_Keyword_State, "0", MM::Integer, false, pAct);
    if (ret != DEVICE_OK)
        return ret;

    AddAllowedValue(MM::g_Keyword_State, "0");  // Closed
    AddAllowedValue(MM::g_Keyword_State, "1");  // Open

    // Add firmware version as read-only property
    std::string version = hub->GetDeviceVersion(DeviceType_DIAShutter);
    if (!version.empty())
    {
        ret = CreateProperty("Firmware Version", version.c_str(), MM::String, true);
        if (ret != DEVICE_OK)
            return ret;
    }

    initialized_ = true;
    return DEVICE_OK;
}

int EvidentDIAShutter::Shutdown()
{
    if (initialized_)
    {
        initialized_ = false;
    }
    return DEVICE_OK;
}

bool EvidentDIAShutter::Busy()
{
    return false;  // Shutter changes are instantaneous
}

int EvidentDIAShutter::SetOpen(bool open)
{
    EvidentHub* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    std::string cmd = BuildCommand(CMD_DIA_SHUTTER, open ? 1 : 0);
    std::string response;
    int ret = hub->ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
        return ret;

    if (!IsPositiveAck(response, CMD_DIA_SHUTTER))
        return ERR_NEGATIVE_ACK;

    return DEVICE_OK;
}

int EvidentDIAShutter::GetOpen(bool& open)
{
    EvidentHub* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    std::string cmd = BuildQuery(CMD_DIA_SHUTTER);
    std::string response;
    int ret = hub->ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
        return ret;

    std::vector<std::string> params = ParseParameters(response);
    if (params.size() > 0)
    {
        int state = ParseIntParameter(params[0]);
        open = (state == 1);
    }

    return DEVICE_OK;
}

int EvidentDIAShutter::Fire(double /*deltaT*/)
{
    // Not implemented for this shutter
    return DEVICE_UNSUPPORTED_COMMAND;
}

int EvidentDIAShutter::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        bool open;
        int ret = GetOpen(open);
        if (ret != DEVICE_OK)
            return ret;
        pProp->Set(open ? 1L : 0L);
    }
    else if (eAct == MM::AfterSet)
    {
        long state;
        pProp->Get(state);
        return SetOpen(state != 0);
    }
    return DEVICE_OK;
}

EvidentHub* EvidentDIAShutter::GetHub()
{
    MM::Hub* hub = GetParentHub();
    if (!hub)
        return nullptr;
    return dynamic_cast<EvidentHub*>(hub);
}

///////////////////////////////////////////////////////////////////////////////
// EvidentEPIShutter1 - EPI (Reflected Light) Shutter 1 Implementation
///////////////////////////////////////////////////////////////////////////////

EvidentEPIShutter1::EvidentEPIShutter1() :
    initialized_(false),
    name_(g_EPIShutter1DeviceName)
{
    InitializeDefaultErrorMessages();
    SetErrorText(ERR_DEVICE_NOT_AVAILABLE, "EPI shutter 1 not available on this microscope");

    CreateHubIDProperty();
}

EvidentEPIShutter1::~EvidentEPIShutter1()
{
    Shutdown();
}

void EvidentEPIShutter1::GetName(char* pszName) const
{
    CDeviceUtils::CopyLimitedString(pszName, name_.c_str());
}

int EvidentEPIShutter1::Initialize()
{
    if (initialized_)
        return DEVICE_OK;

    EvidentHub* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    if (!hub->IsDevicePresent(DeviceType_EPIShutter1))
        return ERR_DEVICE_NOT_AVAILABLE;

    // Create state property
    CPropertyAction* pAct = new CPropertyAction(this, &EvidentEPIShutter1::OnState);
    int ret = CreateProperty(MM::g_Keyword_State, "0", MM::Integer, false, pAct);
    if (ret != DEVICE_OK)
        return ret;

    AddAllowedValue(MM::g_Keyword_State, "0");  // Closed
    AddAllowedValue(MM::g_Keyword_State, "1");  // Open

    // Add firmware version as read-only property
    std::string version = hub->GetDeviceVersion(DeviceType_EPIShutter1);
    if (!version.empty())
    {
        ret = CreateProperty("Firmware Version", version.c_str(), MM::String, true);
        if (ret != DEVICE_OK)
            return ret;
    }

    initialized_ = true;
    return DEVICE_OK;
}

int EvidentEPIShutter1::Shutdown()
{
    if (initialized_)
    {
        initialized_ = false;
    }
    return DEVICE_OK;
}

bool EvidentEPIShutter1::Busy()
{
    return false;  // Shutter changes are instantaneous
}

int EvidentEPIShutter1::SetOpen(bool open)
{
    EvidentHub* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    std::string cmd = BuildCommand(CMD_EPI_SHUTTER1, open ? 1 : 0);
    std::string response;
    int ret = hub->ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
        return ret;

    if (!IsPositiveAck(response, CMD_EPI_SHUTTER1))
        return ERR_NEGATIVE_ACK;

    return DEVICE_OK;
}

int EvidentEPIShutter1::GetOpen(bool& open)
{
    EvidentHub* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    std::string cmd = BuildQuery(CMD_EPI_SHUTTER1);
    std::string response;
    int ret = hub->ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
        return ret;

    std::vector<std::string> params = ParseParameters(response);
    if (params.size() > 0)
    {
        int state = ParseIntParameter(params[0]);
        open = (state == 1);
    }

    return DEVICE_OK;
}

int EvidentEPIShutter1::Fire(double /*deltaT*/)
{
    // Not implemented for this shutter
    return DEVICE_UNSUPPORTED_COMMAND;
}

int EvidentEPIShutter1::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        bool open;
        int ret = GetOpen(open);
        if (ret != DEVICE_OK)
            return ret;
        pProp->Set(open ? 1L : 0L);
    }
    else if (eAct == MM::AfterSet)
    {
        long state;
        pProp->Get(state);
        return SetOpen(state != 0);
    }
    return DEVICE_OK;
}

EvidentHub* EvidentEPIShutter1::GetHub()
{
    MM::Hub* hub = GetParentHub();
    if (!hub)
        return nullptr;
    return dynamic_cast<EvidentHub*>(hub);
}

///////////////////////////////////////////////////////////////////////////////
// EvidentMirrorUnit1 - Mirror Unit 1 (Filter Cube Turret) Implementation
///////////////////////////////////////////////////////////////////////////////

EvidentMirrorUnit1::EvidentMirrorUnit1() :
    initialized_(false),
    name_(g_MirrorUnit1DeviceName),
    numPos_(6)
{
    InitializeDefaultErrorMessages();
    SetErrorText(ERR_DEVICE_NOT_AVAILABLE, "Mirror unit 1 not available on this microscope");

    CreateHubIDProperty();
}

EvidentMirrorUnit1::~EvidentMirrorUnit1()
{
    Shutdown();
}

void EvidentMirrorUnit1::GetName(char* pszName) const
{
    CDeviceUtils::CopyLimitedString(pszName, name_.c_str());
}

int EvidentMirrorUnit1::Initialize()
{
    if (initialized_)
        return DEVICE_OK;

    EvidentHub* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    if (!hub->IsDevicePresent(DeviceType_MirrorUnit1))
        return ERR_DEVICE_NOT_AVAILABLE;

    numPos_ = hub->GetModel()->GetNumPositions(DeviceType_MirrorUnit1);

    // Create properties
    CPropertyAction* pAct = new CPropertyAction(this, &EvidentMirrorUnit1::OnState);
    int ret = CreateProperty(MM::g_Keyword_State, "0", MM::Integer, false, pAct);
    if (ret != DEVICE_OK)
        return ret;

    SetPropertyLimits(MM::g_Keyword_State, 0, numPos_ - 1);

    // Create label property
    pAct = new CPropertyAction(this, &CStateDeviceBase::OnLabel);
    ret = CreateProperty(MM::g_Keyword_Label, "", MM::String, false, pAct);
    if (ret != DEVICE_OK)
        return ret;

    // Define labels
    for (unsigned int i = 0; i < numPos_; i++)
    {
        std::ostringstream label;
        label << "Position-" << (i + 1);
        SetPositionLabel(i, label.str().c_str());
    }

    // Note: MirrorUnit uses NMUINIT1 which is an initialization notification,
    // not a position change notification, so we use query-based position tracking

    // Add firmware version as read-only property
    std::string version = hub->GetDeviceVersion(DeviceType_MirrorUnit1);
    if (!version.empty())
    {
        ret = CreateProperty("Firmware Version", version.c_str(), MM::String, true);
        if (ret != DEVICE_OK)
            return ret;
    }

    initialized_ = true;
    return DEVICE_OK;
}

int EvidentMirrorUnit1::Shutdown()
{
    if (initialized_)
    {
        initialized_ = false;
    }
    return DEVICE_OK;
}

bool EvidentMirrorUnit1::Busy()
{
    return false;  // Mirror unit changes are instantaneous
}

unsigned long EvidentMirrorUnit1::GetNumberOfPositions() const
{
    return numPos_;
}

int EvidentMirrorUnit1::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        EvidentHub* hub = GetHub();
        if (!hub)
            return DEVICE_ERR;

        // Query current position from hardware
        std::string cmd = BuildQuery(CMD_MIRROR_UNIT1);
        std::string response;
        int ret = hub->ExecuteCommand(cmd, response);
        if (ret != DEVICE_OK)
            return ret;

        std::vector<std::string> params = ParseParameters(response);
        if (params.size() > 0)
        {
            int pos = ParseIntParameter(params[0]);
            if (pos >= 1 && pos <= static_cast<int>(numPos_))
            {
                // Convert from 1-based to 0-based
                pProp->Set(static_cast<long>(pos - 1));
            }
        }
    }
    else if (eAct == MM::AfterSet)
    {
        long pos;
        pProp->Get(pos);

        EvidentHub* hub = GetHub();
        if (!hub)
            return DEVICE_ERR;

        // Convert from 0-based to 1-based for the microscope
        std::string cmd = BuildCommand(CMD_MIRROR_UNIT1, static_cast<int>(pos + 1));
        std::string response;
        int ret = hub->ExecuteCommand(cmd, response);
        if (ret != DEVICE_OK)
            return ret;

        if (!IsPositiveAck(response, CMD_MIRROR_UNIT1))
            return ERR_NEGATIVE_ACK;

        // Update MCU indicator I2 with new mirror position (1-based)
        hub->UpdateMirrorUnitIndicator(static_cast<int>(pos + 1));
    }
    return DEVICE_OK;
}

EvidentHub* EvidentMirrorUnit1::GetHub()
{
    MM::Hub* hub = GetParentHub();
    if (!hub)
        return nullptr;
    return dynamic_cast<EvidentHub*>(hub);
}


//int EvidentMirrorUnit1::EnableNotifications(bool /*enable*/)
//{
    // NMUINIT1 is an initialization notification, not a position change notification
    // MirrorUnit1 uses query-based position tracking instead
//    return DEVICE_OK;
//}

///////////////////////////////////////////////////////////////////////////////
// EvidentPolarizer - Polarizer Implementation
///////////////////////////////////////////////////////////////////////////////

EvidentPolarizer::EvidentPolarizer() :
    initialized_(false),
    name_(g_PolarizerDeviceName),
    numPos_(6)
{
    InitializeDefaultErrorMessages();
    SetErrorText(ERR_DEVICE_NOT_AVAILABLE, "Polarizer not available on this microscope");

    CreateHubIDProperty();
}

EvidentPolarizer::~EvidentPolarizer()
{
    Shutdown();
}

void EvidentPolarizer::GetName(char* pszName) const
{
    CDeviceUtils::CopyLimitedString(pszName, name_.c_str());
}

int EvidentPolarizer::Initialize()
{
    if (initialized_)
        return DEVICE_OK;

    EvidentHub* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    if (!hub->IsDevicePresent(DeviceType_Polarizer))
        return ERR_DEVICE_NOT_AVAILABLE;

    numPos_ = hub->GetModel()->GetNumPositions(DeviceType_Polarizer);

    // Create properties
    CPropertyAction* pAct = new CPropertyAction(this, &EvidentPolarizer::OnState);
    int ret = CreateProperty(MM::g_Keyword_State, "0", MM::Integer, false, pAct);
    if (ret != DEVICE_OK)
        return ret;

    SetPropertyLimits(MM::g_Keyword_State, 0, numPos_ - 1);

    // Create label property
    pAct = new CPropertyAction(this, &CStateDeviceBase::OnLabel);
    ret = CreateProperty(MM::g_Keyword_Label, "", MM::String, false, pAct);
    if (ret != DEVICE_OK)
        return ret;

    // Define labels - Polarizer has Out (0) and In (1)
    SetPositionLabel(0, "Out");
    SetPositionLabel(1, "In");

    // Add firmware version as read-only property
    std::string version = hub->GetDeviceVersion(DeviceType_Polarizer);
    if (!version.empty())
    {
        ret = CreateProperty("Firmware Version", version.c_str(), MM::String, true);
        if (ret != DEVICE_OK)
            return ret;
    }

    // Enable notifications
    ret = EnableNotifications(true);
    if (ret != DEVICE_OK)
        return ret;

    initialized_ = true;
    return DEVICE_OK;
}

int EvidentPolarizer::Shutdown()
{
    if (initialized_)
    {
        EnableNotifications(false);
        initialized_ = false;
    }
    return DEVICE_OK;
}

bool EvidentPolarizer::Busy()
{
    EvidentHub* hub = GetHub();
    if (!hub)
        return false;

    return hub->GetModel()->IsBusy(DeviceType_Polarizer);
}

unsigned long EvidentPolarizer::GetNumberOfPositions() const
{
    return numPos_;
}

int EvidentPolarizer::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        EvidentHub* hub = GetHub();
        if (!hub)
            return DEVICE_ERR;

        long pos = hub->GetModel()->GetPosition(DeviceType_Polarizer);
        if (pos < 0)
            return ERR_POSITION_UNKNOWN;

        // Polarizer uses 0-based indexing (PO 0 = Out, PO 1 = In), no conversion needed
        pProp->Set(pos);
    }
    else if (eAct == MM::AfterSet)
    {
        long pos;
        pProp->Get(pos);

        EvidentHub* hub = GetHub();
        if (!hub)
            return DEVICE_ERR;

        // Set target position BEFORE sending command
        // Polarizer uses 0-based indexing (PO 0 = Out, PO 1 = In)
        hub->GetModel()->SetTargetPosition(DeviceType_Polarizer, pos);
        hub->GetModel()->SetBusy(DeviceType_Polarizer, true);

        std::string cmd = BuildCommand(CMD_POLARIZER, static_cast<int>(pos));
        std::string response;
        int ret = hub->ExecuteCommand(cmd, response);
        if (ret != DEVICE_OK)
        {
            hub->GetModel()->SetBusy(DeviceType_Polarizer, false);
            return ret;
        }

        if (!IsPositiveAck(response, CMD_POLARIZER))
        {
            hub->GetModel()->SetBusy(DeviceType_Polarizer, false);
            return ERR_NEGATIVE_ACK;
        }

        // Polarizer does not send notifications (NPO) when movement completes.
        // The positive ack ("PO +") is only returned after movement completes,
        // so we can clear busy immediately and update position.
        hub->GetModel()->SetPosition(DeviceType_Polarizer, pos);
        hub->GetModel()->SetBusy(DeviceType_Polarizer, false);
    }
    return DEVICE_OK;
}

EvidentHub* EvidentPolarizer::GetHub()
{
    MM::Hub* hub = GetParentHub();
    if (!hub)
        return nullptr;
    return dynamic_cast<EvidentHub*>(hub);
}

int EvidentPolarizer::EnableNotifications(bool enable)
{
    EvidentHub* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    // Use PO (command tag) not NPO (notification tag) to enable notifications
    return hub->EnableNotification(CMD_POLARIZER, enable);
}

///////////////////////////////////////////////////////////////////////////////
// EvidentDICPrism - DIC Prism Implementation
///////////////////////////////////////////////////////////////////////////////

EvidentDICPrism::EvidentDICPrism() :
    initialized_(false),
    name_(g_DICPrismDeviceName),
    numPos_(6)
{
    InitializeDefaultErrorMessages();
    SetErrorText(ERR_DEVICE_NOT_AVAILABLE, "DIC prism not available on this microscope");

    CreateHubIDProperty();
}

EvidentDICPrism::~EvidentDICPrism()
{
    Shutdown();
}

void EvidentDICPrism::GetName(char* pszName) const
{
    CDeviceUtils::CopyLimitedString(pszName, name_.c_str());
}

int EvidentDICPrism::Initialize()
{
    if (initialized_)
        return DEVICE_OK;

    EvidentHub* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    if (!hub->IsDevicePresent(DeviceType_DICPrism))
        return ERR_DEVICE_NOT_AVAILABLE;

    numPos_ = hub->GetModel()->GetNumPositions(DeviceType_DICPrism);

    // Create properties
    CPropertyAction* pAct = new CPropertyAction(this, &EvidentDICPrism::OnState);
    int ret = CreateProperty(MM::g_Keyword_State, "0", MM::Integer, false, pAct);
    if (ret != DEVICE_OK)
        return ret;

    SetPropertyLimits(MM::g_Keyword_State, 0, numPos_ - 1);

    // Create label property
    pAct = new CPropertyAction(this, &CStateDeviceBase::OnLabel);
    ret = CreateProperty(MM::g_Keyword_Label, "", MM::String, false, pAct);
    if (ret != DEVICE_OK)
        return ret;

    // Define labels
    for (unsigned int i = 0; i < numPos_; i++)
    {
        std::ostringstream label;
        label << "Position-" << (i + 1);
        SetPositionLabel(i, label.str().c_str());
    }

    // Add firmware version as read-only property
    std::string version = hub->GetDeviceVersion(DeviceType_DICPrism);
    if (!version.empty())
    {
        ret = CreateProperty("Firmware Version", version.c_str(), MM::String, true);
        if (ret != DEVICE_OK)
            return ret;
    }

    initialized_ = true;
    return DEVICE_OK;
}

int EvidentDICPrism::Shutdown()
{
    if (initialized_)
    {
        initialized_ = false;
    }
    return DEVICE_OK;
}

bool EvidentDICPrism::Busy()
{
    return false;  // DIC prism changes are instantaneous
}

unsigned long EvidentDICPrism::GetNumberOfPositions() const
{
    return numPos_;
}

int EvidentDICPrism::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        EvidentHub* hub = GetHub();
        if (!hub)
            return DEVICE_ERR;

        // Query current position
        std::string cmd = BuildQuery(CMD_DIC_PRISM);
        std::string response;
        int ret = hub->ExecuteCommand(cmd, response);
        if (ret != DEVICE_OK)
            return ret;

        std::vector<std::string> params = ParseParameters(response);
        if (params.size() > 0)
        {
            int pos = ParseIntParameter(params[0]);
            if (pos >= 1 && pos <= static_cast<int>(numPos_))
            {
                // Convert from 1-based to 0-based
                pProp->Set(static_cast<long>(pos - 1));
            }
        }
    }
    else if (eAct == MM::AfterSet)
    {
        long pos;
        pProp->Get(pos);

        EvidentHub* hub = GetHub();
        if (!hub)
            return DEVICE_ERR;

        // Convert from 0-based to 1-based for the microscope
        std::string cmd = BuildCommand(CMD_DIC_PRISM, static_cast<int>(pos + 1));
        std::string response;
        int ret = hub->ExecuteCommand(cmd, response);
        if (ret != DEVICE_OK)
            return ret;

        if (!IsPositiveAck(response, CMD_DIC_PRISM))
            return ERR_NEGATIVE_ACK;
    }
    return DEVICE_OK;
}

EvidentHub* EvidentDICPrism::GetHub()
{
    MM::Hub* hub = GetParentHub();
    if (!hub)
        return nullptr;
    return dynamic_cast<EvidentHub*>(hub);
}

///////////////////////////////////////////////////////////////////////////////
// EvidentEPIND - EPI ND Filter Implementation
///////////////////////////////////////////////////////////////////////////////

EvidentEPIND::EvidentEPIND() :
    initialized_(false),
    name_(g_EPINDDeviceName),
    numPos_(6)
{
    InitializeDefaultErrorMessages();
    SetErrorText(ERR_DEVICE_NOT_AVAILABLE, "EPI ND filter not available on this microscope");

    CreateHubIDProperty();
}

EvidentEPIND::~EvidentEPIND()
{
    Shutdown();
}

void EvidentEPIND::GetName(char* pszName) const
{
    CDeviceUtils::CopyLimitedString(pszName, name_.c_str());
}

int EvidentEPIND::Initialize()
{
    if (initialized_)
        return DEVICE_OK;

    EvidentHub* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    if (!hub->IsDevicePresent(DeviceType_EPIND))
        return ERR_DEVICE_NOT_AVAILABLE;

    numPos_ = hub->GetModel()->GetNumPositions(DeviceType_EPIND);

    // Create properties
    CPropertyAction* pAct = new CPropertyAction(this, &EvidentEPIND::OnState);
    int ret = CreateProperty(MM::g_Keyword_State, "0", MM::Integer, false, pAct);
    if (ret != DEVICE_OK)
        return ret;

    SetPropertyLimits(MM::g_Keyword_State, 0, numPos_ - 1);

    // Create label property
    pAct = new CPropertyAction(this, &CStateDeviceBase::OnLabel);
    ret = CreateProperty(MM::g_Keyword_Label, "", MM::String, false, pAct);
    if (ret != DEVICE_OK)
        return ret;

    // Define labels
    for (unsigned int i = 0; i < numPos_; i++)
    {
        std::ostringstream label;
        label << "Position-" << (i + 1);
        SetPositionLabel(i, label.str().c_str());
    }

    // Add firmware version as read-only property
    std::string version = hub->GetDeviceVersion(DeviceType_EPIND);
    if (!version.empty())
    {
        ret = CreateProperty("Firmware Version", version.c_str(), MM::String, true);
        if (ret != DEVICE_OK)
            return ret;
    }

    initialized_ = true;
    return DEVICE_OK;
}

int EvidentEPIND::Shutdown()
{
    if (initialized_)
    {
        initialized_ = false;
    }
    return DEVICE_OK;
}

bool EvidentEPIND::Busy()
{
    return false;  // ND filter changes are instantaneous
}

unsigned long EvidentEPIND::GetNumberOfPositions() const
{
    return numPos_;
}

int EvidentEPIND::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        EvidentHub* hub = GetHub();
        if (!hub)
            return DEVICE_ERR;

        // Query current position
        std::string cmd = BuildQuery(CMD_EPI_ND);
        std::string response;
        int ret = hub->ExecuteCommand(cmd, response);
        if (ret != DEVICE_OK)
            return ret;

        std::vector<std::string> params = ParseParameters(response);
        if (params.size() > 0)
        {
            int pos = ParseIntParameter(params[0]);
            if (pos >= 1 && pos <= static_cast<int>(numPos_))
            {
                // Convert from 1-based to 0-based
                pProp->Set(static_cast<long>(pos - 1));
            }
        }
    }
    else if (eAct == MM::AfterSet)
    {
        long pos;
        pProp->Get(pos);

        EvidentHub* hub = GetHub();
        if (!hub)
            return DEVICE_ERR;

        // Convert from 0-based to 1-based for the microscope
        std::string cmd = BuildCommand(CMD_EPI_ND, static_cast<int>(pos + 1));
        std::string response;
        int ret = hub->ExecuteCommand(cmd, response);
        if (ret != DEVICE_OK)
            return ret;

        if (!IsPositiveAck(response, CMD_EPI_ND))
            return ERR_NEGATIVE_ACK;
    }
    return DEVICE_OK;
}

EvidentHub* EvidentEPIND::GetHub()
{
    MM::Hub* hub = GetParentHub();
    if (!hub)
        return nullptr;
    return dynamic_cast<EvidentHub*>(hub);
}

///////////////////////////////////////////////////////////////////////////////
// EvidentCorrectionCollar - Correction Collar Implementation
///////////////////////////////////////////////////////////////////////////////

EvidentCorrectionCollar::EvidentCorrectionCollar() :
    initialized_(false),
    name_(g_CorrectionCollarDeviceName)
{
    InitializeDefaultErrorMessages();
    SetErrorText(ERR_DEVICE_NOT_AVAILABLE, "Correction collar not available on this microscope");

    CreateHubIDProperty();
}

EvidentCorrectionCollar::~EvidentCorrectionCollar()
{
    Shutdown();
}

void EvidentCorrectionCollar::GetName(char* pszName) const
{
    CDeviceUtils::CopyLimitedString(pszName, name_.c_str());
}

int EvidentCorrectionCollar::Initialize()
{
    if (initialized_)
        return DEVICE_OK;

    EvidentHub* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    if (!hub->IsDevicePresent(DeviceType_CorrectionCollar))
        return ERR_DEVICE_NOT_AVAILABLE;

    // Create position property (0-100 range typically)
    CPropertyAction* pAct = new CPropertyAction(this, &EvidentCorrectionCollar::OnPosition);
    int ret = CreateProperty("Position", "0", MM::Integer, false, pAct);
    if (ret != DEVICE_OK)
        return ret;

    SetPropertyLimits("Position", 0, 100);

    initialized_ = true;
    return DEVICE_OK;
}

int EvidentCorrectionCollar::Shutdown()
{
    if (initialized_)
    {
        initialized_ = false;
    }
    return DEVICE_OK;
}

bool EvidentCorrectionCollar::Busy()
{
    return false;  // Correction collar changes are instantaneous
}

int EvidentCorrectionCollar::OnPosition(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        EvidentHub* hub = GetHub();
        if (!hub)
            return DEVICE_ERR;

        // Query current position
        std::string cmd = BuildQuery(CMD_CORRECTION_COLLAR);
        std::string response;
        int ret = hub->ExecuteCommand(cmd, response);
        if (ret != DEVICE_OK)
            return ret;

        std::vector<std::string> params = ParseParameters(response);
        if (params.size() > 0)
        {
            int pos = ParseIntParameter(params[0]);
            if (pos >= 0)
                pProp->Set(static_cast<long>(pos));
        }
    }
    else if (eAct == MM::AfterSet)
    {
        long pos;
        pProp->Get(pos);

        EvidentHub* hub = GetHub();
        if (!hub)
            return DEVICE_ERR;

        std::string cmd = BuildCommand(CMD_CORRECTION_COLLAR, static_cast<int>(pos));
        std::string response;
        int ret = hub->ExecuteCommand(cmd, response);
        if (ret != DEVICE_OK)
            return ret;

        if (!IsPositiveAck(response, CMD_CORRECTION_COLLAR))
            return ERR_NEGATIVE_ACK;
    }
    return DEVICE_OK;
}

EvidentHub* EvidentCorrectionCollar::GetHub()
{
    MM::Hub* hub = GetParentHub();
    if (!hub)
        return nullptr;
    return dynamic_cast<EvidentHub*>(hub);
}

///////////////////////////////////////////////////////////////////////////////
// EvidentEPIShutter2 - EPI (Reflected Light) Shutter 2 Implementation
///////////////////////////////////////////////////////////////////////////////

EvidentEPIShutter2::EvidentEPIShutter2() :
    initialized_(false),
    name_(g_EPIShutter2DeviceName)
{
    InitializeDefaultErrorMessages();
    SetErrorText(ERR_DEVICE_NOT_AVAILABLE, "EPI shutter 2 not available on this microscope");

    CreateHubIDProperty();
}

EvidentEPIShutter2::~EvidentEPIShutter2()
{
    Shutdown();
}

void EvidentEPIShutter2::GetName(char* pszName) const
{
    CDeviceUtils::CopyLimitedString(pszName, name_.c_str());
}

int EvidentEPIShutter2::Initialize()
{
    if (initialized_)
        return DEVICE_OK;

    EvidentHub* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    if (!hub->IsDevicePresent(DeviceType_EPIShutter2))
        return ERR_DEVICE_NOT_AVAILABLE;

    // Create state property
    CPropertyAction* pAct = new CPropertyAction(this, &EvidentEPIShutter2::OnState);
    int ret = CreateProperty(MM::g_Keyword_State, "0", MM::Integer, false, pAct);
    if (ret != DEVICE_OK)
        return ret;

    AddAllowedValue(MM::g_Keyword_State, "0");  // Closed
    AddAllowedValue(MM::g_Keyword_State, "1");  // Open

    // Add firmware version as read-only property
    std::string version = hub->GetDeviceVersion(DeviceType_EPIShutter2);
    if (!version.empty())
    {
        ret = CreateProperty("Firmware Version", version.c_str(), MM::String, true);
        if (ret != DEVICE_OK)
            return ret;
    }

    initialized_ = true;
    return DEVICE_OK;
}

int EvidentEPIShutter2::Shutdown()
{
    if (initialized_)
    {
        initialized_ = false;
    }
    return DEVICE_OK;
}

bool EvidentEPIShutter2::Busy()
{
    return false;  // Shutter changes are instantaneous
}

int EvidentEPIShutter2::SetOpen(bool open)
{
    EvidentHub* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    std::string cmd = BuildCommand(CMD_EPI_SHUTTER2, open ? 1 : 0);
    std::string response;
    int ret = hub->ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
        return ret;

    if (!IsPositiveAck(response, CMD_EPI_SHUTTER2))
        return ERR_NEGATIVE_ACK;

    return DEVICE_OK;
}

int EvidentEPIShutter2::GetOpen(bool& open)
{
    EvidentHub* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    std::string cmd = BuildQuery(CMD_EPI_SHUTTER2);
    std::string response;
    int ret = hub->ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
        return ret;

    std::vector<std::string> params = ParseParameters(response);
    if (params.size() > 0)
    {
        int state = ParseIntParameter(params[0]);
        open = (state == 1);
    }

    return DEVICE_OK;
}

int EvidentEPIShutter2::Fire(double /*deltaT*/)
{
    // Not implemented for this shutter
    return DEVICE_UNSUPPORTED_COMMAND;
}

int EvidentEPIShutter2::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        bool open;
        int ret = GetOpen(open);
        if (ret != DEVICE_OK)
            return ret;
        pProp->Set(open ? 1L : 0L);
    }
    else if (eAct == MM::AfterSet)
    {
        long state;
        pProp->Get(state);
        return SetOpen(state != 0);
    }
    return DEVICE_OK;
}

EvidentHub* EvidentEPIShutter2::GetHub()
{
    MM::Hub* hub = GetParentHub();
    if (!hub)
        return nullptr;
    return dynamic_cast<EvidentHub*>(hub);
}

///////////////////////////////////////////////////////////////////////////////
// EvidentMirrorUnit2 - Mirror Unit 2 (Filter Cube Turret) Implementation
///////////////////////////////////////////////////////////////////////////////

EvidentMirrorUnit2::EvidentMirrorUnit2() :
    initialized_(false),
    name_(g_MirrorUnit2DeviceName),
    numPos_(6)
{
    InitializeDefaultErrorMessages();
    SetErrorText(ERR_DEVICE_NOT_AVAILABLE, "Mirror unit 2 not available on this microscope");

    CreateHubIDProperty();
}

EvidentMirrorUnit2::~EvidentMirrorUnit2()
{
    Shutdown();
}

void EvidentMirrorUnit2::GetName(char* pszName) const
{
    CDeviceUtils::CopyLimitedString(pszName, name_.c_str());
}

int EvidentMirrorUnit2::Initialize()
{
    if (initialized_)
        return DEVICE_OK;

    EvidentHub* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    if (!hub->IsDevicePresent(DeviceType_MirrorUnit2))
        return ERR_DEVICE_NOT_AVAILABLE;

    numPos_ = hub->GetModel()->GetNumPositions(DeviceType_MirrorUnit2);

    // Create properties
    CPropertyAction* pAct = new CPropertyAction(this, &EvidentMirrorUnit2::OnState);
    int ret = CreateProperty(MM::g_Keyword_State, "0", MM::Integer, false, pAct);
    if (ret != DEVICE_OK)
        return ret;

    SetPropertyLimits(MM::g_Keyword_State, 0, numPos_ - 1);

    // Create label property
    pAct = new CPropertyAction(this, &CStateDeviceBase::OnLabel);
    ret = CreateProperty(MM::g_Keyword_Label, "", MM::String, false, pAct);
    if (ret != DEVICE_OK)
        return ret;

    // Define labels
    for (unsigned int i = 0; i < numPos_; i++)
    {
        std::ostringstream label;
        label << "Position-" << (i + 1);
        SetPositionLabel(i, label.str().c_str());
    }

    // Note: MirrorUnit uses NMUINIT2 which is an initialization notification,
    // not a position change notification, so we use query-based position tracking

    // Add firmware version as read-only property
    std::string version = hub->GetDeviceVersion(DeviceType_MirrorUnit2);
    if (!version.empty())
    {
        ret = CreateProperty("Firmware Version", version.c_str(), MM::String, true);
        if (ret != DEVICE_OK)
            return ret;
    }

    initialized_ = true;
    return DEVICE_OK;
}

int EvidentMirrorUnit2::Shutdown()
{
    if (initialized_)
    {
        initialized_ = false;
    }
    return DEVICE_OK;
}

bool EvidentMirrorUnit2::Busy()
{
    return false;  // Mirror unit changes are instantaneous
}

unsigned long EvidentMirrorUnit2::GetNumberOfPositions() const
{
    return numPos_;
}

int EvidentMirrorUnit2::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        EvidentHub* hub = GetHub();
        if (!hub)
            return DEVICE_ERR;

        // Query current position from hardware
        std::string cmd = BuildQuery(CMD_MIRROR_UNIT2);
        std::string response;
        int ret = hub->ExecuteCommand(cmd, response);
        if (ret != DEVICE_OK)
            return ret;

        std::vector<std::string> params = ParseParameters(response);
        if (params.size() > 0)
        {
            int pos = ParseIntParameter(params[0]);
            if (pos >= 1 && pos <= static_cast<int>(numPos_))
            {
                // Convert from 1-based to 0-based
                pProp->Set(static_cast<long>(pos - 1));
            }
        }
    }
    else if (eAct == MM::AfterSet)
    {
        long pos;
        pProp->Get(pos);

        EvidentHub* hub = GetHub();
        if (!hub)
            return DEVICE_ERR;

        // Convert from 0-based to 1-based for the microscope
        std::string cmd = BuildCommand(CMD_MIRROR_UNIT2, static_cast<int>(pos + 1));
        std::string response;
        int ret = hub->ExecuteCommand(cmd, response);
        if (ret != DEVICE_OK)
            return ret;

        if (!IsPositiveAck(response, CMD_MIRROR_UNIT2))
            return ERR_NEGATIVE_ACK;
    }
    return DEVICE_OK;
}

EvidentHub* EvidentMirrorUnit2::GetHub()
{
    MM::Hub* hub = GetParentHub();
    if (!hub)
        return nullptr;
    return dynamic_cast<EvidentHub*>(hub);
}

int EvidentMirrorUnit2::EnableNotifications(bool /*enable*/)
{
    // NMUINIT2 is an initialization notification, not a position change notification
    // MirrorUnit2 uses query-based position tracking instead
    return DEVICE_OK;
}
