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

#include "EvidentIX85Win.h"
#include "EvidentObjectiveSetup.h"
#include "ModuleInterface.h"
#include <sstream>
#include <iomanip>

using namespace EvidentIX85Win;

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
const char* g_AutofocusDeviceName = "IX85-Autofocus";
const char* g_OffsetLensDeviceName = "IX85-OffsetLens";
const char* g_ZDCVirtualOffsetDeviceName = "IX85-ZDCVirtualOffset";
const char* g_ObjectiveSetupDeviceName = "IX85-ObjectiveSetup";

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
    RegisterDevice(g_CorrectionCollarDeviceName, MM::StageDevice, "Evident IX85 Correction Collar");
    RegisterDevice(g_AutofocusDeviceName, MM::AutoFocusDevice, "Evident IX85 ZDC Autofocus");
    RegisterDevice(g_OffsetLensDeviceName, MM::StageDevice, "Evident IX85 Offset Lens");
    RegisterDevice(g_ObjectiveSetupDeviceName, MM::GenericDevice, "Evident IX85 Objective Setup");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
    if (deviceName == nullptr)
        return nullptr;

    if (strcmp(deviceName, g_HubDeviceName) == 0)
        return new EvidentHubWin();
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
    else if (strcmp(deviceName, g_AutofocusDeviceName) == 0)
        return new EvidentAutofocus();
    else if (strcmp(deviceName, g_OffsetLensDeviceName) == 0)
        return new EvidentOffsetLens();
    else if (strcmp(deviceName, g_ZDCVirtualOffsetDeviceName) == 0)
        return new EvidentZDCVirtualOffset();
    else if (strcmp(deviceName, g_ObjectiveSetupDeviceName) == 0)
        return new EvidentObjectiveSetup();

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
    SetErrorText(ERR_POSITION_OUT_OF_RANGE, "Requested focus position is out of range");

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

    EvidentHubWin* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    if (!hub->IsDevicePresent(DeviceType_Focus))
        return ERR_DEVICE_NOT_AVAILABLE;

    // Create propertiesn
    CPropertyAction* pAct = new CPropertyAction(this, &EvidentFocus::OnPosition);
    int ret = CreateProperty(MM::g_Keyword_Position, "0.0", MM::Float, false, pAct);
    if (ret != DEVICE_OK)
        return ret;

    pAct = new CPropertyAction(this, &EvidentFocus::OnSpeed);
    ret = CreateProperty("Speed (um/s)", "30.0", MM::Float, false, pAct);
    if (ret != DEVICE_OK)
        return ret;

    // Enable active notifications
    ret = EnableNotifications(true);
    if (ret != DEVICE_OK)
        return ret;

    // Register with hub so notification handler can call OnStagePositionChanged
    hub->RegisterDeviceAsUsed(DeviceType_Focus, this);

    initialized_ = true;
    return DEVICE_OK;
}

int EvidentFocus::Shutdown()
{
    if (initialized_)
    {
        EnableNotifications(false);

        // Unregister from hub
        EvidentHubWin* hub = GetHub();
        if (hub)
            hub->UnRegisterDeviceAsUsed(DeviceType_Focus);

        initialized_ = false;
    }
    return DEVICE_OK;
}

bool EvidentFocus::Busy()
{
    EvidentHubWin* hub = GetHub();
    if (!hub)
        return false;

    return hub->GetModel()->IsBusy(DeviceType_Focus);
}

int EvidentFocus::SetPositionUm(double pos)
{
    EvidentHubWin* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    // Convert μm to 10nm units
    long steps = static_cast<long>(pos / stepSizeUm_);

    // Clamp to limits
    if (steps < FOCUS_MIN_POS) steps = FOCUS_MIN_POS;
    if (steps > FOCUS_MAX_POS) steps = FOCUS_MAX_POS;

    return hub->SetFocusPositionSteps(steps);
}

int EvidentFocus::GetPositionUm(double& pos)
{
    EvidentHubWin* hub = GetHub();
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
    EvidentHubWin* hub = GetHub();
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
        EvidentHubWin* hub = GetHub();
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

        EvidentHubWin* hub = GetHub();
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

EvidentHubWin* EvidentFocus::GetHub()
{
    MM::Hub* hub = GetParentHub();
    if (!hub)
        return nullptr;
    return dynamic_cast<EvidentHubWin*>(hub);
}

int EvidentFocus::EnableNotifications(bool enable)
{
    EvidentHubWin* hub = GetHub();
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
    nearLimits_(NOSEPIECE_MAX_POS, FOCUS_MAX_POS),  // Initialize with max values
    parfocalPositions_(NOSEPIECE_MAX_POS, 0),  // Initialize with zeros
    parfocalEnabled_(false),
    escapeDistance_(3)  // Default to 3.0 mm 
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

    EvidentHubWin* hub = GetHub();
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

    // Define labels using objective names from hub
    const std::vector<EvidentIX85Win::ObjectiveInfo>& objectives = hub->GetObjectiveInfo();
    for (unsigned int i = 0; i < numPos_; i++)
    {
        if (i < objectives.size() && !objectives[i].name.empty())
        {
            SetPositionLabel(i, objectives[i].name.c_str());
        }
        else
        {
            std::ostringstream label;
            label << "Position-" << (i + 1);
            SetPositionLabel(i, label.str().c_str());
        }
    }

    // Create read-only properties for current objective info
    pAct = new CPropertyAction(this, &EvidentNosepiece::OnObjectiveNA);
    ret = CreateProperty("Objective-NA", "", MM::String, true, pAct);
    if (ret != DEVICE_OK)
        return ret;

    pAct = new CPropertyAction(this, &EvidentNosepiece::OnObjectiveMagnification);
    ret = CreateProperty("Objective-Magnification", "", MM::String, true, pAct);
    if (ret != DEVICE_OK)
        return ret;

    pAct = new CPropertyAction(this, &EvidentNosepiece::OnObjectiveMedium);
    ret = CreateProperty("Objective-Medium", "", MM::String, true, pAct);
    if (ret != DEVICE_OK)
        return ret;

    pAct = new CPropertyAction(this, &EvidentNosepiece::OnObjectiveWD);
    ret = CreateProperty("Objective-WD", "", MM::String, true, pAct);
    if (ret != DEVICE_OK)
        return ret;

    // Add firmware version as read-only property
    std::string version = hub->GetDeviceVersion(DeviceType_Nosepiece);
    if (!version.empty())
    {
        ret = CreateProperty("Firmware Version", version.c_str(), MM::String, true);
        if (ret != DEVICE_OK)
            return ret;
    }

    // Query focus near limits from microscope
    ret = QueryNearLimits();
    if (ret != DEVICE_OK)
    {
        LogMessage("Warning: Failed to query near limits, using defaults");
        // Don't fail initialization, just use default values
    }

    // Create Focus-Near-Limit-um read-only property
    pAct = new CPropertyAction(this, &EvidentNosepiece::OnNearLimit);
    ret = CreateProperty("Focus-Near-Limit-um", "0.0", MM::Float, true, pAct);
    if (ret != DEVICE_OK)
        return ret;

    // Create Set-Focus-Near-Limit action property
    pAct = new CPropertyAction(this, &EvidentNosepiece::OnSetNearLimit);
    ret = CreateProperty("Set-Focus-Near-Limit", "", MM::String, false, pAct);
    if (ret != DEVICE_OK)
        return ret;
    AddAllowedValue("Set-Focus-Near-Limit", "");
    AddAllowedValue("Set-Focus-Near-Limit", "Set");
    AddAllowedValue("Set-Focus-Near-Limit", "Clear");

    // Query parfocal settings from microscope
    ret = QueryParfocalSettings();
    if (ret != DEVICE_OK)
    {
        LogMessage("Warning: Failed to query parfocal settings, using defaults");
        // Don't fail initialization, just use default values
    }

    // Create Parfocal-Position-um read-only property
    pAct = new CPropertyAction(this, &EvidentNosepiece::OnParfocalPosition);
    ret = CreateProperty("Parfocal-Position-um", "0.0", MM::Float, true, pAct);
    if (ret != DEVICE_OK)
        return ret;

    // Create Set-Parfocal-Position action property
    pAct = new CPropertyAction(this, &EvidentNosepiece::OnSetParfocalPosition);
    ret = CreateProperty("Set-Parfocal-Position", "", MM::String, false, pAct);
    if (ret != DEVICE_OK)
        return ret;
    AddAllowedValue("Set-Parfocal-Position", "");
    AddAllowedValue("Set-Parfocal-Position", "Set");
    AddAllowedValue("Set-Parfocal-Position", "Clear");

    // Create Parfocal-Enabled property
    pAct = new CPropertyAction(this, &EvidentNosepiece::OnParfocalEnabled);
    ret = CreateProperty("Parfocal-Enabled", "Disabled", MM::String, false, pAct);
    if (ret != DEVICE_OK)
        return ret;
    AddAllowedValue("Parfocal-Enabled", "Disabled");
    AddAllowedValue("Parfocal-Enabled", "Enabled");

    // Query focus escape distance
    std::string escCmd = BuildQuery(CMD_FOCUS_ESCAPE);
    std::string escResponse;
    ret = hub->ExecuteCommand(escCmd, escResponse);
    if (ret == DEVICE_OK)
    {
        std::vector<std::string> escParams = ParseParameters(escResponse);
        if (escParams.size() > 0)
        {
            escapeDistance_ = ParseIntParameter(escParams[0]);
        }
    }

    // Create Focus-Escape-Distance property
    pAct = new CPropertyAction(this, &EvidentNosepiece::OnEscapeDistance);
    ret = CreateProperty("Focus-Escape-Distance", "3.0 mm", MM::String, false, pAct);
    if (ret != DEVICE_OK)
        return ret;
    AddAllowedValue("Focus-Escape-Distance", "0.0 mm");
    AddAllowedValue("Focus-Escape-Distance", "1.0 mm");
    AddAllowedValue("Focus-Escape-Distance", "2.0 mm");
    AddAllowedValue("Focus-Escape-Distance", "3.0 mm");
    AddAllowedValue("Focus-Escape-Distance", "4.0 mm");
    AddAllowedValue("Focus-Escape-Distance", "5.0 mm");
    AddAllowedValue("Focus-Escape-Distance", "6.0 mm");
    AddAllowedValue("Focus-Escape-Distance", "7.0 mm");
    AddAllowedValue("Focus-Escape-Distance", "8.0 mm");
    AddAllowedValue("Focus-Escape-Distance", "9.0 mm");

    // Enable notifications
    ret = EnableNotifications(true);
    if (ret != DEVICE_OK)
        return ret;

    // Register with hub so notification handler can notify property changes
    hub->RegisterDeviceAsUsed(DeviceType_Nosepiece, this);

    initialized_ = true;
    return DEVICE_OK;
}

int EvidentNosepiece::Shutdown()
{
    if (initialized_)
    {
        EnableNotifications(false);

        // Unregister from hub
        EvidentHubWin* hub = GetHub();
        if (hub)
            hub->UnRegisterDeviceAsUsed(DeviceType_Nosepiece);

        initialized_ = false;
    }
    return DEVICE_OK;
}

bool EvidentNosepiece::Busy()
{
    EvidentHubWin* hub = GetHub();
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
        EvidentHubWin* hub = GetHub();
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

        EvidentHubWin* hub = GetHub();
        if (!hub)
            return DEVICE_ERR;

        // Convert from 0-based to 1-based for the microscope
        long targetPos = pos + 1;

        // Set target position BEFORE sending command so notifications can check against it
        hub->GetModel()->SetTargetPosition(DeviceType_Nosepiece, targetPos);

        // Check if already at target position
        long currentPos = hub->GetModel()->GetPosition(DeviceType_Nosepiece);
        if (currentPos == targetPos)
        {
            // Already at target, no need to move
            hub->GetModel()->SetBusy(DeviceType_Nosepiece, false);
            return DEVICE_OK;
        }

        hub->GetModel()->SetBusy(DeviceType_Nosepiece, true);

        // Use OBSEQ command - SDK handles focus escape and parfocality automatically
        std::string cmd = BuildCommand(CMD_NOSEPIECE_SEQ, static_cast<int>(targetPos));
        std::string response;
        int ret = hub->ExecuteCommand(cmd, response);
        if (ret != DEVICE_OK)
        {
            // Command failed - clear busy state
            hub->GetModel()->SetBusy(DeviceType_Nosepiece, false);
            return ret;
        }

        if (!IsPositiveAck(response, CMD_NOSEPIECE_SEQ))
        {
            // Command rejected - clear busy state
            hub->GetModel()->SetBusy(DeviceType_Nosepiece, false);
            return ERR_NEGATIVE_ACK;
        }

        // Command accepted - busy state already set, will be cleared by notification when target reached
    }
    return DEVICE_OK;
}

EvidentHubWin* EvidentNosepiece::GetHub()
{
    MM::Hub* hub = GetParentHub();
    if (!hub)
        return nullptr;
    return dynamic_cast<EvidentHubWin*>(hub);
}

int EvidentNosepiece::EnableNotifications(bool enable)
{
    EvidentHubWin* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    return hub->EnableNotification(CMD_NOSEPIECE_NOTIFY, enable);
}

int EvidentNosepiece::OnObjectiveNA(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        EvidentHubWin* hub = GetHub();
        if (hub)
        {
            long pos = hub->GetModel()->GetPosition(DeviceType_Nosepiece);
            const std::vector<EvidentIX85Win::ObjectiveInfo>& objectives = hub->GetObjectiveInfo();
            if (pos >= 1 && pos <= (long)objectives.size())
            {
                double na = objectives[pos - 1].na;
                if (na >= 0)
                {
                    std::ostringstream ss;
                    ss << std::fixed << std::setprecision(2) << na;
                    pProp->Set(ss.str().c_str());
                }
                else
                {
                    pProp->Set("N/A");
                }
            }
        }
    }
    return DEVICE_OK;
}

int EvidentNosepiece::OnObjectiveMagnification(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        EvidentHubWin* hub = GetHub();
        if (hub)
        {
            long pos = hub->GetModel()->GetPosition(DeviceType_Nosepiece);
            const std::vector<EvidentIX85Win::ObjectiveInfo>& objectives = hub->GetObjectiveInfo();
            if (pos >= 1 && pos <= (long)objectives.size())
            {
                int mag = objectives[pos - 1].magnification;
                if (mag >= 0)
                {
                    pProp->Set((long)mag);
                }
                else
                {
                    pProp->Set("N/A");
                }
            }
        }
    }
    return DEVICE_OK;
}

int EvidentNosepiece::OnObjectiveMedium(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        EvidentHubWin* hub = GetHub();
        if (hub)
        {
            long pos = hub->GetModel()->GetPosition(DeviceType_Nosepiece);
            const std::vector<EvidentIX85Win::ObjectiveInfo>& objectives = hub->GetObjectiveInfo();
            if (pos >= 1 && pos <= (long)objectives.size())
            {
                int medium = objectives[pos - 1].medium;
                const char* mediumStr = "N/A";
                switch (medium)
                {
                    case 1: mediumStr = "Dry"; break;
                    case 2: mediumStr = "Water"; break;
                    case 3: mediumStr = "Oil"; break;
                    case 4: mediumStr = "Silicone Oil"; break;
                    case 5: mediumStr = "Silicone Gel"; break;
                }
                pProp->Set(mediumStr);
            }
        }
    }
    return DEVICE_OK;
}

int EvidentNosepiece::OnObjectiveWD(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        EvidentHubWin* hub = GetHub();
        if (hub)
        {
            long pos = hub->GetModel()->GetPosition(DeviceType_Nosepiece);
            const std::vector<EvidentIX85Win::ObjectiveInfo>& objectives = hub->GetObjectiveInfo();
            if (pos >= 1 && pos <= (long)objectives.size())
            {
                double wd = objectives[pos - 1].wd;
                if (wd >= 0)
                {
                    std::ostringstream ss;
                    ss << std::fixed << std::setprecision(2) << wd << " mm";
                    pProp->Set(ss.str().c_str());
                }
                else
                {
                    pProp->Set("N/A");
                }
            }
        }
    }
    return DEVICE_OK;
}

int EvidentNosepiece::OnNearLimit(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        EvidentHubWin* hub = GetHub();
        if (hub)
        {
            long pos = hub->GetModel()->GetPosition(DeviceType_Nosepiece);
            if (pos >= 1 && pos <= (long)nearLimits_.size())
            {
                // Get near limit for current objective position (convert 1-based to 0-based)
                long nearLimitSteps = nearLimits_[pos - 1];
                // Convert steps to micrometers
                double nearLimitUm = nearLimitSteps * FOCUS_STEP_SIZE_UM;
                pProp->Set(nearLimitUm);
            }
            else
            {
                pProp->Set(0.0);
            }
        }
    }
    return DEVICE_OK;
}

int EvidentNosepiece::OnSetNearLimit(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::AfterSet)
    {
        std::string value;
        pProp->Get(value);

        if (value == "Set")
        {
            EvidentHubWin* hub = GetHub();
            if (!hub)
                return DEVICE_ERR;

            // Get current nosepiece position
            long nosepiecePos = hub->GetModel()->GetPosition(DeviceType_Nosepiece);
            if (nosepiecePos < 1 || nosepiecePos > (long)nearLimits_.size())
                return ERR_INVALID_PARAMETER;

            // Check if focus device is available
            if (!hub->IsDevicePresent(DeviceType_Focus))
            {
                LogMessage("Focus device not available, cannot set near limit");
                pProp->Set("");  // Reset to empty
                return ERR_DEVICE_NOT_AVAILABLE;
            }

            // Get current focus position
            long focusPos = hub->GetModel()->GetPosition(DeviceType_Focus);
            if (focusPos < 0)
            {
                LogMessage("Focus position unknown, cannot set near limit");
                pProp->Set("");  // Reset to empty
                return ERR_POSITION_UNKNOWN;
            }

            // Update near limit for current objective (convert 1-based to 0-based)
            nearLimits_[nosepiecePos - 1] = focusPos;

            // Build NL command with all 6 positions: "NL p1,p2,p3,p4,p5,p6"
            std::ostringstream cmd;
            cmd << CMD_FOCUS_NEAR_LIMIT << TAG_DELIMITER;
            for (size_t i = 0; i < nearLimits_.size(); i++)
            {
                if (i > 0)
                    cmd << DATA_DELIMITER;
                cmd << nearLimits_[i];
            }

            // Execute command
            std::string response;
            int ret = hub->ExecuteCommand(cmd.str(), response);
            if (ret != DEVICE_OK)
            {
                pProp->Set("");  // Reset to empty
                return ret;
            }

            if (!IsPositiveAck(response, CMD_FOCUS_NEAR_LIMIT))
            {
                pProp->Set("");  // Reset to empty
                return ERR_NEGATIVE_ACK;
            }

            // Log success
            std::ostringstream logMsg;
            logMsg << "Set near limit for objective " << nosepiecePos
                   << " to " << (focusPos * FOCUS_STEP_SIZE_UM) << " um";
            LogMessage(logMsg.str().c_str());

            // Reset property to empty
            pProp->Set("");
        }
        else if (value == "Clear")
        {
            EvidentHubWin* hub = GetHub();
            if (!hub)
                return DEVICE_ERR;

            // Get current nosepiece position
            long nosepiecePos = hub->GetModel()->GetPosition(DeviceType_Nosepiece);
            if (nosepiecePos < 1 || nosepiecePos > (long)nearLimits_.size())
                return ERR_INVALID_PARAMETER;

            // Set near limit to maximum (effectively removing the limit)
            nearLimits_[nosepiecePos - 1] = FOCUS_MAX_POS;

            // Build NL command with all 6 positions: "NL p1,p2,p3,p4,p5,p6"
            std::ostringstream cmd;
            cmd << CMD_FOCUS_NEAR_LIMIT << TAG_DELIMITER;
            for (size_t i = 0; i < nearLimits_.size(); i++)
            {
                if (i > 0)
                    cmd << DATA_DELIMITER;
                cmd << nearLimits_[i];
            }

            // Execute command
            std::string response;
            int ret = hub->ExecuteCommand(cmd.str(), response);
            if (ret != DEVICE_OK)
            {
                pProp->Set("");  // Reset to empty
                return ret;
            }

            if (!IsPositiveAck(response, CMD_FOCUS_NEAR_LIMIT))
            {
                pProp->Set("");  // Reset to empty
                return ERR_NEGATIVE_ACK;
            }

            // Log success
            std::ostringstream logMsg;
            logMsg << "Cleared near limit for objective " << nosepiecePos
                   << " (set to maximum: " << (FOCUS_MAX_POS * FOCUS_STEP_SIZE_UM) << " um)";
            LogMessage(logMsg.str().c_str());

            // Reset property to empty
            pProp->Set("");
        }
    }
    return DEVICE_OK;
}

int EvidentNosepiece::OnParfocalPosition(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        EvidentHubWin* hub = GetHub();
        if (hub)
        {
            long pos = hub->GetModel()->GetPosition(DeviceType_Nosepiece);
            if (pos >= 1 && pos <= (long)parfocalPositions_.size())
            {
                // Get parfocal position for current objective (convert 1-based to 0-based)
                long parfocalSteps = parfocalPositions_[pos - 1];
                // Convert steps to micrometers
                double parfocalUm = parfocalSteps * FOCUS_STEP_SIZE_UM;
                pProp->Set(parfocalUm);
            }
            else
            {
                pProp->Set(0.0);
            }
        }
    }
    return DEVICE_OK;
}

int EvidentNosepiece::OnSetParfocalPosition(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::AfterSet)
    {
        std::string value;
        pProp->Get(value);

        if (value == "Set")
        {
            EvidentHubWin* hub = GetHub();
            if (!hub)
                return DEVICE_ERR;

            // Get current nosepiece position
            long nosepiecePos = hub->GetModel()->GetPosition(DeviceType_Nosepiece);
            if (nosepiecePos < 1 || nosepiecePos > (long)parfocalPositions_.size())
                return ERR_INVALID_PARAMETER;

            // Check if focus device is available
            if (!hub->IsDevicePresent(DeviceType_Focus))
            {
                LogMessage("Focus device not available, cannot set parfocal position");
                pProp->Set("");  // Reset to empty
                return ERR_DEVICE_NOT_AVAILABLE;
            }

            // Get current focus position
            long focusPos = hub->GetModel()->GetPosition(DeviceType_Focus);
            if (focusPos < 0)
            {
                LogMessage("Focus position unknown, cannot set parfocal position");
                pProp->Set("");  // Reset to empty
                return ERR_POSITION_UNKNOWN;
            }

            // Update parfocal position for current objective (convert 1-based to 0-based)
            parfocalPositions_[nosepiecePos - 1] = focusPos;

            // Build PF command with all 6 positions: "PF p1,p2,p3,p4,p5,p6"
            std::ostringstream cmd;
            cmd << CMD_PARFOCAL << TAG_DELIMITER;
            for (size_t i = 0; i < parfocalPositions_.size(); i++)
            {
                if (i > 0)
                    cmd << DATA_DELIMITER;
                cmd << parfocalPositions_[i];
            }

            // Execute command
            std::string response;
            int ret = hub->ExecuteCommand(cmd.str(), response);
            if (ret != DEVICE_OK)
            {
                pProp->Set("");  // Reset to empty
                return ret;
            }

            if (!IsPositiveAck(response, CMD_PARFOCAL))
            {
                pProp->Set("");  // Reset to empty
                return ERR_NEGATIVE_ACK;
            }

            // Log success
            std::ostringstream logMsg;
            logMsg << "Set parfocal position for objective " << nosepiecePos
                   << " to " << (focusPos * FOCUS_STEP_SIZE_UM) << " um";
            LogMessage(logMsg.str().c_str());

            // Reset property to empty
            pProp->Set("");
        }
        else if (value == "Clear")
        {
            EvidentHubWin* hub = GetHub();
            if (!hub)
                return DEVICE_ERR;

            // Get current nosepiece position
            long nosepiecePos = hub->GetModel()->GetPosition(DeviceType_Nosepiece);
            if (nosepiecePos < 1 || nosepiecePos > (long)parfocalPositions_.size())
                return ERR_INVALID_PARAMETER;

            // Set parfocal position to zero (no offset)
            parfocalPositions_[nosepiecePos - 1] = 0;

            // Build PF command with all 6 positions: "PF p1,p2,p3,p4,p5,p6"
            std::ostringstream cmd;
            cmd << CMD_PARFOCAL << TAG_DELIMITER;
            for (size_t i = 0; i < parfocalPositions_.size(); i++)
            {
                if (i > 0)
                    cmd << DATA_DELIMITER;
                cmd << parfocalPositions_[i];
            }

            // Execute command
            std::string response;
            int ret = hub->ExecuteCommand(cmd.str(), response);
            if (ret != DEVICE_OK)
            {
                pProp->Set("");  // Reset to empty
                return ret;
            }

            if (!IsPositiveAck(response, CMD_PARFOCAL))
            {
                pProp->Set("");  // Reset to empty
                return ERR_NEGATIVE_ACK;
            }

            // Log success
            std::ostringstream logMsg;
            logMsg << "Cleared parfocal position for objective " << nosepiecePos;
            LogMessage(logMsg.str().c_str());

            // Reset property to empty
            pProp->Set("");
        }
    }
    return DEVICE_OK;
}

int EvidentNosepiece::OnParfocalEnabled(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    EvidentHubWin* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    if (eAct == MM::BeforeGet)
    {
        pProp->Set(parfocalEnabled_ ? "Enabled" : "Disabled");
    }
    else if (eAct == MM::AfterSet)
    {
        std::string value;
        pProp->Get(value);
        int enabled = (value == "Enabled") ? 1 : 0;

        std::string cmd = BuildCommand(CMD_ENABLE_PARFOCAL, enabled);
        std::string response;
        int ret = hub->ExecuteCommand(cmd, response);
        if (ret != DEVICE_OK)
            return ret;

        if (!IsPositiveAck(response, CMD_ENABLE_PARFOCAL))
            return ERR_NEGATIVE_ACK;

        // Update internal state
        parfocalEnabled_ = (enabled == 1);

        // Log success
        LogMessage(parfocalEnabled_ ? "Parfocal enabled" : "Parfocal disabled");
    }
    return DEVICE_OK;
}

int EvidentNosepiece::OnEscapeDistance(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    EvidentHubWin* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    if (eAct == MM::BeforeGet)
    {
        // Convert escapeDistance_ (0-9) to display string "X.0 mm"
        std::ostringstream ss;
        ss << escapeDistance_ << ".0 mm";
        pProp->Set(ss.str().c_str());
    }
    else if (eAct == MM::AfterSet)
    {
        std::string value;
        pProp->Get(value);

        // Parse "X.0 mm" to extract the integer (0-9)
        int newDistance = 0;
        if (value.length() >= 1 && value[0] >= '0' && value[0] <= '9')
        {
            newDistance = value[0] - '0';
        }

        // Send ESC2 command with new value
        std::string cmd = BuildCommand(CMD_FOCUS_ESCAPE, newDistance);
        std::string response;
        int ret = hub->ExecuteCommand(cmd, response);
        if (ret != DEVICE_OK)
            return ret;

        if (!IsPositiveAck(response, CMD_FOCUS_ESCAPE))
            return ERR_NEGATIVE_ACK;

        // Update internal state
        escapeDistance_ = newDistance;

        // Log success
        std::ostringstream logMsg;
        logMsg << "Focus escape distance set to " << newDistance << ".0 mm";
        LogMessage(logMsg.str().c_str());
    }
    return DEVICE_OK;
}

int EvidentNosepiece::QueryNearLimits()
{
    EvidentHubWin* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    // Query near limits from microscope
    std::string cmd = BuildQuery(CMD_FOCUS_NEAR_LIMIT);
    std::string response;
    int ret = hub->ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
        return ret;

    // Parse response: "NL p1,p2,p3,p4,p5,p6"
    std::vector<std::string> params = ParseParameters(response);
    if (params.size() >= NOSEPIECE_MAX_POS)
    {
        for (size_t i = 0; i < NOSEPIECE_MAX_POS; i++)
        {
            nearLimits_[i] = ParseLongParameter(params[i]);
        }
    }
    else
    {
        LogMessage("Warning: NL? response has fewer than expected parameters");
        return ERR_INVALID_RESPONSE;
    }

    return DEVICE_OK;
}

int EvidentNosepiece::QueryParfocalSettings()
{
    EvidentHubWin* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    // Query parfocal positions from microscope
    std::string cmd = BuildQuery(CMD_PARFOCAL);
    std::string response;
    int ret = hub->ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
        return ret;

    // Parse response: "PF p1,p2,p3,p4,p5,p6"
    std::vector<std::string> params = ParseParameters(response);
    if (params.size() >= NOSEPIECE_MAX_POS)
    {
        for (size_t i = 0; i < NOSEPIECE_MAX_POS; i++)
        {
            parfocalPositions_[i] = ParseLongParameter(params[i]);
        }
    }
    else
    {
        LogMessage("Warning: PF? response has fewer than expected parameters");
        return ERR_INVALID_RESPONSE;
    }

    // Query parfocal enabled state
    cmd = BuildQuery(CMD_ENABLE_PARFOCAL);
    ret = hub->ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
        return ret;

    // Parse response: "ENPF 0" or "ENPF 1"
    params = ParseParameters(response);
    if (params.size() > 0)
    {
        int enabled = ParseIntParameter(params[0]);
        parfocalEnabled_ = (enabled == 1);
    }
    else
    {
        LogMessage("Warning: ENPF? response has no parameters");
        return ERR_INVALID_RESPONSE;
    }

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

    EvidentHubWin* hub = GetHub();
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
    EvidentHubWin* hub = GetHub();
    if (!hub)
        return false;

    return hub->GetModel()->IsBusy(DeviceType_Magnification);
}

double EvidentMagnification::GetMagnification()
{
    EvidentHubWin* hub = GetHub();
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

EvidentHubWin* EvidentMagnification::GetHub()
{
    MM::Hub* hub = GetParentHub();
    if (!hub)
        return nullptr;
    return dynamic_cast<EvidentHubWin*>(hub);
}

int EvidentMagnification::EnableNotifications(bool enable)
{
    EvidentHubWin* hub = GetHub();
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

    EvidentHubWin* hub = GetHub();
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

    // Register with hub so OnState can notify indicator changes
    hub->RegisterDeviceAsUsed(DeviceType_LightPath, this);

    initialized_ = true;
    return DEVICE_OK;
}

int EvidentLightPath::Shutdown()
{
    if (initialized_)
    {
        // Unregister from hub
        EvidentHubWin* hub = GetHub();
        if (hub)
            hub->UnRegisterDeviceAsUsed(DeviceType_LightPath);

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
        EvidentHubWin* hub = GetHub();
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

        EvidentHubWin* hub = GetHub();
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

        // Update MCU indicator I4 with new light path position (1-based)
        hub->UpdateLightPathIndicator(static_cast<int>(pos + 1));
    }
    return DEVICE_OK;
}

EvidentHubWin* EvidentLightPath::GetHub()
{
    MM::Hub* hub = GetParentHub();
    if (!hub)
        return nullptr;
    return dynamic_cast<EvidentHubWin*>(hub);
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

    EvidentHubWin* hub = GetHub();
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
    EvidentHubWin* hub = GetHub();
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
        EvidentHubWin* hub = GetHub();
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

        EvidentHubWin* hub = GetHub();
        if (!hub)
            return DEVICE_ERR;

        // Convert from 0-based to 1-based for the microscope
        long targetPos = pos + 1;

        // Check if already at target position
        long currentPos = hub->GetModel()->GetPosition(DeviceType_CondenserTurret);
        if (currentPos == targetPos)
        {
            // Already at target, no need to move
            hub->GetModel()->SetBusy(DeviceType_CondenserTurret, false);
            return DEVICE_OK;
        }

        // Set busy before sending command
        hub->GetModel()->SetBusy(DeviceType_CondenserTurret, true);

        std::string cmd = BuildCommand(CMD_CONDENSER_TURRET, static_cast<int>(targetPos));
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
        hub->GetModel()->SetPosition(DeviceType_CondenserTurret, targetPos);
        hub->GetModel()->SetBusy(DeviceType_CondenserTurret, false);
    }
    return DEVICE_OK;
}

EvidentHubWin* EvidentCondenserTurret::GetHub()
{
    MM::Hub* hub = GetParentHub();
    if (!hub)
        return nullptr;
    return dynamic_cast<EvidentHubWin*>(hub);
}

int EvidentCondenserTurret::EnableNotifications(bool enable)
{
    EvidentHubWin* hub = GetHub();
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

    EvidentHubWin* hub = GetHub();
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

    // Create Brightness property (DIA illumination intensity)
    pAct = new CPropertyAction(this, &EvidentDIAShutter::OnBrightness);
    ret = CreateProperty("Brightness", "0", MM::Integer, false, pAct);
    if (ret != DEVICE_OK)
        return ret;

    SetPropertyLimits("Brightness", 0, 255);

    // Query current brightness value
    std::string cmd = BuildQuery(CMD_DIA_ILLUMINATION);
    std::string response;
    ret = hub->ExecuteCommand(cmd, response);
    if (ret == DEVICE_OK)
    {
        std::vector<std::string> params = ParseParameters(response);
        if (params.size() > 0)
        {
            int brightness = ParseIntParameter(params[0]);
            if (brightness >= 0 && brightness <= 255)
            {
                hub->GetModel()->SetPosition(DeviceType_DIABrightness, brightness);
                SetProperty("Brightness", CDeviceUtils::ConvertToString(brightness));
            }
        }
    }

    // Create Mechanical Shutter property (controls physical shutter, independent of logical shutter)
    pAct = new CPropertyAction(this, &EvidentDIAShutter::OnMechanicalShutter);
    ret = CreateProperty("Mechanical Shutter", "Closed", MM::String, false, pAct);
    if (ret != DEVICE_OK)
        return ret;

    AddAllowedValue("Mechanical Shutter", "Closed");
    AddAllowedValue("Mechanical Shutter", "Open");

    // Query current mechanical shutter state
    cmd = BuildQuery(CMD_DIA_SHUTTER);
    ret = hub->ExecuteCommand(cmd, response);
    if (ret == DEVICE_OK)
    {
        std::vector<std::string> params = ParseParameters(response);
        if (params.size() > 0)
        {
            int state = ParseIntParameter(params[0]);
            // Note: DSH 0 = Open, DSH 1 = Closed (reversed)
            SetProperty("Mechanical Shutter", (state == 0) ? "Open" : "Closed");
        }
    }

    // Enable brightness change notifications
    ret = EnableNotifications(true);
    if (ret != DEVICE_OK)
        return ret;

    // Register with hub so notification handler can call OnPropertyChanged
    hub->RegisterDeviceAsUsed(DeviceType_DIAShutter, this);

    initialized_ = true;

    // Close logical shutter on startup (set brightness to 0)
    SetOpen(false);

    return DEVICE_OK;
}

int EvidentDIAShutter::Shutdown()
{
    if (initialized_)
    {
        // Disable brightness change notifications
        EnableNotifications(false);

        // Unregister from hub
        EvidentHubWin* hub = GetHub();
        if (hub)
            hub->UnRegisterDeviceAsUsed(DeviceType_DIAShutter);

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
    EvidentHubWin* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    if (open)
    {
        // Logical open: Set brightness to remembered value
        int rememberedBrightness = hub->GetRememberedDIABrightness();
        std::string cmd = BuildCommand(CMD_DIA_ILLUMINATION, rememberedBrightness);
        std::string response;
        int ret = hub->ExecuteCommand(cmd, response);
        if (ret != DEVICE_OK)
            return ret;

        if (!IsPositiveAck(response, CMD_DIA_ILLUMINATION))
            return ERR_NEGATIVE_ACK;

        // Update model
        hub->GetModel()->SetPosition(DeviceType_DIABrightness, rememberedBrightness);
    }
    else
    {
        // Logical close: Remember current brightness, then set to 0
        std::string cmd = BuildQuery(CMD_DIA_ILLUMINATION);
        std::string response;
        int ret = hub->ExecuteCommand(cmd, response);
        if (ret != DEVICE_OK)
            return ret;

        std::vector<std::string> params = ParseParameters(response);
        if (params.size() > 0)
        {
            int brightness = ParseIntParameter(params[0]);
            if (brightness > 0)
                hub->SetRememberedDIABrightness(brightness);
        }

        // Set brightness to 0
        cmd = BuildCommand(CMD_DIA_ILLUMINATION, 0);
        ret = hub->ExecuteCommand(cmd, response);
        if (ret != DEVICE_OK)
            return ret;

        if (!IsPositiveAck(response, CMD_DIA_ILLUMINATION))
            return ERR_NEGATIVE_ACK;

        // Update model
        hub->GetModel()->SetPosition(DeviceType_DIABrightness, 0);
    }

    return DEVICE_OK;
}

int EvidentDIAShutter::GetOpen(bool& open)
{
    EvidentHubWin* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    // Logical shutter state is based on brightness: open if brightness > 0
    std::string cmd = BuildQuery(CMD_DIA_ILLUMINATION);
    std::string response;
    int ret = hub->ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
        return ret;

    std::vector<std::string> params = ParseParameters(response);
    if (params.size() > 0)
    {
        int brightness = ParseIntParameter(params[0]);
        open = (brightness > 0);

        // Update model
        hub->GetModel()->SetPosition(DeviceType_DIABrightness, brightness);
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

int EvidentDIAShutter::OnBrightness(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    EvidentHubWin* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    if (eAct == MM::BeforeGet)
    {
        // Return remembered brightness (what brightness will be when shutter opens)
        int rememberedBrightness = hub->GetRememberedDIABrightness();
        pProp->Set(static_cast<long>(rememberedBrightness));
    }
    else if (eAct == MM::AfterSet)
    {
        long brightness;
        pProp->Get(brightness);

        // Always update remembered brightness
        hub->SetRememberedDIABrightness(static_cast<int>(brightness));

        // Only send DIL command if logical shutter is open (actual brightness > 0)
        long currentBrightness = hub->GetModel()->GetPosition(DeviceType_DIABrightness);
        if (currentBrightness > 0)
        {
            // Shutter is open: update actual lamp brightness
            std::string cmd = BuildCommand(CMD_DIA_ILLUMINATION, static_cast<int>(brightness));
            std::string response;
            int ret = hub->ExecuteCommand(cmd, response);
            if (ret != DEVICE_OK)
                return ret;

            if (!IsPositiveAck(response, CMD_DIA_ILLUMINATION))
                return ERR_NEGATIVE_ACK;

            // Update model
            hub->GetModel()->SetPosition(DeviceType_DIABrightness, brightness);
        }
        // If shutter is closed, don't send DIL command, don't update model
    }

    return DEVICE_OK;
}

int EvidentDIAShutter::OnMechanicalShutter(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    EvidentHubWin* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    if (eAct == MM::BeforeGet)
    {
        // Query physical shutter state
        std::string cmd = BuildQuery(CMD_DIA_SHUTTER);
        std::string response;
        int ret = hub->ExecuteCommand(cmd, response);
        if (ret != DEVICE_OK)
            return ret;

        std::vector<std::string> params = ParseParameters(response);
        if (params.size() > 0)
        {
            int state = ParseIntParameter(params[0]);
            // Note: DSH 0 = Open, DSH 1 = Closed (reversed)
            pProp->Set((state == 0) ? "Open" : "Closed");
        }
    }
    else if (eAct == MM::AfterSet)
    {
        std::string value;
        pProp->Get(value);

        // Convert "Open"/"Closed" to 0/1 (reversed: Open=0, Closed=1)
        int state = (value == "Open") ? 0 : 1;

        // Send DSH command to control physical shutter
        std::string cmd = BuildCommand(CMD_DIA_SHUTTER, state);
        std::string response;
        int ret = hub->ExecuteCommand(cmd, response);
        if (ret != DEVICE_OK)
            return ret;

        if (!IsPositiveAck(response, CMD_DIA_SHUTTER))
            return ERR_NEGATIVE_ACK;
    }

    return DEVICE_OK;
}

int EvidentDIAShutter::EnableNotifications(bool enable)
{
    EvidentHubWin* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    return hub->EnableNotification(CMD_DIA_ILLUMINATION_NOTIFY, enable);
}

EvidentHubWin* EvidentDIAShutter::GetHub()
{
    MM::Hub* hub = GetParentHub();
    if (!hub)
        return nullptr;
    return dynamic_cast<EvidentHubWin*>(hub);
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

    EvidentHubWin* hub = GetHub();
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

    // Register with hub so SetOpen can notify indicator changes
    hub->RegisterDeviceAsUsed(DeviceType_EPIShutter1, this);

    initialized_ = true;

    // Close shutter on startup
    SetOpen(false);

    return DEVICE_OK;
}

int EvidentEPIShutter1::Shutdown()
{
    if (initialized_)
    {
        // Unregister from hub
        EvidentHubWin* hub = GetHub();
        if (hub)
            hub->UnRegisterDeviceAsUsed(DeviceType_EPIShutter1);

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
    EvidentHubWin* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    std::string cmd = BuildCommand(CMD_EPI_SHUTTER1, open ? 1 : 0);
    std::string response;
    int ret = hub->ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
        return ret;

    if (!IsPositiveAck(response, CMD_EPI_SHUTTER1))
        return ERR_NEGATIVE_ACK;

    // Update MCU indicator I5 with new shutter state (0=closed, 1=open)
    hub->UpdateEPIShutter1Indicator(open ? 1 : 0);

    return DEVICE_OK;
}

int EvidentEPIShutter1::GetOpen(bool& open)
{
    EvidentHubWin* hub = GetHub();
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

EvidentHubWin* EvidentEPIShutter1::GetHub()
{
    MM::Hub* hub = GetParentHub();
    if (!hub)
        return nullptr;
    return dynamic_cast<EvidentHubWin*>(hub);
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

    EvidentHubWin* hub = GetHub();
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

    // Register with hub so encoder can notify property changes
    hub->RegisterDeviceAsUsed(DeviceType_MirrorUnit1, this);

    initialized_ = true;
    return DEVICE_OK;
}

int EvidentMirrorUnit1::Shutdown()
{
    if (initialized_)
    {
        // Unregister from hub
        EvidentHubWin* hub = GetHub();
        if (hub)
            hub->UnRegisterDeviceAsUsed(DeviceType_MirrorUnit1);

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
        EvidentHubWin* hub = GetHub();
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

        EvidentHubWin* hub = GetHub();
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

EvidentHubWin* EvidentMirrorUnit1::GetHub()
{
    MM::Hub* hub = GetParentHub();
    if (!hub)
        return nullptr;
    return dynamic_cast<EvidentHubWin*>(hub);
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

    EvidentHubWin* hub = GetHub();
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
    EvidentHubWin* hub = GetHub();
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
        EvidentHubWin* hub = GetHub();
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

        EvidentHubWin* hub = GetHub();
        if (!hub)
            return DEVICE_ERR;

        // Set target position BEFORE sending command
        // Polarizer uses 0-based indexing (PO 0 = Out, PO 1 = In)
        hub->GetModel()->SetTargetPosition(DeviceType_Polarizer, pos);

        // Check if already at target position
        long currentPos = hub->GetModel()->GetPosition(DeviceType_Polarizer);
        if (currentPos == pos)
        {
            // Already at target, no need to move
            hub->GetModel()->SetBusy(DeviceType_Polarizer, false);
            return DEVICE_OK;
        }

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

EvidentHubWin* EvidentPolarizer::GetHub()
{
    MM::Hub* hub = GetParentHub();
    if (!hub)
        return nullptr;
    return dynamic_cast<EvidentHubWin*>(hub);
}

int EvidentPolarizer::EnableNotifications(bool enable)
{
    EvidentHubWin* hub = GetHub();
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

    EvidentHubWin* hub = GetHub();
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
        EvidentHubWin* hub = GetHub();
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

        EvidentHubWin* hub = GetHub();
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

EvidentHubWin* EvidentDICPrism::GetHub()
{
    MM::Hub* hub = GetParentHub();
    if (!hub)
        return nullptr;
    return dynamic_cast<EvidentHubWin*>(hub);
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

    EvidentHubWin* hub = GetHub();
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
        EvidentHubWin* hub = GetHub();
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

        EvidentHubWin* hub = GetHub();
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

EvidentHubWin* EvidentEPIND::GetHub()
{
    MM::Hub* hub = GetParentHub();
    if (!hub)
        return nullptr;
    return dynamic_cast<EvidentHubWin*>(hub);
}

///////////////////////////////////////////////////////////////////////////////
// EvidentCorrectionCollar - Correction Collar Implementation
///////////////////////////////////////////////////////////////////////////////

EvidentCorrectionCollar::EvidentCorrectionCollar() :
    initialized_(false),
    linked_(false),
    name_(g_CorrectionCollarDeviceName),
    stepSizeUm_(CORRECTION_COLLAR_STEP_SIZE_UM)
{
    InitializeDefaultErrorMessages();
    SetErrorText(ERR_DEVICE_NOT_AVAILABLE, "Correction collar not available on this microscope");
    SetErrorText(ERR_CORRECTION_COLLAR_NOT_LINKED, "Correction Collar must be linked before setting position. Set Activate property to 'Linked'.");
    SetErrorText(ERR_CORRECTION_COLLAR_LINK_FAILED, "Correction Collar linking failed. Ensure correct objective is installed (typically objective 6).");

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

    EvidentHubWin* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    if (!hub->IsDevicePresent(DeviceType_CorrectionCollar))
        return ERR_DEVICE_NOT_AVAILABLE;

    // Create Activate property for linking/unlinking
    CPropertyAction* pAct = new CPropertyAction(this, &EvidentCorrectionCollar::OnActivate);
    int ret = CreateProperty("Activate", "Unlinked", MM::String, false, pAct);
    if (ret != DEVICE_OK)
        return ret;

    AddAllowedValue("Activate", "Linked");
    AddAllowedValue("Activate", "Unlinked");

    // Add firmware version as read-only property
    std::string version = hub->GetDeviceVersion(DeviceType_CorrectionCollar);
    if (!version.empty())
    {
        ret = CreateProperty("Firmware Version", version.c_str(), MM::String, true);
        if (ret != DEVICE_OK)
            return ret;
    }

    initialized_ = true;
    return DEVICE_OK;
}

int EvidentCorrectionCollar::Shutdown()
{
    if (initialized_)
    {
        // Auto-unlink on shutdown if linked
        if (linked_)
        {
            EvidentHubWin* hub = GetHub();
            if (hub)
            {
                std::string cmd = BuildCommand(CMD_CORRECTION_COLLAR_LINK, 0);  // 0 = Unlink
                std::string response;
                hub->ExecuteCommand(cmd, response);
                // Don't check response - best effort unlink
            }
            linked_ = false;

            // Notify core that position changed to 0
            GetCoreCallback()->OnStagePositionChanged(this, 0.0);
        }
        initialized_ = false;
    }
    return DEVICE_OK;
}

bool EvidentCorrectionCollar::Busy()
{
    return false;  // Correction collar changes are instantaneous
}

int EvidentCorrectionCollar::OnActivate(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    EvidentHubWin* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    if (eAct == MM::BeforeGet)
    {
        // Return current linked state
        pProp->Set(linked_ ? "Linked" : "Unlinked");
    }
    else if (eAct == MM::AfterSet)
    {
        std::string state;
        pProp->Get(state);

        if (state == "Linked" && !linked_)
        {
            // Link the correction collar
            std::string cmd = BuildCommand(CMD_CORRECTION_COLLAR_LINK, 1);  // 1 = Link
            std::string response;
            int ret = hub->ExecuteCommand(cmd, response);
            if (ret != DEVICE_OK)
                return ret;

            if (!IsPositiveAck(response, CMD_CORRECTION_COLLAR_LINK))
                return ERR_CORRECTION_COLLAR_LINK_FAILED;

            // Wait for link to complete before initializing
            // The hardware needs time after linking before accepting init command
            CDeviceUtils::SleepMs(500);

            // Initialize the correction collar
            cmd = BuildCommand(CMD_CORRECTION_COLLAR_INIT);
            ret = hub->ExecuteCommand(cmd, response);
            if (ret != DEVICE_OK)
            {
                // Link succeeded but init failed - try to unlink
                cmd = BuildCommand(CMD_CORRECTION_COLLAR_LINK, 0);
                hub->ExecuteCommand(cmd, response);
                return ERR_CORRECTION_COLLAR_LINK_FAILED;
            }

            if (!IsPositiveAck(response, CMD_CORRECTION_COLLAR_INIT))
            {
                // Link succeeded but init failed - try to unlink
                cmd = BuildCommand(CMD_CORRECTION_COLLAR_LINK, 0);
                hub->ExecuteCommand(cmd, response);
                return ERR_CORRECTION_COLLAR_LINK_FAILED;
            }

            // Wait for correction collar initialization to complete
            // The hardware needs time to initialize before accepting position commands
            CDeviceUtils::SleepMs(500);

            // Successfully linked and initialized
            linked_ = true;
        }
        else if (state == "Unlinked" && linked_)
        {
            // Unlink the correction collar
            std::string cmd = BuildCommand(CMD_CORRECTION_COLLAR_LINK, 0);  // 0 = Unlink
            std::string response;
            int ret = hub->ExecuteCommand(cmd, response);
            if (ret != DEVICE_OK)
                return ret;

            if (!IsPositiveAck(response, CMD_CORRECTION_COLLAR_LINK))
                return ERR_NEGATIVE_ACK;

            // Successfully unlinked
            linked_ = false;

            // Notify core that position changed to 0
            GetCoreCallback()->OnStagePositionChanged(this, 0.0);
        }
    }
    return DEVICE_OK;
}

EvidentHubWin* EvidentCorrectionCollar::GetHub()
{
    MM::Hub* hub = GetParentHub();
    if (!hub)
        return nullptr;
    return dynamic_cast<EvidentHubWin*>(hub);
}

int EvidentCorrectionCollar::SetPositionUm(double pos)
{
    // Convert μm to steps (1:1 ratio)
    long steps = static_cast<long>(pos / stepSizeUm_);
    return SetPositionSteps(steps);
}

int EvidentCorrectionCollar::GetPositionUm(double& pos)
{
    long steps;
    int ret = GetPositionSteps(steps);
    if (ret != DEVICE_OK)
        return ret;

    pos = steps * stepSizeUm_;
    return DEVICE_OK;
}

int EvidentCorrectionCollar::SetPositionSteps(long steps)
{
    // Check if linked
    if (!linked_)
        return ERR_CORRECTION_COLLAR_NOT_LINKED;

    EvidentHubWin* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    // Clamp to limits
    if (steps < CORRECTION_COLLAR_MIN_POS) steps = CORRECTION_COLLAR_MIN_POS;
    if (steps > CORRECTION_COLLAR_MAX_POS) steps = CORRECTION_COLLAR_MAX_POS;

    // Send CC command with position
    std::string cmd = BuildCommand(CMD_CORRECTION_COLLAR, static_cast<int>(steps));
    std::string response;
    int ret = hub->ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
        return ret;

    if (!IsPositiveAck(response, CMD_CORRECTION_COLLAR))
        return ERR_NEGATIVE_ACK;

    return DEVICE_OK;
}

int EvidentCorrectionCollar::GetPositionSteps(long& steps)
{
    // If not linked, return 0 (no error)
    if (!linked_)
    {
        steps = 0;
        return DEVICE_OK;
    }

    EvidentHubWin* hub = GetHub();
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
        if (pos >= CORRECTION_COLLAR_MIN_POS && pos <= CORRECTION_COLLAR_MAX_POS)
        {
            steps = pos;
            return DEVICE_OK;
        }
    }

    return ERR_INVALID_RESPONSE;
}

int EvidentCorrectionCollar::SetOrigin()
{
    // Not supported by IX85 correction collar
    return DEVICE_UNSUPPORTED_COMMAND;
}

int EvidentCorrectionCollar::GetLimits(double& lower, double& upper)
{
    lower = CORRECTION_COLLAR_MIN_POS * stepSizeUm_;
    upper = CORRECTION_COLLAR_MAX_POS * stepSizeUm_;
    return DEVICE_OK;
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

    EvidentHubWin* hub = GetHub();
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

    // Close shutter on startup
    SetOpen(false);

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
    EvidentHubWin* hub = GetHub();
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
    EvidentHubWin* hub = GetHub();
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

EvidentHubWin* EvidentEPIShutter2::GetHub()
{
    MM::Hub* hub = GetParentHub();
    if (!hub)
        return nullptr;
    return dynamic_cast<EvidentHubWin*>(hub);
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

    EvidentHubWin* hub = GetHub();
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
        EvidentHubWin* hub = GetHub();
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

        EvidentHubWin* hub = GetHub();
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

EvidentHubWin* EvidentMirrorUnit2::GetHub()
{
    MM::Hub* hub = GetParentHub();
    if (!hub)
        return nullptr;
    return dynamic_cast<EvidentHubWin*>(hub);
}

int EvidentMirrorUnit2::EnableNotifications(bool /*enable*/)
{
    // NMUINIT2 is an initialization notification, not a position change notification
    // MirrorUnit2 uses query-based position tracking instead
    return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// EvidentAutofocus - ZDC Autofocus Implementation
///////////////////////////////////////////////////////////////////////////////

EvidentAutofocus::EvidentAutofocus() :
    initialized_(false),
    name_(g_AutofocusDeviceName),
    continuousFocusing_(false),
    afStatus_(0),
    nearLimit_(1050000),  // Near = upper limit (closer to sample)
    farLimit_(0),         // Far = lower limit (farther from sample)
    lastNosepiecePos_(-1),
    lastCoverslipType_(-1),
    zdcInitNeeded_(false),
    workflowMode_(2)      // Default to Find-Focus-With-Offset mode
{
    InitializeDefaultErrorMessages();
    SetErrorText(ERR_DEVICE_NOT_AVAILABLE, "ZDC Autofocus not available on this microscope");

    CreateHubIDProperty();
}

EvidentAutofocus::~EvidentAutofocus()
{
    Shutdown();
}

void EvidentAutofocus::GetName(char* pszName) const
{
    CDeviceUtils::CopyLimitedString(pszName, name_.c_str());
}

int EvidentAutofocus::Initialize()
{
    if (initialized_)
        return DEVICE_OK;

    EvidentHubWin* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    if (!hub->IsDevicePresent(EvidentIX85Win::DeviceType_Autofocus))
        return ERR_DEVICE_NOT_AVAILABLE;

    // Query initial AF status
    std::string cmd = BuildQuery(CMD_AF_STATUS);
    std::string response;
    int ret = hub->ExecuteCommand(cmd, response);
    if (ret == DEVICE_OK)
    {
        std::vector<std::string> params = ParseParameters(response);
        if (params.size() > 0 && params[0] != "X")
        {
            afStatus_ = ParseIntParameter(params[0]);
        }
    }

    // Query initial AF limits
    cmd = BuildQuery(CMD_AF_LIMIT);
    ret = hub->ExecuteCommand(cmd, response);
    if (ret == DEVICE_OK)
    {
        std::vector<std::string> params = ParseParameters(response);
        if (params.size() >= 2 && params[0] != "X" && params[1] != "X")
        {
            nearLimit_ = ParseIntParameter(params[0]);
            farLimit_ = ParseIntParameter(params[1]);
        }
    }

    // Create AF Status property (read-only)
    CPropertyAction* pAct = new CPropertyAction(this, &EvidentAutofocus::OnAFStatus);
    ret = CreateProperty("AF Status", GetAFStatusString(afStatus_).c_str(), MM::String, true, pAct);
    if (ret != DEVICE_OK)
        return ret;

    // Create Near Limit property
    pAct = new CPropertyAction(this, &EvidentAutofocus::OnNearLimit);
    ret = CreateProperty("Near Limit (um)", CDeviceUtils::ConvertToString(nearLimit_ * 0.01), MM::Float, false, pAct);
    if (ret != DEVICE_OK)
        return ret;

    // Create Far Limit property
    pAct = new CPropertyAction(this, &EvidentAutofocus::OnFarLimit);
    ret = CreateProperty("Far Limit (um)", CDeviceUtils::ConvertToString(farLimit_ * 0.01), MM::Float, false, pAct);
    if (ret != DEVICE_OK)
        return ret;

    // Create Cover Slip Type property
    pAct = new CPropertyAction(this, &EvidentAutofocus::OnCoverSlipType);
    ret = CreateProperty("Cover Slip Type", "Glass", MM::String, false, pAct);
    if (ret != DEVICE_OK)
        return ret;
    AddAllowedValue("Cover Slip Type", "Glass");
    AddAllowedValue("Cover Slip Type", "Plastic");

    // Create Cover Slip Thickness Glass property
    pAct = new CPropertyAction(this, &EvidentAutofocus::OnCoverSlipThicknessGlass);
    ret = CreateProperty("Cover Slip Thickness Glass (um)", "170", MM::Float, false, pAct);
    if (ret != DEVICE_OK)
        return ret;
    SetPropertyLimits("Cover Slip Thickness Glass (um)", 150, 500);

    // Create Cover Slip Thickness Plastic property
    pAct = new CPropertyAction(this, &EvidentAutofocus::OnCoverSlipThicknessPlastic);
    ret = CreateProperty("Cover Slip Thickness Plastic (um)", "1000", MM::Float, false, pAct);
    if (ret != DEVICE_OK)
        return ret;
    SetPropertyLimits("Cover Slip Thickness Plastic (um)", 700, 1500);

    // Create DIC Mode property
    pAct = new CPropertyAction(this, &EvidentAutofocus::OnDICMode);
    ret = CreateProperty("DIC Mode", "Off", MM::String, false, pAct);
    if (ret != DEVICE_OK)
        return ret;
    AddAllowedValue("DIC Mode", "Off");
    AddAllowedValue("DIC Mode", "On");

    // Create Buzzer Success property
    pAct = new CPropertyAction(this, &EvidentAutofocus::OnBuzzerSuccess);
    ret = CreateProperty("Buzzer Success", "On", MM::String, false, pAct);
    if (ret != DEVICE_OK)
        return ret;
    AddAllowedValue("Buzzer Success", "Off");
    AddAllowedValue("Buzzer Success", "On");

    // Create Buzzer Failure property
    pAct = new CPropertyAction(this, &EvidentAutofocus::OnBuzzerFailure);
    ret = CreateProperty("Buzzer Failure", "On", MM::String, false, pAct);
    if (ret != DEVICE_OK)
        return ret;
    AddAllowedValue("Buzzer Failure", "Off");
    AddAllowedValue("Buzzer Failure", "On");

    // Create AF Workflow Mode property
    pAct = new CPropertyAction(this, &EvidentAutofocus::OnWorkflowMode);
    ret = CreateProperty("AF-Workflow-Mode", "Find-Focus-With-Offset", MM::String, false, pAct);
    if (ret != DEVICE_OK)
        return ret;
    AddAllowedValue("AF-Workflow-Mode", "Measure-Offset");
    AddAllowedValue("AF-Workflow-Mode", "Find-Focus-With-Offset");
    AddAllowedValue("AF-Workflow-Mode", "Continuous-Focus");

    // Create Measured Focus Offset property
    pAct = new CPropertyAction(this, &EvidentAutofocus::OnMeasuredFocusOffset);
    ret = CreateProperty("Measured-Focus-Offset-um", "0.0", MM::Float, false, pAct);
    if (ret != DEVICE_OK)
        return ret;
    SetPropertyLimits("Measured-Focus-Offset-um", -10000.0, 10000.0);  // ±10mm range

    // Enable AF status notifications
    EnableNotifications(true);

    // Mark that ZDC needs initialization (deferred to first FullFocus() call)
    zdcInitNeeded_ = true;

    hub->RegisterDeviceAsUsed(EvidentIX85Win::DeviceType_Autofocus, this);
    initialized_ = true;
    return DEVICE_OK;
}

int EvidentAutofocus::Shutdown()
{
    if (initialized_)
    {
        // Stop AF if running
        if (continuousFocusing_)
        {
            StopAF();
        }

        EvidentHubWin* hub = GetHub();
        if (hub)
        {
            EnableNotifications(false);
            hub->UnRegisterDeviceAsUsed(EvidentIX85Win::DeviceType_Autofocus);
        }
        initialized_ = false;
    }
    return DEVICE_OK;
}

bool EvidentAutofocus::Busy()
{
    EvidentHubWin* hub = GetHub();
    if (!hub)
        return false;

    // AF is busy during One-Shot or Focus Search operations
    // also check the focus drive as we may be moving it to an in focus position
    return (afStatus_ == 4) || hub->GetModel()->IsBusy(DeviceType_Focus);  // 4 = Search
}

void EvidentAutofocus::UpdateAFStatus(int status)
{
    if (afStatus_ != status)
    {
        afStatus_ = status;
        OnPropertyChanged("AF Status", GetAFStatusString(afStatus_).c_str());
    }
}

void EvidentAutofocus::UpdateMeasuredZOffset(long offsetSteps)
{
    // Update property to reflect new offset
    double offsetUm = offsetSteps * FOCUS_STEP_SIZE_UM;
    std::ostringstream valStr;
    valStr << offsetUm;
    OnPropertyChanged("Measured-Focus-Offset-um", valStr.str().c_str());
}

int EvidentAutofocus::SetContinuousFocusing(bool state)
{
    EvidentHubWin* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    if (state)
    {
        // Check if ZDC needs re-initialization (objective changed, or settings changed)
        long nosepiecePos = hub->GetModel()->GetPosition(EvidentIX85Win::DeviceType_Nosepiece);
        if (nosepiecePos != lastNosepiecePos_ || zdcInitNeeded_)
        {
            int ret = InitializeZDC();
            if (ret != DEVICE_OK)
                return ret;

            // Clear the flag after successful initialization
            zdcInitNeeded_ = false;
        }

        // Determine AF mode to use
        int afMode;
        if (workflowMode_ == 3)  // Continuous Focus workflow
        {
            // Use AF mode 2 (Continuous Focus Drive)
            afMode = 2;
            LogMessage("Continuous Focus mode: Using AF 2 (Focus Drive tracking)");
        }
        else
        {
            // Use AF mode 2 (Focus Drive) as default
            afMode = 2;
        }

        // Start continuous AF
        std::string cmd = BuildCommand(CMD_AF_START_STOP, afMode);
        std::string response;
        int ret = hub->ExecuteCommand(cmd, response);
        if (ret != DEVICE_OK)
            return ret;

        if (!IsPositiveAck(response, CMD_AF_START_STOP))
            return ERR_NEGATIVE_ACK;

        continuousFocusing_ = true;
    }
    else
    {
        // Stop AF
        int ret = StopAF();
        if (ret != DEVICE_OK)
            return ret;

        continuousFocusing_ = false;
    }

    return DEVICE_OK;
}

int EvidentAutofocus::GetContinuousFocusing(bool& state)
{
    state = continuousFocusing_;
    return DEVICE_OK;
}

bool EvidentAutofocus::IsContinuousFocusLocked()
{
    EvidentHubWin* hub = GetHub();
    if (!hub)
        return false;

    // Query current AF status
    std::string cmd = BuildQuery(CMD_AF_STATUS);
    std::string response;
    int ret = hub->ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
        return false;

    std::vector<std::string> params = ParseParameters(response);
    if (params.size() > 0 && params[0] != "X")
    {
        int newStatus = ParseIntParameter(params[0]);
        if (afStatus_ != newStatus)
        {
            afStatus_ = newStatus;
            OnPropertyChanged("AF Status", GetAFStatusString(afStatus_).c_str());
        }
    }

    // Locked when in Focus (1) or Track (2) state
    return (afStatus_ == 1 || afStatus_ == 2);
}

int EvidentAutofocus::FullFocus()
{
    EvidentHubWin* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    // Handle workflow-specific modes
    if (workflowMode_ == 1)  // Measure Offset mode
    {
        return MeasureZOffset();
    }
    else if (workflowMode_ == 2)  // Find Focus with Offset mode
    {
        return FindFocusWithOffset();
    }

    // Check if ZDC needs re-initialization (objective changed, or settings changed)
    long nosepiecePos = hub->GetModel()->GetPosition(EvidentIX85Win::DeviceType_Nosepiece);
    if (nosepiecePos != lastNosepiecePos_ || zdcInitNeeded_)
    {
        int ret = InitializeZDC();
        if (ret != DEVICE_OK)
            return ret;

        // Clear the flag after successful initialization
        zdcInitNeeded_ = false;
    }

    // Use continuous AF approach to avoid resetting offset lens position
    // Start continuous AF with AF mode 2 (Focus Drive)
    std::string cmd = BuildCommand(CMD_AF_START_STOP, 2);
    std::string response;
    int ret = hub->ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
        return ret;

    if (!IsPositiveAck(response, CMD_AF_START_STOP))
        return ERR_NEGATIVE_ACK;

    // Wait for focus to be achieved (afStatus_ becomes 1=Focus)
    // afStatus_ is updated by notifications (NAFST), no polling needed
    // Timeout: max 30 seconds
    const int maxWaitMs = 30000;
    const int sleepIntervalMs = 100;
    int elapsedMs = 0;

    while (elapsedMs < maxWaitMs)
    {
        CDeviceUtils::SleepMs(sleepIntervalMs);
        elapsedMs += sleepIntervalMs;

        // Check afStatus_ which is updated by NAFST notifications
        if (afStatus_ == 1)  // 1 = Focus achieved
        {
            // AF 2 (Focus drive): manually stop after achieving focus
            ret = StopAF();
            if (ret != DEVICE_OK)
                return ret;
            return DEVICE_OK;
        }
        else if (afStatus_ == 0)  // Stopped
        {
            // AF 2 stopped unexpectedly - failure
            return ERR_NEGATIVE_ACK;
        }
    }

    // Timeout - stop AF and return error
    StopAF();
    return ERR_COMMAND_TIMEOUT;
}

int EvidentAutofocus::IncrementalFocus()
{
    EvidentHubWin* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    // Execute Focus Search
    std::string cmd = BuildCommand(CMD_AF_START_STOP, 3);  // 3 = Focus Search
    std::string response;
    int ret = hub->ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
        return ret;

    if (!IsPositiveAck(response, CMD_AF_START_STOP))
        return ERR_NEGATIVE_ACK;

    return DEVICE_OK;
}

int EvidentAutofocus::GetLastFocusScore(double& score)
{
    score = 0.0;
    return DEVICE_OK;
}

int EvidentAutofocus::GetCurrentFocusScore(double& score)
{
    score = 0.0;
    return DEVICE_OK;
}

int EvidentAutofocus::GetOffset(double& offset)
{
    // Get offset lens position
    EvidentHubWin* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    long pos = hub->GetModel()->GetPosition(EvidentIX85Win::DeviceType_OffsetLens);
    offset = pos * OFFSET_LENS_STEP_SIZE_UM;
    return DEVICE_OK;
}

int EvidentAutofocus::SetOffset(double offset)
{
    // Set offset lens position
    EvidentHubWin* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    long steps = static_cast<long>(offset / OFFSET_LENS_STEP_SIZE_UM);

    std::string cmd = BuildCommand(CMD_OFFSET_LENS_GOTO, static_cast<int>(steps));
    std::string response;
    int ret = hub->ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
        return ret;

    if (!IsPositiveAck(response, CMD_OFFSET_LENS_GOTO))
        return ERR_NEGATIVE_ACK;

    hub->GetModel()->SetPosition(EvidentIX85Win::DeviceType_OffsetLens, steps);
    return DEVICE_OK;
}

int EvidentAutofocus::GetMeasuredZOffset(double& offset)
{
    EvidentHubWin* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    // Get stored Z-offset in micrometers from the model
    offset = hub->GetModel()->GetMeasuredZOffset() * FOCUS_STEP_SIZE_UM;
    return DEVICE_OK;
}

int EvidentAutofocus::SetMeasuredZOffset(double offset)
{
    EvidentHubWin* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    // Set stored Z-offset (for manual override)
    long offsetSteps = static_cast<long>(offset / FOCUS_STEP_SIZE_UM);
    hub->GetModel()->SetMeasuredZOffset(offsetSteps);

    // Notify both devices that the offset has changed
    hub->NotifyMeasuredZOffsetChanged(offsetSteps);

    return DEVICE_OK;
}

int EvidentAutofocus::StopAF()
{
    EvidentHubWin* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    std::string cmd = BuildCommand(CMD_AF_STOP);
    std::string response;
    int ret = hub->ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
        return ret;

    // AFSTP returns + when stopped successfully
    if (!IsPositiveAck(response, CMD_AF_STOP))
        return ERR_NEGATIVE_ACK;

    if (afStatus_ != 0)
    {
        afStatus_ = 0;
        OnPropertyChanged("AF Status", GetAFStatusString(afStatus_).c_str());
    }
    return DEVICE_OK;
}

int EvidentAutofocus::InitializeZDC()
{
    EvidentHubWin* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    std::string cmd;
    std::string response;
    int ret;

    // Get current nosepiece position
    long nosepiecePos = hub->GetModel()->GetPosition(EvidentIX85Win::DeviceType_Nosepiece);
    if (nosepiecePos < 1 || nosepiecePos > 6)
        nosepiecePos = 1;  // Default to position 1

    // Get objective name for current nosepiece position
    const std::vector<EvidentIX85Win::ObjectiveInfo>& objectives = hub->GetObjectiveInfo();
    std::string objectiveName = "Unknown";
    if (nosepiecePos >= 1 && nosepiecePos <= (long)objectives.size())
    {
        objectiveName = objectives[nosepiecePos - 1].name;
    }

    // Step 1: Enter Setting status (OPE 1)
    cmd = BuildCommand(CMD_OPERATION_MODE, 1);
    ret = hub->ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
        return ret;
    if (!IsPositiveAck(response, CMD_OPERATION_MODE))
        return ERR_NEGATIVE_ACK;

    // Step 2: Set Coverslip type - query current setting first
    cmd = BuildQuery(CMD_COVERSLIP_TYPE);
    ret = hub->ExecuteCommand(cmd, response);
    int coverslipType = 1;  // Default to Glass
    if (ret == DEVICE_OK)
    {
        std::vector<std::string> params = ParseParameters(response);
        if (params.size() > 0)
        {
            coverslipType = ParseIntParameter(params[0]);
        }
    }
    cmd = BuildCommand(CMD_COVERSLIP_TYPE, coverslipType);
    ret = hub->ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
    {
        // Exit setting mode before returning error
        hub->ExecuteCommand(BuildCommand(CMD_OPERATION_MODE, 0), response);
        return ret;
    }
    if (!IsPositiveAck(response, CMD_COVERSLIP_TYPE))
    {
        hub->ExecuteCommand(BuildCommand(CMD_OPERATION_MODE, 0), response);
        return ERR_NEGATIVE_ACK;
    }

    // Step 3: Set objective lens for AF (S_OB position,name)
    std::ostringstream sobCmd;
    sobCmd << CMD_AF_SET_OBJECTIVE << " " << nosepiecePos << "," << objectiveName;
    ret = hub->ExecuteCommand(sobCmd.str(), response);
    if (ret != DEVICE_OK)
    {
        hub->ExecuteCommand(BuildCommand(CMD_OPERATION_MODE, 0), response);
        return ret;
    }
    if (!IsPositiveAck(response, CMD_AF_SET_OBJECTIVE))
    {
        hub->ExecuteCommand(BuildCommand(CMD_OPERATION_MODE, 0), response);
        return ERR_NEGATIVE_ACK;
    }

    // Step 4: Exit Setting status (OPE 0)
    cmd = BuildCommand(CMD_OPERATION_MODE, 0);
    ret = hub->ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
        return ret;
    if (!IsPositiveAck(response, CMD_OPERATION_MODE))
        return ERR_NEGATIVE_ACK;

    // Step 5: Set ZDC DM In (AFDM 1)
    cmd = BuildCommand(CMD_AF_DICHROIC, 1);
    ret = hub->ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
        return ret;
    if (!IsPositiveAck(response, CMD_AF_DICHROIC))
        return ERR_NEGATIVE_ACK;

    // Step 6: Move offset lens to base position for current objective (ABBP)
    cmd = BuildCommand(CMD_OFFSET_LENS_BASE_POSITION, static_cast<int>(nosepiecePos));
    ret = hub->ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
        return ret;
    if (!IsPositiveAck(response, CMD_OFFSET_LENS_BASE_POSITION))
        return ERR_NEGATIVE_ACK;

    // Step 7: Enable Focus Jog (JG 1)
    cmd = BuildCommand(CMD_JOG, 1);
    ret = hub->ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
        return ret;
    if (!IsPositiveAck(response, CMD_JOG))
        return ERR_NEGATIVE_ACK;

    // Step 8: Set AF search range (AFL nearLimit,farLimit)
    cmd = BuildCommand(CMD_AF_LIMIT, static_cast<int>(nearLimit_), static_cast<int>(farLimit_));
    ret = hub->ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
        return ret;
    if (!IsPositiveAck(response, CMD_AF_LIMIT))
        return ERR_NEGATIVE_ACK;

    // Update tracking variables
    lastNosepiecePos_ = nosepiecePos;
    lastCoverslipType_ = coverslipType;

    return DEVICE_OK;
}

int EvidentAutofocus::MeasureZOffset()
{
    EvidentHubWin* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    // Re-initialize ZDC if needed
    long nosepiecePos = hub->GetModel()->GetPosition(EvidentIX85Win::DeviceType_Nosepiece);
    if (nosepiecePos != lastNosepiecePos_ || zdcInitNeeded_)
    {
        int ret = InitializeZDC();
        if (ret != DEVICE_OK)
            return ret;
        zdcInitNeeded_ = false;
    }

    // Step 1: Store current focus position (user should have focused on sample)
    long originalZPos = hub->GetModel()->GetPosition(EvidentIX85Win::DeviceType_Focus);
    if (originalZPos < 0)
    {
        LogMessage("Focus position unknown, cannot measure offset");
        return ERR_POSITION_UNKNOWN;
    }

    std::ostringstream logMsg1;
    logMsg1 << "Measuring Z-offset: Starting from position " << originalZPos;
    LogMessage(logMsg1.str().c_str());

    // Step 2: Run AF mode 1 (One-Shot Z-only)
    std::string cmd = BuildCommand(CMD_AF_START_STOP, 1);
    std::string response;
    int ret = hub->ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
        return ret;

    if (!IsPositiveAck(response, CMD_AF_START_STOP))
        return ERR_NEGATIVE_ACK;

    // Step 3: Wait for AF to achieve focus (max 30 seconds)
    const int maxWaitMs = 30000;
    const int sleepIntervalMs = 100;
    int elapsedMs = 0;

    while (elapsedMs < maxWaitMs)
    {
        CDeviceUtils::SleepMs(sleepIntervalMs);
        elapsedMs += sleepIntervalMs;

        if (afStatus_ == 1)  // Focus achieved
        {
            // Stop AF (mode 1 requires manual stop)
            ret = StopAF();
            if (ret != DEVICE_OK)
                return ret;

            // Step 4: Read new focus position
            long newZPos = hub->GetModel()->GetPosition(EvidentIX85Win::DeviceType_Focus);

            // Step 5: Calculate and store offset
            // Offset = how to correct from ZDC's focus to user's desired focus
            long measuredZOffset = originalZPos - newZPos;
            hub->GetModel()->SetMeasuredZOffset(measuredZOffset);

            // Notify both devices and core of property change
            double offsetUm = measuredZOffset * FOCUS_STEP_SIZE_UM;
            std::ostringstream valStr;
            valStr << offsetUm;
            OnPropertyChanged("Measured-Focus-Offset-um", valStr.str().c_str());

            std::ostringstream logMsg2;
            logMsg2 << "Measured Z-offset: " << measuredZOffset <<
                      " steps (" << offsetUm << " um)";
            LogMessage(logMsg2.str().c_str());

            // Notify other devices of offset change
            hub->NotifyMeasuredZOffsetChanged(measuredZOffset);

            // Step 6: Return focus to original position
            cmd = BuildCommand(CMD_FOCUS_GOTO, static_cast<int>(originalZPos));
            ret = hub->ExecuteCommand(cmd, response);
            if (ret != DEVICE_OK)
                return ret;

            if (!IsPositiveAck(response, CMD_FOCUS_GOTO))
                return ERR_NEGATIVE_ACK;

            // Wait for focus to return to original position
            elapsedMs = 0;
            while (elapsedMs < maxWaitMs)
            {
                CDeviceUtils::SleepMs(sleepIntervalMs);
                elapsedMs += sleepIntervalMs;

                long currentPos = hub->GetModel()->GetPosition(EvidentIX85Win::DeviceType_Focus);
                if (abs(currentPos - originalZPos) <= FOCUS_POSITION_TOLERANCE)
                {
                    LogMessage("Z-offset measurement complete, returned to original position");
                    return DEVICE_OK;
                }
            }

            LogMessage("Warning: Timeout returning to original position");
            return DEVICE_OK;  // Offset measured successfully even if return timed out
        }
        else if (afStatus_ == 0)  // Stopped
        {
            // AF mode 1 auto-stops after completion (like AF mode 3)
            // Check if focus position changed from original - if so, AF succeeded
            long currentZPos = hub->GetModel()->GetPosition(EvidentIX85Win::DeviceType_Focus);

            if (currentZPos != originalZPos)
            {
                // Position changed - AF succeeded and auto-stopped
                LogMessage("AF mode 1 completed and auto-stopped");

                // Step 4: Read new focus position (already have it as currentZPos)
                long newZPos = currentZPos;

                // Step 5: Calculate and store offset
                // Offset = how to correct from ZDC's focus to user's desired focus
                long measuredZOffset = originalZPos - newZPos;
                hub->GetModel()->SetMeasuredZOffset(measuredZOffset);

                // Notify core of property change
                double offsetUm = measuredZOffset * FOCUS_STEP_SIZE_UM;
                std::ostringstream valStr;
                valStr << offsetUm;
                OnPropertyChanged("Measured-Focus-Offset-um", valStr.str().c_str());

                std::ostringstream logMsg2;
                logMsg2 << "Measured Z-offset: " << measuredZOffset <<
                          " steps (" << offsetUm << " um)";
                LogMessage(logMsg2.str().c_str());

                // Notify other devices of offset change
                hub->NotifyMeasuredZOffsetChanged(measuredZOffset);

                // Step 6: Return focus to original position
                cmd = BuildCommand(CMD_FOCUS_GOTO, static_cast<int>(originalZPos));
                ret = hub->ExecuteCommand(cmd, response);
                if (ret != DEVICE_OK)
                    return ret;

                if (!IsPositiveAck(response, CMD_FOCUS_GOTO))
                    return ERR_NEGATIVE_ACK;

                // Wait for focus to return to original position
                elapsedMs = 0;
                while (elapsedMs < maxWaitMs)
                {
                    CDeviceUtils::SleepMs(sleepIntervalMs);
                    elapsedMs += sleepIntervalMs;

                    long currentPos = hub->GetModel()->GetPosition(EvidentIX85Win::DeviceType_Focus);
                    if (abs(currentPos - originalZPos) <= FOCUS_POSITION_TOLERANCE)
                    {
                        LogMessage("Z-offset measurement complete, returned to original position");
                        return DEVICE_OK;
                    }
                }

                LogMessage("Warning: Timeout returning to original position");
                return DEVICE_OK;  // Offset measured successfully even if return timed out
            }
            else
            {
                // Position didn't change - AF failed
                LogMessage("AF stopped without finding focus (position unchanged)");
                return ERR_NEGATIVE_ACK;
            }
        }
    }

    // Timeout - stop AF and return error
    StopAF();
    LogMessage("Timeout during Z-offset measurement");
    return ERR_COMMAND_TIMEOUT;
}

int EvidentAutofocus::FindFocusWithOffset()
{
   // TODO: evaluate if we should set Busy to true and execute this in a separate thread
    EvidentHubWin* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    // Re-initialize ZDC if needed
    long nosepiecePos = hub->GetModel()->GetPosition(EvidentIX85Win::DeviceType_Nosepiece);
    if (nosepiecePos != lastNosepiecePos_ || zdcInitNeeded_)
    {
        int ret = InitializeZDC();
        if (ret != DEVICE_OK)
            return ret;
        zdcInitNeeded_ = false;
    }

    // Step 1: Run AF mode 3 (Offset lens mode)
    std::string cmd = BuildCommand(CMD_AF_START_STOP, 3);
    std::string response;
    int ret = hub->ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
        return ret;

    if (!IsPositiveAck(response, CMD_AF_START_STOP))
        return ERR_NEGATIVE_ACK;

    LogMessage("Find Focus with Offset: Running AF mode 3");

    // Step 2: Wait for AF to achieve focus (max 30 seconds)
    const int maxWaitMs = 30000;
    const int sleepIntervalMs = 100;
    int elapsedMs = 0;
    bool focusAchieved = false;

    while (elapsedMs < maxWaitMs)
    {
        CDeviceUtils::SleepMs(sleepIntervalMs);
        elapsedMs += sleepIntervalMs;

        if (afStatus_ == 1 || afStatus_ == 0)  // Focus achieved or auto-stopped
        {
            focusAchieved = true;
            break;
        }
    }

    if (!focusAchieved)
    {
        StopAF();
        LogMessage("Timeout during Find Focus with Offset");
        return ERR_COMMAND_TIMEOUT;
    }

    // Step 3: Stop AF (mode 3 may auto-stop, but ensure it's stopped)
    ret = StopAF();
    if (ret != DEVICE_OK)
        LogMessage("Warning: Failed to stop AF, but continuing with offset application");

    // Step 4: Apply stored offset to Focus Drive
    long currentZPos = hub->GetModel()->GetPosition(EvidentIX85Win::DeviceType_Focus);
    long measuredZOffset = hub->GetModel()->GetMeasuredZOffset();
    long targetZPos = currentZPos + measuredZOffset;

    std::ostringstream logMsg;
    logMsg << "Applying Z-offset: " << measuredZOffset <<
              " steps (from " << currentZPos <<
              " to " << targetZPos << ")";
    LogMessage(logMsg.str().c_str());

    // Step 5: Move Focus Drive to new position
    // Sets the Focus drive busy flag
    return hub->SetFocusPositionSteps(targetZPos);
}

std::string EvidentAutofocus::GetAFStatusString(int status)
{
    switch (status)
    {
        case 0: return "Stop";
        case 1: return "Focus";
        case 2: return "Track";
        case 3: return "Wait";
        case 4: return "Search";
        default: return "Unknown";
    }
}

EvidentHubWin* EvidentAutofocus::GetHub()
{
    MM::Hub* hub = GetParentHub();
    if (!hub)
        return nullptr;
    return dynamic_cast<EvidentHubWin*>(hub);
}

int EvidentAutofocus::EnableNotifications(bool enable)
{
    EvidentHubWin* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    return hub->EnableNotification(CMD_AF_STATUS, enable);
}

int EvidentAutofocus::OnAFStatus(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        // Query current status
        EvidentHubWin* hub = GetHub();
        if (hub)
        {
            std::string cmd = BuildQuery(CMD_AF_STATUS);
            std::string response;
            int ret = hub->ExecuteCommand(cmd, response);
            if (ret == DEVICE_OK)
            {
                std::vector<std::string> params = ParseParameters(response);
                if (params.size() > 0 && params[0] != "X")
                {
                    afStatus_ = ParseIntParameter(params[0]);
                }
            }
        }
        pProp->Set(GetAFStatusString(afStatus_).c_str());
    }
    return DEVICE_OK;
}

int EvidentAutofocus::OnNearLimit(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    EvidentHubWin* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    if (eAct == MM::BeforeGet)
    {
        pProp->Set(nearLimit_ * 0.01);  // Convert 0.01um units to um
    }
    else if (eAct == MM::AfterSet)
    {
        double val;
        pProp->Get(val);
        long newNear = static_cast<long>(val * 100);  // Convert um to 0.01um units

        // Validate: Near limit must be > Far limit
        if (newNear <= farLimit_)
        {
            return ERR_INVALID_PARAMETER;
        }

        std::string cmd = BuildCommand(CMD_AF_LIMIT, static_cast<int>(newNear), static_cast<int>(farLimit_));
        std::string response;
        int ret = hub->ExecuteCommand(cmd, response);
        if (ret != DEVICE_OK)
            return ret;

        if (!IsPositiveAck(response, CMD_AF_LIMIT))
            return ERR_NEGATIVE_ACK;

        nearLimit_ = newNear;

        // Mark that ZDC needs re-initialization (deferred until next AF operation)
        zdcInitNeeded_ = true;
    }
    return DEVICE_OK;
}

int EvidentAutofocus::OnFarLimit(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    EvidentHubWin* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    if (eAct == MM::BeforeGet)
    {
        pProp->Set(farLimit_ * 0.01);  // Convert 0.01um units to um
    }
    else if (eAct == MM::AfterSet)
    {
        double val;
        pProp->Get(val);
        long newFar = static_cast<long>(val * 100);  // Convert um to 0.01um units

        // Validate: Near limit must be > Far limit
        if (nearLimit_ <= newFar)
        {
            return ERR_INVALID_PARAMETER;
        }

        std::string cmd = BuildCommand(CMD_AF_LIMIT, static_cast<int>(nearLimit_), static_cast<int>(newFar));
        std::string response;
        int ret = hub->ExecuteCommand(cmd, response);
        if (ret != DEVICE_OK)
            return ret;

        if (!IsPositiveAck(response, CMD_AF_LIMIT))
            return ERR_NEGATIVE_ACK;

        farLimit_ = newFar;

        // Mark that ZDC needs re-initialization (deferred until next AF operation)
        zdcInitNeeded_ = true;
    }
    return DEVICE_OK;
}

int EvidentAutofocus::OnCoverSlipType(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    EvidentHubWin* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    if (eAct == MM::BeforeGet)
    {
        std::string cmd = BuildQuery(CMD_COVERSLIP_TYPE);
        std::string response;
        int ret = hub->ExecuteCommand(cmd, response);
        if (ret == DEVICE_OK)
        {
            std::vector<std::string> params = ParseParameters(response);
            if (params.size() > 0)
            {
                int type = ParseIntParameter(params[0]);
                pProp->Set(type == 1 ? "Glass" : "Plastic");
            }
        }
    }
    else if (eAct == MM::AfterSet)
    {
        std::string val;
        pProp->Get(val);
        int type = (val == "Glass") ? 1 : 2;

        std::string cmd = BuildCommand(CMD_COVERSLIP_TYPE, type);
        std::string response;
        int ret = hub->ExecuteCommand(cmd, response);
        if (ret != DEVICE_OK)
            return ret;

        if (!IsPositiveAck(response, CMD_COVERSLIP_TYPE))
            return ERR_NEGATIVE_ACK;

        // Track coverslip type change
        lastCoverslipType_ = type;

        // Mark that ZDC needs re-initialization (deferred until next AF operation)
        zdcInitNeeded_ = true;
    }
    return DEVICE_OK;
}

int EvidentAutofocus::OnCoverSlipThicknessGlass(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    EvidentHubWin* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    if (eAct == MM::BeforeGet)
    {
        std::string cmd = BuildQuery(CMD_COVERSLIP_THICKNESS);
        std::string response;
        int ret = hub->ExecuteCommand(cmd, response);
        if (ret == DEVICE_OK)
        {
            std::vector<std::string> params = ParseParameters(response);
            if (params.size() >= 1 && params[0] != "X")
            {
                int thickness = ParseIntParameter(params[0]);
                pProp->Set(thickness * 10.0);  // Convert 10um units to um
            }
        }
    }
    else if (eAct == MM::AfterSet)
    {
        double val;
        pProp->Get(val);
        int thickness = static_cast<int>(val / 10.0);  // Convert um to 10um units

        // Query current plastic thickness to preserve it
        std::string cmd = BuildQuery(CMD_COVERSLIP_THICKNESS);
        std::string response;
        int ret = hub->ExecuteCommand(cmd, response);
        int plasticThickness = 100;  // Default
        if (ret == DEVICE_OK)
        {
            std::vector<std::string> params = ParseParameters(response);
            if (params.size() >= 2 && params[1] != "X")
            {
                plasticThickness = ParseIntParameter(params[1]);
            }
        }

        cmd = BuildCommand(CMD_COVERSLIP_THICKNESS, thickness, plasticThickness);
        ret = hub->ExecuteCommand(cmd, response);
        if (ret != DEVICE_OK)
            return ret;

        if (!IsPositiveAck(response, CMD_COVERSLIP_THICKNESS))
            return ERR_NEGATIVE_ACK;
    }
    return DEVICE_OK;
}

int EvidentAutofocus::OnCoverSlipThicknessPlastic(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    EvidentHubWin* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    if (eAct == MM::BeforeGet)
    {
        std::string cmd = BuildQuery(CMD_COVERSLIP_THICKNESS);
        std::string response;
        int ret = hub->ExecuteCommand(cmd, response);
        if (ret == DEVICE_OK)
        {
            std::vector<std::string> params = ParseParameters(response);
            if (params.size() >= 2 && params[1] != "X")
            {
                int thickness = ParseIntParameter(params[1]);
                pProp->Set(thickness * 10.0);  // Convert 10um units to um
            }
        }
    }
    else if (eAct == MM::AfterSet)
    {
        double val;
        pProp->Get(val);
        int thickness = static_cast<int>(val / 10.0);  // Convert um to 10um units

        // Query current glass thickness to preserve it
        std::string cmd = BuildQuery(CMD_COVERSLIP_THICKNESS);
        std::string response;
        int ret = hub->ExecuteCommand(cmd, response);
        int glassThickness = 17;  // Default
        if (ret == DEVICE_OK)
        {
            std::vector<std::string> params = ParseParameters(response);
            if (params.size() >= 1 && params[0] != "X")
            {
                glassThickness = ParseIntParameter(params[0]);
            }
        }

        cmd = BuildCommand(CMD_COVERSLIP_THICKNESS, glassThickness, thickness);
        ret = hub->ExecuteCommand(cmd, response);
        if (ret != DEVICE_OK)
            return ret;

        if (!IsPositiveAck(response, CMD_COVERSLIP_THICKNESS))
            return ERR_NEGATIVE_ACK;
    }
    return DEVICE_OK;
}

int EvidentAutofocus::OnDICMode(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    EvidentHubWin* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    if (eAct == MM::BeforeGet)
    {
        std::string cmd = BuildQuery(CMD_AF_DIC);
        std::string response;
        int ret = hub->ExecuteCommand(cmd, response);
        if (ret == DEVICE_OK)
        {
            std::vector<std::string> params = ParseParameters(response);
            if (params.size() > 0)
            {
                int mode = ParseIntParameter(params[0]);
                pProp->Set(mode == 0 ? "Off" : "On");
            }
        }
    }
    else if (eAct == MM::AfterSet)
    {
        std::string val;
        pProp->Get(val);
        int mode = (val == "Off") ? 0 : 1;

        std::string cmd = BuildCommand(CMD_AF_DIC, mode);
        std::string response;
        int ret = hub->ExecuteCommand(cmd, response);
        if (ret != DEVICE_OK)
            return ret;

        if (!IsPositiveAck(response, CMD_AF_DIC))
            return ERR_NEGATIVE_ACK;
    }
    return DEVICE_OK;
}

int EvidentAutofocus::OnBuzzerSuccess(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    EvidentHubWin* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    if (eAct == MM::BeforeGet)
    {
        std::string cmd = BuildQuery(CMD_AF_BUZZER);
        std::string response;
        int ret = hub->ExecuteCommand(cmd, response);
        if (ret == DEVICE_OK)
        {
            std::vector<std::string> params = ParseParameters(response);
            if (params.size() >= 1)
            {
                int success = ParseIntParameter(params[0]);
                pProp->Set(success == 0 ? "Off" : "On");
            }
        }
    }
    else if (eAct == MM::AfterSet)
    {
        std::string val;
        pProp->Get(val);
        int success = (val == "Off") ? 0 : 1;

        // Query current failure setting to preserve it
        std::string cmd = BuildQuery(CMD_AF_BUZZER);
        std::string response;
        int ret = hub->ExecuteCommand(cmd, response);
        int failure = 1;  // Default
        if (ret == DEVICE_OK)
        {
            std::vector<std::string> params = ParseParameters(response);
            if (params.size() >= 2)
            {
                failure = ParseIntParameter(params[1]);
            }
        }

        cmd = BuildCommand(CMD_AF_BUZZER, success, failure);
        ret = hub->ExecuteCommand(cmd, response);
        if (ret != DEVICE_OK)
            return ret;

        if (!IsPositiveAck(response, CMD_AF_BUZZER))
            return ERR_NEGATIVE_ACK;
    }
    return DEVICE_OK;
}

int EvidentAutofocus::OnBuzzerFailure(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    EvidentHubWin* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    if (eAct == MM::BeforeGet)
    {
        std::string cmd = BuildQuery(CMD_AF_BUZZER);
        std::string response;
        int ret = hub->ExecuteCommand(cmd, response);
        if (ret == DEVICE_OK)
        {
            std::vector<std::string> params = ParseParameters(response);
            if (params.size() >= 2)
            {
                int failure = ParseIntParameter(params[1]);
                pProp->Set(failure == 0 ? "Off" : "On");
            }
        }
    }
    else if (eAct == MM::AfterSet)
    {
        std::string val;
        pProp->Get(val);
        int failure = (val == "Off") ? 0 : 1;

        // Query current success setting to preserve it
        std::string cmd = BuildQuery(CMD_AF_BUZZER);
        std::string response;
        int ret = hub->ExecuteCommand(cmd, response);
        int success = 1;  // Default
        if (ret == DEVICE_OK)
        {
            std::vector<std::string> params = ParseParameters(response);
            if (params.size() >= 1)
            {
                success = ParseIntParameter(params[0]);
            }
        }

        cmd = BuildCommand(CMD_AF_BUZZER, success, failure);
        ret = hub->ExecuteCommand(cmd, response);
        if (ret != DEVICE_OK)
            return ret;

        if (!IsPositiveAck(response, CMD_AF_BUZZER))
            return ERR_NEGATIVE_ACK;
    }
    return DEVICE_OK;
}

int EvidentAutofocus::OnWorkflowMode(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        std::string mode;
        switch (workflowMode_)
        {
            case 1: mode = "Measure-Offset"; break;
            case 2: mode = "Find-Focus-With-Offset"; break;
            case 3: mode = "Continuous-Focus"; break;
            default: mode = "Continuous-Focus"; break;
        }
        pProp->Set(mode.c_str());
    }
    else if (eAct == MM::AfterSet)
    {
        std::string val;
        pProp->Get(val);
        if (val == "Measure-Offset")
            workflowMode_ = 1;
        else if (val == "Find-Focus-With-Offset")
            workflowMode_ = 2;
        else if (val == "Continuous-Focus")
            workflowMode_ = 3;
    }
    return DEVICE_OK;
}

int EvidentAutofocus::OnMeasuredFocusOffset(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        double offset;
        GetMeasuredZOffset(offset);  // Already converts steps to µm
        pProp->Set(offset);
    }
    else if (eAct == MM::AfterSet)
    {
        double offset;
        pProp->Get(offset);
        SetMeasuredZOffset(offset);  // Already converts µm to steps

        // Notify core of property change
        std::ostringstream valStr;
        valStr << offset;
        OnPropertyChanged("Measured-Focus-Offset-um", valStr.str().c_str());

        // Log the change
        std::ostringstream logMsg;
        logMsg << "Measured focus offset set to " << offset << " um";
        LogMessage(logMsg.str().c_str());
    }
    return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// EvidentOffsetLens - Offset Lens Implementation
///////////////////////////////////////////////////////////////////////////////

EvidentOffsetLens::EvidentOffsetLens() :
    initialized_(false),
    name_(g_OffsetLensDeviceName),
    stepSizeUm_(OFFSET_LENS_STEP_SIZE_UM)
{
    InitializeDefaultErrorMessages();
    SetErrorText(ERR_DEVICE_NOT_AVAILABLE, "Offset lens not available on this microscope");

    CreateHubIDProperty();
}

EvidentOffsetLens::~EvidentOffsetLens()
{
    Shutdown();
}

void EvidentOffsetLens::GetName(char* pszName) const
{
    CDeviceUtils::CopyLimitedString(pszName, name_.c_str());
}

int EvidentOffsetLens::Initialize()
{
    if (initialized_)
        return DEVICE_OK;

    EvidentHubWin* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    if (!hub->IsDevicePresent(EvidentIX85Win::DeviceType_OffsetLens))
        return ERR_DEVICE_NOT_AVAILABLE;

    // Query initial position
    std::string cmd = BuildQuery(CMD_OFFSET_LENS_POSITION);
    std::string response;
    int ret = hub->ExecuteCommand(cmd, response);
    if (ret == DEVICE_OK)
    {
        std::vector<std::string> params = ParseParameters(response);
        if (params.size() > 0 && params[0] != "X")
        {
            long pos = ParseIntParameter(params[0]);
            hub->GetModel()->SetPosition(EvidentIX85Win::DeviceType_OffsetLens, pos);
        }
    }

    // Create Position property
    CPropertyAction* pAct = new CPropertyAction(this, &EvidentOffsetLens::OnPosition);
    ret = CreateProperty("Position (um)", "0", MM::Float, false, pAct);
    if (ret != DEVICE_OK)
        return ret;

    // Enable notifications
    EnableNotifications(true);

    hub->RegisterDeviceAsUsed(EvidentIX85Win::DeviceType_OffsetLens, this);
    initialized_ = true;
    return DEVICE_OK;
}

int EvidentOffsetLens::Shutdown()
{
    if (initialized_)
    {
        EvidentHubWin* hub = GetHub();
        if (hub)
        {
            EnableNotifications(false);
            hub->UnRegisterDeviceAsUsed(EvidentIX85Win::DeviceType_OffsetLens);
        }
        initialized_ = false;
    }
    return DEVICE_OK;
}

bool EvidentOffsetLens::Busy()
{
    EvidentHubWin* hub = GetHub();
    if (!hub)
        return false;

    return hub->GetModel()->IsBusy(EvidentIX85Win::DeviceType_OffsetLens);
}

int EvidentOffsetLens::SetPositionUm(double pos)
{
    EvidentHubWin* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    // Convert μm to steps
    long steps = static_cast<long>(pos / stepSizeUm_);

    // Clamp to limits
    if (steps < OFFSET_LENS_MIN_POS) steps = OFFSET_LENS_MIN_POS;
    if (steps > OFFSET_LENS_MAX_POS) steps = OFFSET_LENS_MAX_POS;

    hub->GetModel()->SetBusy(EvidentIX85Win::DeviceType_OffsetLens, true);

    std::string cmd = BuildCommand(CMD_OFFSET_LENS_GOTO, static_cast<int>(steps));
    std::string response;
    int ret = hub->ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
    {
        hub->GetModel()->SetBusy(EvidentIX85Win::DeviceType_OffsetLens, false);
        return ret;
    }

    if (!IsPositiveAck(response, CMD_OFFSET_LENS_GOTO))
    {
        hub->GetModel()->SetBusy(EvidentIX85Win::DeviceType_OffsetLens, false);
        return ERR_NEGATIVE_ACK;
    }

    hub->GetModel()->SetPosition(EvidentIX85Win::DeviceType_OffsetLens, steps);
    hub->GetModel()->SetBusy(EvidentIX85Win::DeviceType_OffsetLens, false);

    return DEVICE_OK;
}

int EvidentOffsetLens::GetPositionUm(double& pos)
{
    EvidentHubWin* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    long steps = hub->GetModel()->GetPosition(EvidentIX85Win::DeviceType_OffsetLens);
    pos = steps * stepSizeUm_;
    return DEVICE_OK;
}

int EvidentOffsetLens::SetPositionSteps(long steps)
{
    return SetPositionUm(steps * stepSizeUm_);
}

int EvidentOffsetLens::GetPositionSteps(long& steps)
{
    EvidentHubWin* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    steps = hub->GetModel()->GetPosition(EvidentIX85Win::DeviceType_OffsetLens);
    return DEVICE_OK;
}

int EvidentOffsetLens::SetOrigin()
{
    // Not supported - origin is factory set
    return DEVICE_UNSUPPORTED_COMMAND;
}

int EvidentOffsetLens::GetLimits(double& lower, double& upper)
{
    lower = OFFSET_LENS_MIN_POS * stepSizeUm_;
    upper = OFFSET_LENS_MAX_POS * stepSizeUm_;
    return DEVICE_OK;
}

int EvidentOffsetLens::OnPosition(MM::PropertyBase* pProp, MM::ActionType eAct)
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

EvidentHubWin* EvidentOffsetLens::GetHub()
{
    MM::Hub* hub = GetParentHub();
    if (!hub)
        return nullptr;
    return dynamic_cast<EvidentHubWin*>(hub);
}

int EvidentOffsetLens::EnableNotifications(bool enable)
{
    EvidentHubWin* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    return hub->EnableNotification(CMD_OFFSET_LENS_NOTIFY, enable);
}


///////////////////////////////////////////////////////////////////////////////
// EvidentZDCVirtualOffset - Virtual Offset Implementation
///////////////////////////////////////////////////////////////////////////////


 EvidentZDCVirtualOffset::EvidentZDCVirtualOffset() :
    initialized_(false),
    name_(g_ZDCVirtualOffsetDeviceName),
    // since we operate on the Focus drive, the step size is the same as Focus
    stepSizeUm_(FOCUS_STEP_SIZE_UM)
{
    InitializeDefaultErrorMessages();
    SetErrorText(ERR_DEVICE_NOT_AVAILABLE, "Offset lens not available on this microscope");

    CreateHubIDProperty();
}

EvidentZDCVirtualOffset::~EvidentZDCVirtualOffset()
{
    Shutdown();
}

void EvidentZDCVirtualOffset::GetName(char* pszName) const
{
    CDeviceUtils::CopyLimitedString(pszName, name_.c_str());
}

int EvidentZDCVirtualOffset::Initialize()
{
    if (initialized_)
        return DEVICE_OK;

    EvidentHubWin* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    if (!hub->IsDevicePresent(EvidentIX85Win::DeviceType_Autofocus))
        return ERR_DEVICE_NOT_AVAILABLE;

    // Get initial position from model
    long offset = hub->GetModel()->GetMeasuredZOffset();
    double offsetUm = offset * stepSizeUm_;

    // Create Position property
    CPropertyAction* pAct = new CPropertyAction(this, &EvidentZDCVirtualOffset::OnPosition);
    int ret = CreateProperty(MM::g_Keyword_Position, std::to_string(offsetUm).c_str(), MM::Float, false, pAct);
    if (ret != DEVICE_OK)
        return ret;

    hub->RegisterDeviceAsUsed(DeviceType_ZDCVirtualOffset, this);

    initialized_ = true;
    return DEVICE_OK;
}

int EvidentZDCVirtualOffset::Shutdown()
{
    if (initialized_)
    {
        // Unregister from hub
        EvidentHubWin* hub = GetHub();
        if (hub)
            hub->UnRegisterDeviceAsUsed(DeviceType_ZDCVirtualOffset);

        initialized_ = false;
    }
    return DEVICE_OK;
}

bool EvidentZDCVirtualOffset::Busy()
{
    // Virtual offset is never busy
    return false;
}

int EvidentZDCVirtualOffset::SetPositionUm(double pos)
{
    EvidentHubWin* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    // Convert μm to steps
    long steps = static_cast<long>(pos / stepSizeUm_);

    // Update the model
    hub->GetModel()->SetMeasuredZOffset(steps);

    // Notify both devices that the offset has changed
    hub->NotifyMeasuredZOffsetChanged(steps);

    return DEVICE_OK;
}

int EvidentZDCVirtualOffset::GetPositionUm(double& pos)
{
    EvidentHubWin* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    long steps = hub->GetModel()->GetMeasuredZOffset();
    pos = steps * stepSizeUm_;
    return DEVICE_OK;
}

int EvidentZDCVirtualOffset::SetPositionSteps(long steps)
{
    return SetPositionUm(steps * stepSizeUm_);
}

int EvidentZDCVirtualOffset::GetPositionSteps(long& steps)
{
    EvidentHubWin* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    steps = hub->GetModel()->GetMeasuredZOffset();
    return DEVICE_OK;
}

int EvidentZDCVirtualOffset::SetOrigin()
{
    // Set current position as zero offset
    EvidentHubWin* hub = GetHub();
    if (!hub)
        return DEVICE_ERR;

    hub->GetModel()->SetMeasuredZOffset(0);
    hub->NotifyMeasuredZOffsetChanged(0);

    return DEVICE_OK;
}

int EvidentZDCVirtualOffset::GetLimits(double& lower, double& upper)
{
    // Use focus limits as reference
    lower = FOCUS_MIN_POS * stepSizeUm_;
    upper = FOCUS_MAX_POS * stepSizeUm_;
    return DEVICE_OK;
}

int EvidentZDCVirtualOffset::OnPosition(MM::PropertyBase* pProp, MM::ActionType eAct)
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

EvidentHubWin* EvidentZDCVirtualOffset::GetHub()
{
    MM::Hub* hub = GetParentHub();
    if (!hub)
        return nullptr;
    return dynamic_cast<EvidentHubWin*>(hub);
}

void EvidentZDCVirtualOffset::UpdateMeasuredZOffset(long offsetSteps)
{
    // Update property to reflect new offset
    double offsetUm = offsetSteps * stepSizeUm_;
    std::ostringstream valStr;
    valStr << offsetUm;
    OnPropertyChanged(MM::g_Keyword_Position, valStr.str().c_str());
    OnStagePositionChanged(offsetUm);
}
