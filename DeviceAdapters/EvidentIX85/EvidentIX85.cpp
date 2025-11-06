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
const char* g_FocusDeviceName = "EvidentIX85-Focus";
const char* g_NosepieceDeviceName = "EvidentIX85-Nosepiece";
const char* g_MagnificationDeviceName = "EvidentIX85-Magnification";
const char* g_LightPathDeviceName = "EvidentIX85-LightPath";
const char* g_CondenserTurretDeviceName = "EvidentIX85-CondenserTurret";
const char* g_DIAShutterDeviceName = "EvidentIX85-DIAShutter";
const char* g_EPIShutter1DeviceName = "EvidentIX85-EPIShutter1";
const char* g_MirrorUnit1DeviceName = "EvidentIX85-MirrorUnit1";
const char* g_PolarizerDeviceName = "EvidentIX85-Polarizer";
const char* g_DICPrismDeviceName = "EvidentIX85-DICPrism";
const char* g_EPINDDeviceName = "EvidentIX85-EPIND";
const char* g_CorrectionCollarDeviceName = "EvidentIX85-CorrectionCollar";

///////////////////////////////////////////////////////////////////////////////
// MODULE_API - Exported MMDevice interface
///////////////////////////////////////////////////////////////////////////////

MODULE_API void InitializeModuleData()
{
    RegisterDevice(g_HubDeviceName, MM::HubDevice, "Evident IX85 Hub");
    RegisterDevice(g_FocusDeviceName, MM::StageDevice, "Evident IX85 Focus Drive");
    RegisterDevice(g_NosepieceDeviceName, MM::StateDevice, "Evident IX85 Nosepiece");
    RegisterDevice(g_MagnificationDeviceName, MM::StateDevice, "Evident IX85 Magnification Changer");
    RegisterDevice(g_LightPathDeviceName, MM::StateDevice, "Evident IX85 Light Path");
    RegisterDevice(g_CondenserTurretDeviceName, MM::StateDevice, "Evident IX85 Condenser Turret");
    RegisterDevice(g_DIAShutterDeviceName, MM::ShutterDevice, "Evident IX85 DIA Shutter");
    RegisterDevice(g_EPIShutter1DeviceName, MM::ShutterDevice, "Evident IX85 EPI Shutter 1");
    RegisterDevice(g_MirrorUnit1DeviceName, MM::StateDevice, "Evident IX85 Mirror Unit 1");
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
    else if (strcmp(deviceName, g_MirrorUnit1DeviceName) == 0)
        return new EvidentMirrorUnit1();
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

    std::string cmd = BuildCommand(CMD_FOCUS_GOTO, static_cast<int>(steps));
    std::string response;
    int ret = hub->ExecuteCommand(cmd, response);
    if (ret != DEVICE_OK)
        return ret;

    if (!IsPositiveAck(response, CMD_FOCUS_GOTO))
        return ERR_NEGATIVE_ACK;

    hub->GetModel()->SetTargetPosition(DeviceType_Focus, steps);
    hub->GetModel()->SetBusy(DeviceType_Focus, true);

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
    numPos_(NOSEPIECE_MAX_POS)
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

        // Convert from 0-based to 1-based
        std::string cmd = BuildCommand(CMD_NOSEPIECE, static_cast<int>(pos + 1));
        std::string response;
        int ret = hub->ExecuteCommand(cmd, response);
        if (ret != DEVICE_OK)
            return ret;

        if (!IsPositiveAck(response, CMD_NOSEPIECE))
            return ERR_NEGATIVE_ACK;

        hub->GetModel()->SetTargetPosition(DeviceType_Nosepiece, pos + 1);
        hub->GetModel()->SetBusy(DeviceType_Nosepiece, true);
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

///////////////////////////////////////////////////////////////////////////////
// EvidentMagnification - Magnification Changer Implementation
///////////////////////////////////////////////////////////////////////////////

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

    // Create properties
    CPropertyAction* pAct = new CPropertyAction(this, &EvidentMagnification::OnState);
    int ret = CreateProperty(MM::g_Keyword_State, "0", MM::Integer, true, pAct);
    if (ret != DEVICE_OK)
        return ret;

    SetPropertyLimits(MM::g_Keyword_State, 0, numPos_ - 1);

    // Create label property
    pAct = new CPropertyAction(this, &CStateDeviceBase::OnLabel);
    ret = CreateProperty(MM::g_Keyword_Label, "", MM::String, true, pAct);
    if (ret != DEVICE_OK)
        return ret;

    // Define labels (1x, 1.6x, 2x)
    SetPositionLabel(0, "1x");
    SetPositionLabel(1, "1.6x");
    SetPositionLabel(2, "2x");

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

unsigned long EvidentMagnification::GetNumberOfPositions() const
{
    return numPos_;
}

int EvidentMagnification::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        EvidentHub* hub = GetHub();
        if (!hub)
            return DEVICE_ERR;

        long pos = hub->GetModel()->GetPosition(DeviceType_Magnification);
        if (pos < 0)
            return ERR_POSITION_UNKNOWN;

        // Convert from 1-based to 0-based
        pProp->Set(pos - 1);
    }
    else if (eAct == MM::AfterSet)
    {
       // nothing to do, this is a read-only property
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

// Placeholder implementations for other devices
// These can be expanded similarly to the above

EvidentLightPath::EvidentLightPath() : initialized_(false), name_(g_LightPathDeviceName) { CreateHubIDProperty(); }
EvidentLightPath::~EvidentLightPath() { Shutdown(); }
void EvidentLightPath::GetName(char* pszName) const { CDeviceUtils::CopyLimitedString(pszName, name_.c_str()); }
int EvidentLightPath::Initialize() { initialized_ = true; return DEVICE_OK; }
int EvidentLightPath::Shutdown() { initialized_ = false; return DEVICE_OK; }
bool EvidentLightPath::Busy() { return false; }
unsigned long EvidentLightPath::GetNumberOfPositions() const { return 4; }
int EvidentLightPath::OnState(MM::PropertyBase*, MM::ActionType) { return DEVICE_OK; }
EvidentHub* EvidentLightPath::GetHub() { return dynamic_cast<EvidentHub*>(GetParentHub()); }

EvidentCondenserTurret::EvidentCondenserTurret() : initialized_(false), name_(g_CondenserTurretDeviceName), numPos_(6) { CreateHubIDProperty(); }
EvidentCondenserTurret::~EvidentCondenserTurret() { Shutdown(); }
void EvidentCondenserTurret::GetName(char* pszName) const { CDeviceUtils::CopyLimitedString(pszName, name_.c_str()); }
int EvidentCondenserTurret::Initialize() { initialized_ = true; return DEVICE_OK; }
int EvidentCondenserTurret::Shutdown() { initialized_ = false; return DEVICE_OK; }
bool EvidentCondenserTurret::Busy() { return false; }
unsigned long EvidentCondenserTurret::GetNumberOfPositions() const { return numPos_; }
int EvidentCondenserTurret::OnState(MM::PropertyBase*, MM::ActionType) { return DEVICE_OK; }
EvidentHub* EvidentCondenserTurret::GetHub() { return dynamic_cast<EvidentHub*>(GetParentHub()); }
int EvidentCondenserTurret::EnableNotifications(bool) { return DEVICE_OK; }

EvidentDIAShutter::EvidentDIAShutter() : initialized_(false), name_(g_DIAShutterDeviceName) { CreateHubIDProperty(); }
EvidentDIAShutter::~EvidentDIAShutter() { Shutdown(); }
void EvidentDIAShutter::GetName(char* pszName) const { CDeviceUtils::CopyLimitedString(pszName, name_.c_str()); }
int EvidentDIAShutter::Initialize() { initialized_ = true; return DEVICE_OK; }
int EvidentDIAShutter::Shutdown() { initialized_ = false; return DEVICE_OK; }
bool EvidentDIAShutter::Busy() { return false; }
int EvidentDIAShutter::SetOpen(bool) { return DEVICE_OK; }
int EvidentDIAShutter::GetOpen(bool& open) { open = false; return DEVICE_OK; }
int EvidentDIAShutter::Fire(double) { return DEVICE_OK; }
int EvidentDIAShutter::OnState(MM::PropertyBase*, MM::ActionType) { return DEVICE_OK; }
EvidentHub* EvidentDIAShutter::GetHub() { return dynamic_cast<EvidentHub*>(GetParentHub()); }

EvidentEPIShutter1::EvidentEPIShutter1() : initialized_(false), name_(g_EPIShutter1DeviceName) { CreateHubIDProperty(); }
EvidentEPIShutter1::~EvidentEPIShutter1() { Shutdown(); }
void EvidentEPIShutter1::GetName(char* pszName) const { CDeviceUtils::CopyLimitedString(pszName, name_.c_str()); }
int EvidentEPIShutter1::Initialize() { initialized_ = true; return DEVICE_OK; }
int EvidentEPIShutter1::Shutdown() { initialized_ = false; return DEVICE_OK; }
bool EvidentEPIShutter1::Busy() { return false; }
int EvidentEPIShutter1::SetOpen(bool) { return DEVICE_OK; }
int EvidentEPIShutter1::GetOpen(bool& open) { open = false; return DEVICE_OK; }
int EvidentEPIShutter1::Fire(double) { return DEVICE_OK; }
int EvidentEPIShutter1::OnState(MM::PropertyBase*, MM::ActionType) { return DEVICE_OK; }
EvidentHub* EvidentEPIShutter1::GetHub() { return dynamic_cast<EvidentHub*>(GetParentHub()); }

EvidentMirrorUnit1::EvidentMirrorUnit1() : initialized_(false), name_(g_MirrorUnit1DeviceName), numPos_(6) { CreateHubIDProperty(); }
EvidentMirrorUnit1::~EvidentMirrorUnit1() { Shutdown(); }
void EvidentMirrorUnit1::GetName(char* pszName) const { CDeviceUtils::CopyLimitedString(pszName, name_.c_str()); }
int EvidentMirrorUnit1::Initialize() { initialized_ = true; return DEVICE_OK; }
int EvidentMirrorUnit1::Shutdown() { initialized_ = false; return DEVICE_OK; }
bool EvidentMirrorUnit1::Busy() { return false; }
unsigned long EvidentMirrorUnit1::GetNumberOfPositions() const { return numPos_; }
int EvidentMirrorUnit1::OnState(MM::PropertyBase*, MM::ActionType) { return DEVICE_OK; }
EvidentHub* EvidentMirrorUnit1::GetHub() { return dynamic_cast<EvidentHub*>(GetParentHub()); }
int EvidentMirrorUnit1::EnableNotifications(bool) { return DEVICE_OK; }

EvidentPolarizer::EvidentPolarizer() : initialized_(false), name_(g_PolarizerDeviceName), numPos_(6) { CreateHubIDProperty(); }
EvidentPolarizer::~EvidentPolarizer() { Shutdown(); }
void EvidentPolarizer::GetName(char* pszName) const { CDeviceUtils::CopyLimitedString(pszName, name_.c_str()); }
int EvidentPolarizer::Initialize() { initialized_ = true; return DEVICE_OK; }
int EvidentPolarizer::Shutdown() { initialized_ = false; return DEVICE_OK; }
bool EvidentPolarizer::Busy() { return false; }
unsigned long EvidentPolarizer::GetNumberOfPositions() const { return numPos_; }
int EvidentPolarizer::OnState(MM::PropertyBase*, MM::ActionType) { return DEVICE_OK; }
EvidentHub* EvidentPolarizer::GetHub() { return dynamic_cast<EvidentHub*>(GetParentHub()); }
int EvidentPolarizer::EnableNotifications(bool) { return DEVICE_OK; }

EvidentDICPrism::EvidentDICPrism() : initialized_(false), name_(g_DICPrismDeviceName), numPos_(6) { CreateHubIDProperty(); }
EvidentDICPrism::~EvidentDICPrism() { Shutdown(); }
void EvidentDICPrism::GetName(char* pszName) const { CDeviceUtils::CopyLimitedString(pszName, name_.c_str()); }
int EvidentDICPrism::Initialize() { initialized_ = true; return DEVICE_OK; }
int EvidentDICPrism::Shutdown() { initialized_ = false; return DEVICE_OK; }
bool EvidentDICPrism::Busy() { return false; }
unsigned long EvidentDICPrism::GetNumberOfPositions() const { return numPos_; }
int EvidentDICPrism::OnState(MM::PropertyBase*, MM::ActionType) { return DEVICE_OK; }
EvidentHub* EvidentDICPrism::GetHub() { return dynamic_cast<EvidentHub*>(GetParentHub()); }

EvidentEPIND::EvidentEPIND() : initialized_(false), name_(g_EPINDDeviceName), numPos_(6) { CreateHubIDProperty(); }
EvidentEPIND::~EvidentEPIND() { Shutdown(); }
void EvidentEPIND::GetName(char* pszName) const { CDeviceUtils::CopyLimitedString(pszName, name_.c_str()); }
int EvidentEPIND::Initialize() { initialized_ = true; return DEVICE_OK; }
int EvidentEPIND::Shutdown() { initialized_ = false; return DEVICE_OK; }
bool EvidentEPIND::Busy() { return false; }
unsigned long EvidentEPIND::GetNumberOfPositions() const { return numPos_; }
int EvidentEPIND::OnState(MM::PropertyBase*, MM::ActionType) { return DEVICE_OK; }
EvidentHub* EvidentEPIND::GetHub() { return dynamic_cast<EvidentHub*>(GetParentHub()); }

EvidentCorrectionCollar::EvidentCorrectionCollar() : initialized_(false), name_(g_CorrectionCollarDeviceName) { CreateHubIDProperty(); }
EvidentCorrectionCollar::~EvidentCorrectionCollar() { Shutdown(); }
void EvidentCorrectionCollar::GetName(char* pszName) const { CDeviceUtils::CopyLimitedString(pszName, name_.c_str()); }
int EvidentCorrectionCollar::Initialize() { initialized_ = true; return DEVICE_OK; }
int EvidentCorrectionCollar::Shutdown() { initialized_ = false; return DEVICE_OK; }
bool EvidentCorrectionCollar::Busy() { return false; }
int EvidentCorrectionCollar::OnPosition(MM::PropertyBase*, MM::ActionType) { return DEVICE_OK; }
EvidentHub* EvidentCorrectionCollar::GetHub() { return dynamic_cast<EvidentHub*>(GetParentHub()); }
