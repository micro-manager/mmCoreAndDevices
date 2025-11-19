///////////////////////////////////////////////////////////////////////////////
// FILE:          EvidentIX85Win.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Evident IX85 microscope device classes
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

#pragma once

#include "DeviceBase.h"
#include "EvidentHubWin.h"
#include "EvidentModelWin.h"
#include "EvidentProtocolWin.h"

//////////////////////////////////////////////////////////////////////////////
// Focus Drive (Z-Stage)
//////////////////////////////////////////////////////////////////////////////

class EvidentHubWin;

class EvidentFocus : public CStageBase<EvidentFocus>
{
public:
    EvidentFocus();
    ~EvidentFocus();


    // MMDevice API
    int Initialize();
    int Shutdown();
    void GetName(char* pszName) const;
    bool Busy();

    // Stage API
    int SetPositionUm(double pos);
    int GetPositionUm(double& pos);
    int SetPositionSteps(long steps);
    int GetPositionSteps(long& steps);
    int SetOrigin();
    int GetLimits(double& lower, double& upper);
    int IsStageSequenceable(bool& isSequenceable) const { isSequenceable = false; return DEVICE_OK; };
    bool IsContinuousFocusDrive() const { return false; };

    // Action interface
    int OnPosition(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnSpeed(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
    EvidentHubWin* GetHub();
    int EnableNotifications(bool enable);

    bool initialized_;
    std::string name_;
    double stepSizeUm_;
};

//////////////////////////////////////////////////////////////////////////////
// Nosepiece (Objective Turret)
//////////////////////////////////////////////////////////////////////////////

class EvidentNosepiece : public CStateDeviceBase<EvidentNosepiece>
{
public:
    EvidentNosepiece();
    ~EvidentNosepiece();

    // MMDevice API
    int Initialize();
    int Shutdown();
    void GetName(char* pszName) const;
    bool Busy();
    unsigned long GetNumberOfPositions() const;

    // Action interface
    int OnState(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnSafeChange(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnObjectiveNA(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnObjectiveMagnification(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnObjectiveMedium(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnObjectiveWD(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
    EvidentHubWin* GetHub();
    int EnableNotifications(bool enable);
    int SafeNosepieceChange(long targetPosition);

    bool initialized_;
    std::string name_;
    unsigned int numPos_;
    bool safeNosepieceChange_;  // When true, lower focus before changing nosepiece
};

//////////////////////////////////////////////////////////////////////////////
// Magnification Changer
//////////////////////////////////////////////////////////////////////////////

class EvidentMagnification : public CMagnifierBase<EvidentMagnification>
{
public:
    EvidentMagnification();
    ~EvidentMagnification();

    // MMDevice API
    int Initialize();
    int Shutdown();
    void GetName(char* pszName) const;
    bool Busy();

    // CMagnifierBase API
    double GetMagnification();

    // Action interface
    int OnMagnification(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
    EvidentHubWin* GetHub();
    int EnableNotifications(bool enable);

    bool initialized_;
    std::string name_;
    unsigned int numPos_;
    static const double magnifications_[3];  // 1.0x, 1.6x, 2.0x
};

//////////////////////////////////////////////////////////////////////////////
// Light Path Selector
//////////////////////////////////////////////////////////////////////////////

class EvidentLightPath : public CStateDeviceBase<EvidentLightPath>
{
public:
    EvidentLightPath();
    ~EvidentLightPath();

    // MMDevice API
    int Initialize();
    int Shutdown();
    void GetName(char* pszName) const;
    bool Busy();
    unsigned long GetNumberOfPositions() const;

    // Action interface
    int OnState(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
    EvidentHubWin* GetHub();

    bool initialized_;
    std::string name_;
};

//////////////////////////////////////////////////////////////////////////////
// Condenser Turret
//////////////////////////////////////////////////////////////////////////////

class EvidentCondenserTurret : public CStateDeviceBase<EvidentCondenserTurret>
{
public:
    EvidentCondenserTurret();
    ~EvidentCondenserTurret();

    // MMDevice API
    int Initialize();
    int Shutdown();
    void GetName(char* pszName) const;
    bool Busy();
    unsigned long GetNumberOfPositions() const;

    // Action interface
    int OnState(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
    EvidentHubWin* GetHub();
    int EnableNotifications(bool enable);

    bool initialized_;
    std::string name_;
    unsigned int numPos_;
};

//////////////////////////////////////////////////////////////////////////////
// DIA Shutter
//////////////////////////////////////////////////////////////////////////////

class EvidentDIAShutter : public CShutterBase<EvidentDIAShutter>
{
public:
    EvidentDIAShutter();
    ~EvidentDIAShutter();

    // MMDevice API
    int Initialize();
    int Shutdown();
    void GetName(char* pszName) const;
    bool Busy();

    // Shutter API
    int SetOpen(bool open = true);
    int GetOpen(bool& open);
    int Fire(double deltaT);

    // Action interface
    int OnState(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnBrightness(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnMechanicalShutter(MM::PropertyBase* pProp, MM::ActionType eAct);

    // Notification control
    int EnableNotifications(bool enable);

private:
    EvidentHubWin* GetHub();

    bool initialized_;
    std::string name_;
};

//////////////////////////////////////////////////////////////////////////////
// EPI Shutter 1
//////////////////////////////////////////////////////////////////////////////

class EvidentEPIShutter1 : public CShutterBase<EvidentEPIShutter1>
{
public:
    EvidentEPIShutter1();
    ~EvidentEPIShutter1();

    // MMDevice API
    int Initialize();
    int Shutdown();
    void GetName(char* pszName) const;
    bool Busy();

    // Shutter API
    int SetOpen(bool open = true);
    int GetOpen(bool& open);
    int Fire(double deltaT);

    // Action interface
    int OnState(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
    EvidentHubWin* GetHub();

    bool initialized_;
    std::string name_;
};

//////////////////////////////////////////////////////////////////////////////
// Mirror Unit 1 (Filter Cube Turret)
//////////////////////////////////////////////////////////////////////////////

class EvidentMirrorUnit1 : public CStateDeviceBase<EvidentMirrorUnit1>
{
public:
    EvidentMirrorUnit1();
    ~EvidentMirrorUnit1();

    // MMDevice API
    int Initialize();
    int Shutdown();
    void GetName(char* pszName) const;
    bool Busy();
    unsigned long GetNumberOfPositions() const;

    // Action interface
    int OnState(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
    EvidentHubWin* GetHub();
    int EnableNotifications(bool enable);

    bool initialized_;
    std::string name_;
    unsigned int numPos_;
};

//////////////////////////////////////////////////////////////////////////////
// EPI Shutter 2
//////////////////////////////////////////////////////////////////////////////

class EvidentEPIShutter2 : public CShutterBase<EvidentEPIShutter2>
{
public:
    EvidentEPIShutter2();
    ~EvidentEPIShutter2();

    // MMDevice API
    int Initialize();
    int Shutdown();
    void GetName(char* pszName) const;
    bool Busy();

    // Shutter API
    int SetOpen(bool open = true);
    int GetOpen(bool& open);
    int Fire(double deltaT);

    // Action interface
    int OnState(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
    EvidentHubWin* GetHub();

    bool initialized_;
    std::string name_;
};

//////////////////////////////////////////////////////////////////////////////
// Mirror Unit 2 (Filter Cube Turret)
//////////////////////////////////////////////////////////////////////////////

class EvidentMirrorUnit2 : public CStateDeviceBase<EvidentMirrorUnit2>
{
public:
    EvidentMirrorUnit2();
    ~EvidentMirrorUnit2();

    // MMDevice API
    int Initialize();
    int Shutdown();
    void GetName(char* pszName) const;
    bool Busy();
    unsigned long GetNumberOfPositions() const;

    // Action interface
    int OnState(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
    EvidentHubWin* GetHub();
    int EnableNotifications(bool enable);

    bool initialized_;
    std::string name_;
    unsigned int numPos_;
};

//////////////////////////////////////////////////////////////////////////////
// Polarizer
//////////////////////////////////////////////////////////////////////////////

class EvidentPolarizer : public CStateDeviceBase<EvidentPolarizer>
{
public:
    EvidentPolarizer();
    ~EvidentPolarizer();

    // MMDevice API
    int Initialize();
    int Shutdown();
    void GetName(char* pszName) const;
    bool Busy();
    unsigned long GetNumberOfPositions() const;

    // Action interface
    int OnState(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
    EvidentHubWin* GetHub();
    int EnableNotifications(bool enable);

    bool initialized_;
    std::string name_;
    unsigned int numPos_;
};

//////////////////////////////////////////////////////////////////////////////
// DIC Prism
//////////////////////////////////////////////////////////////////////////////

class EvidentDICPrism : public CStateDeviceBase<EvidentDICPrism>
{
public:
    EvidentDICPrism();
    ~EvidentDICPrism();

    // MMDevice API
    int Initialize();
    int Shutdown();
    void GetName(char* pszName) const;
    bool Busy();
    unsigned long GetNumberOfPositions() const;

    // Action interface
    int OnState(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
    EvidentHubWin* GetHub();
    int EnableNotifications(bool enable);

    bool initialized_;
    std::string name_;
    unsigned int numPos_;
};

//////////////////////////////////////////////////////////////////////////////
// EPI ND Filter
//////////////////////////////////////////////////////////////////////////////

class EvidentEPIND : public CStateDeviceBase<EvidentEPIND>
{
public:
    EvidentEPIND();
    ~EvidentEPIND();

    // MMDevice API
    int Initialize();
    int Shutdown();
    void GetName(char* pszName) const;
    bool Busy();
    unsigned long GetNumberOfPositions() const;

    // Action interface
    int OnState(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
    EvidentHubWin* GetHub();
    int EnableNotifications(bool enable);

    bool initialized_;
    std::string name_;
    unsigned int numPos_;
};

//////////////////////////////////////////////////////////////////////////////
// Correction Collar
//////////////////////////////////////////////////////////////////////////////

class EvidentCorrectionCollar : public CStageBase<EvidentCorrectionCollar>
{
public:
    EvidentCorrectionCollar();
    ~EvidentCorrectionCollar();

    // MMDevice API
    int Initialize();
    int Shutdown();
    void GetName(char* pszName) const;
    bool Busy();

    // Stage API
    int SetPositionUm(double pos);
    int GetPositionUm(double& pos);
    int SetPositionSteps(long steps);
    int GetPositionSteps(long& steps);
    int SetOrigin();
    int GetLimits(double& lower, double& upper);
    int IsStageSequenceable(bool& isSequenceable) const { isSequenceable = false; return DEVICE_OK; };
    bool IsContinuousFocusDrive() const { return false; };

    // Action interface
    int OnActivate(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
    EvidentHubWin* GetHub();

    bool initialized_;
    bool linked_;
    std::string name_;
    double stepSizeUm_;
};

//////////////////////////////////////////////////////////////////////////////
// ZDC Autofocus
//////////////////////////////////////////////////////////////////////////////

class EvidentAutofocus : public CAutoFocusBase<EvidentAutofocus>
{
public:
    EvidentAutofocus();
    ~EvidentAutofocus();

    // MMDevice API
    int Initialize();
    int Shutdown();
    void GetName(char* pszName) const;
    bool Busy();

    // AutoFocus API
    int SetContinuousFocusing(bool state);
    int GetContinuousFocusing(bool& state);
    bool IsContinuousFocusLocked();
    int FullFocus();
    int IncrementalFocus();
    int GetLastFocusScore(double& score);
    int GetCurrentFocusScore(double& score);
    int GetOffset(double& offset);
    int SetOffset(double offset);

    // Action interface
    int OnAFMode(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnAFStatus(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnNearLimit(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnFarLimit(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnCoverSlipType(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnCoverSlipThicknessGlass(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnCoverSlipThicknessPlastic(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnDICMode(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnBuzzerSuccess(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnBuzzerFailure(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
    EvidentHubWin* GetHub();
    int EnableNotifications(bool enable);
    int StopAF();
    int InitializeZDC();  // Run full ZDC initialization sequence
    std::string GetAFStatusString(int status);

    bool initialized_;
    std::string name_;
    bool continuousFocusing_;
    int afStatus_;  // 0=Stop, 1=Focus, 2=Track, 3=Wait, 4=Search
    long nearLimit_;
    long farLimit_;
    long lastNosepiecePos_;  // Track objective changes
    int lastCoverslipType_;   // Track coverslip type changes
    bool zdcInitNeeded_;      // Flag to defer ZDC initialization
};

//////////////////////////////////////////////////////////////////////////////
// Offset Lens (ZDC)
//////////////////////////////////////////////////////////////////////////////

class EvidentOffsetLens : public CStageBase<EvidentOffsetLens>
{
public:
    EvidentOffsetLens();
    ~EvidentOffsetLens();

    // MMDevice API
    int Initialize();
    int Shutdown();
    void GetName(char* pszName) const;
    bool Busy();

    // Stage API
    int SetPositionUm(double pos);
    int GetPositionUm(double& pos);
    int SetPositionSteps(long steps);
    int GetPositionSteps(long& steps);
    int SetOrigin();
    int GetLimits(double& lower, double& upper);
    int IsStageSequenceable(bool& isSequenceable) const { isSequenceable = false; return DEVICE_OK; };
    bool IsContinuousFocusDrive() const { return false; };

    // Action interface
    int OnPosition(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
    EvidentHubWin* GetHub();
    int EnableNotifications(bool enable);

    bool initialized_;
    std::string name_;
    double stepSizeUm_;
};
