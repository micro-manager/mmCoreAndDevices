///////////////////////////////////////////////////////////////////////////////
// FILE:          ASICRISP.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   ASI CRISP autofocus device adapter
//
// COPYRIGHT:     Applied Scientific Instrumentation, Eugene OR
//
// LICENSE:       This file is distributed under the BSD license.
//
//                This file is distributed in the hope that it will be useful,
//                but WITHOUT ANY WARRANTY; without even the implied warranty
//                of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//                IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//                CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//                INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.
//
// AUTHOR:        Jon Daniels (jon@asiimaging.com) 09/2013
//

#ifndef ASICRISP_H
#define ASICRISP_H

#include "ASIPeripheralBase.h"
#include "MMDevice.h"
#include "DeviceBase.h"
#include <optional>
#include <string>
#include <string_view>


// The serial command set for a device property.
struct Command {
    const std::string get;      // read from hardware (e.g. "2LR Z?")
    const std::string getReply; // reply from get (e.g. ":A Z=")
    const std::string set;      // write to hardware (e.g. "2LR Z=")
    const std::string setReply; // reply from set (e.g. ":A")
};

// The command table for CRISP.
// Each device property has a serial command set.
struct CommandTable {
    // Read/write
    const Command ledIntensity;
    const Command objectiveNA;
    const Command gainMultiplier;
    const Command numberAverages;
    const Command numberSkips;
    const Command calibrationGain;
    const Command calibrationRangeUm;
    const Command inFocusRangeUm;
    const Command maxLockRangeMm;
    // Read-only
    const Command state;
    const Command stateChar;
    const Command signalNoiseRatio;
    const Command lockOffset;
    const Command sum;
    const Command ditherError;
    const Command logAmpAGC;
    // Advanced
    const Command setLogAmpAGC;
    const Command setLockOffset;
    // MM Autofocus API
    const Command focusScore;
    const Command unlock;
};

// ASI CRISP Autofocus Device
// Documentation: https://asiimaging.com/docs/crisp_manual
class CCRISP : public ASIPeripheralBase<CAutoFocusBase, CCRISP> {
public:
    explicit CCRISP(const char* name);
    ~CCRISP() = default;

    // MM Device API
    int Initialize();
    bool Busy();

    // MM Autofocus API
    int SetContinuousFocusing(bool state);
    int GetContinuousFocusing(bool& state);
    bool IsContinuousFocusLocked();
    int FullFocus();
    int IncrementalFocus();
    int GetLastFocusScore(double& score) { return GetCurrentFocusScore(score); }
    int GetCurrentFocusScore(double& score);
    int GetOffset(double& offset);
    int SetOffset(double offset);

    // action interface
    int OnNA(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnCalGain(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnCalRange(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnLockRange(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnLEDIntensity(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnLoopGainMultiplier(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnNumAvg(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnLogAmpAGC(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnNumSkips(MM::PropertyBase* pProp, MM::ActionType eAct);
    int OnInFocusRange(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
    int UpdateFocusState();
    int SetFocusState(const std::string& focusState);
    int ForceSetFocusState(const std::string& focusState);

    CommandTable BuildCommandTable(std::string_view cardAddress) const;
    void LogFirmwareSupport(const bool hasLockQueries, const bool hasExShortcut) const;

    // Properties

    // Software-only
    void CreateRefreshPropertiesProperty();
    void CreateWaitAfterLockProperty();
    // Read-only
    void CreateFocusStateProperty();
    void CreateStateProperty();
    void CreateSNRProperty();
    void CreateLockOffsetProperty();
    void CreateSumProperty();
    void CreateDitherErrorProperty();
    void CreateLogAmpAGCProperty();
    // Advanced
    void CreateSetLogAmpAGCProperty();
    void CreateSetLockOffsetProperty();

    std::string axisLetter_;
    std::string focusState_;
    long waitAfterLock_;

    // The CommandTable is created in Initialize() once we know the card address and firmware version.
    // std::optional allows the table to be late-initialized while keeping its members const for immutability.
    std::optional<CommandTable> commands_;
};

#endif // ASICRISP_H
