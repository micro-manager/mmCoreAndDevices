///////////////////////////////////////////////////////////////////////////////
// FILE:          ASICRISP.cpp
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

#include "ASICRISP.h"
#include "ASIHub.h"
#include "ModuleInterface.h"
#include "DeviceUtils.h"
#include "DeviceBase.h"
#include "MMDevice.h"


// shared properties not implemented for CRISP because as of mid-2017 only can have one per card

namespace {
namespace Properties {

// Software-only
constexpr char WaitMsAfterLock[] = "Wait ms after Lock";

// Read/Write
constexpr char LEDIntensity[] = "LED Intensity";
constexpr char ObjectiveNA[] = "Objective NA";
constexpr char GainMultiplier[] = "GainMultiplier";
constexpr char NumberAverages[] = "Number of Averages";
constexpr char NumberSkips[] = "Number of Skips";
constexpr char CalibrationGain[] = "Calibration Gain";
constexpr char CalibrationRangeUm[] = "Calibration Range(um)";
constexpr char InFocusRangeUm[] = "In Focus Range(um)";
constexpr char MaxLockRangeMm[] = "Max Lock Range(mm)";

// Always read
constexpr char State[] = "CRISP State";
constexpr char StateChar[] = "CRISP State Character";
constexpr char SignalNoiseRatio[] = "Signal Noise Ratio";
constexpr char LockOffset[] = "Lock Offset";
constexpr char Sum[] = "Sum";
constexpr char DitherError[] = "Dither Error";
constexpr char LogAmpAGC[] = "LogAmpAGC";

// Advanced
constexpr char SetLogAmpAGC[] = "Set LogAmpAGC (Advanced Users Only)";
constexpr char SetLockOffset[] = "Set Lock Offset (Advanced Users Only)";

} // namespace Properties

namespace Props = Properties;
} // namespace


CCRISP::CCRISP(const char* name) :
    ASIPeripheralBase<::CAutoFocusBase, CCRISP>(name),
    axisLetter_(g_EmptyAxisLetterStr), // value determined by extended name
    waitAfterLock_(1000) {

    // only set up these properties if we have the required information in the name
    if (IsExtendedName(name)) {
        axisLetter_ = GetAxisLetterFromExtName(name);
        CreateStringProperty(g_AxisLetterPropertyName, axisLetter_.c_str(), true);
    }
}

CommandTable CCRISP::BuildCommandTable(std::string_view cardAddress) const {
    // create local owned copy to use + for table alignment
    const std::string a(cardAddress);

    // select serial commands based on firmware version
    const bool hasLockQueries = FirmwareVersionAtLeast(3.40);
    const bool hasExShortcut = hub_ && hub_->FirmwareVersionAtLeast(3.53) // Tiger Comm has "EX"
                                    && FirmwareVersionAtLeast(3.53);      // Stage Card has "EX"

    const std::string snr =    hasExShortcut  ? "EX Y?" : "EXTRA Y?";
    const std::string sum =    hasLockQueries ? "LK T?" : "EXTRA X?";
    const std::string dither = hasLockQueries ? "LK Y?" : "EXTRA X?";
    const std::string reply =  hasLockQueries ? ":A"    : ""; // same for both sum and dither

    LogFirmwareSupport(hasLockQueries, hasExShortcut);

    // Note: "-" indicates the operation is not supported.
    return CommandTable {        //   get           getReply  set           setReply
        /* .ledIntensity =       */ { a + "UL X?",  ":A X=",  a + "UL X=",  ":A" }, // Read/write
        /* .objectiveNA =        */ { a + "LR Y?",  ":A Y=",  a + "LR Y=",  ":A" },
        /* .gainMultiplier =     */ { a + "LR T?",  ":A T=",  a + "LR T=",  ":A" },
        /* .numberAverages =     */ { a + "RT F?",  ":A F=",  a + "RT F=",  ":A" },
        /* .numberSkips =        */ { a + "UL Y?",  ":A Y=",  a + "UL Y=",  ":A" },
        /* .calibrationGain =    */ { a + "LR X?",  ":A X=",  a + "LR X=",  ":A" },
        /* .calibrationRangeUm = */ { a + "LR F?",  ":A F=",  a + "LR F=",  ":A" },
        /* .inFocusRangeUm =     */ { a + "AL Z?",  ":A Z=",  a + "AL Z=",  ":A" },
        /* .maxLockRangeMm =     */ { a + "LR Z?",  ":A Z=",  a + "LR Z=",  ":A" },
        /* .state =              */ { a + "LK X?",  ":A",     a + "LK F=",  ":A" },
        /* .stateChar =          */ { a + "LK X?",  ":A",     "-",          "-"  }, // Read-only
        /* .signalNoiseRatio =   */ { a + snr,      "",       "-",          "-"  },
        /* .lockOffset =         */ { a + "LK Z?",  ":A",     "-",          "-"  },
        /* .sum =                */ { a + sum,      reply,    "-",          "-"  },
        /* .ditherError =        */ { a + dither,   reply,    "-",          "-"  },
        /* .logAmpAGC =          */ { a + "AL X?",  ":A X=",  "-",          "-"  },
        /* .setLogAmpAGC =       */ { "-",          "-",      a + "LK M=",  ":A" }, // Advanced
        /* .setLockOffset =      */ { "-",          "-",      a + "LK Z=",  ":A" },
        /* .focusScore =         */ { a + "LK Y?",  ":A",     "-",          "-"  }, // MM Autofocus API
        /* .unlock =             */ { "",           "-",      a + "UL",     ":A" },
    };
}

void CCRISP::LogFirmwareSupport(const bool hasLockQueries, const bool hasExShortcut) const {
    // use LOCK command instead of EXTRA
    LogMessage(hasLockQueries ?
        "CRISP: firmware >= 3.40; using 'LK T?' for Sum." :
        "CRISP: firmware < 3.40; using legacy 'EXTRA X?' for Sum.", false);
    LogMessage(hasLockQueries ?
        "CRISP: firmware >= 3.40; using 'LK Y?' for Dither Error." :
        "CRISP: firmware < 3.40; using legacy 'EXTRA X?' for Dither Error.", false);

    // Tiger Comm and Stage Card firmware required for shortcut
    if (hasExShortcut) {
        LogMessage("CRISP: firmware >= 3.53; using 'EX Y?' for SNR.", false);
    } else {
        if (FirmwareVersionAtLeast(3.53)) { // Stage Card has "EX"
            LogMessage("CRISP: Stage Card is >= 3.53 but Tiger Comm < 3.53; "
                       "falling back to 'EXTRA Y?' for SNR.", false);
        } else {
            LogMessage("CRISP: firmware < 3.53; using legacy 'EXTRA Y?' for SNR.", false);
        }
    }
}

int CCRISP::Initialize() {
    // call generic Initialize first, this gets hub
    if (const int error = PeripheralInitialize()) {
        return error;
    }

    // create CommandTable after PeripheralInitialize()
    // gets the card address and firmware version
    commands_.emplace(BuildCommandTable(addressChar_));

    // create MM description; requires runtime values not
    // available during the hardware configuration wizard
    const std::string description = std::string(g_CRISPDeviceDescription)
        + " Axis=" + axisLetter_ + " HexAddr=" + addressString_;
    CreateStringProperty(MM::g_Keyword_Description, description.c_str(), true);

    // refresh properties from controller every time - default is not to refresh
    // (speeds things up by not redoing as much serial communication)
    CreateRefreshPropertiesProperty();
    CreateWaitAfterLockProperty();

    // create properties and corresponding action handlers
    CPropertyAction* pAct;

    pAct = new CPropertyAction(this, &CCRISP::OnNA);
    CreateProperty(Props::ObjectiveNA, "0.8", MM::Float, false, pAct);
    SetPropertyLimits(Props::ObjectiveNA, 0, 1.65);
    UpdateProperty(Props::ObjectiveNA);

    pAct = new CPropertyAction(this, &CCRISP::OnLockRange);
    CreateProperty(Props::MaxLockRangeMm, "0.05", MM::Float, false, pAct);
    UpdateProperty(Props::MaxLockRangeMm);

    pAct = new CPropertyAction(this, &CCRISP::OnCalGain);
    CreateProperty(Props::CalibrationGain, "0", MM::Integer, false, pAct);
    UpdateProperty(Props::CalibrationGain);

    pAct = new CPropertyAction(this, &CCRISP::OnCalRange);
    CreateProperty(Props::CalibrationRangeUm, "0", MM::Float, false, pAct);
    UpdateProperty(Props::CalibrationRangeUm);

    pAct = new CPropertyAction(this, &CCRISP::OnLEDIntensity);
    CreateProperty(Props::LEDIntensity, "50", MM::Integer, false, pAct);
    SetPropertyLimits(Props::LEDIntensity, 0, 100);
    UpdateProperty(Props::LEDIntensity);

    pAct = new CPropertyAction(this, &CCRISP::OnLoopGainMultiplier);
    CreateProperty(Props::GainMultiplier, "10", MM::Integer, false, pAct);
    SetPropertyLimits(Props::GainMultiplier, 0, 100);
    UpdateProperty(Props::GainMultiplier);

    pAct = new CPropertyAction(this, &CCRISP::OnNumAvg);
    CreateProperty(Props::NumberAverages, "1", MM::Integer, false, pAct);
    SetPropertyLimits(Props::NumberAverages, 0, 8);
    UpdateProperty(Props::NumberAverages);

    if (FirmwareVersionAtLeast(3.12)) {
        pAct = new CPropertyAction(this, &CCRISP::OnNumSkips);
        CreateProperty(Props::NumberSkips, "0", MM::Integer, false, pAct);
        SetPropertyLimits(Props::NumberSkips, 0, 100);
        UpdateProperty(Props::NumberSkips);

        pAct = new CPropertyAction(this, &CCRISP::OnInFocusRange);
        CreateProperty(Props::InFocusRangeUm, "0.1", MM::Float, false, pAct);
        UpdateProperty(Props::InFocusRangeUm);
    }

    // Always read
    CreateFocusStateProperty();
    CreateStateProperty();
    CreateSNRProperty();
    CreateLockOffsetProperty();
    CreateSumProperty();
    CreateDitherErrorProperty();
    CreateLogAmpAGCProperty();

    // LK M requires firmware version 3.39 or higher.
    // Enable these properties as a group to modify calibration settings.
    if (FirmwareVersionAtLeast(3.39)) {
        CreateSetLogAmpAGCProperty();
        CreateSetLockOffsetProperty();
    }

    initialized_ = true;
    return DEVICE_OK;
}

bool CCRISP::Busy() {
    // not sure how to define it, Nico's ASIStage adapter hard-codes it false so I'll do same thing
    return false;
}

int CCRISP::SetContinuousFocusing(bool state) {
    bool isFocusing = false;
    // will update focusState_
    if (const int error = GetContinuousFocusing(isFocusing)) {
        return error;
    }
    if (isFocusing && !state) {
        // was on, turn off
        const Command& cmd = commands_->unlock;
        if (const int error = hub_->QueryCommandVerify(cmd.set, cmd.setReply)) {
            return error;
        }
    } else if (!isFocusing && state) {
        // was off, turn on
        if (focusState_ == g_CRISP_R) {
            return SetFocusState(g_CRISP_K);
        } else {
            // need to move to ready state, then turn on
            if (const int error = SetFocusState(g_CRISP_R)) {
                return error;
            }
            if (const int error = SetFocusState(g_CRISP_K)) {
                return error;
            }
        }
    }
    // if already in state requested we don't need to do anything
    return DEVICE_OK;
}

// Update focusState_ from the controller and check if focus is locked or trying to lock ('F' or 'K' state).
int CCRISP::GetContinuousFocusing(bool& state) {
    if (const int error = UpdateFocusState()) {
        return error;
    }
    state = (focusState_ == g_CRISP_K) || (focusState_ == g_CRISP_F);
    return DEVICE_OK;
}

// Update focusState_ from the controller and check if focus is locked ('F' state).
bool CCRISP::IsContinuousFocusLocked() {
    return (UpdateFocusState() == DEVICE_OK) && (focusState_ == g_CRISP_F);
}

// Does a "one-shot" autofocus: locks and then unlocks again
int CCRISP::FullFocus() {
    if (const int error = SetContinuousFocusing(true)) {
        return error;
    }

    const MM::MMTime startTime = GetCurrentMMTime();
    const MM::MMTime wait(0, waitAfterLock_ * 1000);
    while (!IsContinuousFocusLocked() && ((GetCurrentMMTime() - startTime) < wait)) {
        CDeviceUtils::SleepMs(25);
    }

    CDeviceUtils::SleepMs(waitAfterLock_);

    if (!IsContinuousFocusLocked()) {
        SetContinuousFocusing(false);
        return ERR_CRISP_NOT_LOCKED;
    }

    return SetContinuousFocusing(false);
}

int CCRISP::IncrementalFocus() {
    return FullFocus();
}

int CCRISP::GetCurrentFocusScore(double& score) {
    score = 0.0; // default to 0 if serial read fails
    const Command& cmd = commands_->focusScore;
    if (const int error = hub_->QueryCommandVerify(cmd.get, cmd.getReply)) {
        return error;
    }
    return hub_->ParseAnswerAfterPosition3(score);
}

int CCRISP::GetOffset(double& offset) {
    const Command& cmd = commands_->lockOffset;
    if (const int error = hub_->QueryCommandVerify(cmd.get, cmd.getReply)) {
        return error;
    }
    return hub_->ParseAnswerAfterPosition3(offset);
}

int CCRISP::SetOffset(double offset) {
    const Command& cmd = commands_->setLockOffset;
    const std::string command = cmd.set + std::to_string(offset);
    if (const int error = hub_->QueryCommandVerify(command, cmd.setReply)) {
        return error;
    }
    return DEVICE_OK;
}

int CCRISP::UpdateFocusState() {
    // get the state
    const Command& cmd = commands_->state;
    if (const int error = hub_->QueryCommandVerify(cmd.get, cmd.getReply)) {
        return error;
    }

    // get the state character
    char state = '\0';
    if (const int error = hub_->GetAnswerCharAtPosition3(state)) {
        return error;
    }

    switch (state) {
        case 'I': focusState_ = g_CRISP_I; break;
        case 'R': focusState_ = g_CRISP_R; break;
        case 'D': focusState_ = g_CRISP_D; break;
        case 'K': focusState_ = g_CRISP_K; break;  // trying to lock, goes to F when locked
        case 'F': focusState_ = g_CRISP_F; break;  // this is read-only state
        case 'N': focusState_ = g_CRISP_N; break;
        case 'E': focusState_ = g_CRISP_E; break;
        case 'G': focusState_ = g_CRISP_G; break;
        case 'H':
        case 'C': focusState_ = g_CRISP_Cal; break;
        case 'o':
        case 'l': focusState_ = g_CRISP_RFO; break;
        case 'f': focusState_ = g_CRISP_f; break;
        case '1':
        case '2':
        case '3':
        case '4':
        case '5':
        case 'g':
        case 'h':
        case 'i':
        case 'j':
        case 't': focusState_ = g_CRISP_Cal; break;
        case 'B': focusState_ = g_CRISP_B; break;
        case 'a':
        case 'b':
        case 'c':
        case 'd':
        case 'e': focusState_ = g_CRISP_C; break;
        default:  focusState_ = g_CRISP_Unknown; break;
    }

    return DEVICE_OK;
}

// TODO: make a table to precompute commands
int CCRISP::ForceSetFocusState(const std::string& focusState) {
    std::string command = "";
    if (focusState == g_CRISP_R) {
        command = "LK F=85";
    } else if (focusState == g_CRISP_K) {
        command = "LK F=83";
    } else if (focusState == g_CRISP_I) {
        command = "LK F=79"; // Idle (switch off LED)
    } else if (focusState == g_CRISP_G) {
        command = "LK F=72"; // log-amp calibration
    } else if (focusState == g_CRISP_SG) {
        command = "LK F=67"; // gain_cal (servo) calibration
    } else if (focusState == g_CRISP_f) {
        command = "LK F=102"; // dither
    } else if (focusState == g_CRISP_RFO) {
        command = "LK F=111"; // reset focus offset
    } else if (focusState == g_CRISP_SSZ) {
        command = "SS Z"; // save settings to controller (least common, should be checked last)
    }

    if (command.empty()) {
        return DEVICE_OK; // don't complain if we try to use an unknown state
    }

    return hub_->QueryCommandVerify(addressChar_ + command, ":A");
}

int CCRISP::SetFocusState(const std::string& focusState) {
    // avoid serial communication if already in the state
    if (focusState == focusState_) {
        return DEVICE_OK;
    }

    if (const int error = ForceSetFocusState(focusState)) {
        return error;
    }

    focusState_ = focusState;

    return DEVICE_OK;
}

// Properties

// Software-only property (no serial communication)
void CCRISP::CreateRefreshPropertiesProperty() {
    CreateStringProperty(
        g_RefreshPropValsPropertyName, g_NoState, false,
        new MM::ActionLambda([this](MM::PropertyBase* pProp, MM::ActionType eAct) {
            if (eAct == MM::AfterSet) {
                std::string tmp;
                pProp->Get(tmp);
                refreshProps_ = (tmp == g_YesState);
            }
            return DEVICE_OK;
        })
    );

    AddAllowedValue(g_RefreshPropValsPropertyName, g_NoState);
    AddAllowedValue(g_RefreshPropValsPropertyName, g_YesState);

    // No need to call UpdateProperty() because we already initialized
    // refreshProps_ in the ASIBase constructor and CreateStringProperty()
}

// Software-only property (no serial communication)
void CCRISP::CreateWaitAfterLockProperty() {
    CreateIntegerProperty(
        Props::WaitMsAfterLock, 1000, false,
        new MM::ActionLambda([this](MM::PropertyBase* pProp, MM::ActionType eAct) {
            if (eAct == MM::BeforeGet) {
                pProp->Set(waitAfterLock_);
            } else if (eAct == MM::AfterSet) {
                pProp->Get(waitAfterLock_);
            }
            return DEVICE_OK;
        })
    );

    // No need to call UpdateProperty() because we already initialized
    // waitAfterLock_ in the constructor and CreateIntegerProperty()
}

// Read/write Properties

int CCRISP::OnNA(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   double tmp = 0;
   const Command& cmd = commands_->objectiveNA;
   if (eAct == MM::BeforeGet)
   {
      if (!refreshProps_ && initialized_) {
          return DEVICE_OK;
      }
      if (const int error = hub_->QueryCommandVerify(cmd.get, cmd.getReply)) {
          return error;
      }
      if (const int error = hub_->ParseAnswerAfterEquals(tmp)) {
          return error;
      }
      if (!pProp->Set(tmp)) {
          return DEVICE_INVALID_PROPERTY_VALUE;
      }
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(tmp);
      const std::string command = cmd.set + std::to_string(tmp);
      if (const int error = hub_->QueryCommandVerify(command, cmd.setReply)) {
          return error;
      }
      refreshOverride_ = true;
      // update dependent properties
      UpdateProperty(Props::CalibrationRangeUm);
      if (FirmwareVersionAtLeast(3.12)) {
          UpdateProperty(Props::InFocusRangeUm);
      }
      // Note: do not return early on UpdateProperty errors
      refreshOverride_ = false; // must reach this point
   }
   return DEVICE_OK;
}

int CCRISP::OnCalGain(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   double tmp = 0;
   const Command& cmd = commands_->calibrationGain;
   if (eAct == MM::BeforeGet)
   {
      if (!refreshProps_ && initialized_) {
          return DEVICE_OK;
      }
      if (const int error = hub_->QueryCommandVerify(cmd.get, cmd.getReply)) {
          return error;
      }
      if (const int error = hub_->ParseAnswerAfterEquals(tmp)) {
          return error;
      }
      if (!pProp->Set(tmp)) {
          return DEVICE_INVALID_PROPERTY_VALUE;
      }
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(tmp);
      const std::string command = cmd.set + std::to_string(tmp);
      if (const int error = hub_->QueryCommandVerify(command, cmd.setReply)) {
          return error;
      }
   }
   return DEVICE_OK;
}

// The LR F command uses millimeters, convert the value to microns for the Micro-Manager property.
int CCRISP::OnCalRange(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    double tmp = 0;
    const Command& cmd = commands_->calibrationRangeUm;
    if (eAct == MM::BeforeGet)
    {
        if (!refreshProps_ && initialized_ && !refreshOverride_) {
            return DEVICE_OK;
        }
        if (const int error = hub_->QueryCommandVerify(cmd.get, cmd.getReply)) {
            return error;
        }
        if (const int error = hub_->ParseAnswerAfterEquals(tmp)) {
            return error;
        }
        if (!pProp->Set(tmp * 1000.0)) {
            return DEVICE_INVALID_PROPERTY_VALUE;
        }
    }
    else if (eAct == MM::AfterSet)
    {
        pProp->Get(tmp);
        const std::string command = cmd.set + std::to_string(tmp / 1000.0);
        if (const int error = hub_->QueryCommandVerify(command, cmd.setReply)) {
            return error;
        }
    }
    return DEVICE_OK;
}

int CCRISP::OnLockRange(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   double tmp = 0;
   const Command& cmd = commands_->maxLockRangeMm;
   if (eAct == MM::BeforeGet)
   {
      if (!refreshProps_ && initialized_) {
          return DEVICE_OK;
      }
      if (const int error = hub_->QueryCommandVerify(cmd.get, cmd.getReply)) {
          return error;
      }
      if (const int error = hub_->ParseAnswerAfterEquals(tmp)) {
          return error;
      }
      if (!pProp->Set(tmp)) {
          return DEVICE_INVALID_PROPERTY_VALUE;
      }
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(tmp);
      const std::string command = cmd.set + std::to_string(tmp);
      if (const int error = hub_->QueryCommandVerify(command, cmd.setReply)) {
          return error;
      }
   }
   return DEVICE_OK;
}

int CCRISP::OnLEDIntensity(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   long tmp = 0;
   const Command& cmd = commands_->ledIntensity;
   if (eAct == MM::BeforeGet)
   {
      if (!refreshProps_ && initialized_) {
          return DEVICE_OK;
      }
      if (const int error = hub_->QueryCommandVerify(cmd.get, cmd.getReply)) {
          return error;
      }
      if (const int error = hub_->ParseAnswerAfterEquals(tmp)) {
          return error;
      }
      if (!pProp->Set(tmp)) {
          return DEVICE_INVALID_PROPERTY_VALUE;
      }
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(tmp);
      const std::string command = cmd.set + std::to_string(tmp);
      if (const int error = hub_->QueryCommandVerify(command, cmd.setReply)) {
          return error;
      }
   }
   return DEVICE_OK;
}

int CCRISP::OnLoopGainMultiplier(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   long tmp = 0;
   const Command& cmd = commands_->gainMultiplier;
   if (eAct == MM::BeforeGet)
   {
      if (!refreshProps_ && initialized_) {
          return DEVICE_OK;
      }
      if (const int error = hub_->QueryCommandVerify(cmd.get, cmd.getReply)) {
          return error;
      }
      if (const int error = hub_->ParseAnswerAfterEquals(tmp)) {
          return error;
      }
      if (!pProp->Set(tmp)) {
          return DEVICE_INVALID_PROPERTY_VALUE;
      }
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(tmp);
      const std::string command = cmd.set + std::to_string(tmp);
      if (const int error = hub_->QueryCommandVerify(command, cmd.setReply)) {
          return error;
      }
   }
   return DEVICE_OK;
}

int CCRISP::OnNumAvg(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   long tmp = 0;
   const Command& cmd = commands_->numberAverages;
   if (eAct == MM::BeforeGet)
   {
      if (!refreshProps_ && initialized_) {
          return DEVICE_OK;
      }
      if (const int error = hub_->QueryCommandVerify(cmd.get, cmd.getReply)) {
          return error;
      }
      if (const int error = hub_->ParseAnswerAfterEquals(tmp)) {
          return error;
      }
      if (!pProp->Set(tmp)) {
          return DEVICE_INVALID_PROPERTY_VALUE;
      }
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(tmp);
      const std::string command = cmd.set + std::to_string(tmp);
      if (const int error = hub_->QueryCommandVerify(command, cmd.setReply)) {
          return error;
      }
   }
   return DEVICE_OK;
}

int CCRISP::OnNumSkips(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   long tmp = 0;
   const Command& cmd = commands_->numberSkips;
   if (eAct == MM::BeforeGet)
   {
      if (!refreshProps_ && initialized_) {
          return DEVICE_OK;
      }
      if (const int error = hub_->QueryCommandVerify(cmd.get, cmd.getReply)) {
          return error;
      }
      if (const int error = hub_->ParseAnswerAfterEquals(tmp)) {
          return error;
      }
      if (!pProp->Set(tmp)) {
          return DEVICE_INVALID_PROPERTY_VALUE;
      }
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(tmp);
      const std::string command = cmd.set + std::to_string(tmp);
      if (const int error = hub_->QueryCommandVerify(command, cmd.setReply)) {
          return error;
      }
   }
   return DEVICE_OK;
}

// The AL Z command uses millimeters, convert the value to microns for the Micro-Manager property.
int CCRISP::OnInFocusRange(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   double tmp = 0.0;
   const Command& cmd = commands_->inFocusRangeUm;
   if (eAct == MM::BeforeGet)
   {
      if (!refreshProps_ && initialized_ && !refreshOverride_) {
          return DEVICE_OK;
      }
      if (const int error = hub_->QueryCommandVerify(cmd.get, cmd.getReply)) {
          return error;
      }
      if (const int error = hub_->ParseAnswerAfterEquals(tmp)) {
          return error;
      }
      if (!pProp->Set(tmp * 1000.0)) {
          return DEVICE_INVALID_PROPERTY_VALUE;
      }
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(tmp);
      const std::string command = cmd.set + std::to_string(tmp / 1000.0);
      if (const int error = hub_->QueryCommandVerify(command, cmd.setReply)) {
          return error;
      }
   }
   return DEVICE_OK;
}

// Read-only Properties

// Always read
void CCRISP::CreateFocusStateProperty() {
    CreateStringProperty(
        Props::State, "Idle", false,
        new MM::ActionLambda([this](MM::PropertyBase* pProp, MM::ActionType eAct) {
            if (eAct == MM::BeforeGet) {
                if (const int error = UpdateFocusState()) {
                    return error;
                }
                pProp->Set(focusState_.c_str());
            } else if (eAct == MM::AfterSet) {
                std::string focusState;
                pProp->Get(focusState);
                if (const int error = SetFocusState(focusState)) {
                    return error;
                }
            }
            return DEVICE_OK;
        })
    );

    AddAllowedValue(Props::State, g_CRISP_I, 79);
    AddAllowedValue(Props::State, g_CRISP_R, 85);
    AddAllowedValue(Props::State, g_CRISP_D);
    AddAllowedValue(Props::State, g_CRISP_K, 83);
    AddAllowedValue(Props::State, g_CRISP_F);
    AddAllowedValue(Props::State, g_CRISP_N);
    AddAllowedValue(Props::State, g_CRISP_E);
    AddAllowedValue(Props::State, g_CRISP_G, 72);
    AddAllowedValue(Props::State, g_CRISP_SG, 67);
    AddAllowedValue(Props::State, g_CRISP_f, 102);
    AddAllowedValue(Props::State, g_CRISP_C, 97);
    AddAllowedValue(Props::State, g_CRISP_B, 66);
    AddAllowedValue(Props::State, g_CRISP_RFO, 111);
    AddAllowedValue(Props::State, g_CRISP_SSZ);

    UpdateProperty(Props::State);
}

// Always read
void CCRISP::CreateStateProperty() {
    CreateStringProperty(
        Props::StateChar, "", true,
        new MM::ActionLambda([this](MM::PropertyBase* pProp, MM::ActionType eAct) {
            if (eAct == MM::BeforeGet) {
                char tmp = '\0';
                const Command& cmd = commands_->stateChar;
                if (const int error = hub_->QueryCommandVerify(cmd.get, cmd.getReply)) {
                    return error;
                }
                if (const int error = hub_->GetAnswerCharAtPosition3(tmp)) {
                    return error;
                }
                const std::string str(1, tmp);
                if (!pProp->Set(str.c_str())) {
                    return DEVICE_INVALID_PROPERTY_VALUE;
                }
            }
            return DEVICE_OK;
        })
    );

    UpdateProperty(Props::StateChar);
}

// Always read
void CCRISP::CreateSNRProperty() {
    CreateFloatProperty(
        Props::SignalNoiseRatio, 0.0, true,
        new MM::ActionLambda([this](MM::PropertyBase* pProp, MM::ActionType eAct) {
            if (eAct == MM::BeforeGet) {
                double tmp = 0.0;
                const Command& cmd = commands_->signalNoiseRatio;
                if (const int error = hub_->QueryCommand(cmd.get)) {
                    return error;
                }
                if (const int error = hub_->ParseAnswerAfterPosition(0, tmp)) {
                    return error;
                }
                if (!pProp->Set(tmp)) {
                    return DEVICE_INVALID_PROPERTY_VALUE;
                }
            }
            return DEVICE_OK;
        })
    );

    UpdateProperty(Props::SignalNoiseRatio);
}

// Always read
void CCRISP::CreateLockOffsetProperty() {
    CreateIntegerProperty(
        Props::LockOffset, 0, true,
        new MM::ActionLambda([this](MM::PropertyBase* pProp, MM::ActionType eAct) {
            if (eAct == MM::BeforeGet) {
                if (!refreshProps_ && initialized_ && !refreshOverride_) {
                    return DEVICE_OK;
                }
                double tmp = 0.0; // Note: autofocus API requires double
                if (const int error = GetOffset(tmp)) {
                    return error;
                }
                // convert to long for integer property
                if (!pProp->Set(static_cast<long>(tmp))) {
                    return DEVICE_INVALID_PROPERTY_VALUE;
                }
            }
            return DEVICE_OK;
        })
    );

    UpdateProperty(Props::LockOffset);
}

// Always read
void CCRISP::CreateSumProperty() {

    if (FirmwareVersionAtLeast(3.40)) {
        // The LOCK command can query the value directly
        // The command responds with => ":A 0 \r\n"
        CreateIntegerProperty(
            Props::Sum, 0, true,
            new MM::ActionLambda([this](MM::PropertyBase* pProp, MM::ActionType eAct) {
                if (eAct == MM::BeforeGet) {
                    long tmp = 0;
                    const Command& cmd = commands_->sum;
                    if (const int error = hub_->QueryCommandVerify(cmd.get, cmd.getReply)) {
                        return error;
                    }
                    if (const int error = hub_->ParseAnswerAfterPosition3(tmp)) {
                        return error;
                    }
                    if (!pProp->Set(tmp)) {
                        return DEVICE_INVALID_PROPERTY_VALUE;
                    }
                }
                return DEVICE_OK;
            })
        );

    } else { // Firmware < 3.40

        // The old version uses the EXTRA command and requires parsing
        // The command responds with => "I    0    0 \r\n"
        CreateIntegerProperty(
            Props::Sum, 0, true,
            new MM::ActionLambda([this](MM::PropertyBase* pProp, MM::ActionType eAct) {
                if (eAct == MM::BeforeGet) {
                    const Command& cmd = commands_->sum;
                    if (const int error = hub_->QueryCommand(cmd.get)) {
                        return error;
                    }
                    const std::vector<std::string> reply = hub_->SplitAnswerOnSpace();
                    if (reply.size() <= 2) {
                        return DEVICE_INVALID_PROPERTY_VALUE;
                    }
                    if (!pProp->Set(reply[1].c_str())) {
                        return DEVICE_INVALID_PROPERTY_VALUE;
                    }
                }
                return DEVICE_OK;
            })
        );
    }

    UpdateProperty(Props::Sum);
}

// Always read
void CCRISP::CreateDitherErrorProperty() {

    if (FirmwareVersionAtLeast(3.40)) {
        // The LOCK command can query the value directly
        // The command responds with => ":A 0 \r\n"
        CreateIntegerProperty(
            Props::DitherError, 0, true,
            new MM::ActionLambda([this](MM::PropertyBase* pProp, MM::ActionType eAct) {
                if (eAct == MM::BeforeGet) {
                    long tmp = 0;
                    const Command& cmd = commands_->ditherError;
                    if (const int error = hub_->QueryCommandVerify(cmd.get, cmd.getReply)) {
                        return error;
                    }
                    if (const int error = hub_->ParseAnswerAfterPosition3(tmp)) {
                        return error;
                    }
                    if (!pProp->Set(tmp)) {
                        return DEVICE_INVALID_PROPERTY_VALUE;
                    }
                }
                return DEVICE_OK;
            })
        );

    } else { // Firmware < 3.40

        // The old version uses the EXTRA command and requires parsing
        // The command responds with => "I    0    0 \r\n"
        CreateIntegerProperty(
            Props::DitherError, 0, true,
            new MM::ActionLambda([this](MM::PropertyBase* pProp, MM::ActionType eAct) {
                if (eAct == MM::BeforeGet) {
                    const Command& cmd = commands_->ditherError;
                    if (const int error = hub_->QueryCommand(cmd.get)) {
                        return error;
                    }
                    const std::vector<std::string> reply = hub_->SplitAnswerOnSpace();
                    if (reply.size() <= 2) {
                        return DEVICE_INVALID_PROPERTY_VALUE;
                    }
                    if (!pProp->Set(reply[2].c_str())) {
                        return DEVICE_INVALID_PROPERTY_VALUE;
                    }
                }
                return DEVICE_OK;
            })
        );
    }

    UpdateProperty(Props::DitherError);
}

// Always read
void CCRISP::CreateLogAmpAGCProperty() {
    CreateIntegerProperty(
        Props::LogAmpAGC, 0, true,
        new MM::ActionLambda([this](MM::PropertyBase* pProp, MM::ActionType eAct) {
            if (eAct == MM::BeforeGet) {
                double tmp = 0.0; // Note: response is ":A X=1.000000", parse as double
                const Command& cmd = commands_->logAmpAGC;
                if (const int error = hub_->QueryCommandVerify(cmd.get, cmd.getReply)) {
                    return error;
                }
                if (const int error = hub_->ParseAnswerAfterEquals(tmp)) {
                    return error;
                }
                // convert to long for integer property
                if (!pProp->Set(static_cast<long>(tmp))) {
                    return DEVICE_INVALID_PROPERTY_VALUE;
                }
            }
            return DEVICE_OK;
        })
    );

    UpdateProperty(Props::LogAmpAGC);
}

// Advanced Properties

void CCRISP::CreateSetLogAmpAGCProperty() {
    CreateIntegerProperty(
        Props::SetLogAmpAGC, 0, false,
        new MM::ActionLambda([this](MM::PropertyBase* pProp, MM::ActionType eAct) {
            if (eAct == MM::BeforeGet) {
                pProp->Set(0L);
            } else if (eAct == MM::AfterSet) {
                long logAmpAGC = 0;
                pProp->Get(logAmpAGC);
                if (logAmpAGC != 0) {
                    const Command& cmd = commands_->setLogAmpAGC;
                    const std::string command = cmd.set + std::to_string(logAmpAGC);
                    if (const int error = hub_->QueryCommandVerify(command, cmd.setReply)) {
                        return error;
                    }
                }
            }
            return DEVICE_OK;
        })
    );

    // No need to call UpdateProperty() because the value is always set to 0 to avoid updates
}

void CCRISP::CreateSetLockOffsetProperty() {
    CreateIntegerProperty(
        Props::SetLockOffset, 0, false,
        new MM::ActionLambda([this](MM::PropertyBase* pProp, MM::ActionType eAct) {
            if (eAct == MM::BeforeGet) {
                pProp->Set(0L);
            } else if (eAct == MM::AfterSet) {
                long offset = 0;
                pProp->Get(offset);
                if (offset != 0) {
                    const Command& cmd = commands_->setLockOffset;
                    const std::string command = cmd.set + std::to_string(offset);
                    if (const int error = hub_->QueryCommandVerify(command, cmd.setReply)) {
                        return error;
                    }
                    refreshOverride_ = true;
                    UpdateProperty(Props::LockOffset);
                    refreshOverride_ = false;
                }
            }
            return DEVICE_OK;
        })
    );

    // No need to call UpdateProperty() because the value is always set to 0 to avoid updates
}
