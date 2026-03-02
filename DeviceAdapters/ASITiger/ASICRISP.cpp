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
#include <sstream>
#include <string>


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
    waitAfterLock_(1000L) {

    // only set up these properties if we have the required information in the name
    if (IsExtendedName(name)) {
        axisLetter_ = GetAxisLetterFromExtName(name);
        CreateStringProperty(g_AxisLetterPropertyName, axisLetter_.c_str(), true);
    }
}

int CCRISP::Initialize() {
    // call generic Initialize first, this gets hub
    RETURN_ON_MM_ERROR(PeripheralInitialize());

    // create MM description; this doesn't work during hardware configuration wizard but will work afterwards
    std::ostringstream command;
    command << g_CRISPDeviceDescription << " Axis=" << axisLetter_ << " HexAddr=" << addressString_;
    CreateStringProperty(MM::g_Keyword_Description, command.str().c_str(), true);

    // refresh properties from controller every time - default is not to refresh (speeds things up by not redoing so much serial comm)
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
    CreateOffsetProperty();
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

bool CCRISP::Busy()
{
    // not sure how to define it, Nico's ASIStage adapter hard-codes it false so I'll do same thing
    return false;
}

int CCRISP::SetContinuousFocusing(bool state)
{
    bool focusingOn = false;
    RETURN_ON_MM_ERROR( GetContinuousFocusing(focusingOn) );  // will update focusState_
    if (focusingOn && !state)
    {
        // was on, turning off
        const std::string command = addressChar_ + "UL";
        RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command, ":A"));
    }
    else if (!focusingOn && state)
    {
        // was off, turning on
        if (focusState_ == g_CRISP_R)
        {
            return ForceSetFocusState(g_CRISP_K);
        }
        else
        {
            // need to move to ready state, then turn on
            RETURN_ON_MM_ERROR( ForceSetFocusState(g_CRISP_R) );
            RETURN_ON_MM_ERROR( ForceSetFocusState(g_CRISP_K) );
        }
    }
    // if already in state requested we don't need to do anything
    return DEVICE_OK;
}

// Update focusState_ from the controller and check if focus is locked or trying to lock ('F' or 'K' state).
int CCRISP::GetContinuousFocusing(bool& state)
{
   RETURN_ON_MM_ERROR( UpdateFocusState() );
   state = (focusState_ == g_CRISP_K) || (focusState_ == g_CRISP_F);
   return DEVICE_OK;
}

// Update focusState_ from the controller and check if focus is locked ('F' state).
bool CCRISP::IsContinuousFocusLocked()
{
    return (UpdateFocusState() == DEVICE_OK) && (focusState_ == g_CRISP_F);
}

int CCRISP::FullFocus()
{
   // Does a "one-shot" autofocus: locks and then unlocks again
   RETURN_ON_MM_ERROR ( SetContinuousFocusing(true) );

   MM::MMTime startTime = GetCurrentMMTime();
   MM::MMTime wait(0, waitAfterLock_ * 1000);
   while (!IsContinuousFocusLocked() && ((GetCurrentMMTime() - startTime) < wait))
   {
      CDeviceUtils::SleepMs(25);
   }

   CDeviceUtils::SleepMs(waitAfterLock_);

   if (!IsContinuousFocusLocked())
   {
      SetContinuousFocusing(false);
      return ERR_CRISP_NOT_LOCKED;
   }

   return SetContinuousFocusing(false);
}

int CCRISP::IncrementalFocus()
{
   return FullFocus();
}

int CCRISP::GetLastFocusScore(double& score)
{
    score = 0.0; // init in case we can't read it
    const std::string command = addressChar_ + "LK Y?";
    RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command, ":A"));
    return hub_->ParseAnswerAfterPosition3(score);
}

int CCRISP::GetCurrentFocusScore(double& score)
{
   return GetLastFocusScore(score);
}

int CCRISP::GetOffset(double& offset) {
    const std::string command = addressChar_ + "LK Z?";
    RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command, ":A"));
    return hub_->ParseAnswerAfterPosition3(offset);
}

int CCRISP::SetOffset(double offset)
{
   std::ostringstream command;
   command << addressChar_ << "LK Z=" << offset;
   RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(),":A") );
   return DEVICE_OK;
}

int CCRISP::UpdateFocusState() {
   const std::string command = addressChar_ + "LK X?";
   RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command, ":A"));

   char state = '\0';
   RETURN_ON_MM_ERROR(hub_->GetAnswerCharAtPosition3(state));
   switch (state)
   {
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

int CCRISP::ForceSetFocusState(const std::string& focusState)
{
    std::ostringstream command;
    if (focusState == g_CRISP_R)
    {
        command << addressChar_ << "LK F=85";
    }
    else if (focusState == g_CRISP_K)
    {
        command << addressChar_ << "LK F=83";
    }
    else if (focusState == g_CRISP_I)  // Idle (switch off LED)
    {
        command << addressChar_ << "LK F=79";
    }
    else if (focusState == g_CRISP_G) // log-amp calibration
    {
        command << addressChar_ << "LK F=72";
    }
    else if (focusState == g_CRISP_SG) // gain_cal (servo) calibration
    {
        command << addressChar_ << "LK F=67";
    }
    else if (focusState == g_CRISP_f) // dither
    {
        command << addressChar_ << "LK F=102";
    }
    else if (focusState == g_CRISP_RFO) // reset focus offset
    {
        command << addressChar_ << "LK F=111";
    }
    else if (focusState == g_CRISP_SSZ) // save settings to controller (least common, should be checked last)
    {
        command << addressChar_ << "SS Z";
    }

    if (command.str().empty())
    {
        return DEVICE_OK; // don't complain if we try to use an unknown state
    }
    else
    {
        return hub_->QueryCommandVerify(command.str(), ":A");
    }
}

int CCRISP::SetFocusState(const std::string& focusState) {
    RETURN_ON_MM_ERROR(UpdateFocusState());
    if (focusState == focusState_) {
        return DEVICE_OK;
    }
    return ForceSetFocusState(focusState);
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
        Props::WaitMsAfterLock, 1000L, false,
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

int CCRISP::OnNA(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   std::ostringstream command;
   double tmp = 0;
   if (eAct == MM::BeforeGet)
   {
      if (!refreshProps_ && initialized_)
      {
          return DEVICE_OK;
      }
      command << addressChar_ << "LR Y?";
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), ":A Y="));
      RETURN_ON_MM_ERROR( hub_->ParseAnswerAfterEquals(tmp) );
      if (!pProp->Set(tmp))
      {
          return DEVICE_INVALID_PROPERTY_VALUE;
      }
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(tmp);
      command << addressChar_ << "LR Y=" << tmp;
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), ":A") );
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
   std::ostringstream command;
   double tmp = 0;
   if (eAct == MM::BeforeGet)
   {
      if (!refreshProps_ && initialized_)
      {
          return DEVICE_OK;
      }
      command << addressChar_ << "LR X?";
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), ":A X="));
      RETURN_ON_MM_ERROR( hub_->ParseAnswerAfterEquals(tmp) );
      if (!pProp->Set(tmp))
      {
          return DEVICE_INVALID_PROPERTY_VALUE;
      }
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(tmp);
      command << addressChar_ << "LR X=" << tmp;
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), ":A") );
   }
   return DEVICE_OK;
}

// The LR F command uses millimeters, convert the value to microns for the Micro-Manager property.
int CCRISP::OnCalRange(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    std::ostringstream command;
    double tmp = 0;
    if (eAct == MM::BeforeGet)
    {
        if (!refreshProps_ && initialized_ && !refreshOverride_)
        {
            return DEVICE_OK;
        }
        command << addressChar_ << "LR F?";
        RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), ":A F="));
        RETURN_ON_MM_ERROR(hub_->ParseAnswerAfterEquals(tmp));
        if (!pProp->Set(tmp * 1000.0))
        {
            return DEVICE_INVALID_PROPERTY_VALUE;
        }
    }
    else if (eAct == MM::AfterSet)
    {
        pProp->Get(tmp);
        command << addressChar_ << "LR F=" << tmp / 1000.0;
        RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), ":A"));
    }
    return DEVICE_OK;
}

int CCRISP::OnLockRange(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   std::ostringstream command;
   double tmp = 0;
   if (eAct == MM::BeforeGet)
   {
      if (!refreshProps_ && initialized_)
      {
          return DEVICE_OK;
      }
      command << addressChar_ << "LR Z?";
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), ":A Z="));
      RETURN_ON_MM_ERROR( hub_->ParseAnswerAfterEquals(tmp) );
      if (!pProp->Set(tmp))
      {
          return DEVICE_INVALID_PROPERTY_VALUE;
      }
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(tmp);
      command << addressChar_ << "LR Z=" << tmp;
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), ":A") );
   }
   return DEVICE_OK;
}

int CCRISP::OnLEDIntensity(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   std::ostringstream command;
   double tmp = 0;
   if (eAct == MM::BeforeGet)
   {
      if (!refreshProps_ && initialized_)
      {
          return DEVICE_OK;
      }
      command << addressChar_ << "UL X?";
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), ":A X="));
      RETURN_ON_MM_ERROR( hub_->ParseAnswerAfterEquals(tmp) );
      if (!pProp->Set(tmp))
      {
          return DEVICE_INVALID_PROPERTY_VALUE;
      }
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(tmp);
      command << addressChar_ << "UL X=" << tmp;
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), ":A") );
   }
   return DEVICE_OK;
}

int CCRISP::OnLoopGainMultiplier(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   std::ostringstream command;
   long tmp = 0;
   if (eAct == MM::BeforeGet)
   {
      if (!refreshProps_ && initialized_)
      {
          return DEVICE_OK;
      }
      command << addressChar_ << "LR T?";
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), ":A"));
      RETURN_ON_MM_ERROR ( hub_->ParseAnswerAfterEquals(tmp) );
      if (!pProp->Set(tmp))
      {
          return DEVICE_INVALID_PROPERTY_VALUE;
      }
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(tmp);
      command << addressChar_ << "LR T=" << tmp;
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), ":A") );
   }
   return DEVICE_OK;
}

int CCRISP::OnNumAvg(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   std::ostringstream command;
   long tmp = 0;
   if (eAct == MM::BeforeGet)
   {
      if (!refreshProps_ && initialized_)
      {
          return DEVICE_OK;
      }
      command << addressChar_ << "RT F?";
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), ":A F="));
      RETURN_ON_MM_ERROR ( hub_->ParseAnswerAfterEquals(tmp) );
      if (!pProp->Set(tmp))
      {
          return DEVICE_INVALID_PROPERTY_VALUE;
      }
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(tmp);
      command << addressChar_ << "RT F=" << tmp;
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), ":A") );
   }
   return DEVICE_OK;
}

int CCRISP::OnNumSkips(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   std::ostringstream command;
   long tmp = 0;
   if (eAct == MM::BeforeGet)
   {
      if (!refreshProps_ && initialized_)
      {
          return DEVICE_OK;
      }
      command << addressChar_ << "UL Y?";
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), ":A Y="));
      RETURN_ON_MM_ERROR ( hub_->ParseAnswerAfterEquals(tmp) );
      if (!pProp->Set(tmp))
      {
          return DEVICE_INVALID_PROPERTY_VALUE;
      }
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(tmp);
      command << addressChar_ << "UL Y=" << tmp;
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), ":A") );
   }
   return DEVICE_OK;
}

// The AL Z command uses millimeters, convert the value to microns for the Micro-Manager property.
int CCRISP::OnInFocusRange(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   std::ostringstream command;
   double tmp = 0;
   if (eAct == MM::BeforeGet)
   {
      if (!refreshProps_ && initialized_ && !refreshOverride_)
      {
          return DEVICE_OK;
      }
      command << addressChar_ << "AL Z?";
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), ":A Z="));
      RETURN_ON_MM_ERROR( hub_->ParseAnswerAfterEquals(tmp) );
      if (!pProp->Set(tmp * 1000.0))
      {
          return DEVICE_INVALID_PROPERTY_VALUE;
      }
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(tmp);
      command << addressChar_ << "AL Z=" << tmp / 1000.0;
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), ":A") );
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
                RETURN_ON_MM_ERROR(UpdateFocusState());
                pProp->Set(focusState_.c_str());
            } else if (eAct == MM::AfterSet) {
                std::string focusState;
                pProp->Get(focusState);
                RETURN_ON_MM_ERROR(SetFocusState(focusState));
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
    const std::string command = addressChar_ + "LK X?";

    CreateStringProperty(
        Props::StateChar, "", true,
        new MM::ActionLambda([this, command](MM::PropertyBase* pProp, MM::ActionType eAct) {
            if (eAct == MM::BeforeGet) {
                char tmp = '\0';
                RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command, ":A"));
                RETURN_ON_MM_ERROR(hub_->GetAnswerCharAtPosition3(tmp));
                std::string str(1, tmp);
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
    std::string command = addressChar_;
    if (FirmwareVersionAtLeast(3.53)) {
        command += "EX Y?";
        LogMessage("CRISP: firmware >= 3.53; using shortcut 'EX Y?' for SNR.", true);
    } else {
        command += "EXTRA Y?";
        LogMessage("CRISP: firmware < 3.53; using full 'EXTRA Y?' for SNR.", true);
    }

    CreateFloatProperty(
        Props::SignalNoiseRatio, 0.0, true,
        new MM::ActionLambda([this, command](MM::PropertyBase* pProp, MM::ActionType eAct) {
            if (eAct == MM::BeforeGet) {
                double tmp = 0.0;
                RETURN_ON_MM_ERROR(hub_->QueryCommand(command));
                RETURN_ON_MM_ERROR(hub_->ParseAnswerAfterPosition(0, tmp));
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
void CCRISP::CreateOffsetProperty() {
    CreateIntegerProperty(
        Props::LockOffset, 0L, true,
        new MM::ActionLambda([this](MM::PropertyBase* pProp, MM::ActionType eAct) {
            if (eAct == MM::BeforeGet) {
                if (!refreshProps_ && initialized_ && !refreshOverride_) {
                    return DEVICE_OK;
                }
                double tmp = 0.0;
                const int result = GetOffset(tmp);
                if (result != DEVICE_OK) {
                    return result;
                }
                if (!pProp->Set(tmp)) {
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
        LogMessage("CRISP: firmware >= 3.40; using 'LK T?' for Sum.", true);

        const std::string command = addressChar_ + "LK T?";

        CreateIntegerProperty(
            Props::Sum, 0L, true,
            new MM::ActionLambda([this, command](MM::PropertyBase* pProp, MM::ActionType eAct) {
                if (eAct == MM::BeforeGet) {
                    long tmp = 0;
                    RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command, ":A"));
                    RETURN_ON_MM_ERROR(hub_->ParseAnswerAfterPosition3(tmp));
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
        LogMessage("CRISP: firmware < 3.40; using 'EXTRA X?' for Sum", true);

        const std::string command = addressChar_ + "EXTRA X?";

        CreateIntegerProperty(
            Props::Sum, 0L, true,
            new MM::ActionLambda([this, command](MM::PropertyBase* pProp, MM::ActionType eAct) {
                if (eAct == MM::BeforeGet) {
                    RETURN_ON_MM_ERROR(hub_->QueryCommand(command));
                    std::vector<std::string> vReply = hub_->SplitAnswerOnSpace();
                    if (vReply.size() <= 2) {
                        return DEVICE_INVALID_PROPERTY_VALUE;
                    }
                    if (!pProp->Set(vReply[1].c_str())) {
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
        LogMessage("CRISP: firmware >= 3.40; using 'LK Y?' for Dither Error.", true);

        const std::string command = addressChar_ + "LK Y?";

        CreateIntegerProperty(
            Props::DitherError, 0L, true,
            new MM::ActionLambda([this, command](MM::PropertyBase* pProp, MM::ActionType eAct) {
                if (eAct == MM::BeforeGet) {
                    long tmp = 0;
                    RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command, ":A"));
                    RETURN_ON_MM_ERROR(hub_->ParseAnswerAfterPosition3(tmp));
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
        LogMessage("CRISP: firmware < 3.40; using 'EXTRA X?' for Dither Error.", true);

        const std::string command = addressChar_ + "EXTRA X?";

        CreateIntegerProperty(
            Props::DitherError, 0L, true,
            new MM::ActionLambda([this, command](MM::PropertyBase* pProp, MM::ActionType eAct) {
                if (eAct == MM::BeforeGet) {
                    RETURN_ON_MM_ERROR(hub_->QueryCommand(command));
                    std::vector<std::string> vReply = hub_->SplitAnswerOnSpace();
                    if (vReply.size() <= 2) {
                        return DEVICE_INVALID_PROPERTY_VALUE;
                    }
                    if (!pProp->Set(vReply[2].c_str())) {
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
    const std::string command = addressChar_ + "AL X?";

    CreateIntegerProperty(
        Props::LogAmpAGC, 0L, true,
        new MM::ActionLambda([this, command](MM::PropertyBase* pProp, MM::ActionType eAct) {
            if (eAct == MM::BeforeGet) {
                long tmp = 0;
                RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command, ":A X="));
                RETURN_ON_MM_ERROR(hub_->ParseAnswerAfterEquals(tmp));
                if (!pProp->Set(tmp)) {
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
    const std::string command = addressChar_ + "LK M=";

    CreateIntegerProperty(
        Props::SetLogAmpAGC, 0L, false,
        new MM::ActionLambda([this, command](MM::PropertyBase* pProp, MM::ActionType eAct) {
            if (eAct == MM::BeforeGet) {
                pProp->Set(0L);
            } else if (eAct == MM::AfterSet) {
                long logAmpAGC = 0L;
                pProp->Get(logAmpAGC);
                if (logAmpAGC != 0L) {
                    RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command + std::to_string(logAmpAGC), ":A"));
                }
            }
            return DEVICE_OK;
        })
    );

    // No need to call UpdateProperty() because the value is always set to 0 to avoid updates
}

void CCRISP::CreateSetLockOffsetProperty() {
    const std::string command = addressChar_ + "LK Z=";

    CreateIntegerProperty(
        Props::SetLockOffset, 0L, false,
        new MM::ActionLambda([this, command](MM::PropertyBase* pProp, MM::ActionType eAct) {
            if (eAct == MM::BeforeGet) {
                pProp->Set(0L);
            } else if (eAct == MM::AfterSet) {
                long offset = 0L;
                pProp->Get(offset);
                if (offset != 0L) {
                    refreshOverride_ = true;
                    const int result = hub_->QueryCommandVerify(command + std::to_string(offset), ":A");
                    if (result != DEVICE_OK) {
                        refreshOverride_ = false;
                        return result;
                    }
                    UpdateProperty(Props::LockOffset);
                    refreshOverride_ = false;
                }
            }
            return DEVICE_OK;
        })
    );

    // No need to call UpdateProperty() because the value is always set to 0 to avoid updates
}
