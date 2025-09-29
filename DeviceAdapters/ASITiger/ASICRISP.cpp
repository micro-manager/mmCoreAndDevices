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
#include <iostream>
#include <cmath>
#include <sstream>
#include <string>


// shared properties not implemented for CRISP because as of mid-2017 only can have one per card

///////////////////////////////////////////////////////////////////////////////
// CCRISP
//
CCRISP::CCRISP(const char* name) :
   ASIPeripheralBase< ::CAutoFocusBase, CCRISP >(name),
   axisLetter_(g_EmptyAxisLetterStr),    // value determined by extended name
   waitAfterLock_(1000)
{
   if (IsExtendedName(name))  // only set up these properties if we have the required information in the name
   {
      axisLetter_ = GetAxisLetterFromExtName(name);
      CreateProperty(g_AxisLetterPropertyName, axisLetter_.c_str(), MM::String, true);
   }
}

int CCRISP::Initialize()
{
   // call generic Initialize first, this gets hub
   RETURN_ON_MM_ERROR( PeripheralInitialize() );

   // create MM description; this doesn't work during hardware configuration wizard but will work afterwards
   std::ostringstream command;
   command << g_CRISPDeviceDescription << " Axis=" << axisLetter_ << " HexAddr=" << addressString_;
   CreateProperty(MM::g_Keyword_Description, command.str().c_str(), MM::String, true);

   // create properties and corresponding action handlers

   CPropertyAction* pAct;

   // refresh properties from controller every time - default is not to refresh (speeds things up by not redoing so much serial comm)
   pAct = new CPropertyAction (this, &CCRISP::OnRefreshProperties);
   CreateProperty(g_RefreshPropValsPropertyName, g_NoState, MM::String, false, pAct);
   AddAllowedValue(g_RefreshPropValsPropertyName, g_NoState);
   AddAllowedValue(g_RefreshPropValsPropertyName, g_YesState);

   pAct = new CPropertyAction(this, &CCRISP::OnFocusState);
   CreateProperty (g_CRISPState, g_CRISP_I, MM::String, false, pAct);
   AddAllowedValue(g_CRISPState, g_CRISP_I, 79);
   AddAllowedValue(g_CRISPState, g_CRISP_R, 85);
   AddAllowedValue(g_CRISPState, g_CRISP_D);
   AddAllowedValue(g_CRISPState, g_CRISP_K, 83);
   AddAllowedValue(g_CRISPState, g_CRISP_F);
   AddAllowedValue(g_CRISPState, g_CRISP_N);
   AddAllowedValue(g_CRISPState, g_CRISP_E);
   AddAllowedValue(g_CRISPState, g_CRISP_G, 72);
   AddAllowedValue(g_CRISPState, g_CRISP_SG, 67);
   AddAllowedValue(g_CRISPState, g_CRISP_f, 102);
   AddAllowedValue(g_CRISPState, g_CRISP_C, 97);
   AddAllowedValue(g_CRISPState, g_CRISP_B, 66);
   AddAllowedValue(g_CRISPState, g_CRISP_RFO, 111);
   AddAllowedValue(g_CRISPState, g_CRISP_SSZ);

   pAct = new CPropertyAction(this, &CCRISP::OnWaitAfterLock);
   CreateProperty(g_CRISPWaitAfterLockPropertyName, "1000", MM::Integer, false, pAct);
   UpdateProperty(g_CRISPWaitAfterLockPropertyName);

   pAct = new CPropertyAction(this, &CCRISP::OnNA);
   CreateProperty(g_CRISPObjectiveNAPropertyName, "0.8", MM::Float, false, pAct);
   SetPropertyLimits(g_CRISPObjectiveNAPropertyName, 0, 1.65);
   UpdateProperty(g_CRISPObjectiveNAPropertyName);

   pAct = new CPropertyAction(this, &CCRISP::OnLockRange);
   CreateProperty(g_CRISPLockRangePropertyName, "0.05", MM::Float, false, pAct);
   UpdateProperty(g_CRISPLockRangePropertyName);

   pAct = new CPropertyAction(this, &CCRISP::OnCalGain);
   CreateProperty(g_CRISPCalibrationGainPropertyName, "0", MM::Integer, false, pAct);
   UpdateProperty(g_CRISPCalibrationGainPropertyName);

   pAct = new CPropertyAction(this, &CCRISP::OnCalRange);
   CreateProperty(g_CRISPCalibrationRangePropertyName, "0", MM::Float, false, pAct);
   UpdateProperty(g_CRISPCalibrationRangePropertyName);

   pAct = new CPropertyAction(this, &CCRISP::OnLEDIntensity);
   CreateProperty(g_CRISPLEDIntensityPropertyName, "50", MM::Integer, false, pAct);
   SetPropertyLimits(g_CRISPLEDIntensityPropertyName, 0, 100);
   UpdateProperty(g_CRISPLEDIntensityPropertyName);

   pAct = new CPropertyAction(this, &CCRISP::OnLoopGainMultiplier);
   CreateProperty(g_CRISPLoopGainMultiplierPropertyName, "10", MM::Integer, false, pAct);
   SetPropertyLimits(g_CRISPLoopGainMultiplierPropertyName, 0, 100);
   UpdateProperty(g_CRISPLoopGainMultiplierPropertyName);

   pAct = new CPropertyAction(this, &CCRISP::OnNumAvg);
   CreateProperty(g_CRISPNumberAveragesPropertyName, "1", MM::Integer, false, pAct);
   SetPropertyLimits(g_CRISPNumberAveragesPropertyName, 0, 8);
   UpdateProperty(g_CRISPNumberAveragesPropertyName);

   pAct = new CPropertyAction(this, &CCRISP::OnSNR);
   CreateProperty(g_CRISPSNRPropertyName, "", MM::Float, true, pAct);
   UpdateProperty(g_CRISPSNRPropertyName);

   pAct = new CPropertyAction(this, &CCRISP::OnLogAmpAGC);
   CreateProperty(g_CRISPLogAmpAGCPropertyName, "", MM::Integer, true, pAct);
   UpdateProperty(g_CRISPLogAmpAGCPropertyName);

   pAct = new CPropertyAction(this, &CCRISP::OnOffset);
   CreateProperty(g_CRISPOffsetPropertyName, "", MM::Integer, true, pAct);
   UpdateProperty(g_CRISPOffsetPropertyName);

   pAct = new CPropertyAction(this, &CCRISP::OnState);
   CreateProperty(g_CRISPStatePropertyName, "", MM::String, true, pAct);
   UpdateProperty(g_CRISPStatePropertyName);

   if (FirmwareVersionAtLeast(3.12))
   {
    pAct = new CPropertyAction(this, &CCRISP::OnNumSkips);
    CreateProperty(g_CRISPNumberSkipsPropertyName, "0", MM::Integer, false, pAct);
    SetPropertyLimits(g_CRISPNumberSkipsPropertyName, 0, 100);
    UpdateProperty(g_CRISPNumberSkipsPropertyName);

    pAct = new CPropertyAction(this, &CCRISP::OnInFocusRange);
    CreateProperty(g_CRISPInFocusRangePropertyName, "0.1", MM::Float, false, pAct);
    UpdateProperty(g_CRISPInFocusRangePropertyName);
   }

   // Note: Older firmware could only query the properties "Dither Error" and "Sum" through EXTRA X?, 
   // new firmware has commands to query the values much faster.
   if (FirmwareVersionAtLeast(3.40))
   {
       LogMessage("CRISP: firmware >= 3.40; use LK T? and LK Y? for the \"Sum\" and \"Dither Error\" properties.", true);
       pAct = new CPropertyAction(this, &CCRISP::OnSum);
       CreateProperty(g_CRISPSumPropertyName, "", MM::Integer, true, pAct);
       UpdateProperty(g_CRISPSumPropertyName);

       pAct = new CPropertyAction(this, &CCRISP::OnDitherError);
       CreateProperty(g_CRISPDitherErrorPropertyName, "", MM::Integer, true, pAct);
       UpdateProperty(g_CRISPDitherErrorPropertyName);
   }
   else
   {
       LogMessage("CRISP: firmware < 3.40; use EXTRA X? for both the \"Sum\" and \"Dither Error\" properties.", true);
       pAct = new CPropertyAction(this, &CCRISP::OnSumLegacy);
       CreateProperty(g_CRISPSumPropertyName, "", MM::Integer, true, pAct);
       UpdateProperty(g_CRISPSumPropertyName);

       pAct = new CPropertyAction(this, &CCRISP::OnDitherErrorLegacy);
       CreateProperty(g_CRISPDitherErrorPropertyName, "", MM::Integer, true, pAct);
       UpdateProperty(g_CRISPDitherErrorPropertyName);
   }

   // Always read
   CreateSumProperty();
   CreateDitherErrorProperty();

   // LK M requires firmware version 3.39 or higher.
   // Enable these properties as a group to modify calibration settings.
   if (FirmwareVersionAtLeast(3.39))
   {
       // No need to call UpdateProperty() because the values for these properties
       // are always set to 0 to avoid unnecessary updates.
       pAct = new CPropertyAction(this, &CCRISP::OnSetLogAmpAGC);
       CreateProperty(g_CRISPSetLogAmpAGCPropertyName, "0", MM::Integer, false, pAct);

       pAct = new CPropertyAction(this, &CCRISP::OnSetLockOffset);
       CreateProperty(g_CRISPSetOffsetPropertyName, "0", MM::Integer, false, pAct);
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
    std::ostringstream command;
    bool focusingOn = false;
    RETURN_ON_MM_ERROR( GetContinuousFocusing(focusingOn) );  // will update focusState_
    if (focusingOn && !state)
    {
        // was on, turning off
        command << addressChar_ << "UL";
        RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), ":A") );
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
   std::ostringstream command;
   command << addressChar_ << "LK Y?";
   RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(),":A") );
   return hub_->ParseAnswerAfterPosition3(score);
}

int CCRISP::GetCurrentFocusScore(double& score)
{
   return GetLastFocusScore(score);
}

int CCRISP::GetOffset(double& offset)
{
   std::ostringstream command;
   command << addressChar_ << "LK Z?";
   RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(),":A") );
   return hub_->ParseAnswerAfterPosition3(offset);
}

int CCRISP::SetOffset(double offset)
{
   std::ostringstream command;
   command << addressChar_ << "LK Z=" << offset;
   RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(),":A") );
   return DEVICE_OK;
}

int CCRISP::UpdateFocusState()
{
   std::ostringstream command;
   command << addressChar_ << "LK X?";
   RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(),":A") );

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

int CCRISP::SetFocusState(const std::string& focusState)
{
    RETURN_ON_MM_ERROR ( UpdateFocusState() );
    if (focusState == focusState_)
    {
        return DEVICE_OK;
    }
    return ForceSetFocusState(focusState);
}

// action handlers

int CCRISP::OnRefreshProperties(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::AfterSet)
    {
        std::string tmpstr;
        pProp->Get(tmpstr);
        refreshProps_ = (tmpstr == g_YesState) ? true : false;
    }
    return DEVICE_OK;
}

// read this every time
int CCRISP::OnFocusState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      RETURN_ON_MM_ERROR( UpdateFocusState() );
      pProp->Set(focusState_.c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      std::string focusState;
      pProp->Get(focusState);
      RETURN_ON_MM_ERROR( SetFocusState(focusState) );
   }
   return DEVICE_OK;
}

// property value set in MM only, not read from nor written to controller
int CCRISP::OnWaitAfterLock(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(waitAfterLock_);
    }
    else if (eAct == MM::AfterSet)
    {
        pProp->Get(waitAfterLock_);
    }
    return DEVICE_OK;
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
      // also update the "Calibration Range(um)" property
      UpdateProperty(g_CRISPCalibrationRangePropertyName);
      // also update "In Focus Range(um)" property
      if (FirmwareVersionAtLeast(3.12))
      {
          UpdateProperty(g_CRISPInFocusRangePropertyName);
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

// always read
int CCRISP::OnSNR(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      double tmp = 0.0;
      std::ostringstream command;
      command << addressChar_ << "EXTRA Y?";
      RETURN_ON_MM_ERROR( hub_->QueryCommand(command.str()) );
      RETURN_ON_MM_ERROR( hub_->ParseAnswerAfterPosition(0, tmp));
      if (!pProp->Set(tmp))
      {
          return DEVICE_INVALID_PROPERTY_VALUE;
      }
   }
   return DEVICE_OK;
}

// always read
int CCRISP::OnDitherError(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        long tmp = 0;
        std::ostringstream command;
        command << addressChar_ << "LK Y?";
        RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), ":A"));
        RETURN_ON_MM_ERROR(hub_->ParseAnswerAfterPosition3(tmp));

        if (!pProp->Set(tmp))
        {
            return DEVICE_INVALID_PROPERTY_VALUE;
        }
    }
    return DEVICE_OK;
}

// always read
int CCRISP::OnSum(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        long tmp = 0;
        std::ostringstream command;
        command << addressChar_ << "LK T?";
        RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), ":A"));
        RETURN_ON_MM_ERROR(hub_->ParseAnswerAfterPosition3(tmp));

        if (!pProp->Set(tmp))
        {
            return DEVICE_INVALID_PROPERTY_VALUE;
        }
    }
    return DEVICE_OK;
}

int CCRISP::OnOffset(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    //std::ostringstream command;
    //long tmp = 0;
    if (eAct == MM::BeforeGet)
    {
        if (!refreshProps_ && initialized_ && !refreshOverride_)
        {
            return DEVICE_OK;
        }
        //command << addressChar_ << "LK Z?";
        //RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(), ":A") );
        //RETURN_ON_MM_ERROR ( hub_->ParseAnswerAfterPosition2(tmp) );
        double tmp = 0.0;
        int ret = GetOffset(tmp);
        if (ret != DEVICE_OK)
        {
            return ret;
        }
        if (!pProp->Set(tmp))
        {
            return DEVICE_INVALID_PROPERTY_VALUE;
        }
    }
    return DEVICE_OK;
}

// always read
int CCRISP::OnLogAmpAGC(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      long tmp = 0;
      std::ostringstream command;
      command << addressChar_ << "AL X?";
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), ":A X="));
      RETURN_ON_MM_ERROR ( hub_->ParseAnswerAfterEquals(tmp) );
      if (!pProp->Set(tmp))
      {
          return DEVICE_INVALID_PROPERTY_VALUE;
      }
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

// always read
int CCRISP::OnState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        char tmp = '\0';
        std::ostringstream command;
        command << addressChar_ << "LK X?";
        RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), ":A"));
        RETURN_ON_MM_ERROR(hub_->GetAnswerCharAtPosition3(tmp));
        std::string str(1, tmp);
        if (!pProp->Set(str.c_str()))
        {
            return DEVICE_INVALID_PROPERTY_VALUE;
        }
    }
    return DEVICE_OK;
}

// Provide support for Tiger firmware < 3.40
int CCRISP::OnDitherErrorLegacy(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        std::ostringstream command;
        command << addressChar_ << "EXTRA X?";
        RETURN_ON_MM_ERROR(hub_->QueryCommand(command.str()));
        std::vector<std::string> vReply = hub_->SplitAnswerOnSpace();
        if (vReply.size() <= 2)
        {
            return DEVICE_INVALID_PROPERTY_VALUE;
        }
        if (!pProp->Set(vReply[2].c_str()))
        {
            return DEVICE_INVALID_PROPERTY_VALUE;
        }
    }
    return DEVICE_OK;
}

// Provide support for Tiger firmware < 3.40
int CCRISP::OnSumLegacy(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        std::ostringstream command;
        command << addressChar_ << "EXTRA X?";
        RETURN_ON_MM_ERROR(hub_->QueryCommand(command.str()));
        std::vector<std::string> vReply = hub_->SplitAnswerOnSpace();
        if (vReply.size() <= 2)
        {
            return DEVICE_INVALID_PROPERTY_VALUE;
        }
        if (!pProp->Set(vReply[1].c_str()))
        {
            return DEVICE_INVALID_PROPERTY_VALUE;
        }
    }
    return DEVICE_OK;
}

// Advanced Properties

int CCRISP::OnSetLogAmpAGC(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set("0");
    }
    else if (eAct == MM::AfterSet)
    {
        double logAmpAGC = 0.0;
        pProp->Get(logAmpAGC);
        if (logAmpAGC != 0.0)
        {
            std::ostringstream command;
            command << addressChar_ << "LK M=" << logAmpAGC;
            RETURN_ON_MM_ERROR(hub_->QueryCommandVerify(command.str(), ":A"));
        }
    }
    return DEVICE_OK;
}

int CCRISP::OnSetLockOffset(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set("0");
    }
    else if (eAct == MM::AfterSet)
    {
        double offset = 0.0;
        pProp->Get(offset);
        if (offset != 0.0)
        {
            refreshOverride_ = true;
            std::ostringstream command;
            command << addressChar_ << "LK Z=" << offset;
            const int result = hub_->QueryCommandVerify(command.str(), ":A");
            if (result != DEVICE_OK)
            {
                refreshOverride_ = false;
                return result;
            }
            UpdateProperty(g_CRISPOffsetPropertyName);
            refreshOverride_ = false;
        }
    }
    return DEVICE_OK;
}
