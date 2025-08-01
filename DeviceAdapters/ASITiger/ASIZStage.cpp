///////////////////////////////////////////////////////////////////////////////
// FILE:          ASIZStage.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   ASI motorized one-axis stage device adapter
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
// BASED ON:      ASIStage.cpp and others
//

#include "ASIZStage.h"
#include "ASITiger.h"
#include "ASIHub.h"
#include "ModuleInterface.h"
#include "DeviceUtils.h"
#include "DeviceBase.h"
#include "MMDevice.h"
#include <iostream>
#include <cmath>
#include <sstream>
#include <string>


///////////////////////////////////////////////////////////////////////////////
// CZStage
//
CZStage::CZStage(const char* name) :
   ASIPeripheralBase< ::CStageBase, CZStage >(name),
   unitMult_(g_StageDefaultUnitMult),  // later will try to read actual setting
   stepSizeUm_(g_StageMinStepSize),    // we'll use 1 nm as our smallest possible step size, this is somewhat arbitrary and doesn't change during the program
   axisLetter_(g_EmptyAxisLetterStr),   // value determined by extended name
   advancedPropsEnabled_(false),
   speedTruth_(false),
   lastSpeed_(1.0),
   ring_buffer_supported_(false),
   ring_buffer_capacity_(0),
   ttl_trigger_supported_(false),
   ttl_trigger_enabled_(false),
   runningFastSequence_(false),
   axisIndex_(0)
{
   if (IsExtendedName(name))  // only set up these properties if we have the required information in the name
   {
      axisLetter_ = GetAxisLetterFromExtName(name);
      CreateProperty(g_AxisLetterPropertyName, axisLetter_.c_str(), MM::String, true);
   }
}

int CZStage::Initialize()
{
   // call generic Initialize first, this gets hub
   RETURN_ON_MM_ERROR( PeripheralInitialize() );

   // read the unit multiplier
   // ASI's unit multiplier is how many units per mm, so divide by 1000 here to get units per micron
   // we store the micron-based unit multiplier for MM use, not the mm-based one ASI uses
   std::ostringstream command;
   double tmp;
   command << "UM " << axisLetter_ << "?";
   RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(),":") );
   RETURN_ON_MM_ERROR( hub_->ParseAnswerAfterEquals(tmp) );
   unitMult_ = tmp/1000;
   command.str("");

   // set controller card to return positions with 1 decimal places (3 is max allowed currently, 1 gives 10nm resolution)
   command.str("");
   command << addressChar_ << "VB Z=1";
   RETURN_ON_MM_ERROR ( hub_->QueryCommand(command.str()) );

   // expose the step size to user as read-only property (no need for action handler)
   command.str("");
   command << g_StageMinStepSize;
   CreateProperty(g_StepSizePropertyName , command.str().c_str(), MM::Float, true);

   // create MM description; this doesn't work during hardware configuration wizard but will work afterwards
   command.str("");
   command << g_ZStageDeviceDescription << " Axis=" << axisLetter_ << " HexAddr=" << addressString_;
   CreateProperty(MM::g_Keyword_Description, command.str().c_str(), MM::String, true);

   // min and max motor speeds - read only properties; do this way instead of via to-be-created properties to minimize serial
   //   traffic with updating speed based on speedTruth_ (and seems to do a better job of preserving decimal points)
   command.str("");
   command << "S " << axisLetter_ << "?";
   RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), ":A"));
   double origSpeed;
   RETURN_ON_MM_ERROR( hub_->ParseAnswerAfterEquals(origSpeed) );
   std::ostringstream command2;
   command2 << "S " << axisLetter_ << "=10000";
   RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command2.str(), ":A")); // set too high
   RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), ":A"));  // read actual max
   double maxSpeed;
   RETURN_ON_MM_ERROR( hub_->ParseAnswerAfterEquals(maxSpeed) );
   command2.str("");
   command2 << "S " << axisLetter_ << "=0.000001";
   RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command2.str(), ":A")); // set too low
   RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), ":A"));  // read actual min
   double minSpeed;
   RETURN_ON_MM_ERROR( hub_->ParseAnswerAfterEquals(minSpeed) );
   command2.str("");
   command2 << "S " << axisLetter_ << "=" << origSpeed;
   RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command2.str(), ":A")); // restore
   command2.str("");
   command2 << maxSpeed;
   CreateProperty(g_MaxMotorSpeedPropertyName, command2.str().c_str(), MM::Float, true);
   command2.str("");
   command2 << (minSpeed*1000);
   CreateProperty(g_MinMotorSpeedPropertyName, command2.str().c_str(), MM::Float, true);

   command.str(""); command << "Z2B " << axisLetter_ << "?";
   RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), ":A"));
   RETURN_ON_MM_ERROR ( hub_->ParseAnswerAfterEquals(axisIndex_) );

   // now for properties that are read-write, mostly parameters that set aspects of stage behavior
   // parameters exposed for user to set easily: SL, SU, PC, E, S, AC, WT, MA, JS X=, JS Y=, JS mirror
   // parameters maybe exposed with some hurdle to user: B, OS, AA, AZ, KP, KI, KD, AZ (in OnAdvancedProperties())

   CPropertyAction* pAct;

   // refresh properties from controller every time - default is not to refresh (speeds things up by not redoing so much serial comm)
   pAct = new CPropertyAction (this, &CZStage::OnRefreshProperties);
   CreateProperty(g_RefreshPropValsPropertyName, g_NoState, MM::String, false, pAct);
   AddAllowedValue(g_RefreshPropValsPropertyName, g_NoState);
   AddAllowedValue(g_RefreshPropValsPropertyName, g_YesState);

   // save settings to controller if requested
   pAct = new CPropertyAction (this, &CZStage::OnSaveCardSettings);
   CreateProperty(g_SaveSettingsPropertyName, g_SaveSettingsOrig, MM::String, false, pAct);
   AddAllowedValue(g_SaveSettingsPropertyName, g_SaveSettingsX);
   AddAllowedValue(g_SaveSettingsPropertyName, g_SaveSettingsY);
   AddAllowedValue(g_SaveSettingsPropertyName, g_SaveSettingsZ);
   AddAllowedValue(g_SaveSettingsPropertyName, g_SaveSettingsZJoystick);
   AddAllowedValue(g_SaveSettingsPropertyName, g_SaveSettingsOrig);
   AddAllowedValue(g_SaveSettingsPropertyName, g_SaveSettingsDone);

   // Motor speed (S)
   pAct = new CPropertyAction (this, &CZStage::OnSpeedMicronsPerSec);  // allow reading actual speed at higher precision by using different units
   CreateProperty(g_MotorSpeedMicronsPerSecPropertyName , "1000", MM::Float, true, pAct);  // read-only property updated when speed is set
   pAct = new CPropertyAction (this, &CZStage::OnSpeed);
   CreateProperty(g_MotorSpeedPropertyName, "1", MM::Float, false, pAct);
   SetPropertyLimits(g_MotorSpeedPropertyName, minSpeed, maxSpeed);
   UpdateProperty(g_MotorSpeedPropertyName);

   // Backlash (B)
   pAct = new CPropertyAction (this, &CZStage::OnBacklash);
   CreateProperty(g_BacklashPropertyName, "0", MM::Float, false, pAct);
   UpdateProperty(g_BacklashPropertyName);

   // drift error (E)
   pAct = new CPropertyAction (this, &CZStage::OnDriftError);
   CreateProperty(g_DriftErrorPropertyName, "0", MM::Float, false, pAct);
   UpdateProperty(g_DriftErrorPropertyName);

   // finish error (PC)
   pAct = new CPropertyAction (this, &CZStage::OnFinishError);
   CreateProperty(g_FinishErrorPropertyName, "0", MM::Float, false, pAct);
   UpdateProperty(g_FinishErrorPropertyName);

   // acceleration (AC)
   pAct = new CPropertyAction (this, &CZStage::OnAcceleration);
   CreateProperty(g_AccelerationPropertyName, "0", MM::Integer, false, pAct);
   UpdateProperty(g_AccelerationPropertyName);

   // upper and lower limits (SU and SL)
   pAct = new CPropertyAction (this, &CZStage::OnLowerLim);
   CreateProperty(g_LowerLimPropertyName, "0", MM::Float, false, pAct);
   UpdateProperty(g_LowerLimPropertyName);
   pAct = new CPropertyAction (this, &CZStage::OnUpperLim);
   CreateProperty(g_UpperLimPropertyName, "0", MM::Float, false, pAct);
   UpdateProperty(g_UpperLimPropertyName);

   // maintain behavior (MA)
   pAct = new CPropertyAction (this, &CZStage::OnMaintainState);
   CreateProperty(g_MaintainStatePropertyName, g_StageMaintain_0, MM::String, false, pAct);
   AddAllowedValue(g_MaintainStatePropertyName, g_StageMaintain_0);
   AddAllowedValue(g_MaintainStatePropertyName, g_StageMaintain_1);
   AddAllowedValue(g_MaintainStatePropertyName, g_StageMaintain_2);
   AddAllowedValue(g_MaintainStatePropertyName, g_StageMaintain_3);
   UpdateProperty(g_MaintainStatePropertyName);

   // Wait time, default is 0 (WT)
   pAct = new CPropertyAction (this, &CZStage::OnWaitTime);
   CreateProperty(g_StageWaitTimePropertyName, "0", MM::Integer, false, pAct);
   UpdateProperty(g_StageWaitTimePropertyName);

   // joystick fast speed (JS X=)
   pAct = new CPropertyAction (this, &CZStage::OnJoystickFastSpeed);
   CreateProperty(g_JoystickFastSpeedPropertyName, "100", MM::Float, false, pAct);
   SetPropertyLimits(g_JoystickFastSpeedPropertyName, 0, 100);
   UpdateProperty(g_JoystickFastSpeedPropertyName);

   // joystick slow speed (JS Y=)
   pAct = new CPropertyAction (this, &CZStage::OnJoystickSlowSpeed);
   CreateProperty(g_JoystickSlowSpeedPropertyName, "10", MM::Float, false, pAct);
   SetPropertyLimits(g_JoystickSlowSpeedPropertyName, 0, 100);
   UpdateProperty(g_JoystickSlowSpeedPropertyName);

   // joystick mirror (changes joystick fast/slow speeds to negative)
   pAct = new CPropertyAction (this, &CZStage::OnJoystickMirror);
   CreateProperty(g_JoystickMirrorPropertyName, g_NoState, MM::String, false, pAct);
   AddAllowedValue(g_JoystickMirrorPropertyName, g_NoState);
   AddAllowedValue(g_JoystickMirrorPropertyName, g_YesState);
   UpdateProperty(g_JoystickMirrorPropertyName);

   // joystick disable and select which knob
   pAct = new CPropertyAction (this, &CZStage::OnJoystickSelect);
   CreateProperty(g_JoystickSelectPropertyName, g_JSCode_0, MM::String, false, pAct);
   AddAllowedValue(g_JoystickSelectPropertyName, g_JSCode_0);
   AddAllowedValue(g_JoystickSelectPropertyName, g_JSCode_2);
   AddAllowedValue(g_JoystickSelectPropertyName, g_JSCode_3);
   AddAllowedValue(g_JoystickSelectPropertyName, g_JSCode_22);
   AddAllowedValue(g_JoystickSelectPropertyName, g_JSCode_23);
   UpdateProperty(g_JoystickSelectPropertyName);

   // Motor enable/disable (MC)
   pAct = new CPropertyAction (this, &CZStage::OnMotorControl);
   CreateProperty(g_MotorControlPropertyName, g_OnState, MM::String, false, pAct);
   AddAllowedValue(g_MotorControlPropertyName, g_OnState);
   AddAllowedValue(g_MotorControlPropertyName, g_OffState);
   UpdateProperty(g_MotorControlPropertyName);

   if (FirmwareVersionAtLeast(2.87))  // changed behavior of JS F and T as of v2.87
   {
      // fast wheel speed (JS F) (per-card, not per-axis)
      pAct = new CPropertyAction (this, &CZStage::OnWheelFastSpeed);
      CreateProperty(g_WheelFastSpeedPropertyName, "10", MM::Float, false, pAct);
      SetPropertyLimits(g_WheelFastSpeedPropertyName, 0, 100);
      UpdateProperty(g_WheelFastSpeedPropertyName);

      // slow wheel speed (JS T) (per-card, not per-axis)
      pAct = new CPropertyAction (this, &CZStage::OnWheelSlowSpeed);
      CreateProperty(g_WheelSlowSpeedPropertyName, "5", MM::Float, false, pAct);
      SetPropertyLimits(g_WheelSlowSpeedPropertyName, 0, 100);
      UpdateProperty(g_WheelSlowSpeedPropertyName);

      // wheel mirror (changes wheel fast/slow speeds to negative) (per-card, not per-axis)
      pAct = new CPropertyAction (this, &CZStage::OnWheelMirror);
      CreateProperty(g_WheelMirrorPropertyName, g_NoState, MM::String, false, pAct);
      AddAllowedValue(g_WheelMirrorPropertyName, g_NoState);
      AddAllowedValue(g_WheelMirrorPropertyName, g_YesState);
      UpdateProperty(g_WheelMirrorPropertyName);
   }

   // generates a set of additional advanced properties that are rarely used
   pAct = new CPropertyAction (this, &CZStage::OnAdvancedProperties);
   CreateProperty(g_AdvancedPropertiesPropertyName, g_NoState, MM::String, false, pAct);
   AddAllowedValue(g_AdvancedPropertiesPropertyName, g_NoState);
   AddAllowedValue(g_AdvancedPropertiesPropertyName, g_YesState);
   UpdateProperty(g_AdvancedPropertiesPropertyName);

   // is negative towards sample (ASI firmware convention) or away from sample (Micro-manager convention)
   pAct = new CPropertyAction (this, &CZStage::OnAxisPolarity);
   CreateProperty(g_AxisPolarity, g_FocusPolarityASIDefault, MM::String, false, pAct);
   AddAllowedValue(g_AxisPolarity, g_FocusPolarityASIDefault);
   AddAllowedValue(g_AxisPolarity, g_FocusPolarityMicroManagerDefault);
   UpdateProperty(g_AxisPolarity);

   // get build info so we can add optional properties
   FirmwareBuild build;
   RETURN_ON_MM_ERROR( hub_->GetBuildInfo(addressChar_, build) );

   // populate speedTruth_, which is whether the controller will tell us the actual speed
   if (FirmwareVersionAtLeast(3.27))
   {
      speedTruth_ = ! hub_->IsDefinePresent(build, "SPEED UNTRUTH");
   }
   else  // before v3.27
   {
      speedTruth_ = hub_->IsDefinePresent(build, "SPEED TRUTH");
   }

   // add single-axis properties if supported
   // (single-axis support existed prior pre-2.8 firmware, but now we have easier way to tell if it's present using axis properties
   //   and it wasn't used very much before SPIM)
   if(build.vAxesProps[0] & BIT5)//      if(hub_->IsDefinePresent(build, g_Define_SINGLEAXIS_FUNCTION))
   {
      // copied from ASIMMirror.cpp
      pAct = new CPropertyAction (this, &CZStage::OnSAAmplitude);
      CreateProperty(g_SAAmplitudePropertyName, "0", MM::Float, false, pAct);
      UpdateProperty(g_SAAmplitudePropertyName);
      pAct = new CPropertyAction (this, &CZStage::OnSAOffset);
      CreateProperty(g_SAOffsetPropertyName, "0", MM::Float, false, pAct);
      UpdateProperty(g_SAOffsetPropertyName);
      pAct = new CPropertyAction (this, &CZStage::OnSAPeriod);
      CreateProperty(g_SAPeriodPropertyName, "0", MM::Integer, false, pAct);
      UpdateProperty(g_SAPeriodPropertyName);
      pAct = new CPropertyAction (this, &CZStage::OnSAMode);
      CreateProperty(g_SAModePropertyName, g_SAMode_0, MM::String, false, pAct);
      AddAllowedValue(g_SAModePropertyName, g_SAMode_0);
      AddAllowedValue(g_SAModePropertyName, g_SAMode_1);
      AddAllowedValue(g_SAModePropertyName, g_SAMode_2);
      AddAllowedValue(g_SAModePropertyName, g_SAMode_3);
      UpdateProperty(g_SAModePropertyName);
      pAct = new CPropertyAction (this, &CZStage::OnSAPattern);
      CreateProperty(g_SAPatternPropertyName, g_SAPattern_0, MM::String, false, pAct);
      AddAllowedValue(g_SAPatternPropertyName, g_SAPattern_0);
      AddAllowedValue(g_SAPatternPropertyName, g_SAPattern_1);
      AddAllowedValue(g_SAPatternPropertyName, g_SAPattern_2);
	  if (FirmwareVersionAtLeast(3.14))
	   {	//sin pattern was implemeted much later atleast firmware 3/14 needed
		   AddAllowedValue(g_SAPatternPropertyName, g_SAPattern_3);
	   }
      UpdateProperty(g_SAPatternPropertyName);
      // generates a set of additional advanced properties that are rarely used
      pAct = new CPropertyAction (this, &CZStage::OnSAAdvanced);
      CreateProperty(g_AdvancedSAPropertiesPropertyName, g_NoState, MM::String, false, pAct);
      AddAllowedValue(g_AdvancedSAPropertiesPropertyName, g_NoState);
      AddAllowedValue(g_AdvancedSAPropertiesPropertyName, g_YesState);
      UpdateProperty(g_AdvancedSAPropertiesPropertyName);
   }

   // add ring buffer properties if supported (starting version 2.81)
   if (FirmwareVersionAtLeast(2.81) && (build.vAxesProps[0] & BIT1))
   {
      // get the number of ring buffer positions from the BU X output
      std::string rb_define = hub_->GetDefineString(build, "RING BUFFER");

      ring_buffer_capacity_ = 0;
      if (rb_define.size() > 12)
      {
         ring_buffer_capacity_ = atol(rb_define.substr(11).c_str());
      }

      if (ring_buffer_capacity_ != 0)
      {
         ring_buffer_supported_ = true;

         pAct = new CPropertyAction (this, &CZStage::OnRBMode);
         CreateProperty(g_RB_ModePropertyName, g_RB_OnePoint_1, MM::String, false, pAct);
         AddAllowedValue(g_RB_ModePropertyName, g_RB_OnePoint_1);
         AddAllowedValue(g_RB_ModePropertyName, g_RB_PlayOnce_2);
         AddAllowedValue(g_RB_ModePropertyName, g_RB_PlayRepeat_3);
         UpdateProperty(g_RB_ModePropertyName);

         pAct = new CPropertyAction (this, &CZStage::OnRBDelayBetweenPoints);
         CreateProperty(g_RB_DelayPropertyName, "0", MM::Integer, false, pAct);
         UpdateProperty(g_RB_DelayPropertyName);

         // "do it" property to do TTL trigger via serial
         pAct = new CPropertyAction (this, &CZStage::OnRBTrigger);
         CreateProperty(g_RB_TriggerPropertyName, g_IdleState, MM::String, false, pAct);
         AddAllowedValue(g_RB_TriggerPropertyName, g_IdleState, 0);
         AddAllowedValue(g_RB_TriggerPropertyName, g_DoItState, 1);
         AddAllowedValue(g_RB_TriggerPropertyName, g_DoneState, 2);
         UpdateProperty(g_RB_TriggerPropertyName);

         pAct = new CPropertyAction (this, &CZStage::OnRBRunning);
         CreateProperty(g_RB_AutoplayRunningPropertyName, g_NoState, MM::String, false, pAct);
         AddAllowedValue(g_RB_AutoplayRunningPropertyName, g_NoState);
         AddAllowedValue(g_RB_AutoplayRunningPropertyName, g_YesState);
         UpdateProperty(g_RB_AutoplayRunningPropertyName);

         pAct = new CPropertyAction (this, &CZStage::OnUseSequence);
         CreateProperty(g_UseSequencePropertyName, g_NoState, MM::String, false, pAct);
         AddAllowedValue(g_UseSequencePropertyName, g_NoState);
         AddAllowedValue(g_UseSequencePropertyName, g_YesState);
         ttl_trigger_enabled_ = false;

         pAct = new CPropertyAction (this, &CZStage::OnFastSequence);
         CreateProperty(g_UseFastSequencePropertyName, g_NoState, MM::String, false, pAct);
         AddAllowedValue(g_UseFastSequencePropertyName, g_NoState);
         AddAllowedValue(g_UseFastSequencePropertyName, g_ArmedState);
         runningFastSequence_ = false;
      }

   }

   if (FirmwareVersionAtLeast(3.09) && (hub_->IsDefinePresent(build, "IN0_INT"))
         && ring_buffer_supported_)
   {
      ttl_trigger_supported_ = true;

      // Vik implemented a similar property for tuneable lens as integer-valued, but this is string-valued with slightly different name
      // would be nice if property handler automatically parsed things so you could use either string or integer from script one but this appears impossible
      pAct = new CPropertyAction (this, &CZStage::OnTTLInputMode);
      CreateProperty(g_TTLInputModeName, g_TTLInputMode_0, MM::String, false, pAct);
      AddAllowedValue(g_TTLInputModeName, g_TTLInputMode_0, 0);
      AddAllowedValue(g_TTLInputModeName, g_TTLInputMode_1, 1);
      AddAllowedValue(g_TTLInputModeName, g_TTLInputMode_2, 2);
      AddAllowedValue(g_TTLInputModeName, g_TTLInputMode_7, 7);
      UpdateProperty(g_TTLInputModeName);

   }

   //VectorMove
   pAct = new CPropertyAction (this, &CZStage::OnVector);
   CreateProperty(g_VectorPropertyName, "0", MM::Float, false, pAct);
   SetPropertyLimits(g_VectorPropertyName, maxSpeed*-1, maxSpeed);
   UpdateProperty(g_VectorPropertyName);

   initialized_ = true;
   return DEVICE_OK;
}

int CZStage::GetPositionUm(double& pos)
{
   std::ostringstream command;
   command << "W " << axisLetter_;
   RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(),":A") );
   RETURN_ON_MM_ERROR ( hub_->ParseAnswerAfterPosition2(pos) );
   pos = pos/unitMult_;
   return DEVICE_OK;
}

int CZStage::SetPositionUm(double pos)
{
   std::ostringstream command;
   command << "M " << axisLetter_ << "=" << pos*unitMult_;
   return hub_->QueryCommandVerify(command.str(),":A");
}

int CZStage::GetPositionSteps(long& steps)
{
   std::ostringstream command;
   command << "W " << axisLetter_;
   RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(),":A") );
   double tmp;
   RETURN_ON_MM_ERROR ( hub_->ParseAnswerAfterPosition2(tmp) );
   steps = (long)(tmp/unitMult_/stepSizeUm_);
   return DEVICE_OK;
}

int CZStage::SetPositionSteps(long steps)
{
   std::ostringstream command;
   command << "M " << axisLetter_ << "=" << steps*unitMult_*stepSizeUm_;
   return hub_->QueryCommandVerify(command.str(),":A");
}

int CZStage::SetRelativePositionUm(double d)
{
   std::ostringstream command;
   command << "R " << axisLetter_ << "=" << d*unitMult_;
   return hub_->QueryCommandVerify(command.str(),":A");
}

int CZStage::GetLimits(double& min, double& max)
{
   // ASI limits are always reported in terms of mm, independent of unit multiplier
   std::ostringstream command;
   command << "SL " << axisLetter_ << "?";
   double tmp;
   RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(),":A") );
   RETURN_ON_MM_ERROR( hub_->ParseAnswerAfterEquals(tmp) );
   min = tmp*1000;
   command.str("");
   command << "SU " << axisLetter_ << "?";
   RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(),":A") );
   RETURN_ON_MM_ERROR( hub_->ParseAnswerAfterEquals(tmp) );
   max = tmp*1000;
   return DEVICE_OK;
}

int CZStage::Stop()
{
   // note this stops the card (including if there are other stages on same card), \ stops all stages
   std::ostringstream command;
   command << addressChar_ << "halt";
   return hub_->QueryCommand(command.str());
}

int CZStage::Home()
{
   // single-axis possible in recent firmware of motorized Z but rarely
   // used so for now don't support in Micro-Manager; when added to the
   // device adapter then stop move here like in ASIPiezo
   std::ostringstream command;
   command << "! " << axisLetter_;
   RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(), ":A") );
   return DEVICE_OK;
}

bool CZStage::Busy()
{
   std::ostringstream command;
   if (runningFastSequence_)
   {
      return false;
   }
   if (FirmwareVersionAtLeast(2.7)) // can use more accurate RS <axis>?
   {
      command << "RS " << axisLetter_ << "?";
      if (hub_->QueryCommandVerify(command.str(),":A") != DEVICE_OK)  // say we aren't busy if we can't communicate
         return false;
      char c;
      if (hub_->GetAnswerCharAtPosition3(c) != DEVICE_OK)
         return false;
      return (c == 'B');
   }
   else  // use LSB of the status byte as approximate status, not quite equivalent
   {
      command << "RS " << axisLetter_;
      if (hub_->QueryCommandVerify(command.str(),":A") != DEVICE_OK)  // say we aren't busy if we can't communicate
         return false;
      unsigned int i;
      if (hub_->ParseAnswerAfterPosition2(i) != DEVICE_OK)  // say we aren't busy if we can't parse
         return false;
      return (i & (int)BIT0);  // mask everything but LSB
   }
}

int CZStage::SetOrigin()
{
   std::ostringstream command;
   command << "H " << axisLetter_ << "=0";
   return hub_->QueryCommandVerify(command.str(),":A");
}

// Disables TTL triggering; doesn't actually stop anything already happening on controller
int CZStage::StopStageSequence()
{
   std::ostringstream command;
   if (runningFastSequence_)
   {
      return DEVICE_OK;
   }
   if (!ttl_trigger_supported_)
   {
      return DEVICE_UNSUPPORTED_COMMAND;
   }
   command << addressChar_ << "TTL X=0";  // switch off TTL triggering
   RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(),":A") );
   return DEVICE_OK;
}

// Enables TTL triggering; doesn't actually start anything going on controller
int CZStage::StartStageSequence()
{
   std::ostringstream command;
   if (runningFastSequence_)
   {
      return DEVICE_OK;
   }
   if (!ttl_trigger_supported_)
   {
      return DEVICE_UNSUPPORTED_COMMAND;
   }
   // ensure that ringbuffer pointer points to first entry and
   // that we only trigger the first axis (assume only 1 axis on piezo card)
   // TODO fix this so it doesn't assume 1st axis on card, this is copy/paste remnant from piezo
   command << addressChar_ << "RM Y=1 Z=0";
   RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(),":A") );

   command.str("");
   command << addressChar_ << "TTL X=1";  // switch on TTL triggering
   RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(),":A") );
   return DEVICE_OK;
}

int CZStage::SendStageSequence()
{
   std::ostringstream command;
   if (runningFastSequence_)
   {
      return DEVICE_OK;
   }
   if (!ttl_trigger_supported_)
   {
      return DEVICE_UNSUPPORTED_COMMAND;
   }
   command << addressChar_ << "RM X=0"; // clear ring buffer
   RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(),":A") );
   for (unsigned i=0; i< sequence_.size(); i++)  // send new points
   {
      command.str("");
      command << "LD " << axisLetter_ << "=" << sequence_[i]*unitMult_;
      RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(),":A") );
   }

   // turn off ring buffer on all other axes; this will be the right thing 90% of the time
   unsigned int mask = 1 << axisIndex_;
   command.str("");
   command << addressChar_ << "RM Y=" << mask;
   RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(),":A") );

   return DEVICE_OK;
}

int CZStage::ClearStageSequence()
{
   std::ostringstream command;
   if (runningFastSequence_)
   {
      return DEVICE_OK;
   }
   if (!ttl_trigger_supported_)
   {
      return DEVICE_UNSUPPORTED_COMMAND;
   }
   sequence_.clear();
   command << addressChar_ << "RM X=0";  // clear ring buffer
   RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(),":A") );
   return DEVICE_OK;
}

int CZStage::AddToStageSequence(double position)
{
   if (runningFastSequence_)
   {
      return DEVICE_OK;
   }
   if (!ttl_trigger_supported_)
   {
      return DEVICE_UNSUPPORTED_COMMAND;
   }
   sequence_.push_back(position);
   return DEVICE_OK;
}

int CZStage::Move(double velocity)
{
    std::ostringstream command;
    command << "VE " << axisLetter_ << "=" << velocity;
    return hub_->QueryCommandVerify(command.str(), ":A");
}

// action handlers

// redoes the joystick settings so they can be saved using SS Z
int CZStage::OnSaveJoystickSettings()
{
   long tmp;
   std::string tmpstr;
   std::ostringstream command;
   std::ostringstream response;
   command << "J " << axisLetter_ << "?";
   response << ":A " << axisLetter_ << "=";
   RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), response.str()));
   RETURN_ON_MM_ERROR( hub_->ParseAnswerAfterEquals(tmp) );
   tmp += 100;
   command.str("");
   command << "J " << axisLetter_ << "=" << tmp;
   RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), ":A"));
   return DEVICE_OK;
}

int CZStage::OnSaveCardSettings(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   std::string tmpstr;
   std::ostringstream command;
   if (eAct == MM::AfterSet) {
      if (hub_->UpdatingSharedProperties())
         return DEVICE_OK;
      command << addressChar_ << "SS ";
      pProp->Get(tmpstr);
      if (tmpstr == g_SaveSettingsOrig)
         return DEVICE_OK;
      if (tmpstr == g_SaveSettingsDone)
         return DEVICE_OK;
      if (tmpstr == g_SaveSettingsX)
         command << 'X';
      else if (tmpstr == g_SaveSettingsY)
         command << 'Y';
      else if (tmpstr == g_SaveSettingsZ)
         command << 'Z';
      else if (tmpstr == g_SaveSettingsZJoystick)
      {
         command << 'Z';
         // do save joystick settings first
         RETURN_ON_MM_ERROR (OnSaveJoystickSettings());
      }
      RETURN_ON_MM_ERROR (hub_->QueryCommandVerify(command.str(), ":A", (long)200));  // note added 200ms delay
      pProp->Set(g_SaveSettingsDone);
      command.str(""); command << g_SaveSettingsDone;
      RETURN_ON_MM_ERROR ( hub_->UpdateSharedProperties(addressChar_, pProp->GetName(), command.str()) );
   }
   return DEVICE_OK;
}

int CZStage::OnRefreshProperties(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::AfterSet)
    {
        std::string tmpstr;
        pProp->Get(tmpstr);
        refreshProps_ = (tmpstr == g_YesState) ? true : false;
    }
    return DEVICE_OK;
}

// special property, when set to "yes" it creates a set of little-used properties that can be manipulated thereafter
// these parameters exposed with some hurdle to user: B, OS, AA, AZ, KP, KI, KD, AZ
int CZStage::OnAdvancedProperties(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      return DEVICE_OK; // do nothing
   }
   else if (eAct == MM::AfterSet)
   {
      std::string tmpstr;
      pProp->Get(tmpstr);
      if (tmpstr == g_YesState && !advancedPropsEnabled_)  // after creating advanced properties once no need to repeat
      {
         CPropertyAction* pAct;
         advancedPropsEnabled_ = true;

         // make sure that the new properties are initialized, set to true at the end of creating them
         initialized_ = false;

         // overshoot (OS)
         pAct = new CPropertyAction (this, &CZStage::OnOvershoot);
         CreateProperty(g_OvershootPropertyName, "0", MM::Float, false, pAct);
         UpdateProperty(g_OvershootPropertyName);

         // servo integral term (KI)
         pAct = new CPropertyAction (this, &CZStage::OnKIntegral);
         CreateProperty(g_KIntegralPropertyName, "0", MM::Integer, false, pAct);
         UpdateProperty(g_KIntegralPropertyName);

         // servo proportional term (KP)
         pAct = new CPropertyAction (this, &CZStage::OnKProportional);
         CreateProperty(g_KProportionalPropertyName, "0", MM::Integer, false, pAct);
         UpdateProperty(g_KProportionalPropertyName);

         // servo derivative term (KD)
         pAct = new CPropertyAction (this, &CZStage::OnKDerivative);
         CreateProperty(g_KDerivativePropertyName, "0", MM::Integer, false, pAct);
         UpdateProperty(g_KDerivativePropertyName);

         // motor proportional term (KV)
         pAct = new CPropertyAction (this, &CZStage::OnKDrive);
         CreateProperty(g_KDrivePropertyName, "0", MM::Integer, false, pAct);
         UpdateProperty(g_KDrivePropertyName);

         // motor feedforward term (KA)
         pAct = new CPropertyAction (this, &CZStage::OnKFeedforward);
         CreateProperty(g_KFeedforwardPropertyName, "0", MM::Integer, false, pAct);
         UpdateProperty(g_KFeedforwardPropertyName);

         // Align calibration/setting for pot in drive electronics (AA)
         pAct = new CPropertyAction (this, &CZStage::OnAAlign);
         CreateProperty(g_AAlignPropertyName, "0", MM::Integer, false, pAct);
         UpdateProperty(g_AAlignPropertyName);

         // Autozero drive electronics (AZ)
         pAct = new CPropertyAction (this, &CZStage::OnAZero);
         CreateProperty(g_AZeroXPropertyName, "0", MM::String, false, pAct);
         UpdateProperty(g_AZeroXPropertyName);

         initialized_ = true;
      }
   }
   return DEVICE_OK;
}

int CZStage::OnWaitTime(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   std::ostringstream command;
   std::ostringstream response;
   long tmp = 0;
   if (eAct == MM::BeforeGet)
   {
      if (!refreshProps_ && initialized_)
         return DEVICE_OK;
      command << "WT " << axisLetter_ << "?";
      response << ":" << axisLetter_ << "=";
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), response.str()));
      RETURN_ON_MM_ERROR ( hub_->ParseAnswerAfterEquals(tmp) );
      pProp->Set(tmp);
   }
   else if (eAct == MM::AfterSet) {
      pProp->Get(tmp);
      command << "WT " << axisLetter_ << "=" << tmp;
      RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(), ":A") );
   }
   return DEVICE_OK;
}

int CZStage::OnSpeedMicronsPerSec(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet || eAct == MM::AfterSet)
   {
      if (!pProp->Set(lastSpeed_*1000))
         return DEVICE_INVALID_PROPERTY_VALUE;
   }
   return DEVICE_OK;
}

int CZStage::OnSpeed(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   std::ostringstream command;
   std::ostringstream response;
   double tmp = 0;
   if (eAct == MM::BeforeGet)
   {
      if (!refreshProps_ && initialized_ && !refreshOverride_)
         return DEVICE_OK;
      refreshOverride_ = false;
      command << "S " << axisLetter_ << "?";
      response << ":A " << axisLetter_ << "=";
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), response.str()) );
      RETURN_ON_MM_ERROR( hub_->ParseAnswerAfterEquals(tmp) );
      if (!pProp->Set(tmp))
         return DEVICE_INVALID_PROPERTY_VALUE;
      lastSpeed_ = tmp;
      RETURN_ON_MM_ERROR( SetProperty(g_MotorSpeedMicronsPerSecPropertyName, "1") );  // set to a dummy value, will read from lastSpeed_ variable
   }
   else if (eAct == MM::AfterSet) {
      pProp->Get(tmp);
      command << "S " << axisLetter_ << "=" << tmp;
      RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(), ":A") );
      if (speedTruth_) {
         refreshOverride_ = true;
         return OnSpeed(pProp, MM::BeforeGet);
      }
      else
      {
         lastSpeed_ = tmp;
         RETURN_ON_MM_ERROR( SetProperty(g_MotorSpeedMicronsPerSecPropertyName, "1") );  // set to a dummy value, will read from lastSpeedX_ variable
      }
   }
   return DEVICE_OK;
}

// Note: ASI units are in millimeters but MM units are in micrometers
int CZStage::OnDriftError(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   std::ostringstream command;
   std::ostringstream response;
   double tmp = 0;
   if (eAct == MM::BeforeGet)
   {
      if (!refreshProps_ && initialized_)
         return DEVICE_OK;
      command << "E " << axisLetter_ << "?";
      response << ":" << axisLetter_ << "=";
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), response.str()));
      RETURN_ON_MM_ERROR( hub_->ParseAnswerAfterEquals(tmp) );
      tmp = 1000*tmp;
      if (!pProp->Set(tmp))
         return DEVICE_INVALID_PROPERTY_VALUE;
   }
   else if (eAct == MM::AfterSet) {
      pProp->Get(tmp);
      command << "E " << axisLetter_ << "=" << tmp/1000;
      RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(), ":A") );
   }
   return DEVICE_OK;
}

// Note: ASI units are in millimeters but MM units are in micrometers
int CZStage::OnFinishError(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   std::ostringstream command;
   std::ostringstream response;
   double tmp = 0;
   if (eAct == MM::BeforeGet)
   {
      if (!refreshProps_ && initialized_)
         return DEVICE_OK;
      command << "PC " << axisLetter_ << "?";
      response << ":A " << axisLetter_ << "=";
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), response.str()));
      RETURN_ON_MM_ERROR( hub_->ParseAnswerAfterEquals(tmp) );
      tmp = 1000*tmp;
      if (!pProp->Set(tmp))
         return DEVICE_INVALID_PROPERTY_VALUE;
   }
   else if (eAct == MM::AfterSet) {
      pProp->Get(tmp);
      command << "PC " << axisLetter_ << "=" << tmp/1000;
      RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(), ":A") );
   }
   return DEVICE_OK;
}

int CZStage::OnLowerLim(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   std::ostringstream command;
   std::ostringstream response;
   double tmp = 0;
   if (eAct == MM::BeforeGet)
   {
      if (!refreshProps_ && initialized_)
         return DEVICE_OK;
      command << "SL " << axisLetter_ << "?";
      response << ":A " << axisLetter_ << "=";
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), response.str()));
      RETURN_ON_MM_ERROR( hub_->ParseAnswerAfterEquals(tmp) );
      if (!pProp->Set(tmp))
         return DEVICE_INVALID_PROPERTY_VALUE;
   }
   else if (eAct == MM::AfterSet) {
      pProp->Get(tmp);
      command << "SL " << axisLetter_ << "=" << tmp;
      RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(), ":A") );
   }
   return DEVICE_OK;
}

int CZStage::OnUpperLim(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   std::ostringstream command;
   std::ostringstream response;
   double tmp = 0;
   if (eAct == MM::BeforeGet)
   {
      if (!refreshProps_ && initialized_)
         return DEVICE_OK;
      command << "SU " << axisLetter_ << "?";
      response << ":A " << axisLetter_ << "=";
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), response.str()));
      RETURN_ON_MM_ERROR( hub_->ParseAnswerAfterEquals(tmp) );
      if (!pProp->Set(tmp))
         return DEVICE_INVALID_PROPERTY_VALUE;
   }
   else if (eAct == MM::AfterSet) {
      pProp->Get(tmp);
      command << "SU " << axisLetter_ << "=" << tmp;
      RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(), ":A") );
   }
   return DEVICE_OK;
}

int CZStage::OnAcceleration(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   std::ostringstream command;
   long tmp = 0;
   if (eAct == MM::BeforeGet)
   {
      if (!refreshProps_ && initialized_)
         return DEVICE_OK;
      command << "AC " << axisLetter_ << "?";
      std::ostringstream response;
      response << ":" << axisLetter_ << "=";
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), response.str()));
      RETURN_ON_MM_ERROR ( hub_->ParseAnswerAfterEquals(tmp) );
      if (!pProp->Set(tmp))
         return DEVICE_INVALID_PROPERTY_VALUE;
   }
   else if (eAct == MM::AfterSet) {
      pProp->Get(tmp);
      command << "AC " << axisLetter_ << "=" << tmp;
      RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(), ":A") );
   }
   return DEVICE_OK;
}

int CZStage::OnMaintainState(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   std::ostringstream command;
   long tmp = 0;
   if (eAct == MM::BeforeGet)
   {
      if (!refreshProps_ && initialized_)
         return DEVICE_OK;
      command << "MA " << axisLetter_ << "?";
      std::ostringstream response;
      response << ":A " << axisLetter_ << "=";
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), response.str()));
      RETURN_ON_MM_ERROR ( hub_->ParseAnswerAfterEquals(tmp) );
      bool success = 0;
      switch (tmp)
      {
         case 0: success = pProp->Set(g_StageMaintain_0); break;
         case 1: success = pProp->Set(g_StageMaintain_1); break;
         case 2: success = pProp->Set(g_StageMaintain_2); break;
         case 3: success = pProp->Set(g_StageMaintain_3); break;
         default:success = 0;                             break;
      }
      if (!success)
         return DEVICE_INVALID_PROPERTY_VALUE;
   }
   else if (eAct == MM::AfterSet)
   {
      std::string tmpstr;
      pProp->Get(tmpstr);
      if (tmpstr == g_StageMaintain_0)
         tmp = 0;
      else if (tmpstr == g_StageMaintain_1)
         tmp = 1;
      else if (tmpstr == g_StageMaintain_2)
         tmp = 2;
      else if (tmpstr == g_StageMaintain_3)
         tmp = 3;
      else
         return DEVICE_INVALID_PROPERTY_VALUE;
      command << "MA " << axisLetter_ << "=" << tmp;
      RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(), ":A") );
   }
   return DEVICE_OK;
}

// Note: ASI units are in millimeters but MM units are in micrometers
int CZStage::OnBacklash(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   std::ostringstream command;
   std::ostringstream response;
   double tmp = 0;
   if (eAct == MM::BeforeGet)
   {
      if (!refreshProps_ && initialized_)
         return DEVICE_OK;
      command << "B " << axisLetter_ << "?";
      response << ":" << axisLetter_ << "=";
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), response.str()));
      RETURN_ON_MM_ERROR( hub_->ParseAnswerAfterEquals(tmp) );
      tmp = 1000*tmp;
      if (!pProp->Set(tmp))
         return DEVICE_INVALID_PROPERTY_VALUE;
   }
   else if (eAct == MM::AfterSet) {
      pProp->Get(tmp);
      command << "B " << axisLetter_ << "=" << tmp/1000;
      RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(), ":A") );
   }
   return DEVICE_OK;
}

int CZStage::OnOvershoot(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   std::ostringstream command;
   std::ostringstream response;
   double tmp = 0;
   if (eAct == MM::BeforeGet)
   {
      if (!refreshProps_ && initialized_)
         return DEVICE_OK;
      command << "OS " << axisLetter_ << "?";
      response << ":A " << axisLetter_ << "=";
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), response.str()));
      RETURN_ON_MM_ERROR( hub_->ParseAnswerAfterEquals(tmp) );
      tmp = 1000*tmp;
      if (!pProp->Set(tmp))
         return DEVICE_INVALID_PROPERTY_VALUE;
   }
   else if (eAct == MM::AfterSet) {
      pProp->Get(tmp);
      command << "OS " << axisLetter_ << "=" << tmp/1000;
      RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(), ":A") );
   }
   return DEVICE_OK;
}

int CZStage::OnKIntegral(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   std::ostringstream command;
   std::ostringstream response;
   long tmp = 0;
   if (eAct == MM::BeforeGet)
   {
      if (!refreshProps_ && initialized_)
         return DEVICE_OK;
      command << "KI " << axisLetter_ << "?";
      response << ":A " << axisLetter_ << "=";
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), response.str()));
      RETURN_ON_MM_ERROR ( hub_->ParseAnswerAfterEquals(tmp) );
      if (!pProp->Set(tmp))
         return DEVICE_INVALID_PROPERTY_VALUE;
   }
   else if (eAct == MM::AfterSet) {
      pProp->Get(tmp);
      command << "KI " << axisLetter_ << "=" << tmp;
      RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(), ":A") );
   }
   return DEVICE_OK;
}

int CZStage::OnKProportional(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   std::ostringstream command;
   std::ostringstream response;
   long tmp = 0;
   if (eAct == MM::BeforeGet)
   {
      if (!refreshProps_ && initialized_)
         return DEVICE_OK;
      command << "KP " << axisLetter_ << "?";
      response << ":A " << axisLetter_ << "=";
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), response.str()));
      RETURN_ON_MM_ERROR ( hub_->ParseAnswerAfterEquals(tmp) );
      if (!pProp->Set(tmp))
         return DEVICE_INVALID_PROPERTY_VALUE;
   }
   else if (eAct == MM::AfterSet) {
      pProp->Get(tmp);
      command << "KP " << axisLetter_ << "=" << tmp;
      RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(), ":A") );
   }
   return DEVICE_OK;
}

int CZStage::OnKDerivative(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   std::ostringstream command;
   std::ostringstream response;
   long tmp = 0;
   if (eAct == MM::BeforeGet)
   {
      if (!refreshProps_ && initialized_)
         return DEVICE_OK;
      command << "KD " << axisLetter_ << "?";
      response << ":A " << axisLetter_ << "=";
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), response.str()));
      RETURN_ON_MM_ERROR ( hub_->ParseAnswerAfterEquals(tmp) );
      if (!pProp->Set(tmp))
         return DEVICE_INVALID_PROPERTY_VALUE;
   }
   else if (eAct == MM::AfterSet) {
      pProp->Get(tmp);
      command << "KD " << axisLetter_ << "=" << tmp;
      RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(), ":A") );
   }
   return DEVICE_OK;
}

int CZStage::OnKDrive(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   std::ostringstream command;
   std::ostringstream response;
   long tmp = 0;
   if (eAct == MM::BeforeGet)
   {
      if (!refreshProps_ && initialized_)
         return DEVICE_OK;
      command << "KV " << axisLetter_ << "?";
      response << ":A " << axisLetter_ << "=";
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), response.str()));
      RETURN_ON_MM_ERROR ( hub_->ParseAnswerAfterEquals(tmp) );
      if (!pProp->Set(tmp))
         return DEVICE_INVALID_PROPERTY_VALUE;
   }
   else if (eAct == MM::AfterSet) {
      pProp->Get(tmp);
      command << "KV " << axisLetter_ << "=" << tmp;
      RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(), ":A") );
   }
   return DEVICE_OK;
}

int CZStage::OnKFeedforward(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   std::ostringstream command;
   std::ostringstream response;
   long tmp = 0;
   if (eAct == MM::BeforeGet)
   {
      if (!refreshProps_ && initialized_)
         return DEVICE_OK;
      command << "KA " << axisLetter_ << "?";
      response << ":A " << axisLetter_ << "=";
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), response.str()));
      RETURN_ON_MM_ERROR ( hub_->ParseAnswerAfterEquals(tmp) );
      if (!pProp->Set(tmp))
         return DEVICE_INVALID_PROPERTY_VALUE;
   }
   else if (eAct == MM::AfterSet) {
      pProp->Get(tmp);
      command << "KA " << axisLetter_ << "=" << tmp;
      RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(), ":A") );
   }
   return DEVICE_OK;
}

int CZStage::OnAAlign(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   std::ostringstream command;
   std::ostringstream response;
   long tmp = 0;
   if (eAct == MM::BeforeGet)
   {
      if (!refreshProps_ && initialized_)
         return DEVICE_OK;
      command << "AA " << axisLetter_ << "?";
      response << ":A " << axisLetter_ << "=";
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), response.str()));
      RETURN_ON_MM_ERROR ( hub_->ParseAnswerAfterEquals(tmp) );
      if (!pProp->Set(tmp))
         return DEVICE_INVALID_PROPERTY_VALUE;
   }
   else if (eAct == MM::AfterSet) {
      pProp->Get(tmp);
      command << "AA " << axisLetter_ << "=" << tmp;
      RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(), ":A") );
   }
   return DEVICE_OK;
}

// On property change the AZ command is issued, and the reported result becomes the property value
int CZStage::OnAZero(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   std::ostringstream command;
   std::ostringstream response;
   if (eAct == MM::BeforeGet)
   {
      return DEVICE_OK; // do nothing
   }
   else if (eAct == MM::AfterSet) {
      command << "AZ " << axisLetter_;
      RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(), ":A") );
      // last line has result, echo result to user as property
      std::vector<std::string> vReply = hub_->SplitAnswerOnCR();
      if (!pProp->Set(vReply.back().c_str()))
         return DEVICE_INVALID_PROPERTY_VALUE;
   }
   return DEVICE_OK;
}

int CZStage::OnMotorControl(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   std::ostringstream command;
   std::ostringstream response;
   long tmp = 0;
   if (eAct == MM::BeforeGet)
   {
      if (!refreshProps_ && initialized_)
         return DEVICE_OK;
      command << "MC " << axisLetter_ << "?";
      response << ":A ";
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), response.str()));
      RETURN_ON_MM_ERROR( hub_->ParseAnswerAfterPosition3(tmp));
      bool success = 0;
      if (tmp)
         success = pProp->Set(g_OnState);
      else
         success = pProp->Set(g_OffState);
      if (!success)
         return DEVICE_INVALID_PROPERTY_VALUE;
   }
   else if (eAct == MM::AfterSet)
   {
      std::string tmpstr;
      pProp->Get(tmpstr);
      if (tmpstr == g_OffState)
         command << "MC " << axisLetter_ << "-";
      else
         command << "MC " << axisLetter_ << "+";
      RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(), ":A") );
   }
   return DEVICE_OK;
}

// ASI controller mirrors by having negative speed, but here we have separate property for mirroring
//   and for speed (which is strictly positive)... that makes this code a bit odd
// note that this setting is per-card, not per-axis
int CZStage::OnJoystickFastSpeed(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   std::ostringstream command;
   std::ostringstream response;
   double tmp = 0;
   if (eAct == MM::BeforeGet)
   {
      if (!refreshProps_ && initialized_)
         return DEVICE_OK;
      command << addressChar_ << "JS X?";
      response << ":A X=";
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), response.str()));
      RETURN_ON_MM_ERROR( hub_->ParseAnswerAfterEquals(tmp) );
      tmp = abs(tmp);
      if (!pProp->Set(tmp))
         return DEVICE_INVALID_PROPERTY_VALUE;
   }
   else if (eAct == MM::AfterSet) {
      if (hub_->UpdatingSharedProperties())
         return DEVICE_OK;
      pProp->Get(tmp);
      char joystickMirror[MM::MaxStrLength];
      RETURN_ON_MM_ERROR ( GetProperty(g_JoystickMirrorPropertyName, joystickMirror) );
      if (strcmp(joystickMirror, g_YesState) == 0)
         command << addressChar_ << "JS X=-" << tmp;
      else
         command << addressChar_ << "JS X=" << tmp;
      RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(), ":A") );
      command.str(""); command << tmp;
      RETURN_ON_MM_ERROR ( hub_->UpdateSharedProperties(addressChar_, pProp->GetName(), command.str()) );
   }
   return DEVICE_OK;
}

// ASI controller mirrors by having negative speed, but here we have separate property for mirroring
//   and for speed (which is strictly positive)... that makes this code a bit odd
// note that this setting is per-card, not per-axis
int CZStage::OnJoystickSlowSpeed(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   std::ostringstream command;
   std::ostringstream response;
   double tmp = 0;
   if (eAct == MM::BeforeGet)
   {
      if (!refreshProps_ && initialized_)
         return DEVICE_OK;
      command << addressChar_ << "JS Y?";
      response << ":A Y=";
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), response.str()));
      RETURN_ON_MM_ERROR( hub_->ParseAnswerAfterEquals(tmp) );
      tmp = abs(tmp);
      if (!pProp->Set(tmp))
         return DEVICE_INVALID_PROPERTY_VALUE;
   }
   else if (eAct == MM::AfterSet) {
      if (hub_->UpdatingSharedProperties())
         return DEVICE_OK;
      pProp->Get(tmp);
      char joystickMirror[MM::MaxStrLength];
      RETURN_ON_MM_ERROR ( GetProperty(g_JoystickMirrorPropertyName, joystickMirror) );
      if (strcmp(joystickMirror, g_YesState) == 0)
         command << addressChar_ << "JS Y=-" << tmp;
      else
         command << addressChar_ << "JS Y=" << tmp;
      RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(), ":A") );
      command.str(""); command << tmp;
      RETURN_ON_MM_ERROR ( hub_->UpdateSharedProperties(addressChar_, pProp->GetName(), command.str()) );
   }
   return DEVICE_OK;
}

// ASI controller mirrors by having negative speed, but here we have separate property for mirroring
//   and for speed (which is strictly positive)... that makes this code a bit odd
// note that this setting is per-card, not per-axis
int CZStage::OnJoystickMirror(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   std::ostringstream command;
   std::ostringstream response;
   double tmp = 0;
   if (eAct == MM::BeforeGet)
   {
      if (!refreshProps_ && initialized_)
         return DEVICE_OK;
      command << addressChar_ << "JS X?";  // query only the fast setting to see if already mirrored
      response << ":A X=";
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), response.str()));
      RETURN_ON_MM_ERROR( hub_->ParseAnswerAfterEquals(tmp) );
      bool success = 0;
      if (tmp < 0) // speed negative <=> mirrored
         success = pProp->Set(g_YesState);
      else
         success = pProp->Set(g_NoState);
      if (!success)
         return DEVICE_INVALID_PROPERTY_VALUE;
   }
   else if (eAct == MM::AfterSet) {
      if (hub_->UpdatingSharedProperties())
         return DEVICE_OK;
      std::string tmpstr;
      pProp->Get(tmpstr);
      double joystickFast = 0.0;
      RETURN_ON_MM_ERROR ( GetProperty(g_JoystickFastSpeedPropertyName, joystickFast) );
      double joystickSlow = 0.0;
      RETURN_ON_MM_ERROR ( GetProperty(g_JoystickSlowSpeedPropertyName, joystickSlow) );
      if (tmpstr == g_YesState)
         command << addressChar_ << "JS X=-" << joystickFast << " Y=-" << joystickSlow;
      else
         command << addressChar_ << "JS X=" << joystickFast << " Y=" << joystickSlow;
      RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(), ":A") );
      RETURN_ON_MM_ERROR ( hub_->UpdateSharedProperties(addressChar_, pProp->GetName(), tmpstr.c_str()) );
   }
   return DEVICE_OK;
}

int CZStage::OnJoystickSelect(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   std::ostringstream command;
   std::ostringstream response;
   long tmp = 0;
   if (eAct == MM::BeforeGet)
   {
      if (!refreshProps_ && initialized_)
         return DEVICE_OK;
      command << "J " << axisLetter_ << "?";
      response << ":A " << axisLetter_ << "=";
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), response.str()));
      RETURN_ON_MM_ERROR ( hub_->ParseAnswerAfterEquals(tmp) );
      bool success = 0;
      switch (tmp)
      {
         case 0: success = pProp->Set(g_JSCode_0); break;
         case 1: success = pProp->Set(g_JSCode_1); break;
         case 2: success = pProp->Set(g_JSCode_2); break;
         case 3: success = pProp->Set(g_JSCode_3); break;
         case 22: success = pProp->Set(g_JSCode_22); break;
         case 23: success = pProp->Set(g_JSCode_23); break;
         default: success=0;
      }
      // don't complain if value is unsupported, just leave as-is
   }
   else if (eAct == MM::AfterSet)
   {
      std::string tmpstr;
      pProp->Get(tmpstr);
      if (tmpstr == g_JSCode_0)
         tmp = 0;
      else if (tmpstr == g_JSCode_1)
         tmp = 1;
      else if (tmpstr == g_JSCode_2)
         tmp = 2;
      else if (tmpstr == g_JSCode_3)
         tmp = 3;
      else if (tmpstr == g_JSCode_22)
         tmp = 22;
      else if (tmpstr == g_JSCode_23)
         tmp = 23;
      else
         return DEVICE_INVALID_PROPERTY_VALUE;
      command << "J " << axisLetter_ << "=" << tmp;
      RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(), ":A") );
   }
   return DEVICE_OK;
}

// ASI controller mirrors by having negative speed, but here we have separate property for mirroring
//   and for speed (which is strictly positive)... that makes this code a bit odd
// note that this setting is per-card, not per-axis
int CZStage::OnWheelFastSpeed(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   std::ostringstream command;
   double tmp = 0;
   if (eAct == MM::BeforeGet)
   {
      if (!refreshProps_ && initialized_)
         return DEVICE_OK;
      command << addressChar_ << "JS F?";
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), ":A F="));
      RETURN_ON_MM_ERROR( hub_->ParseAnswerAfterEquals(tmp) );
      tmp = abs(tmp);
      if (!pProp->Set(tmp))
         return DEVICE_INVALID_PROPERTY_VALUE;
   }
   else if (eAct == MM::AfterSet) {
      if (hub_->UpdatingSharedProperties())
         return DEVICE_OK;
      pProp->Get(tmp);
      char wheelMirror[MM::MaxStrLength];
      RETURN_ON_MM_ERROR ( GetProperty(g_WheelMirrorPropertyName, wheelMirror) );
      if (strcmp(wheelMirror, g_YesState) == 0)
         command << addressChar_ << "JS F=-" << tmp;
      else
         command << addressChar_ << "JS F=" << tmp;
      RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(), ":A") );
      command.str(""); command << tmp;
      RETURN_ON_MM_ERROR ( hub_->UpdateSharedProperties(addressChar_, pProp->GetName(), command.str()) );
   }
   return DEVICE_OK;
}

// ASI controller mirrors by having negative speed, but here we have separate property for mirroring
//   and for speed (which is strictly positive)... that makes this code a bit odd
// note that this setting is per-card, not per-axis
int CZStage::OnWheelSlowSpeed(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   std::ostringstream command;
   double tmp = 0;
   if (eAct == MM::BeforeGet)
   {
      if (!refreshProps_ && initialized_)
         return DEVICE_OK;
      command << addressChar_ << "JS T?";
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), ":A T="));
      RETURN_ON_MM_ERROR( hub_->ParseAnswerAfterEquals(tmp) );
      tmp = abs(tmp);
      if (!pProp->Set(tmp))
         return DEVICE_INVALID_PROPERTY_VALUE;
   }
   else if (eAct == MM::AfterSet) {
      if (hub_->UpdatingSharedProperties())
         return DEVICE_OK;
      pProp->Get(tmp);
      char wheelMirror[MM::MaxStrLength];
      RETURN_ON_MM_ERROR ( GetProperty(g_JoystickMirrorPropertyName, wheelMirror) );
      if (strcmp(wheelMirror, g_YesState) == 0)
         command << addressChar_ << "JS T=-" << tmp;
      else
         command << addressChar_ << "JS T=" << tmp;
      RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(), ":A") );
      command.str(""); command << tmp;
      RETURN_ON_MM_ERROR ( hub_->UpdateSharedProperties(addressChar_, pProp->GetName(), command.str()) );
   }
   return DEVICE_OK;
}

// ASI controller mirrors by having negative speed, but here we have separate property for mirroring
//   and for speed (which is strictly positive)... that makes this code a bit odd
// note that this setting is per-card, not per-axis
int CZStage::OnWheelMirror(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   std::ostringstream command;
   double tmp = 0;
   if (eAct == MM::BeforeGet)
   {
      if (!refreshProps_ && initialized_)
         return DEVICE_OK;
      command << addressChar_ << "JS F?";  // query only the fast setting to see if already mirrored
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), ":A F="));
      RETURN_ON_MM_ERROR( hub_->ParseAnswerAfterEquals(tmp) );
      bool success = 0;
      if (tmp < 0) // speed negative <=> mirrored
         success = pProp->Set(g_YesState);
      else
         success = pProp->Set(g_NoState);
      if (!success)
         return DEVICE_INVALID_PROPERTY_VALUE;
   }
   else if (eAct == MM::AfterSet) {
      if (hub_->UpdatingSharedProperties())
         return DEVICE_OK;
      std::string tmpstr;
      pProp->Get(tmpstr);
      double wheelFast = 0.0;
      RETURN_ON_MM_ERROR ( GetProperty(g_WheelFastSpeedPropertyName, wheelFast) );
      double wheelSlow = 0.0;
      RETURN_ON_MM_ERROR ( GetProperty(g_WheelSlowSpeedPropertyName, wheelSlow) );
      if (tmpstr == g_YesState)
         command << addressChar_ << "JS F=-" << wheelFast << " T=-" << wheelSlow;
      else
         command << addressChar_ << "JS F=" << wheelFast << " T=" << wheelSlow;
      RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(), ":A") );
      RETURN_ON_MM_ERROR ( hub_->UpdateSharedProperties(addressChar_, pProp->GetName(), tmpstr.c_str()) );
   }
   return DEVICE_OK;
}

int CZStage::OnAxisPolarity(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      // do nothing
   }
   else if (eAct == MM::AfterSet)
   {
      std::string tmpstr;
      pProp->Get(tmpstr);
      // change the unit mult that converts controller coordinates to micro-manager coordinates
      // micro-manager defines positive towards sample, ASI controllers just opposite
      if (tmpstr == g_FocusPolarityMicroManagerDefault) {
         unitMult_ = -1*abs(unitMult_);
      } else {
         unitMult_ = abs(unitMult_);
      }
   }
   return DEVICE_OK;
}

// Special property, when set to "yes" it creates a set of little-used properties that can be manipulated thereafter
int CZStage::OnSAAdvanced(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      return DEVICE_OK; // do nothing
   }
   else if (eAct == MM::AfterSet) {
      std::string tmpstr;
      pProp->Get(tmpstr);
      if (tmpstr == g_YesState)
      {
         CPropertyAction* pAct;

         pAct = new CPropertyAction (this, &CZStage::OnSAClkSrc);
         CreateProperty(g_SAClkSrcPropertyName, g_SAClkSrc_0, MM::String, false, pAct);
         AddAllowedValue(g_SAClkSrcPropertyName, g_SAClkSrc_0);
         AddAllowedValue(g_SAClkSrcPropertyName, g_SAClkSrc_1);
         UpdateProperty(g_SAClkSrcPropertyName);

         pAct = new CPropertyAction (this, &CZStage::OnSAClkPol);
         CreateProperty(g_SAClkPolPropertyName, g_SAClkPol_0, MM::String, false, pAct);
         AddAllowedValue(g_SAClkPolPropertyName, g_SAClkPol_0);
         AddAllowedValue(g_SAClkPolPropertyName, g_SAClkPol_1);
         UpdateProperty(g_SAClkPolPropertyName);

         pAct = new CPropertyAction (this, &CZStage::OnSATTLOut);
         CreateProperty(g_SATTLOutPropertyName, g_SATTLOut_0, MM::String, false, pAct);
         AddAllowedValue(g_SATTLOutPropertyName, g_SATTLOut_0);
         AddAllowedValue(g_SATTLOutPropertyName, g_SATTLOut_1);
         UpdateProperty(g_SATTLOutPropertyName);

         pAct = new CPropertyAction (this, &CZStage::OnSATTLPol);
         CreateProperty(g_SATTLPolPropertyName, g_SATTLPol_0, MM::String, false, pAct);
         AddAllowedValue(g_SATTLPolPropertyName, g_SATTLPol_0);
         AddAllowedValue(g_SATTLPolPropertyName, g_SATTLPol_1);
         UpdateProperty(g_SATTLPolPropertyName);

         pAct = new CPropertyAction (this, &CZStage::OnSAPatternByte);
         CreateProperty(g_SAPatternModePropertyName, "0", MM::Integer, false, pAct);
         SetPropertyLimits(g_SAPatternModePropertyName, 0, 255);
         UpdateProperty(g_SAPatternModePropertyName);
      }
   }
   return DEVICE_OK;
}

int CZStage::OnSAAmplitude(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   std::ostringstream command;
   std::ostringstream response;
   double tmp = 0;
   if (eAct == MM::BeforeGet)
   {
      if (!refreshProps_ && initialized_)
         return DEVICE_OK;
      command << "SAA " << axisLetter_ << "?";
      response << ":A " << axisLetter_ << "=";
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), response.str()));
      RETURN_ON_MM_ERROR( hub_->ParseAnswerAfterEquals(tmp) );
      tmp = tmp/unitMult_;
      if (!pProp->Set(tmp))
         return DEVICE_INVALID_PROPERTY_VALUE;
   }
   else if (eAct == MM::AfterSet) {
      pProp->Get(tmp);
      command << "SAA " << axisLetter_ << "=" << tmp*unitMult_;
      RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(), ":A") );
   }
   return DEVICE_OK;
}

int CZStage::OnSAOffset(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   std::ostringstream command;
   std::ostringstream response;
   double tmp = 0;
   if (eAct == MM::BeforeGet)
   {
      if (!refreshProps_ && initialized_)
         return DEVICE_OK;
      command << "SAO " << axisLetter_ << "?";
      response << ":A " << axisLetter_ << "=";
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), response.str()));
      RETURN_ON_MM_ERROR( hub_->ParseAnswerAfterEquals(tmp) );
      tmp = tmp/unitMult_;
      if (!pProp->Set(tmp))
         return DEVICE_INVALID_PROPERTY_VALUE;
   }
   else if (eAct == MM::AfterSet) {
      pProp->Get(tmp);
      command << "SAO " << axisLetter_ << "=" << tmp*unitMult_;
      RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(), ":A") );
   }
   return DEVICE_OK;
}

int CZStage::OnSAPeriod(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   std::ostringstream command;
   std::ostringstream response;
   long tmp = 0;
   if (eAct == MM::BeforeGet)
   {
      if (!refreshProps_ && initialized_)
         return DEVICE_OK;
      command << "SAF " << axisLetter_ << "?";
      response << ":A " << axisLetter_ << "=";
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), response.str()));
      RETURN_ON_MM_ERROR ( hub_->ParseAnswerAfterEquals(tmp) );
      if (!pProp->Set(tmp))
         return DEVICE_INVALID_PROPERTY_VALUE;
   }
   else if (eAct == MM::AfterSet) {
      pProp->Get(tmp);
      command << "SAF " << axisLetter_ << "=" << tmp;
      RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(), ":A") );
   }
   return DEVICE_OK;
}

int CZStage::OnSAMode(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   static bool justSet = false;
   std::ostringstream command;
   std::ostringstream response;
   long tmp = 0;
   if (eAct == MM::BeforeGet)
   {
      if (!refreshProps_ && initialized_ && !justSet)
         return DEVICE_OK;
      command << "SAM " << axisLetter_ << "?";
      response << ":A " << axisLetter_ << "=";
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), response.str()));
      RETURN_ON_MM_ERROR ( hub_->ParseAnswerAfterEquals(tmp) );
      bool success;
      switch (tmp)
      {
         case 0: success = pProp->Set(g_SAMode_0); break;
         case 1: success = pProp->Set(g_SAMode_1); break;
         case 2: success = pProp->Set(g_SAMode_2); break;
         case 3: success = pProp->Set(g_SAMode_3); break;
         default:success = 0;                      break;
      }
      if (!success)
         return DEVICE_INVALID_PROPERTY_VALUE;
      justSet = false;
   }
   else if (eAct == MM::AfterSet)
   {
      std::string tmpstr;
      pProp->Get(tmpstr);
      if (tmpstr == g_SAMode_0)
         tmp = 0;
      else if (tmpstr == g_SAMode_1)
         tmp = 1;
      else if (tmpstr == g_SAMode_2)
         tmp = 2;
      else if (tmpstr == g_SAMode_3)
         tmp = 3;
      else
         return DEVICE_INVALID_PROPERTY_VALUE;
      command << "SAM " << axisLetter_ << "=" << tmp;
      RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(), ":A") );
      // get the updated value right away
      justSet = true;
      return OnSAMode(pProp, MM::BeforeGet);
   }
   return DEVICE_OK;
}

int CZStage::OnSAPattern(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   std::ostringstream command;
   std::ostringstream response;
   long tmp = 0;
   if (eAct == MM::BeforeGet)
   {
      if (!refreshProps_ && initialized_)
         return DEVICE_OK;
      command << "SAP " << axisLetter_ << "?";
      response << ":A " << axisLetter_ << "=";
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), response.str()));
      RETURN_ON_MM_ERROR ( hub_->ParseAnswerAfterEquals(tmp) );
      bool success;
      tmp = tmp & ((long)(BIT2|BIT1|BIT0));  // zero all but the lowest 3 bits
      switch (tmp)
      {
         case 0: success = pProp->Set(g_SAPattern_0); break;
         case 1: success = pProp->Set(g_SAPattern_1); break;
         case 2: success = pProp->Set(g_SAPattern_2); break;
         case 3: success = pProp->Set(g_SAPattern_3); break;
		 default:success = 0;                      break;
      }
      if (!success)
         return DEVICE_INVALID_PROPERTY_VALUE;
   }
   else if (eAct == MM::AfterSet)
   {
      std::string tmpstr;
      pProp->Get(tmpstr);
      if (tmpstr == g_SAPattern_0)
         tmp = 0;
      else if (tmpstr == g_SAPattern_1)
         tmp = 1;
      else if (tmpstr == g_SAPattern_2)
         tmp = 2;
      else if (tmpstr == g_SAPattern_3)
         tmp = 3;
	  else
         return DEVICE_INVALID_PROPERTY_VALUE;
      // have to get current settings and then modify bits 0-2 from there
      command << "SAP " << axisLetter_ << "?";
      response << ":A " << axisLetter_ << "=";
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), response.str()));
      long current;
      RETURN_ON_MM_ERROR ( hub_->ParseAnswerAfterEquals(current) );
      current = current & (~(long)(BIT2|BIT1|BIT0));  // set lowest 3 bits to zero
      tmp += current;
      command.str("");
      command << "SAP " << axisLetter_ << "=" << tmp;
      RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(), ":A") );
   }
   return DEVICE_OK;
}

// get every single time
int CZStage::OnSAPatternByte(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   std::ostringstream command;
   std::ostringstream response;
   long tmp = 0;
   if (eAct == MM::BeforeGet)
   {
      command << "SAP " << axisLetter_ << "?";
      response << ":A " << axisLetter_ << "=";
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), response.str()));
      RETURN_ON_MM_ERROR ( hub_->ParseAnswerAfterEquals(tmp) );
      if (!pProp->Set(tmp))
         return DEVICE_INVALID_PROPERTY_VALUE;
   }
   else if (eAct == MM::AfterSet) {
      pProp->Get(tmp);
      command << "SAP " << axisLetter_ << "=" << tmp;
      RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(), ":A") );
   }
   return DEVICE_OK;
}

int CZStage::OnSAClkSrc(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   std::ostringstream command;
   std::ostringstream response;
   long tmp = 0;
   if (eAct == MM::BeforeGet)
   {
      if (!refreshProps_ && initialized_)
         return DEVICE_OK;
      command << "SAP " << axisLetter_ << "?";
      response << ":A " << axisLetter_ << "=";
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), response.str()));
      RETURN_ON_MM_ERROR ( hub_->ParseAnswerAfterEquals(tmp) );
      bool success;
      tmp = tmp & ((long)(BIT7));  // zero all but bit 7
      switch (tmp)
      {
         case 0: success = pProp->Set(g_SAClkSrc_0); break;
         case BIT7: success = pProp->Set(g_SAClkSrc_1); break;
         default:success = 0;                      break;
      }
      if (!success)
         return DEVICE_INVALID_PROPERTY_VALUE;
   }
   else if (eAct == MM::AfterSet)
   {
      std::string tmpstr;
      pProp->Get(tmpstr);
      if (tmpstr == g_SAClkSrc_0)
         tmp = 0;
      else if (tmpstr == g_SAClkSrc_1)
         tmp = BIT7;
      else
         return DEVICE_INVALID_PROPERTY_VALUE;
      // have to get current settings and then modify bit 7 from there
      command << "SAP " << axisLetter_ << "?";
      response << ":A " << axisLetter_ << "=";
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), response.str()));
      long current;
      RETURN_ON_MM_ERROR ( hub_->ParseAnswerAfterEquals(current) );
      current = current & (~(long)(BIT7));  // clear bit 7
      tmp += current;
      command.str("");
      command << "SAP " << axisLetter_ << "=" << tmp;
      RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(), ":A") );
   }
   return DEVICE_OK;
}

int CZStage::OnSAClkPol(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   std::ostringstream command;
   std::ostringstream response;
   long tmp = 0;
   if (eAct == MM::BeforeGet)
   {
      if (!refreshProps_ && initialized_)
         return DEVICE_OK;
      command << "SAP " << axisLetter_ << "?";
      response << ":A " << axisLetter_ << "=";
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), response.str()));
      RETURN_ON_MM_ERROR ( hub_->ParseAnswerAfterEquals(tmp) );
      bool success;
      tmp = tmp & ((long)(BIT6));  // zero all but bit 6
      switch (tmp)
      {
         case 0: success = pProp->Set(g_SAClkPol_0); break;
         case BIT6: success = pProp->Set(g_SAClkPol_1); break;
         default:success = 0;                      break;
      }
      if (!success)
         return DEVICE_INVALID_PROPERTY_VALUE;
   }
   else if (eAct == MM::AfterSet)
   {
      std::string tmpstr;
      pProp->Get(tmpstr);
      if (tmpstr == g_SAClkPol_0)
         tmp = 0;
      else if (tmpstr == g_SAClkPol_1)
         tmp = BIT6;
      else
         return DEVICE_INVALID_PROPERTY_VALUE;
      // have to get current settings and then modify bit 6 from there
      command << "SAP " << axisLetter_ << "?";
      response << ":A " << axisLetter_ << "=";
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), response.str()));
      long current;
      RETURN_ON_MM_ERROR ( hub_->ParseAnswerAfterEquals(current) );
      current = current & (~(long)(BIT6));  // clear bit 6
      tmp += current;
      command.str("");
      command << "SAP " << axisLetter_ << "=" << tmp;
      RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(), ":A") );
   }
   return DEVICE_OK;
}

int CZStage::OnSATTLOut(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   std::ostringstream command;
   std::ostringstream response;
   long tmp = 0;
   if (eAct == MM::BeforeGet)
   {
      if (!refreshProps_ && initialized_)
         return DEVICE_OK;
      command << "SAP " << axisLetter_ << "?";
      response << ":A " << axisLetter_ << "=";
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), response.str()));
      RETURN_ON_MM_ERROR ( hub_->ParseAnswerAfterEquals(tmp) );
      bool success;
      tmp = tmp & ((long)(BIT5));  // zero all but bit 5
      switch (tmp)
      {
         case 0: success = pProp->Set(g_SATTLOut_0); break;
         case BIT5: success = pProp->Set(g_SATTLOut_1); break;
         default:success = 0;                      break;
      }
      if (!success)
         return DEVICE_INVALID_PROPERTY_VALUE;
   }
   else if (eAct == MM::AfterSet)
   {
      std::string tmpstr;
      pProp->Get(tmpstr);
      if (tmpstr == g_SATTLOut_0)
         tmp = 0;
      else if (tmpstr == g_SATTLOut_1)
         tmp = BIT5;
      else
         return DEVICE_INVALID_PROPERTY_VALUE;
      // have to get current settings and then modify bit 5 from there
      command << "SAP " << axisLetter_ << "?";
      response << ":A " << axisLetter_ << "=";
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), response.str()));
      long current;
      RETURN_ON_MM_ERROR ( hub_->ParseAnswerAfterEquals(current) );
      current = current & (~(long)(BIT5));  // clear bit 5
      tmp += current;
      command.str("");
      command << "SAP " << axisLetter_ << "=" << tmp;
      RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(), ":A") );
   }
   return DEVICE_OK;
}

int CZStage::OnSATTLPol(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   std::ostringstream command;
   std::ostringstream response;
   long tmp = 0;
   if (eAct == MM::BeforeGet)
   {
      if (!refreshProps_ && initialized_)
         return DEVICE_OK;
      command << "SAP " << axisLetter_ << "?";
      response << ":A " << axisLetter_ << "=";
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), response.str()));
      RETURN_ON_MM_ERROR ( hub_->ParseAnswerAfterEquals(tmp) );
      bool success;
      tmp = tmp & ((long)(BIT4));  // zero all but bit 4
      switch (tmp)
      {
         case 0: success = pProp->Set(g_SATTLPol_0); break;
         case BIT4: success = pProp->Set(g_SATTLPol_1); break;
         default:success = 0;                      break;
      }
      if (!success)
         return DEVICE_INVALID_PROPERTY_VALUE;
   }
   else if (eAct == MM::AfterSet)
   {
      std::string tmpstr;
      pProp->Get(tmpstr);
      if (tmpstr == g_SATTLPol_0)
         tmp = 0;
      else if (tmpstr == g_SATTLPol_1)
         tmp = BIT4;
      else
         return DEVICE_INVALID_PROPERTY_VALUE;
      // have to get current settings and then modify bit 4 from there
      command << "SAP " << axisLetter_ << "?";
      response << ":A " << axisLetter_ << "=";
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), response.str()));
      long current;
      RETURN_ON_MM_ERROR ( hub_->ParseAnswerAfterEquals(current) );
      current = current & (~(long)(BIT4));  // clear bit 4
      tmp += current;
      command.str("");
      command << "SAP " << axisLetter_ << "=" << tmp;
      RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(), ":A") );
   }
   return DEVICE_OK;
}

int CZStage::OnRBMode(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   std::ostringstream command;
   std::ostringstream response;
   std::string pseudoAxisChar = FirmwareVersionAtLeast(2.89) ? "F" : "X";
   long tmp;
   if (eAct == MM::BeforeGet)
   {
      if (!refreshProps_ && initialized_)
         return DEVICE_OK;
      command << addressChar_ << "RM " << pseudoAxisChar << "?";
      response << ":A " << pseudoAxisChar << "=";
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), response.str()) );
      RETURN_ON_MM_ERROR( hub_->ParseAnswerAfterEquals(tmp) );
      if (tmp >= 128)
      {
         tmp -= 128;  // remove the "running now" code if present
      }
      bool success;
      switch ( tmp )
      {
         case 1: success = pProp->Set(g_RB_OnePoint_1); break;
         case 2: success = pProp->Set(g_RB_PlayOnce_2); break;
         case 3: success = pProp->Set(g_RB_PlayRepeat_3); break;
         default: success = false;
      }
      if (!success)
         return DEVICE_INVALID_PROPERTY_VALUE;
   }
   else if (eAct == MM::AfterSet)
   {
      if (hub_->UpdatingSharedProperties())
         return DEVICE_OK;
      std::string tmpstr;
      pProp->Get(tmpstr);
      if (tmpstr == g_RB_OnePoint_1)
         tmp = 1;
      else if (tmpstr == g_RB_PlayOnce_2)
         tmp = 2;
      else if (tmpstr == g_RB_PlayRepeat_3)
         tmp = 3;
      else
         return DEVICE_INVALID_PROPERTY_VALUE;
      command << addressChar_ << "RM " << pseudoAxisChar << "=" << tmp;
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), ":A"));
      RETURN_ON_MM_ERROR ( hub_->UpdateSharedProperties(addressChar_, pProp->GetName(), tmpstr.c_str()) );
   }
   return DEVICE_OK;
}

int CZStage::OnRBTrigger(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   std::ostringstream command;
   if (eAct == MM::BeforeGet) {
      pProp->Set(g_IdleState);
   }
   else  if (eAct == MM::AfterSet) {
      if (hub_->UpdatingSharedProperties())
         return DEVICE_OK;
      std::string tmpstr;
      pProp->Get(tmpstr);
      if (tmpstr == g_DoItState)
      {
         command << addressChar_ << "RM";
         RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(), ":A") );
         pProp->Set(g_DoneState);
         command.str(""); command << g_DoneState;
         RETURN_ON_MM_ERROR ( hub_->UpdateSharedProperties(addressChar_, pProp->GetName(), command.str()) );
      }
   }
   return DEVICE_OK;
}

int CZStage::OnRBRunning(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   std::ostringstream command;
   std::ostringstream response;
   std::string pseudoAxisChar = FirmwareVersionAtLeast(2.89) ? "F" : "X";
   long tmp = 0;
   static bool justSet;
   if (eAct == MM::BeforeGet)
   {
      if (!refreshProps_ && initialized_ && !justSet)
         return DEVICE_OK;
      command << addressChar_ << "RM " << pseudoAxisChar << "?";
      response << ":A " << pseudoAxisChar << "=";
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), response.str()) );
      RETURN_ON_MM_ERROR( hub_->ParseAnswerAfterEquals(tmp) );
      bool success;
      if (tmp >= 128)
      {
         success = pProp->Set(g_YesState);
      }
      else
      {
         success = pProp->Set(g_NoState);
      }
      if (!success)
         return DEVICE_INVALID_PROPERTY_VALUE;
      justSet = false;
   }
   else if (eAct == MM::AfterSet)
   {
      justSet = true;
      return OnRBRunning(pProp, MM::BeforeGet);
      // TODO determine how to handle this with shared properties since ring buffer is per-card and not per-axis
      // the reason this property exists (and why it's not a read-only property) are a bit hazy as of mid-2017
   }
   return DEVICE_OK;
}

int CZStage::OnRBDelayBetweenPoints(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   std::ostringstream command;
   long tmp = 0;
   if (eAct == MM::BeforeGet)
   {
      if (!refreshProps_ && initialized_)
         return DEVICE_OK;
      command << addressChar_ << "RT Z?";
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), ":A Z="));
      RETURN_ON_MM_ERROR( hub_->ParseAnswerAfterEquals(tmp) );
      if (!pProp->Set(tmp))
         return DEVICE_INVALID_PROPERTY_VALUE;
   }
   else if (eAct == MM::AfterSet) {
      if (hub_->UpdatingSharedProperties())
         return DEVICE_OK;
      pProp->Get(tmp);
      command << addressChar_ << "RT Z=" << tmp;
      RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(), ":A") );
      command.str(""); command << tmp;
      RETURN_ON_MM_ERROR ( hub_->UpdateSharedProperties(addressChar_, pProp->GetName(), command.str()) );
   }
   return DEVICE_OK;
}

int CZStage::OnUseSequence(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   std::ostringstream command;
   if (eAct == MM::BeforeGet)
   {
      if (ttl_trigger_enabled_)
         pProp->Set(g_YesState);
      else
         pProp->Set(g_NoState);
   }
   else if (eAct == MM::AfterSet) {
      std::string tmpstr;
      pProp->Get(tmpstr);
      ttl_trigger_enabled_ = ttl_trigger_supported_ && (tmpstr == g_YesState);
      return OnUseSequence(pProp, MM::BeforeGet);  // refresh value
   }
   return DEVICE_OK;
}

int CZStage::OnFastSequence(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      if (runningFastSequence_)
         pProp->Set(g_ArmedState);
      else
         pProp->Set(g_NoState);
   }
   else if (eAct == MM::AfterSet)
   {
      std::string tmpstr;
      pProp->Get(tmpstr);
      // only let user do fast sequence if regular one is enabled
      if (!ttl_trigger_enabled_) {
         pProp->Set(g_NoState);
         return DEVICE_OK;
      }
      if (tmpstr == g_ArmedState)
      {
         runningFastSequence_ = false;
         RETURN_ON_MM_ERROR ( SendStageSequence() );
         RETURN_ON_MM_ERROR ( StartStageSequence() );
         runningFastSequence_ = true;
      }
      else
      {
         runningFastSequence_ = false;
         RETURN_ON_MM_ERROR ( StopStageSequence() );
      }
   }
   return DEVICE_OK;
}

int CZStage::OnVector(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   std::ostringstream command;
   std::ostringstream response;
   double tmp = 0;
   if (eAct == MM::BeforeGet)
   {
      if (!refreshProps_ && initialized_)
         return DEVICE_OK;
      command << "VE " << axisLetter_ << "?";
      response << ":A " << axisLetter_ << "=";
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), response.str()));
      RETURN_ON_MM_ERROR( hub_->ParseAnswerAfterEquals(tmp) );
      if (!pProp->Set(tmp))
         return DEVICE_INVALID_PROPERTY_VALUE;
   }
   else if (eAct == MM::AfterSet) {
      pProp->Get(tmp);
      command << "VE " << axisLetter_ << "=" << tmp;
      RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(), ":A") );
      // will make stage report busy which can lead to timeout errors
   }
   return DEVICE_OK;
}

int CZStage::OnTTLInputMode(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   std::ostringstream command;
   long tmp;
   if (eAct == MM::BeforeGet) {
      if (!refreshProps_ && initialized_)
         return DEVICE_OK;
      command << addressChar_ << "TTL X?";
      RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(), axisLetter_) );
      RETURN_ON_MM_ERROR ( hub_->ParseAnswerAfterEquals(tmp) );
      bool success = 0;
      switch (tmp) {
         case 0: success = pProp->Set(g_TTLInputMode_0); break;
         case 1: success = pProp->Set(g_TTLInputMode_1); break;
         case 2: success = pProp->Set(g_TTLInputMode_2); break;
         case 7: success = pProp->Set(g_TTLInputMode_7); break;
         default: success=0;
      }
      if (!success)
         return DEVICE_INVALID_PROPERTY_VALUE;
   } else if (eAct == MM::AfterSet) {
      if (hub_->UpdatingSharedProperties())
         return DEVICE_OK;
      RETURN_ON_MM_ERROR ( GetCurrentPropertyData(pProp->GetName().c_str(), tmp) );
      command << addressChar_ << "TTL X=" << tmp;
      RETURN_ON_MM_ERROR ( hub_->QueryCommandVerify(command.str(),":A") );
      std::string tmpstr;
      pProp->Get(tmpstr);
      RETURN_ON_MM_ERROR ( hub_->UpdateSharedProperties(addressChar_, pProp->GetName(), tmpstr.c_str()) );
   }
   return DEVICE_OK;
}
