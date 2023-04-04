///////////////////////////////////////////////////////////////////////////////
// FILE:          PIZStage.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   PI GCS DLL ZStage
//
// AUTHOR:        Nenad Amodaj, nenad@amodaj.com, 08/28/2006
//                Steffen Rau, s.rau@pi.ws, 10/03/2008
// COPYRIGHT:     University of California, San Francisco, 2006
//                Physik Instrumente (PI) GmbH & Co. KG, 2008
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
// CVS:           $Id: PIZStage_DLL.cpp,v 1.20, 2018-10-01 14:25:47Z, Steffen Rau$
//

#include "PIZStage.h"

#include "PI_GCS_2.h"
#include "PIController.h"

const char* PIZStage::DeviceName_ = "PIZStage";
const char* g_PI_ZStageAxisName = "Axis";
const char* g_PI_ZStageAxisLimitUm = "Limit_um";
const char* g_PI_ZStageInvertTravelRange = "Invert travel range";
const char* g_PI_ZStageStageType = "Stage";
const char* g_PI_ZStageStepSize = "StepSizeUm";
const char* g_PI_ZStageControllerName = "Controller Name";
const char* g_PI_ZStageServoControl = "Servo control activated (SVO)";
const char* g_PI_ZStageEnableAxis = "Axis enabled (EAX)";
const char* g_PI_ZStageAlternativeHomingCommand = "Alternative Homing Command";
const char* g_PI_ZStageAxisErrorState = "Axis in error state";


///////////////////////////////////////////////////////////////////////////////
// PIZStage

PIZStage::PIZStage ()
   : axisName_ ("1")
   , stepSizeUm_ (0.1)
   , initialized_ (false)
   , axisLimitUm_ (500.0)
   , invertTravelRange_ (false)
   , servoControl_ (true)
   , axisEnabled_ (true)
   , stageType_ ("")
   , controllerName_ ("")
   , ctrl_ (NULL)
{
   InitializeDefaultErrorMessages ();

   SetErrorText (ERR_GCS_PI_CNTR_POS_OUT_OF_LIMITS, g_msg_CNTR_POS_OUT_OF_LIMITS);
   SetErrorText (ERR_GCS_PI_CNTR_MOVE_WITHOUT_REF_OR_NO_SERVO, g_msg_CNTR_MOVE_WITHOUT_REF_OR_NO_SERVO);
   SetErrorText (ERR_GCS_PI_CNTR_AXIS_UNDER_JOYSTICK_CONTROL, g_msg_CNTR_AXIS_UNDER_JOYSTICK_CONTROL);
   SetErrorText (ERR_GCS_PI_CNTR_INVALID_AXIS_IDENTIFIER, g_msg_CNTR_INVALID_AXIS_IDENTIFIER);
   SetErrorText (ERR_GCS_PI_CNTR_ILLEGAL_AXIS, g_msg_CNTR_ILLEGAL_AXIS);
   SetErrorText (ERR_GCS_PI_CNTR_VEL_OUT_OF_LIMITS, g_msg_CNTR_VEL_OUT_OF_LIMITS);
   SetErrorText (ERR_GCS_PI_CNTR_ON_LIMIT_SWITCH, g_msg_CNTR_ON_LIMIT_SWITCH);
   SetErrorText (ERR_GCS_PI_CNTR_MOTION_ERROR, g_msg_CNTR_MOTION_ERROR);
   SetErrorText (ERR_GCS_PI_MOTION_ERROR, g_msg_MOTION_ERROR);
   SetErrorText (ERR_GCS_PI_CNTR_PARAM_OUT_OF_RANGE, g_msg_CNTR_PARAM_OUT_OF_RANGE);
   SetErrorText (ERR_GCS_PI_NO_CONTROLLER_FOUND, g_msg_NO_CONTROLLER_FOUND);
   SetErrorText (ERR_GCS_PI_AXIS_DISABLED, g_msg_AXIS_DISABLED);
   SetErrorText (ERR_GCS_PI_INVALID_MODE_OF_OPERATION, g_msg_INVALID_MODE_OF_OPERATION);
   SetErrorText (ERR_GCS_PI_PARAM_VALUE_OUT_OF_RANGE, g_msg_PARAM_VALUE_OUT_OF_RANGE);
   SetErrorText (ERR_GCS_PI_CNTR_MOTOR_IS_OFF, g_msg_CNTR_MOTOR_IS_OFF);
   SetErrorText (ERR_GCS_PI_AXIS_IN_FAULT, g_msg_AXIS_IN_FAULT);

   // create pre-initialization properties
   // ------------------------------------

   // Name
   CreateProperty (MM::g_Keyword_Name, DeviceName_, MM::String, true);

   // Description
   CreateProperty (MM::g_Keyword_Description, "Physik Instrumente (PI) GCS DLL Adapter", MM::String, true);

   CPropertyAction* pAct;


   // Controller name
   pAct = new CPropertyAction (this, &PIZStage::OnControllerName);
   CreateProperty (g_PI_ZStageControllerName, controllerName_.c_str (), MM::String, false, pAct, true);

   // Axis name
   pAct = new CPropertyAction (this, &PIZStage::OnAxisName);
   CreateProperty (g_PI_ZStageAxisName, axisName_.c_str (), MM::String, false, pAct, true);

   // Axis stage type
   pAct = new CPropertyAction (this, &PIZStage::OnStageType);
   CreateProperty (g_PI_ZStageStageType, stageType_.c_str (), MM::String, false, pAct, true);

   //// Axis homing mode
   //pAct = new CPropertyAction (this, &PIZStage::OnHoming);
   //CreateProperty(g_PI_ZStageHoming, homingMode_.c_str(), MM::String, false, pAct, true);

   // axis limit in um
   pAct = new CPropertyAction (this, &PIZStage::OnAxisLimit);
   CreateProperty (g_PI_ZStageAxisLimitUm, "500.0", MM::Float, false, pAct, true);

   // axis limit in um
   pAct = new CPropertyAction (this, &PIZStage::OnAxisTravelRange);
   CreateProperty (g_PI_ZStageInvertTravelRange, "0", MM::Integer, false, pAct, true);

   // axis limits (assumed symmetrical)
   pAct = new CPropertyAction (this, &PIZStage::OnPosition);
   CreateProperty (MM::g_Keyword_Position, "0.0", MM::Float, false, pAct);
   SetPropertyLimits (MM::g_Keyword_Position, 0.0/*-axisLimitUm_*/, axisLimitUm_);

   // servo control
   pAct = new CPropertyAction (this, &PIZStage::OnAxisServoControl);
   CreateProperty (g_PI_ZStageServoControl, "1", MM::Integer, false, pAct);

   // enable axis
   pAct = new CPropertyAction (this, &PIZStage::OnEnableAxis);
   CreateProperty (g_PI_ZStageEnableAxis, "1", MM::Integer, false, pAct);

   // alternative homing command
   pAct = new CPropertyAction (this, &PIZStage::OnAlternativeHomingCommand);
   CreateProperty (g_PI_ZStageAlternativeHomingCommand, "", MM::String, false, pAct, true);

   // Error state
   pAct = new CPropertyAction (this, &PIZStage::OnAxisErrorState);
   CreateProperty (g_PI_ZStageAxisErrorState, "0", MM::Integer, false, pAct);
}

PIZStage::~PIZStage ()
{
   Shutdown ();
   ctrl_ = NULL;
}

void PIZStage::GetName (char* Name) const
{
   CDeviceUtils::CopyLimitedString (Name, DeviceName_);
}

int PIZStage::Initialize ()
{
   MM::Device* device = GetDevice (controllerName_.c_str ());
   if (device == NULL)
   {
      return ERR_GCS_PI_NO_CONTROLLER_FOUND;
   }

   int ret = device->Initialize ();
   if (ret != DEVICE_OK)
   {
      return ret;
   }

   ctrl_ = PIController::GetByLabel (controllerName_);
   if (ctrl_ == NULL)
   {
      return ERR_GCS_PI_NO_CONTROLLER_FOUND;
   }
   ctrl_->Attach (this);

   std::string  sBuffer;
   ctrl_->ReadIdentification (sBuffer);
   LogMessage (std::string ("Connected to: ") + sBuffer);

   ret = ctrl_->InitStage (axisName_, stageType_);
   if (ret != DEVICE_OK)
   {
      LogMessage ("Cannot init axis");
      return ret;
   }

   CPropertyAction* pAct = new CPropertyAction (this, &PIZStage::OnHoming);
   CreateProperty ("HOMING", "", MM::String, false, pAct);

   pAct = new CPropertyAction (this, &PIZStage::OnVelocity);
   CreateProperty ("Velocity", "", MM::Float, false, pAct);

   bool hin = false;
   if (ctrl_->ReadHIN (axisName_, hin))
   {
      pAct = new CPropertyAction (this, &PIZStage::OnHID);
      CreateProperty ("HID active", hin ? "1" : "0", MM::Integer, false, pAct);
   }

   initialized_ = true;
   return DEVICE_OK;
}

int PIZStage::Shutdown ()
{
   if (initialized_)
   {
      initialized_ = false;
   }
   return DEVICE_OK;
}

bool PIZStage::Busy ()
{
   if (NULL == ctrl_)
   {
      return false;
   }
   return ctrl_->IsBusy (axisName_);
}

int PIZStage::SetPositionSteps (long steps)
{
   double pos = steps * stepSizeUm_;
   return SetPositionUm (pos);
}

int PIZStage::GetPositionSteps (long& steps)
{
   double pos;
   int ret = GetPositionUm (pos);
   if (ret != DEVICE_OK)
   {
      return ret;
   }
   steps = static_cast<long>((pos / stepSizeUm_) + 0.5);
   return DEVICE_OK;
}

int PIZStage::SetPositionUm (double pos)
{
   if (NULL == ctrl_)
   {
      return DEVICE_ERR;
   }
   if (invertTravelRange_)
   {
      pos = axisLimitUm_ - pos;
   }
   pos *= ctrl_->GetUmToDefaultUnit ();
   if (!ctrl_->MoveAxis (axisName_, pos))
   {
      return ctrl_->GetTranslatedError ();
   }

   return DEVICE_OK;
}

int PIZStage::GetPositionUm (double& pos)
{
   if (NULL == ctrl_)
   {
      return DEVICE_ERR;
   }
   if (!ctrl_->ReadPosition (axisName_, &pos))
   {
      return ctrl_->GetTranslatedError ();
   }
   pos /= ctrl_->GetUmToDefaultUnit ();
   if (invertTravelRange_)
   {
      pos = axisLimitUm_ - pos;
   }
   return DEVICE_OK;
}

int PIZStage::SetOrigin ()
{
   return DEVICE_UNSUPPORTED_COMMAND;
}

int PIZStage::GetLimits (double& min, double& max)
{
   min = 0;
   max = axisLimitUm_;
   return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// Action handlers
///////////////////////////////////////////////////////////////////////////////

int PIZStage::OnControllerName (MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set (controllerName_.c_str ());
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get (controllerName_);
   }

   return DEVICE_OK;
}



int PIZStage::OnAxisName (MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set (axisName_.c_str ());
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get (axisName_);
   }

   return DEVICE_OK;
}

int PIZStage::OnStepSizeUm (MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set (stepSizeUm_);
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get (stepSizeUm_);
   }

   return DEVICE_OK;
}

int PIZStage::OnAxisLimit (MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set (axisLimitUm_);
      SetPropertyLimits (MM::g_Keyword_Position, 0.0/*-axisLimitUm_*/, axisLimitUm_);
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get (axisLimitUm_);
   }

   return DEVICE_OK;
}

int PIZStage::OnAxisTravelRange (MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set (long (invertTravelRange_ ? 1 : 0));
   }
   else if (eAct == MM::AfterSet)
   {
      long value;
      pProp->Get (value);
      invertTravelRange_ = (value != 0);
   }

   return DEVICE_OK;
}

int PIZStage::OnAxisServoControl (MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      if (initialized_)
      {
         bool svo;
         if (!ctrl_->ReadServo (axisName_, svo))
         {
            return ctrl_->GetTranslatedError ();
         }
         servoControl_ = svo;
      }
      pProp->Set (long (servoControl_ ? 1 : 0));
   }
   else if (eAct == MM::AfterSet)
   {
      long value;
      pProp->Get (value);
      servoControl_ = (value != 0);
      if (!ctrl_->WriteServo (axisName_, servoControl_))
      {
         return ctrl_->GetTranslatedError ();
      }
   }

   return DEVICE_OK;
}

int PIZStage::OnEnableAxis (MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      if (initialized_)
      {
         bool eax = true;
         if (ctrl_->CanEnableAxis () && !ctrl_->ReadEnableAxis (axisName_, eax))
         {
            return ctrl_->GetTranslatedError ();
         }
         axisEnabled_ = eax;
      }
      pProp->Set (long (axisEnabled_ ? 1 : 0));
   }
   else if (eAct == MM::AfterSet)
   {
      long value;
      pProp->Get (value);
      axisEnabled_ = (value != 0);
      if (ctrl_->CanEnableAxis () && !ctrl_->WriteEnableAxis (axisName_, axisEnabled_))
      {
         return ctrl_->GetTranslatedError ();
      }
   }

   return DEVICE_OK;
}

int PIZStage::OnAlternativeHomingCommand (MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set (alternativeHomingCommand_.c_str ());
   }
   else if (eAct == MM::AfterSet)
   {
      std::string value;
      pProp->Get (value);
      alternativeHomingCommand_ = value;
   }

   return DEVICE_OK;
}

int PIZStage::OnAxisErrorState (MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      if (initialized_)
      {
         pProp->Set (long (ctrl_->IsInErrorState (axisName_) ? 1 : 0));
      }
      else
      {
         pProp->Set (0L);
      }
   }
   else if (eAct == MM::AfterSet)
   {
      long value;
      pProp->Get (value);
      // Error state should only be reset to 0 by the user.
      if (value == 0 && ctrl_->IsGCS30 ())
      {
         if (!ctrl_->Reset(axisName_))
         {
            return ctrl_->GetTranslatedError ();
         }
      }
   }

   return DEVICE_OK;
}

int PIZStage::OnPosition (MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      if (!initialized_)
      {
         pProp->Set (0.0);
         return DEVICE_OK;
      }
      double pos;
      int ret = GetPositionUm (pos);
      if (ret != DEVICE_OK)
      {
         return ret;
      }

      pProp->Set (pos);
   }
   else if (eAct == MM::AfterSet)
   {
      if (!initialized_)
      {
         return DEVICE_OK;
      }
      double pos;
      pProp->Get (pos);
      int ret = SetPositionUm (pos);
      if (ret != DEVICE_OK)
      {
         return ret;
      }

   }

   return DEVICE_OK;
}

int PIZStage::OnStageType (MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set (stageType_.c_str ());
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get (stageType_);
   }

   return DEVICE_OK;
}

int PIZStage::OnHoming (MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (NULL == ctrl_)
   {
      return DEVICE_ERR;
   }
   if (eAct == MM::BeforeGet)
   {
      pProp->Set (0.0);
   }
   else if (eAct == MM::AfterSet)
   {
      if (alternativeHomingCommand_.length () != 0)
      {
         return ctrl_->SendCommand (alternativeHomingCommand_);
      }

      std::string homingMode;
      pProp->Get (homingMode);
      int ret = ctrl_->Home (axisName_, homingMode);
      if (ret != DEVICE_OK)
      {
         return ret;
      }
      while (Busy ())
      {};
      (void)ctrl_->WriteServo (axisName_, true);
      return DEVICE_OK;

   }

   return DEVICE_OK;
}

int PIZStage::OnVelocity (MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (NULL == ctrl_)
   {
      return DEVICE_ERR;
   }
   if (eAct == MM::BeforeGet)
   {
      double velocity = 0.0;
      if (!ctrl_->ReadVelocity (axisName_, velocity))
      {
         return ctrl_->GetTranslatedError ();
      }
      pProp->Set (velocity);
   }
   else if (eAct == MM::AfterSet)
   {
      double velocity = 0.0;
      pProp->Get (velocity);
      if (!ctrl_->WriteVelocity (axisName_, velocity))
      {
         int err = ctrl_->GetTranslatedError ();
         if (ERR_GCS_PI_PARAM_VALUE_OUT_OF_RANGE == err)
         {
            err = ERR_GCS_PI_CNTR_VEL_OUT_OF_LIMITS;
         }
         return err;
      }
   }

   return DEVICE_OK;
}


int PIZStage::OnHID (MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      bool state;
      if (!ctrl_->ReadHIN (axisName_, state))
      {
         return ctrl_->GetTranslatedError ();
      }
      pProp->Set (state ? 1L : 0L);
   }
   else if (eAct == MM::AfterSet)
   {
      long state;
      pProp->Get (state);
      if (!ctrl_->WriteHIN (axisName_, state == TRUE))
      {
         return ctrl_->GetTranslatedError ();
      }
   }

   return DEVICE_OK;
}

void PIZStage::OnControllerDeleted ()
{
   ctrl_ = NULL;
   initialized_ = false;
}
