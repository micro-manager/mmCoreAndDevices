///////////////////////////////////////////////////////////////////////////////
// FILE:          PIXYStage.cpp
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
// CVS:           $Id: PIXYStage_DLL.cpp,v 1.22, 2018-10-01 14:25:47Z, Steffen Rau$
//

#include "PIXYStage.h"
#include "PIController.h"

const char* PIXYStage::DeviceName_ = "PIXYStage";

const char* g_PI_XYStageAxisXName = "Axis X: Name";
const char* g_PI_XYStageAxisXStageType = "Axis X: Stage";
const char* g_PI_XYStageAxisXHoming = "Axis X: HomingMode";
const char* g_PI_XYStageAxisYName = "Axis Y: Name";
const char* g_PI_XYStageAxisYStageType = "Axis Y: Stage";
const char* g_PI_XYStageAxisYHoming = "Axis Y: HomingMode";
const char* g_PI_XYStageControllerName = "Controller Name";
const char* g_PI_XYStageControllerNameYAxis = "Controller Name for Y axis";
const char* g_PI_XYStageServoControl = "Servo control activated (SVO)";
const char* g_PI_XYStageEnableAxes = "Axes enabled (EAX)";
const char* g_PI_XYStageAxisErrorState = "Axis in error state";
const char* g_PI_XYStageAlternativeHomingCommandXAxis = "Axis X: Alternative Homing Command";
const char* g_PI_XYStageAlternativeHomingCommandYAxis = "Axis Y: Alternative Homing Command";


// valid interface types: "PCI", "RS-232"
// for Interface type "RS-232" Interface Parameter is a string: "<portnumber>;<baudrate>"
// for Interface type "PCI" Interface Parameter is a string: "<board index>"

// valid homing modes: "REF", "PLM", "NLM"

///////////////////////////////////////////////////////////////////////////////
// PIXYStage

PIXYStage::PIXYStage ()
   : CXYStageBase<PIXYStage> ()
   , axisXName_ ("1")
   , axisXStageType_ ("")
   , axisXHomingMode_ ("REF")
   , axisYName_ ("2")
   , axisYStageType_ ("")
   , axisYHomingMode_ ("REF")
   , controllerName_ ("")
   , controllerNameYAxis_ ("")
   , ctrl_ (NULL)
   , ctrlYAxis_ (NULL)
   , stepSize_um_ (0.01)
   , originX_ (0.0)
   , originY_ (0.0)
   , initialized_ (false)
   , servoControl_ (true)
   , axesEnabled_ (true)
{
   InitializeDefaultErrorMessages ();

   // create pre-initialization properties
   // ------------------------------------

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

   CreateProperties ();

}

void PIXYStage::CreateProperties ()
{
   // Name
   CreateProperty (MM::g_Keyword_Name, DeviceName_, MM::String, true);

   // Description
   CreateProperty (MM::g_Keyword_Description, "Physik Instrumente (PI) GCS DLL Adapter", MM::String, true);

   CPropertyAction* pAct;

   // Controller name
   pAct = new CPropertyAction (this, &PIXYStage::OnControllerName);
   CreateProperty (g_PI_XYStageControllerName, controllerName_.c_str (), MM::String, false, pAct, true);

   // Axis X name
   pAct = new CPropertyAction (this, &PIXYStage::OnAxisXName);
   CreateProperty (g_PI_XYStageAxisXName, axisXName_.c_str (), MM::String, false, pAct, true);

   // Axis X stage type
   pAct = new CPropertyAction (this, &PIXYStage::OnAxisXStageType);
   CreateProperty (g_PI_XYStageAxisXStageType, axisXStageType_.c_str (), MM::String, false, pAct, true);

   // Axis X homing mode
   pAct = new CPropertyAction (this, &PIXYStage::OnAxisXHoming);
   CreateProperty (g_PI_XYStageAxisXHoming, axisXHomingMode_.c_str (), MM::String, false, pAct, true);

   // Axis Y name
   pAct = new CPropertyAction (this, &PIXYStage::OnAxisYName);
   CreateProperty (g_PI_XYStageAxisYName, axisYName_.c_str (), MM::String, false, pAct, true);

   // Axis Y stage type
   pAct = new CPropertyAction (this, &PIXYStage::OnAxisYStageType);
   CreateProperty (g_PI_XYStageAxisYStageType, axisYStageType_.c_str (), MM::String, false, pAct, true);

   // Axis Y homing mode
   pAct = new CPropertyAction (this, &PIXYStage::OnAxisYHoming);
   CreateProperty (g_PI_XYStageAxisYHoming, axisYHomingMode_.c_str (), MM::String, false, pAct, true);

   // Controller name for Y axis
   pAct = new CPropertyAction (this, &PIXYStage::OnControllerNameYAxis);
   CreateProperty (g_PI_XYStageControllerNameYAxis, controllerNameYAxis_.c_str (), MM::String, false, pAct, true);

   // servo control
   pAct = new CPropertyAction (this, &PIXYStage::OnServoControl);
   CreateProperty (g_PI_XYStageServoControl, "1", MM::Integer, false, pAct);

   // enable axis
   pAct = new CPropertyAction (this, &PIXYStage::OnEnableAxes);
   CreateProperty (g_PI_XYStageEnableAxes, "1", MM::Integer, false, pAct);

   // Axis error state
   pAct = new CPropertyAction (this, &PIXYStage::OnAxisErrorState);
   CreateProperty (g_PI_XYStageAxisErrorState, "0", MM::Integer, false, pAct);

   pAct = new CPropertyAction (this, &PIXYStage::OnAlternativeHomingCommandXAxis);
   CreateProperty (g_PI_XYStageAlternativeHomingCommandXAxis, "", MM::String, false, pAct, true);

   pAct = new CPropertyAction (this, &PIXYStage::OnAlternativeHomingCommandYAxis);
   CreateProperty (g_PI_XYStageAlternativeHomingCommandYAxis, "", MM::String, false, pAct, true);

}

PIXYStage::~PIXYStage ()
{
   Shutdown ();
   ctrl_ = NULL;
   ctrlYAxis_ = NULL;
}

void PIXYStage::GetName (char* Name) const
{
   CDeviceUtils::CopyLimitedString (Name, DeviceName_);
}

int PIXYStage::Initialize ()
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
   std::string sBuffer;
   ctrl_->ReadIdentification (sBuffer);
   LogMessage (std::string ("Connected to: ") + sBuffer);

   ctrlYAxis_ = ctrl_;
   if (controllerNameYAxis_ != "")
   {
      device = GetDevice (controllerNameYAxis_.c_str ());
      if (device == NULL)
      {
         return ERR_GCS_PI_NO_CONTROLLER_FOUND;
      }

      ret = device->Initialize ();
      if (ret != DEVICE_OK)
      {
         return ret;
      }

      ctrlYAxis_ = PIController::GetByLabel (controllerNameYAxis_);
      if (ctrlYAxis_ == NULL)
      {
         return ERR_GCS_PI_NO_CONTROLLER_FOUND;
      }
      ctrlYAxis_->Attach (this);

      ctrlYAxis_->ReadIdentification (sBuffer);
      LogMessage (std::string ("Y axis Connected to: ") + sBuffer);
   }

   ret = ctrl_->InitStage (axisXName_, axisXStageType_);
   if (ret != DEVICE_OK)
   {
      LogMessage ("Cannot init axis x");
      return ret;
   }

   ret = ctrlYAxis_->InitStage (axisYName_, axisYStageType_);
   if (ret != DEVICE_OK)
   {
      LogMessage ("Cannot init axis y");
      return ret;
   }

   CPropertyAction* pAct = new CPropertyAction (this, &PIXYStage::OnXVelocity);
   CreateProperty ("Axis X: Velocity", "", MM::Float, false, pAct);

   pAct = new CPropertyAction (this, &PIXYStage::OnYVelocity);
   CreateProperty ("Axis Y: Velocity", "", MM::Float, false, pAct);

   bool hinX = false;
   bool hinY = false;
   if (ctrl_->ReadHIN (axisXName_, hinX) && ctrlYAxis_->ReadHIN (axisYName_, hinY))
   {
      pAct = new CPropertyAction (this, &PIXYStage::OnHID);
      CreateProperty ("HID active", hinX || hinY ? "1" : "0", MM::Integer, false, pAct);
   }

   initialized_ = true;
   return DEVICE_OK;
}

int PIXYStage::Shutdown ()
{
   if (initialized_)
   {
      initialized_ = false;
   }
   return DEVICE_OK;
}

bool PIXYStage::Busy ()
{
   bool busy = ctrl_->IsBusy (axisXName_);
   if (ctrl_->IsGCS30 () || ctrlYAxis_ != ctrl_)
   {
      // we need to check y axis also
      //  - with GCS 2.1 we check the busy/moving state for each axis
      //  - or different controller is used for Y axis, so busy state of this one needs to be asked also
      busy = busy || ctrlYAxis_->IsBusy (axisYName_);
   }
   if (!busy)
   {
      ctrl_->ClearReferenceMoveActive ();
      ctrlYAxis_->ClearReferenceMoveActive ();
   }
   return busy;
}

double PIXYStage::GetStepSize ()
{
   return stepSize_um_;
}

int PIXYStage::SetPositionSteps (long x, long y)
{
   double umToDefaultUnitYAxis = ctrl_->GetUmToDefaultUnit ();
   if (ctrlYAxis_ != ctrl_)
   {
      umToDefaultUnitYAxis = ctrlYAxis_->GetUmToDefaultUnit ();
   }

   double posX, posY;
   posX = x * stepSize_um_ * ctrl_->GetUmToDefaultUnit ();
   posY = y * stepSize_um_ * umToDefaultUnitYAxis;

   posX += originX_;
   posY += originY_;
   if (ctrlYAxis_ == ctrl_)
   {
      if (!ctrl_->MoveAxes (axisXName_, posX, axisYName_, posY))
      {
         return ctrl_->GetTranslatedError ();
      }
   }
   else
   {
      if (!ctrl_->MoveAxis (axisXName_, posX))
      {
         return ctrl_->GetTranslatedError ();
      }
      if (!ctrlYAxis_->MoveAxis (axisYName_, posY))
      {
         return ctrl_->GetTranslatedError ();
      }
   }

   return DEVICE_OK;

}


int PIXYStage::ReadPositions (double& posX, double& posY)
{
   if ((ctrlYAxis_ == ctrl_) && !ctrl_->IsGCS30 ())
   {
      if (!ctrl_->ReadPositions (axisXName_, &posX, axisYName_, &posY))
      {
         return ctrl_->GetTranslatedError ();
      }
   }
   else
   {
      if (!ctrl_->ReadPosition (axisXName_, &posX))
      {
         return ctrl_->GetTranslatedError ();
      }
      if (!ctrlYAxis_->ReadPosition (axisYName_, &posY))
      {
         return ctrlYAxis_->GetTranslatedError ();
      }
   }
   return DEVICE_OK;
}


int PIXYStage::GetPositionSteps (long& x, long& y)
{
   double posX, posY;
   int result = ReadPositions (posX, posY);
   if (DEVICE_OK != result)
   {
      return result;
   }

   posX -= originX_;
   posY -= originY_;

   x = static_cast<long>(ceil ((posX / (stepSize_um_ * ctrl_->GetUmToDefaultUnit ())) - 0.5));
   y = static_cast<long>(ceil ((posY / (stepSize_um_ * ctrlYAxis_->GetUmToDefaultUnit ())) - 0.5));
   return DEVICE_OK;
}

int PIXYStage::HomeXAxis ()
{
   if (alternativeHomingCommandXAxis_.length () != 0)
   {
      return ctrl_->SendCommand (alternativeHomingCommandXAxis_);
   }
   return ctrl_->Home (axisXName_, axisXHomingMode_);
}

int PIXYStage::HomeYAxis (PIController* ctrl)
{
   if (alternativeHomingCommandYAxis_.length () != 0)
   {
      return ctrl->SendCommand (alternativeHomingCommandYAxis_);
   }
   return ctrl->Home (axisYName_, axisYHomingMode_);
}

int PIXYStage::Home ()
{
   if ((axisXHomingMode_ == axisYHomingMode_)
       && (ctrlYAxis_ == ctrl_)
       && (alternativeHomingCommandXAxis_.length () == 0)
       && (alternativeHomingCommandYAxis_.length () == 0)
       )
   {
      // same mode for both axes, both axes on same controller
      int err = ctrl_->Home (ctrl_->MakeAxesString (axisXName_, axisYName_), axisXHomingMode_);
      if (err != DEVICE_OK)
      {
         return err;
      }
   }
   else
   {
      int ret = HomeXAxis ();
      if (ret != DEVICE_OK)
      {
         return ret;
      }

      while (Busy ())
      {};

      ret = HomeYAxis (ctrlYAxis_);
      if (ret != DEVICE_OK)
      {
         return ret;
      }
   }
   while (Busy ())
   {};

   (void)ctrl_->WriteServo (axisXName_, true);
   (void)ctrlYAxis_->WriteServo (axisYName_, true);
   return DEVICE_OK;
}

int PIXYStage::Stop ()
{
   bool bStop1 = ctrl_->Halt (axisXName_);
   bool bStop2 = ctrlYAxis_->Halt (axisYName_);

   if (bStop1 && bStop2)
   {
      return DEVICE_OK;
   }
   return DEVICE_ERR;
}

int PIXYStage::SetOrigin ()
{
   double posX, posY;
   if ( ReadPositions (posX, posY) != DEVICE_OK)
   {
      return DEVICE_ERR;
   }

   originX_ = posX;
   originY_ = posY;

   return DEVICE_OK;
}

int PIXYStage::GetLimitsUm (double& /*xMin*/, double& /*xMax*/, double& /*yMin*/, double& /*yMax*/)
{
   return DEVICE_UNSUPPORTED_COMMAND;
}

int PIXYStage::GetStepLimits (long& /*xMin*/, long& /*xMax*/, long& /*yMin*/, long& /*yMax*/)
{
   return DEVICE_UNSUPPORTED_COMMAND;
}

double PIXYStage::GetStepSizeXUm ()
{
   return stepSize_um_;
}

double PIXYStage::GetStepSizeYUm ()
{
   return stepSize_um_;
}

bool PIXYStage::ResetBothAxes ()
{
   if (!ctrl_->Reset (axisXName_) || !ctrlYAxis_->Reset (axisYName_))
   {
      return false;
   }
   return true;
}


///////////////////////////////////////////////////////////////////////////////
// Action handlers
///////////////////////////////////////////////////////////////////////////////

int PIXYStage::OnControllerName (MM::PropertyBase* pProp, MM::ActionType eAct)
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

int PIXYStage::OnControllerNameYAxis (MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set (controllerNameYAxis_.c_str ());
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get (controllerNameYAxis_);
   }

   return DEVICE_OK;
}


int PIXYStage::OnAxisXName (MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set (axisXName_.c_str ());
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get (axisXName_);
   }

   return DEVICE_OK;
}

int PIXYStage::OnAxisXStageType (MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set (axisXStageType_.c_str ());
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get (axisXStageType_);
   }

   return DEVICE_OK;
}

int PIXYStage::OnAxisXHoming (MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set (axisXHomingMode_.c_str ());
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get (axisXHomingMode_);
   }

   return DEVICE_OK;
}

int PIXYStage::OnAxisYName (MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set (axisYName_.c_str ());
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get (axisYName_);
   }

   return DEVICE_OK;
}

int PIXYStage::OnAxisYStageType (MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set (axisYStageType_.c_str ());
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get (axisYStageType_);
   }

   return DEVICE_OK;
}

int PIXYStage::OnAxisYHoming (MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set (axisYHomingMode_.c_str ());
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get (axisYHomingMode_);
   }

   return DEVICE_OK;
}

int PIXYStage::OnVelocity (MM::PropertyBase* pProp, MM::ActionType eAct, PIController* controller, std::string& axis)
{
   if (eAct == MM::BeforeGet)
   {
      double velocity = 0.0;
      if (initialized_)
      {
         if (!controller->ReadVelocity (axis, velocity))
         {
            return controller->GetTranslatedError ();
         }
      }
      pProp->Set (velocity);
   }
   else if (eAct == MM::AfterSet)
   {
      double velocity = 0.0;
      pProp->Get (velocity);
      if (initialized_ && !controller->WriteVelocity (axis, velocity))
      {
         int err = controller->GetTranslatedError ();
         if (ERR_GCS_PI_PARAM_VALUE_OUT_OF_RANGE == err)
         {
            err = ERR_GCS_PI_CNTR_VEL_OUT_OF_LIMITS;
         }
         return err;
      }
   }

   return DEVICE_OK;
}

int PIXYStage::OnXVelocity (MM::PropertyBase* pProp, MM::ActionType eAct)
{
   return OnVelocity (pProp, eAct, ctrl_, axisXName_);
}


int PIXYStage::OnYVelocity (MM::PropertyBase* pProp, MM::ActionType eAct)
{
   return OnVelocity (pProp, eAct, ctrlYAxis_, axisYName_);
}

int PIXYStage::OnServoControl (MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      if (initialized_)
      {
         bool svoX;
         if (!ctrl_->ReadServo (axisXName_, svoX))
         {
            return ctrl_->GetTranslatedError ();
         }
         bool svoY;
         if (!ctrlYAxis_->ReadServo (axisYName_, svoY))
         {
            return ctrlYAxis_->GetTranslatedError ();
         }
         servoControl_ = svoX && svoY;
      }
      pProp->Set (long (servoControl_ ? 1 : 0));
   }
   else if (eAct == MM::AfterSet)
   {
      long value;
      pProp->Get (value);
      servoControl_ = (value != 0);
      if (!ctrl_->WriteServo (axisXName_, servoControl_))
      {
         return ctrl_->GetTranslatedError ();
      }
      if (!ctrlYAxis_->WriteServo (axisYName_, servoControl_))
      {
         return ctrlYAxis_->GetTranslatedError ();
      }
   }

   return DEVICE_OK;
}

int PIXYStage::OnEnableAxes (MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      if (initialized_)
      {
         bool eaxX = true;
         if (ctrl_->CanEnableAxis () && !ctrl_->ReadEnableAxis (axisXName_, eaxX))
         {
            return ctrl_->GetTranslatedError ();
         }
         bool eaxY = true;
         if (ctrlYAxis_->CanEnableAxis () && !ctrlYAxis_->ReadEnableAxis (axisYName_, eaxY))
         {
            return ctrlYAxis_->GetTranslatedError ();
         }
         axesEnabled_ = eaxX && eaxY;
      }
      pProp->Set (long (axesEnabled_ ? 1 : 0));
   }
   else if (eAct == MM::AfterSet)
   {
      long value;
      pProp->Get (value);
      axesEnabled_ = (value != 0);
      if (ctrl_->CanEnableAxis () && !ctrl_->WriteEnableAxis (axisXName_, axesEnabled_))
      {
         return ctrl_->GetTranslatedError ();
      }
      if (ctrlYAxis_->CanEnableAxis () && !ctrlYAxis_->WriteEnableAxis (axisYName_, axesEnabled_))
      {
         return ctrlYAxis_->GetTranslatedError ();
      }
   }

   return DEVICE_OK;
}

int PIXYStage::OnAlternativeHomingCommandXAxis (MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set (alternativeHomingCommandXAxis_.c_str ());
   }
   else if (eAct == MM::AfterSet)
   {
      std::string value;
      pProp->Get (value);
      alternativeHomingCommandXAxis_ = value;
   }

   return DEVICE_OK;
}

int PIXYStage::OnAlternativeHomingCommandYAxis (MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set (alternativeHomingCommandYAxis_.c_str ());
   }
   else if (eAct == MM::AfterSet)
   {
      std::string value;
      pProp->Get (value);
      alternativeHomingCommandYAxis_ = value;
   }

   return DEVICE_OK;
}

int PIXYStage::OnAxisErrorState (MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      if (initialized_)
      {
         pProp->Set (long (ctrl_->IsInErrorState (axisXName_) || long (ctrlYAxis_->IsInErrorState (axisYName_)) ? 1 : 0));
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
         if (!ResetBothAxes ())
            return ctrl_->GetTranslatedError ();
      }
   }

   return DEVICE_OK;
}

int PIXYStage::OnHID (MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      bool stateX;
      bool stateY;
      if (!ctrl_->ReadHIN (axisXName_, stateX) || !ctrlYAxis_->ReadHIN (axisYName_, stateY))
      {
         return ctrl_->GetTranslatedError ();
      }
      pProp->Set ((stateX || stateY) ? 1L : 0L);
   }
   else if (eAct == MM::AfterSet)
   {
      long state;
      pProp->Get (state);
      if (!ctrl_->WriteHIN (axisXName_, state == TRUE) || !ctrlYAxis_->WriteHIN (axisYName_, state == TRUE))
      {
         return ctrl_->GetTranslatedError ();
      }
   }

   return DEVICE_OK;
}

void PIXYStage::OnControllerDeleted ()
{
   ctrl_ = ctrlYAxis_ = NULL;
   initialized_ = false;
}
