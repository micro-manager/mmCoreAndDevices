///////////////////////////////////////////////////////////////////////////////
// FILE:          PIGCSControllerComDevice.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   PI GCS DLL Controller Driver
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
// CVS:           $Id: PIGCSControllerCom.cpp,v 1.23, 2019-01-09 10:47:09Z, Steffen Rau$
//


#include "PIGCSControllerComDevice.h"
#include "PIController.h"
#include "PI_GCS_2.h"

const char* PIGCSControllerComDevice::DeviceName_ = "PI_GCSController";
const char* PIGCSControllerComDevice::UmToDefaultUnitName_ = "um in default unit";
const char* PIGCSControllerComDevice::ErrorCheckAfterMOV_ = "Error Check after MOV command";
const char* PIGCSControllerComDevice::SendCommand_ = "Send command";

PIGCSControllerComDevice::PIGCSControllerComDevice ()
   : umToDefaultUnit_ (0.001)
   , port_ ("")
   , lastError_ (DEVICE_OK)
   , initialized_ (false)
   , bShowProperty_UmToDefaultUnit_ (true)
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
}

PIGCSControllerComDevice::~PIGCSControllerComDevice ()
{
   Shutdown ();
   ctrl_ = NULL;
}

void PIGCSControllerComDevice::SetFactor_UmToDefaultUnit (double dUmToDefaultUnit, bool bHideProperty)
{
   umToDefaultUnit_ = dUmToDefaultUnit;
   if (bHideProperty)
   {
      bShowProperty_UmToDefaultUnit_ = false;
   }

}

void PIGCSControllerComDevice::CreateProperties ()
{
   // create pre-initialization properties
   // ------------------------------------

   // Name
   CreateProperty (MM::g_Keyword_Name, DeviceName_, MM::String, true);

   // Description
   CreateProperty (MM::g_Keyword_Description, "Physik Instrumente (PI) GCS DLL Adapter", MM::String, true);

   CPropertyAction* pAct;

   // Port
   pAct = new CPropertyAction (this, &PIGCSControllerComDevice::OnPort);
   CreateProperty (MM::g_Keyword_Port, "Undefined", MM::String, false, pAct, true);

   if (bShowProperty_UmToDefaultUnit_)
   {
      // axis limit in um
      pAct = new CPropertyAction (this, &PIGCSControllerComDevice::OnUmInDefaultUnit);
      CreateProperty (PIGCSControllerComDevice::UmToDefaultUnitName_, "0.001", MM::Float, false, pAct, true);
   }

   // ErrorCheckAfterMOV
   pAct = new CPropertyAction (this, &PIGCSControllerComDevice::OnErrorCheck);
   CreateProperty (PIGCSControllerComDevice::ErrorCheckAfterMOV_, "1", MM::Integer, false, pAct, false);

   pAct = new CPropertyAction (this, &PIGCSControllerComDevice::OnSendCommand);
   CreateProperty (PIGCSControllerComDevice::SendCommand_, "", MM::String, false, pAct);
}

int PIGCSControllerComDevice::OnPort (MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set (port_.c_str ());
   }
   else if (eAct == MM::AfterSet)
   {
      if (initialized_)
      {
         // revert
         pProp->Set (port_.c_str ());
         return ERR_PORT_CHANGE_FORBIDDEN;
      }

      pProp->Get (port_);
   }

   return DEVICE_OK;
}

int PIGCSControllerComDevice::OnUmInDefaultUnit (MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set (umToDefaultUnit_);
   }
   else if (eAct == MM::AfterSet)
   {
      double value;
      pProp->Get (value);
      if (value < 1e-15)
      {
          return DEVICE_INVALID_PROPERTY_VALUE;
      }
      umToDefaultUnit_ = value;
      if (ctrl_)
      {
         ctrl_->SetUmToDefaultUnit (umToDefaultUnit_);
      }
   }

   return DEVICE_OK;
}

int PIGCSControllerComDevice::OnErrorCheck (MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (ctrl_)
   {
      if (eAct == MM::BeforeGet)
      {
         pProp->Set (long (GetErrorCheckAfterMOV () ? 1 : 0));
      }
      else if (eAct == MM::AfterSet)
      {
         long value;
         pProp->Get (value);
         SetErrorCheckAfterMOV (value != 0);
      }
   }
   return DEVICE_OK;
}

int PIGCSControllerComDevice::OnSendCommand (MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set ("");
   }
   else if (eAct == MM::AfterSet)
   {
      std::string command;
      pProp->Get (command);
      if (command.length () > 0)
      {
         if (!SendGCSCommand (command))
         {
            return lastError_;
         }
      }
   }

   return DEVICE_OK;
}

int PIGCSControllerComDevice::Initialize ()
{
   if (initialized_)
   {
      return DEVICE_OK;
   }

   char szLabel[MM::MaxStrLength];
   GetLabel (szLabel);
   ctrl_ = new PIController (szLabel, GetCoreCallback (), this);
   ctrl_->SetGCSCommands (this);

   if (!ctrl_->Connect ())
   {
      LogMessage (std::string ("Cannot connect"));
      Shutdown ();
      return DEVICE_ERR;
   }

   ctrl_->SetUmToDefaultUnit (umToDefaultUnit_);
   int nrJoysticks = ctrl_->FindNrJoysticks ();
   if (nrJoysticks > 0)
   {
      CPropertyAction* pAct = new CPropertyAction (this, &PIGCSControllerComDevice::OnJoystick1);
      CreateProperty ("Joystick 1", "0", MM::Integer, false, pAct);
   }
   if (nrJoysticks > 1)
   {
      CPropertyAction* pAct = new CPropertyAction (this, &PIGCSControllerComDevice::OnJoystick2);
      CreateProperty ("Joystick 2", "0", MM::Integer, false, pAct);
   }
   if (nrJoysticks > 2)
   {
      CPropertyAction* pAct = new CPropertyAction (this, &PIGCSControllerComDevice::OnJoystick3);
      CreateProperty ("Joystick 3", "0", MM::Integer, false, pAct);
   }
   if (nrJoysticks > 3)
   {
      CPropertyAction* pAct = new CPropertyAction (this, &PIGCSControllerComDevice::OnJoystick4);
      CreateProperty ("Joystick 4", "0", MM::Integer, false, pAct);
   }


   initialized_ = true;
   return DEVICE_OK;
}

int PIGCSControllerComDevice::Shutdown ()
{
   if (!initialized_)
   {
      return DEVICE_OK;
   }
   char szLabel[MM::MaxStrLength];
   GetLabel (szLabel);
   PIController::DeleteByLabel (szLabel);
   initialized_ = false;

   return DEVICE_OK;
}

bool PIGCSControllerComDevice::Busy ()
{
   return false;
}

void PIGCSControllerComDevice::GetName (char* Name) const
{
   CDeviceUtils::CopyLimitedString (Name, DeviceName_);
}

bool PIGCSControllerComDevice::SendGCSCommand (unsigned char singlebyte)
{
   int ret = WriteToComPort (port_.c_str (), &singlebyte, 1);
   if (ret != DEVICE_OK)
   {
      lastError_ = ret;
      return false;
   }
   return true;
}

bool PIGCSControllerComDevice::SendGCSCommand (const std::string& command)
{
   int ret = SendSerialCommand (port_.c_str (), command.c_str (), "\n");
   if (ret != DEVICE_OK)
   {
      lastError_ = ret;
      return false;
   }
   return true;
}

bool PIGCSControllerComDevice::ReadGCSAnswer (std::vector<std::string>& answer, int nExpectedLines)
{
   answer.clear ();
   std::string line;
   do
   {
      int start = GetTickCountInMs ();
      int ret = DEVICE_OK;
      // sometimes timeout cannot be changed according to GCS-device needs (TCP/IP adapter?)
      // or user forget to set it => use our own timeout
      do
      {
         ret = GetSerialAnswer (port_.c_str (), "\n", line);
         if (ret != DEVICE_OK && ret != DEVICE_SERIAL_COMMAND_FAILED)
         {
            lastError_ = ret;
            return false;
         }
      } while ((ret == DEVICE_SERIAL_COMMAND_FAILED) && ((GetTickCountInMs () - start) < timeout_));
      if (ret != DEVICE_OK)
      {
         lastError_ = ret;
         return false;
      }
      answer.push_back (line);
   } while (!line.empty () && line[line.length () - 1] == ' ');
   if ((nExpectedLines >= 0) && (int (answer.size ()) != nExpectedLines))
   {
      return false;
   }
   return true;
}

int PIGCSControllerComDevice::OnJoystick1 (MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (NULL == ctrl_)
   {
      return DEVICE_ERR;
   }
   return ctrl_->OnJoystick (pProp, eAct, 1);
}

int PIGCSControllerComDevice::OnJoystick2 (MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (NULL == ctrl_)
   {
      return DEVICE_ERR;
   }
   return ctrl_->OnJoystick (pProp, eAct, 2);
}

int PIGCSControllerComDevice::OnJoystick3 (MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (NULL == ctrl_)
   {
      return DEVICE_ERR;
   }
   return ctrl_->OnJoystick (pProp, eAct, 3);
}

int PIGCSControllerComDevice::OnJoystick4 (MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (NULL == ctrl_)
   {
      return DEVICE_ERR;
   }
   return ctrl_->OnJoystick (pProp, eAct, 4);
}
