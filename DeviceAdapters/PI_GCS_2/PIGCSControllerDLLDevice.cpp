///////////////////////////////////////////////////////////////////////////////
// FILE:          PIGCSControllerDLLDevice.cpp
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
// CVS:           $Id: PIGCSCommandsDLL.cpp,v 1.25, 2019-01-09 10:45:26Z, Steffen Rau$
//

#ifndef __APPLE__

#ifndef WIN32
#include <dlfcn.h>
#endif

#include "PI_GCS_2.h"
#include "PIController.h"
#include "PIGCSControllerDLLDevice.h"
#include "PIGCSCommandsDLL.h"

const char* PIGCSControllerDLLDevice::DeviceName_ = "PI_GCSController_DLL";
const char* PIGCSControllerDLLDevice::PropName_ = "DLL Name";
const char* PIGCSControllerDLLDevice::PropInterfaceType_ = "Interface Type";
const char* PIGCSControllerDLLDevice::PropInterfaceParameter_ = "Interface Parameter";
const char* PIGCSControllerDLLDevice::UmToDefaultUnitName_ = "um in default unit";
const char* PIGCSControllerDLLDevice::SendCommand_ = "Send command";

PIGCSControllerDLLDevice::PIGCSControllerDLLDevice ()
   : umToDefaultUnit_ (0.001)
   , ctrl_ (NULL)
   , dllName_ ("PI_GCS2_DLL.dll")
   , interfaceType_ ("")
   , interfaceParameter_ ("")
   , initialized_ (false)
   , bShowInterfaceProperties_ (true)
{
   InitializeDefaultErrorMessages ();

   SetErrorText (ERR_DLL_PI_DLL_NOT_FOUND, g_msg_DLL_NOT_FOUND);
   SetErrorText (ERR_DLL_PI_INVALID_INTERFACE_NAME, g_msg_INVALID_INTERFACE_NAME);
   SetErrorText (ERR_DLL_PI_INVALID_INTERFACE_PARAMETER, g_msg_INVALID_INTERFACE_PARAMETER);
}


PIGCSControllerDLLDevice::~PIGCSControllerDLLDevice ()
{
   Shutdown ();
   ctrl_ = NULL;
}

void PIGCSControllerDLLDevice::SetDLL (std::string dll_name)
{
   dllName_ = dll_name;
}

void PIGCSControllerDLLDevice::SetInterface (std::string type, std::string parameter)
{
   interfaceType_ = type;
   interfaceParameter_ = parameter;
}

void PIGCSControllerDLLDevice::ShowInterfaceProperties (bool bShow)
{
   bShowInterfaceProperties_ = bShow;
}

void PIGCSControllerDLLDevice::CreateProperties ()
{
   // create pre-initialization properties
   // ------------------------------------

   // Name
   CreateProperty (MM::g_Keyword_Name, DeviceName_, MM::String, true);

   // Description
   CreateProperty (MM::g_Keyword_Description, "Physik Instrumente (PI) GCS DLL Adapter", MM::String, true);

   CPropertyAction* pAct;

   // DLL name
   pAct = new CPropertyAction (this, &PIGCSControllerDLLDevice::OnDLLName);
   CreateProperty (PIGCSControllerDLLDevice::PropName_, dllName_.c_str (), MM::String, false, pAct, true);

   pAct = new CPropertyAction (this, &PIGCSControllerDLLDevice::OnUmInDefaultUnit);
   CreateProperty (PIGCSControllerDLLDevice::UmToDefaultUnitName_, "0.001", MM::Float, false, pAct, true);

   pAct = new CPropertyAction (this, &PIGCSControllerDLLDevice::OnSendCommand);
   CreateProperty (PIGCSControllerDLLDevice::SendCommand_, "", MM::String, false, pAct);

   CreateInterfaceProperties ();

}

void PIGCSControllerDLLDevice::CreateInterfaceProperties (void)
{
   CPropertyAction* pAct;
   std::string interfaceParameterLabel = "";

   // Interface type
   if (bShowInterfaceProperties_)
   {
      pAct = new CPropertyAction (this, &PIGCSControllerDLLDevice::OnInterfaceType);
      CreateProperty (PIGCSControllerDLLDevice::PropInterfaceType_, interfaceType_.c_str (), MM::String, false, pAct, true);

      interfaceParameterLabel = PIGCSControllerDLLDevice::PropInterfaceParameter_;
   }
   else
   {
      if (strcmp (interfaceType_.c_str (), "PCI") == 0)
      {
         interfaceParameterLabel = "PCI Board";
      }
      else if (strcmp (interfaceType_.c_str (), "RS-232") == 0)
      {
         interfaceParameterLabel = "ComPort ; Baudrate";
      }
      else if (strcmp (interfaceType_.c_str (), "TCP/IP") == 0)
      {
         interfaceParameterLabel = "IP-Address:port";
      }
   }

   // Interface parameter
   if (interfaceParameterLabel.empty ())
   {
      return;
   }

   pAct = new CPropertyAction (this, &PIGCSControllerDLLDevice::OnInterfaceParameter);
   CreateProperty (interfaceParameterLabel.c_str (), interfaceParameter_.c_str (), MM::String, false, pAct, true);
}

int PIGCSControllerDLLDevice::Initialize ()
{
   if (initialized_)
   {
      return DEVICE_OK;
   }

   char szLabel[MM::MaxStrLength];
   GetLabel (szLabel);
   ctrl_ = new PIController (szLabel, GetCoreCallback (), this);
   ctrl_->SetGCSCommands (&dll_);

   int ret = dll_.LoadDLL (dllName_, ctrl_);
   if (ret != DEVICE_OK)
   {
      LogMessage (std::string ("Cannot load dll ") + dllName_);
      Shutdown ();
      return ret;
   }

   ctrl_->SetUmToDefaultUnit (umToDefaultUnit_);

   ret = dll_.ConnectInterface (interfaceType_, interfaceParameter_);
   if (ret != DEVICE_OK)
   {
      LogMessage ("Cannot connect");
      Shutdown ();
      return ret;
   }

   if (!ctrl_->Connect ())
   {
      LogMessage (std::string ("Cannot connect"));
      Shutdown ();
      return DEVICE_ERR;
   }

   initialized_ = true;

   int nrJoysticks = ctrl_->FindNrJoysticks ();
   if (nrJoysticks > 0)
   {
      CPropertyAction* pAct = new CPropertyAction (this, &PIGCSControllerDLLDevice::OnJoystick1);
      CreateProperty ("Joystick 1", "0", MM::Integer, false, pAct);
   }
   if (nrJoysticks > 1)
   {
      CPropertyAction* pAct = new CPropertyAction (this, &PIGCSControllerDLLDevice::OnJoystick2);
      CreateProperty ("Joystick 2", "0", MM::Integer, false, pAct);
   }
   if (nrJoysticks > 2)
   {
      CPropertyAction* pAct = new CPropertyAction (this, &PIGCSControllerDLLDevice::OnJoystick3);
      CreateProperty ("Joystick 3", "0", MM::Integer, false, pAct);
   }
   if (nrJoysticks > 3)
   {
      CPropertyAction* pAct = new CPropertyAction (this, &PIGCSControllerDLLDevice::OnJoystick4);
      CreateProperty ("Joystick 4", "0", MM::Integer, false, pAct);
   }


   return ret;
}

int PIGCSControllerDLLDevice::Shutdown ()
{
   if (!initialized_)
   {
      return DEVICE_OK;
   }
   char szLabel[MM::MaxStrLength];
   GetLabel (szLabel);
   PIController::DeleteByLabel (szLabel);
   dll_.CloseAndUnload ();
   initialized_ = false;

   return DEVICE_OK;
}

bool PIGCSControllerDLLDevice::Busy ()
{
   return false;
}

void PIGCSControllerDLLDevice::GetName (char* Name) const
{
   CDeviceUtils::CopyLimitedString (Name, DeviceName_);
}

int PIGCSControllerDLLDevice::OnDLLName (MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set (dllName_.c_str ());
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get (dllName_);
   }

   return DEVICE_OK;
}

int PIGCSControllerDLLDevice::OnUmInDefaultUnit (MM::PropertyBase* pProp, MM::ActionType eAct)
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
   }

   return DEVICE_OK;
}

int PIGCSControllerDLLDevice::OnSendCommand (MM::PropertyBase* pProp, MM::ActionType eAct)
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
         return dll_.SendGCSCommand (command);
      }
   }

   return DEVICE_OK;
}

int PIGCSControllerDLLDevice::OnInterfaceType (MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set (interfaceType_.c_str ());
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get (interfaceType_);
   }

   return DEVICE_OK;
}

int PIGCSControllerDLLDevice::OnInterfaceParameter (MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set (interfaceParameter_.c_str ());
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get (interfaceParameter_);
   }

   return DEVICE_OK;
}

int PIGCSControllerDLLDevice::OnJoystick1 (MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (NULL == ctrl_)
   {
      return DEVICE_ERR;
   }
   return ctrl_->OnJoystick (pProp, eAct, 1);
}

int PIGCSControllerDLLDevice::OnJoystick2 (MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (NULL == ctrl_)
   {
      return DEVICE_ERR;
   }
   return ctrl_->OnJoystick (pProp, eAct, 2);
}

int PIGCSControllerDLLDevice::OnJoystick3 (MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (NULL == ctrl_)
   {
      return DEVICE_ERR;
   }
   return ctrl_->OnJoystick (pProp, eAct, 3);
}

int PIGCSControllerDLLDevice::OnJoystick4 (MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (NULL == ctrl_)
   {
      return DEVICE_ERR;
   }
   return ctrl_->OnJoystick (pProp, eAct, 4);
}

#endif
