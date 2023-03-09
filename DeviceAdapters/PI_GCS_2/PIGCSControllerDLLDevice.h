///////////////////////////////////////////////////////////////////////////////
// FILE:          PIGCSControllerDLLDevice.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   PI GCS Controller Driver
//
// AUTHOR:        Nenad Amodaj, nenad@amodaj.com, 08/28/2006
//                Steffen Rau, s.rau@pi.ws, 28/03/2008
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
// CVS:           $Id: PIGCSCommandsDLL.h,v 1.19, 2019-01-09 10:45:26Z, Steffen Rau$
//

#ifndef PI_GCS_CONTROLLER_DLL_DEVICE_H_INCLUDED
#define PI_GCS_CONTROLLER_DLL_DEVICE_H_INCLUDED

#include "DeviceBase.h"
#include "PIGCSCommandsDLL.h"
#include <string>


class PIController;
class PIGCSControllerDLLDevice : public CGenericBase<PIGCSControllerDLLDevice>
{
public:
   PIGCSControllerDLLDevice ();
   ~PIGCSControllerDLLDevice ();

   // Device API
   // ----------
   int Initialize ();
   int Shutdown ();

   void SetDLL (std::string dll_name);
   void SetInterface (std::string type, std::string parameter);
   void ShowInterfaceProperties (bool bShow);

   void CreateProperties ();
   void CreateInterfaceProperties ();

   static const char* DeviceName_;
   void GetName (char* pszName) const;
   bool Busy ();


   static const char* PropName_;
   static const char* PropInterfaceType_;
   static const char* PropInterfaceParameter_;
   static const char* UmToDefaultUnitName_;
   static const char* SendCommand_;

   int OnDLLName (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnInterfaceType (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnInterfaceParameter (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnUmInDefaultUnit (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnSendCommand (MM::PropertyBase* pProp, MM::ActionType eAct);

private:
   int OnJoystick1 (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnJoystick2 (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnJoystick3 (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnJoystick4 (MM::PropertyBase* pProp, MM::ActionType eAct);

   PIController* ctrl_;
   PIGCSCommandsDLL dll_;
   std::string dllName_;
   std::string interfaceType_;
   std::string interfaceParameter_;
   bool initialized_;
   bool bShowInterfaceProperties_;
   double umToDefaultUnit_;
};


#endif // PI_GCS_CONTROLLER_DLL_DEVICE_H_INCLUDED
