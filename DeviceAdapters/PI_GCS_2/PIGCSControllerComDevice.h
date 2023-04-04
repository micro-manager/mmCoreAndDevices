///////////////////////////////////////////////////////////////////////////////
// FILE:          PIGCSControllerComDevice.h
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
// CVS:           $Id: PIGCSControllerCom.h,v 1.17, 2018-10-05 05:58:53Z, Steffen Rau$
//

#ifndef PI_GCS_CONTROLLER_COM_DEVICE_H_INCLUDED
#define PI_GCS_CONTROLLER_COM_DEVICE_H_INCLUDED

#include "DeviceBase.h"
#include "PIGCSCommands.h"
#include <string>

class PIController;
class PIGCSControllerComDevice : public CGenericBase<PIGCSControllerComDevice>, public PIGCSCommands
{
public:
   PIGCSControllerComDevice ();
   ~PIGCSControllerComDevice ();

   // Device API
   // ----------
   int Initialize ();
   int Shutdown ();

   void SetFactor_UmToDefaultUnit (double dUmToDefaultUnit, bool bHideProperty = true);

   void CreateProperties ();

   static const char* DeviceName_;
   static const char* UmToDefaultUnitName_;
   static const char* ErrorCheckAfterMOV_;
   static const char* SendCommand_;
   void GetName (char* pszName) const;
   bool Busy ();


   int OnPort (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnUmInDefaultUnit (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnErrorCheck (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnSendCommand (MM::PropertyBase* pProp, MM::ActionType eAct);


   virtual bool SendGCSCommand (const std::string& command);
   virtual bool SendGCSCommand (unsigned char singlebyte);
   virtual bool ReadGCSAnswer (std::vector<std::string>& answer, int nExpectedLines = -1);
   int GetLastError () const
   {
      return lastError_;
   }

   double umToDefaultUnit_;
private:
   int OnJoystick1 (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnJoystick2 (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnJoystick3 (MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnJoystick4 (MM::PropertyBase* pProp, MM::ActionType eAct);

   std::string port_;
   int lastError_;
   bool initialized_;
   bool bShowProperty_UmToDefaultUnit_;
   PIController* ctrl_;
};



#endif //_PI_GCS_CONTROLLER_COM_DEVICE_H_
