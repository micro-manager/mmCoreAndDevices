///////////////////////////////////////////////////////////////////////////////
// FILE:          XT900Ctrl.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Definition of X-Cite XT900 Led Controller Class
//
// COPYRIGHT:     S3L GmbH 2021
//
// LICENSE:       This file is distributed under the BSD license.
//                License text is included with the source distribution.
//
//                This file is distributed in the hope that it will be useful,
//                but WITHOUT ANY WARRANTY; without even the implied warranty
//                of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//                IN NO EVENT SHALL THE COPYRIGHT OWNER(S) OR
//                CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//                INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.
//
// AUTHOR:        Steffen Leidenbach
//                based on XCiteXT600 code from Lon Chu (lonchu@yahoo.com) created on July 2011
//

#pragma once

#include "MMDevice.h"
#include "DeviceBase.h"
	
//
// XT900 is a controller from Excelitas.
// It accept remote serial input to conrol micromanipulator.
//

class XLedCtrl : public CGenericBase<XLedCtrl>
{
   public:

	  // contructor & destructor
	  // .......................
      XLedCtrl();
      ~XLedCtrl();

      // Device API
      // ---------
      int Initialize();
      int Shutdown();  // shutdown the controller

      void GetName(char* pszName) const;
      bool Busy() { return false; }

      int ReadAllProperty();
      char* GetXLedStatus(unsigned char* sResp, char* sXLedStatus);
      int GetStatusDescription(long lStatus, char* sStatus);


      // action interface
      // ---------------
      int OnPort(MM::PropertyBase* pProp, MM::ActionType eAct);
      int OnDebugLogFlag(MM::PropertyBase* pProp, MM::ActionType eAct);
      int OnState(MM::PropertyBase* pProp, MM::ActionType eAct);
      int OnAllOnOff(MM::PropertyBase* pProp, MM::ActionType eAct);
      int OnPWMStatus(MM::PropertyBase* pProp, MM::ActionType pAct);
      int OnPWMMode(MM::PropertyBase* pProp, MM::ActionType pAct);
      int OnFrontPanelLock(MM::PropertyBase* pProp, MM::ActionType pAct);
      int OnLCDScrnNumber(MM::PropertyBase* pProp, MM::ActionType pAct);
      int OnLCDScrnBrite(MM::PropertyBase* pProp, MM::ActionType pAct);
      int OnLCDScrnSaver(MM::PropertyBase* pProp, MM::ActionType pAct);
      int OnClearAlarm(MM::PropertyBase* pProp, MM::ActionType pAct);
      int OnSpeakerVolume(MM::PropertyBase* pProp, MM::ActionType pAct);
      int ConnectXLed(unsigned char* sResp);


   private:
      int XLedSerialIO(unsigned char* sCmd, unsigned char* sResp);  // write comand to serial port and read message from serial port
      int WriteCommand(const unsigned char* sCommand);              // write command to serial port
      int ReadMessage(unsigned char* sMessage);                     // read message from serial port

      double        m_dAnswerTimeoutMs;     // maximum waiting time for receiving reolied message
      bool          m_yInitialized;         // controller initialized flag
      long          m_lAllOnOff;            // all on/off flag
      long          m_lPWMState;            // PWM status
      long          m_lPWMMode;             // PWM mode
      long          m_lScrnLock;            // front panel lock
      long          m_lScrnNumber;          // screen number
      long          m_lScrnBrite;           // screen brightness
      long          m_lScrnTimeout;         // screen saver time out
      long          m_lSpeakerVol;          // speaker volume
};
