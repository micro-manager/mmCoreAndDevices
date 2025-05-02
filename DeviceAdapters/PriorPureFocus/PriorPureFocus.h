///////////////////////////////////////////////////////////////////////////////
// FILE:          PriorPureFocus.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Device adapter for Prior PureFocus Autofocus System
//                
//                
// AUTHOR:        Your Name
//
// COPYRIGHT:     Your Institution, 2025
//
// LICENSE:       This file is distributed under the BSD license.
//

#ifndef _PRIORPUREFOCUS_H_
#define _PRIORPUREFOCUS_H_

#include "MMDevice.h"
#include "DeviceBase.h"
#include "DeviceThreads.h"
#include <string>

//////////////////////////////////////////////////////////////////////////////
// Error codes
//
#define ERR_PORT_CHANGE_FORBIDDEN 101
#define ERR_DEVICE_NOT_FOUND      102
#define ERR_UNEXPECTED_RESPONSE   103
#define ERR_RESPONSE_TIMEOUT      104
#define ERR_COMMUNICATION         105
#define ERR_AUTOFOCUS_LOCKED      106

class CPureFocus : public CAutoFocusBase<CPureFocus>
{
public:
   CPureFocus();
   ~CPureFocus();

   // Device API
   bool Busy();
   void GetName(char* pszName) const;
   int Initialize();
   int Shutdown();

   // AutoFocus API
   int SetContinuousFocusing(bool state);
   int GetContinuousFocusing(bool& state);
   bool IsContinuousFocusLocked();
   int FullFocus();
   int IncrementalFocus();
   int GetLastFocusScore(double& score);
   int GetCurrentFocusScore(double& score);
   int GetOffset(double& offset);
   int SetOffset(double offset);

   // Property Action Handlers
   int OnPort(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnStatus(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnLock(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnFocusScore(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnOffset(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnPinholeColumns(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnPinholeWidth(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnLaserPower(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnObjective(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnVersion(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnBuildDate(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
   bool initialized_;
   std::string name_;
   std::string port_;
   bool locked_;
   bool busy_;
   long answerTimeoutMs_;
   int statusMode_;
   int pinholeColumns_;
   int pinholeWidth_;
   int laserPower_;
   int objective_;
   std::string dateString_;
   std::string version_;
   std::string date_;
   MMThreadLock lock_; // Thread lock for serial communication

   // Communication methods
   int GetResponse(std::string& resp);
   std::string RemoveLineEndings(std::string input);

   // Helper methods for properties
   int UpdatePinholeProperties();
   int UpdateLaserPower();
   int UpdateObjective();
   size_t findNthChar(const std::string& str, char targetChar, int n);
};

#endif // _PRIORPUREFOCUS_H_#pragma once
