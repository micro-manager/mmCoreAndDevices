///////////////////////////////////////////////////////////////////////////////
// FILE:          PureFocus.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Device adapter for Prior PureFocus Autofocus System
//                
//                
// AUTHOR:        Nico Stuurman
//
// COPYRIGHT:     Regents of the University of California, 2025
//
// LICENSE:       This file is distributed under the BSD license.
//


#ifndef _PUREFOCUS_H_
#define _PUREFOCUS_H_

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
#define ERR_NOT_IN_RANGE          107


extern const char* g_PureFocusDevice;
extern const char* g_PureFocusDeviceName;
extern const char* g_PureFocusOffsetDeviceName;
extern const char* g_PureFocusAutoFocusDeviceName;
extern const char* g_PureFocusDeviceDescription;
extern const char* g_PureFocusAutoFocusDescription;
extern const char* g_PureFocusOffsetDescription;
extern const char* g_Stepper;
extern const char* g_Piezo;
extern const char* g_Measure;

class PureFocusOffset;
class PureFocusAutoFocus;


class PureFocusHub : public HubBase<PureFocusHub>
{
public:
   PureFocusHub();
   ~PureFocusHub();

   // Device API
   bool Busy();
   void GetName(char* pszName) const;
   int Initialize();
   int Shutdown();

   // Hub API
   int DetectInstalledDevices();

   // Property Action Handlers 
   int OnPort(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnVersion(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnBuildDate(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnFocusControl(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnPiezoPosition(MM::PropertyBase* pProp, MM::ActionType eAct);

   int GetOffset(long& offset);
   int SetOffset(long offset);
   int SetServo(bool state);
   int GetServo(bool& state);
   bool IsInFocus();
   bool IsSampleDetected();
   int GetFocusScore(double& score);

   // Helper methods for properties
   int GetPinholeProperties(int& pinholeColumns, int& pinholeWidth);
   int SetPinholeProperties(int columns, int width);
   int GetLaserPower(int& laserPower);
   int SetLaserPower(int laserPower);
   int GetObjective(int& objective);
   int SetObjective(int objective);
   int GetList(std::string& list);

private:
   bool initialized_;
   std::string name_;
   std::string port_;
   bool locked_;
   bool busy_;
   long answerTimeoutMs_;
   int statusMode_;
   std::string dateString_;
   std::string version_;
   std::string date_;
   int piezoRange_;
   MMThreadLock lock_; // Thread lock for serial communication

   // Communication methods
   int GetResponse(std::string& resp);
   std::string RemoveLineEndings(std::string input);

   size_t findNthChar(const std::string& str, char targetChar, int n);
};

class PureFocusOffset : public CStageBase<PureFocusOffset> {
public:
 
   PureFocusOffset();
   ~PureFocusOffset();

   // Device APIr
   bool Busy();
   void GetName(char* pszName) const;
   int Initialize();
   int Shutdown();

   // Stage API

   int SetPositionUm(double pos);
   int GetPositionUm(double& pos);
   int SetPositionSteps(long steps);
   int GetPositionSteps(long& steps);
   int SetOrigin()
   {
      return DEVICE_UNSUPPORTED_COMMAND;
   };
   int GetLimits(double& /* lower */, double& /* upper */)
   {
      return DEVICE_UNSUPPORTED_COMMAND;
   };
   int IsStageSequenceable(bool& isSequenceable) const {
      isSequenceable = false;
      return DEVICE_OK;
   };
   bool IsContinuousFocusDrive() const {
      return false;
   };


private:
   bool initialized_;
   PureFocusHub* pHub_;
   double stepSize_;
};

class PureFocusAutoFocus : public CAutoFocusBase<PureFocusAutoFocus>
{
public:

   PureFocusAutoFocus();
   ~PureFocusAutoFocus();

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

   int OnServo(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnFocus(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnSampleDetected(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnFocusScore(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnPinholeCenter(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnPinholeWidth(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnLaserPower(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnObjective(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnList(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
   bool initialized_;
   PureFocusHub* pHub_;
   int pinholeCenter_;
   int pinholeWidth_;
   int laserPower_;
   int objective_;
};

#endif // _PUREFOCUS_H_#pragma once
