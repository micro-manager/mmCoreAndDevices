///////////////////////////////////////////////////////////////////////////////
// FILE:          PIController.h
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
// CVS:           $Id: Controller.h,v 1.17, 2018-09-26 11:11:16Z, Steffen Rau$
//

#ifndef PI_CONTROLLER_H_INCLUDED
#define PI_CONTROLLER_H_INCLUDED


#include "DeviceBase.h"
#include <string>
#include <set>

#ifndef WIN32
#define WINAPI
#define BOOL int
#define TRUE 1
#define FALSE 0
#endif

size_t ci_find (const std::string& str1, const std::string& str2);

class PIGCSCommands;
class PIControllerObserver;
class PIController
{
public:
   explicit PIController (const std::string& label, MM::Core* logsink, MM::Device* logdevice);
   virtual ~PIController ();

   void SetGCSCommands (PIGCSCommands* gcsCommands);
   bool Connect ();

   static PIController* GetByLabel (const std::string& label);
   static void DeleteByLabel (const std::string& label);

   int InitStage (const std::string& axisName, const std::string& stageType);

   bool IsBusy (const std::string& axisName);

   int Home (const std::string& axesNames, const std::string& homingMode);

   bool IsGCS30 ();

   bool WriteVelocity (const std::string& axisName, double velocity);
   bool ReadVelocity (const std::string& axisName, double& velocity);

   bool WriteServo (const std::string& axisName, bool servo);
   bool ReadServo (const std::string& axisName, bool& servo);

   bool WriteEnableAxis (const std::string& axisName, bool eax);
   bool ReadEnableAxis (const std::string& axisName, bool& eax);
   virtual bool CanEnableAxis ();

   bool ReadIdentification (std::string& idn);

   bool MoveAxis (const std::string& axis, double target);
   bool MoveAxes (const std::string& axis1, double target1, const std::string& axis2, double target2);
   bool ReadPosition (const std::string& axis, double* position);
   bool ReadPositions (const std::string& axis1, double* position1, const std::string& axis2, double* position2);

   virtual bool Stop ();
   virtual bool Halt (const std::string& axis);
   virtual bool Reset (const std::string& axis);

   std::string MakeAxesString (const std::string& axis1Name, const std::string& axis2Name) const;

   int GetTranslatedError ();

   virtual int SendCommand (const std::string& command);


   int FindNrJoysticks ();
   int OnJoystick (MM::PropertyBase* pProp, MM::ActionType eAct, int joystick);
   int GetNrOutputChannels ();
   bool ReadHIN (const std::string& axis, bool& hin);
   bool WriteHIN (const std::string& axis, bool hin);

   void SetUmToDefaultUnit (double umToDefaultUnit);
   double GetUmToDefaultUnit () const;
   void ClearReferenceMoveActive ();
   void SetGCS2 (bool gcs2);
   void SetOnlyIDSTAGEvalid (bool onlyIDSTAGEvalid);
   void SetNeedResetStages (bool needResetStages);

   bool IsInErrorState (const std::string& axisName);
   bool IsInErrorStateGCS20 ();
   bool IsInErrorStateGCS30 (const std::string& axisName);

   void Attach (PIControllerObserver* observer);

protected:
   PIController() {}

   bool PrepareAxesForReference (const std::vector<std::string>& axes);
   bool PrepareAxesForReferenceGCS30 (const std::vector<std::string>& axes);
   bool PrepareAxesForReferenceGCS20 (const std::vector<std::string>& axes);
   bool HomeWithFRF (const std::string& axesNames);
   void SetChannelsOnline();

   bool IsBusyGCS20 ();
   bool IsBusyGCS30 (const std::string& axisName);

   void LogMessage (const std::string& msg) const;

   PIGCSCommands* gcsCommands_;
   double umToDefaultUnit_;
   bool gcs2_;
   std::string label_;
   bool onlyIDSTAGEvalid_;
   bool needResetStages_;
   static std::map<std::string, PIController*> allControllersByLabel_;
   bool referenceMoveActive_;
   double syntaxVersion_;
   std::string identification_;
   int timeoutForTestMessages_;

   MM::Core* logsink_;
   MM::Device* logdevice_;
   std::set<PIControllerObserver*> observers_;
};



#endif //PI_CONTROLLER_H_INCLUDED
