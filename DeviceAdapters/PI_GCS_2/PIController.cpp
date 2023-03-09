///////////////////////////////////////////////////////////////////////////////
// FILE:          PIController.cpp
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
// CVS:           $Id: Controller.cpp,v 1.20, 2018-09-26 11:09:17Z, Steffen Rau$
//

#include "PIController.h"
#include "PIControllerObserver.h"
#include "PIGCSCommands.h"
#include "PI_GCS_2.h"

std::map<std::string, PIController*> PIController::allControllersByLabel_;

PIController::PIController (const std::string& label, MM::Core* logsink, MM::Device* logdevice)
   : gcsCommands_ (NULL)
   , umToDefaultUnit_ (0.001)
   , gcs2_ (true)
   , label_ (label)
   , onlyIDSTAGEvalid_ (false)
   , needResetStages_ (false)
   , referenceMoveActive_ (false)
   , syntaxVersion_ (0.0)
   , timeoutForTestMessages_ (1000)
   , logsink_ (logsink)
   , logdevice_ (logdevice)
{
   allControllersByLabel_[label_] = this;
}

PIController::~PIController ()
{
   for (std::set<PIControllerObserver*>::iterator observer = observers_.begin(); observer != observers_.end(); ++observer)
   {
       (*observer)->OnControllerDeleted ();
   }
   logsink_ = NULL;
   logdevice_ = NULL;
}

void PIController::Attach (PIControllerObserver* observer)
{
   observers_.insert (observer);
}

void PIController::SetGCSCommands (PIGCSCommands* gcsCommands)
{
   gcsCommands_ = gcsCommands;
}

void PIController::SetUmToDefaultUnit (const double umToDefaultUnit)
{
   umToDefaultUnit_ = umToDefaultUnit;
}

double PIController::GetUmToDefaultUnit () const
{
   return umToDefaultUnit_;
}

void PIController::ClearReferenceMoveActive ()
{
   referenceMoveActive_ = false;
}

void PIController::SetGCS2 (bool gcs2)
{
   gcs2_ = gcs2;
}

void PIController::SetOnlyIDSTAGEvalid (const bool onlyIDSTAGEvalid)
{
   onlyIDSTAGEvalid_ = onlyIDSTAGEvalid;
}

void PIController::SetNeedResetStages (const bool needResetStages)
{
   needResetStages_ = needResetStages;
}

bool PIController::Connect ()
{
   if (!gcsCommands_)
   {
      return false;
   }
   gcsCommands_->SetGCS2 (gcs2_);
   std::string idn;
   if (!ReadIdentification (idn))
   {
      return false;
   }
   LogMessage (std::string ("PIController::Connect() connected to ") + idn);

   SetChannelsOnline();
   return true;
}

void PIController::SetChannelsOnline()
{
   int oldTimeout = gcsCommands_->GetTimeout();
   gcsCommands_->SetTimeout(timeoutForTestMessages_);
   if (gcsCommands_->HasONL())
   {
      int nrOutputChannels = GetNrOutputChannels();
      if (nrOutputChannels > 0)
      {
         std::vector<int> outputChannels(nrOutputChannels);
         std::vector<int> values(nrOutputChannels, 1);
         for (int i = 0; i < nrOutputChannels; i++)
         {
            outputChannels[i] = i + 1;
         }
         if (gcsCommands_->HasONL())
         {
            gcsCommands_->ONL(outputChannels, values);
         }
      }
   }
   gcsCommands_->SetTimeout(oldTimeout);
}

int PIController::SendCommand (const std::string& command)
{
   if (!gcsCommands_->SendGCSCommand (command))
   {
      return DEVICE_ERR;
   }
   return DEVICE_OK;
}

void PIController::LogMessage (const std::string& msg) const
{
   if (logsink_ == NULL || logdevice_ == NULL)
   {
      return;
   }
   (void)logsink_->LogMessage (logdevice_, msg.c_str (), true);
}

PIController* PIController::GetByLabel (const std::string& label)
{
   std::map< std::string, PIController*>::iterator ctrl = allControllersByLabel_.find (label);
   if (ctrl == allControllersByLabel_.end ())
   {
      return NULL;
   }
   return (*ctrl).second;
}

void PIController::DeleteByLabel (const std::string& label)
{
   std::map< std::string, PIController*>::iterator ctrl = allControllersByLabel_.find (label);
   if (ctrl == allControllersByLabel_.end ())
   {
      return;
   }
   delete (*ctrl).second;
   allControllersByLabel_.erase (label);
}

int PIController::InitStage (const std::string& axisName, const std::string& stageType)
{
   if (!gcs2_ && gcsCommands_->HasCST () && !stageType.empty ())
   {
      if (needResetStages_)
      {
         if (!gcsCommands_->CST (axisName, "NOSTAGE"))
         {
            return GetTranslatedError ();
         }
      }
      std::string stageType_local = stageType;
      if (onlyIDSTAGEvalid_)
      {
         if (strcmp (stageType.c_str (), "NOSTAGE") != 0)
         {
            stageType_local = "ID-STAGE";
         }
      }
      if (!gcsCommands_->CST (axisName, stageType_local))
      {
         return GetTranslatedError ();
      }
   }
   if (!gcs2_ && gcsCommands_->HasINI ())
   {
      if (!gcsCommands_->INI (axisName))
      {
         return GetTranslatedError ();
      }

   }
   if (gcsCommands_->HasEAX ())
   {
      // try to enable servo
      // This does not work for all stages, see wiki
      if (!gcsCommands_->EAX (axisName, TRUE))
      {
         LogMessage (std::string ("PIController::InitStage() EAX failed"));
         (void)gcsCommands_->GetError ();
      }
   }
   if (gcsCommands_->HasSVO ())
   {
      // try to enable servo
      // This does not work for all stages, see wiki
      if (!gcsCommands_->SVO (axisName, true))
      {
         LogMessage (std::string ("PIController::InitStage() SVO failed"));
         (void)gcsCommands_->GetError ();
      }
   }
   return DEVICE_OK;
}

bool PIController::IsBusy (const std::string& axisName)
{
   return IsGCS30 () ? IsBusyGCS30 (axisName) : IsBusyGCS20 ();
}

bool PIController::IsBusyGCS20 ()
{
   if (referenceMoveActive_)
   {
      LogMessage (std::string ("PIController::IsBusyGCS20(): active referenceMoveActive_"));
      bool ready = true;
      if (gcsCommands_->IsControllerReady (ready))
      {
         if (!ready)
         {
            LogMessage (std::string ("PIController::IsBusyGCS20(): IsControllerReady -> not READY"));
            return true;
         }
      }
   }

   bool moving = false;
   if (gcsCommands_->HasIsMoving () && gcsCommands_->IsMoving (moving))
   {
      if (moving)
      {
         LogMessage (std::string ("PIController::IsBusyGCS20(): IsMoving ->BUSY"));
         return true;
      }
   }

   referenceMoveActive_ = false;
   return false;
}

bool PIController::IsBusyGCS30 (const std::string& axisName)
{
   static const unsigned int INTERNAL_PROCESS_RUNNING = 1 << 20;
   static const unsigned int IN_MOTION = 1 << 18;
   unsigned int status = 0;
   if (gcsCommands_->qSTV (axisName, status))
   {
      if (referenceMoveActive_)
      {
         if (0 != (status & INTERNAL_PROCESS_RUNNING))
         {
            LogMessage (std::string ("PIController::IsBusyGCS30(): IsReferencing ->BUSY"));
            return true;
         }
      }
      if (0 != (status & IN_MOTION))
      {
         LogMessage (std::string ("PIController::IsBusyGCS30(): IsMoving ->BUSY"));
         return true;
      }
   }
   return false;
}

bool PIController::IsInErrorState (const std::string& axisName)
{
   return IsGCS30 () ? IsInErrorStateGCS30 (axisName) : IsInErrorStateGCS20 ();
}

bool PIController::IsInErrorStateGCS20 ()
{
   return false;
}

bool PIController::IsInErrorStateGCS30 (const std::string& axisName)
{
	static const unsigned int ERROR_STATE = 1 << 0;
	unsigned int status = 0;
	if (gcsCommands_->qSTV(axisName, status))
	{
		if (0 != (status & ERROR_STATE))
		{
			LogMessage (std::string ("PIController::IsInErrorStateGCS30(): IsInErrorState ->ERROR"));
			return true;
		}
	}
	return false;
}

std::string PIController::MakeAxesString (const std::string& axis1Name, const std::string& axis2Name) const
{
   if (gcs2_)
   {
      return axis1Name + " \n" + axis2Name;
   }
   else
   {
      return axis1Name + axis2Name;
   }
}


bool PIController::PrepareAxesForReference (const std::vector<std::string>& axes)
{
   if (IsGCS30 ())
   {
      return PrepareAxesForReferenceGCS30(axes);
   }
   else
   {
      return PrepareAxesForReferenceGCS20(axes);
   }
   return true;
}


bool PIController::PrepareAxesForReferenceGCS30 (const std::vector<std::string>& axes)
{
   for (std::vector<std::string>::const_iterator axis = axes.begin (); axis != axes.end (); ++axis)
   {
      if (!gcsCommands_->EAX (*axis, true) || !gcsCommands_->SAM (*axis, 0))
      {
         return false;
      }
   }
   return true;
}


bool PIController::PrepareAxesForReferenceGCS20 (const std::vector<std::string>& axes)
{
   if (gcsCommands_->HasEAX ())
   {
      for (std::vector<std::string>::const_iterator axis = axes.begin (); axis != axes.end (); ++axis)
      {
         if (!gcsCommands_->EAX (*axis, true))
         {
            return false;
         }
      }
   }
   return true;
}


bool PIController::HomeWithFRF (const std::string& axesNames)
{
   std::vector<std::string> axes = Tokenize (axesNames);
   if (!PrepareAxesForReference (axes))
   {
      return false;
   }

   if (IsGCS30 ())
   {
      for (std::vector<std::string>::iterator axis = axes.begin (); axis != axes.end (); ++axis)
      {
         if (!gcsCommands_->FRF (*axis))
         {
            return false;
         }
      }
      return true;
   }
   else
   {
      return gcsCommands_->FRF (axesNames);
   }
}

int PIController::Home (const std::string& axesNames, const std::string& homingMode)
{
   if (homingMode.empty ())
      return DEVICE_OK;

   if ((homingMode == "REF") || (homingMode == "FRF"))
   {
      if (gcsCommands_->HasFRF ())
      {
         if (HomeWithFRF (axesNames))
         {
            referenceMoveActive_ = true;
            return DEVICE_OK;
         }
         else
         {
            return GetTranslatedError ();
         }
      }
      else if (gcsCommands_->HasREF ())
      {
         if (gcsCommands_->REF (axesNames))
         {
            referenceMoveActive_ = true;
            return DEVICE_OK;
         }
         else
         {
            return GetTranslatedError ();
         }
      }
      else
         return DEVICE_OK;
   }
   else if ((homingMode == "MNL") || (homingMode == "FNL"))
   {
      if (gcsCommands_->HasFNL ())
      {
         if (gcsCommands_->FNL (axesNames))
         {
            referenceMoveActive_ = true;
            return DEVICE_OK;
         }
         else
         {
            return GetTranslatedError ();
         }
      }
      else if (gcsCommands_->HasMNL ())
      {
         if (gcsCommands_->MNL (axesNames))
         {
            referenceMoveActive_ = true;
            return DEVICE_OK;
         }
         else
         {
            return GetTranslatedError ();
         }
      }
      else
      {
         return DEVICE_OK;
      }
   }
   else if ((homingMode == "MPL") || (homingMode == "FPL"))
   {
      if (gcsCommands_->HasFPL ())
      {
         if (gcsCommands_->FPL (axesNames))
         {
            referenceMoveActive_ = true;
            return DEVICE_OK;
         }
         else
         {
            return GetTranslatedError ();
         }
      }
      else if (gcsCommands_->HasMPL ())
      {
         if (gcsCommands_->MPL (axesNames))
         {
            referenceMoveActive_ = true;
            return DEVICE_OK;
         }
         else
         {
            return GetTranslatedError ();
         }
      }
      else
      {
         return DEVICE_OK;
      }
   }
   else
   {
      return DEVICE_INVALID_PROPERTY_VALUE;
   }
}

int PIController::GetTranslatedError ()
{
   return gcsCommands_->GetTranslatedError ();
}

int PIController::FindNrJoysticks ()
{
   if (!gcsCommands_->HasJON ())
   {
      return 0;
   }

   int oldTimeout = gcsCommands_->GetTimeout ();
   gcsCommands_->SetTimeout (timeoutForTestMessages_);
   int nrJoysticks = 0;
   int state;
   while (gcsCommands_->qJON (nrJoysticks + 1, state) && nrJoysticks < 5)
   {
      nrJoysticks++;
   }
   gcsCommands_->SetTimeout (oldTimeout);
   GetTranslatedError ();
   return nrJoysticks;
}

int PIController::OnJoystick (MM::PropertyBase* pProp, MM::ActionType eAct, int joystick)
{
   if (eAct == MM::BeforeGet)
   {
      int state;
      if (!gcsCommands_->qJON (joystick, state))
      {
         return GetTranslatedError ();
      }
      pProp->Set (long (state));
   }
   else if (eAct == MM::AfterSet)
   {
      long lstate;
      pProp->Get (lstate);
      int state = int (lstate);
      if (!gcsCommands_->JON (joystick, state))
      {
         return GetTranslatedError ();
      }
   }

   return DEVICE_OK;
}

int PIController::GetNrOutputChannels ()
{
   if (!gcsCommands_->Has_qTPC ())
   {
      return 0;
   }

   int nrOutputChannels = 0;
   gcsCommands_->qTPC (nrOutputChannels);

   gcsCommands_->GetError ();
   return nrOutputChannels;
}

bool PIController::IsGCS30 ()
{
   if (syntaxVersion_ < 0.1)
   {
      if (!gcs2_)
      {
         syntaxVersion_ = 1.0;
      }
      else
      {
         gcsCommands_->qCSV (syntaxVersion_);
      }
   }
   return syntaxVersion_ > 2.09;
}

bool PIController::WriteVelocity (const std::string& axisName, const double velocity)
{
   if (!gcsCommands_->HasVEL ())
   {
      return true;
   }
   if (IsGCS30 ())
   {
      return gcsCommands_->SPV ("RAM", axisName, "TRAJ_1", "0x104", velocity);
   }
   else
   {
      return gcsCommands_->VEL (axisName, velocity);
   }
}

bool PIController::ReadVelocity (const std::string& axisName, double& velocity)
{
   if (!gcsCommands_->HasVEL ())
   {
      return true;
   }
   if (IsGCS30 ())
   {
      return gcsCommands_->qSPV ("RAM", axisName, "TRAJ_1", "0x104", velocity);
   }
   else
   {
      return gcsCommands_->qVEL (axisName, velocity);
   }
}

bool PIController::ReadIdentification (std::string& idn)
{
   if (identification_.empty ())
   {
      if (!gcsCommands_->qIDN (identification_))
      {
         identification_.clear ();
         return false;
      }
      bool isC885 = identification_.find ("C-885") != std::string::npos;
      gcsCommands_->SetAlwaysUseEAX (isC885);
      timeoutForTestMessages_ = isC885 ? 3000 : 1000;
   }
   idn = identification_;
   return true;
}

bool PIController::WriteServo (const std::string& axisName, const bool servo)
{
   return gcsCommands_->SVO (axisName, servo);
}

bool PIController::ReadServo (const std::string& axisName, bool& servo)
{
   return gcsCommands_->qSVO (axisName, servo);
}

bool PIController::CanEnableAxis ()
{
   return gcsCommands_->HasEAX ();
}

bool PIController::WriteEnableAxis (const std::string& axisName, bool eax)
{
   return gcsCommands_->EAX (axisName, eax);
}

bool PIController::ReadHIN (const std::string& axisName, bool& hin)
{
   return gcsCommands_->qHIN (axisName, hin);
}

bool PIController::WriteHIN (const std::string& axisName, bool hin)
{
   return gcsCommands_->HIN (axisName, hin);
}

bool PIController::ReadEnableAxis (const std::string& axisName, bool& eax)
{
   return gcsCommands_->qEAX (axisName, eax);
}

bool PIController::MoveAxis (const std::string& axis, const double target)
{
   return gcsCommands_->MOV (axis, target);
}

bool PIController::MoveAxes (const std::string& axis1, const double target1, const std::string& axis2, const double target2)
{
   return gcsCommands_->MOV (axis1, target1, axis2, target2);
}

bool PIController::ReadPosition (const std::string& axis, double* position)
{
   return gcsCommands_->qPOS (axis, position);
}

bool PIController::ReadPositions (const std::string& axis1, double* position1, const std::string& axis2, double* position2)
{
   return gcsCommands_->qPOS (axis1, position1, axis2, position2);
}

bool PIController::Stop ()
{
   return gcsCommands_->STP ();
}

bool PIController::Halt (const std::string& axis)
{
   return gcsCommands_->HLT (axis);
}

bool PIController::Reset (const std::string& axis)
{
	return gcsCommands_->RES (axis);
}