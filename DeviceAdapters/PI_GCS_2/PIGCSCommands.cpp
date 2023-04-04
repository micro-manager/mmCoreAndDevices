///////////////////////////////////////////////////////////////////////////////
// FILE:          PIGCSCommands.cpp
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

#include "PIGCSCommands.h"
#include "PI_GCS_2.h"
#include "PIController.h" // error codes


PIGCSCommands::PIGCSCommands ()
   : controllerError_ (PI_CNTR_NO_ERROR)
   , alwaysUseEAX_ (false)
   , gcs2_ (true)
   , errorCheckAfterMOV_ (true)
   , timeout_ (5000)
   , hasqCSV_ (true)
   , hasINI_ (true)
   , hasCST_ (true)
   , hasSVO_ (true)
   , hasEAX_ (true)
   , hasFRF_ (true)
   , hasREF_ (true)
   , hasFNL_ (true)
   , hasMNL_ (true)
   , hasFPL_ (true)
   , hasMPL_ (true)
   , hasVEL_ (true)
   , hasHLT_ (true)
   , hasRES_ (true)
   , hasSAM_ (true)
   , hasJON_ (true)
   , has_qTPC_ (true)
   , hasONL_ (true)
   , hasIsMoving_ (true)
   , hasHIN_ (true)
{}

PIGCSCommands::~PIGCSCommands ()
{}

void PIGCSCommands::SetAlwaysUseEAX (bool alwaysUseEAX)
{
   alwaysUseEAX_ = alwaysUseEAX;
}

void PIGCSCommands::SetGCS2 (bool gcs2)
{
   gcs2_ = gcs2;
}

bool PIGCSCommands::GetErrorCheckAfterMOV () const
{
   return errorCheckAfterMOV_;
}

void PIGCSCommands::SetErrorCheckAfterMOV (bool errorCheck)
{
   errorCheckAfterMOV_ = errorCheck;
}


bool PIGCSCommands::SendGCSCommandAndReadAnswer (const std::string& command, std::vector<std::string>& answer, int nExpectedLines)
{
   if (!SendGCSCommand (command))
   {
      return false;
   }
   return ReadGCSAnswer (answer, nExpectedLines);
}

bool PIGCSCommands::SendGCSCommandAndReadAnswer (unsigned char singlebyte, std::vector<std::string>& answer, int nExpectedLines)
{
   if (!SendGCSCommand (singlebyte))
   {
      return false;
   }
   return ReadGCSAnswer (answer, nExpectedLines);
}


int PIGCSCommands::GetError ()
{
   if (PI_CNTR_NO_ERROR != controllerError_)
   {
      int error = controllerError_;
      controllerError_ = PI_CNTR_NO_ERROR;
      return error;
   }
   std::vector<std::string> answer;
   if (!SendGCSCommandAndReadAnswer ("ERR?", answer))
   {
      return COM_ERROR;
   }
   // empty answer => try again
   if (answer[0].empty ())
   {
      if (!SendGCSCommandAndReadAnswer ("ERR?", answer))
      {
         return COM_ERROR;
      }
   }
   long error;
   if (!GetValue (answer[0], error))
   {
      return COM_ERROR;
   }
   return error;
}

bool PIGCSCommands::CheckError (bool& hasCmdFlag)
{
   int err = GetError ();
   bool timeout = false;
   if (err == COM_TIMEOUT_ERROR)
   {
      // unknown query commands may also set timeout. Ask again to find out cause for timeout.
      timeout = true;
      err = GetError ();
   }
   if (err == PI_CNTR_UNKNOWN_COMMAND
       || err == PI_ERROR_CMD_CMD_UNKNOWN_COMMAND)
   {
      hasCmdFlag = false;
      return true;
   }
   if (timeout && (err == PI_CNTR_NO_ERROR))
   {
      // do not accidentally clear timeout error
      err = COM_TIMEOUT_ERROR;
   }
   controllerError_ = err;
   return (err == PI_CNTR_NO_ERROR);
}

bool PIGCSCommands::CheckError (void)
{
   controllerError_ = GetError ();
   return (controllerError_ == PI_CNTR_NO_ERROR);
}


int PIGCSCommands::GetTranslatedError ()
{
   return TranslateError (GetError ());
}

bool PIGCSCommands::qIDN (std::string& sIDN)
{
   std::vector<std::string> answer;
   if (!SendGCSCommandAndReadAnswer ("*IDN?", answer))
   {
      return false;
   }
   sIDN = answer[0];
   return true;
}

bool PIGCSCommands::qCSV (double& syntaxVersion)
{
   if (!hasqCSV_)
   {
      return false;
   }
   std::ostringstream command;
   command << "CSV?";
   std::vector<std::string> answer;
   if (!SendGCSCommandAndReadAnswer (command.str (), answer, 1))
   {
      return false;
   }

   double value;
   if (!GetValue (answer[0], value))
   {
      return false;
   }
   syntaxVersion = value;
   return CheckError (hasqCSV_);
}

bool PIGCSCommands::INI (const std::string& axis)
{
   if (!hasINI_)
   {
      return false;
   }
   std::ostringstream command;
   command << "INI " << axis;
   if (!SendGCSCommand (command.str ()))
   {
      return false;
   }

   return CheckError (hasINI_);
}

bool PIGCSCommands::CST (const std::string& axis, const std::string& stagetype)
{
   if (!hasCST_)
   {
      return false;
   }
   std::ostringstream command;
   command << "CST " << axis << " " << stagetype;
   if (!SendGCSCommand (command.str ()))
   {
      return false;
   }

   return CheckError (hasCST_);
}

bool PIGCSCommands::SVO (const std::string& axis, bool svo)
{
   if (!hasSVO_)
   {
      return false;
   }
   std::ostringstream command;
   command << "SVO " << axis << " " << (svo ? "1" : "0");
   if (!SendGCSCommand (command.str ()))
   {
      return false;
   }

   return CheckError (hasSVO_);
};

bool PIGCSCommands::qSVO (const std::string& axis, bool& svo)
{
   if (!hasSVO_)
   {
      return false;
   }
   std::ostringstream command;
   command << "SVO? " << axis;
   std::vector<std::string> answer;
   if (!SendGCSCommandAndReadAnswer (command.str (), answer, 1))
   {
      return false;
   }
   long value;
   if (!GetValue (answer[0], value))
   {
      return false;
   }
   svo = (TRUE == value);
   return true;
};

bool PIGCSCommands::EAX (const std::string& axis, bool enableAxis)
{
   if (!hasEAX_ && !alwaysUseEAX_)
   {
      return false;
   }
   std::ostringstream command;
   command << "EAX " << axis << " " << (enableAxis ? "1" : "0");
   if (!SendGCSCommand (command.str ()))
   {
      return false;
   }

   return CheckError (hasEAX_);
};

bool PIGCSCommands::qEAX (const std::string& axis, bool& eax)
{
   if (!hasEAX_ && !alwaysUseEAX_)
   {
      return false;
   }
   std::ostringstream command;
   command << "EAX? " << axis;
   std::vector<std::string> answer;
   if (SendGCSCommandAndReadAnswer (command.str (), answer, 1))
   {
      long value;
      if (GetValue (answer[0], value))
      {
         eax = (TRUE == value);
         return true;
      }
   }
   return CheckError (hasEAX_);
};

bool PIGCSCommands::FRF (const std::string& axes)
{
   std::ostringstream command;
   command << "FRF " << ConvertToAxesStringWithSpaces (axes);
   if (!SendGCSCommand (command.str ()))
   {
      return false;
   }
   return CheckError ();
}

bool PIGCSCommands::REF (const std::string& axes)
{
   std::ostringstream command;
   command << "REF " << ConvertToAxesStringWithSpaces (axes);
   if (!SendGCSCommand (command.str ()))
   {
      return false;
   }
   return CheckError ();
}

bool PIGCSCommands::MNL (const std::string& axes)
{
   std::ostringstream command;
   command << "MNL " << ConvertToAxesStringWithSpaces (axes);
   if (!SendGCSCommand (command.str ()))
   {
      return false;
   }
   return CheckError ();
}

bool PIGCSCommands::FNL (const std::string& axes)
{
   std::ostringstream command;
   command << "FNL " << ConvertToAxesStringWithSpaces (axes);
   if (!SendGCSCommand (command.str ()))
   {
      return false;
   }
   return CheckError ();
}

bool PIGCSCommands::FPL (const std::string& axes)
{
   std::ostringstream command;
   command << "FPL " << ConvertToAxesStringWithSpaces (axes);
   if (!SendGCSCommand (command.str ()))
   {
      return false;
   }
   return CheckError ();
}

bool PIGCSCommands::MPL (const std::string& axes)
{
   std::ostringstream command;
   command << "MPL " << ConvertToAxesStringWithSpaces (axes);
   if (!SendGCSCommand (command.str ()))
   {
      return false;
   }
   return CheckError ();
}


bool PIGCSCommands::qPOS (const std::string& axis, double* position)
{
   std::ostringstream command;
   command << "POS? " << axis;
   std::vector<std::string> answer;
   if (!SendGCSCommandAndReadAnswer (command.str (), answer, 1))
   {
      return false;
   }
   double value;
   if (!GetValue (answer[0], value))
   {
      return false;
   }
   *position = value;
   return true;
}

bool PIGCSCommands::qPOS (const std::string& axis1, double* position1, const std::string& axis2, double* position2)
{
   std::ostringstream command;
   command << "POS? " << axis1 << " " << axis2;
   std::vector<std::string> answer;
   if (!SendGCSCommandAndReadAnswer (command.str (), answer, 2))
   {
      return false;
   }
   double value[2];
   if (!GetValue (answer[0], value[0]))
   {
      return false;
   }
   if (!GetValue (answer[1], value[1]))
   {
      return false;
   }
   *position1 = value[0];
   *position2 = value[1];
   return true;
}

bool PIGCSCommands::MOV (const std::string& axis, double target)
{
   std::ostringstream command;
   command << "MOV " << axis << " " << target;
   if (!SendGCSCommand (command.str ()))
   {
      return false;
   }
   if (errorCheckAfterMOV_)
   {
      return CheckError ();
   }
   return true;
}

bool PIGCSCommands::MOV (const std::string& axis1, double target1, const std::string& axis2, double target2)
{
   std::ostringstream command;
   command << "MOV " << axis1 << " " << target1 << " " << axis2 << " " << target2;
   if (!SendGCSCommand (command.str ()))
   {
      return false;
   }
   if (errorCheckAfterMOV_)
   {
      return CheckError ();
   }
   return true;
}

std::string PIGCSCommands::ConvertToAxesStringWithSpaces (const std::string& axes) const
{
   if (!gcs2_)
   {
      return axes;
   }

   std::string axesstring;
   std::vector<std::string> lines = Tokenize (axes);
   std::vector<std::string>::iterator line;
   for (line = lines.begin (); line != lines.end (); ++line)
   {
      axesstring += (*line) + " ";
   }
   return axesstring;
}

bool PIGCSCommands::SPV (const std::string& memory,
                         const std::string& containerUnit,
                         const std::string& functionUnit,
                         const std::string& parameter,
                         double value)
{
   std::ostringstream command;
   command << "SPV " << memory << " "
      << containerUnit << " "
      << functionUnit << " "
      << parameter << " "
      << value;
   if (!SendGCSCommand (command.str ()))
   {
      return false;
   }
   return CheckError ();
}

bool PIGCSCommands::qSPV (const std::string& memory,
                          const std::string& containerUnit,
                          const std::string& functionUnit,
                          const std::string& parameter,
                          double& value)
{
   std::ostringstream command;
   command << "SPV? " << memory << " "
      << containerUnit << " "
      << functionUnit << " "
      << parameter;
   std::vector<std::string> answer;
   if (SendGCSCommandAndReadAnswer (command.str (), answer, 1))
   {
      if (GetValue (answer[0], value))
      {
         return true;
      }
   }
   return false;
}

bool PIGCSCommands::VEL (const std::string& axis, const double velocity)
{
   if (!hasVEL_)
   {
      return false;
   }
   std::ostringstream command;
   command << "VEL " << axis << " " << velocity;
   if (!SendGCSCommand (command.str ()))
   {
      return false;
   }
   return CheckError (hasVEL_);
}

bool PIGCSCommands::qVEL (const std::string& axis, double& velocity)
{
   if (!hasVEL_)
   {
      return false;
   }
   std::ostringstream command;
   command << "VEL? " << axis;
   std::vector<std::string> answer;
   if (SendGCSCommandAndReadAnswer (command.str (), answer, 1))
   {
      double value;
      if (GetValue (answer[0], value))
      {
         velocity = value;
         return true;
      }

   }
   return CheckError (hasVEL_);
}

bool PIGCSCommands::STP ()
{
   return SendGCSCommand ("STP");
}

bool PIGCSCommands::HLT (const std::string& axis)
{
   if (!hasHLT_)
   {
      return false;
   }
   std::ostringstream command;
   command << "HLT " << axis;
   if (!SendGCSCommand (command.str ()))
   {
      return false;
   }
   return CheckError (hasHLT_);
}

bool PIGCSCommands::RES(const std::string& axis)
{
   if (!hasRES_)
   {
      return false;
   }
   std::ostringstream command;
   command << "RES " << axis;
   if (!SendGCSCommand (command.str ()))
   {
      return false;
   }
   return CheckError (hasRES_);
}

bool PIGCSCommands::SAM (const std::string& axis, const unsigned int mode)
{
   if (!hasSAM_)
   {
      return false;
   }
   std::ostringstream command;
   command << "SAM " << axis << " " << mode;
   if (!SendGCSCommand(command.str()))
   {
      return false;
   }
   return CheckError (hasSAM_);
}

bool PIGCSCommands::JON (int joystick, int state)
{
   if (!hasJON_)
   {
      return false;
   }
   std::ostringstream command;
   command << "JON " << joystick << " " << state;
   if (!SendGCSCommand (command.str ()))
   {
      return false;
   }

   return CheckError (hasJON_);
};

bool PIGCSCommands::qJON (int joystick, int& state)
{
   if (!hasJON_)
   {
      return false;
   }
   std::ostringstream command;
   command << "JON? " << joystick;
   std::vector<std::string> answer;
   if (!SendGCSCommandAndReadAnswer (command.str (), answer, 1))
   {
      return false;
   }
   double value;
   if (!GetValue (answer[0], value))
   {
      return false;
   }
   state = (value > 0.9);
   return CheckError (hasJON_);
};


bool PIGCSCommands::qTPC (int& nrOutputChannels)
{
   if (!has_qTPC_)
   {
      return false;
   }
   std::ostringstream command;
   command << "TPC?";
   std::vector<std::string> answer;
   if (!SendGCSCommandAndReadAnswer (command.str (), answer, 1))
   {
      return false;
   }

   double value;
   if (!GetValue (answer[0], value))
   {
      return false;
   }
   nrOutputChannels = int (value + 0.1);
   return CheckError (has_qTPC_);
}

bool PIGCSCommands::ONL (const std::vector<int> outputChannels, const std::vector<int> values)
{
   size_t nrChannels = outputChannels.size ();

   if (nrChannels < 1)
   {
      return true;
   }

   std::ostringstream command;
   command << "ONL";

   size_t i = 0;
   for (; i < nrChannels; i++)
   {
      command << " " << outputChannels[i] << " " << values[i];
   }
   if (!SendGCSCommand (command.str ()))
   {
      return false;
   }
   return CheckError (hasONL_);
}

bool PIGCSCommands::qSTV (const std::string& axis, unsigned int& value)
{
   std::ostringstream command;
   command << "STV? " << axis;
   std::vector<std::string> answer;
   if (SendGCSCommandAndReadAnswer (command.str (), answer, 1))
   {
      unsigned long lValue = 0;
      if (GetValue (answer[0], lValue))
      {
         value = static_cast<unsigned int>(lValue);
         return true;
      }
   }
   return false;
}

bool PIGCSCommands::IsControllerReady (bool& ready)
{
   std::vector<std::string> answer;
   if (!SendGCSCommandAndReadAnswer (static_cast<unsigned char>(7), answer, 1))
   {
      return false;
   }
   ready = (static_cast<unsigned char>(answer[0][0]) == static_cast<unsigned char>(128 + '1'));
   return true;
}

bool PIGCSCommands::IsMoving (bool& moving)
{
   if (!hasIsMoving_)
   {
      return false;
   }
   std::vector<std::string> answer;
   if (!SendGCSCommandAndReadAnswer (static_cast<unsigned char>(5), answer, 1))
   {
      return CheckError (hasIsMoving_);
   }
   long value;
   if (!GetValue (answer[0], value))
   {
      return CheckError (hasIsMoving_);
   }
   moving = (value != 0);
   return CheckError (hasIsMoving_);
}

bool PIGCSCommands::HIN (const std::string& axis, bool state)
{
   if (!hasHIN_)
   {
      return false;
   }
   std::ostringstream command;
   command << "HIN " << axis << " " << (state ? 1 : 0);
   if (!SendGCSCommand (command.str ()))
   {
      return false;
   }

   return CheckError (hasHIN_);
};

bool PIGCSCommands::qHIN (const std::string& axis, bool& state)
{
   if (!hasHIN_)
   {
      return false;
   }
   std::ostringstream command;
   command << "HIN? " << axis;
   std::vector<std::string> answer;
   if (SendGCSCommandAndReadAnswer (command.str (), answer, 1))
   {
      long value;
      if (GetValue (answer[0], value))
      {
         state = (TRUE == value);
         return true;
      }
   }
   return CheckError (hasHIN_);
};


void PIGCSCommands::SetTimeout (int timeoutInMs)
{
   timeout_ = timeoutInMs;
}

int PIGCSCommands::GetTimeout ()
{
   return timeout_;
}

bool PIGCSCommands::HasqCSV ()
{
   return hasqCSV_;
}

bool PIGCSCommands::HasINI ()
{
   return hasINI_;
}

bool PIGCSCommands::HasCST ()
{
   return hasCST_;
}

bool PIGCSCommands::HasSVO ()
{
   return hasSVO_;
}

bool PIGCSCommands::HasEAX ()
{
   return hasEAX_ || alwaysUseEAX_;
}
bool PIGCSCommands::HasFRF ()
{
   return hasFRF_;
}
bool PIGCSCommands::HasREF ()
{
   return hasREF_;
}
bool PIGCSCommands::HasFNL ()
{
   return hasFNL_;
}
bool PIGCSCommands::HasMNL ()
{
   return hasMNL_;
}
bool PIGCSCommands::HasFPL ()
{
   return hasFPL_;
}
bool PIGCSCommands::HasMPL ()
{
   return hasMPL_;
}
bool PIGCSCommands::HasVEL ()
{
   return hasVEL_;
}
bool PIGCSCommands::HasHLT ()
{
   return hasHLT_;
}
bool PIGCSCommands::HasRES ()
{
   return hasRES_;
}
bool PIGCSCommands::HasSAM ()
{
   return hasSAM_;
}
bool PIGCSCommands::HasJON ()
{
   return hasJON_;
}
bool PIGCSCommands::Has_qTPC ()
{
   return has_qTPC_;
}
bool PIGCSCommands::HasONL ()
{
   return hasONL_;
}
bool PIGCSCommands::HasIsMoving ()
{
   return hasIsMoving_;
}
bool PIGCSCommands::HasHIN ()
{
   return hasHIN_;
}

int PIGCSCommands::GetTickCountInMs ()
{
#ifdef WIN32
   return static_cast<int>(::GetTickCount ());
#else
   struct timespec tm;
   clock_gettime (CLOCK_MONOTONIC, &tm);
   return static_cast<int>(tm.tv_sec * 1000 + tm.tv_nsec / 1000000);
#endif
}
