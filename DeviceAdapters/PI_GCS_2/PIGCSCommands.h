///////////////////////////////////////////////////////////////////////////////
// FILE:          PIGCSCommandsDLL.h
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

#ifndef PI_GCS_COMMANDS_H_INCLUDED
#define PI_GCS_COMMANDS_H_INCLUDED

#include <string>
#include <vector>

#ifndef WIN32
#define BOOL int
#define TRUE 1
#define FALSE 0
#endif


class PIGCSCommands
{
public:
   PIGCSCommands ();
   virtual ~PIGCSCommands ();

   virtual bool SendGCSCommand (const std::string& command) = 0;
   virtual bool SendGCSCommand (unsigned char singlebyte) = 0;
   virtual bool ReadGCSAnswer (std::vector<std::string>& answer, int nExpectedLines = -1) = 0;

   virtual bool SendGCSCommandAndReadAnswer (const std::string& command, std::vector<std::string>& answer, int nExpectedLines = -1);
   virtual bool SendGCSCommandAndReadAnswer (unsigned char singleByte, std::vector<std::string>& answer, int nExpectedLines = -1);

   virtual int GetError ();
   virtual int GetTranslatedError ();

   virtual void SetTimeout (int timeoutInMs);
   virtual int GetTimeout ();

   void SetAlwaysUseEAX (bool alwaysUseEAX);
   void SetGCS2 (bool gcs2);
   bool GetErrorCheckAfterMOV () const;
   void SetErrorCheckAfterMOV (bool errorCheck);

   virtual bool qCSV (double& syntaxVersion);
   virtual bool qIDN (std::string&);
   virtual bool INI (const std::string&);
   virtual bool CST (const std::string&, const std::string&);
   virtual bool SVO (const std::string&, bool svo);
   virtual bool qSVO (const std::string&, bool& svo);
   virtual bool EAX (const std::string&, bool eax);
   virtual bool qEAX (const std::string&, bool& eax);
   virtual bool FRF (const std::string& axes);
   virtual bool REF (const std::string& axes);
   virtual bool MNL (const std::string& axes);
   virtual bool FNL (const std::string& axes);
   virtual bool FPL (const std::string& axes);
   virtual bool MPL (const std::string& axes);
   virtual bool MOV (const std::string& axis, double target);
   virtual bool MOV (const std::string& axis1, double target1, const std::string& axis2, double target2);
   virtual bool qPOS (const std::string& axis, double* position);
   virtual bool qPOS (const std::string& axis1, double* position1, const std::string& axis2, double* position2);
   virtual bool SPV (const std::string& memory,
                     const std::string& containerUnit,
                     const std::string& functionUnit,
                     const std::string& parameter,
                     double value);
   virtual bool qSPV (const std::string& memory,
                      const std::string& containerUnit,
                      const std::string& functionUnit,
                      const std::string& parameter,
                      double& value);
   virtual bool VEL (const std::string& axis, double velocity);
   virtual bool qVEL (const std::string& axis, double& velocity);
   virtual bool STP ();
   virtual bool HLT (const std::string& axis);
   virtual bool RES (const std::string& axis);
   virtual bool SAM (const std::string& axis, unsigned int mode);
   virtual bool JON (int joystick, int state);
   virtual bool qJON (int joystick, int& state);
   virtual bool qTPC (int& nrOutputChannels);
   virtual bool ONL (std::vector<int> outputChannels, std::vector<int> values);
   virtual bool qSTV (const std::string& axis, unsigned int& value);
   virtual bool IsControllerReady (bool& ready);
   virtual bool IsMoving (bool& moving);
   virtual bool HIN (const std::string& axis, bool state);
   virtual bool qHIN (const std::string& axis, bool& state);

   virtual bool HasqCSV ();
   virtual bool HasINI ();
   virtual bool HasCST ();
   virtual bool HasSVO ();
   virtual bool HasEAX ();
   virtual bool HasFRF ();
   virtual bool HasREF ();
   virtual bool HasFNL ();
   virtual bool HasMNL ();
   virtual bool HasFPL ();
   virtual bool HasMPL ();
   virtual bool HasVEL ();
   virtual bool HasHLT ();
   virtual bool HasRES ();
   virtual bool HasSAM ();
   virtual bool HasJON ();
   virtual bool Has_qTPC ();
   virtual bool HasONL ();
   virtual bool HasIsMoving ();
   virtual bool HasHIN ();

protected:
   bool CheckError (bool& hasCmdFlag);
   bool CheckError ();

   std::string ConvertToAxesStringWithSpaces (const std::string& axes) const;
   int GetTickCountInMs ();

   int controllerError_;
   bool alwaysUseEAX_;
   bool gcs2_;
   bool errorCheckAfterMOV_;
   int timeout_;

   bool hasqCSV_;
   bool hasINI_;
   bool hasCST_;
   bool hasSVO_;
   bool hasEAX_;
   bool hasFRF_;
   bool hasREF_;
   bool hasFNL_;
   bool hasMNL_;
   bool hasFPL_;
   bool hasMPL_;
   bool hasVEL_;
   bool hasHLT_;
   bool hasRES_;
   bool hasSAM_;
   bool hasJON_;
   bool has_qTPC_;
   bool hasONL_;
   bool hasIsMoving_;
   bool hasHIN_;
};



#endif //PI_GCS_COMMANDS_H_INCLUDED
