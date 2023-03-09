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
// CVS:           $Id: PIGCSCommandsDLL.h,v 1.19, 2019-01-09 10:45:26Z, Steffen Rau$
//

#ifndef PI_GCS_COMMANDS_DLL_H_INCLUDED
#define PI_GCS_COMMANDS_DLL_H_INCLUDED

#include "DeviceBase.h"
#include "PIGCSCommands.h"
#include <string>
#include <stdint.h>

class PIController;

class PIGCSCommandsDLL : public PIGCSCommands
{
public:
   PIGCSCommandsDLL ();
   ~PIGCSCommandsDLL ();

   virtual bool SendGCSCommand (const std::string& command);
   virtual bool SendGCSCommand (unsigned char singlebyte);
   virtual bool ReadGCSAnswer (std::vector<std::string>& answer, int nExpectedLines = -1);

   int LoadDLL (const std::string& dllName, PIController* controller);
   int ConnectInterface (const std::string& interfaceType, const std::string& interfaceParameter);
   void CloseAndUnload ();

protected:
   int GetSizeOfNextLine (int timeoutInMs);

private:
   typedef int (WINAPI* FP_GcsCommandset) (int32_t, const char*);
   typedef int (WINAPI* FP_GcsGetAnswer) (int32_t, char*, int32_t);
   typedef int (WINAPI* FP_GcsGetAnswerSize) (int32_t, int32_t*);
   typedef int (WINAPI* FP_ConnectRS232) (int32_t, int32_t);
   typedef int (WINAPI* FP_Connect) (int32_t);
   typedef int (WINAPI* FP_IsConnected) (int32_t);
   typedef int (WINAPI* FP_CloseConnection) (int32_t);
   typedef int (WINAPI* FP_EnumerateUSB) (char*, int32_t, const char*);
   typedef int (WINAPI* FP_ConnectUSB) (const char*);
   typedef int (WINAPI* FP_ConnectTCPIP) (const char*, int32_t);

   FP_GcsCommandset GcsCommandset_;
   FP_GcsGetAnswer GcsGetAnswer_;
   FP_GcsGetAnswerSize GcsGetAnswerSize_;
   FP_ConnectRS232 ConnectRS232_;
   FP_Connect Connect_;
   FP_IsConnected IsConnected_;
   FP_CloseConnection CloseConnection_;
   FP_EnumerateUSB EnumerateUSB_;
   FP_ConnectUSB ConnectUSB_;
   FP_ConnectTCPIP ConnectTCPIP_;


   void* LoadDLLFunc (const char* funcName);
   int ConnectPCI (const std::string& interfaceParameter);
   int ConnectRS232 (const std::string& interfaceParameter);
   int ConnectUSB (const std::string& interfaceParameter);
   int ConnectTCPIP (const std::string& interfaceParameter);
   std::string FindDeviceNameInUSBList (const char* szDevices, std::string interfaceParameter) const;

   std::string dllPrefix_;
   int ID_;

#ifdef WIN32
   HMODULE module_;
#else
   void* module_;
#endif

};


#endif // PI_GCS_COMMANDS_DLL_H_INCLUDED
