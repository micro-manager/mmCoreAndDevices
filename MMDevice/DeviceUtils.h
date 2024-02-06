///////////////////////////////////////////////////////////////////////////////
// FILE:          DeviceUtils.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     MMDevice - Device adapter kit
//-----------------------------------------------------------------------------
// DESCRIPTION:   Class with utility methods for building device adapters
// AUTHOR:        Nenad Amodaj, nenad@amodaj.com 06/08/2005
// COPYRIGHT:     University of California, San Francisco, 2006
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
//

#pragma once

#include "MMDeviceConstants.h"
#include <vector>
#include <string>

class CDeviceUtils
{
public:
   static bool CopyLimitedString(char* pszTarget, const char* pszSource);
   static unsigned GetMaxStringLength();
   static const char* ConvertToString(long lnVal);
   static const char* ConvertToString(double dVal);
   static const char* ConvertToString(int val);
   static const char* ConvertToString(bool val);
   static void Tokenize(const std::string& str, std::vector<std::string>& tokens, const std::string& delimiters = ",");
   static void SleepMs(long ms);
   static void NapMicros(unsigned long microsecs);
   static std::string HexRep(std::vector<unsigned char>  );
private:
   static char m_pszBuffer[MM::MaxStrLength];
};
