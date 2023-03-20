///////////////////////////////////////////////////////////////////////////////
// FILE:          PIGCSCommandsDLL.cpp
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
// CVS:           $Id: PIGCSCommandsDLL.cpp,v 1.25, 2019-01-09 10:45:26Z, Steffen Rau$
//

#ifndef __APPLE__

#ifndef WIN32
#include <dlfcn.h>
#include <time.h>
#endif

#include "PIController.h"
#include "PI_GCS_2.h"
#include "PIGCSCommandsDLL.h"

#include <locale>
#include <algorithm>

struct ToUpper
{
   ToUpper (std::locale const& l) : loc (l)
   {
      ;
   }
   char operator() (char c) const
   {
      return std::toupper (c, loc);
   }
private:
   std::locale const loc;
   ToUpper& operator=(const ToUpper&);
   ToUpper ();
};

PIGCSCommandsDLL::PIGCSCommandsDLL ()
   : GcsCommandset_ (NULL)
   , GcsGetAnswer_ (NULL)
   , GcsGetAnswerSize_ (NULL)
   , ConnectRS232_ (NULL)
   , Connect_ (NULL)
   , IsConnected_ (NULL)
   , CloseConnection_ (NULL)
   , EnumerateUSB_ (NULL)
   , ConnectUSB_ (NULL)
   , ConnectTCPIP_ (NULL)
   , ID_ (-1)
   , module_ (NULL)
{
}

PIGCSCommandsDLL::~PIGCSCommandsDLL ()
{
   CloseAndUnload ();
}



int PIGCSCommandsDLL::LoadDLL (const std::string& dllName, PIController* controller)
{
   if (ci_find (dllName, "C843_GCS_DLL") != std::string::npos)
   {
      dllPrefix_ = "C843_";
      controller->SetGCS2 (false);
   }
   else if (ci_find (dllName, "PI_Mercury_GCS") != std::string::npos)
   {
      dllPrefix_ = "Mercury_";
      controller->SetGCS2 (false);
   }
   else if (ci_find (dllName, "E7XX_GCS_DLL") != std::string::npos)
   {
      dllPrefix_ = "E7XX_";
      controller->SetGCS2 (false);
      controller->SetUmToDefaultUnit (1.0);
      controller->SetNeedResetStages (true);
      controller->SetOnlyIDSTAGEvalid (true);
   }
   else if (ci_find (dllName, "E816_DLL") != std::string::npos)
   {
      dllPrefix_ = "E816_";
      controller->SetUmToDefaultUnit (1.0);
   }
   else if (ci_find (dllName, "PI_HydraPollux_GCS2_DLL") != std::string::npos)
   {
      dllPrefix_ = "Hydra_";
   }
   else if (ci_find (dllName, "C865_GCS_DLL") != std::string::npos)
   {
      dllPrefix_ = "C865_";
      controller->SetGCS2 (false);
   }
   else if (ci_find (dllName, "C866_GCS_DLL") != std::string::npos)
   {
      dllPrefix_ = "C866_";
      controller->SetGCS2 (false);
   }
   else if (ci_find (dllName, "PI_GCS2_DLL") != std::string::npos)
   {
      dllPrefix_ = "PI_";
      controller->SetGCS2 (true);
   }

#ifdef WIN32
   module_ = LoadLibrary (dllName.c_str ());
#else
   module_ = dlopen (dllName.c_str (), RTLD_LAZY);
#endif
   if (module_ == NULL)
   {
      printf ("load module failed\n");
      return ERR_DLL_PI_DLL_NOT_FOUND;
   }
   GcsCommandset_ = reinterpret_cast<FP_GcsCommandset>(LoadDLLFunc ("GcsCommandset"));
   GcsGetAnswer_ = reinterpret_cast<FP_GcsGetAnswer>(LoadDLLFunc ("GcsGetAnswer"));
   GcsGetAnswerSize_ = reinterpret_cast<FP_GcsGetAnswerSize>(LoadDLLFunc ("GcsGetAnswerSize"));
   ConnectRS232_ = reinterpret_cast<FP_ConnectRS232>(LoadDLLFunc ("ConnectRS232"));
   Connect_ = reinterpret_cast<FP_Connect>(LoadDLLFunc ("Connect"));
   IsConnected_ = reinterpret_cast<FP_IsConnected>(LoadDLLFunc ("IsConnected"));
   CloseConnection_ = reinterpret_cast<FP_CloseConnection>(LoadDLLFunc ("CloseConnection"));
   EnumerateUSB_ = reinterpret_cast<FP_EnumerateUSB>(LoadDLLFunc ("EnumerateUSB"));
   ConnectUSB_ = reinterpret_cast<FP_ConnectUSB>(LoadDLLFunc ("ConnectUSB"));
   ConnectTCPIP_ = reinterpret_cast<FP_ConnectTCPIP>(LoadDLLFunc ("ConnectTCPIP"));

   return DEVICE_OK;
}

void* PIGCSCommandsDLL::LoadDLLFunc (const char* funcName)
{
#ifdef WIN32
   return GetProcAddress (module_, (dllPrefix_ + funcName).c_str ());
#else
   return(dlsym (module_, (dllPrefix_ + funcName).c_str ()));
#endif
}

void PIGCSCommandsDLL::CloseAndUnload ()
{
   if (module_ == NULL)
   {
      return;
   }

   if (ID_ >= 0 && CloseConnection_ != NULL)
   {
      CloseConnection_ (ID_);
   }

   ID_ = -1;

   GcsCommandset_ = NULL;
   GcsGetAnswer_ = NULL;
   GcsGetAnswerSize_ = NULL;
   ConnectRS232_ = NULL;
   Connect_ = NULL;
   IsConnected_ = NULL;
   CloseConnection_ = NULL;
   EnumerateUSB_ = NULL;
   ConnectUSB_ = NULL;
#ifdef WIN32
   FreeLibrary (module_);
   module_ = NULL;
#else
#endif
}

int PIGCSCommandsDLL::ConnectInterface (const std::string& interfaceType, const std::string& interfaceParameter)
{
#ifdef WIN32
   if (module_ == NULL)
   {
      return DEVICE_NOT_CONNECTED;
   }
#else
#endif

   int ret = ERR_DLL_PI_INVALID_INTERFACE_NAME;
   if (interfaceType == "PCI")
   {
      ret = ConnectPCI (interfaceParameter);
   }
   if (interfaceType == "RS-232")
   {
      ret = ConnectRS232 (interfaceParameter);
   }
   if (interfaceType == "USB")
   {
      ret = ConnectUSB (interfaceParameter);
   }
   if (interfaceType == "TCP/IP")
   {
      ret = ConnectTCPIP (interfaceParameter);
   }

   if (ret != DEVICE_OK)
   {
      return ret;
   }

   return ret;
}

int PIGCSCommandsDLL::ConnectPCI (const std::string& interfaceParameter)
{
   if (Connect_ == NULL)
   {
      return DEVICE_NOT_SUPPORTED;
   }

   long board;
   if (!GetValue (interfaceParameter, board))
   {
      return ERR_DLL_PI_INVALID_INTERFACE_PARAMETER;
   }
   ID_ = Connect_ (static_cast<int>(board));
   if (ID_ < 0)
   {
      return DEVICE_NOT_CONNECTED;
   }
   return DEVICE_OK;
}

int PIGCSCommandsDLL::ConnectRS232 (const std::string& interfaceParameter)
{
   if (ConnectRS232_ == NULL)
   {
      return DEVICE_NOT_SUPPORTED;
   }

   size_t pos = interfaceParameter.find (';');
   if (pos == std::string::npos)
   {
      return ERR_DLL_PI_INVALID_INTERFACE_PARAMETER;
   }
   std::string sport = interfaceParameter.substr (0, pos);
   std::string sbaud = interfaceParameter.substr (pos + 1);

   long port, baud;
   if (!GetValue (sport, port))
   {
      return DEVICE_INVALID_PROPERTY_VALUE;
   }
   if (!GetValue (sbaud, baud))
   {
      return DEVICE_INVALID_PROPERTY_VALUE;
   }

   ID_ = ConnectRS232_ (static_cast<int>(port), static_cast<int>(baud));
   if (ID_ < 0)
   {
      return DEVICE_NOT_CONNECTED;
   }

   return DEVICE_OK;
}



int PIGCSCommandsDLL::ConnectUSB (const std::string& interfaceParameter)
{
   if (ConnectUSB_ == NULL || EnumerateUSB_ == NULL)
   {
      return DEVICE_NOT_SUPPORTED;
   }

   char szDevices[128 * 80 + 1];
   int nrDevices = EnumerateUSB_ (szDevices, 128 * 80, NULL);
   if (nrDevices < 0)
   {
      return TranslateError (nrDevices);
   }
   if (nrDevices == 0)
   {
      return DEVICE_NOT_CONNECTED;
   }

   std::string deviceName;
   if (interfaceParameter.empty ())
   {
      if (nrDevices != 1)
      {
         return ERR_DLL_PI_INVALID_INTERFACE_PARAMETER;
      }
      deviceName = szDevices;
   }
   else
   {
      deviceName = FindDeviceNameInUSBList (szDevices, interfaceParameter);
   }
   if (deviceName.empty ())
   {
      return DEVICE_NOT_CONNECTED;
   }

   ID_ = ConnectUSB_ (deviceName.c_str ());
   if (ID_ < 0)
   {
      return DEVICE_NOT_CONNECTED;
   }

   return DEVICE_OK;
}

std::string PIGCSCommandsDLL::FindDeviceNameInUSBList (const char* szDevices, std::string interfaceParameter) const
{
   std::string sDevices (szDevices);
   static ToUpper up (std::locale::classic ());
   std::transform (interfaceParameter.begin (), interfaceParameter.end (), interfaceParameter.begin (), up);

   std::vector<std::string> lines = Tokenize (sDevices);
   std::vector<std::string>::iterator line;
   for (line = lines.begin (); line != lines.end (); ++line)
   {
      std::string LINE (*line);
      std::transform (LINE.begin (), LINE.end (), LINE.begin (), up);

      if (LINE.find (interfaceParameter) != std::string::npos)
      {
         return (*line);
      }
   }
   return "";
}

int PIGCSCommandsDLL::ConnectTCPIP (const std::string& interfaceParameter)
{
   if (ConnectTCPIP_ == NULL)
   {
      return DEVICE_NOT_SUPPORTED;
   }

   size_t pos = interfaceParameter.find (':');
   if (pos == std::string::npos)
   {
      return ERR_DLL_PI_INVALID_INTERFACE_PARAMETER;
   }
   std::string ipaddr = interfaceParameter.substr (0, pos);
   std::string sport = interfaceParameter.substr (pos + 1);

   long port;
   if (!GetValue (sport, port))
   {
      return DEVICE_INVALID_PROPERTY_VALUE;
   }

   ID_ = ConnectTCPIP_ (ipaddr.c_str (), port);
   if (ID_ < 0)
   {
      return DEVICE_NOT_CONNECTED;
   }

   return DEVICE_OK;
}

bool PIGCSCommandsDLL::SendGCSCommand (const std::string& command)
{
   if (!GcsCommandset_)
   {
      return false;
   }
   if (GcsCommandset_ (ID_, command.c_str ()) == FALSE)
   {
      return false;
   }
   return true;
}

bool PIGCSCommandsDLL::SendGCSCommand (unsigned char singlebyte)
{
   if (!GcsCommandset_)
   {
      return false;
   }
   const char cmd[2] = { static_cast<char>(singlebyte), '\0' };
   if (GcsCommandset_ (ID_, cmd) == FALSE)
   {
      return false;
   }
   return true;

}

bool PIGCSCommandsDLL::ReadGCSAnswer (std::vector<std::string>& answer, int nExpectedLines)
{
   if (!GcsGetAnswer_ || !GcsGetAnswerSize_ )
   {
      return false;
   }
   answer.clear ();
   for (;;)
   {
      int size = GetSizeOfNextLine (timeout_);
      if (size < 0)
      {
         return false;
      }
      char* buffer = new char [size + 2];
      if (!GcsGetAnswer_ (ID_, buffer, size+1))
      {
         delete[] buffer;
         return false;
      }
      std::string line (buffer);
      delete[] buffer;
      answer.push_back (line);
      if (line.length ()>1 && (line.substr (line.length ()-2) == " \n"))
      {
         continue;
      }
      if (nExpectedLines >= 0)
      {
         return (nExpectedLines == static_cast<int>(answer.size ()));
      }
      return true;
   }
}

int PIGCSCommandsDLL::GetSizeOfNextLine (int timeoutInMs)
{
   int32_t size;
   int start = GetTickCountInMs ();
   while ((GetTickCountInMs () - start) < timeoutInMs)
   {
      if (!GcsGetAnswerSize_ (ID_, &size))
      {
         return -1;
      }
      if (size > 0)
      {
         return static_cast<int>(size);
      }
   }
   return -1;
}

#endif
