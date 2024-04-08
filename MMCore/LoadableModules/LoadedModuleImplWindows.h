// PROJECT:       Micro-Manager
// SUBSYSTEM:     MMCore
//
// DESCRIPTION:   Loadable module implementation for Windows.
//
// COPYRIGHT:     University of California, San Francisco, 2013,
//                All Rights reserved
//
// LICENSE:       This file is distributed under the "Lesser GPL" (LGPL) license.
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
// AUTHOR:        Mark Tsuchida,
//                based on parts of CPluginManager by Nenad Amodaj

#pragma once

#ifdef _WIN32

#include "LoadedModuleImpl.h"

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>


class LoadedModuleImplWindows: public LoadedModuleImpl
{
public:
   explicit LoadedModuleImplWindows(const std::string& filename);
   virtual void Unload();

   virtual void* GetFunction(const char* funcName);

private:
   HMODULE handle_;
};

#endif // _WIN32