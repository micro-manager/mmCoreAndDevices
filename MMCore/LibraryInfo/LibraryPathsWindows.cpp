// PROJECT:       Micro-Manager
// SUBSYSTEM:     MMCore
//
// COPYRIGHT:     University of California, San Francisco, 2014,
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
// AUTHOR:        Mark Tsuchida

// Note: This code must reside in the same binary image as the rest of the
// Core.

#ifdef _WIN32 // whole file

#include "LibraryPaths.h"


#include "../CoreUtils.h"
#include "../Error.h"

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

#include <string>
#include <vector>


static HMODULE GetHandleOfThisModule()
{
   // This function is located in this module (obviously), so get the module
   // containing the address of this function.
   HMODULE hModule;
   BOOL ok = GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
         GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
         reinterpret_cast<LPCSTR>(&GetHandleOfThisModule),
         &hModule);
   if (!ok) {
      DWORD err = GetLastError();
      // TODO FormatMessage()
      throw CMMError("Cannot get handle to DLL or executable "
            "(Windows system error code " + ToString(err) + ")");
   }

   return hModule;
}


static std::string GetPathOfModule(HMODULE hModule)
{
   for (size_t bufsize = 1024, len = bufsize; ; bufsize += 1024)
   {
      std::vector<char> filename(bufsize);
      len = GetModuleFileNameA(hModule, filename.data(),
            static_cast<DWORD>(bufsize));
      if (!len)
      {
         DWORD err = GetLastError();
         // TODO FormatMessage()
         throw CMMError("Cannot get filename of DLL or executable "
               "(Windows system error code " + ToString(err) + ")");
      }

      if (len == bufsize) // Filename may not have fit in buffer
         continue;

      return filename.data();
   }
}


namespace MMCorePrivate {

std::string GetPathOfThisModule()
{
   return GetPathOfModule(GetHandleOfThisModule());
}

} // namespace MMCorePrivate

#endif // _WIN32
