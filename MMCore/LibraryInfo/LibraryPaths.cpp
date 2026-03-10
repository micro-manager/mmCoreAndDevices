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

#include "LibraryPaths.h"

#if defined(__APPLE__) || defined(__linux__)

#include "../Error.h"

#if defined(__linux__) && !defined(_GNU_SOURCE)
// Provide dladdr()
#   define _GNU_SOURCE
#endif

#include <dlfcn.h>

#ifdef __linux__
#   include <libgen.h> // for basename()
#   include <unistd.h> // for readlink()
#endif

#include <algorithm>
#include <string>
#include <vector>


#ifdef __linux__

static std::string GetExecutablePath()
{
   std::vector<char> path;
   for (size_t bufsize = 1024; bufsize <= 32768; bufsize *= 2)
   {
      path.resize(bufsize);
      size_t len = readlink("/proc/self/exe", path.data(), bufsize);
      if (!len)
         throw CMMError("Cannot get path to executable");
      if (len >= bufsize)
         continue;
      return path.data();
   }
   throw CMMError("Path to executable too long");
}

static std::string GetExecutableName()
{
   const std::string path = GetExecutablePath();
   // basename() can modify the buffer, so make a copy
   std::vector<char> mutablePath(path.size() + 1);
   std::copy(path.begin(), path.end(), mutablePath.begin());
   const char* name = basename(mutablePath.data());
   if (!name)
      throw CMMError("Cannot get executable name");
   return name;
}

#endif // __linux__


namespace mmcore {
namespace internal {

// Note: This can return a relative path on Linux. On OS X, an absolute (though
// not necessarily normalized) path is returned. This should not normally be a
// problem.
std::string GetPathOfThisModule()
{
   // This function is located in this module (obviously), so get info on the
   // dynamic library containing the address of this function.
   Dl_info info;
   int ok = dladdr(reinterpret_cast<void*>(&GetPathOfThisModule), &info);
   if (!ok || !info.dli_fname)
      throw CMMError("Cannot get path to library or executable");

   const std::string path(info.dli_fname);

#ifdef __linux__
   // On Linux, the filename returned by dladdr() may not be a path if we are
   // statically linked into the executable (it appears that the equivalent of
   // argv[0] is returned). In that case, obtain the correct executable path.
   if (path.find('/') == std::string::npos) // not a path
   {
      if (path == GetExecutableName())
         return GetExecutablePath();
   }
#endif // __linux__

   return path;
}

} // namespace internal
} // namespace mmcore

#endif // __APPLE__ || __linux__

#ifdef _WIN32

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


namespace mmcore {
namespace internal {

std::string GetPathOfThisModule()
{
   return GetPathOfModule(GetHandleOfThisModule());
}

} // namespace internal
} // namespace mmcore

#endif // _WIN32
