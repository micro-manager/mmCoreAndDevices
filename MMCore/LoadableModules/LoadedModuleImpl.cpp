// PROJECT:       Micro-Manager
// SUBSYSTEM:     MMCore
//
// DESCRIPTION:   Abstract base class for platform-specific loadable module
//                implementation
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
// AUTHOR:        Mark Tsuchida

#include "LoadedModuleImpl.h"

#include "../Error.h"

namespace mmcore {
namespace internal {

#ifdef _WIN32
typedef LoadedModuleImplWindows PlatformLoadedModuleImpl;
#else
typedef LoadedModuleImplUnix PlatformLoadedModuleImpl;
#endif

LoadedModuleImpl*
LoadedModuleImpl::NewPlatformImpl(const std::string& filename)
{
   return new PlatformLoadedModuleImpl(filename);
}

} // namespace internal
} // namespace mmcore

#ifndef _WIN32

#include <dlfcn.h>

namespace mmcore {
namespace internal {

static void __attribute__((noreturn))
ThrowDlError()
{
   const char* errorText = dlerror();
   if (!errorText)
      errorText = "Operating system error message not available";
   throw CMMError(errorText);
}


LoadedModuleImplUnix::LoadedModuleImplUnix(const std::string& filename)
{
   int mode = RTLD_NOW | RTLD_LOCAL;

   // Hack to make Andor adapter on Linux work
   // TODO Check if this is still necessary, and if so, why. If it is
   // necessary, add a more generic 'enable-lazy' mechanism.
   if (filename.find("libmmgr_dal_Andor.so") != std::string::npos)
      mode = RTLD_LAZY | RTLD_LOCAL;

   handle_ = dlopen(filename.c_str(), mode);
   if (!handle_)
      ThrowDlError();
}


void
LoadedModuleImplUnix::Unload()
{
   if (!handle_)
      return;

   int err = dlclose(handle_);
   handle_ = 0;
   if (err)
      ThrowDlError();
}


void*
LoadedModuleImplUnix::GetFunction(const char* funcName)
{
   if (!handle_)
      throw CMMError("Cannot get function from unloaded module");

   void* proc = dlsym(handle_, funcName);
   if (!proc)
      ThrowDlError();
   return proc;
}

} // namespace internal
} // namespace mmcore

#endif // !defined(_WIN32)

#ifdef _WIN32

#include <string>

namespace mmcore {
namespace internal {

static void __declspec(noreturn)
ThrowLastError()
{
   std::string errorText;

   DWORD err = GetLastError();
   LPSTR pMsgBuf(0);
   if (FormatMessageA(
         FORMAT_MESSAGE_ALLOCATE_BUFFER |
         FORMAT_MESSAGE_FROM_SYSTEM |
         FORMAT_MESSAGE_IGNORE_INSERTS,
         NULL,
         err,
         MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
         (LPSTR)&pMsgBuf,
         0,
         NULL) && pMsgBuf)
   {
      errorText = pMsgBuf;

      // Windows error messages sometimes have trailing newlines
      const std::string whitespace(" \f\n\r\t\v");
      errorText.erase(errorText.find_last_not_of(whitespace) + 1);

      // This particular message can be rather misleading.
      if (errorText == "The specified module could not be found.") {
         errorText = "The module, or a module it depends upon, could not be found "
            "(Windows error: " + errorText + ")";
      }
   }
   if (pMsgBuf)
   {
      LocalFree(pMsgBuf);
   }

   if (errorText.empty()) {
      errorText = "Operating system error message not available";
   }

   throw CMMError(errorText);
}


LoadedModuleImplWindows::LoadedModuleImplWindows(const std::string& filename)
{
   int saveErrorMode = SetErrorMode(SEM_NOOPENFILEERRORBOX | SEM_FAILCRITICALERRORS);
   handle_ = LoadLibrary(filename.c_str());
   SetErrorMode(saveErrorMode);
   if (!handle_)
      ThrowLastError();
}


void
LoadedModuleImplWindows::Unload()
{
   if (!handle_)
      return;

   BOOL ok = FreeLibrary(handle_);
   handle_ = 0;
   if (!ok)
      ThrowLastError();
}


void*
LoadedModuleImplWindows::GetFunction(const char* funcName)
{
   if (!handle_)
      throw CMMError("Cannot get function from unloaded module");

   void* proc = GetProcAddress(handle_, funcName);
   if (!proc)
      ThrowLastError();
   return proc;
}

} // namespace internal
} // namespace mmcore

#endif // _WIN32
