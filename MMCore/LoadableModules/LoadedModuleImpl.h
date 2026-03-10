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

#pragma once

#include "LoadedModule.h"

#include <memory>

namespace mmcore {
namespace internal {

class LoadedModuleImpl
{
public:
   LoadedModuleImpl(const LoadedModuleImpl&) = delete;
   LoadedModuleImpl& operator=(const LoadedModuleImpl&) = delete;

   static std::unique_ptr<LoadedModuleImpl> NewPlatformImpl(const std::string& filename);

   virtual ~LoadedModuleImpl() {}

   virtual void Unload() = 0;
   virtual void* GetFunction(const char* funcName) = 0;

protected:
   LoadedModuleImpl() {}
};

} // namespace internal
} // namespace mmcore

#ifndef _WIN32

namespace mmcore {
namespace internal {

class LoadedModuleImplUnix : public LoadedModuleImpl
{
public:
   explicit LoadedModuleImplUnix(const std::string& filename);
   virtual void Unload();

   virtual void* GetFunction(const char* funcName);

private:
   void* handle_;
};

} // namespace internal
} // namespace mmcore

#endif // !defined(_WIN32)

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

namespace mmcore {
namespace internal {

class LoadedModuleImplWindows: public LoadedModuleImpl
{
public:
   explicit LoadedModuleImplWindows(const std::string& filename);
   virtual void Unload();

   virtual void* GetFunction(const char* funcName);

private:
   HMODULE handle_;
};

} // namespace internal
} // namespace mmcore

#endif // _WIN32
