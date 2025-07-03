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

#ifdef _WIN32
#  include "LoadedModuleImplWindows.h"
typedef LoadedModuleImplWindows PlatformLoadedModuleImpl;
#else
#  include "LoadedModuleImplUnix.h"
typedef LoadedModuleImplUnix PlatformLoadedModuleImpl;
#endif


LoadedModuleImpl*
LoadedModuleImpl::NewPlatformImpl(const std::string& filename)
{
   return new PlatformLoadedModuleImpl(filename);
}
