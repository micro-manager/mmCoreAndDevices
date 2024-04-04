// PROJECT:       Micro-Manager
// SUBSYSTEM:     MMCore
//
// DESCRIPTION:   Camera device instance wrapper
//
// COPYRIGHT:     University of California, San Francisco, 2014,
//                Nenad Amodaj 2024
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
// AUTHOR:        Nenad Amodaj

#include "StorageInstance.h"

int StorageInstance::Create(const char* path, const char* name, std::vector<int> shape, const char* meta, std::string& handle)
{
   RequireInitialized(__func__);

   char cHandle[MM::MaxStrLength];
   int ret = GetImpl()->Create(path, name, (int)shape.size(), &shape[0], meta, cHandle);
   if (ret != DEVICE_OK)
      return ret;

   return DEVICE_OK;
}
