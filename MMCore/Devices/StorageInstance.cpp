// PROJECT:       Micro-Manager
// SUBSYSTEM:     MMCore
//
// DESCRIPTION:   Camera device instance wrapper
//
// COPYRIGHT:     Nenad Amodaj 2024
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

int StorageInstance::Create(const char* path, const char* name, const std::vector<int>& shape, const char* meta, std::string& handle)
{
   RequireInitialized(__func__);

   char cHandle[MM::MaxStrLength];
   int ret = GetImpl()->Create(path, name, (int)shape.size(), &shape[0], meta, cHandle);
   if (ret == DEVICE_OK)
      handle = cHandle;

   return DEVICE_OK;
}

int StorageInstance::ConfigureDimension(const char* handle, int dimension, const char* name, const char* meaning)
{
   RequireInitialized(__func__);

   return GetImpl()->ConfigureDimension(handle, dimension, name, meaning);
}

int StorageInstance::ConfigureCoordinate(const char* handle, int dimension, int coordinate, const char* name)
{
   RequireInitialized(__func__);

   return GetImpl()->ConfigureCoordinate(handle, dimension, coordinate, name);
}

int StorageInstance::Close(const char* handle)
{
   RequireInitialized(__func__);

   return GetImpl()->Close(handle);
}

int StorageInstance::Load(const char* path, const char* name, std::string& handle)
{
   RequireInitialized(__func__);

   char cHandle[MM::MaxStrLength];
   int ret = GetImpl()->Load(path, name, cHandle);
   if (ret == DEVICE_OK)
      handle = cHandle;

   return ret;
}

int StorageInstance::Delete(char* handle)
{
   RequireInitialized(__func__);

   return GetImpl()->Delete(handle);
}

int StorageInstance::List(const char* path, std::vector<std::string>& listOfDatasets)
{
   RequireInitialized(__func__);
   const int maxItems(5000);
   const int maxItemLength(1024);
   std::vector<char*> cList(maxItems, nullptr);
   for (auto c : cList)
   {
      c = new char[maxItemLength];
      memset(c, 0, maxItemLength);
   }
   int ret = GetImpl()->List(path, &cList[0], maxItems, maxItemLength);
   if (ret == DEVICE_OK)
   {
      listOfDatasets.clear();

      for (auto c : cList)
      {
         if (strlen(c) == 0) break;
         listOfDatasets.push_back(std::string(c));
      }
   }

   for (auto c : cList) delete[] c;

   return ret;
}

int StorageInstance::AddImage(const char* handle, unsigned char* pixels, int width, int height, int depth, std::vector<int>& coordinates, const char* imageMeta)
{
   RequireInitialized(__func__);
   return GetImpl()->AddImage(handle, pixels, width, height, depth, &coordinates[0], coordinates.size(), imageMeta);
}

int StorageInstance::GetSummaryMeta(const char* handle, char* meta)
{
   return 0;
}

int StorageInstance::GetImageMeta(const char* handle, const std::vector<int>& coordinates, char* meta)
{
   return 0;
}

const unsigned char* StorageInstance::GetImage(const char* handle, const std::vector<int>& coordinates)
{
   return nullptr;
}

int StorageInstance::GetNumberOfDimensions(const char* handle)
{
   return 0;
}

int StorageInstance::GetDimension(const char* handle, int dimension, std::string& name, std::string& meaning)
{
   return 0;
}

int StorageInstance::GetCoordinate(const char* handle, int dimension, int coordinate, std::string& name)
{
   return 0;
}
