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

int StorageInstance::Create(const char* path, const char* name, const std::vector<int>& shape, MM::StorageDataType pixType, const char* meta, std::string& handle)
{
   RequireInitialized(__func__);

   char cHandle[MM::MaxStrLength];
	memset(cHandle, 0, MM::MaxStrLength);
   int ret = GetImpl()->Create(path, name, (int)shape.size(), &shape[0], pixType, meta, cHandle);
   if (ret != DEVICE_OK)
      return ret;
   
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

int StorageInstance::Load(const char* path, std::string& handle)
{
   RequireInitialized(__func__);

   char cHandle[MM::MaxStrLength];
	memset(cHandle, 0, MM::MaxStrLength);
   int ret = GetImpl()->Load(path, cHandle);
   if (ret != DEVICE_OK)
      return ret;
   
   handle = cHandle;
   return DEVICE_OK;
}

int StorageInstance::GetShape(const char* handle, std::vector<long>& shape)
{
   RequireInitialized(__func__);
   int numDim(0);
   int ret = GetImpl()->GetNumberOfDimensions(handle, numDim);
   if (ret != DEVICE_OK)
      return ret;
   int* shapeArray = new int[numDim];
   ret = GetImpl()->GetShape(handle, shapeArray);
   if (ret != DEVICE_OK)
      return ret;
   shape.clear();
   for (int i = 0; i < numDim; i++)
      shape.push_back(shapeArray[i]);

   return DEVICE_OK;
}

int StorageInstance::GetPixelType(const char* handle, MM::StorageDataType& dataType)
{
   RequireInitialized(__func__);
   int ret = GetImpl()->GetDataType(handle, dataType);
   if (ret != DEVICE_OK)
      return ret;

   return DEVICE_OK;
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

int StorageInstance::AddImage(const char* handle, int sizeInBytes, unsigned char* pixels, std::vector<int>& coordinates, const char* imageMeta)
{
   RequireInitialized(__func__);
   return GetImpl()->AddImage(handle, sizeInBytes, pixels, &coordinates[0], coordinates.size(), imageMeta);
}

int StorageInstance::GetSummaryMeta(const char* handle, std::string& meta)
{
	char cMeta[MM::MaxMetadataLength];
	memset(cMeta, 0, MM::MaxMetadataLength);
   RequireInitialized(__func__);
   int ret = GetImpl()->GetSummaryMeta(handle, cMeta, MM::MaxMetadataLength);
   meta = cMeta;
   return ret;
}

int StorageInstance::GetImageMeta(const char* handle, const std::vector<int>& coordinates, std::string& meta)
{
	char cMeta[MM::MaxMetadataLength];
	memset(cMeta, 0, MM::MaxMetadataLength);
   RequireInitialized(__func__);
   int ret = GetImpl()->GetImageMeta(handle, const_cast<int*>(&coordinates[0]), coordinates.size(), cMeta, MM::MaxMetadataLength);
   meta = cMeta;
   return ret;
}

const unsigned char* StorageInstance::GetImage(const char* handle, const std::vector<int>& coordinates)
{
   RequireInitialized(__func__);
   return GetImpl()->GetImage(handle, const_cast<int*>(&coordinates[0]), coordinates.size());
}

int StorageInstance::GetNumberOfDimensions(const char* handle, int& numDim)
{
   RequireInitialized(__func__);
   return GetImpl()->GetNumberOfDimensions(handle, numDim);
}

int StorageInstance::GetDimension(const char* handle, int dimension, std::string& name, std::string& meaning)
{
	char nameStr[MM::MaxStrLength];
	char meaningStr[MM::MaxStrLength];
	memset(nameStr, 0, MM::MaxStrLength);
	memset(meaningStr, 0, MM::MaxStrLength);
   int ret = GetImpl()->GetDimension(handle, dimension, nameStr, MM::MaxStrLength, meaningStr, MM::MaxStrLength);
   name = nameStr;
   meaning = meaningStr;
   return ret;
}

int StorageInstance::GetCoordinate(const char* handle, int dimension, int coordinate, std::string& name)
{
	char cName[MM::MaxStrLength];
	memset(cName, 0, MM::MaxStrLength);
   int ret = GetImpl()->GetCoordinate(handle, dimension, coordinate, cName, MM::MaxStrLength);
   name = cName;
   return ret;
}

int StorageInstance::GetImageCount(const char* handle, int& imgcount)
{
	RequireInitialized(__func__);
	return GetImpl()->GetImageCount(handle, imgcount);
}

int StorageInstance::GetPath(const char* handle, std::string& path)
{
	char cPath[MM::MaxStrLength];
	memset(cPath, 0, MM::MaxStrLength);
	int ret = GetImpl()->GetPath(handle, cPath, MM::MaxStrLength);
	path = cPath;
	return ret;
}

bool StorageInstance::IsOpen(const char* handle)
{
	RequireInitialized(__func__);
	return GetImpl()->IsOpen(handle);
}

bool StorageInstance::IsReadOnly(const char* handle)
{
	RequireInitialized(__func__);
	return GetImpl()->IsReadOnly(handle);
}
