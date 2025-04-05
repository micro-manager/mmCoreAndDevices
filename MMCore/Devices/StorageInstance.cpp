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

int StorageInstance::Create(const char* path, const char* name, const std::vector<int>& shape, MM::StorageDataType pixType, const char* meta, int metaLength, int& handle)
{
   RequireInitialized(__func__);

   int ret = GetImpl()->Create(path, name, (int)shape.size(), &shape[0], pixType, meta, metaLength, &handle);
   if (ret != DEVICE_OK)
      return ret;
   
   return DEVICE_OK;
}

int StorageInstance::ConfigureDimension(int handle, int dimension, const char* name, const char* meaning)
{
   RequireInitialized(__func__);

   return GetImpl()->ConfigureDimension(handle, dimension, name, meaning);
}

int StorageInstance::ConfigureCoordinate(int handle, int dimension, int coordinate, const char* name)
{
   RequireInitialized(__func__);

   return GetImpl()->ConfigureCoordinate(handle, dimension, coordinate, name);
}

int StorageInstance::Close(int handle)
{
   RequireInitialized(__func__);

   return GetImpl()->Close(handle);
}

int StorageInstance::Load(const char* path, int& handle)
{
   RequireInitialized(__func__);

   int ret = GetImpl()->Load(path, &handle);
   if (ret != DEVICE_OK)
      return ret;
   
   return DEVICE_OK;
}

int StorageInstance::GetShape(int handle, std::vector<long>& shape)
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

int StorageInstance::GetPixelType(int handle, MM::StorageDataType& dataType)
{
   RequireInitialized(__func__);
   int ret = GetImpl()->GetDataType(handle, dataType);
   if (ret != DEVICE_OK)
      return ret;

   return DEVICE_OK;
}

int StorageInstance::Delete(int handle)
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

int StorageInstance::AddImage(int handle, int sizeInBytes, unsigned char* pixels, std::vector<int>& coordinates, const char* imageMeta, int metaLength)
{
   RequireInitialized(__func__);
   return GetImpl()->AddImage(handle, sizeInBytes, pixels, &coordinates[0], (int)coordinates.size(), imageMeta, metaLength);
}

int StorageInstance::AppendImage(int handle, int sizeInBytes, unsigned char* pixels, const char* imageMeta, int metaLength)
{
   RequireInitialized(__func__);
   return GetImpl()->AppendImage(handle, sizeInBytes, pixels, imageMeta, metaLength);
}

int StorageInstance::GetSummaryMeta(int handle, std::string& meta)
{
	char* cMeta(nullptr);
   RequireInitialized(__func__);
   int ret = GetImpl()->GetSummaryMeta(handle, &cMeta);
   if (ret == DEVICE_OK)
   {
      if (cMeta)
         meta = cMeta;
      GetImpl()->ReleaseStringBuffer(cMeta);
   }
   return ret;
}

int StorageInstance::GetImageMeta(int handle, const std::vector<int>& coordinates, std::string& meta)
{
	char* cMeta(nullptr);
   RequireInitialized(__func__);
   int ret = GetImpl()->GetImageMeta(handle, const_cast<int*>(&coordinates[0]), (int)coordinates.size(), &cMeta);
   if (ret == DEVICE_OK)
   {
      if (cMeta)
         meta = cMeta;
      GetImpl()->ReleaseStringBuffer(cMeta);
   }
   return ret;
}

int StorageInstance::GetCustomMeta(int handle, const std::string& key, std::string& meta)
{
   char* cMeta(nullptr);
   RequireInitialized(__func__);
   int ret = GetImpl()->GetCustomMetadata(handle, key.c_str(), &cMeta);
   if (ret == DEVICE_OK)
   {
      if (cMeta)
         meta = cMeta;
      GetImpl()->ReleaseStringBuffer(cMeta);
   }
   return ret;
}

int StorageInstance::SetCustomMeta(int handle, const std::string& key, const char* meta, int metaLength)
{
   RequireInitialized(__func__);
   return GetImpl()->SetCustomMetadata(handle, key.c_str(), meta, metaLength);
}

const unsigned char* StorageInstance::GetImage(int handle, const std::vector<int>& coordinates)
{
   RequireInitialized(__func__);
   return GetImpl()->GetImage(handle, const_cast<int*>(&coordinates[0]), (int)coordinates.size());
}

int StorageInstance::GetNumberOfDimensions(int handle, int& numDim)
{
   RequireInitialized(__func__);
   return GetImpl()->GetNumberOfDimensions(handle, numDim);
}

int StorageInstance::GetDimension(int handle, int dimension, std::string& name, std::string& meaning)
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

int StorageInstance::GetCoordinate(int handle, int dimension, int coordinate, std::string& name)
{
	char cName[MM::MaxStrLength];
	memset(cName, 0, MM::MaxStrLength);
   int ret = GetImpl()->GetCoordinate(handle, dimension, coordinate, cName, MM::MaxStrLength);
   name = cName;
   return ret;
}

int StorageInstance::GetImageCount(int handle, int& imgcount)
{
	RequireInitialized(__func__);
	return GetImpl()->GetImageCount(handle, imgcount);
}

int StorageInstance::GetPath(int handle, std::string& path)
{
	char cPath[MM::MaxStrLength];
	memset(cPath, 0, MM::MaxStrLength);
	int ret = GetImpl()->GetPath(handle, cPath, MM::MaxStrLength);
	path = cPath;
	return ret;
}

bool StorageInstance::IsOpen(int handle)
{
	RequireInitialized(__func__);
	return GetImpl()->IsOpen(handle);
}

bool StorageInstance::IsReadOnly(int handle)
{
	RequireInitialized(__func__);
	return GetImpl()->IsReadOnly(handle);
}
