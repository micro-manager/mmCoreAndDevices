// PROJECT:       Micro-Manager
// SUBSYSTEM:     MMCore
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

#pragma once

#include "DeviceInstanceBase.h"


class StorageInstance : public DeviceInstanceBase<MM::Storage>
{
public:
   StorageInstance(CMMCore* core,
      std::shared_ptr<LoadedDeviceAdapter> adapter,
      const std::string& name,
      MM::Device* pDevice,
      DeleteDeviceFunction deleteFunction,
      const std::string& label,
      mm::logging::Logger deviceLogger,
      mm::logging::Logger coreLogger) :
      DeviceInstanceBase<MM::Storage>(core, adapter, name, pDevice, deleteFunction, label, deviceLogger, coreLogger) {}

   int Create(const char* path, const char* name, const std::vector<int>& shape, MM::StorageDataType pixType, const char* meta, int metaLength, int& handle);
   int ConfigureDimension(int handle, int dimension, const char* name, const char* meaning);
   int ConfigureCoordinate(int handle, int dimension, int coordinate, const char* name);
	int GetPath(int handle, std::string& path);
   int Close(int handle);
   int Load(const char* path, int& handle);
   int GetShape(int handle, std::vector<long>& shape);
   int GetPixelType(int handle, MM::StorageDataType& dataType);
   int Delete(int handle);
   int List(const char* path, std::vector<std::string>& datasets);
   int AddImage(int handle, int sizeInBytes, unsigned char* pixels, std::vector<int>& coordinates, const char* imageMeta, int imageMetaLength);
   int AppendImage(int handle, int sizeInBytes, unsigned char* pixels, const char* imageMeta, int imageMetaLength);
   int GetSummaryMeta(int handle, std::string& meta);
   int GetImageMeta(int handle, const std::vector<int>& coordinates, std::string& meta);
   int GetCustomMeta(int handle, const std::string& key, std::string& meta);
   int SetCustomMeta(int handle, const std::string& key, const char* meta, int metaLength);
   const unsigned char* GetImage(int handle, const std::vector<int>& coordinates);
   int GetNumberOfDimensions(int handle, int& numDimensions);
   int GetDimension(int handle, int dimension, std::string& name, std::string& meaning);
   int GetCoordinate(int handle, int dimension, int coordinate, std::string& name);
	int GetImageCount(int handle, int& imgcnt);
	bool IsOpen(int handle);
	bool IsReadOnly(int handle);
};
