///////////////////////////////////////////////////////////////////////////////
// FILE:          AcqZarrStorage.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Go2Scope devices. Includes the experimental StorageDevice
//
// AUTHOR:        Nenad Amodaj
//
// COPYRIGHT:     Nenad Amodaj, Chan Zuckerberg Initiative, 2024
//
// LICENSE:       This file is distributed under the BSD license.
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
// NOTE:          Storage Device development is supported in part by
//                Chan Zuckerberg Initiative (CZI)
// 
///////////////////////////////////////////////////////////////////////////////

#pragma once
#include "MMDevice.h"
#include "DeviceBase.h"

struct ZarrStream_s;

class AcqZarrStorage : public CStorageBase<AcqZarrStorage>
{
public:
   AcqZarrStorage();
   virtual ~AcqZarrStorage();

   // Device API
   // ----------
   int Initialize();
   int Shutdown();

   void GetName(char* pszName) const;
   bool Busy();

   // Storage API
   // -----------
   int Create(int handle, const char* path, const char* name, int numberOfDimensions, const int shape[], MM::StorageDataType pixType, const char* meta, int metaLength);
   int ConfigureDimension(int handle, int dimension, const char* name, const char* meaning);
   int ConfigureCoordinate(int handle, int dimension, int coordinate, const char* name);
   int Close(int handle);
   int Load(int handle, const char* path);
   int GetShape(int handle, int shape[]);
   int GetDataType(int handle, MM::StorageDataType& pixelDataType) { return dataType; }

   int Delete(int handle);
   int List(const char* path, char** listOfDatasets, int maxItems, int maxItemLength);
   int AddImage(int handle, int sizeInBytes, unsigned char* pixels, int coordinates[], int numCoordinates, const char* imageMeta, int metaLength);
   int AppendImage(int handle, int sizeInBytes, unsigned char* pixels, const char* imageMeta, int metaLength);
   int GetSummaryMeta(int handle, char** meta);
   int GetImageMeta(int handle, int coordinates[], int numCoordinates, char** meta);
   const unsigned char* GetImage(int handle, int coordinates[], int numCoordinates);
   int GetNumberOfDimensions(int handle, int& numDimensions);
   int GetDimension(int handle, int dimension, char* name, int nameLength, char* meaning, int meaningLength);
   int GetCoordinate(int handle, int dimension, int coordinate, char* name, int nameLength);
	int GetImageCount(int handle, int& imgcnt);
   bool IsOpen(int handle);
	bool IsReadOnly(int handle);
   int GetPath(int handle, char* path, int maxPathLength);
	int SetCustomMetadata(int handle, const char* key, const char* content, int contentLength) { return DEVICE_UNSUPPORTED_COMMAND; }
	int GetCustomMetadata(int handle, const char* key, char** content) { return DEVICE_UNSUPPORTED_COMMAND; }


   // action interface
   // ----------------

private:
   bool initialized;
   ZarrStream_s* zarrStream;
   std::vector<int> streamDimensions;
   MM::StorageDataType dataType;
   std::vector<int> currentCoordinate;
   int currentImageNumber;
   bool datasetIsOpen = false; // May be redundant with zarrStream != nullptr
   int theHandle = -1; // Only one dataset/handle supported at a time
   std::string getErrorMessage(int code);
   void destroyStream();
   int ConvertToZarrType(MM::StorageDataType type);
   std::string streamPath;
};

