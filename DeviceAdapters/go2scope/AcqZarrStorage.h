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
   int Create(const char* path, const char* name, int numberOfDimensions, const int shape[], MM::StorageDataType pixType, const char* meta, char* handle);
   int ConfigureDimension(const char* handle, int dimension, const char* name, const char* meaning);
   int ConfigureCoordinate(const char* handle, int dimension, int coordinate, const char* name);
   int Close(const char* handle);
   int Load(const char* path, char* handle);
   int GetShape(const char* handle, int shape[]);
   int GetDataType(const char* handle, MM::StorageDataType& pixelDataType) { return dataType; }

   int Delete(char* handle);
   int List(const char* path, char** listOfDatasets, int maxItems, int maxItemLength);
   int AddImage(const char* handle, int sizeInBytes, unsigned char* pixels, int coordinates[], int numCoordinates, const char* imageMeta);
   int GetSummaryMeta(const char* handle, char* meta, int bufSize);
   int GetImageMeta(const char* handle, int coordinates[], int numCoordinates, char* meta, int bufSize);
   const unsigned char* GetImage(const char* handle, int coordinates[], int numCoordinates);
   int GetNumberOfDimensions(const char* handle, int& numDimensions);
   int GetDimension(const char* handle, int dimension, char* name, int nameLength, char* meaning, int meaningLength);
   int GetCoordinate(const char* handle, int dimension, int coordinate, char* name, int nameLength);
   bool IsOpen(const char* handle);
   int  GetPath(const char* handle, char* path, int maxPathLength);


   // action interface
   // ----------------

private:
   bool initialized;
   ZarrStream_s* zarrStream;
   std::vector<int> streamDimensions;
   MM::StorageDataType dataType;
   std::vector<int> currentCoordinate;
   int currentImageNumber;
   std::string streamHandle;
   std::string getErrorMessage(int code);
   void destroyStream();
   int ConvertToZarrType(MM::StorageDataType type);
   std::string streamPath;
};

