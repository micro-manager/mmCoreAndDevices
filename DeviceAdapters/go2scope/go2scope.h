///////////////////////////////////////////////////////////////////////////////
// FILE:          Go2Scope.cpp
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

//////////////////////////////////////////////////////////////////////////////
// Error codes
//

#define ERR_PARAMETER_ERROR          144001
#define ERR_INTERNAL						 144002
#define ERR_ZARR                     140100
#define ERR_ZARR_SETTINGS            140101

#define ERR_TIFF                     140500

static const char* g_Go2Scope = "Go2Scope";
static const char* g_MMV1Storage = "MMV1Storage";
static const char* g_AcqZarrStorage = "AcquireZarrStorage";

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
   int Create(const char* path, const char* name, int numberOfDimensions, const int shape[], const char* meta, char* handle);
   int ConfigureDimension(const char* handle, int dimension, const char* name, const char* meaning);
   int ConfigureCoordinate(const char* handle, int dimension, int coordinate, const char* name);
   int Close(const char* handle);
   int Load(const char* path, const char* name, char* handle);
   int Delete(char* handle);
   int List(const char* path, char** listOfDatasets, int maxItems, int maxItemLength);
   int AddImage(const char* handle, unsigned char* pixels, int width, int height, int depth, int coordinates[], int numCoordinates, const char* imageMeta);
   int GetSummaryMeta(const char* handle, char* meta, int bufSize);
   int GetImageMeta(const char* handle, int coordinates[], int numCoordinates, char* meta, int bufSize);
   const unsigned char* GetImage(const char* handle, int coordinates[], int numCoordinates);
   int GetNumberOfDimensions(const char* handle, int& numDimensions);
   int GetDimension(const char* handle, int dimension, char* name, int nameLength, char* meaning, int meaningLength);
   int GetCoordinate(const char* handle, int dimension, int coordinate, char* name, int nameLength);

   // action interface
   // ----------------

private:
   bool initialized;
};

