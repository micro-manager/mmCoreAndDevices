///////////////////////////////////////////////////////////////////////////////
// FILE:          G2SBigTiffStorage.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Go2Scope devices. Includes the experimental StorageDevice
//
// AUTHOR:        Milos Jovanovic <milos@tehnocad.rs>
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
#include "G2SStorage.h"
#include "G2SBigTiffDataset.h"

/**
 * Storage writer driver for Go2Scope BigTIFF format
 * @author Miloš Jovanović <milos@tehnocad.rs>
 * @version 1.0
 */
class G2SBigTiffStorage : public CStorageBase<G2SBigTiffStorage>
{
public:
   //=========================================================================================================================
	// Constructors & destructors
	//=========================================================================================================================
	G2SBigTiffStorage();
   virtual ~G2SBigTiffStorage() noexcept { Shutdown(); }

public:
   //=========================================================================================================================
	// Public interface - Device API
	//=========================================================================================================================
   int                                             Initialize();
   int                                             Shutdown() noexcept;
   void                                            GetName(char* pszName) const;
   bool                                            Busy() noexcept { return false; }

public:
   //=========================================================================================================================
   // Public interface - Storage API
   //=========================================================================================================================
   int                                             Create(int handle, const char* path, const char* name, int numberOfDimensions, const int shape[], MM::StorageDataType pixType,
                                                          const char* meta, int metaLength) noexcept;
   int                                             ConfigureDimension(int handle, int dimension, const char* name, const char* meaning) noexcept;
   int                                             ConfigureCoordinate(int handle, int dimension, int coordinate, const char* name) noexcept;
   int                                             Close(int handle) noexcept;
   int                                             Load(int handle, const char* path) noexcept;
   int                                             GetShape(int handle, int shape[]) noexcept;
   int                                             GetDataType(int handle, MM::StorageDataType& pixelDataType) noexcept;
   int                                             Delete(int handle) noexcept;
   int                                             List(const char* path, char** listOfDatasets, int maxItems, int maxItemLength) noexcept;
   int                                             AddImage(int handle, int sizeInBytes, unsigned char* pixels, int coordinates[], int numCoordinates, const char* imageMeta, int metaLength) noexcept;
   int                                             AppendImage(int handle, int sizeInBytes, unsigned char* pixels, const char* imageMeta, int metaLength) noexcept;
   int                                             GetSummaryMeta(int handle, char** meta) noexcept;
   int                                             GetImageMeta(int handle, int coordinates[], int numCoordinates, char** meta) noexcept;
   const unsigned char*                            GetImage(int handle, int coordinates[], int numCoordinates) noexcept;
   int                                             GetNumberOfDimensions(int handle, int& numDimensions) noexcept;
   int                                             GetDimension(int handle, int dimension, char* name, int nameLength, char* meaning, int meaningLength) noexcept;
   int                                             GetCoordinate(int handle, int dimension, int coordinate, char* name, int nameLength) noexcept;
	int															GetImageCount(int handle, int& imgcnt) noexcept;
	int															SetCustomMetadata(int handle, const char* key, const char* content, int contentLength) noexcept;
	int															GetCustomMetadata(int handle, const char* key, char** content) noexcept;
   bool                                            IsOpen(int handle) noexcept;
	bool                                            IsReadOnly(int handle) noexcept;
   int                                             GetPath(int handle, char* path, int maxPathLength) noexcept;
	bool															CanLoad(const char* path) noexcept;

protected:
   //=========================================================================================================================
   // Internal methods
   //=========================================================================================================================
   void                                            cacheReduce() noexcept;
   bool                                            scanDir(const std::string& path, char** listOfDatasets, int maxItems, int maxItemLength, int cpos) noexcept;
	bool															validateCoordinates(const G2SBigTiffDataset* fs, int coordinates[], int numCoordinates, bool flexaxis0 = false) noexcept;
	bool															getDirectIO() const noexcept;
	int															getFlushCycle() const noexcept;
	int															getChunkSize() const noexcept;

private:
   //=========================================================================================================================
   // Data members
   //=========================================================================================================================
   std::map<int, G2SStorageEntry>          cache;                                 ///< Storage entries cache
   std::vector<std::string>                supportedFormats = { "g2s" };          ///< Supported file formats
   bool                                    initialized = false;                   ///< Is driver initialized
};
