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
   int                                             Shutdown();
   void                                            GetName(char* pszName) const;
   bool                                            Busy() { return false; }

public:
   //=========================================================================================================================
   // Public interface - Storage API
   //=========================================================================================================================
   int                                             Create(const char* path, const char* name, int numberOfDimensions, const int shape[], MM::StorageDataType pixType, const char* meta, char* handle);
   int                                             ConfigureDimension(const char* handle, int dimension, const char* name, const char* meaning);
   int                                             ConfigureCoordinate(const char* handle, int dimension, int coordinate, const char* name);
   int                                             Close(const char* handle);
   int                                             Load(const char* path, char* handle);
   int                                             GetShape(const char* handle, int shape[]);
   int                                             GetDataType(const char* handle, MM::StorageDataType& pixelDataType);
   int                                             Delete(char* handle);
   int                                             List(const char* path, char** listOfDatasets, int maxItems, int maxItemLength);
   int                                             AddImage(const char* handle, int sizeInBytes, unsigned char* pixels, int coordinates[], int numCoordinates, const char* imageMeta);
   int                                             GetSummaryMeta(const char* handle, char* meta, int bufSize);
   int                                             GetImageMeta(const char* handle, int coordinates[], int numCoordinates, char* meta, int bufSize);
   const unsigned char*                            GetImage(const char* handle, int coordinates[], int numCoordinates);
   int                                             GetNumberOfDimensions(const char* handle, int& numDimensions);
   int                                             GetDimension(const char* handle, int dimension, char* name, int nameLength, char* meaning, int meaningLength);
   int                                             GetCoordinate(const char* handle, int dimension, int coordinate, char* name, int nameLength);
   bool                                            IsOpen(const char* handle);
   int                                             GetPath(const char* handle, char* path, int maxPathLength);


public:
   //=========================================================================================================================
   // Public interface - Action interface
   //=========================================================================================================================

protected:
   //=========================================================================================================================
   // Internal methods
   //=========================================================================================================================
   void                                            cacheReduce() noexcept;
   bool                                            scanDir(const std::string& path, char** listOfDatasets, int maxItems, int maxItemLength, int cpos) noexcept;
	bool															validateCoordinates(const G2SBigTiffDataset* fs, int coordinates[], int numCoordinates) noexcept;
   std::string                                     getImageKey(int coordinates[], int numCoordinates) noexcept;
	bool															getDirectIO() const noexcept;
	int															getFlushCycle() const noexcept;
	int															getChunkSize() const noexcept;

private:
   //=========================================================================================================================
   // Data members
   //=========================================================================================================================
   std::map<std::string, G2SStorageEntry>          cache;                                 ///< Storage entries cache
   std::vector<std::string>                        supportedFormats;                      ///< Supported file formats
   bool                                            initialized;                           ///< Is driver initialized
};
