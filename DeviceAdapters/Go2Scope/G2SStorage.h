///////////////////////////////////////////////////////////////////////////////
// FILE:          G2SStorage.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Go2Scope devices. Includes the experimental StorageDevice
//
// AUTHORS:       Milos Jovanovic <milos@tehnocad.rs>
//						Nenad Amodaj <nenad@go2scope.com>
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
//////////////////////////////////////////////////////////////////////////////
#define ERR_INTERNAL						 144002

#define ERR_TIFF                     140500
#define ERR_TIFF_STREAM_UNAVAILABLE  140501
#define ERR_TIFF_INVALID_PATH			 140502
#define ERR_TIFF_INVALID_DIMENSIONS	 140503
#define ERR_TIFF_INVALID_PIXEL_TYPE  140504
#define ERR_TIFF_CACHE_OVERFLOW      140505
#define ERR_TIFF_OPEN_FAILED			 140506
#define ERR_TIFF_CACHE_INSERT			 140507
#define ERR_TIFF_HANDLE_INVALID		 140508
#define ERR_TIFF_STRING_TOO_LONG		 140509
#define ERR_TIFF_INVALID_COORDINATE	 140510
#define ERR_TIFF_DATASET_CLOSED		 140511
#define ERR_TIFF_DATASET_READONLY	 140512
#define ERR_TIFF_DELETE_FAILED		 140513
#define ERR_TIFF_ALLOCATION_FAILED	 140514
#define ERR_TIFF_CORRUPTED_METADATA	 140515
#define ERR_TIFF_UPDATE_FAIL			 140516
#define ERR_TIFF_FILESYSTEM_ERROR	 140517
#define ERR_TIFF_INVALID_META_KEY	 140518

//////////////////////////////////////////////////////////////////////////////
// Cache configuration
//
//////////////////////////////////////////////////////////////////////////////
#define MAX_CACHE_SIZE					1024
#define CACHE_HARD_LIMIT				0

static const char* g_BigTiffStorage = "G2SBigTiffStorage";

/**
 * Storage entry descriptor
 * @author Miloš Jovanović <milos@tehnocad.rs>
 * @version 1.0
 */
struct G2SStorageEntry
{
	/**
	 * Default initializer
	 * @param vpath Absoulute path on disk
	 * @param shape Axis sizes
	 */
	G2SStorageEntry(const std::string& vpath) noexcept : Path(vpath), FileHandle(nullptr) { }

	/**
	 * Close the descriptor
	 */
	void close() noexcept { FileHandle = nullptr; ImageData.clear(); }
	/**
	 * Check if file handle is open
	 * @return Is file handle open
	 */
	bool isOpen() noexcept { return FileHandle != nullptr; }

	std::string													Path;												///< Absoulute path on disk
	std::vector<unsigned char>								ImageData;										///< Current image data
	void*															FileHandle;										///< File handle
};
