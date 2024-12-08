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
#define ERR_PARAMETER_ERROR          144001
#define ERR_INTERNAL						 144002
#define ERR_FAILED_CREATING_FILE		 144003
#define ERR_ZARR                     140100
#define ERR_ZARR_SETTINGS            140101
#define ERR_ZARR_NUMDIMS             140102
#define ERR_ZARR_STREAM_CREATE       140103
#define ERR_ZARR_STREAM_CLOSE        140104
#define ERR_ZARR_STREAM_LOAD         140105
#define ERR_ZARR_STREAM_APPEND       140106
#define ERR_ZARR_STREAM_ACCESS       140107

#define ERR_TIFF                     140500
#define ERR_TIFF_STREAM_UNAVAILABLE  140501
#define ERR_TIFF_INVALID_PATH			 140502
#define ERR_TIFF_INVALID_DIMENSIONS	 140503
#define ERR_TIFF_INVALID_PIXEL_TYPE  140504
#define ERR_TIFF_CACHE_OVERFLOW      140505
#define ERR_TIFF_OPEN_FAILED			 140506
#define ERR_TIFF_CACHE_INSERT			 140507
#define ERR_TIFF_HANDLE_INVALID		 140508

//////////////////////////////////////////////////////////////////////////////
// Cache configuration
//
//////////////////////////////////////////////////////////////////////////////
#define MAX_CACHE_SIZE					1024
#define CACHE_HARD_LIMIT				0

static const char* g_Go2Scope = "Go2Scope";
static const char* g_MMV1Storage = "MMV1Storage";
static const char* g_AcqZarrStorage = "AcquireZarrStorage";
static const char* g_BigTiffStorage = "G2SBigTiffStorage";

/**
 * Dataset dimension descriptor
 * @author Miloš Jovanović <milos@tehnocad.rs>
 * @version 1.0
 */
struct G2SDimensionInfo
{
	/**
	 * Default initializer
	 * @param vname Axis name
	 * @param ndim Axis size
	 */
	G2SDimensionInfo(int ndim = 0) noexcept : Name(""), Coordinates(ndim) { }

	/**
	 * Set dimensions size
	 * @param sz Number of axis coordinates
	 */
	void setSize(std::size_t sz) noexcept { Coordinates.resize(sz); }
	/**
	 * Get dimension size
	 * @return Number of axis coordinates
	 */
	std::size_t getSize() const noexcept { return Coordinates.size(); }

	std::string													Name;												///< Axis name
	std::string													Metadata;										///< Axis metadata
	std::vector<std::string>								Coordinates;									///< Axis coordinates
};

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
	 * @param ndim Number of dimensions
	 * @param shape Axis sizes
	 * @param vmeta Dataset metadata
	 */
	G2SStorageEntry(const std::string& vpath, int ndim, const int* shape = nullptr, const char* vmeta = nullptr) noexcept : Path(vpath), Dimensions(ndim)
	{
		if(shape != nullptr)
		{
			for(std::size_t i = 0; i < Dimensions.size(); i++)
				Dimensions[i].setSize((std::size_t)shape[i]);
		}
		if(vmeta != nullptr)
			Metadata = std::string(vmeta);
		FileHandle = nullptr;
	}

	/**
	 * Close the descriptor
	 */
	void close() noexcept { FileHandle = nullptr; Metadata.clear(); ImageMetadata.clear(); ImageData.clear(); }
	/**
	 * Check if file handle is open
	 * @return Is file handle open
	 */
	bool isOpen() noexcept { return FileHandle != nullptr; }
	/**
	 * Get number of dimensions
	 * @return Number of dataset dimensions
	 */
	std::size_t getDimSize() const noexcept { return Dimensions.size(); }

	std::string													Path;												///< Absoulute path on disk
	std::string													Metadata;										///< Dataset metadata
	std::vector<G2SDimensionInfo>							Dimensions;										///< Dataset dimensions vector
	std::map<std::string, std::string>					ImageMetadata;									///< Per-image metadata
	std::vector<unsigned char>								ImageData;										///< Current image cache
	void*															FileHandle;										///< File handle
};
