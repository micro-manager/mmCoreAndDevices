///////////////////////////////////////////////////////////////////////////////
// FILE:          G2STiffFile.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   BIGTIFF storage device driver
//
// AUTHOR:        Milos Jovanovic <milos@tehnocad.rs>
//
// COPYRIGHT:     Luminous Point LLC, Lumencor Inc. 2024
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
//// 
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <string>
#include <map>
#include <vector>
#include <cmath>
#include "G2SBigTiffStream.h"

/**
 * Go2Scope BigTiff File Parser
 * Go2Scope BigTiff format is the extension of the BIgTIFF format (v6)
 * Supports both storing the entire datasets (image acquisitions) in a single file
 *  or
 * Chunking the dataset in multiple files, 
 * Chunk contains a subset of images that have a common slowest changing dimension
 * Support for dataset and per-image metadata
 * Support for large files (>2GB)
 * Support for Direct and Cached I/O
 * Support for sequential & random image access (read / write)
 * @author Miloš Jovanović <milos@tehnocad.rs>
 * @version 1.0
 */
class G2SBigTiffDataset
{
public:
	//============================================================================================================================
	// Constructors & Destructors
	//============================================================================================================================
	G2SBigTiffDataset() noexcept;
	G2SBigTiffDataset(const G2SBigTiffDataset& src) noexcept = default;
	~G2SBigTiffDataset() noexcept { close(); }

public:
	//============================================================================================================================
	// Public interface
	//============================================================================================================================
	void															create(const std::string& path, bool dio = DEFAULT_DIRECT_IO, bool fbig = DEFAULT_BIGTIFF, std::uint32_t chunksz = 0);
	void															load(const std::string& path, bool dio = DEFAULT_DIRECT_IO);
	void															close() noexcept;
	void															setFlushCycles(std::uint32_t val) noexcept { flushcnt = val; }
	std::uint32_t												getChunkSize() const noexcept { return chunksize; }
	void															setShape(const std::vector<std::uint32_t>& dims);
	void															setShape(std::initializer_list<std::uint32_t> dims);
	std::vector<std::uint32_t>								getShape() const noexcept { return shape; }
	std::size_t													getDimension() const noexcept { return shape.size(); }
	std::uint32_t												getAxisSize(std::size_t ind) const noexcept { return ind < shape.size() ? shape[ind] : 0; }
	std::uint32_t												getWidth() const noexcept { return shape.size() < 2 ? 0 : shape[shape.size() - 1]; }
	std::uint32_t												getHeight() const noexcept { return shape.size() < 2 ? 0 : shape[shape.size() - 2]; }
	void															setPixelFormat(std::uint8_t depth, std::uint8_t vsamples = 1);
	int															getBitDepth() const noexcept { return (int)bitdepth; }
	int															getBpp() const noexcept { return (int)std::ceil(bitdepth / 8.0); }
	int															getSamples() const noexcept { return (int)samples; }
	void															setMetadata(const std::string& meta);
	std::string													getMetadata() const noexcept;
	void															setUID(const std::string& val);
	std::string													getUID() const noexcept { return datasetuid; }
	std::string													getImageMetadata(const std::vector<std::uint32_t>& coord = {});
	void															addImage(const std::vector<unsigned char>& buff, const std::string& meta = "") { addImage(&buff[0], buff.size(), meta); }
	void															addImage(const unsigned char* buff, std::size_t len, const std::string& meta = "");
	std::vector<unsigned char>								getImage(const std::vector<std::uint32_t>& coord = {});
	std::uint32_t												getDatasetImageCount() const noexcept { std::uint32_t ret = 1; for(std::size_t i = 0; i < shape.size() - 2; i++) ret *= shape[i]; return ret; }
	std::uint32_t												getImageCount() const noexcept { return imgcounter; }
	std::string													getPath() const noexcept { return dspath; }
	std::string													getName() const noexcept { return dsname; }
	bool															isDirectIO() const noexcept { return directIo; }
	bool															isBigTIFF() const noexcept { return bigTiff; }
	bool															isInWriteMode() const noexcept { return writemode; }
	bool															isInReadMode() const noexcept { return !writemode; }
	bool															isOpen() const noexcept { return !datachunks.empty() && activechunk; }


private:
	//============================================================================================================================
	// Internal methods
	//============================================================================================================================
	void															switchDataChunk(std::uint32_t chunkind);
	void															validateDataChunk(std::uint32_t chunkind, bool index);
	void															selectImage(const std::vector<std::uint32_t>& coord);
	void															advanceImage();
	std::uint32_t												getChunkImageCount() const noexcept;
	void															calcImageIndex(const std::vector<std::uint32_t>& coord, std::uint32_t& chunkind, std::uint32_t& imgind) const;

private:
	//============================================================================================================================
	// Data members - Dataset properties
	//============================================================================================================================
	std::string													dspath;											///< Dataset (directory) path
	std::string													dsname;											///< Dataset name
	std::string													datasetuid;										///< Dataset UID
	std::vector<std::uint32_t>								shape;											///< Dataset shape (dimension / axis sizes)
	std::vector<G2SFileStreamHandle>						datachunks;										///< Data chunks / File stream descriptors
	G2SFileStreamHandle										activechunk;									///< Active data chunk
	std::vector<unsigned char>								metadata;										///< Dataset metdata (cache)
	std::uint32_t												imgcounter;										///< Image counter
	std::uint32_t												flushcnt;										///< Image flush cycles
	std::uint32_t												chunksize;										///< Chunk size
	std::uint8_t												bitdepth;										///< Bit depth
	std::uint8_t												samples;											///< Samples per pixel
	bool															directIo;										///< Use direct I/O for file operations
	bool															bigTiff;											///< Use big TIFF format
	bool															writemode;										///< Is dataset opened for writing
};

