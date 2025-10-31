///////////////////////////////////////////////////////////////////////////////
// FILE:          G2SBigTiffStream.h
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
#include <string>
#include <vector>
#include <climits>
#include "G2SFileUtil.h"

/**
 * BigTIFF file stream descriptor / state cache
 * @author Miloš Jovanović <milos@tehnocad.rs>
 * @version 1.0
 */
class G2SBigTiffStream
{
public:
	//============================================================================================================================
	// Constructors & Destructors
	//============================================================================================================================
	G2SBigTiffStream(const std::string& path, bool dio, bool fbig = DEFAULT_BIGTIFF, std::uint32_t chunk = 0) noexcept;
	G2SBigTiffStream(const G2SBigTiffStream& src) noexcept = default;
	~G2SBigTiffStream() noexcept { close(); }

public:
	//============================================================================================================================
	// Public interface - Common methods
	//============================================================================================================================
	void															open(bool trunc);
	void															close() noexcept;
	void															parse(std::string& datasetuid, std::vector<std::uint32_t>& shape, std::uint32_t& chunksize, std::vector<unsigned char>& metadata, std::uint8_t& bitdepth, bool index = true);
	void															formHeader() noexcept;
	void															addImage(const unsigned char* buff, std::size_t len, std::uint32_t imgw, std::uint32_t imgh, std::uint32_t imgdepth, const std::string& meta = "");
	std::vector<unsigned char>								getImage();
	std::string													getImageMetadata() const;
	void															writeShapeInfo(const std::vector<std::uint32_t>& shape, std::uint32_t chunksz) noexcept;
	void															writeDatasetUid(const std::string& datasetuid) noexcept;
	void															advanceIFD() noexcept { currentifd.clear(); currentifdpos = nextifdpos; }
	void															setChunkIndex(std::uint32_t val) noexcept;
	std::uint32_t												getChunkIndex() const noexcept { return chunkindex; }
	void															setCurrentImage(std::uint32_t val) noexcept { currentimage = val; }
	std::uint32_t												getCurrentImage() const noexcept { return currentimage; }
	void															appendMetadata(const std::vector<unsigned char>& meta);
	const std::vector<unsigned char>&					getHeader() const noexcept { return header; }
	std::vector<unsigned char>&							getHeader() noexcept { return header; }
	const std::vector<unsigned char>&					getCurrentIFD() const noexcept { return currentifd; }
	const std::vector<unsigned char>&					getLastIFD() const noexcept { return lastifd; }
	const std::vector<std::uint64_t>&					getIFDOffsets() const noexcept { return ifdcache; }
	std::uint64_t												getCurrentIFDOffset() const noexcept { return currentifdpos; }
	std::uint64_t												getNextIFDOffset() const noexcept { return nextifdpos; }
	std::string													getFilePath() const noexcept { return fpath; }
	std::uint64_t												getFileSize() const noexcept;
	std::uint64_t												getMaxFileSize() const noexcept { return bigTiff ? std::numeric_limits<std::uint64_t>::max() : std::numeric_limits<std::uint32_t>::max(); }
	std::uint32_t												getImageCount() const noexcept { return imgcounter; }
	bool															isBigTiff() const noexcept { return bigTiff; }
#ifdef _WIN32
	bool															isOpen() const noexcept { return fhandle != nullptr; }
#else
	bool															isOpen() const noexcept { return fhandle > 0; }
#endif

public:
	//============================================================================================================================
	// Public interface - File stream manipulation
	//============================================================================================================================
	std::size_t													commit(const unsigned char* buff, std::size_t len);
	std::size_t													write(const unsigned char* buff, std::size_t len);
	std::size_t													fetch(unsigned char* buff, std::size_t len);
	std::size_t													read(unsigned char* buff, std::size_t len);
	std::uint64_t												seek(std::int64_t pos, bool beg = true);
	std::uint64_t												offset(std::int64_t off);
	void															flush() const;

public:
	//============================================================================================================================
	// Public interface - Helper methods
	//============================================================================================================================
	void															appendIFD(std::uint32_t imgw, std::uint32_t imgh, std::uint32_t imgdepth, std::size_t imagelen, const std::string& meta);
	std::size_t													setIFDTag(unsigned char* ifd, std::uint16_t tag, std::uint16_t dtype, std::uint64_t val, std::uint64_t cnt = 1) const noexcept;
	std::uint32_t												getTagCount(const std::string& meta) const noexcept { return meta.empty() ? G2STIFF_TAG_COUNT_NOMETA : G2STIFF_TAG_COUNT; }
	std::uint64_t												parseIFD(std::vector<unsigned char>& ifd, std::uint32_t& ifdsz);
	void															loadNextIFD() { loadIFD(currentifdpos); }
	void															loadIFD(std::uint64_t off);
	void															calcDescSize(std::size_t metalen, std::uint32_t tags, std::uint32_t* ifd, std::uint32_t* desc, std::uint32_t* tot) noexcept;
	void															moveReadCursor(std::uint64_t pos) noexcept;
	void															moveWriteCursor(std::uint64_t pos) noexcept;

private:
	//============================================================================================================================
	// Data members - Stream state / cache
	//============================================================================================================================
	std::vector<unsigned char>								header;											///< Header (cache)
	std::vector<unsigned char>								lastifd;											///< Last IFD (cache)
	std::vector<unsigned char>								currentifd;										///< Current IFD (cache)
	std::vector<unsigned char>								writebuff;										///< Write buffer for direct I/O
	std::vector<unsigned char>								readbuff;										///< Read buffer for direct I/O
	std::size_t													readbuffoff;									///< Read buffer offset
	std::uint64_t												currpos;											///< Current file stream offset
	std::uint64_t												writepos;										///< Write stream offset
	std::uint64_t												readpos;											///< Read stream offset
	std::uint64_t												lastifdpos;										///< Offset of the last image descriptor
	std::uint32_t												lastifdsize;									///< Last IFD size
	std::uint64_t												currentifdpos;									///< Offset of the current image descriptor
	std::uint32_t												currentifdsize;								///< Current IFD size
	std::uint64_t												nextifdpos;										///< Offset of the next image descriptor
	std::uint32_t												currentimage;									///< Current image index (used for reading only)
	std::uint32_t												imgcounter;										///< Image counter
	std::uint32_t												chunkindex;										///< Chunk index
	std::vector<std::uint64_t>								ifdcache;										///< IFD offset cache
#ifdef _WIN32
	void*															fhandle;											///< File handle
#else
	int															fhandle;											///< File descriptor
#endif

private:
	//============================================================================================================================
	// Data members - Configuration
	//============================================================================================================================
	std::string													fpath;											///< File path
	std::uint32_t												ssize;											///< Sector size (for direct I/O)
	bool															directIo;										///< Use direct I/O for file operations
	bool															bigTiff;											///< Use big TIFF format
};

typedef std::shared_ptr<G2SBigTiffStream> G2SFileStreamHandle;