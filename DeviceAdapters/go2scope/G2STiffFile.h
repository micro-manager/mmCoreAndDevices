///////////////////////////////////////////////////////////////////////////////
// FILE:          G2STiffFile.h
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
#include <map>
#include <vector>
#include <cmath>
#include <climits>

#define DEFAULT_BIGTIFF					true
#define DEFAULT_DIRECT_IO				false
#define TIFF_MAX_BUFFER_SIZE			2147483648U
#define G2STIFF_HEADER_SIZE			512
#define G2STIFF_TAG_COUNT				12
#define G2STIFF_TAG_COUNT_NOMETA		11

/**
 * Go2Scope BigTiff File Parser
 * Go2Scope BigTiff format is the extension of the BIgTIFF format (v6)
 * Stores entire datasets (image acquisitions) in a single file
 * Support for dataset and per-image metadata
 * Support for large files (>2GB)
 * Support for Direct and Cached I/O
 * Support for sequential image access (read / write)
 * @author Miloš Jovanović <milos@tehnocad.rs>
 * @version 1.0
 */
class G2STiffFile
{
public:
	//============================================================================================================================
	// Constructors & Destructors
	//============================================================================================================================
	G2STiffFile(const std::string& path, bool fbig = DEFAULT_BIGTIFF) noexcept;
	G2STiffFile(const G2STiffFile& src) noexcept = default;
	~G2STiffFile() noexcept { close(); }

public:
	//============================================================================================================================
	// Public interface
	//============================================================================================================================
	void															open(bool trunc = false, bool dio = DEFAULT_DIRECT_IO);
	void															close() noexcept;
	void															setShape(const std::vector<std::uint32_t>& dims);
	void															setShape(std::initializer_list<std::uint32_t> dims);
	std::vector<std::uint32_t>								getShape() const noexcept { return shape; }
	std::size_t													getDimension() const noexcept { return shape.size(); }
	std::uint32_t												getAxisSize(std::size_t ind) const noexcept { return ind < shape.size() ? shape[ind] : 0; }
	std::uint32_t												getWidth() const noexcept { return shape.size() < 1 ? 0 : shape[0]; }
	std::uint32_t												getHeight() const noexcept { return shape.size() < 2 ? 0 : shape[1]; }
	void															setPixelFormat(std::uint8_t depth, std::uint8_t vsamples = 1);
	int															getBitDepth() const noexcept { return (int)bitdepth; }
	int															getBpp() const noexcept { return (int)std::ceil(bitdepth / 8.0); }
	int															getSamples() const noexcept { return (int)samples; }
	void															setMetadata(const std::string& meta) noexcept;
	std::string													getMetadata();
	void															setUID(const std::string& val);
	std::string													getUID() const noexcept { return datasetuid; }
	std::string													getImageMetadata();
	void															addImage(const std::vector<unsigned char>& buff, const std::string& meta = "") { addImage(&buff[0], buff.size(), meta); }
	void															addImage(const unsigned char* buff, std::size_t len, const std::string& meta = "");
	std::vector<unsigned char>								getImage();
	std::uint32_t												getDatasetImageCount() const noexcept { std::uint32_t ret = 1; for(std::size_t i = 2; i < shape.size(); i++) ret *= shape[i]; return ret; }
	std::uint32_t												getImageCount() const noexcept { return imgcounter; }
	std::uint64_t												getFileSize() const noexcept;
	std::uint64_t												getMaxFileSize() const noexcept { return bigTiff ? std::numeric_limits<std::uint64_t>::max() : std::numeric_limits<std::uint32_t>::max(); }
	bool															isDirectIO() const noexcept { return directIo; }
	bool															isBigTIFF() const noexcept { return bigTiff; }
#ifdef _WIN32
	bool															isOpen() const noexcept { return fhandle != nullptr; }
#else
	bool															isOpen() const noexcept { return fhandle > 0; }
#endif

private:
	//============================================================================================================================
	// Internal methods
	//============================================================================================================================
	std::size_t													commit(const unsigned char* buff, std::size_t len);
	std::size_t													write(const unsigned char* buff, std::size_t len);
	std::size_t													fetch(unsigned char* buff, std::size_t len);
	std::size_t													read(unsigned char* buff, std::size_t len);
	std::uint64_t												seek(std::int64_t pos, bool beg = true);
	std::uint64_t												offset(std::int64_t off);
	void															formHeader() noexcept;
	void															appendIFD(std::size_t imagelen, const std::string& meta);
	void															loadNextIFD();
	std::uint64_t												parseIFD(std::vector<unsigned char>& ifd, std::uint32_t& ifdsz);
	std::size_t													setIFDTag(unsigned char* ifd, std::uint16_t tag, std::uint16_t dtype, std::uint64_t val, std::uint64_t cnt = 1) const noexcept;
	std::uint32_t												getTagCount(const std::string& meta) const noexcept { return meta.empty() ? G2STIFF_TAG_COUNT_NOMETA : G2STIFF_TAG_COUNT; }
	void															calcDescSize(std::size_t metalen, std::uint32_t tags, std::uint32_t* ifd, std::uint32_t* desc, std::uint32_t* tot) noexcept;
	void															writeShapeInfo() noexcept;
	void															writeInt(unsigned char* buff, std::uint8_t len, std::uint64_t val) const noexcept;
	std::uint64_t												readInt(const unsigned char* buff, std::uint8_t len) const noexcept;
	void															moveReadCursor(std::uint64_t pos) noexcept;
	void															moveWriteCursor(std::uint64_t pos) noexcept;

private:
	//============================================================================================================================
	// Data members - Dataset properties
	//============================================================================================================================
	std::string													fpath;											///< File path
	std::string													datasetuid;										///< Dataset UID
	std::vector<std::uint32_t>								shape;											///< Dataset shape (dimension / axis sizes)
	std::uint32_t												ssize;											///< Sector size (for direct I/O)
	std::uint32_t												imgcounter;										///< Image counter
	std::uint32_t												currentimage;									///< Current image index (used for reading only)
	std::uint8_t												bitdepth;										///< Bit depth
	std::uint8_t												samples;											///< Samples per pixel
	bool															directIo;										///< Use direct I/O for file operations
	bool															bigTiff;											///< Use big TIFF format
	bool															configset;										///< Is dataset configuration set
#ifdef _WIN32
	void*															fhandle;											///< File handle
#else
	int															fhandle;											///< File descriptor
#endif

private:
	//============================================================================================================================
	// Data members - Stream state
	//============================================================================================================================
	std::vector<unsigned char>								header;											///< Header (cache)
	std::vector<unsigned char>								metadata;										///< Dataset metdata (cache)
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
};

