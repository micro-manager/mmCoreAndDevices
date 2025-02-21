﻿///////////////////////////////////////////////////////////////////////////////
// FILE:          G2STiffFile.cpp
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
#define _LARGEFILE64_SOURCE
#include <sstream>
#include <filesystem>
#include <cstring>
#include "G2STiffFile.h"
#ifdef _WIN32
#include <Windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/fs.h>
#endif

/**
 * Class constructor
 * Constructor doesn't open the file, just creates an object set sets the configuration
 * @param path File path
 * @param fbig Use BigTIFF format
 */
G2STiffFile::G2STiffFile(const std::string& path, bool fbig) noexcept
{
	fpath = path;
	datasetuid = "";
	ssize = 0;
	bitdepth = 8;
	samples = 1;
	imgcounter = 0;
	currentimage = 0;
#ifdef _WIN32
	fhandle = nullptr;
#else
	fhandle = 0;
#endif
	directIo = false;
	bigTiff = fbig;
	configset = false;
	currpos = 0;
	writepos = 0;
	readpos = 0;
	lastifdpos = 0;
	lastifdsize = 0;
	currentifdpos = 0;
	currentifdsize = 0;
	nextifdpos = 0;
	readbuffoff = 0;
}

/**
 * Open / create a file
 * File path is optional (if it's already been set)
 * If the file doesn't exist it will be created (write mode)
 * If the file exists and 'trunc' is set to true existing file will be discared and new one will be created (write mode)
 * If the file exists and 'trunc' is set to false dataset shape, pixel format and metadata will be parsed (read / append mode)
 * @param trunc Trucate existing file
 * @param dio Use direct I/O
 * @throws std::runtime_error
 */
void G2STiffFile::open(bool trunc, bool dio)
{
	if(isOpen())
		throw std::runtime_error("Invalid operation. File stream is already open");
	if(fpath.empty())
		throw std::runtime_error("Unable to open a file stream. File path is undefined");
	directIo = dio;
	auto fexists = std::filesystem::exists(std::filesystem::u8path(fpath));
	auto xpath = std::filesystem::u8path(fpath);

	// Obtain a file handle
#ifdef _WIN32
	// Convert file path
	int slength = (int)fpath.length() + 1;
	int len = MultiByteToWideChar(CP_UTF8, 0, fpath.c_str(), slength, nullptr, 0);
	wchar_t* buf = new wchar_t[len];
	MultiByteToWideChar(CP_UTF8, 0, fpath.c_str(), slength, buf, len);

	// Open a file 
	DWORD cattr = trunc ? CREATE_ALWAYS : OPEN_ALWAYS;
	DWORD fattr = directIo ? FILE_FLAG_WRITE_THROUGH | FILE_FLAG_NO_BUFFERING : 0;
	fhandle = CreateFile(buf, GENERIC_WRITE | GENERIC_READ, 0, NULL, cattr, FILE_ATTRIBUTE_NORMAL | fattr, NULL);
	if(fhandle == INVALID_HANDLE_VALUE)
	{
		delete[] buf;
		fhandle = nullptr;
		throw std::runtime_error("File '" + fpath + "' open failed, error code " + std::to_string(GetLastError()));
	}
	delete[] buf;
#else
	int flags = O_LARGEFILE | O_RDWR | O_SYNC | (trunc ? O_CREAT | O_TRUNC : 0) | (directIo ? O_DIRECT : 0);
	fhandle = ::open(fpath.c_str(), flags);
	if(fhandle < 0)
	{
		fhandle = 0;
		throw std::runtime_error("File '" + fpath + "' open failed, error code " + std::to_string(errno));
	}
#endif

	// Obtain a block size
	if(ssize == 0)
	{
#ifdef _WIN32
		auto apath = std::filesystem::absolute(xpath);
		auto dsepind = apath.u8string().find_first_of('\\');	
		auto drivepath = dsepind == std::string::npos ? apath.u8string() : apath.u8string().substr(0, dsepind + 1);
		slength = (int)drivepath.length() + 1;
		len = MultiByteToWideChar(CP_UTF8, 0, drivepath.c_str(), slength, nullptr, 0);
		wchar_t* dbuf = new wchar_t[len];
		MultiByteToWideChar(CP_UTF8, 0, drivepath.c_str(), slength, dbuf, len);

		GetDiskFreeSpace(dbuf, NULL, (LPDWORD)&ssize, NULL, NULL);
		delete[] dbuf;
#elif __linux__
		size_t blockSize = 0;
		ioctl(fhandle, BLKSSZGET, &blockSize);
		ssize = (std::uint32_t)blockSize;
#elif __APPLE__
		size_t blockSize = 0;
		ioctl(fhandle, DKIOCGETPHYSICALBLOCKSIZE, &blockSize);
		ssize = (std::uint32_t)blockSize;
#endif
	}
	if(directIo)
		writebuff.reserve(ssize);

	bool freshfile = !fexists || trunc;
	currpos = 0;
	writepos = 0;
	readpos = 0;
	lastifdpos = 0;
	lastifdsize = 0;
	currentifdpos = 0;
	currentifdsize = 0;
	nextifdpos = 0;
	currentimage = 0;
	readbuffoff = 0;
	header.clear();
	ifdcache.clear();

	if(freshfile)
	{
		// New / empty file / write mode
		// Prepare a file header
		// File header will be commited when the first image is added
		formHeader();
	}
	else
	{
		// Existing file / read mode
		// Parse file header
		header.resize(G2STIFF_HEADER_SIZE);
		metadata.clear();
		datasetuid = "";
		auto nb = fetch(&header[0], header.size());
		if(nb == 0)
		{
			// Empty file detected -> form header / write mode
			formHeader();
		}
		else 
		{
			if(nb != header.size())
				throw std::runtime_error("File '" + fpath + "' open failed. File header is missing");

			// Check TIFF format header
			if(header[0] != 0x49 || header[1] != 0x49 || (header[2] != 0x2a && header[2] != 0x2b))
				throw std::runtime_error("File '" + fpath + "' open failed. Unsupported file format");
			if(header[2] == 0x2a)
			{
				// Regular TIFF format detected
				if(header[3] != 0)
					throw std::runtime_error("File '" + fpath + "' open failed. Unsupported file format");
				// Check G2S Format header
				if(header[8] != 0x3c || header[9] != 0x1d || header[10] != 0x59 || header[11] != 0x69)
					throw std::runtime_error("File '" + fpath + "' open failed. Unsupported file format");
				if(header[12] != 0x00 || header[13] != 0x00 || header[14] != 0x00 || header[15] != 0x00)
					throw std::runtime_error("File '" + fpath + "' open failed. Unsupported file format");
				bigTiff = false;
			}
			else 
			{
				// BigTIFF format detected
				if(header[4] != 0x08 || header[5] != 0x00 || header[6] != 0x00 || header[7] != 0x00)
					throw std::runtime_error("File '" + fpath + "' open failed. Unsupported file format");
				// Check G2S Format header
				if(header[16] != 0x3c || header[17] != 0x1d || header[18] != 0x59 || header[19] != 0x69)
					throw std::runtime_error("File '" + fpath + "' open failed. Unsupported file format");
				if(header[20] != 0x00 || header[21] != 0x00 || header[22] != 0x00 || header[23] != 0x00)
					throw std::runtime_error("File '" + fpath + "' open failed. Unsupported file format");
				bigTiff = true;
			}

			// Parse dataset UID
			auto sind = bigTiff ? 24 : 16;
			for(std::size_t i = 0; i < 16; i++)
			{
				if(i == 4 || i == 6 || i == 8 || i == 10)
					datasetuid += "-"; 
				char dig1 = (header[sind + i] & 0xf0) >> 4;
				char dig2 = (header[sind + i] & 0x0f);
				if(0 <= dig1 && dig1 <= 9)
					dig1 += 48;
				if(10 <= dig1 && dig1 <= 15)
					dig1 += 97 - 10;
				if(0 <= dig2 && dig2 <= 9)
					dig2 += 48;
				if(10 <= dig2 && dig2 <= 15)
					dig2 += 97 - 10;
				datasetuid.append(&dig1, 1);
				datasetuid.append(&dig2, 1);
			}
			if(datasetuid == "00000000-0000-0000-0000-000000000000")
				datasetuid = "";

			// Parse shape data
			imgcounter = (std::uint32_t)readInt(&header[bigTiff ? 48 : 36], 4);
			auto shapedim = (std::uint32_t)readInt(&header[bigTiff ? 52 : 40], 4);
			shape.clear();
			for(std::uint32_t i = 0; i < shapedim; i++)
				shape.push_back((std::uint32_t)readInt(&header[(bigTiff ? 56 : 44) + i * 4], 4));

			// Get file size
			auto fsize = std::filesystem::file_size(xpath);

			// Parse metadata
			auto metaoffset = readInt(&header[bigTiff ? 40 : 32], bigTiff ? 8 : 4);
			seek(metaoffset);

			metadata.resize(fsize - metaoffset);
			fetch(&metadata[0], metadata.size());

			// Locate first IFD
			currentifdpos = readInt(&header[bigTiff ? 8 : 4], bigTiff ? 8 : 4);

			// Set write position at the metadata section start or at the end of the file stream
			writepos = metaoffset == 0 ? fsize : metaoffset;

			// Locate last IFD
			bool pixformatset = false;
			if(currentifdpos > 0)
			{
				seek(currentifdpos);
				ifdcache.push_back(currentifdpos);

				int i = 0;
				std::vector<unsigned char> lbuff;
				lbuff.reserve(G2STIFF_HEADER_SIZE);
				while(true)
				{
					lbuff.clear();
					lbuff.resize(8);
					nb = fetch(&lbuff[0], lbuff.size());
					if(nb == 0)
						throw std::runtime_error("File '" + fpath + "' open failed. IFD " + std::to_string(i) + " is corrupted");
					std::uint32_t tagcount = (std::uint32_t)readInt(&lbuff[0], bigTiff ? 8 : 2);
					
					std::uint32_t ifdsz = 0, basesz = 0;
					calcDescSize(0, tagcount, &ifdsz, &basesz, nullptr);
					lbuff.resize(basesz);
					nb = fetch(&lbuff[8], lbuff.size() - 8);

					// Obtain pixel format from the first IFD)
					if(!pixformatset)
					{
						bitdepth = (uint8_t)readInt(&lbuff[bigTiff ? 60 : 34], 2);
						pixformatset = true;
					}

					auto nextoffset = readInt(&lbuff[ifdsz - (bigTiff ? 8 : 4)], bigTiff ? 8 : 4);
					if(nextoffset >= fsize)
						throw std::runtime_error("File '" + fpath + "' open failed. IFD " + std::to_string(i) + " link is corrupted");
					if(nextoffset == 0)
						break;
					lastifdpos = nextoffset;
					lastifdsize = ifdsz;
					ifdcache.push_back(lastifdpos);
					seek(lastifdpos);
					i++;
				}
			}

			// Rewind read cursor
			seek(currentifdpos);
			moveReadCursor(currentifdpos);

			configset = true;
		}
	}
	nextifdpos = currentifdpos;
}

/**
 * Close / commit a file stream
 * If a file hasn't been open this method will have no effect
 * File handle will be released / closed
 * In the write / append mode during closing final section (dataset metadat) is commited to file
 * File header is also updated with the offset of the final section
 * File opened for writing, but without any images should be empty after closing
 */
void G2STiffFile::close() noexcept
{
	if(!isOpen())
		return;
	
	std::uint64_t fsize = 0;
	if(writepos > 0 && lastifdsize > 0 && !lastifd.empty())
	{
		// This section is skipped if file is empty, or no write operation has been peformed
		try
		{
			// Reposition file cursor if last operation was a file read
			if(writepos != currpos)
				seek(writepos);

			// Commit final sections
			// Write metadata
			fsize = writepos + metadata.size();
			if(!metadata.empty())
			{
				writeInt(&header[bigTiff ? 40 : 32], bigTiff ? 8 : 4, writepos);
				commit(&metadata[0], metadata.size());
			}

			// Commit last sector
			if(directIo && !writebuff.empty())
			{
				auto padsize = ssize - writebuff.size();
				std::vector<unsigned char> pbuff(padsize);
				commit(&pbuff[0], pbuff.size());
			}

			// Clear last IFD chain offset
			if(bigTiff)
				writeInt(&lastifd[lastifdsize - 8], 8, 0);
			else
				writeInt(&lastifd[lastifdsize - 4], 4, 0);
			seek(lastifdpos);
			commit(&lastifd[0], lastifd.size());

			// Update file header
			// Set image count
			writeInt(&header[bigTiff ? 48 : 36], 4, imgcounter);
			seek(0);
			commit(&header[0], header.size());

			// Move cursor at the end of the file stream
			seek(0, false);
		}
		catch(...) { }

		// Set actual file size limit for direct I/O
		if(directIo)
		{
#ifdef _WIN32
			FILE_END_OF_FILE_INFO inf = {};
			inf.EndOfFile.QuadPart = (LONGLONG)fsize;
			SetFileInformationByHandle((HANDLE)fhandle, FileEndOfFileInfo, &inf, sizeof(inf));
#else
			// TODO
#endif
		}
	}

	// Close the file
#ifdef _WIN32
	CloseHandle((HANDLE)fhandle);
	fhandle = nullptr;
#else
	::close(fhandle);
	fhandle = 0;
#endif
	imgcounter = 0;
	currentimage = 0;
	currpos = 0;
	writepos = 0;
	readpos = 0;
	lastifdpos = 0;
	lastifdsize = 0;
	currentifdpos = 0;
	currentifdsize = 0;
	nextifdpos = 0;
	readbuffoff = 0;
	bitdepth = 8;
	samples = 1;
	configset = false;

	writebuff.clear();
	readbuff.clear();
	lastifd.clear();
	currentifd.clear();
	header.clear();
	metadata.clear();
	shape.clear();
	ifdcache.clear();
}

/**
 * Set dataset shape / dimension & axis sizes
 * First two axis are always width and height
 * If the shape info is invalid, or the dataset configuration has already been set
 * this method will take no effect
 * @param dims Axis sizes list
 * @throws std::runtime_error
 */
void G2STiffFile::setShape(const std::vector<std::uint32_t>& dims)
{
	if(dims.size() < 2)
		throw std::runtime_error("Unable to set dataset shape. Invalid shape info");
	if(configset && imgcounter > 0 && shape.size() >= 2)
	{
		if(dims[0] != shape[0] || dims[1] != shape[1])
			throw std::runtime_error("Unable to set dataset shape. Image dimensions don't match the existing image dimensions");
	}
	shape = dims;
	
	// Write Shape info to the header cache
	writeShapeInfo();
}

/**
 * Set dataset shape / dimension & axis sizes
 * First two axis are always width and height
 * If the shape info is invalid, or the dataset configuration has already been set
 * this method will take no effect
 * @param dims Axis sizes list
 * @throws std::runtime_error
 */
void G2STiffFile::setShape(std::initializer_list<std::uint32_t> dims)
{
	if(dims.size() < 2)
		throw std::runtime_error("Unable to set dataset shape. Invalid shape info");
	if(configset && imgcounter > 0 && shape.size() >= 2)
	{
		if(*dims.begin() != shape[0] || *(dims.begin() + 1) != shape[1])
			throw std::runtime_error("Unable to set dataset shape. Image dimensions don't match the existing image dimensions");
	}
	shape = dims;

	// Write Shape info to the header cache
	writeShapeInfo();
}

/**
 * Set pixel format
 * If the shape info is invalid, or the dataset configuration has already been set
 * this method will take no effect
 * @param depth Bit depth (bits per sample)
 * @parma vsamples Samples per pixel
 * @throws std::runtime_error
 */
void G2STiffFile::setPixelFormat(std::uint8_t depth, std::uint8_t vsamples)
{
	if(configset && imgcounter > 0)
	{
		if(bitdepth != depth || samples != vsamples)
			throw std::runtime_error("Unable to set pixel format. Specified pixel format doesn't match current pixel format");
	}
	bitdepth = depth;
	samples = vsamples;
}

/**
 * Set dataset metadata
 * Metadata will be stored in byte buffer whose size is 1 byte larger than the metadata string length
 * @param meta Metadata string
 */
void G2STiffFile::setMetadata(const std::string& meta) noexcept
{
	metadata.clear();
	if(meta.empty())
		return;
	metadata.resize(meta.size() + 1);
	std::copy(meta.begin(), meta.end(), metadata.begin());
}

/**
 * Set dataset UID
 * UID must be in a standard UUID format, 16-bytes long hex string with or without the dash delimiters: 
 * XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
 * @param val Dataset UID
 * @throws std::runtime_error
 */
void G2STiffFile::setUID(const std::string& val)
{
	if(val.empty())
	{
		datasetuid = val;
		return;
	}
	if(val.size() != 32 && val.size() != 36)
		throw std::runtime_error("Unable to set the dataset UID. Invalid UID format");
	auto hasdashes = val.size() == 36;
	if(hasdashes && (val[8] != '-' || val[13] != '-' || val[18] != '-' || val[23] != '-'))
		throw std::runtime_error("Unable to set the dataset UID. Invalid UID format");
	for(std::size_t i = 0; i < val.size(); i++)
	{
		if(hasdashes && (i == 8 || i == 13 || i == 18 || i == 23))
			continue;
		if(val[i] < 48 || val[i] > 102 || (val[i] > 57 && val[i] < 65) || (val[i] > 70 && val[i] < 97))
			throw std::runtime_error("Unable to set the dataset UID. Invalid UID format");
	}
	datasetuid = hasdashes ? val : val.substr(0, 8) + "-" + val.substr(8, 4) + "-" + val.substr(12, 4) + "-" + val.substr(16, 4) + "-" + val.substr(20);
	if(!header.empty())
	{
		// Write UID to the header cache
		auto startind = bigTiff ? 24 : 16;
		auto cind = 0;
		for(int i = 0; i < 16; i++)
		{
			if(i == 4 || i == 6 || i == 8 || i == 10)
				cind++;
			char cv1 = datasetuid[cind++];
			char cv2 = datasetuid[cind++];
			std::uint8_t vx1 = cv1 >= 48 && cv1 <= 57 ? cv1 - 48 : (cv1 >= 65 && cv1 <= 70 ? cv1 - 55 : cv1 - 87);
			std::uint8_t vx2 = cv2 >= 48 && cv2 <= 57 ? cv2 - 48 : (cv2 >= 65 && cv2 <= 70 ? cv2 - 55 : cv2 - 87);
			auto xval = (std::uint8_t)(((vx1 & 0x0f) << 4) | (vx2 & 0x0f));
			header[startind + i] = xval;
		}
	}
}

/**
 * Get dataset metadata
 * If metadata is specified value will be returned from cache, otherwise it will be read from a file stream
 * @return Metadata string
 * @throws std::runtime_error
 */
std::string G2STiffFile::getMetadata()
{
	// Check metadata cache
	if(metadata.empty())
		return "";
	std::string str(metadata.begin(), metadata.end() - 1);
	return str;
}

/**
 * Get image metadata
 * If the coordinates are not specified images are read sequentially, metadata for the current image 
 * will be returned, in which case the current image won't be changed
 * If no metadata is defined this method will return an empty string
 * If no images are defined this method will return an empty string
 * In the sequential mode the image IFD will be loaded if this method is called before getImage() (only for the first image)
 * For other images getImage() should always be called prior to calling getImageMetadata()
 * @param coord Image coordinates
 * @return Image metadata
 */
std::string G2STiffFile::getImageMetadata(const std::vector<std::uint32_t>& coord)
{
	if(!isOpen())
		throw std::runtime_error("Invalid operation. No open file stream available");
	if(imgcounter == 0)
		throw std::runtime_error("Invalid operation. No images available");
	
	// Select current image (IFD)
	if(!coord.empty())
	{
		auto ind = calcImageIndex(coord);
		if(ind >= ifdcache.size())
			throw std::runtime_error("Invalid operation. Invalid image coordinates");
		currentimage = ind;
		loadIFD(ifdcache[ind]);
	}
	else if(currentifd.empty())
		// Load IFD
		loadIFD(currentifdpos);

	// Check IFD tag count
	auto tagcount = readInt(&currentifd[0], bigTiff ? 8 : 2);
	if(tagcount == G2STIFF_TAG_COUNT_NOMETA)
		return "";

	// Obtain metadata OFFSET and length
	auto metatagind = (bigTiff ? 8 : 2) + G2STIFF_TAG_COUNT_NOMETA * (bigTiff ? 20 : 12);
	auto metalen = readInt(&currentifd[metatagind + 4], bigTiff ? 8 : 4);
	auto metaoffset = readInt(&currentifd[metatagind + (bigTiff ? 12 : 8)], bigTiff ? 8 : 4);
	if(metalen == 0 || metaoffset == 0)
		return "";
	if(metaoffset < currentifdpos)
		throw std::runtime_error("Unable to obtain image metadata. File is corrupted");

	// Copy metadata from the IFD
	auto roff = metaoffset - currentifdpos;
	auto strlen = roff + metalen > currentifd.size() ? currentifd.size() - roff - metalen : metalen - 1;
	std::string str(&currentifd[roff], &currentifd[roff + strlen]);
	return str;
}

/**
 * Add image / write image to the file
 * Images are added sequentially
 * Image data is stored uncompressed
 * Metadata is stored in plain text, after the pixel data
 * Image IFD is stored before pixel data
 * @param buff Image buffer
 * @param meta Image metadata (optional)
 * @throws std::runtime_error
 */
void G2STiffFile::addImage(const unsigned char* buff, std::size_t len, const std::string& meta)
{
	if(!isOpen())
		throw std::runtime_error("Invalid operation. No open file stream available");
	if(shape.size() < 2)
		throw std::runtime_error("Invalid operation. Dataset shape is not defined");
	if(!bigTiff && len > TIFF_MAX_BUFFER_SIZE)
		throw std::runtime_error("Invalid operation. Image data is too long");
	if(!bigTiff && meta.size() > TIFF_MAX_BUFFER_SIZE)
		throw std::runtime_error("Invalid operation. Metadata string is too large");

	// Check file size limits
	std::uint32_t tot = 0;
	calcDescSize(meta.empty() ? 0 : meta.size() + 1, getTagCount(meta), nullptr, nullptr, &tot);
	if(meta.size() + len + currpos + tot > getMaxFileSize())
		throw std::runtime_error("Invalid operation. File size limit exceeded");

	// Commit header if empty file
	if(writepos == 0)
	{
		commit(&header[0], header.size());
		lastifdpos = readInt(&header[bigTiff ? 8 : 4], bigTiff ? 8 : 4);
		configset = true;
	}
	// Update last IFD for images in read mode
	else if(lastifd.empty() && lastifdpos > 0)
	{
		// Move read cursor to the last IFD
		auto lreadpos = readpos;
		auto lwritepos = writepos;
		seek(lastifdpos);
		moveReadCursor(currpos);

		// Load last IFD and change the next IFD offset
		auto nextoff = parseIFD(lastifd, lastifdsize);
		if(nextoff == 0)
			writeInt(&lastifd[lastifdsize - (bigTiff ? 8 : 4)], bigTiff ? 8 : 4, writepos);

		// Update last IFD
		seek(lastifdpos);
		commit(&lastifd[0], lastifd.size());

		// Reset cursors
		moveReadCursor(lreadpos);
		moveWriteCursor(lwritepos);
	}

	// Reposition file cursor if last operation was a file read
	if(writepos != currpos)
		seek(writepos);

	// Compose next IFD and write image metadata
	appendIFD(len, meta);

	// Write pixel data
	commit(buff, len);

	// Add padding bytes
	auto alignsz = directIo ? ssize : 2;
	if(len % alignsz != 0)
	{
		auto padsize = len - (len / alignsz) * alignsz;
		std::vector<unsigned char> pbuff(padsize);
		commit(&pbuff[0], pbuff.size());
	}
	ifdcache.push_back(lastifdpos);
	imgcounter++;
}

/**
 * Get image data (pixel buffer)
 * If the coordinates are not specified images are read sequentially
 * This method will change (advance) the current image
 * If this method is called after the last available image (in sequential mode), or with invalid coordinates an exception will be thrown
 * @param coord Image coordinates
 * @return Image data
 */
std::vector<unsigned char> G2STiffFile::getImage(const std::vector<std::uint32_t>& coord)
{
	if(!isOpen())
		throw std::runtime_error("Invalid operation. No open file stream available");
	if(imgcounter == 0 || (currentimage + 1 > imgcounter) || nextifdpos == 0)
		throw std::runtime_error("Invalid operation. No images available");

	// Select current image (IFD)
	if(!coord.empty())
	{
		auto ind = calcImageIndex(coord);
		if(ind >= ifdcache.size())
			throw std::runtime_error("Invalid operation. Invalid image coordinates");
		currentimage = ind;
		loadIFD(ifdcache[ind]);
	}
	else
	{
		// Clear current IFD before advancing
		// In a case where getImageMetadata() is called before any getImage() call
		// we should skip clearing of current IFD, this works only for the first image
		if(currentimage > 0)
		{
			currentifd.clear();
			currentifdpos = nextifdpos;
		}

		// Advance current image
		currentimage++;

		// Load IFD (skip if already loaded by the getImageMetadata())
		if(currentifd.empty())
			loadNextIFD();
	}

	// Obtain pixel data strip locations
	auto offind = (bigTiff ? 8 : 2) + 5 * (bigTiff ? 20 : 12);
	auto lenind = (bigTiff ? 8 : 2) + 7 * (bigTiff ? 20 : 12);
	auto dataoffset = readInt(&currentifd[offind + (bigTiff ? 12 : 8)], bigTiff ? 8 : 4);
	auto datalen = readInt(&currentifd[lenind + (bigTiff ? 12 : 8)], bigTiff ? 8 : 4);
	if(dataoffset == 0 || datalen == 0)
		return {};

	std::vector<unsigned char> ret(datalen);
	moveReadCursor(seek(dataoffset));
	fetch(&ret[0], ret.size());
	return ret;
}

/**
 * Get file size
 * @return File size in bytes
 */
std::uint64_t G2STiffFile::getFileSize() const noexcept
{
	if(isOpen())
	{
#ifdef _WIN32
		LARGE_INTEGER fsize = {};
		auto sc = GetFileSizeEx((HANDLE)fhandle, &fsize);
		if(sc == 0)
			return 0;
		return (std::uint64_t)fsize.QuadPart;
#else
		auto sz = lseek(fhandle, 0L, SEEK_END);
		lseek(fhandle, (off_t)currpos, SEEK_SET);
		return (std::uint64_t)sz;
#endif
	}
	if(fpath.empty())
		return 0;
	return std::filesystem::file_size(std::filesystem::u8path(fpath));
}

/**
 * Send data to be written to the file stream
 * if direct I/O is used for small data sizes data will be stored in the temporary buffer 
 * For cached I/O and large buffer sizes for direct I/O data is written to the file stream
 * @param buff Source buffer
 * @param len Source buffer length
 * @return Number of bytes written
 * @throws std::runtime_error
 */
std::size_t G2STiffFile::commit(const unsigned char* buff, std::size_t len)
{
	if(!isOpen())
		throw std::runtime_error("File write failed. No valid file stream available");
	if(directIo)
	{
		// Clear write cache if file cursor has been moved
		if(currpos != writepos)
			writebuff.clear();

		// Direct I/O - Buffer size must be a multiple of a disk sector size (usually 512 B)
		std::size_t ret = 0;

		// Fill pending data with the first sector / write pending data
		std::size_t boffset = 0;
		if(!writebuff.empty())
		{
			auto curr = writebuff.size();
			auto cnt = curr + len > ssize ? ssize - curr : len;
			writebuff.resize(writebuff.size() + cnt);
			std::memcpy(&writebuff[curr], buff, cnt);
			boffset += cnt;
			if(writebuff.size() == ssize)
			{
				ret += write(&writebuff[0], ssize);
				writebuff.clear();
			}
			else
				return ret;
		}
		if(len - boffset == 0)
			return ret;

		// Write middle sectors directly
		if(len - boffset >= ssize)
		{
			auto wcnt = ((len - boffset) / ssize) * ssize;
			ret += write(buff + boffset, wcnt);
			boffset += wcnt;
		}

		// Write remaining data (last sector) to the pending data buffer
		// Pending data buffer should be empty at this point
		if(len - boffset > 0)
		{
			auto cpos = writebuff.size();
			writebuff.resize(len - boffset);
			std::memcpy(&writebuff[cpos], &buff[boffset], len - boffset);
		}
		return ret;
	}
	else
		// Standard - Cached I/O
		return write(buff, len);
}

/**
 * Write data to the file stream
 * @param buff Source buffer
 * @param len Source buffer length
 * @return Number of bytes written
 * @throws std::runtime_error
 */
std::size_t G2STiffFile::write(const unsigned char* buff, std::size_t len)
{	
	std::size_t pos = 0;
	std::size_t ret = 0;
	while(pos < len)
	{
		std::uint32_t trans = len - pos < std::numeric_limits<std::uint32_t>::max() ? (std::uint32_t)(len - pos) : std::numeric_limits<std::uint32_t>::max();
#ifdef _WIN32
		DWORD nb = 0;
		auto sc = WriteFile((HANDLE)fhandle, (LPCVOID)&buff[pos], (DWORD)trans, &nb, NULL);
		if(sc == FALSE)
			throw std::runtime_error("File write failed. Error code: " + std::to_string(GetLastError()));
		currpos += nb;
#else
		auto nb = ::write(fhandle, &buff[pos], trans);
		if(nb < 0)
			throw std::runtime_error("File write failed. Error code: " + std::to_string(errno));
		currpos += (std::uint64_t)nb;
#endif
		ret += nb;
		pos += trans;
	}
	writepos = currpos;
	return ret;
}

/**
 * Fetch data from the file stream
 * if direct I/O is used for small data sizes entire block is read from file and stored in the temporary buffer
 * For cached I/O and large buffer sizes for direct I/O data is read from the file stream
 * @param buff Destination buffer
 * @param len Destination buffer length
 * @return Number of bytes read
 * @throws std::runtime_error
 */
std::size_t G2STiffFile::fetch(unsigned char* buff, std::size_t len)
{
	if(!isOpen())
		throw std::runtime_error("File read failed. No valid file stream available");
	if(directIo)
	{
		// Clear write cache if file cursor has been moved
		if(currpos != readpos)
		{
			readbuff.clear();
			readbuffoff = 0;
		}

		// Direct I/O - Buffer size must be a multiple of a disk sector size (usually 512 B)
		std::size_t ret = 0;

		// Fetch pending data before requesting another read operation
		std::size_t boffset = 0;
		if(!readbuff.empty())
		{
			std::size_t cnt = (readbuff.size() - readbuffoff) >= len ? len : (readbuff.size() - readbuffoff);
			std::memcpy(buff, &readbuff[readbuffoff], cnt);
			readbuffoff += cnt;
			if(readbuffoff == readbuff.size())
			{
				readbuff.clear();
				readbuffoff = 0;
			}
			boffset += cnt;
		}

		// Check if all data has been fetched and if not
		if(len - boffset == 0)
			return ret;

		// Read middle sectors directly and prefetch last sector
		if(len - boffset >= ssize)
		{
			auto rcnt = ((len - boffset) / ssize) * ssize;
			ret += read(&buff[boffset], rcnt);
			boffset += rcnt;
		}

		// Prefetch last sector
		if(len - boffset > 0)
		{
			readbuff.resize(ssize);
			readbuffoff = 0;
			ret += read(&readbuff[0], ssize);
			std::size_t cnt = len - boffset;
			std::memcpy(&buff[boffset], &readbuff[0], cnt);
			readbuffoff += cnt;
		}
		return ret;
	}
	else
		return read(buff, len);
}

/**
 * Read data from the file stream
 * @param buff Destination buffer
 * @param len Destination buffer length
 * @return Number of bytes read
 * @throws std::runtime_error
 */
std::size_t G2STiffFile::read(unsigned char* buff, std::size_t len)
{
	if(!isOpen())
		return 0;
	std::size_t pos = 0;
	std::size_t ret = 0;
	while(pos < len)
	{
		std::uint32_t trans = len - pos < std::numeric_limits<std::uint32_t>::max() ? (std::uint32_t)(len - pos) : std::numeric_limits<std::uint32_t>::max();
#ifdef _WIN32
		DWORD nb = 0;
		auto sc = ReadFile((HANDLE)fhandle, (LPVOID)&buff[pos], trans, &nb, NULL);
		if(sc == FALSE)
			throw std::runtime_error("File read failed. Error code: " + std::to_string(GetLastError()));
		currpos += nb;
#else
		auto nb = ::read(fhandle, &buff[pos], trans);
		if(nb < 0)
			throw std::runtime_error("File read failed. Error code: " + std::to_string(errno));
		currpos += (std::uint64_t)nb;
#endif
		ret += nb;
		pos += trans;
	}
	readpos = currpos;
	return ret;
}

/**
 * Set cursor position (position in the file stream)
 * Moving the cursor will clear temporary buffer (for direct I/O)
 * @param pos Position / Offset
 * @param beg Use stream begining as a base point (or stream end)
 * @return Current position
 * @throws std::runtime_error
 */
std::uint64_t G2STiffFile::seek(std::int64_t pos, bool beg)
{
	if(!isOpen())
		throw std::runtime_error("File seek failed. No valid file stream available");
	if(beg && pos < 0)
		throw std::runtime_error("File seek failed. Invalid file position");
	if(beg && (std::uint64_t)pos == currpos)
		return currpos;
#ifdef _WIN32
	LARGE_INTEGER li = {};
	li.QuadPart = pos;
	auto ret = SetFilePointer((HANDLE)fhandle, li.LowPart, &li.HighPart, beg ? FILE_BEGIN : FILE_END);
	if(ret == INVALID_SET_FILE_POINTER)
		throw std::runtime_error("File seek failed. Error code: " + std::to_string(GetLastError()));
	ULARGE_INTEGER ri = {};
	ri.LowPart = ret;
	ri.HighPart = (DWORD)li.HighPart;
	currpos = ri.QuadPart;
#else
	auto ret = lseek64(fhandle, (off64_t)pos, beg ? SEEK_SET : SEEK_END);
	if(ret < 0)
		throw std::runtime_error("File seek failed. Error code: " + std::to_string(errno));
	currpos = (std::uint64_t)ret;
#endif
	return currpos;
}

/**
 * Advance cursor position in the relation to the current cursor position
 * Moving the cursor will clear temporary buffer (for direct I/O)
 * @param pos Offset
 * @return Current position
 * @throws std::runtime_error
 */
std::uint64_t G2STiffFile::offset(std::int64_t off)
{
	if(!isOpen())
		throw std::runtime_error("File seek failed. No valid file stream available");
	if(off == 0)
		return currpos;
#ifdef _WIN32
	LARGE_INTEGER li = {};
	li.QuadPart = off;
	auto ret = SetFilePointer((HANDLE)fhandle, li.LowPart, &li.HighPart, FILE_CURRENT);
	if(ret == INVALID_SET_FILE_POINTER)
		throw std::runtime_error("File seek failed. Error code: " + std::to_string(GetLastError()));
	ULARGE_INTEGER ri = {};
	ri.LowPart = ret;
	ri.HighPart = (DWORD)li.HighPart;
	currpos = ri.QuadPart;
#else
	auto ret = lseek64(fhandle, (off64_t)off, SEEK_CUR);
	if(ret < 0)
		throw std::runtime_error("File seek failed. Error code: " + std::to_string(errno));
	currpos = (std::uint64_t)ret;
#endif
	return currpos;
}

/**
 * Construct header cache (write mode only)
 */
void G2STiffFile::formHeader() noexcept
{
	header.clear();
	header.resize(G2STIFF_HEADER_SIZE);
	if(bigTiff)
	{
		// BigTIFF file header
		writeInt(&header[0], 4, 0x002b4949);
		writeInt(&header[4], 4, 0x08);

		// Set first IFD to 0x00 00 02 00 (512)
		writeInt(&header[8], 8, G2STIFF_HEADER_SIZE);

		// Write G2STIFF format signature
		writeInt(&header[16], 8, 0x69591d3c);
	}
	else
	{
		// TIFF file header
		writeInt(&header[0], 4, 0x002a4949);

		// Set first IFD to 0x00 00 02 00 (512)
		writeInt(&header[4], 4, G2STIFF_HEADER_SIZE);

		// Write G2STIFF format signature
		writeInt(&header[8], 8, 0x69591d3c);
	}

	// Write shape info
	writeShapeInfo();

	currentifdpos = G2STIFF_HEADER_SIZE;
	lastifdpos = G2STIFF_HEADER_SIZE;
}

/**
 * Form IFD and write it to file stream at the current position
 * Metadata is placed immediately after pixel data
 * Image / Matadata offsets  calculated automatically
 * Image sections are block aligned
 * @param imagelen Pixel data length
 * @param meta Image metadata
 * @throws std::runtime_error
 */
void G2STiffFile::appendIFD(std::size_t imagelen, const std::string& meta)
{
	std::uint32_t descsz = 0, totsz = 0, tagcnt = getTagCount(meta);
	auto actimglen = imagelen + (imagelen % ssize == 0 ? 0 : (ssize - (imagelen % ssize)));
	calcDescSize(meta.empty() ? 0 : meta.size() + 1, tagcnt, &lastifdsize, &descsz, &totsz);
	lastifdpos = currpos;
	lastifd.clear();
	lastifd.resize(totsz);

	// Set TAG count
	writeInt(&lastifd[0], bigTiff ? 8 : 2, tagcnt);

	// Add TAGS
	std::size_t ind = bigTiff ? 8 : 2;
	ind += setIFDTag(&lastifd[ind], 0x0100, 4, getWidth());
	ind += setIFDTag(&lastifd[ind], 0x0101, 4, getHeight());
	ind += setIFDTag(&lastifd[ind], 0x0102, 3, bitdepth);
	ind += setIFDTag(&lastifd[ind], 0x0103, 3, 1);
	ind += setIFDTag(&lastifd[ind], 0x0106, 3, 1);
	ind += setIFDTag(&lastifd[ind], 0x0111, bigTiff ? 16 : 4, currpos + totsz);
	ind += setIFDTag(&lastifd[ind], 0x0116, 4, getHeight());
	ind += setIFDTag(&lastifd[ind], 0x0117, bigTiff ? 16 : 4, imagelen);
	if(bigTiff)
	{
		ind += setIFDTag(&lastifd[ind], 0x011a, 5, 0x0100000000);
		ind += setIFDTag(&lastifd[ind], 0x011b, 5, 0x0100000000);
	}
	else
	{
		ind += setIFDTag(&lastifd[ind], 0x011a, 5, currpos + lastifdsize);
		ind += setIFDTag(&lastifd[ind], 0x011b, 5, currpos + lastifdsize + 8);
	}
	ind += setIFDTag(&lastifd[ind], 0x0128, 3, 1);
	if(!meta.empty())
		ind += setIFDTag(&lastifd[ind], 0x010e, 2, currpos + descsz, meta.size() + 1);

	// Write next IFD offset
	std::uint64_t nextifd = currpos + totsz + actimglen;
	writeInt(&lastifd[ind], bigTiff ? 8 : 4, nextifd);

	// Write image resolution values
	if(!bigTiff)
	{
		writeInt(&lastifd[lastifdsize + 0], 8, 0x0100000000);
		writeInt(&lastifd[lastifdsize + 8], 8, 0x0100000000);
	}

	// Write metadata
	if(!meta.empty())
		std::copy(meta.begin(), meta.end(), lastifd.begin() + descsz);

	// Write IFD + metadata
	commit(&lastifd[0], lastifd.size());
}

/**
 * Load IFD for reading
 * Selected IFD will become the current IFD
 * Next IFD offset will be updated
 * If the last (written) IFD offset is the same as the selected IFD offset
 * no file operation will be performed (IFD will be copied from cache)
 * @throws std::runtime_error
 */
void G2STiffFile::loadIFD(std::uint64_t off)
{
	if(!currentifd.empty() && off == currentifdpos)
		return;
	if(off == 0)
	{
		// Reset current IFD
		currentifd.clear();
		currentifdpos = 0;
		currentifdsize = 0;
		nextifdpos = 0;
	}
	else if(!lastifd.empty() && lastifdpos == off)
	{
		// Copy IFD from cache
		currentifd = lastifd;
		currentifdpos = off;
		currentifdsize = lastifdsize;
		nextifdpos = readInt(&currentifd[currentifdsize - (bigTiff ? 8 : 4)], bigTiff ? 8 : 4);
	}
	else
	{
		// Load IFD from the file stream
		moveReadCursor(seek(off));
		currentifdpos = off;
		nextifdpos = parseIFD(currentifd, currentifdsize);
	}
}

/**
 * Parse IFD at the current read cursor
 * @param ifd IFD buffer [out]
 * @param ifdsz IFD size [out]
 * @return Next IFD offset
 * @throws std::runtime_error
 */
std::uint64_t G2STiffFile::parseIFD(std::vector<unsigned char>& ifd, std::uint32_t& ifdsz)
{
	if(readpos != currpos)
		seek(readpos);

	ifd.clear();
	ifd.resize(8);
	auto nb = fetch(&ifd[0], ifd.size());
	if(nb == 0)
		throw std::runtime_error("IFD loading failed. File is corrupted");

	// Obtain tag count
	std::uint32_t tagcount = (std::uint32_t)readInt(&ifd[0], bigTiff ? 8 : 2);
	std::uint32_t ifdsize = 0, basesz = 0, totsz = 0;
	calcDescSize(0, tagcount, nullptr, &basesz, nullptr);
	if(basesz <= 8)
		throw std::runtime_error("IFD loading failed. File is corrupted");

	// Load base IFD
	ifd.resize(basesz);
	fetch(&ifd[8], ifd.size() - 8);

	// Obtain image metadata length
	auto metatagind = (bigTiff ? 8 : 2) + G2STIFF_TAG_COUNT_NOMETA * (bigTiff ? 20 : 12);
	auto metalen = (std::size_t)readInt(&ifd[metatagind + 4], bigTiff ? 8 : 4);
	calcDescSize(metalen, tagcount, &ifdsize, &basesz, &totsz);

	// Load entire image descriptor (with metadata)
	ifd.resize(totsz);
	fetch(&ifd[basesz], ifd.size() - basesz);
	
	// Set current IFD offset to next IFD offset
	auto nextifd = readInt(&ifd[ifdsize - (bigTiff ? 8 : 4)], bigTiff ? 8 : 4);
	ifdsz = ifdsize;
	return nextifd;
}

/**
 * Set IFD field
 * Field data will be appended to the IFD buffer
 * @param ifd IFD buffer
 * @param tag Field tag
 * @param dtype Data type
 * @param val Field value / offset
 * @param cnt Value count
 * @return Number of bytes written
 */
std::size_t G2STiffFile::setIFDTag(unsigned char* ifd, std::uint16_t tag, std::uint16_t dtype, std::uint64_t val, std::uint64_t cnt) const noexcept
{
	// Write tag
	writeInt(&ifd[0], 2, tag);

	// Write data type
	writeInt(&ifd[2], 2, dtype);

	// Write value count
	writeInt(&ifd[4], bigTiff ? 8 : 4, cnt);

	// Write value / offset
	writeInt(&ifd[bigTiff ? 12 : 8], bigTiff ? 8 : 4, val);

	return bigTiff ? 20 : 12;
}

/**
 * Calculate image descriptor / IFD size
 * Image descriptor contains IFD, additional heap for large values (X/Y resolution), image metadata and padding bytes
 * Image descriptors are block aligned
 * @param metalen Image metadata length
 * @param tags Number of IFD tags
 * @param ifd IFD size [out]
 * @param desc Base descriptor (IFD + additional heap) size [out]
 * @parma tot Total image descriptor size [out]
 */
void G2STiffFile::calcDescSize(std::size_t metalen, std::uint32_t tags, std::uint32_t* ifd, std::uint32_t* desc, std::uint32_t* tot) noexcept 
{ 
	// Calculate IFD size
	std::uint32_t lifd = bigTiff ? 8 + tags * 20 + 8 : 2 + tags * 12 + 4;
	
	// Calculate base descriptor size (IFD + additional heap)
	auto heapsz = bigTiff ? 0 : 16;
	std::uint32_t ldesc = lifd + heapsz;

	// Calculate total size (IFD + additional heap + metadata + padding)
	auto basesz = ldesc + (std::uint32_t)metalen;
	auto padsz = basesz % ssize == 0 ? 0 : ssize - (basesz % ssize);
	std::uint32_t ltot = basesz + padsz;

	// Assign values
	if(ifd != nullptr)
		*ifd = lifd;
	if(desc != nullptr)
		*desc = ldesc;
	if(tot != nullptr)
		*tot = ltot;
}

/**
 * Write shape info to the header cache
 */
void G2STiffFile::writeShapeInfo() noexcept
{
	if(header.size() < G2STIFF_HEADER_SIZE || shape.empty())
		return;

	auto startind = bigTiff ? 52 : 40;
	writeInt(&header[startind], 4, shape.size());
	for(std::size_t i = 0; i < shape.size(); i++)
		writeInt(&header[startind + (i + 1) * 4], 4, shape[i]);
}

/**
 * Write integer value to a byte buffer
 * @param buff Byte buffer
 * @param len Value length (in bytes)
 * @param val Integer value
 */
void G2STiffFile::writeInt(unsigned char* buff, std::uint8_t len, std::uint64_t val) const noexcept
{
	if(buff == nullptr || len == 0)
		return;
	for(auto i = 0; i < len; i++)
		buff[i] = (val >> (i * 8)) & 0xff;
}

/**
 * Read integer value from a byte buffer
 * @param buff Byte buffer
 * @param len Value length (in bytes)
 * @return Integer value
 */
std::uint64_t G2STiffFile::readInt(const unsigned char* buff, std::uint8_t len) const noexcept
{
	if(buff == nullptr || len == 0 || len > 8)
		return 0;
	std::uint64_t ret = 0;
	for(std::uint8_t i = 0; i < len; i++)
	{
		auto shift = i * 8;
		std::uint64_t xval = (std::uint64_t)buff[i] << shift;
		ret |= xval;
	}
	return ret;
}

/**
 * Move file read cursor
 * This wont affect current file position (cursor)
 * @param pos New position
 */
void G2STiffFile::moveReadCursor(std::uint64_t pos) noexcept
{
	if(pos == readpos)
		return;
	readpos = pos;
	if(directIo && !readbuff.empty())
	{
		readbuff.clear();
		readbuffoff = 0;
	}
}

/**
 * Move file write cursor
 * This wont affect current file position (cursor)
 * @param pos New position
 */
void G2STiffFile::moveWriteCursor(std::uint64_t pos) noexcept
{
	if(pos == writepos)
		return;
	writepos = pos;
	if(directIo)
		writebuff.clear();
}

/**
 * Calculate image index from image coordinates
 * Image coordiantes should not contain indices for first two dimensions (width & height)
 * By convention image acquisitions loops through the coordinates in the ascending order (lower coordinates are looped first)
 * E.g. CTZ order means that all channels are acquired before moving changing the time point, and all specified time points 
 * are acquired before moving the Z-stage, in which case dataset with the shape 3-4-2 for coordinates 1-2-1 will return 19 (=1*12 + 2*3 + 1)
 * Last image coordinate can go beyond the specified shape size
 * @param coord Image coordinates
 * @return Image index
 * @throws std::runtime_error
 */
std::uint32_t G2STiffFile::calcImageIndex(const std::vector<std::uint32_t>& coord) const
{
	if(coord.size() > shape.size() - 2)
		throw std::runtime_error("Invalid number of coordinates");
	for(std::size_t i = 0; i < coord.size() - 1; i++)
	{
		if(coord[i] >= shape[i + 2])
			throw std::runtime_error("Invalid coordinate for dimension " + std::to_string(i + 2));
	}
	std::uint32_t ind = 0;
	for(int i = (int)coord.size() - 1; i >= 0; i--)
	{
		std::uint32_t sum = 1;
		for(int j = i - 1; j >= 0; j--)
			sum *= shape[j + 2];
		ind += sum * coord[i];
	}
	return ind;
}
