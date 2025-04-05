///////////////////////////////////////////////////////////////////////////////
// FILE:          G2SBigTiffStorage.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   BIGTIFF storage device driver
//
// AUTHOR:        Milos Jovanovic <milos@tehnocad.rs>
//
// COPYRIGHT:     Luminous Point LLC, Lumencor Inc., 2024
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
///////////////////////////////////////////////////////////////////////////////
#include <filesystem>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <boost/lexical_cast.hpp>
#include "G2SBigTiffStorage.h"

#define MAX_FILE_SEARCH_INDEX			128
#define PROP_DIRECTIO					"DirectIO"
#define PROP_FLUSHCYCLE					"FlushCycle"
#define PROP_CHUNKSIZE					"ChunkSize"

/**
 * Default class constructor
 */
G2SBigTiffStorage::G2SBigTiffStorage() : initialized(false), handleCounter(-1)
{
   supportedFormats = { "g2s" };

   InitializeDefaultErrorMessages();

   // set device specific error messages
   SetErrorText(ERR_INTERNAL, "Internal driver error, see log file for details");
   SetErrorText(ERR_TIFF, "Generic TIFF error. See log for more info.");
	SetErrorText(ERR_TIFF_STREAM_UNAVAILABLE, "BigTIFF storage error. File stream is not available.");
	SetErrorText(ERR_TIFF_INVALID_PATH, "Invalid path or name.");
	SetErrorText(ERR_TIFF_INVALID_DIMENSIONS, "Invalid number of dimensions. Minimum is 3.");
	SetErrorText(ERR_TIFF_INVALID_PIXEL_TYPE, "Invalid or unsupported pixel type.");
	SetErrorText(ERR_TIFF_OPEN_FAILED, "Failed opening TIF file.");
	SetErrorText(ERR_TIFF_HANDLE_INVALID, "Dataset handle is not valid.");
	SetErrorText(ERR_TIFF_STRING_TOO_LONG, "Requested string is too long for the provided buffer.");
	SetErrorText(ERR_TIFF_INVALID_COORDINATE, "Dataset coordinates are valid.");
	SetErrorText(ERR_TIFF_DATASET_CLOSED, "Operation unavailable - dataset is closed.");
	SetErrorText(ERR_TIFF_DATASET_READONLY, "Operation unavailable - dataset is read-only.");
	SetErrorText(ERR_TIFF_DELETE_FAILED, "File / folder delete failed.");
	SetErrorText(ERR_TIFF_ALLOCATION_FAILED, "Dataset memory allocation failed.");
	SetErrorText(ERR_TIFF_CORRUPTED_METADATA, "Metadata corrupted / invalid");
	SetErrorText(ERR_TIFF_UPDATE_FAIL, "Dataset structure update failed");
	SetErrorText(ERR_TIFF_FILESYSTEM_ERROR, "Filesystem error");
	SetErrorText(ERR_TIFF_INVALID_META_KEY, "Invalid metadata key");

   // create pre-initialization properties                                   
   // ------------------------------------
   //
   
   // Name                                                                   
   CreateProperty(MM::g_Keyword_Name, g_BigTiffStorage, MM::String, true);
   //
   // Description
   std::ostringstream os;
	os << "BigTIFF Storage " << G2STIFF_VERSION;
   CreateProperty(MM::g_Keyword_Description, os.str().c_str(), MM::String, true);
}

/**
 * Get device name
 * @param Name String buffer [out]
 */
void G2SBigTiffStorage::GetName(char* Name) const
{
   CDeviceUtils::CopyLimitedString(Name, g_BigTiffStorage);
}

/**
 * Device driver initialization routine
 */
int G2SBigTiffStorage::Initialize()
{
   if(initialized)
      return DEVICE_OK;
	
	// Add DirectIO property
	int nRet = CreateIntegerProperty(PROP_DIRECTIO, 0, false);
	assert(nRet == DEVICE_OK);
	AddAllowedValue(PROP_DIRECTIO,"0");
	AddAllowedValue(PROP_DIRECTIO,"1");

	// Add flush counter property
	nRet = CreateIntegerProperty(PROP_FLUSHCYCLE, 0, false);
	assert(nRet == DEVICE_OK);

	// Add chunk size property
	nRet = CreateIntegerProperty(PROP_CHUNKSIZE, 0, false);
	assert(nRet == DEVICE_OK);

   UpdateStatus();

   initialized = true;
   return DEVICE_OK;
}

/**
 * Device driver shutdown routine
 * During device shutdown cache will be emptied, 
 * and all open file handles will be closed
 */
int G2SBigTiffStorage::Shutdown() noexcept
{
   initialized = false;
   for(auto it = cache.begin(); it != cache.end(); it++)
   {
      if(it->second.isOpen())
		{
			auto fs = reinterpret_cast<G2SBigTiffDataset*>(it->second.FileHandle);
			fs->close();
			it->second.close();
			delete fs;
		}
   }
   cache.clear();
   return DEVICE_OK;
}

/**
 * Create storage entry
 * Dataset storage descriptor will open a file handle, to close a file handle call Close()
 * Dataset storage descriptor will reside in device driver cache
 * If the file already exists this method will fail with 'DEVICE_DUPLICATE_PROPERTY' status code
 * @param path Absolute file path (TIFF file)
 * @param name Dataset name
 * @param numberOfDimensions Number of dimensions
 * @param shape Axis sizes
 * @param pixType Pixel format
 * @param meta Metadata
 * @param metaLength length of the metadata string
 * @param handle [out]
 * @return Status code
 */
int G2SBigTiffStorage::Create(const char* path, const char* name, int numberOfDimensions, const int shape[], MM::StorageDataType pixType, const char* meta, int metaLength, int* handle) noexcept
{
   if(path == nullptr)
      return ERR_TIFF_INVALID_PATH;
	if(numberOfDimensions < 3)
		return ERR_TIFF_INVALID_DIMENSIONS;
	if(!(pixType == MM::StorageDataType::StorageDataType_GRAY16 || pixType == MM::StorageDataType::StorageDataType_GRAY8 || pixType == MM::StorageDataType::StorageDataType_RGB32)) 
		return ERR_TIFF_INVALID_PIXEL_TYPE;
	if(shape == nullptr || handle == nullptr)
		return DEVICE_INVALID_INPUT_PARAM;

	try
	{
		// Check cache size limits
		if(cache.size() >= MAX_CACHE_SIZE)
		{
			cacheReduce();
			if(CACHE_HARD_LIMIT && cache.size() >= MAX_CACHE_SIZE)
				return ERR_TIFF_CACHE_OVERFLOW;
		}

		// Compose dataset path
		std::string dsname(name);
		if(dsname.find(".tiff") == dsname.size() - 5)
			dsname = dsname.substr(0, dsname.size() - 5);
		else if(dsname.find(".tif") == dsname.size() - 4 || dsname.find(".tf8") == dsname.size() - 4)
			dsname = dsname.substr(0, dsname.size() - 4);
		if(dsname.find(".g2s") != dsname.size() - 4)
			dsname += ".g2s";
		std::filesystem::path saveRoot = std::filesystem::u8path(path);
		std::filesystem::path dsName = saveRoot / dsname;
	
		// Create dataset storage descriptor
		*handle = ++handleCounter;
  
		// Create a file on disk and store the file handle
		auto fhandle = new G2SBigTiffDataset();
		if(fhandle == nullptr)
		{
			LogMessage("Error obtaining file handle for: " + dsName.u8string());
			return ERR_TIFF_STREAM_UNAVAILABLE;
		}
	
		try
		{
			fhandle->create(dsName.u8string(), getDirectIO(), true, (std::uint32_t)getChunkSize());
			if(!fhandle->isOpen())
			{
				LogMessage("Failed to open file: " + dsName.u8string());
				return ERR_TIFF_OPEN_FAILED;
			}
			fhandle->setFlushCycles((std::uint32_t)getFlushCycle());
		}
		catch(std::exception& err)
		{
			delete fhandle;
			LogMessage(std::string(err.what()) + " for " + dsName.u8string());
			return ERR_TIFF_OPEN_FAILED;
		}

		// Create dataset storage descriptor
		std::string guid = boost::lexical_cast<std::string>(boost::uuids::random_generator()());           // Entry UUID
		if (guid.size() > MM::MaxStrLength)
		{
#ifdef _NDEBUG
			guid = guid.substr(0, MM::MaxStrLength);
#else
			assert("Dataset handle size is too long");
#endif
		}


		G2SStorageEntry sdesc(fhandle->getPath());
		sdesc.FileHandle = fhandle;
		try
		{

			// Set dataset UUID / shape / metadata
			std::vector<std::uint32_t> vshape;
			vshape.assign(shape, shape + numberOfDimensions);
			fhandle->setUID(guid);
			fhandle->setHandle(*handle);
			fhandle->setShape(vshape);
			std::string metadataStr(meta, metaLength);
			fhandle->setMetadata(metadataStr);

			// Set pixel format
			if(pixType == MM::StorageDataType::StorageDataType_GRAY8)
				fhandle->setPixelFormat(8, 1);
			else if(pixType == MM::StorageDataType::StorageDataType_GRAY16)
				fhandle->setPixelFormat(16, 1);
			else if(pixType == MM::StorageDataType::StorageDataType_RGB32)
				fhandle->setPixelFormat(8, 4);
		}
		catch(std::exception& err)
		{
			delete fhandle;
			LogMessage(std::string(err.what()) + " for " + dsName.u8string());
			return ERR_TIFF_UPDATE_FAIL;
		}

		// Append dataset storage descriptor to cache
		auto it = cache.insert(std::make_pair(*handle, sdesc));
		if(it.first == cache.end())
		{
			delete fhandle;
			LogMessage("Adding BigTIFF dataset to cache failed. Path: " + dsName.u8string() + ", handle: " + std::to_string(*handle));
			return ERR_TIFF_CACHE_INSERT;
		}
		if(!it.second)
		{
			// Dataset already exists
			delete fhandle;
			LogMessage("Adding BigTIFF dataset to cache failed. Path: " + dsName.u8string() + ", handle: " + std::to_string(*handle));
			return ERR_TIFF_CACHE_INSERT;
		}

		return DEVICE_OK;
	}
	catch(std::exception& e)
	{
		LogMessage("Create error: " +std::string(e.what()));
		return ERR_INTERNAL;
	}
}

/**
 * Load dataset from disk
 * Dataset storage descriptor will be read from file
 * Dataset storage descriptor will open a file handle, to close a file handle call Close()
 * Dataset storage descriptor will reside in device driver cache
 * @param path Absolute file path (TIFF file) / Absolute dataset folder path
 * @param name Dataset name
 * @param handle Entry GUID [out]
 * @return Status code
 */
int G2SBigTiffStorage::Load(const char* path, int* handle) noexcept
{
   if(path == nullptr || handle == nullptr)
      return DEVICE_INVALID_INPUT_PARAM;

	try
	{
		// Check if the file exists
		std::error_code ec;
		std::filesystem::path actpath = std::filesystem::u8path(path);
		auto ex = std::filesystem::exists(actpath, ec);
		if(ec)
		{
			LogMessage("Load filesystem error " + std::to_string(ec.value()) + ". " + ec.message());
			return ERR_TIFF_FILESYSTEM_ERROR;
		}
		auto rf = ex ? std::filesystem::is_regular_file(actpath, ec) : false;
		if(ec)
		{
			LogMessage("Load filesystem error " + std::to_string(ec.value()) + ". " + ec.message());
			return ERR_TIFF_FILESYSTEM_ERROR;
		}

		if(!ex)
		{
			// Try finding the folder by adding the index (number suffix)
			bool fnd = false;
			auto dir = actpath.parent_path();
			auto fname = actpath.stem().u8string();
			auto ext = actpath.extension().u8string();
			for(int i = 0; i < MAX_FILE_SEARCH_INDEX; i++)
			{
				actpath = dir / (fname + "_" + std::to_string(i) + ext);
				if(std::filesystem::exists(actpath))
				{
					fnd = true;
					break;
				}
			}
			if(!fnd)
				return ERR_TIFF_INVALID_PATH;
		}
		else if(rf)
		{
			// File path of the first data chunk is specified -> use parent folder path
			actpath = actpath.parent_path();
		}

		// Check if file is already in cache
		auto cit = cache.begin();
		while(cit != cache.end())
		{
			if(std::filesystem::u8path(cit->second.Path) == actpath)
				break;
			cit++;
		}

		G2SBigTiffDataset* fhandle = nullptr;
		if(cit == cache.end())
			// Open a file on disk and store the file handle
			fhandle = new G2SBigTiffDataset();
		else if(cit->second.FileHandle == nullptr)
		{
			// Open a file on disk and update the cache file handle 
			fhandle = new G2SBigTiffDataset();
			cit->second.FileHandle = fhandle;
		}
		else
			// Use existing object descriptor
			fhandle = (G2SBigTiffDataset*)cit->second.FileHandle;
		if(fhandle == nullptr)
		{
			LogMessage("Loading BigTIFF dataset failed (" + actpath.u8string() + "). Dataset allocation failed");
			return ERR_TIFF_ALLOCATION_FAILED;
		}

		try
		{
			if(!fhandle->isOpen())
			{
				fhandle->load(std::filesystem::absolute(actpath).u8string(), getDirectIO());
				if(!fhandle->isOpen())
					return ERR_TIFF_OPEN_FAILED;
			}
			fhandle->setFlushCycles((std::uint32_t)getFlushCycle());
		}
		catch(std::exception& e)
		{
			delete fhandle;
			LogMessage("Loading BigTIFF dataset failed (" + actpath.u8string() + "). " + std::string(e.what()));
			return ERR_TIFF_OPEN_FAILED;
		}
	
		// Obtain / generate dataset UID
		std::string guid = fhandle->getUID().empty() ? boost::lexical_cast<std::string>(boost::uuids::random_generator()()) : fhandle->getUID();
		if(guid.size() > MM::MaxStrLength)
		{
			delete fhandle;
			return ERR_TIFF_STRING_TOO_LONG;
		}

		// Append dataset storage descriptor to cache
		if(cit == cache.end())
		{
			// Create dataset storage descriptor
			G2SStorageEntry sdesc(std::filesystem::absolute(actpath).u8string());
			sdesc.FileHandle = fhandle;

			auto it = cache.insert(std::make_pair(++handleCounter, sdesc));
			if(it.first == cache.end())
			{
				delete fhandle;
				LogMessage("Loading BigTIFF dataset failed (" + actpath.u8string() + "). Dataset cache is full");
				return DEVICE_OUT_OF_MEMORY;
			}
		}

		*handle = handleCounter;
		return DEVICE_OK;
	}
	catch(std::exception& e)
	{
		LogMessage("Load error: " +std::string(e.what()));
		return ERR_INTERNAL;
	}
}

/**
 * Get actual dataset shape
 * Shape contains image width and height as first two dimensions
 * @param handle Entry GUID
 * @param shape Dataset shape [out]
 * @return Status code
 */
int G2SBigTiffStorage::GetShape(int handle, int shape[]) noexcept
{
	if(shape == nullptr)
		return DEVICE_INVALID_INPUT_PARAM;

	// Obtain dataset descriptor from cache
	auto it = cache.find(handle);
	if(it == cache.end())
		return ERR_TIFF_HANDLE_INVALID;

	auto fs = reinterpret_cast<G2SBigTiffDataset*>(it->second.FileHandle);
	for(std::size_t i = 0; i < fs->getDimension(); i++)
		shape[i] = fs->getActualShape()[i];
   return DEVICE_OK;
}

/**
 * Get dataset pixel format
 * @param handle Entry GUID
 * @param pixelDataType Pixel format [out]
 * @return Status code
 */
int G2SBigTiffStorage::GetDataType(int handle, MM::StorageDataType& pixelDataType) noexcept
{
	// Obtain dataset descriptor from cache
	auto it = cache.find(handle);
	if(it == cache.end())
		return ERR_TIFF_HANDLE_INVALID;

	// Get pixel format
	if(!it->second.isOpen())
		pixelDataType = MM::StorageDataType_UNKNOWN;
	else
	{
		auto fs = reinterpret_cast<G2SBigTiffDataset*>(it->second.FileHandle);
		switch(fs->getBpp())
		{
			case 1:
				pixelDataType = MM::StorageDataType_GRAY8;
				break;
			case 2:
				pixelDataType = MM::StorageDataType_GRAY16;
				break;
			case 3:
			case 4:
				pixelDataType = MM::StorageDataType_RGB32;
				break;
			default:
				pixelDataType = MM::StorageDataType_UNKNOWN;
				break;
		}
	}
   return DEVICE_OK;
}

/**
 * Close the dataset
 * File handle will be closed
 * Metadata will be discarded
 * Storage entry descriptor will remain in cache
 * @param handle Entry GUID
 * @return Status code
 */
int G2SBigTiffStorage::Close(int handle) noexcept
{
   auto it = cache.find(handle);
   if(it == cache.end())
      return ERR_TIFF_HANDLE_INVALID;
   if(it->second.isOpen())
   {
		auto fs = reinterpret_cast<G2SBigTiffDataset*>(it->second.FileHandle);
		fs->close();
		it->second.close();
		delete fs;
   }
   return DEVICE_OK;
}

/**
 * Delete existing dataset (file on disk)
 * If the file doesn't exist this method will return 'DEVICE_NO_PROPERTY_DATA' status code
 * Dataset storage descriptor will be removed from cache
 * @param handle Entry GUID
 * @return Status code
 */
int G2SBigTiffStorage::Delete(int handle) noexcept
{

   auto it = cache.find(handle);
   if(it == cache.end())
      return ERR_TIFF_HANDLE_INVALID;

	try
	{
		// Check if the file exists
		std::error_code ec;
		auto fp = std::filesystem::u8path(it->second.Path);
		auto ex = std::filesystem::exists(fp, ec);
		if(ec)
		{
			LogMessage("Delete filesystem error " + std::to_string(ec.value()) + ". " + ec.message());
			return ERR_TIFF_FILESYSTEM_ERROR;
		}
		if(!ex)
			return ERR_TIFF_INVALID_PATH;

		// Close the file handle
		if(it->second.isOpen())
		{
			auto fs = reinterpret_cast<G2SBigTiffDataset*>(it->second.FileHandle);
			fs->close();
			it->second.close();
			delete fs;
		}
   
		// Delete the file
		bool succ = std::filesystem::remove(fp, ec);
		if(ec)
		{
			LogMessage("Delete filesystem error " + std::to_string(ec.value()) + ". " + ec.message());
			return ERR_TIFF_FILESYSTEM_ERROR;
		}
		if(!succ)
			return ERR_TIFF_DELETE_FAILED;

		// Discard the cache entry
		cache.erase(it);
		return DEVICE_OK;
	}
	catch(std::exception& e)
	{
		LogMessage("Delete error: " +std::string(e.what()));
		return ERR_INTERNAL;
	}
}

/**
 * List datasets in the specified folder / path
 * If the list of found datasets is longer than 'maxItems' only first [maxItems] will be 
 * returned and 'DEVICE_SEQUENCE_TOO_LARGE' status code will be returned
 * If the dataset path is longer than 'maxItemLength' dataset path will be truncated
 * If the specified path doesn't exist, or it's not a valid folder path 'DEVICE_INVALID_INPUT_PARAM' status code will be returned
 * @param path Folder path
 * @param listOfDatasets Dataset path list [out]
 * @param maxItems Max dataset count
 * @param maxItemLength Max dataset path length
 * @return Status code
 */
int G2SBigTiffStorage::List(const char* path, char** listOfDatasets, int maxItems, int maxItemLength) noexcept
{
   if(path == nullptr || listOfDatasets == nullptr || maxItems <= 0 || maxItemLength <= 0)
      return DEVICE_INVALID_INPUT_PARAM;
	
	try
	{
		std::error_code ec;
		auto dp = std::filesystem::u8path(path);
		auto exs = std::filesystem::exists(dp, ec);
		if(ec)
		{
			LogMessage("List filesystem error " + std::to_string(ec.value()) + ". " + ec.message());
			return ERR_TIFF_FILESYSTEM_ERROR;
		}
		auto isdir = std::filesystem::is_directory(dp, ec);
		if(ec)
		{
			LogMessage("List filesystem error " + std::to_string(ec.value()) + ". " + ec.message());
			return ERR_TIFF_FILESYSTEM_ERROR;
		}
		if(!exs || !isdir)
			return ERR_TIFF_INVALID_PATH;
		auto allfnd = scanDir(path, listOfDatasets, maxItems, maxItemLength, 0);

		// TODO: review memory allocation and whether the limit can be removed
		return allfnd ? DEVICE_OK : ERR_TIFF_STRING_TOO_LONG;
	}
	catch(std::exception& e)
	{
		LogMessage("List error: " +std::string(e.what()));
		return ERR_INTERNAL;
	}
}

/**
 * Add image / write image to file
 * Image metadata will be stored in cache
 * @param handle Entry GUID
 * @param pixels Pixel data buffer
 * @param sizeInBytes pixel array size
 * @param coordinates Image coordinates
 * @param numCoordinates Coordinate count
 * @param imageMeta Image metadata
 * @param metaLength metadata length
 * @return Status code
 */
int G2SBigTiffStorage::AddImage(int handle, int sizeInBytes, unsigned char* pixels, int coordinates[], int numCoordinates, const char* imageMeta, int metaLength) noexcept
{
	if(pixels == nullptr || sizeInBytes <= 0 || numCoordinates <= 0)
		return DEVICE_INVALID_INPUT_PARAM;

	// Obtain dataset descriptor from cache
	auto it = cache.find(handle);
	if(it == cache.end())
		return ERR_TIFF_HANDLE_INVALID;
	if(!it->second.isOpen())
		return ERR_TIFF_DATASET_CLOSED;

	// Validate image dimensions / coordinates
	auto fs = reinterpret_cast<G2SBigTiffDataset*>(it->second.FileHandle);
	if(fs->isInReadMode())
		return ERR_TIFF_DATASET_READONLY;
	if(!validateCoordinates(fs, coordinates, numCoordinates, true))
		return ERR_TIFF_INVALID_COORDINATE;
	if(fs->isCoordinateSet(coordinates, numCoordinates))
		return ERR_TIFF_INVALID_COORDINATE;

	try
	{
		// Add image
		std::string imageMetaStr(imageMeta, metaLength);
		fs->addImage(pixels, sizeInBytes, imageMetaStr);
		return DEVICE_OK;
	}
	catch(std::exception& e)
	{
		LogMessage("AddImage error: " +std::string(e.what()));
		return ERR_TIFF_UPDATE_FAIL;
	}
}

/**
 * Append image / write image to file
 * Image metadata will be stored in cache
 * @param handle Entry GUID
 * @param pixels Pixel data buffer
 * @param sizeInBytes pixel array size
 * @param imageMeta Image metadata
 * @param metaLength length of the metadata
 * @return Status code
 */
int G2SBigTiffStorage::AppendImage(int handle, int sizeInBytes, unsigned char* pixels, const char* imageMeta, int metaLength) noexcept
{
	if(pixels == nullptr || sizeInBytes <= 0)
		return DEVICE_INVALID_INPUT_PARAM;

	// Obtain dataset descriptor from cache
	auto it = cache.find(handle);
	if(it == cache.end())
		return ERR_TIFF_HANDLE_INVALID;
	if(!it->second.isOpen())
		return ERR_TIFF_DATASET_CLOSED;

	try
	{
		// Append image
		auto fs = reinterpret_cast<G2SBigTiffDataset*>(it->second.FileHandle);
		if(fs->isInReadMode())
			return ERR_TIFF_DATASET_READONLY;
		std::string imageMetaStr(imageMeta, metaLength);
		fs->addImage(pixels, sizeInBytes, imageMetaStr);
		return DEVICE_OK;
	}
	catch(std::exception& e)
	{
		LogMessage("AppendImage error: " +std::string(e.what()));
		return ERR_TIFF_UPDATE_FAIL;
	}
}

/**
 * Get dataset summary metadata
 * If the netadata size is longer than the provided buffer, only the first [bufSize] bytes will be
 * copied, and the status code 'DEVICE_SEQUENCE_TOO_LARGE' will be returned
 * @param handle Entry GUID
 * @param meta Metadata buffer [out]
 * @return Status code
 */
int G2SBigTiffStorage::GetSummaryMeta(int handle, char** meta) noexcept
{
	// Obtain dataset descriptor from cache
   auto it = cache.find(handle);
   if(it == cache.end())
      return ERR_TIFF_HANDLE_INVALID;

	if(!it->second.isOpen())
		return ERR_TIFF_DATASET_CLOSED;

	try
	{
		// Copy metadata string
		auto fs = reinterpret_cast<G2SBigTiffDataset*>(it->second.FileHandle);
		*meta = new char[fs->getMetadata().size() + 1];
		strncpy(*meta, fs->getMetadata().c_str(), fs->getMetadata().size() + 1);
		return DEVICE_OK;
	}
	catch(std::exception& e)
	{
		LogMessage("GetSummaryMeta error: " + std::string(e.what()));
		return ERR_TIFF_CORRUPTED_METADATA;
	}
}

/**
 * Get dataset image metadata
 * If the netadata size is longer than the provided buffer, only the first [bufSize] bytes will be
 * copied, and the status code 'DEVICE_SEQUENCE_TOO_LARGE' will be returned
 * @param handle Entry GUID
 * @param coordinates Image coordinates
 * @param numCoordinates Coordinate count
 * @param meta Metadata buffer [out]
 * @param bufSize Buffer size
 * @return Status code
 */
int G2SBigTiffStorage::GetImageMeta(int handle, int coordinates[], int numCoordinates, char** meta) noexcept
{
   if(coordinates == nullptr || numCoordinates == 0)
      return DEVICE_INVALID_INPUT_PARAM;

	// Obtain dataset descriptor from cache
   auto it = cache.find(handle);
   if(it == cache.end())
      return ERR_TIFF_HANDLE_INVALID;

	auto fs = reinterpret_cast<G2SBigTiffDataset*>(it->second.FileHandle);
	if(!validateCoordinates(fs, coordinates, numCoordinates))
		return ERR_TIFF_INVALID_COORDINATE;
   
	// Obtain metadata from the file stream
	if(!it->second.isOpen())
		return ERR_TIFF_DATASET_CLOSED;

	// Copy coordinates without including the width and height
	std::vector<std::uint32_t> coords(fs->getDimension() - 2);
	for(int i = 0; i < coords.size(); i++)
	{
		if(i >= numCoordinates)
			break;
		coords[i] = coordinates[i];
	}

	try
	{
		auto fmeta = fs->getImageMetadata(coords);
		*meta = new char[fmeta.size() + 1];
		strncpy(*meta, fmeta.c_str(), fmeta.size() + 1);
		return DEVICE_OK;
	}
	catch(std::exception& e)
	{
		LogMessage("GetImageMeta error: " + std::string(e.what()));
		return ERR_TIFF_CORRUPTED_METADATA;
	}
}

/**
 * Get image / pixel data
 * Image buffer will be created inside this method, so
 * object (buffer) destruction becomes callers responsibility
 * @param handle Entry GUID
 * @param coordinates Image coordinates
 * @param numCoordinates Coordinate count
 * @return Pixel buffer pointer
 */
const unsigned char* G2SBigTiffStorage::GetImage(int handle, int coordinates[], int numCoordinates) noexcept
{
	if(numCoordinates <= 0)
		return nullptr;
	try 
	{
		// Obtain dataset descriptor from cache
		auto it = cache.find(handle);
		if(it == cache.end())
			return nullptr;

		auto fs = reinterpret_cast<G2SBigTiffDataset*>(it->second.FileHandle);
		if(!validateCoordinates(fs, coordinates, numCoordinates))
			return nullptr;

		if(!it->second.isOpen())
			return nullptr;

		// Copy coordinates without including the width and height
		std::vector<std::uint32_t> coords(fs->getDimension() - 2);
		for(int i = 0; i < coords.size(); i++)
		{
			if (i >= numCoordinates)
				break;
			coords[i] = coordinates[i];
		}

		it->second.ImageData = fs->getImage(coords);
		return &it->second.ImageData[0];
	}
	catch(std::runtime_error& e)
	{
		LogMessage("GetImage error: " +std::string(e.what()));
		return nullptr;
	}
}

/**
 * Configure metadata for a given dimension
 * @param handle Entry GUID
 * @param dimension Dimension index
 * @param name Name of the dimension
 * @param meaning Z,T,C, etc. (physical meaning)
 * @return Status code
 */
int G2SBigTiffStorage::ConfigureDimension(int handle, int dimension, const char* name, const char* meaning) noexcept
{
   if(dimension < 0 || name == nullptr || meaning == nullptr)
      return DEVICE_INVALID_INPUT_PARAM;
   auto it = cache.find(handle);
   if(it == cache.end())
      return ERR_TIFF_HANDLE_INVALID;
	if(!it->second.isOpen())
		return ERR_TIFF_DATASET_CLOSED;
	auto fs = reinterpret_cast<G2SBigTiffDataset*>(it->second.FileHandle);
	if(fs->isInReadMode())
		return ERR_TIFF_DATASET_READONLY;

   if((std::size_t)dimension >= fs->getDimension())
      return ERR_TIFF_INVALID_DIMENSIONS;
	fs->configureAxis(dimension, std::string(name), std::string(meaning));
   return DEVICE_OK;
}

/**
 * Configure a particular coordinate name. e.g. channel name / position name ...
 * @param handle Entry GUID
 * @param dimension Dimension index
 * @param coordinate Coordinate index
 * @param name Coordinate name
 * @return Status code
 */
int G2SBigTiffStorage::ConfigureCoordinate(int handle, int dimension, int coordinate, const char* name) noexcept
{
   if(dimension < 0 || coordinate < 0 || name == nullptr)
      return DEVICE_INVALID_INPUT_PARAM;
   auto it = cache.find(handle);
   if(it == cache.end())
      return ERR_TIFF_HANDLE_INVALID;
	if(!it->second.isOpen())
		return ERR_TIFF_DATASET_CLOSED;
	auto fs = reinterpret_cast<G2SBigTiffDataset*>(it->second.FileHandle);
	if(fs->isInReadMode())
		return ERR_TIFF_DATASET_READONLY;

	if(dimension < 0 || (std::size_t)dimension >= fs->getDimension())
		return ERR_TIFF_INVALID_DIMENSIONS;
   if(coordinate < 0 || ((std::size_t)coordinate >= fs->getShape()[dimension] && dimension > 0))
      return ERR_TIFF_INVALID_COORDINATE;
   fs->configureCoordinate(dimension, coordinate, std::string(name));
   return DEVICE_OK;
}

/**
 * Get number of dimensions
 * @param handle Entry GUID
 * @param numDimensions Number of dimensions [out]
 * @return Status code
 */
int G2SBigTiffStorage::GetNumberOfDimensions(int handle, int& numDimensions) noexcept
{
   auto it = cache.find(handle);
   if(it == cache.end())
      return ERR_TIFF_HANDLE_INVALID;
	if(!it->second.isOpen())
		return ERR_TIFF_DATASET_CLOSED;

	auto fs = reinterpret_cast<G2SBigTiffDataset*>(it->second.FileHandle);
   numDimensions = (int)fs->getDimension();
   return DEVICE_OK;
}

/**
 * Get number of dimensions
 * @param handle Entry GUID
 * @return Status code
 */
int G2SBigTiffStorage::GetDimension(int handle, int dimension, char* name, int nameLength, char* meaning, int meaningLength) noexcept
{
   if(dimension < 0 || meaningLength <= 0 || name == nullptr || meaning == nullptr)
      return DEVICE_INVALID_INPUT_PARAM;
   auto it = cache.find(handle);
   if(it == cache.end())
      return ERR_TIFF_HANDLE_INVALID;
	if(!it->second.isOpen())
		return ERR_TIFF_DATASET_CLOSED;
	auto fs = reinterpret_cast<G2SBigTiffDataset*>(it->second.FileHandle);

	if(dimension < 0 || (std::size_t)dimension >= fs->getDimension())
		return ERR_TIFF_INVALID_DIMENSIONS;

	try
	{
		const auto& axinf = fs->getAxisInfo((std::uint32_t)dimension);
		if(axinf.Name.size() > (std::size_t)nameLength)
			return ERR_TIFF_STRING_TOO_LONG;
		if(axinf.Description.size() > (std::size_t)meaningLength)
			return ERR_TIFF_STRING_TOO_LONG;
		strncpy(name, axinf.Name.c_str(), nameLength);
		strncpy(meaning, axinf.Description.c_str(), meaningLength);
		return DEVICE_OK;
	}
	catch(std::exception& e)
	{
		LogMessage("GetDimension error: " + std::string(e.what()));
		return ERR_TIFF_INVALID_COORDINATE;
	}
}

/**
 * Get number of dimensions
 * @param handle Entry GUID
 * @return Status code
 */
int G2SBigTiffStorage::GetCoordinate(int handle, int dimension, int coordinate, char* name, int nameLength) noexcept
{
   if(dimension < 0 || coordinate < 0 || nameLength <= 0 || name == nullptr)
      return DEVICE_INVALID_INPUT_PARAM;
   auto it = cache.find(handle);
   if(it == cache.end())
      return ERR_TIFF_HANDLE_INVALID;
	if(!it->second.isOpen())
		return ERR_TIFF_DATASET_CLOSED;
	auto fs = reinterpret_cast<G2SBigTiffDataset*>(it->second.FileHandle);

	if(dimension < 0 || (std::size_t)dimension >= fs->getDimension())
		return ERR_TIFF_INVALID_DIMENSIONS;
	if(coordinate < 0)
		return ERR_TIFF_INVALID_COORDINATE;
	if((std::size_t)coordinate >= fs->getShape()[dimension])
	{
		if(dimension > 0)
			return ERR_TIFF_INVALID_COORDINATE;
		if((std::size_t)coordinate >= fs->getActualShape()[dimension])
			return ERR_TIFF_INVALID_COORDINATE;
	}

	try
	{
		const auto& axinf = fs->getAxisInfo((std::uint32_t)dimension);
		if((std::size_t)coordinate < axinf.Coordinates.size())
		{
			if(axinf.Coordinates[coordinate].size() > (std::size_t)nameLength)
				return ERR_TIFF_STRING_TOO_LONG;
			strncpy(name, axinf.Coordinates[coordinate].c_str(), nameLength);
		}
		else if(dimension > 0)
			return ERR_TIFF_INVALID_COORDINATE;
		return DEVICE_OK;
	}
	catch(std::exception& e)
	{
		LogMessage("GetCoordinate error: " + std::string(e.what()));
		return ERR_TIFF_INVALID_COORDINATE;
	}
}

/**
 * Get number of available images
 * @param handle Entry GUID
 * @param imgcount Image count [out]
 * @return Status code
 */
int G2SBigTiffStorage::GetImageCount(int handle, int& imgcount) noexcept
{
	auto it = cache.find(handle);
	if(it == cache.end())
		return ERR_TIFF_HANDLE_INVALID;
	if(!it->second.isOpen())
		return ERR_TIFF_DATASET_CLOSED;
	auto fs = reinterpret_cast<G2SBigTiffDataset*>(it->second.FileHandle);
	imgcount = (int)fs->getImageCount();
	return DEVICE_OK;
}

/**
 * Set custom metadata (key-value pair)
 * @param handle Entry GUID
 * @param key Metadata entry key
 * @param content Metadata entry value / content
 * @return Status code
 */
int G2SBigTiffStorage::SetCustomMetadata(int handle, const char* key, const char* content, int contentLength) noexcept
{
	if (key == nullptr || content == nullptr)
		return DEVICE_INVALID_INPUT_PARAM;
	auto it = cache.find(handle);
	if(it == cache.end())
		return ERR_TIFF_HANDLE_INVALID;
	if(!it->second.isOpen())
		return ERR_TIFF_DATASET_CLOSED;
	auto fs = reinterpret_cast<G2SBigTiffDataset*>(it->second.FileHandle);
	if(fs->isInReadMode())
		return ERR_TIFF_DATASET_READONLY;
	
	std::string contentStr(content, contentLength);
	fs->setCustomMetadata(key, content);
	return DEVICE_OK;
}

/**
 * Get custom metadata (key-value pair)
 * @param handle Entry GUID
 * @param key Metadata entry key
 * @param content Metadata entry value / content [out]
 * @return Status code
 */
int G2SBigTiffStorage::GetCustomMetadata(int handle, const char* key, char** content) noexcept
{
	if(key == nullptr)
		return DEVICE_INVALID_INPUT_PARAM;
	auto it = cache.find(handle);
	if(it == cache.end())
		return ERR_TIFF_HANDLE_INVALID;
	if(!it->second.isOpen())
		return ERR_TIFF_DATASET_CLOSED;
	auto fs = reinterpret_cast<G2SBigTiffDataset*>(it->second.FileHandle);
	if(!fs->hasCustomMetadata(key))
		return ERR_TIFF_INVALID_META_KEY;
	
	try
	{
		auto mval = fs->getCustomMetadata(key);
		*content = new char[mval.size() + 1];
		strncpy(*content, mval.c_str(), mval.size() + 1);
		return DEVICE_OK;
	}
	catch(std::exception& e) 
	{ 
		LogMessage("GetCustomMetadata error: " + std::string(e.what()));
		return ERR_TIFF_INVALID_META_KEY;
	}
}

/**
 * Check if dataset is open
 * If the dataset doesn't exist, or the GUID is invalid this method will return false
 * @param handle Entry GUID
 * @return true if dataset is open
 */
bool G2SBigTiffStorage::IsOpen(int handle) noexcept
{
	auto it = cache.find(handle);
	if(it == cache.end())
		return false;	
	return it->second.isOpen();
}

/**
 * Check if dataset is read-only
 * If the dataset doesn't exist, or the GUID is invalid this method will return true
 * @param handle Entry GUID
 * @return true if images can't be added to the dataset
 */
bool G2SBigTiffStorage::IsReadOnly(int handle) noexcept
{
	auto it = cache.find(handle);
	if(it == cache.end())
		return true;
	if(!it->second.isOpen())
		return true;
	auto fs = reinterpret_cast<G2SBigTiffDataset*>(it->second.FileHandle);
	return fs->isInReadMode();
}

/**
 * Get dataset path
 * @param handle Entry GUID
 * @param path Dataset path [out]
 * @param maxPathLength Max path length
 * @return Status code
 */
int G2SBigTiffStorage::GetPath(int handle, char* path, int maxPathLength) noexcept
{
	if(maxPathLength <= 0 || path == nullptr)
		return DEVICE_INVALID_INPUT_PARAM;
	auto it = cache.find(handle);
	if(it == cache.end())
		return ERR_TIFF_HANDLE_INVALID;
	if(it->second.Path.size() > (std::size_t)maxPathLength)
		return ERR_TIFF_STRING_TOO_LONG;
	strncpy(path, it->second.Path.c_str(), it->second.Path.size());
	return DEVICE_OK;
}

/**
 * Check if there is a valid dataset on the selected path
 * @param path Dataset path
 * @return Path is a valid dataset
 */
bool G2SBigTiffStorage::CanLoad(const char* path) noexcept
{
	if(path == nullptr)
		return false;
	try
	{
		std::error_code ec;
		std::filesystem::path xpath = std::filesystem::u8path(path);
		if(!std::filesystem::exists(xpath, ec))
			return false;
		if(ec)
			return false;

		bool isdir = std::filesystem::is_directory(xpath, ec);
		if(ec)
			return false;
		if(isdir)
		{
			// If directory is selected check if it's not empty and if the name ends with .g2s
			auto dname = xpath.filename().u8string();
			if(dname.find(".g2s") != dname.size() - 4)
				return false;
		
			// Check for valid files
			int validfiles = 0;
			for(const auto& entry : std::filesystem::directory_iterator(xpath))
			{
				// Skip auto folder paths
				auto fname = entry.path().filename().u8string();
				if(fname == "." || fname == "..")
					continue;

				// Skip folders
				bool issubdir = std::filesystem::is_directory(entry, ec);
				if(ec)
				{
					LogMessage("CanLoad filesystem error " + std::to_string(ec.value()) + ". " + ec.message());
					return false;
				}
				if(issubdir)
					continue;

				// Skip unsupported file formats
				auto fext = entry.path().extension().u8string();
				if(fext.size() == 0)
					continue;
				if(fext[0] == '.')
					fext = fext.substr(1);
				std::transform(fext.begin(), fext.end(), fext.begin(), [](char c) { return (char)tolower(c); });
				if(fext != "tiff" && fext != "tif" && fext != "g2s.tiff" && fext != "g2s.tif")
					continue;

				// We found a supported file type -> Increment the counter
				validfiles++;
			}
			return validfiles > 0;
		}
		else
		{
			// If file is selected check file extension
			auto fext = xpath.extension().u8string();
			std::transform(fext.begin(), fext.end(), fext.begin(), [](char c) { return (char)tolower(c); });
			return fext == "tiff" || fext == "tif" || fext == "g2s.tiff" || fext == "g2s.tif";
		}
	}
	catch(std::exception& e) 
	{ 
		LogMessage("CanLoad error: " + std::string(e.what()));
		return false; 
	}
}

/**
 * Discard closed dataset storage descriptors from cache
 * By default storage descriptors are preserved even after the dataset is closed
 * To reclaim memory all closed descritors are evicted from cache
 */
void G2SBigTiffStorage::cacheReduce() noexcept
{
   for(auto it = cache.begin(); it != cache.end(); )
   {
      if(!it->second.isOpen())
         it = cache.erase(it);
      else
         it++;
   }
}

/**
 * Scan folder subtree for supported files
 * @paramm path Folder path
 * @param listOfDatasets Dataset path list [out]
 * @param maxItems Max dataset count
 * @param maxItemLength Max dataset path length
 * @param cpos Current position in the list
 * @return Was provided buffer large enough to store all dataset paths
 */
bool G2SBigTiffStorage::scanDir(const std::string& path, char** listOfDatasets, int maxItems, int maxItemLength, int cpos) noexcept
{
	if(listOfDatasets == nullptr)
		return false;
   try
   {
      auto dp = std::filesystem::u8path(path);
      if(!std::filesystem::exists(dp))
         return true;
      auto dit = std::filesystem::directory_iterator(dp);
      if(!std::filesystem::is_directory(dp))
         return false;
      for(const auto& entry : dit)
      {
         // Skip auto folder paths
         auto fname = entry.path().filename().u8string();
         if(fname == "." || fname == "..")
            continue;

         // Skip regular files
         if(!std::filesystem::is_directory(entry))
            continue;
         
         // If the folder extension is invalid -> scan the subtree
			auto abspath = std::filesystem::absolute(entry).u8string();
         auto fext = entry.path().extension().u8string();
         if(fext.size() == 0)
            return scanDir(abspath, listOfDatasets, maxItems, maxItemLength, cpos);
         if(fext[0] == '.')
            fext = fext.substr(1);
         std::transform(fext.begin(), fext.end(), fext.begin(), [](char c) { return (char)tolower(c); });
         if(std::find(supportedFormats.begin(), supportedFormats.end(), fext) == supportedFormats.end())
				return scanDir(abspath, listOfDatasets, maxItems, maxItemLength, cpos);

         // We found a supported dataset folder
         // Check result buffer limit
         if(cpos >= maxItems || listOfDatasets[cpos] == nullptr)
            return false;

         // Add to results list
			strncpy(listOfDatasets[cpos], abspath.c_str(), maxItemLength);
         cpos++;
      }
      return true;
   }
   catch(std::filesystem::filesystem_error&)
   {
      return false;
   }
}

/**
 * Validate image coordinates
 * @param fs Dataset handle
 * @param coordinates Image coordinates
 * @param numCoordinates Coordinate count
 * @return Are coordinates valid
 */ 
bool G2SBigTiffStorage::validateCoordinates(const G2SBigTiffDataset* fs, int coordinates[], int numCoordinates, bool flexaxis0) noexcept
{
	if((std::size_t)numCoordinates != fs->getDimension() && (std::size_t)numCoordinates != fs->getDimension() - 2)
		return false;
	for(int i = 0; i < (int)fs->getDimension() - 2; i++)
	{
		if(coordinates[i] < 0)
			return false;
		if(coordinates[i] >= (int)fs->getActualShape()[i])
		{
			if(i > 0 || !flexaxis0)
				return false;
		}
	}
	return true;
}

/**
 * Get direct I/O property
 * @return Is direct I/O enabled
 */
bool G2SBigTiffStorage::getDirectIO() const noexcept
{
	char buf[MM::MaxStrLength];
	int ret = GetProperty(PROP_DIRECTIO, buf);
	if(ret != DEVICE_OK)
		return false;
	try
	{
		return std::atoi(buf) != 0;
	}
	catch(...) { return false; }
}

/**
 * Get flush cycle property
 * @return Flush cycle count
 */
int G2SBigTiffStorage::getFlushCycle() const noexcept
{
	char buf[MM::MaxStrLength];
	int ret = GetProperty(PROP_FLUSHCYCLE, buf);
	if(ret != DEVICE_OK)
		return 0;
	try
	{
		return std::atoi(buf);
	}
	catch(...) { return 0; }
}

/**
 * Get chunk size property
 * @return Chunk size - number of slowest changing dimension coordinates in a single file
 */
int G2SBigTiffStorage::getChunkSize() const noexcept
{
	char buf[MM::MaxStrLength];
	int ret = GetProperty(PROP_CHUNKSIZE, buf);
	if(ret != DEVICE_OK)
		return 0;
	try
	{
		return std::atoi(buf);
	}
	catch(...) { return 0; }
}
