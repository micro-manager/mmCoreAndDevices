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
G2SBigTiffStorage::G2SBigTiffStorage() : initialized(false)
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
int G2SBigTiffStorage::Shutdown()
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
 * @param handle Entry GUID [out]
 * @return Status code
 */
int G2SBigTiffStorage::Create(const char* path, const char* name, int numberOfDimensions, const int shape[], MM::StorageDataType pixType, const char* meta, char* handle)
{
   if (path == nullptr)
      return ERR_TIFF_INVALID_PATH;

	if (numberOfDimensions < 3)
		return ERR_TIFF_INVALID_DIMENSIONS;

	if (!(pixType == MM::StorageDataType::StorageDataType_GRAY16 || 
		pixType == MM::StorageDataType::StorageDataType_GRAY8 || 
		pixType == MM::StorageDataType::StorageDataType_RGB32))
		return ERR_TIFF_INVALID_PIXEL_TYPE;

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
	std::string guid = boost::lexical_cast<std::string>(boost::uuids::random_generator()());           // Entry UUID
	if(guid.size() > MM::MaxStrLength)
	{
#ifdef _NDEBUG
		guid = guid.substr(0, MM::MaxStrLength);
#else
		assert("Dataset handle size is too long");
#endif
	}
   
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
   
   G2SStorageEntry sdesc(fhandle->getPath(), numberOfDimensions, shape, meta);
   sdesc.FileHandle = fhandle;

	// Set dataset UUID / shape / metadata
	std::vector<std::uint32_t> vshape;
	vshape.assign(shape, shape + numberOfDimensions);
	fhandle->setUID(guid);
	fhandle->setShape(vshape);
	fhandle->setMetadata(meta);

	// Set pixel format
	if(pixType == MM::StorageDataType::StorageDataType_GRAY8)
		fhandle->setPixelFormat(8, 1);
	else if(pixType == MM::StorageDataType::StorageDataType_GRAY16)
		fhandle->setPixelFormat(16, 1);
	else if(pixType == MM::StorageDataType::StorageDataType_RGB32)
		fhandle->setPixelFormat(8, 4);

   // Append dataset storage descriptor to cache
   auto it = cache.insert(std::make_pair(guid, sdesc));
	if(it.first == cache.end())
	{
		delete fhandle;
		return ERR_TIFF_CACHE_INSERT;
	}
   if(!it.second)
	{
		// Dataset already exists
		delete fhandle;
      return ERR_TIFF_CACHE_INSERT;
	}

   // Copy UUID string to the GUID buffer
	strncpy(handle, guid.c_str(), MM::MaxStrLength);
   return DEVICE_OK;
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
int G2SBigTiffStorage::Load(const char* path, char* handle)
{
   if(path == nullptr)
      return DEVICE_INVALID_INPUT_PARAM;

   // Check if the file exists
	std::filesystem::path actpath = std::filesystem::u8path(path);
   if(!std::filesystem::exists(actpath))
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
			return DEVICE_INVALID_INPUT_PARAM;
	}
	else if(std::filesystem::is_regular_file(actpath))
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
		return ERR_TIFF_OPEN_FAILED;

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
		return ERR_TIFF_OPEN_FAILED;
	}
	
	// Obtain / generate dataset UID
   std::string guid = fhandle->getUID().empty() ? boost::lexical_cast<std::string>(boost::uuids::random_generator()()) : fhandle->getUID();
	if(guid.size() > MM::MaxStrLength)
	{
		delete fhandle;
		return DEVICE_INVALID_PROPERTY_LIMTS;
	}

   // Append dataset storage descriptor to cache
	if(cit == cache.end())
	{
		// Create dataset storage descriptor
		G2SStorageEntry sdesc(std::filesystem::absolute(actpath).u8string(), (int)fhandle->getDimension(), reinterpret_cast<int*>(&fhandle->getShape()[0]), fhandle->getMetadata().empty() ? nullptr : fhandle->getMetadata().c_str());
		sdesc.FileHandle = fhandle;

		auto it = cache.insert(std::make_pair(guid, sdesc));
		if(it.first == cache.end())
		{
			delete fhandle;
			return DEVICE_OUT_OF_MEMORY;
		}
	}

   // Copy UUID string to the GUID buffer
	strncpy(handle, guid.c_str(), MM::MaxStrLength);
   return DEVICE_OK;
}

/**
 * Get dataset shape
 * Shape contains image width and height as first two dimensions
 * @param handle Entry GUID
 * @param shape Dataset shape [out]
 * @return Status code
 */
int G2SBigTiffStorage::GetShape(const char* handle, int shape[])
{
	if(handle == nullptr || shape == nullptr)
		return DEVICE_INVALID_INPUT_PARAM;

	// Obtain dataset descriptor from cache
	auto it = cache.find(handle);
	if(it == cache.end())
		return DEVICE_INVALID_INPUT_PARAM;

	auto fs = reinterpret_cast<G2SBigTiffDataset*>(it->second.FileHandle);
	for(std::size_t i = 0; i < fs->getDimension(); i++)
		shape[i] = fs->getShape()[i];
   return DEVICE_OK;
}

/**
 * Get dataset pixel format
 * @param handle Entry GUID
 * @param pixelDataType Pixel format [out]
 * @return Status code
 */
int G2SBigTiffStorage::GetDataType(const char* handle, MM::StorageDataType& pixelDataType)
{
	if(handle == nullptr)
		return DEVICE_INVALID_INPUT_PARAM;

	// Obtain dataset descriptor from cache
	auto it = cache.find(handle);
	if(it == cache.end())
		return DEVICE_INVALID_INPUT_PARAM;

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
int G2SBigTiffStorage::Close(const char* handle)
{
   auto it = cache.find(handle);
   if(it == cache.end())
      return DEVICE_INVALID_INPUT_PARAM;
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
int G2SBigTiffStorage::Delete(char* handle)
{
   if(handle == nullptr)
      return DEVICE_INVALID_INPUT_PARAM;
   auto it = cache.find(handle);
   if(it == cache.end())
      return DEVICE_INVALID_INPUT_PARAM;

   // Check if the file exists
   auto fp = std::filesystem::u8path(it->second.Path);
   if(!std::filesystem::exists(fp))
      return DEVICE_NO_PROPERTY_DATA;

   // Close the file handle
   if(it->second.isOpen())
   {
		auto fs = reinterpret_cast<G2SBigTiffDataset*>(it->second.FileHandle);
		fs->close();
		it->second.close();
		delete fs;
   }
   
   // Delete the file
   if(!std::filesystem::remove(fp))
      return DEVICE_ERR;

   // Discard the cache entry
   cache.erase(it);
   return DEVICE_OK;
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
int G2SBigTiffStorage::List(const char* path, char** listOfDatasets, int maxItems, int maxItemLength)
{
   if(path == nullptr || listOfDatasets == nullptr || maxItems <= 0 || maxItemLength <= 0)
      return DEVICE_INVALID_INPUT_PARAM;
   auto dp = std::filesystem::u8path(path);
   if(!std::filesystem::exists(dp) || !std::filesystem::is_directory(dp))
      return DEVICE_INVALID_INPUT_PARAM;
   auto allfnd = scanDir(path, listOfDatasets, maxItems, maxItemLength, 0);
   return allfnd ? DEVICE_OK : DEVICE_SEQUENCE_TOO_LARGE;
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
 * @return Status code
 */
int G2SBigTiffStorage::AddImage(const char* handle, int sizeInBytes, unsigned char* pixels, int coordinates[], int numCoordinates, const char* imageMeta)
{
	if(handle == nullptr || pixels == nullptr || sizeInBytes <= 0 || numCoordinates <= 0)
		return DEVICE_INVALID_INPUT_PARAM;

	// Obtain dataset descriptor from cache
	auto it = cache.find(handle);
	if (it == cache.end())
		return DEVICE_INVALID_INPUT_PARAM;

	// Validate image dimensions
	auto fs = reinterpret_cast<G2SBigTiffDataset*>(it->second.FileHandle);
	if(!validateCoordinates(fs, coordinates, numCoordinates))
		return DEVICE_INVALID_INPUT_PARAM;

	// Add image
	fs->addImage(pixels, sizeInBytes, imageMeta);

	// Add image metadata to the dataset cache
	auto ikey = getImageKey(coordinates, numCoordinates);
	it->second.ImageMetadata.insert(std::make_pair(ikey, std::string(imageMeta)));
   return DEVICE_OK;
}

/**
 * Get dataset summary metadata
 * If the netadata size is longer than the provided buffer, only the first [bufSize] bytes will be
 * copied, and the status code 'DEVICE_SEQUENCE_TOO_LARGE' will be returned
 * @param handle Entry GUID
 * @param meta Metadata buffer [out]
 * @param bufSize Buffer size
 * @return Status code
 */
int G2SBigTiffStorage::GetSummaryMeta(const char* handle, char* meta, int bufSize)
{
   if(handle == nullptr || meta == nullptr || bufSize <= 0)
      return DEVICE_INVALID_INPUT_PARAM;

	// Obtain dataset descriptor from cache
   auto it = cache.find(handle);
   if(it == cache.end())
      return DEVICE_INVALID_INPUT_PARAM;

	// Copy metadata string
	strncpy(meta, it->second.Metadata.c_str(), bufSize);
   return it->second.Metadata.size() > (std::size_t)bufSize ? DEVICE_SEQUENCE_TOO_LARGE : DEVICE_OK;
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
int G2SBigTiffStorage::GetImageMeta(const char* handle, int coordinates[], int numCoordinates, char* meta, int bufSize)
{
   if(handle == nullptr || coordinates == nullptr || numCoordinates == 0 || meta == nullptr || bufSize <= 0)
      return DEVICE_INVALID_INPUT_PARAM;

	// Obtain dataset descriptor from cache
   auto it = cache.find(handle);
   if(it == cache.end())
      return DEVICE_INVALID_INPUT_PARAM;

	auto fs = reinterpret_cast<G2SBigTiffDataset*>(it->second.FileHandle);
	if(!validateCoordinates(fs, coordinates, numCoordinates))
		return DEVICE_INVALID_INPUT_PARAM;
   
	// Check the dataset cache first
	auto ikey = getImageKey(coordinates, numCoordinates);
   auto iit = it->second.ImageMetadata.find(ikey);
   if(iit == it->second.ImageMetadata.end())
	{
      // Obtain metadata from the file stream
		if(!it->second.isOpen())
			return ERR_TIFF_STREAM_UNAVAILABLE;

		// Copy coordinates without including the width and height
		std::vector<std::uint32_t> coords(fs->getDimension() - 2);
		for(int i = 0; i < coords.size(); i++)
		{
			if(i >= numCoordinates)
				break;
			coords[i] = coordinates[i];
		}

		auto fmeta = fs->getImageMetadata(coords);
		if(!fmeta.empty())
			strncpy(meta, fmeta.c_str(), bufSize);
	}
	else
	{
		// Copy metadata from cache
		if(!iit->second.empty())
			strncpy(meta, iit->second.c_str(), bufSize);
	}
   return DEVICE_OK;
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
const unsigned char* G2SBigTiffStorage::GetImage(const char* handle, int coordinates[], int numCoordinates)
{
	if(handle == nullptr || numCoordinates <= 0)
		return nullptr;

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
		if(i >= numCoordinates)
			break;
		coords[i] = coordinates[i];
	}
	
	it->second.ImageData = fs->getImage(coords);
   return &it->second.ImageData[0];
}

/**
 * Configure metadata for a given dimension
 * @param handle Entry GUID
 * @param dimension Dimension index
 * @param name Name of the dimension
 * @param meaning Z,T,C, etc. (physical meaning)
 * @return Status code
 */
int G2SBigTiffStorage::ConfigureDimension(const char* handle, int dimension, const char* name, const char* meaning)
{
   if(handle == nullptr || dimension < 0)
      return DEVICE_INVALID_INPUT_PARAM;
   auto it = cache.find(handle);
   if(it == cache.end())
      return DEVICE_INVALID_INPUT_PARAM;
   if((std::size_t)dimension >= it->second.getDimSize())
      return DEVICE_INVALID_INPUT_PARAM;
   it->second.Dimensions[dimension].Name = std::string(name);
   it->second.Dimensions[dimension].Metadata = std::string(meaning);
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
int G2SBigTiffStorage::ConfigureCoordinate(const char* handle, int dimension, int coordinate, const char* name)
{
   if(handle == nullptr || dimension < 0 || coordinate < 0)
      return DEVICE_INVALID_INPUT_PARAM;
   auto it = cache.find(handle);
   if(it == cache.end())
      return DEVICE_INVALID_INPUT_PARAM;
   if((std::size_t)dimension >= it->second.getDimSize())
      return DEVICE_INVALID_INPUT_PARAM;
   if((std::size_t)coordinate >= it->second.Dimensions[dimension].getSize())
      return DEVICE_INVALID_INPUT_PARAM;
   it->second.Dimensions[dimension].Coordinates[coordinate] = std::string(name);
   return DEVICE_OK;
}

/**
 * Get number of dimensions
 * @param handle Entry GUID
 * @param numDimensions Number of dimensions [out]
 * @return Status code
 */
int G2SBigTiffStorage::GetNumberOfDimensions(const char* handle, int& numDimensions)
{
   if(handle == nullptr)
      return DEVICE_INVALID_INPUT_PARAM;
   auto it = cache.find(handle);
   if(it == cache.end())
      return DEVICE_INVALID_INPUT_PARAM;
   numDimensions = (int)it->second.getDimSize();
   return DEVICE_OK;
}

/**
 * Get number of dimensions
 * @param handle Entry GUID
 * @return Status code
 */
int G2SBigTiffStorage::GetDimension(const char* handle, int dimension, char* name, int nameLength, char* meaning, int meaningLength)
{
   if(handle == nullptr || dimension < 0 || meaningLength <= 0)
      return DEVICE_INVALID_INPUT_PARAM;
   auto it = cache.find(handle);
   if(it == cache.end())
      return DEVICE_INVALID_INPUT_PARAM;
   if((std::size_t)dimension >= it->second.getDimSize())
      return DEVICE_INVALID_INPUT_PARAM;
   if(it->second.Dimensions[dimension].Name.size() > (std::size_t)nameLength)
      return DEVICE_INVALID_PROPERTY_LIMTS;
   if(it->second.Dimensions[dimension].Metadata.size() > (std::size_t)meaningLength)
      return DEVICE_INVALID_PROPERTY_LIMTS;
	strncpy(name, it->second.Dimensions[dimension].Name.c_str(), nameLength);
	strncpy(meaning, it->second.Dimensions[dimension].Metadata.c_str(), meaningLength);
   return DEVICE_OK;
}

/**
 * Get number of dimensions
 * @param handle Entry GUID
 * @return Status code
 */
int G2SBigTiffStorage::GetCoordinate(const char* handle, int dimension, int coordinate, char* name, int nameLength)
{
   if(handle == nullptr || dimension < 0 || coordinate < 0 || nameLength <= 0)
      return DEVICE_INVALID_INPUT_PARAM;
   auto it = cache.find(handle);
   if(it == cache.end())
      return DEVICE_INVALID_INPUT_PARAM;
   if((std::size_t)dimension >= it->second.getDimSize())
      return DEVICE_INVALID_INPUT_PARAM;
   if((std::size_t)coordinate >= it->second.Dimensions[dimension].getSize())
      return DEVICE_INVALID_INPUT_PARAM;
   if(it->second.Dimensions[dimension].Coordinates[coordinate].size() > (std::size_t)nameLength)
      return DEVICE_INVALID_PROPERTY_LIMTS;
	strncpy(name, it->second.Dimensions[dimension].Coordinates[coordinate].c_str(), nameLength);
   return DEVICE_OK;
}

/**
 * Returns true if images can be added to the dataset
 */
bool G2SBigTiffStorage::IsOpen(const char* handle)
{
	// TODO: is this the correct way to check if the dataset is accepting images?
	auto it = cache.find(handle);
	if (it == cache.end())
		return false;

	return true;
}

int G2SBigTiffStorage::GetPath(const char* handle, char* path, int maxPathLength)
{
	auto it = cache.find(handle);
	if (it == cache.end())
		return ERR_TIFF_HANDLE_INVALID;
	strncpy(path, it->second.Path.c_str(), it->second.Path.size());
	return DEVICE_OK;
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
         if(cpos >= maxItems)
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
bool G2SBigTiffStorage::validateCoordinates(const G2SBigTiffDataset* fs, int coordinates[], int numCoordinates) noexcept
{
	if((std::size_t)numCoordinates != fs->getDimension() && (std::size_t)numCoordinates != fs->getDimension() - 2)
		return false;
	for(int i = 0; i < (int)fs->getDimension() - 2; i++)
	{
		if(coordinates[i] < 0 || coordinates[i] >= (int)fs->getShape()[i])
			return false;
	}
	return true;
}

/**
 * Calculate image key from the specified image coordinates
 * @param coordinates Image coordinates
 * @param numCoordinates Coordinate count
 * @return Image key (for image cache indices)
 */
std::string G2SBigTiffStorage::getImageKey(int coordinates[], int numCoordinates) noexcept
{
   std::stringstream ss;
   for(int i = 0; i < numCoordinates; i++)
      ss << (i == 0 ? "" : ".") << coordinates[i];
   return ss.str();
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
