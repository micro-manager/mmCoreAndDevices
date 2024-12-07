///////////////////////////////////////////////////////////////////////////////
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
#include "G2SBigTiffDataset.h"
#ifdef _WIN32
#include <Windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/fs.h>
#endif

#define G2SFOLDER_EXT					".g2s"
#define G2SFILE_EXT						".g2s.tif"

/**
 * Class constructor
 * Constructor doesn't open the file, just creates an object set sets the configuration
 * By convention G2S format files end with a .g2s.tif extension
 * First data chunk doesn't have a index (e.g. SampleDataset.g2s.tif)
 * Other data chunks contain an index (1-based, e.g. SampleDataset_1.g2s.tif, SampleDataset_2.g2s.tif..)
 * Dataset files are contained within a directory. The name of the directory matches the dataset name with the .g2s sufix (e.g. SampleDataset.g2s)
 */
G2SBigTiffDataset::G2SBigTiffDataset() noexcept
{
	dspath = "";
	datasetuid = "";
	bitdepth = 8;
	samples = 1;
	imgcounter = 0;
	flushcnt = 0;
	chunksize = 0;
	directIo = false;
	bigTiff = true;
	writemode = false;
}

/**
 * Create a dataset
 * All datasets are stored in separate folders, folder names have a .g2s suffix
 * If the folder with the specified name already exists, a name with the index in the suffix will be used
 * If the dataset is chunked files will be created only when the active chunk is filled
 * @param path Dataset (folder) path
 * @param dio Use direct I/O
 * @param fbig Use BigTIFF format
 * @param chunksz Chunk size
 * @throws std::runtime_error
 */
void G2SBigTiffDataset::create(const std::string& path, bool dio, bool fbig, std::uint32_t chunksz)
{
	if(isOpen())
		throw std::runtime_error("Invalid operation. Dataset is already created");
	if(path.empty())
		throw std::runtime_error("Unable to create a file stream. Dataset path is undefined");
	directIo = dio;
	writemode = true;
	chunksize = chunksz;

	// Extract dataset name
	std::filesystem::path basepath = std::filesystem::u8path(path);
	dsname = basepath.stem().u8string();
	if(dsname.find(".g2s") == dsname.size() - 4)
		dsname = dsname.substr(0, dsname.size() - 4);

	// Determine dataset path
	std::uint32_t counter = 1;
	std::filesystem::path xpath = basepath.parent_path() / (dsname + G2SFOLDER_EXT);
	while(std::filesystem::exists(xpath))
	{
		// If the file path (path + name) exists, it should not be an error
		// nor the file should be overwritten, first available suffix (index) will be appended to the file name
		auto tmpname = dsname + "_" + std::to_string(counter++) + G2SFOLDER_EXT;
		xpath = basepath.parent_path() / tmpname;
	}
	dspath = xpath.u8string();


	// Create a first file (data chunk)
	std::error_code ec;
	std::filesystem::path fp = xpath / (dsname + G2SFILE_EXT);
	std::filesystem::create_directories(fp.parent_path(), ec);
	if(ec.value() != 0)
		throw std::runtime_error("Unable to create a file stream. Directory tree creation failed");
	activechunk = std::make_shared<G2SBigTiffStream>(fp.u8string(), directIo);
	if(!activechunk)
		throw std::runtime_error("Unable to create a file stream. Data chunk allocation failed");
	activechunk->open(true);
	if(activechunk->getHeader().empty())
		throw std::runtime_error("Unable to create a file stream. File header creation failed");
	if(!datasetuid.empty())
	activechunk->writeDatasetUid(datasetuid);
	if(!shape.empty())
		activechunk->writeShapeInfo(shape, chunksize);
	datachunks.push_back(activechunk);
}

/**
 * Load a dataset
 * If the dataset doesn't exist an exception will be thrown
 * If the dataset exists dataset parameters and metadata will be parsed
 * If the dataset is chunked all files will be enumerated, but only the first file will be loaded
 * @param path Dataset (folder) path or File path of the first data chunk
 * @param dio Use direct I/O
 * @throws std::runtime_error
 */
void G2SBigTiffDataset::load(const std::string& path, bool dio)
{
	if(isOpen())
		throw std::runtime_error("Invalid operation. Dataset is already loaded");
	if(path.empty())
		throw std::runtime_error("Unable to load a dataset. Dataset path is undefined");
	directIo = dio;
	writemode = false;

	// Check dataset / file path
	auto xp = std::filesystem::u8path(path);
	if(!std::filesystem::exists(xp))
	{
		// Check if the dataset path has a .g2s extension
		std::string fpath(path);
		if(fpath.find(".g2s") != fpath.size() - 4)
			fpath += ".g2s";
		xp = std::filesystem::u8path(path);
		if(!std::filesystem::exists(xp))
			throw std::runtime_error("Unable to load a dataset. Specified path doesn't exist");
	}

	// If the first data chunk (file) path is specified -> use parent folder path
	if(std::filesystem::is_regular_file(xp))
		xp = xp.parent_path();
	dspath = xp.u8string();
	dsname = xp.stem().u8string();
	if(dsname.find(".g2s") == dsname.size() - 4)
		dsname = dsname.substr(0, dsname.size() - 4);

	// Enumerate files
	for(const auto& entry : std::filesystem::directory_iterator(xp))
	{
		// Skip auto folder paths
		auto fname = entry.path().filename().u8string();
		if(fname == "." || fname == "..")
			continue;

		// Skip folders
		if(std::filesystem::is_directory(entry))
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

		// We found a supported file type -> Add to results list
		auto abspath = std::filesystem::absolute(entry).u8string();
		auto dchunk = std::make_shared<G2SBigTiffStream>(abspath, directIo);
		datachunks.push_back(dchunk);
	}
	if(datachunks.empty())
		throw std::runtime_error("Unable to load a dataset. No files found");

	// Load first data chunk
	samples = 1;
	imgcounter = 0;
	metadata.clear();
	activechunk = datachunks.front();
	activechunk->open(false);
	activechunk->parse(datasetuid, shape, chunksize, metadata, bitdepth);
	imgcounter += activechunk->getImageCount();

	// Validate dataset parameters
	if(activechunk->getChunkIndex() != 0)
	{
		close();
		throw std::runtime_error("Unable to load a dataset. First data chunk is missing");
	}
	if(datasetuid.empty())
	{
		close();
		throw std::runtime_error("Unable to load a dataset. Invalid dataset UID");
	}
	if(shape.size() < 3)
	{
		close();
		throw std::runtime_error("Unable to load a dataset. Invalid dataset shape");
	}
	if(bitdepth < 8 || bitdepth > 16)
	{
		close();
		throw std::runtime_error("Unable to load a dataset. Unsupported pixel format");
	}

	// Parse headers for other data chunks
	for(std::size_t i = 1; i < datachunks.size(); i++)
	{
		validateDataChunk(i, false);
		imgcounter += datachunks[i]->getImageCount();
		datachunks[i]->close();
	}
}

/**
 * Close the dataset
 * If a dataset hasn't been created / loaded this method will have no effect
 * File handles will be released / closed
 * In the create mode during closing final section (dataset metadata) is commited to the first data chunk (file)
 */
void G2SBigTiffDataset::close() noexcept
{
	if(writemode && datachunks.size() == 1 && datachunks[0]->isOpen())
		datachunks[0]->appendMetadata(metadata);
	for(const auto& fx : datachunks)
		fx->close();
	imgcounter = 0;
	bitdepth = 8;
	samples = 1;
	metadata.clear();
	shape.clear();
	datachunks.clear();
	activechunk.reset();
}

/**
 * Set dataset shape / dimension & axis sizes
 * First two axis are always width and height
 * If the shape info is invalid this method will take no effect
 * Shape can only be set in the write mode, before adding any images
 * @param dims Axis sizes list
 * @throws std::runtime_error
 */
void G2SBigTiffDataset::setShape(const std::vector<std::uint32_t>& dims)
{
	if(dims.size() < 2)
		throw std::runtime_error("Unable to set dataset shape. Invalid shape info");
	if(!writemode)
		throw std::runtime_error("Unable to set dataset shape in read mode");
	if(datachunks.size() > 1)
		throw std::runtime_error("Unable to set dataset shape. Dataset configuration is already set");
	if(imgcounter > 0 && shape.size() >= 2)
	{
		if(dims.size() != shape.size())
			throw std::runtime_error("Unable to set dataset shape. Invalid axis count");
		if(dims[dims.size() - 2] != shape[shape.size() - 2] || dims[dims.size() - 1] != shape[shape.size() - 1])
			throw std::runtime_error("Unable to set dataset shape. Image dimensions don't match the existing image dimensions");
		return;
	}
	shape = dims;
	if(activechunk)
		activechunk->writeShapeInfo(shape, chunksize);
}

/**
 * Set dataset shape / dimension & axis sizes
 * First two axis are always width and height
 * If the shape info is invalid this method will take no effect
 * Shape can only be set in the write mode, before adding any images
 * @param dims Axis sizes list
 * @throws std::runtime_error
 */
void G2SBigTiffDataset::setShape(std::initializer_list<std::uint32_t> dims)
{
	if(dims.size() < 2)
		throw std::runtime_error("Unable to set dataset shape. Invalid shape info");
	if(!writemode)
		throw std::runtime_error("Unable to set dataset shape in read mode");
	if(datachunks.size() > 1)
		throw std::runtime_error("Unable to set dataset shape. Dataset configuration is already set");
	if(imgcounter > 0 && shape.size() >= 2)
	{
		if(dims.size() != shape.size())
			throw std::runtime_error("Unable to set dataset shape. Invalid axis count");
		if(*(dims.end() - 2) != shape[shape.size() - 2] || *(dims.end() - 1) != shape[shape.size() - 1])
			throw std::runtime_error("Unable to set dataset shape. Image dimensions don't match the existing image dimensions");
		return;
	}
	shape = dims;
	if(activechunk)
		activechunk->writeShapeInfo(shape, chunksize);
}

/**
 * Set pixel format
 * If the pixel format is invalid this method will take no effect
 * Pixel format can only be set in the write mode, before adding any images
 * @param depth Bit depth (bits per sample)
 * @parma vsamples Samples per pixel
 * @throws std::runtime_error
 */
void G2SBigTiffDataset::setPixelFormat(std::uint8_t depth, std::uint8_t vsamples)
{
	if(!writemode)
		throw std::runtime_error("Unable to set pixel format in read mode");
	if(datachunks.size() > 1)
		throw std::runtime_error("Unable to set pixel format. Dataset configuration is already set");
	if(imgcounter > 0)
	{
		if(bitdepth != depth || samples != vsamples)
			throw std::runtime_error("Unable to set pixel format. Specified pixel format doesn't match current pixel format");
		return;
	}
	bitdepth = depth;
	samples = vsamples;
}

/**
 * Set dataset metadata
 * Metadata will be stored in byte buffer whose size is 1 byte larger than the metadata string length
 * @param meta Metadata string
 */
void G2SBigTiffDataset::setMetadata(const std::string& meta)
{
	if(!writemode)
		throw std::runtime_error("Unable to set dataset metadata in read mode");
	
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
void G2SBigTiffDataset::setUID(const std::string& val)
{
	if(!writemode)
		throw std::runtime_error("Unable to set dataset UID in read mode");
	if(datachunks.size() > 1)
		throw std::runtime_error("Unable to set dataset UID. Dataset configuration is already set");
	
	if(val.empty())
		datasetuid = val;
	else
	{
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
	}

	// Update file header
	if(activechunk)
		activechunk->writeDatasetUid(datasetuid);
}

/**
 * Get dataset metadata
 * If metadata is specified value will be returned from cache, otherwise it will be read from a file stream
 * @return Metadata string
 */
std::string G2SBigTiffDataset::getMetadata() const noexcept
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
 * @throws std::runtime_error
 */
std::string G2SBigTiffDataset::getImageMetadata(const std::vector<std::uint32_t>& coord)
{
	if(!isOpen())
		throw std::runtime_error("Invalid operation. No open file stream available");
	if(imgcounter == 0)
		throw std::runtime_error("Invalid operation. No images available");
	
	// Select current image (IFD)
	if(!coord.empty())
		selectImage(coord);
	else if(activechunk->getCurrentIFD().empty())
		// Load IFD
		activechunk->loadIFD(activechunk->getCurrentIFDOffset());

	return activechunk->getImageMetadata();
}

/**
 * Add image / write image to the file
 * Images are added sequentially
 * Image data is stored uncompressed
 * Metadata is stored in plain text, after the pixel data
 * Image IFD is stored before pixel data
 * If the new image doesn't belong to the current chunk, a new file will be created automatically, and the current one will be closed
 * @param buff Image buffer
 * @param len Image buffer length
 * @param meta Image metadata (optional)
 * @throws std::runtime_error
 */
void G2SBigTiffDataset::addImage(const unsigned char* buff, std::size_t len, const std::string& meta)
{
	if(!isOpen())
		throw std::runtime_error("Invalid operation. No open file stream available");
	if(!writemode)
		throw std::runtime_error("Invalid operation. Unable to add images in read mode");
	if(shape.size() < 2)
		throw std::runtime_error("Invalid operation. Dataset shape is not defined");
	if(!bigTiff && len > TIFF_MAX_BUFFER_SIZE)
		throw std::runtime_error("Invalid operation. Image data is too long");
	if(!bigTiff && meta.size() > TIFF_MAX_BUFFER_SIZE)
		throw std::runtime_error("Invalid operation. Metadata string is too large");

	// Check active data chunk
	if(chunksize > 0 && imgcounter > 0 && imgcounter % getChunkImageCount() == 0)
	{
		// Close current data chunk
		// Only the first data chunk should contain dataset metadata
		if(datachunks.size() == 1)
			activechunk->appendMetadata(metadata);
		activechunk->close();
		
		// Create new data chunk
		std::filesystem::path fp = std::filesystem::path(dspath) / (dsname + "_" + std::to_string(datachunks.size()) + G2SFILE_EXT);
		activechunk = std::make_shared<G2SBigTiffStream>(fp.u8string(), directIo, bigTiff, (std::uint32_t)datachunks.size());
		if(!activechunk)
			throw std::runtime_error("Unable to add an image. Data chunk allocation failed");
		activechunk->open(true);
		if(activechunk->getHeader().empty())
			throw std::runtime_error("Unable to add an image. File header creation failed");
		activechunk->writeDatasetUid(datasetuid);
		activechunk->writeShapeInfo(shape, chunksize);
		datachunks.push_back(activechunk);
	}

	// Check file size limits
	activechunk->addImage(buff, len, getWidth(), getHeight(), bitdepth, meta);
	imgcounter++;

	// Flush pending data
	if(flushcnt > 0 && activechunk->getImageCount() % flushcnt == 0)
		activechunk->flush();
}

/**
 * Get image data (pixel buffer)
 * If the coordinates are not specified images are read sequentially
 * This method will change (advance) the current image
 * If this method is called after the last available image (in sequential mode), or with invalid coordinates an exception will be thrown
 * @param coord Image coordinates
 * @return Image data
 * @throws std::runtime_error
 */
std::vector<unsigned char> G2SBigTiffDataset::getImage(const std::vector<std::uint32_t>& coord)
{
	if(!isOpen())
		throw std::runtime_error("Invalid operation. No open file stream available");
	if(imgcounter == 0)
		throw std::runtime_error("Invalid operation. No images available");

	// Select current image (IFD)
	if(!coord.empty())
		selectImage(coord);
	else
		advanceImage();
	return activechunk->getImage();	
}

/**
 * Change active data chunk
 * This method is used only for reading data
 * Dataset properties from the new data chunk will be validated
 * @param chunkind Data chunk index
 * @throws std::runtime_error
 */
void G2SBigTiffDataset::switchDataChunk(std::uint32_t chunkind)
{
	// Validate next data chunk
	validateDataChunk(chunkind, true);

	// Change active data chunk
	activechunk->close();
	activechunk = datachunks[chunkind];	
}

/**
 * Validate data chunk
 * Data chunk (file stream) will be opened in order to parse the header
 * File stream won't be closed unless validation fails
 * @param chunkind Data chunk index
 * @param index Index data chunk IFDs
 * @throws std::runtime_error
 */
void G2SBigTiffDataset::validateDataChunk(std::uint32_t chunkind, bool index)
{
	std::string ldataseuid = "";
	std::vector<std::uint32_t> lshape;
	std::uint32_t lchunksz = 0;
	std::vector<unsigned char> lmetadata;
	std::uint8_t lbitdepth = 0;
	
	// Open & parse data chunk (file)
	datachunks[chunkind]->open(false);
	datachunks[chunkind]->parse(ldataseuid, lshape, lchunksz, lmetadata, lbitdepth, index);

	// Validate dataset properties
	if(datasetuid != ldataseuid)
	{
		datachunks[chunkind]->close();
		throw std::runtime_error("Invalid data chunk. Dataset UID missmatch");
	}
	if(shape.size() != lshape.size())
	{
		datachunks[chunkind]->close();
		throw std::runtime_error("Invalid data chunk. Dataset shape missmatch");
	}
	for(std::size_t i = 0; i < shape.size(); i++)
	{
		if(shape[i] != lshape[i])
		{
			datachunks[chunkind]->close();
			throw std::runtime_error("Invalid data chunk. Axis " + std::to_string(i) + " size missmatch");
		}
	}
	if(chunksize != lchunksz)
	{
		datachunks[chunkind]->close();
		throw std::runtime_error("Invalid data chunk. Chunk size missmatch");
	}
	if(index && bitdepth != lbitdepth)
	{
		datachunks[chunkind]->close();
		throw std::runtime_error("Invalid data chunk. Pixel format missmatch");
	}
}

/**
 * Select image
 * This method will automatically switch active data chunk
 * @param coord Image coordinates
 * @throws std::runtime_error
 */
void G2SBigTiffDataset::selectImage(const std::vector<std::uint32_t>& coord)
{
	if(coord.empty())
		return;
	if(!activechunk)
		throw std::runtime_error("Invalid operation. Invalid data chunk");

	// Calculate data chunk & image index
	std::uint32_t chunkind = 0, imgind = 0;
	calcImageIndex(coord, chunkind, imgind);
	if(chunkind != activechunk->getChunkIndex())
		switchDataChunk(chunkind);
	if(imgind >= activechunk->getIFDOffsets().size())
		throw std::runtime_error("Invalid operation. Invalid image coordinates");

	// Change current image & load IFD
	activechunk->setCurrentImage(imgind);
	activechunk->loadIFD(activechunk->getIFDOffsets()[imgind]);
}

/**
 * Advance current image
 * This method will automatically switch active data chunk
 * @throws std::runtime_error
 */
void G2SBigTiffDataset::advanceImage()
{
	if(!activechunk)
		throw std::runtime_error("Invalid operation. Invalid data chunk");
	if(activechunk == datachunks.back() && (activechunk->getCurrentImage() + 1 > activechunk->getImageCount() || activechunk->getNextIFDOffset() == 0))
		throw std::runtime_error("Invalid operation. No more images available");

	// Check if need to switch the data chunk
	if(activechunk->getCurrentImage() + 1 == activechunk->getImageCount())
		switchDataChunk(activechunk->getChunkIndex() + 1);

	// Clear current IFD before advancing
	// In a case where getImageMetadata() is called before any getImage() call
	// we should skip clearing the current IFD; this works only for the first image
	if(activechunk->getCurrentImage() > 0)
		activechunk->advanceIFD();

	// Advance current image
	activechunk->setCurrentImage(activechunk->getCurrentImage() + 1);

	// Load IFD (skip if already loaded by the getImageMetadata())
	if(activechunk->getCurrentIFD().empty())
		activechunk->loadNextIFD();
}

/**
 * Get number of images in a data chunk
 * Data chunk represents a dataset subset with one or more slowest changing dimension coordinates
 * If chunking is turned OFF this method will return 0
 * @return Image count
 */
std::uint32_t G2SBigTiffDataset::getChunkImageCount() const noexcept
{
	if(chunksize == 0)
		return 0;
	std::uint32_t ret = 1;
	for(std::size_t i = 1; i < shape.size() - 2; i++)
		ret *= shape[i];
	return chunksize * ret;
}

/**
 * Calculate image index from image coordinates
 * Image coordiantes should not contain indices for the last two dimensions (width & height)
 * By convention image acquisitions loops through the coordinates in the descending order (higher coordinates are looped first)
 * E.g. ZTC order means that all channels are acquired before changing the time point, and all specified time points 
 * are acquired before moving the Z-stage, in which case dataset with the shape 2-4-3 for coordinates 1-2-1 will return 19 (=1*12 + 2*3 + 1*1)
 * First image coordinate can go beyond the specified shape size
 * @param coord Image coordinates
 * @param chunkind Data chunk index [out]
 * @param imgind Image index (in the data chunk) [out]
 * @throws std::runtime_error
 */
void G2SBigTiffDataset::calcImageIndex(const std::vector<std::uint32_t>& coord, std::uint32_t& chunkind, std::uint32_t& imgind) const
{
	// Validate coordinates count
	if(coord.size() > shape.size() - 2)
		throw std::runtime_error("Invalid number of coordinates");
	if(chunkind >= datachunks.size() || (chunkind > 0 && chunksize == 0))
		throw std::runtime_error("Invalid data chunk index");
	if(coord.empty())
	{
		chunkind = 0;
		imgind = 0;
		return;
	}

	// Validate ranges for all axis (except the first)
	for(std::size_t i = 1; i < coord.size(); i++)
	{
		if(coord[i] >= shape[i])
			throw std::runtime_error("Invalid coordinate for dimension " + std::to_string(i + 2));
	}

	// Determine chunk index
	if(chunksize == 0)
		chunkind = 0;
	else
		chunkind = coord[0] / chunksize;

	// Adjust slowest changing dimension index to set the base for image index calculation
	std::vector<std::uint32_t> lcoord = coord;
	std::uint32_t baseind = chunkind * chunksize;
	lcoord[0] -= baseind;

	// Calculate image index
	std::uint32_t ind = 0;
	for(int i = 0; i < lcoord.size(); i++)
	{
		if(lcoord[i] == 0)
			continue;
		std::uint32_t sum = 1;
		for(int j = i + 1; j < shape.size() - 2; j++)
			sum *= shape[j];
		ind += sum * lcoord[i];
	}
	imgind = ind;
}
