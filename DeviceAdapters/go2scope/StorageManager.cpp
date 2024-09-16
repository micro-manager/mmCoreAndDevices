// ==============================================================================================================================
// 
// Go2Scope
// Cloud platform for Digital Microscopy
// http://www.go2scope.com
//
// Go2Scope Server (Device) Components
// Go2Scope Server
// Storage Manager: provides local storage functions for datasets and generic files
// 
// Copyright (C) 2012 by Luminous Point LLC. All rights reserved.
// http://www.luminous-point.com
//
// ==============================================================================================================================

#include "StorageManager.h"
#include <cstdio>
#include <iostream>
#include <sstream>
#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>

#include <windows.h>

#define HOUSEKEEPING_INTERVAL_SEC		60

g2s::device::StorageManager::StorageManager(MM::Device* parentDevice) noexcept :
	dirPath("file-storage"), saveBuffer(saveBufferDepth), execThd(nullptr), workerActive(false), asyncError(false), shutdownFlag(false), mmDevice
{
}

g2s::device::StorageManager::~StorageManager()
{
	shutdownFlag = true;
	if(execThd)
	{
		execThd->interrupt();
		execThd->join();
		delete execThd;
	}
	if(housekeepingThd)
	{
		housekeepingThd->join();
		housekeepingThd.reset();
	}

	auto it = datasets.begin();
	while (it != datasets.end())
	{
		it->second->close();
		delete it->second;
		it++;
	}
	datasets.clear();
}

/**
 * \brief Initialize Storage Manager instance.
 * \return true if successful
 */
bool g2s::device::StorageManager::initialize() noexcept
{
	// start worker save thread
	saveBuffer.clear();
	if(!execThd)
		execThd = new boost::thread(&g2s::device::StorageManager::saveWorker, this);

	// start housekeeping thread
	if(!housekeepingThd)
		housekeepingThd = std::make_unique<std::thread>(&g2s::device::StorageManager::housekeepingWorker, this);

	boost::filesystem::path p = boost::filesystem::path(dirPath);
	if (boost::filesystem::exists(p))
		return true;

	boost::system::error_code ec;
	if(!boost::filesystem::create_directories(p, ec))
	{
		g2s::G2SLogger::Main().log(ec.message());
		return false;
	}
	return true;
}

/**
 * \brief Initializes the Storage Manager with configurable parameters
 * \param devName - qualified go2scope.com microscope name
 * \param path - root path of the storage space
 * \return true if successful
 */
bool g2s::device::StorageManager::initialize(const std::string& devName, const std::string& path) noexcept
{
	deviceName = devName;
	dirPath = path;
	return initialize();
}


/**
 * \brief Insert image to dataset using asynchronous buffer. This function returns immediately, not waiting for file to be saved.
 * \param handle - dataset name
 * \param pixels - pixel buffer
 * \param frame 
 * \param channel 
 * \param slice 
 * \param position 
 * \param imageMeta - image metadata
 */
void g2s::device::StorageManager::insertImageAsync(const std::string& handle, std::vector<uint8_t>&& pixels, int frame, int channel, int slice, int position, const std::string& imageMeta)
{
	auto i = datasets.find(handle);
	if (i == datasets.end())
		throw G2SException("Dataset is not open: " + handle);

	if (!workerActive)
		throw G2SException("Dataset Storage thread is not active");

	// check if there were any save errors
	if (asyncError && (asyncErrorDataset == handle || asyncErrorDataset.empty()))
		throw G2SException(asyncErrorText);

	// place in queue for worker thread to save to disk
	StorageItem* si = new StorageItem(handle, pixels, frame, channel, slice, position, imageMeta);
	saveBuffer.push_front(si);
}

/**
 * \brief Insert image to dataset. This function blocks until the file is commited to disk.
 * \param handle - dataset name
 * \param pixels - pixel buffer
 * \param frame
 * \param channel
 * \param slice
 * \param position
 * \param imageMeta - image metadata
 */void g2s::device::StorageManager::insertImage(const std::string& handle, std::vector<uint8_t>& pixels, int frame, int channel, int slice, int position, const std::string& imageMeta)
{
	std::lock_guard<std::mutex> lock(storageLock);

	auto i = datasets.find(handle);
	if (i == datasets.end()) {
		G2SException e;
		e.setMessage("Dataset is not open: " + handle);
		e.setParameters(handle);
		throw e;
	}
	i->second->addImage(pixels, position, channel, slice, frame, G2SConfig(imageMeta));
}

/**
 * \brief Worker thread for the image save queue.
 */
void g2s::device::StorageManager::saveWorker()
{
	workerActive = true;
	for (;;)
	{
		try
		{
			if (saveBuffer.is_not_empty())
			{
				StorageItem* si;
				saveBuffer.pop_back(&si);
				std::unique_ptr<StorageItem> smartSi(si);
				insertImage(si->handle, si->pixels, si->frame, si->channel, si->slice, si->position, si->imageMeta);
			}
			else
			{
				boost::this_thread::sleep(boost::posix_time::milliseconds(threadSleepMs));
			}
		}
		catch (boost::thread_interrupted&)
		{
			workerActive = false;
			return;
		}
		catch (G2SException& e)
		{
			// error occured during saving images
			asyncError = true;
			asyncErrorText = e.msg();
			asyncErrorDataset = e.parameters();

			// clear the queue
			saveBuffer.clear();
		}
		catch (...)
		{
			asyncError = true;
			asyncErrorText = "unknown";
			asyncErrorDataset = "";
			// clear the queue
			saveBuffer.clear();
		}
	}
}

/**
 * Housekeeping worker thread
 */
void g2s::device::StorageManager::housekeepingWorker() noexcept
{
	while(!shutdownFlag)
	{
		try
		{
			bool autoManage = G2SConfigManager::Instance().isStorageAutoManaged();
			int minFreeSpaceMB = G2SConfigManager::Instance().getStorageThreshold();
			int datasetsInLocalIndex = getLocalIndexSize();
			// TODO: Add datasets access mutex
			if(!isLocalIndexEmpty() && autoManage && minFreeSpaceMB > 0)
			{
				int freespaceMb = getFreeSpaceMB();
				while(!isLocalIndexEmpty() && freespaceMb <= minFreeSpaceMB && datasets.size() < datasetsInLocalIndex)
				{
					// Delete old datasets
					std::string uuid = findOldestDataset();
					if(!uuid.empty() && !isDatasetOpen(uuid))
					{
						deleteDataset(uuid);
						freespaceMb = getFreeSpaceMB();
						g2s::G2SLogger::Main().log("Dataset " + uuid + " discarded by StorageManager auto disk cleanup task.");
					}
				}
			}
			std::this_thread::sleep_for(std::chrono::seconds(HOUSEKEEPING_INTERVAL_SEC));
		}
		catch(std::exception& e) 
		{
			g2s::G2SLogger::Main().error("Error caught during storage manager housekeeping. " + std::string(e.what()), "StorageManager::housekeepingWorker", "", "Storage");
			g2s::G2SLogger::Main().error("Storage Manager automatic disk cleanup task suspended.");
		}
	}
}

/**
 * Find oldest dataset in the local index
 * If local index is empty this method will return an empty string
 * @return Dataset UUID
 */
std::string g2s::device::StorageManager::findOldestDataset() noexcept
{
	std::lock_guard<std::mutex> lock(indexLock);
	if(storageIndex[deviceName].empty())
		return "";
	
	std::string curruuid = "";
	std::string currdate = "";
	auto it = storageIndex[deviceName].begin();
	while(it != storageIndex[deviceName].end())
	{
		if(!it->second.hasValue("Time"))
		{
			it++;
			continue;
		}
		try
		{
			std::string ldate = it->second.getString("Time");
			if(currdate.empty() || compareDates(currdate, ldate) > 0)
			{
				currdate = ldate;
				curruuid = it->first;
			}
		}
		catch(g2s::G2SException& e)
		{
			e.addCall("StorageManager::findOldestDataset");
			g2s::G2SLogger::Main().error(e, "Storage");
		}
		it++;
	}
	return curruuid;
}

/**
 * Compare date/times (timestamps)
 * this method assumes YYYY.MM.dd.HH.mm.ss date form
 * @param dt1 Date/Time A
 * @param dt2 Date/Time B
 * @return 0 if date/times are equal, +1 if [Date/Time A] > [Date/Time B], -1 if [Date/Time A] < [Date/Time B]
 * @throws g2s::G2SException
 */
int g2s::device::StorageManager::compareDates(const std::string& dt1, const std::string& dt2)
{
	if(dt1.empty() && dt2.empty())
		return 0;
	else if(dt1.empty())
		return -1;
	else if(dt2.empty())
		return 1;
	if(dt1 == dt2)
		return 0;

	std::vector<std::string> dtokens1 = g2s::SplitText(dt1, '.');
	std::vector<std::string> dtokens2 = g2s::SplitText(dt2, '.');
	if(dtokens1.size() != 6 || dtokens2.size() != 6)
		throw g2s::G2SException("Unable to compare dates. Invalid date format(s)", "StorageManager::compareDates", "Date A: " + dt1 + SLEND + "Date B: " + dt2);
	std::vector<int> dnums1(6);
	std::vector<int> dnums2(6);
	try
	{
		for(std::size_t i = 0; i < dnums1.size(); i++)
		{
			dnums1[i] = std::stoi(dtokens1[i]);
			dnums2[i] = std::stoi(dtokens2[i]);
		}
	}
	catch(std::exception& e)
	{
		throw g2s::G2SException("Unable to compare dates. " + std::string(e.what()), "StorageManager::compareDates", "Date A: " + dt1 + SLEND + "Date B: " + dt2);
	}
	if(dnums1[0] < 1970 || dnums2[0] < 1970)
		throw g2s::G2SException("Unable to compare dates. Invalid date year" + dt1, "StorageManager::compareDates", "Date A: " + dt1 + SLEND + "Date B: " + dt2);
	if(dnums1[1] < 1 || dnums2[1] < 1 || dnums1[1] > 12 || dnums2[1] > 12)
		throw g2s::G2SException("Unable to compare dates. Invalid date month" + dt1, "StorageManager::compareDates", "Date A: " + dt1 + SLEND + "Date B: " + dt2);
	if(dnums1[2] < 1 || dnums2[2] < 1 || dnums1[2] > 31 || dnums2[2] > 31)
		throw g2s::G2SException("Unable to compare dates. Invalid day of the month" + dt1, "StorageManager::compareDates", "Date A: " + dt1 + SLEND + "Date B: " + dt2);
	if(dnums1[3] < 0 || dnums2[3] < 0 || dnums1[3] > 23 || dnums2[3] > 23)
		throw g2s::G2SException("Unable to compare dates. Invalid time (hours)" + dt1, "StorageManager::compareDates", "Date A: " + dt1 + SLEND + "Date B: " + dt2);
	if(dnums1[4] < 0 || dnums2[4] < 0 || dnums1[4] > 59 || dnums2[4] > 59)
		throw g2s::G2SException("Unable to compare dates. Invalid time (minutes)" + dt1, "StorageManager::compareDates", "Date A: " + dt1 + SLEND + "Date B: " + dt2);
	if(dnums1[5] < 0 || dnums2[5] < 0 || dnums1[5] > 59 || dnums2[5] > 59)
		throw g2s::G2SException("Unable to compare dates. Invalid time (seconds)" + dt1, "StorageManager::compareDates", "Date A: " + dt1 + SLEND + "Date B: " + dt2);

	for(std::size_t i = 0; i < dnums1.size(); i++)
	{
		if(dnums1[i] == dnums2[i])
			continue;
		return dnums1[i] > dnums2[i] ? 1 : -1;
	}
	return 0;
}

/**
 * Check if the selected dataset is already open
 * @param uuid Dataset UUID
 * @return Is dataset open
 */
bool g2s::device::StorageManager::isDatasetOpen(const std::string& uuid) const noexcept 
{ 
	std::lock_guard<std::mutex> lock(storageLock);
	auto it = datasets.cbegin();
	while(it != datasets.cend())
	{
		if(g2s::CompareText(it->second->getUUID(), uuid))
			return true;
		it++;
	}
	//return datasets.find(uuid) != datasets.cend(); 
	return false;
}

/**
 * \brief Blocks until the storage buffer is emptied (subject to timeout). This can be used to wait until all files are commited to disk.
 * \return true if queue is empty, false if it timed out.
 */
bool g2s::device::StorageManager::waitForPending()
{
	static const int timeoutSec(2.0);
	boost::timer timer;
	while (saveBuffer.is_not_empty())
	{
		boost::this_thread::sleep(boost::posix_time::milliseconds(threadSleepMs));
		if (timer.elapsed() > 2.0) // wait for up to two seconds
			return false; // timed out
	}
	return true;
}


g2s::G2SConfig g2s::device::StorageManager::getIndexItemFromSummaryMeta(const G2SConfig& summaryMeta, const std::string& localPath, const std::string& pathKey)
{
	G2SConfig datasetItem;
	datasetItem.set("UUID", summaryMeta.getString("UUID"));
	datasetItem.set("Width", summaryMeta.getInt("Width"));
	datasetItem.set("Name", summaryMeta.getString("Prefix")); // dataset name
	datasetItem.set("Height", summaryMeta.getInt("Height"));
	datasetItem.set("PixelType", summaryMeta.getString("PixelType"));
	datasetItem.set("Channels", summaryMeta.getInt("Channels"));
	datasetItem.set("Slices", summaryMeta.getInt("Slices"));
	datasetItem.set("Frames", summaryMeta.getInt("Frames"));
	datasetItem.set("Positions", summaryMeta.getInt("Positions"));
	datasetItem.set("Time", summaryMeta.getString("Time"));
	datasetItem.set("LocalPath", localPath);
	datasetItem.set("PathKey", pathKey);

	// g2s::G2SLogger::Main().log("Dataset: name=" + summaryMeta.getString("Prefix") + ", key=" + pathKey);

	return datasetItem;
}


/**
 * \brief Saves text file in local storage
 * \param key - directory path (absolute or relative to storage root depending on the "external" flag)
 * \param name - file name
 * \param data - file contents
 * \param external - if true the path will be treated as absolute, otherwise as relative
 */
void g2s::device::StorageManager::writeText(const std::string& key, const std::string& name, const std::string& data, bool external)
{
	boost::filesystem::path p = boost::filesystem::path(constructDirname(key));
	if (external)
		p = boost::filesystem::path(key); // external directory
	
	if (!boost::filesystem::exists(p))
	{
		if (!boost::filesystem::create_directories(p))
			throw G2SException("Failed creating directory: " + p.string());
	}

	std::string fname(p.string() + "/" + name);
	std::ofstream of;
	of.open(fname);
	if (of.good())
	{
		of << data;
		of.close();
	}
	else
	{
		throw G2SException("Failed writeText(): " + fname);
	}
}


/**
 * \brief Reads text file from the local storage
 * \param key - directory path (absolute or relative to storage root depending on the "external" flag)
 * \param name - file name
 * \param external - if true the path will be treated as absolute, otherwise as relative
 * \return - file contents
 */
std::string g2s::device::StorageManager::readText(const std::string& key, const std::string& name, bool external)
{
	std::string fname(constructFilename(key, name));
	if (external)
		fname = key + "/" + name; // external path

	std::ifstream ifs(fname);
	std::string data;
	if (ifs.good())
	{
		return std::string((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
	}
	else
	{
		throw G2SException("Failed readText(): " + fname);
	}
}


/**
 * \brief Write binary file to the local storage
 * \param key - directory path (absolute or relative to storage root depending on the "external" flag)
 * \param name - file name
 * \param data - binary data array
 */
void g2s::device::StorageManager::writeBinary(const std::string& key, const std::string& name, const std::vector<uint8_t>& data)
{
	boost::filesystem::path p = boost::filesystem::path(constructDirname(key));
	if (!boost::filesystem::exists(p))
	{
		if (!boost::filesystem::create_directories(p))
			throw G2SException("Failed creating directory: " + p.string());
	}

	std::string fname(constructFilename(key, name));
	std::ofstream of;
	of.open(fname, std::ios::out | std::ios::binary);
	if (of.good())
	{
		of.write((char*)&data[0], data.size());
		of.close();
	}
	else
	{
		throw G2SException("Failed writeBinary(): " + fname);
	}
}

/**
 * \brief Reads image pixels from the local storage. Only micro-manager TIFs are supported.
 * \param key - directory path relative to the local root
 * \param name - file name
 * \param external - if true the path will be treated as absolute, otherwise as relative
 */
std::vector<uint8_t> g2s::device::StorageManager::readImage(const std::string& key, const std::string& name)
{
	std::string fname(constructFilename(key, name));

	g2s::data::G2SImage img;
	if (!G2SReferenceImageDB::OpenImage(fname, img))
		throw G2SException("Failed reading image file: " + fname);
	return img.getPixels();
}


/**
 * \brief Delete files from the local storage or external file system
 * \param key - directory path (absolute or relative to storage root depending on the "external" flag)
 * \param name - file name
 * \param external - if true the path will be treated as absolute, otherwise as relative
 */
void g2s::device::StorageManager::purge(const std::string& key, const std::string& name, bool external)
{
	std::string fname(constructFilename(key, name));
	if (external)
		fname = key + "/" + name; // external file path

	if (std::remove(fname.c_str()) != 0)
		throw G2SException("Failed deleting file: " + fname);
}


/**
 * \brief Lists files in the local storage. Each file item is a JSON string with some basic information:
 *	"name" - file name
 *	"dir" - true if the file is directory
 *	"mmds" - true if the file is a micro-manager dataset
 * \param key - directory path (absolute or relative to storage root depending on the "external" flag)
 * \param external - if true the path will be treated as absolute, otherwise as relative
 * \return 
 */
std::vector<std::string> g2s::device::StorageManager::list(const std::string& key, bool external)
{
	boost::filesystem::path p = boost::filesystem::path(external ? key : constructDirname(key));
	std::vector<std::string> ret;
	for (auto& entry : boost::make_iterator_range(boost::filesystem::directory_iterator(p), {}))
	{
		G2SConfig fileInfo;
		bool isDir = boost::filesystem::is_directory(entry.path());
		fileInfo.set("name", entry.path().filename().string());
		fileInfo.set("dir", isDir);

		// look for MM dataset signature if it is not an external path
		bool mmDataset = false;
		if (!external) {
			if (isDir) {
				for (auto& subEntry : boost::make_iterator_range(boost::filesystem::directory_iterator(entry.path()), {}))
				{
					if (subEntry.path().filename().string().compare("display_and_comments.txt") == 0)
					{
						mmDataset = true;
						break;
					}
				}
			}
		}
		fileInfo.set("mmds", mmDataset);
		ret.push_back(fileInfo.serialize());
	}
				
	return ret;
}

/**
 * \brief Traverse local storage and build an index of all available datasets
 */
void g2s::device::StorageManager::indexDatasets()
{
	std::lock_guard<std::mutex> lock(indexLock);
	
	// list all datasets in the storage root
	boost::filesystem::path rootPath = boost::filesystem::path(constructDirname(""));
	g2s::G2SLogger::Main().sysmsg("Indexing datasets in " + rootPath.string() + "...");

	// reset index
	storageIndex.clear();

	// build up local index
	std::map<std::string, G2SConfig> localIndex;
	boost::filesystem::recursive_directory_iterator it(rootPath);
	boost::filesystem::recursive_directory_iterator end;

	while (it != end)
	{
		std::string fileItem = it->path().string();
	   if(boost::filesystem::is_directory(it->path())) 
		{
			// test if there is "display and comments" file
			// the presence of this file indicates that we entered a dataset directory
			bool datasetSignature = boost::filesystem::exists(it->path().string() + "/display_and_comments.txt"); 
			if(datasetSignature)
			{
				// we assume we hit the dataset directory
				std::string pathKey = boost::filesystem::relative(it->path().parent_path(), rootPath).string();
				if(pathKey == ".")
					pathKey = ""; // make current dir deisgnator empty string

				// iterate on position dirs within the dataset
				for(auto& dsItem : boost::make_iterator_range(boost::filesystem::directory_iterator(it->path()), {}))
				{
					if(boost::filesystem::is_directory(dsItem.path()))
					{
						std::string mdPath = dsItem.path().string() + "/metadata.txt";
						try 
						{
							// we are assuming this is a position directory
							// we are also assuming that ALL subdirs in the dataset must be position directories
							G2SConfig md;
							md.load(mdPath);
							G2SConfig summaryMeta = md.getElement("Summary");
							G2SConfig datasetItem = getIndexItemFromSummaryMeta(summaryMeta, it->path().string(), pathKey);
							localIndex.insert(std::pair<std::string, G2SConfig>(summaryMeta.getString("UUID"), datasetItem));
							break;
						}
						catch (G2SException& e)
						{
							g2s::G2SLogger::Main().log("Failed parsing metadata on " + mdPath + ", " + e.msg());
							break;
						}
					}
					else if(g2s::CompareText(dsItem.path().filename().string(), "metadata.txt"))
					{
						try 
						{
							// we are assuming this is a position directory
							// we are also assuming that ALL subdirs in the dataset must be position directories
							G2SConfig md;
							md.load(dsItem.path().string());
							G2SConfig summaryMeta = md.getElement("Summary");
							G2SConfig datasetItem = getIndexItemFromSummaryMeta(summaryMeta, it->path().string(), pathKey);
							localIndex.insert(std::pair<std::string, G2SConfig>(summaryMeta.getString("UUID"), datasetItem));
							break;
						}
						catch (G2SException& e)
						{
							g2s::G2SLogger::Main().log("Failed parsing metadata on " + dsItem.path().string() + ", " + e.msg());
							break;
						}
					}
				}
				it.no_push();
				// do not recurse if we are in the dataset dir
				// or if we are in the dir that looks like dataset but in which metadata parsing failed
			}
		}
		++it;
	}

	storageIndex[deviceName] = localIndex;
	g2s::G2SLogger::Main().sysmsg("Indexed datasets: " + std::to_string(localIndex.size()));
}

/**
 * \brief - adds an external device (another microscope in the domain) index to the local index, thus building
 *	up a global index of all datasets stored on all micrscopes
 *
 * \param otherDevice - other device name
 * \param serializedIndex - Index entry serialized to JSON string
 */
void g2s::device::StorageManager::addDeviceIndex(const std::string& otherDevice, const std::string& serializedIndex)
{
	G2SConfig indexRep;
	indexRep.parse(serializedIndex);
	auto itemArray = indexRep.getElementArray("data");

	std::map<std::string, G2SConfig> index;
	for (auto& item : itemArray)
	{
		G2SConfig jsonItem(item);
		index[jsonItem.getString("UUID")] = jsonItem;
	}
	{
		std::lock_guard<std::mutex> lock(indexLock);
		storageIndex[otherDevice] = index;
	}
}

void g2s::device::StorageManager::addDeviceItem(const std::string& device, const std::string& serializedItem)
{
	G2SConfig item;
	item.parse(serializedItem);
	std::string uuid = item.getString("UUID");

	{
		std::lock_guard<std::mutex> lock(indexLock);
		storageIndex[device][uuid] = item;
	}
}

void g2s::device::StorageManager::deleteDeviceItem(const std::string& device, const std::string& serializedItem)
{
	std::lock_guard<std::mutex> lock(indexLock);
	auto it = storageIndex.find(device);
	if (it != storageIndex.end())
	{
		G2SConfig item;
		item.parse(serializedItem);
		std::string uuid = item.getString("UUID");
		auto itItem = it->second.find(uuid);
		if (itItem != it->second.end())
			it->second.erase(uuid);
	}
}

/**
 * \brief - retrieves the index of a given device (local or remote)
 * \param device - device name
 * \return - serialized index
 */
std::vector<g2s::G2SConfig> g2s::device::StorageManager::getDeviceIndex(const std::string& device)
{
	std::lock_guard<std::mutex> lock(indexLock);
	auto storageIndexIt = storageIndex.find(device);
	std::vector<g2s::G2SConfig> itemArray;
	if (storageIndexIt != storageIndex.end())
		// convert map of jsons to array of strings
		std::for_each(storageIndexIt->second.begin(), storageIndexIt->second.end(),
			[&itemArray](std::pair<const std::string, G2SConfig>& entry) {itemArray.push_back(entry.second);});

	return itemArray;
}

std::vector<std::string> g2s::device::StorageManager::getDeviceIndexSerialized(const std::string& device)
{
	std::lock_guard<std::mutex> lock(indexLock);
	auto storageIndexIt = storageIndex.find(device);
	std::vector<std::string> itemArray;
	if (storageIndexIt != storageIndex.end())
		// convert map of jsons to array of strings
		std::for_each(storageIndexIt->second.begin(), storageIndexIt->second.end(),
			[&itemArray](std::pair<const std::string, G2SConfig>& entry) {itemArray.push_back(entry.second.serialize()); });

	return itemArray;
}

g2s::G2SConfig g2s::device::StorageManager::getItem(const std::string& devName, const std::string& uuid)
{
	std::lock_guard<std::mutex> lock(indexLock);
	auto storageIndexIt = storageIndex.find(devName);
	if (storageIndexIt != storageIndex.end())
	{
		auto itemIt = storageIndexIt->second.find(uuid);
		if (itemIt != storageIndexIt->second.end())
			return itemIt->second;
	}
	throw G2SException("Item " + uuid + " is not available in device index " + devName);
}

/**
 * \brief - retrieves the aggregate index of all datasets on all microscopes in the domain
 * \return - serialized aggregate index
 */
std::vector<std::string> g2s::device::StorageManager::getStorageIndexSerialized()
{
	std::vector<std::string> serializedGlobalIndex;
	for (auto& it = storageIndex.begin(); it != storageIndex.end(); it++)
	{
		std::vector<std::string> idx = getDeviceIndexSerialized(it->first);
		serializedGlobalIndex.insert(end(serializedGlobalIndex), begin(idx), end(idx));
	}
	return serializedGlobalIndex;
}

/**
 * \brief - checks the free disk space on the drive used by Storage Manager
 * \return - disk space in MB
 */
int g2s::device::StorageManager::getFreeSpaceMB()
{
		DWORD sectorsPerCluster;
		DWORD bytesPerSector;
		DWORD numberOfFreeClusters;
		DWORD totalNumberOfClusters;
		if (GetDiskFreeSpace((LPCSTR)dirPath.c_str(), &sectorsPerCluster, &bytesPerSector, &numberOfFreeClusters, &totalNumberOfClusters) == 0)
		{
			DWORD errcode = GetLastError();
			LPSTR messageBuffer = nullptr;
			std::size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
				NULL, errcode, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&messageBuffer, 0, NULL);
			std::stringstream ss;
			ss << "Source path: " << dirPath << std::endl;
			ss << "Error code: " << errcode << std::endl;
			ss << "Error description: " << std::string(messageBuffer, size);
			throw G2SException("Failed to obtain disk free space: " + ss.str());
		}

		return (__int64)numberOfFreeClusters * sectorsPerCluster * bytesPerSector / 1048576;
}

/**
 * \brief Copies a file in server storage to an external location, usually the network drive
 * \param keySource - storage key (path)
 * \param nameSource - storage file name
 * \param externalPath - full external path to the destination
 */
void g2s::device::StorageManager::copyToExtern(const std::string& keySource, const std::string& nameSource, const std::string& externalPath, const std::string& externalName)
{
	boost::filesystem::path sourcePath = boost::filesystem::path(constructFilename(keySource, nameSource));
	boost::filesystem::path targetPath = boost::filesystem::path(externalPath);
	boost::filesystem::path targetFullPath = boost::filesystem::path(externalPath + "/" + externalName);
	if (!boost::filesystem::exists(sourcePath))
	{
		throw G2SException("Source file not present in server storage: " + sourcePath.string());
	}

	// create target directory if needed
	if (!boost::filesystem::exists(targetPath))
		if (!boost::filesystem::create_directories(targetPath))
			throw G2SException("Failed creating external target directory: " + targetPath.string());
	
	boost::system::error_code ec;
	boost::filesystem::copy_file(sourcePath, targetFullPath, boost::filesystem::copy_option::overwrite_if_exists, ec);
	if (ec.failed())
		throw G2SException("External file copy failed: " + targetPath.string() + " with error: " + ec.message());
}

/**
 * \brief Create new dataset on the server
 * \param path - storage path
 * \param name - dataset name
 * \param summaryMetadata - summary metadata, must contain critical info for initializing the dataset
 * \return - dataset handle
 */
std::string g2s::device::StorageManager::createDataset(const std::string& path, const std::string& name, const std::string& summaryMetadata)
{
	std::lock_guard<std::mutex> lock(indexLock);
	std::string handle = name;
	g2s::acq::Dataset* ds = g2s::acq::Dataset::createAndInitializeFromMetadata(constructDirname(path), name, summaryMetadata);
	auto i = datasets.find(handle);
	if (i != datasets.end())
		throw G2SException("Dataset already open: " + handle);

	datasets[handle] = ds;
	storageIndex[deviceName][ds->getUUID()] = getIndexItemFromSummaryMeta(ds->getSummaryMetadata(), ds->getPath() + "/" + ds->getName(), path);
	return handle;
}

/**
 *	Returns current status of the acquisition in progress
 *	
 *	\param handle - handle to the open dataset, exception if not open or does not exist
 * \return - Serialized JSON representing acquired ("acq") and saved ("sav") flags for each FrameKey
 */
std::string g2s::device::StorageManager::getDatasetAcqStatus(const std::string handle)
{
	auto i = datasets.find(handle);
	if (i == datasets.end())
		throw G2SException("Dataset is not open: " + handle);
   acq::Dataset* ds = datasets[handle];
	G2SConfig acqStatMd;
	for (int p=0; p<ds->getNumPositions(); p++)
	{
	   for (int f=0; f<ds->getNumFrames(); f++)
	   {
	      for (int c=0; c<ds->getNumChannels(); c++)
	      {
				for (int s=0; s<ds->getNumSlices(); s++)
            {
					G2SConfig asc;
					acq::ImageAcqStatus as = ds->getImageAcqStatus(p, c, s, f);
					asc.set("acq", as.acquired);
					asc.set("sav", as.saved);
					acqStatMd.set(acq::Dataset::generateKey(p, c, s, f), asc);
				}
	      }
	   }
	}

	return acqStatMd.serialize();
}

/**
 * \brief Closes open dataset and saves all metadata, if the dataset is not open
 * it will do nothing
 * 
 * \param handle - handle to the open file
 */
bool g2s::device::StorageManager::closeDataset(const std::string& handle)
{
	auto i = datasets.find(handle);
	if (i == datasets.end())
		return false;

	waitForPending(); // wait for the worker queue

	i->second->close();
	delete i->second;
	datasets.erase(i->first);
	return true;
}


/**
 * \brief Delete dataset from the local storage
 * \param uuid - unique identifier
 */
void g2s::device::StorageManager::deleteDataset(const std::string& uuid)
{
	std::lock_guard<std::mutex> lock(indexLock);
	auto& localIndex = storageIndex[deviceName];
	auto itemIt = localIndex.find(uuid);
	if (itemIt != localIndex.end())
	{
		boost::filesystem::path path(itemIt->second.getString("LocalPath"));
		boost::filesystem::remove_all(path);
		localIndex.erase(uuid);
		g2s::G2SLogger::Main().log("Dataset deleted: " + uuid + ", path: " + path.string());
	}
}

/**
 * \brief - add metadata to Summary metadata of the open dataset
 * \param handle - dataset name		
 * \param meta - metadata to add
 */
void g2s::device::StorageManager::addDatasetProperties(const std::string& handle, const std::string& meta)
{
	auto i = datasets.find(handle);
	if (i == datasets.end())
		throw G2SException("Dataset is not open: " + handle);

	i->second->addSummaryMetadata(G2SConfig(meta));
}

/**
 * \brief Add metadata to specific image coordinates
 * \param handle - open dataset name
 * \param frame 
 * \param channel 
 * \param slice 
 * \param position 
 * \param imageMeta - image metadata 
 */
void g2s::device::StorageManager::addImageProperties(const std::string& handle, int frame, int channel, int slice, int position, const std::string& imageMeta)
{
	auto i = datasets.find(handle);
	if (i == datasets.end())
		throw G2SException("Dataset is not open: " + handle);

	i->second->addImageMetadata(position, channel, slice, frame, G2SConfig(imageMeta));
}

/**
 * \brief Retrieve summary metadata of the currently open dataset
 * \param handle - dataset name
 * \return - summary metadata
 */
std::string g2s::device::StorageManager::getSummaryMeta(const std::string& handle)
{
	auto i = datasets.find(handle);
	if (i == datasets.end())
		throw G2SException("Dataset is not open: " + handle);

	acq::Dataset* ds = i->second;
	std::string serMeta = i->second->getSummaryMetadata().serialize();

	return serMeta;
}

/**
 * \brief Retrieve image metadata from specific image coordinates
 * \param handle - dataset name
 * \param frame 
 * \param channel 
 * \param slice 
 * \param position 
 * \return - image metadata
 */
std::string g2s::device::StorageManager::getImageMeta(const std::string& handle, int frame, int channel, int slice, int position)
{
	auto i = datasets.find(handle);
	if (i == datasets.end())
		throw G2SException("Dataset is not open: " + handle);

	// wait until all files are comited
	if (!waitForPending())
		throw G2SException("Timed out waiting for async save to comit files to disk");

	return i->second->getImageMetadata(position, channel, slice, frame).serialize();
}


/**
 * \brief Get metadata frome the dataset stored in local storage (it does not have to be open)
 * \param path - dataset path relative to storage root
 * \param name - dataset name
 * \return - summary metadata
 */
std::string g2s::device::StorageManager::getDatasetMetadata(const std::string& path, const std::string& name)
{
	G2SConfig md = g2s::acq::Dataset::getMetadata(constructDirname(path), name);
	std::string mdSerialized = md.serialize();
	return mdSerialized;
}

/**
 * \brief Returns an array of image pixels (as byte array) from a specific coordinate
 * \param handle - dataset name
 * \param frame 
 * \param channel 
 * \param slice 
 * \param position 
 * \return - array of bytes
 */
std::vector<uint8_t> g2s::device::StorageManager::getImagePixels(const std::string& handle, int frame, int channel, int slice, int position)
{
	auto i = datasets.find(handle);
	if (i == datasets.end())
		throw G2SException("Dataset is not open: " + handle);

	return i->second->getImagePixels(position, channel, slice, frame);
}


/**
 * \brief Opens a dataset. The dataset is then available for fast access from the server memory
 * \param path - dataset path
 * \param name - dataset name
 * \return - dataset name
 */
std::string g2s::device::StorageManager::loadDataset(const std::string& path, const std::string& name)
{
	std::string handle = path + "/" + name;
	auto i = datasets.find(handle);
	if (i != datasets.end())
		throw G2SException("Dataset already open: " + handle);

	// do not load pixels
	// TODO: consider loading pixels (optionally)
	g2s::acq::Dataset* ds = g2s::acq::Dataset::loadFromPath(constructDirname(path), name, false);

	datasets[handle] = ds;
	return handle;
}

