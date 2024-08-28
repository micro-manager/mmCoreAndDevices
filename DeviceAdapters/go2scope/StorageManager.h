// ==============================================================================================================================
// 
// Storage Manager: provides local storage functions for datasets and generic files
// 
// Copyright (C) 2012 by Luminous Point LLC
// http://www.luminous-point.com
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
// NOTES:         Storage Device development is supported in part by
//                Chan Zuckerberg Initiative (CZI)
//
// 
//						This component is modified to work for the go2scope storage device
//						for micromanager. Supports Micromanager v1 dataset structure.
//
// ==============================================================================================================================
#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include "MMDevice.h"
#include "G2SBoundedBuffer.h"


/**
 * \brief Image with metadata container used for asynchronous saving to disk.
 *	StorageItems are placed in the queue patrolled by the Worker thread saving items to disk.		
 */
class StorageItem
{
	public:
		StorageItem(const std::string& h, std::vector<uint8_t>&& pix, int f, int c, int s, int p, const std::string& im)
			: handle(h), frame(f), channel(c), slice(s), position(p), imageMeta(im)
		{
			pixels.swap(pix);
		}
		StorageItem(const std::string& h, const std::vector<uint8_t>& pix, int f, int c, int s, int p, const std::string& im)
			: handle(h), pixels(pix), frame(f), channel(c), slice(s), position(p), imageMeta(im)
		{}

		virtual ~StorageItem() {}

		std::vector<uint8_t> pixels;
		std::string handle;
		int frame;
		int channel;
		int slice;
		int position;
		std::string imageMeta;
};


namespace g2s {
	namespace device {

      /**
		 * \brief Local storage functions
		 */
		class StorageManager
		{
		public:
			//============================================================================================================================
			// Constructors & destructors
			//============================================================================================================================
			/**
			 * Default class constructor
			 */
			StorageManager(MM::Device* parentDevice) noexcept;
			virtual ~StorageManager();

		public:
			//============================================================================================================================
			// Public interface
			//============================================================================================================================
			bool initialize() noexcept;
			bool initialize(const std::string& devName, const std::string& path) noexcept;

			int getFreeSpaceMB();

			// dataset operations
			std::string createDataset(const std::string& path, const std::string& name, const std::string& summaryMetadata);
			std::string getDatasetAcqStatus(const std::string handle);
			bool closeDataset(const std::string& handle);
			void deleteDataset(const std::string& uuid);
			void insertImageAsync(const std::string& handle, std::vector<uint8_t>&& pixels, int frame, int channel, int slice, int position, const std::string& imageMeta);
			void insertImage(const std::string& handle, std::vector<uint8_t>& pixels, int frame, int channel, int slice, int position, const std::string& imageMeta);
			void addDatasetProperties(const std::string& handle, const std::string&);
			void addImageProperties(const std::string& handle, int frame, int channel, int slice, int position, const std::string& imageMeta);
			std::string getSummaryMeta(const std::string& handle);
			std::string getImageMeta(const std::string& handle, int frame, int channel, int slice, int position);
			std::string getDatasetMetadata(const std::string& path, const std::string& name);
			std::vector<uint8_t> getImagePixels(const std::string& handle, int frame, int channel, int slice, int position);
			std::string loadDataset(const std::string& path, const std::string& name);
			void clearAsyncError() { asyncError = false; asyncErrorText = ""; asyncErrorDataset = ""; }

      private:
			std::string constructFilename(const std::string& key, const std::string& name) { return dirPath + "\\" + key + "\\" + name; }
			std::string constructDirname(const std::string& key) { return dirPath + "\\" + key; }
			void saveWorker();
			void housekeepingWorker() noexcept;
			int compareDates(const std::string& dt1, const std::string& dt2);
			bool waitForPending();
			bool isDatasetOpen(const std::string& uuid) const noexcept;
			
			//============================================================================================================================
			// Data members
			//============================================================================================================================
			std::string dirPath;
			std::map<std::string, g2s::acq::Dataset*> datasets;///< currently open datasets
			bounded_buffer<StorageItem*> saveBuffer;
			boost::thread* execThd;										///< Worker thread
			std::unique_ptr<std::thread> housekeepingThd;		///< Housekeeping thread
			std::atomic<bool> workerActive;
			bool asyncError;
			bool shutdownFlag;	
			std::string asyncErrorText;
			std::string asyncErrorDataset;
			mutable std::mutex storageLock;
			mutable std::mutex indexLock;

			const static int saveBufferDepth = 4;
			const static int threadSleepMs = 300;
		};
	}
}

