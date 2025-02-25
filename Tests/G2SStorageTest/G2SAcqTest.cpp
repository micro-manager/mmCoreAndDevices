///////////////////////////////////////////////////////////////////////////////
// FILE:          G2SAcqTest.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     Device Driver Tests
//-----------------------------------------------------------------------------
// DESCRIPTION:   Go2Scope storage driver acquisition test
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
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include "MMCore.h"

extern std::string generateImageMeta(CMMCore& core, int imgind);

/**
 * Storage acquisition test
 * @param core MM Core instance
 * @param path Data folder path
 * @param name Dataset name
 * @param c Channel count
 * @param t Time points
 * @param p Positions count
 * @throws std::runtime_error
 */
void testAcquisition(CMMCore& core, const std::string& path, const std::string& name, int c, int t, int p)
{
	std::cout << std::endl << "Starting G2SStorage driver acquisition test" << std::endl;

	// Take one image to "warm up" the camera and get actual image dimensions
	core.snapImage();
	int w = (int)core.getImageWidth();
	int h = (int)core.getImageHeight();
	int imgSize = 2 * w * h;
	double imgSizeMb = (double)imgSize / (1024.0 * 1024.0);

	// Shape convention: Z, T, C, Y, X
	std::vector<long> shape = { p, t, c, h, w };
	auto handle = core.createDataset(path.c_str(), name.c_str(), shape, MM::StorageDataType_GRAY16, "");

	std::cout << "Dataset UID: " << handle << std::endl;
	std::cout << "Dataset shape (W-H-C-T-P): " << w << " x " << h << " x " << c << " x " << t << " x " << p << " x 16-bit" << std::endl << std::endl;
	std::cout << "START OF ACQUISITION" << std::endl;
	
	// Start acquisition
	auto start = std::chrono::high_resolution_clock::now();
	core.startSequenceAcquisition(c * t * p, 0.0, true);

	int imgind = 0;
	auto startAcq = start;
	for(int i = 0; i < p; i++) 
	{
		for(int j = 0; j < t; j++) 
		{
			for(int k = 0; k < c; k++)
			{
				if(core.isBufferOverflowed()) 
					throw std::runtime_error("Buffer overflow!!");

				while(core.getRemainingImageCount() == 0)
					std::this_thread::sleep_for(std::chrono::milliseconds(1)); // wait for images to become available
				
				// Reset acquisition timer when the first image becomes available)
				if(start == startAcq)
					startAcq = std::chrono::high_resolution_clock::now();
				
				// fetch the image
				unsigned char* img = reinterpret_cast<unsigned char*>(core.popNextImage());

				// Generate image metadata
				std::string meta = generateImageMeta(core, imgind);
				
				// Add image to the stream
				auto startSave = std::chrono::high_resolution_clock::now();
				core.appendImageToDataset(handle.c_str(), imgSize, img, { i, j, k }, meta.c_str());
				auto endSave = std::chrono::high_resolution_clock::now();

				// Calculate statistics
				double imgSaveTimeMs = (endSave - startSave).count() / 1000000.0;
				double bw = imgSizeMb / (imgSaveTimeMs / 1000.0);
				std::cout << "Saved image " << imgind++ << " in ";
				std::cout << std::fixed << std::setprecision(2) << imgSaveTimeMs << " ms, size ";
				std::cout << std::fixed << std::setprecision(1) << imgSizeMb << " MB, BW: " << bw << " MB/s" << std::endl;
			}
		}
	}

	// We are done so close the dataset
	core.stopSequenceAcquisition();
	core.closeDataset(handle.c_str());
	auto end = std::chrono::high_resolution_clock::now();
	std::cout << "END OF ACQUISITION" << std::endl << std::endl;

	// Calculate storage driver bandwidth
	double totalTimeS = (end - start).count() / 1000000000.0;
	double prepTimeS = (startAcq - start).count() / 1000000000.0;
	double acqTimeS = (end - startAcq).count() / 1000000000.0;
	double totalSizemb = (double)imgSize * p * t * c / (1024.0 * 1024.0);
	double totbw = totalSizemb / totalTimeS;
	double acqbw = totalSizemb / acqTimeS;
	std::cout << std::fixed << std::setprecision(1) << "Dataset size " << totalSizemb << " MB" << std::endl;
	std::cout << std::fixed << std::setprecision(3) << "Camera prep time: " << prepTimeS << " sec" << std::endl;
	std::cout << std::fixed << std::setprecision(3) << "Active acquisition time: " << acqTimeS << " sec" << std::endl;
	std::cout << std::fixed << std::setprecision(1) << "Storage driver bandwidth " << acqbw << " MB/s" << std::endl << std::endl;
	std::cout << std::fixed << std::setprecision(3) << "Acquisition completed in " << totalTimeS << " sec" << std::endl;
	std::cout << std::fixed << std::setprecision(1) << "Acquisition bandwidth " << totbw << " MB/s" << std::endl;
}