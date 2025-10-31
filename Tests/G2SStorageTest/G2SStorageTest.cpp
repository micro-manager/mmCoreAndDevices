///////////////////////////////////////////////////////////////////////////////
// FILE:          G2SStorageTest.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     Device Driver Tests
//-----------------------------------------------------------------------------
// DESCRIPTION:   Go2Scope storage driver test suite
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
#include <algorithm>
#include <filesystem>
#include "MMCore.h"

#define TEST_WRITE				1
#define TEST_READ					2
#define TEST_ACQ					3
#define TEST_VER					"1.0.0"
#define ENGINE_BIGTIFF			1
#define ENGINE_ZARR				2
#define CAMERA_DEMO				1
#define CAMERA_HAMAMATSU		2

extern void testWritter(CMMCore& core, const std::string& path, const std::string& name, int c, int t, int p);
extern void testReader(CMMCore& core, const std::string& path, const std::string& name, bool optimized, bool printmeta);
extern void testAcquisition(CMMCore& core, const std::string& path, const std::string& name, int c, int t, int p);

/**
 * Application entry point
 * @param argc Argument count
 * @param argv Argument list
 * @return Status code
 */
int main(int argc, char** argv)
{
	std::cout << "Starting G2SStorage driver test suite..." << std::endl;
	int selectedTest = TEST_ACQ;
	int storageEngine = ENGINE_BIGTIFF;
	int selectedcamera = CAMERA_DEMO;
	int flushcycle = 0;
	int channels = 4;
	int timepoints = 8;
	int positions = 5;
	int cbuffsize = 16384;
	bool directIO = false;
	bool printmeta = false;
	bool optimalaccess = false;
	std::string savelocation = ".";

	// Parse input arguments
	if(argc < 2)
	{
		std::cout << "Invalid arguments specified. To see program options type G2SStorageTest -help" << std::endl;
		return 2;
	}

	// Obtain selected test
	std::string carg(argv[1]);
	std::transform(carg.begin(), carg.end(), carg.begin(), [](char c) { return std::tolower(c); });
	if(carg == "-v")
	{
		std::cout << "G2SStorageTest " << TEST_VER << std::endl;
		return 0;
	}
	else if(carg == "-help")
	{
		std::cout << "Available test suites: write, read, acq" << std::endl << std::endl;
		std::cout << "For write test type:" << std::endl;
		std::cout << "G2SStorageTest write [storage_engine] [save_location] [camera] [channel_count] [time_points_count] [positions_count] [direct_io] [flush_cycle]" << std::endl << std::endl;
		
		std::cout << "For read test type:" << std::endl;
		std::cout << "G2SStorageTest read [storage_engine] [save_location] [dataset_name] [direct_io] [optimal_access] [print_meta]" << std::endl << std::endl;
		
		std::cout << "For acquisition test type:" << std::endl;
		std::cout << "G2SStorageTest acq [storage_engine] [save_location] [camera] [channel_count] [time_points_count] [positions_count] [direct_io] [flush_cycle]" << std::endl << std::endl;
		
		std::cout << "Available storage engines: zarr, bigtiff (default)" << std::endl;
		std::cout << "Available cameras: demo (default), hamamatsu" << std::endl;
		std::cout << "The following options are basic ON/OFF (1/0) flags: [direct_io] [optimal_access] [print_meta]" << std::endl;
		std::cout << "Default save location is the current directory" << std::endl;
		std::cout << "Default channel count is " << channels << std::endl;
		std::cout << "Default time points count is " << timepoints << std::endl;
		std::cout << "Default positions count is " << positions << std::endl;
		std::cout << "Default camera is " << selectedcamera << std::endl;
		std::cout << "Default DirectIO status is " << directIO << std::endl;
		std::cout << "Default access type is " << optimalaccess << std::endl;
		std::cout << "By default metadata is not printed on the command line" << std::endl;
		std::cout << "Default flush cycle is 0 (file stream is flushed during closing)" << std::endl;
		std::cout << "Default dataset name is test-[storage_engine]" << std::endl;
		return 0;
	}
	else if(carg == "read" || carg == "write" || carg == "acq")
		selectedTest = carg == "write" ? TEST_WRITE : (carg == "read" ? TEST_READ : TEST_ACQ);
	else
	{
		std::cout << "Invalid test suite selected. To see program options type G2SStorageTest -help" << std::endl;
		return 2;
	}

	// Obtain storage engine
	if(argc > 2)
	{
		carg = std::string(argv[2]);
		std::transform(carg.begin(), carg.end(), carg.begin(), [](char c) { return std::tolower(c); });
		if(carg == "bigtiff" || carg == "zarr")
			storageEngine = carg == "bigtiff" ? ENGINE_BIGTIFF : ENGINE_ZARR;
		else
		{
			std::cout << "Invalid storage engine selected. To see program options type G2SStorageTest -help" << std::endl;
			return 2;
		}
	}
	std::string datasetname = "test-" + std::string(storageEngine == ENGINE_BIGTIFF ? "bigtiff" : "zarr");

	// Obtain save location
	if(argc > 3)
	{
		carg = std::string(argv[3]);
		std::filesystem::path spath = std::filesystem::u8path(carg);
		if(std::filesystem::exists(spath) && !std::filesystem::is_directory(spath))
		{
			std::cout << "Invalid save location. To see program options type G2SStorageTest -help" << std::endl;
			return 2;
		}
		savelocation = std::filesystem::absolute(spath).u8string();
	}

	// Obtain dataset name (READ test)
	if(argc > 4 && selectedTest == TEST_READ)
		datasetname = std::string(argv[4]);	
	// Obtain camera type (WRITE, ACQ tests)
	else if(argc > 4)
	{
		carg = std::string(argv[4]);
		std::transform(carg.begin(), carg.end(), carg.begin(), [](char c) { return std::tolower(c); });
		if(carg == "demo" || carg == "hamamatsu")
			selectedcamera = carg == "demo" ? CAMERA_DEMO : CAMERA_HAMAMATSU;
		else
		{
			std::cout << "Invalid camera selected. To see program options type G2SStorageTest -help" << std::endl;
			return 2;
		}
	}

	// Obtain I/O type (READ test)
	if(argc > 5 && selectedTest == TEST_READ)
		try { directIO = std::stoi(argv[5]) != 0; } catch(std::exception& e) { std::cout << "Invalid argument value. " << e.what() << std::endl; return 1; }
	// Obtain channel count (WRITE, ACQ tests)
	else if(argc > 5)
		try { channels = (int)std::stoul(argv[5]); } catch(std::exception& e) { std::cout << "Invalid argument value. " << e.what() << std::endl; return 1; }

	// Obtain access type (READ test)
	if(argc > 6 && selectedTest == TEST_READ)
		try { optimalaccess = std::stoi(argv[6]) != 0; } catch(std::exception& e) { std::cout << "Invalid argument value. " << e.what() << std::endl; return 1; }
	// Obtain time points (WRITE, ACQ tests)
	else if(argc > 6)
		try { timepoints = (int)std::stoul(argv[6]); } catch(std::exception& e) { std::cout << "Invalid argument value. " << e.what() << std::endl; return 1; }
	
	// Obtain print metadata flag (READ test)
	if(argc > 7 && selectedTest == TEST_READ)
		try { printmeta = std::stoi(argv[7]) != 0; } catch(std::exception& e) { std::cout << "Invalid argument value. " << e.what() << std::endl; return 1; }
	// Obtain positions (WRITE, ACQ tests)
	else if(argc > 7)
		try { positions = (int)std::stoul(argv[7]); } catch(std::exception& e) { std::cout << "Invalid argument value. " << e.what() << std::endl; return 1; }

	// Obtain I/O type (WRITE, ACQ tests)
	if(argc > 8 && selectedTest != TEST_READ)
		try { directIO = std::stoi(argv[8]) != 0; } catch(std::exception& e) { std::cout << "Invalid argument value. " << e.what() << std::endl; return 1; }

	// Obtain flush cycle (WRITE, ACQ tests)
	if(argc > 9 && selectedTest != TEST_READ)
		try { flushcycle = (int)std::stoul(argv[9]) != 0; } catch(std::exception& e) { std::cout << "Invalid argument value. " << e.what() << std::endl; return 1; }

	// Print configuration
	std::cout << "Data location " << savelocation << std::endl;
	std::cout << "Dataset name " << datasetname << std::endl << std::endl;
	if(selectedTest != TEST_READ)
	{
		std::cout << "Circular buffer size: " << cbuffsize << " MB" << std::endl;
		std::cout << "Camera " << (selectedcamera == CAMERA_DEMO ? "DEMO" : "HAMAMATSU") << std::endl;
	}
	std::cout << "Direct I/O " << (directIO ? "YES" : "NO") << std::endl;
	std::cout << "Flush Cycle " << flushcycle << std::endl;
	std::cout << "Optimized access " << (optimalaccess ? "YES" : "NO") << std::endl << std::endl;
	try
	{
		// Initialize core
		CMMCore core;
#ifdef NDEBUG
		core.enableDebugLog(false);
		core.enableStderrLog(false);
#else
		core.enableDebugLog(true);
		core.enableStderrLog(true);
#endif


		// Load storage driver
		std::cout << "Loading storage driver..." << std::endl;
		if(storageEngine == ENGINE_ZARR)
			core.loadDevice("Store", "AcquireZarr", "AcquireZarrStorage");
		else
			core.loadDevice("Store", "Go2ScopeTmp", "G2SBigTiffStorage");

		// Load camera driver
		if(selectedTest != TEST_READ)
		{
			// Set circular buffer size
			std::cout << "Setting buffer size..." << std::endl;
			core.setCircularBufferMemoryFootprint(cbuffsize);
			core.clearCircularBuffer();

			std::cout << "Loading camera driver..." << std::endl;
			if(selectedcamera == CAMERA_DEMO)
				core.loadDevice("Camera", "DemoCamera", "DCam");
			else
				core.loadDevice("Camera", "HamamatsuHam", "HamamatsuHam_DCAM");
		}

		// initialize the system, this will in turn initialize each device
		std::cout << "Initializing device drivers..." << std::endl;
		core.initializeAllDevices();

		// Configure camera
		if(selectedTest != TEST_READ)
		{
			std::cout << "Setting camera configuration..." << std::endl;
			if(selectedcamera == CAMERA_DEMO)
			{
				// Configure demo camera
				core.setProperty("Camera", "PixelType", "16bit");
				core.setProperty("Camera", "OnCameraCCDXSize", "4432");
				core.setProperty("Camera", "OnCameraCCDYSize", "2368");
				core.setExposure(10.0);
			}
			else
			{
				// Configure HamamatsuHam
				core.setProperty("Camera", "PixelType", "16bit");
				core.setROI(1032, 0, 2368, 2368);
				core.setExposure(5.0);
			}
		}

		// Set storage engine properties
		if(storageEngine == ENGINE_BIGTIFF) {
			std::cout << "Setting BigTIFF storage configuration..." << std::endl;
			core.setProperty("Store", "DirectIO", directIO ? 1L : 0L);
			core.setProperty("Store", "FlushCycle", (long)flushcycle);
		}
				
		// Run test
		if(selectedTest == TEST_WRITE)
			testWritter(core, savelocation, datasetname, channels, timepoints, positions);
		else if(selectedTest == TEST_READ)
			testReader(core, savelocation, datasetname, optimalaccess, printmeta);
		else if(selectedTest == TEST_ACQ)
			testAcquisition(core, savelocation, datasetname, channels, timepoints, positions);
		else
			std::cout << "Invalid test suite selected. Exiting..." << std::endl;

		core.unloadAllDevices();
	}
	catch(std::exception& e)
	{
		std::cout << "Error: " << e.what() << std::endl;
		return 1;
	}
}
