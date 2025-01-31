///////////////////////////////////////////////////////////////////////////////
// FILE:          Buffer_v2.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     MMCore
//-----------------------------------------------------------------------------
// DESCRIPTION:   Generic implementation of a buffer for storing image data and
//                metadata. Provides thread-safe access for reading and writing
//                with configurable overflow behavior.
//              
// COPYRIGHT:     Henry Pinkard, 2025
//
// LICENSE:       This file is distributed under the "Lesser GPL" (LGPL) license.
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
// AUTHOR:        Henry Pinkard, henry.pinkard@gmail.com, 01/31/2025
// 

#pragma once

#include "Metadata.h"
#include "../MMDevice/MMDevice.h"
#include <mutex>
#include <string>

class DataBuffer {
    public:
        DataBuffer(size_t numBytes, const char* name);
        ~DataBuffer();
			

	/////// Buffer Allocation and Destruction

	// C version
	int AllocateBuffer(size_t numBytes, const char* name);
	//// Is there a need for a name?
	//// Maybe makes sense for Core to assing a unique integer
	int ReleaseBuffer(const char* name);

	// TODO: Other versions for allocating buffers Java, Python


	/////// Monitoring the Buffer ///////
	int GetAvailableBytes(void* buffer, size_t* availableBytes);
	int GetNumSlotsUsed(void* buffer, size_t* numSlotsUsed);



	//// Configuration options
	/**
	 * Configure whether to overwrite old data when buffer is full.
	 * 
	 * If true, when there are no more slots available for writing because
	 * images haven't been read fast enough, then automatically recycle the 
	 * oldest slot(s) in the buffer as needed in order to make space for new images.
	 * This is suitable for situations when its okay to drop frames, like live
	 * view when data is not being saved.
	 * 
	 * If false, then throw an exception if the buffer becomes full.
	 * 
	 * @param overwrite Whether to enable overwriting of old data
	 * @return Error code (0 on success)
	 */
	int SetOverwriteData(bool overwrite);



	/////// Getting Data Out ///////

	/**
	 * Check if a new slot has been fully written in this buffer
	 * @return true if new data is ready, false otherwise
	 */
	bool IsNewDataReady();

	/**
	 * Copy the next available data and metadata from the buffer
	 * @param dataDestination Destination buffer to copy data into
	 * @param dataSize Returns the size of the copied data, or 0 if no data available
	 * @param md Metadata object to populate
	 * @param waitForData If true, block until data becomes available
	 * @return Error code (0 on success)
	 */
	int CopyNextDataAndMetadata(void* dataDestination, 
					size_t* dataSize, Metadata &md,	bool waitForData);


	/**
	 * Copy the next available metadata from the buffer
	 * Returns the size of the copied metadata through metadataSize,
	 * or 0 if no metadata is available
	 */
	int CopyNextMetadata(void* buffer, Metadata &md);

	/**
	 * Get a pointer to the next available data slot in the buffer
	 * The caller must release the slot using ReleaseNextDataAndMetadata
	 * If awaitReady is false and the data was inserted using GetWritingSlot, it
	 * is possible to read the data as it is being written (e.g. to monitor progress)
	 * of large or slow image being written
	 *
	 * Internally this will use a std::shared_ptr<Slot>
	 */
	int GetNextSlotPointer(void** slotPointer,  size_t* dataSize, 
								Metadata &md, bool awaitReady=true);

	/**
	 * Release the next data slot and its associated metadata
	 */
	int ReleaseNextSlot(void** slotPointer);


	////// Writing Data into buffer //////

	/**
	 * @brief Copy data into the next available slot in the buffer.
	 * 
	 * Returns the size of the copied data through dataSize.
	 * Implementing code should check the device type of the caller, and ensure that 
	 * all required metadata for interpreting its image data is there.
	 * Note: this can be implemented in terms of Get/Release slot + memcopy.
	 * 
	 * @param caller The device calling this function.
	 * @param data The data to be copied into the buffer.
	 * @param dataSize The size of the data to be copied.
	 * @param serializedMetadata The serialized metadata associated with the data.
	 * @return Error code (0 on success).
	 */
	int InsertData(const MM::Device *caller, const void* data, size_t dataSize, const char* serializedMetadata);

	/**
	 * @brief Get a pointer to the next available data slot in the buffer for writing.
	 * 
	 * The caller must release the slot using ReleaseDataSlot after writing is complete.
	 * Internally this will use a std::unique_ptr<Slot>.
	 * 
	 * @param caller The device calling this function.
	 * @param slot Pointer to the slot where data will be written.
	 * @param slotSize The size of the slot.
	 * @param serializedMetadata The serialized metadata associated with the data.
	 * @return Error code (0 on success).
	 */
	int GetWritingSlot(const MM::Device *caller, void** slot, size_t slotSize, const char* serializedMetadata);

	/**
	 * @brief Release a data slot after writing is complete.
	 * 
	 * @param caller The device calling this function.
	 * @param buffer The buffer to be released.
	 * @return Error code (0 on success).
	 */
	int ReleaseWritingSlot(const MM::Device *caller, void* buffer);




 	////// Camera API //////

    // Set the buffer for a camera to write into
    int SetCameraBuffer(const char* camera, void* buffer);

    // Get a pointer to a heap allocated Metadata object with the required fields filled in
    int CreateCameraRequiredMetadata(Metadata**, int width, int height, int bitDepth);

    private:
        // Basic buffer management
        void* buffer_;
        size_t bufferSize_;
        std::string bufferName_;

        // Read/write positions
        size_t writePos_;
        size_t readPos_;

        // Configuration
        bool overwriteWhenFull_;

        // Mutex for thread safety
        std::mutex mutex_;
};
