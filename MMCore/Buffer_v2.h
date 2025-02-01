///////////////////////////////////////////////////////////////////////////////
// FILE:          Buffer_v2.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     MMCore
//-----------------------------------------------------------------------------
// DESCRIPTION:   Generic implementation of a buffer for storing image data and
//                metadata. Provides thread-safe access for reading and writing
//                with configurable overflow behavior.
//
// The buffer is organized into slots (BufferSlot objects), each of which
// supports exclusive write access and shared read access. Read access is
// delivered using const pointers and is counted via an atomic counter, while
// write access requires acquiring an exclusive lock. This ensures that once a
// read pointer is given out it cannot be misused for writing.
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
// AUTHOR:        Henry Pinkard,  01/31/2025
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "../MMDevice/ImageMetadata.h"
#include "../MMDevice/MMDevice.h"
#include <mutex>
#include <string>
#include <map>
#include <cstddef>
#include <vector>
#include <atomic>
#include <condition_variable>

/**
 * BufferSlot represents a contiguous slot in the DataBuffer that holds image
 * data and metadata. It manages exclusive (write) and shared (read) access
 * using atomics, a mutex, and a condition variable.
 */
class BufferSlot {
public:
    // Constructor: Initializes the slot with the given start offset and length.
    BufferSlot(std::size_t start, std::size_t length);
    // Destructor.
    ~BufferSlot();

    // Returns the start offset (in bytes) of the slot.
    std::size_t GetStart() const;
    // Returns the length (in bytes) of the slot.
    std::size_t GetLength() const;

    // Stores a detail (e.g., width, height) associated with the slot.
    void SetDetail(const std::string &key, std::size_t value);
    // Retrieves a stored detail; returns 0 if the key is not found.
    std::size_t GetDetail(const std::string &key) const;
    // Clears all stored details.
    void ClearDetails();

    // --- Methods for synchronizing access ---

    /**
     * Try to acquire exclusive write access.
     * Returns true on success, false if the slot is already locked for writing
     * or if active readers exist.
     */
    bool AcquireWriteAccess();
    /**
     * Release exclusive write access.
     * Clears the write flag and notifies waiting readers.
     */
    void ReleaseWriteAccess();
    /**
     * Acquire shared read access by blocking until no writer is active.
     * Once the waiting condition is met, the reader count is incremented.
     * Returns true when read access has been acquired.
     */
    bool AcquireReadAccess();
    /**
     * Release shared read access.
     * Decrements the reader count using release semantics.
     */
    void ReleaseReadAccess();

    /**
     * Return true if the slot is available for acquiring write access (i.e.,
     * no active writer or reader).
     */
    bool IsAvailableForWriting() const;
    /**
     * Return true if the slot is available for acquiring read access 
     * (no active writer).
     */
    bool IsAvailableForReading() const;

private:
    // Basic slot information.
    std::size_t start_;               // Byte offset within the buffer.
    std::size_t length_;              // Length of the slot in bytes.
    std::map<std::string, std::size_t> details_;  // Additional details (e.g., image dimensions).

    // Synchronization primitives.
    std::atomic<int> readAccessCountAtomicInt_;  // Count of active readers.
    std::atomic<bool> writeAtomicBool_;            // True if the slot is locked for writing.
    mutable std::mutex writeCompleteConditionMutex_;  // Mutex for condition variable.
    mutable std::condition_variable writeCompleteCondition_;  // Condition variable for blocking readers.
};


/**
 * DataBuffer manages a large contiguous memory area, divided into BufferSlot
 * objects for storing image data and metadata. It supports two data access
 * patterns: copy-based access and direct pointer access via retrieval of slots.
 */
class DataBuffer {
public:
    DataBuffer(unsigned int memorySizeMB, const std::string& name);
    ~DataBuffer();

    // Buffer Allocation and Destruction
    int AllocateBuffer(unsigned int memorySizeMB, const std::string& name);
    int ReleaseBuffer(const std::string& name);

	// TODO: Other versions for allocating buffers Java, Python

	
    /**
     * Get the total memory size of the buffer in megabytes
     * @return Size of the buffer in MB
     */
    unsigned int GetMemorySizeMB() const;

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
	 * Copy data into the next available slot in the buffer.
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
	int InsertData(const MM::Device *caller, const void* data, size_t dataSize, 
                   const std::string& serializedMetadata);

	/**
	 * Get a pointer to the next available data slot in the buffer for writing.
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
	int GetWritingSlot(const MM::Device *caller, void** slot, size_t slotSize, 
                       const std::string& serializedMetadata);

    /**
     * Release a data slot after writing is complete.
     * @param caller The device calling this function.
     * @param buffer The slot to be released.
     * @return Error code (0 on success).
     */
    int ReleaseWritingSlot(const MM::Device *caller, void* buffer);




 	////// Camera API //////

    // Set the buffer for a camera to write into
    int SetCameraBuffer(const std::string& camera, void* buffer);

    // Get a pointer to a heap allocated Metadata object with the required fields filled in
    int CreateCameraRequiredMetadata(Metadata**, int width, int height, int bitDepth);

private:
    // Basic buffer management.
    char* buffer_;
    size_t bufferSize_;
    std::string bufferName_;

    // Configuration.
    bool overwriteWhenFull_;

    // List of active buffer slots. Each slot manages its own read/write access.
    std::vector<BufferSlot> activeSlots_;
};
