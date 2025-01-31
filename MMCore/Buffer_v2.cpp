///////////////////////////////////////////////////////////////////////////////
// FILE:          Buffer_v2.cpp
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
// AUTHOR:        Henry Pinkard,  01/31/2025


/*
Design Overview:

The buffer is designed as a flexible data structure for storing image data and metadata:

Buffer Structure:
- A large block of contiguous memory divided into slots

Slots:
- Contiguous sections within the buffer that can vary in size
- Support exclusive write access with shared read access
- Memory management through reference counting:
  - Writers get exclusive ownership during writes
  - Readers can get shared read-only access
  - Slots are recycled when all references are released

Data Access:
- Two access patterns supported:
  1. Copy-based access
  2. Direct pointer access with explicit release
- Reference counting ensures safe memory management
- Slots become available for recycling when:
  - Writing is complete (via Insert or GetWritingSlot+Release)
  - All readers have released their references

Metadata Handling:
- Devices must specify PixelType when adding data
- Device-specific metadata requirements (e.g. image dimensions) are handled at the
  device API level rather than in the buffer API to maintain clean separation
*/


#include "Buffer_v2.h"
#include <cstring>

DataBuffer::DataBuffer(size_t numBytes, const char* name) {
    // Initialize basic buffer with given size
    AllocateBuffer(numBytes, name);
}

DataBuffer::~DataBuffer() {
    // Cleanup
    delete[] buffer_;
}

/**
 * Allocate a character buffer
 * @param numBytes The size of the buffer to allocate.
 * @param name The name of the buffer.
 * @return Error code (0 on success).
 */
int DataBuffer::AllocateBuffer(size_t numBytes, const char* name) {
    buffer_ = new char[numBytes];
    bufferSize_ = numBytes;
    bufferName_ = name;
    // TODO: Store the buffer in a member variable for later use
    return DEVICE_OK;
}

/**
 * Release the buffer if it matches the given name.
 * @param name The name of the buffer to release.
 * @return Error code (0 on success, error if buffer not found or already released).
 */
int DataBuffer::ReleaseBuffer(const char* name) {
    if (buffer_ != nullptr && bufferName_ == name) {
        delete[] buffer_;
        buffer_ = nullptr;
        bufferSize_ = 0;
        bufferName_ = nullptr;
        return DEVICE_OK;
    }
    // TODO: throw errors if other code holds pointers on stuff
    return DEVICE_ERR; // Return an error if the buffer is not found or already released
}

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
int DataBuffer::InsertData(const MM::Device *caller, const void* data, size_t dataSize, const char* serializedMetadata) {
    // Basic implementation - just copy data
    // TODO: Add proper buffer management and metadata handling
    return 0;
}

/**
 * Check if a new slot has been fully written in this buffer
 * @return true if new data is ready, false otherwise
 */
bool DataBuffer::IsNewDataReady() {
    // Basic implementation
    return false;
}

/**
 * Copy the next available data and metadata from the buffer
 * @param dataDestination Destination buffer to copy data into
 * @param dataSize Returns the size of the copied data, or 0 if no data available
 * @param md Metadata object to populate
 * @param waitForData If true, block until data becomes available
 * @return Error code (0 on success)
 */
int DataBuffer::CopyNextDataAndMetadata(void* dataDestination, 
                size_t* dataSize, Metadata &md, bool waitForData) {
    // Basic implementation
    // TODO: Add proper data copying and metadata handling
    return 0;
}

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
int DataBuffer::SetOverwriteData(bool overwrite) {
    // Basic implementation
    return 0;
}

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
int DataBuffer::GetWritingSlot(const MM::Device *caller, void** slot, size_t slotSize, const char* serializedMetadata) {
    // Basic implementation
    return 0;
}

/**
 * @brief Release a data slot after writing is complete.
 * 
 * @param caller The device calling this function.
 * @param buffer The buffer to be released.
 * @return Error code (0 on success).
 */
int DataBuffer::ReleaseWritingSlot(const MM::Device *caller, void* buffer) {
    // Basic implementation
    return 0;
}

/**
 * Get a pointer to a heap allocated Metadata object with the required fields filled in
 * @param md Pointer to the Metadata object to be created.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param bitDepth The bit depth of the image.
 * @return Error code (0 on success).
 */
int DataBuffer::CreateCameraRequiredMetadata(Metadata** md, int width, int height, int bitDepth) {
    // Basic implementation
    // TODO: Create metadata with required camera fields
    return 0;
}
