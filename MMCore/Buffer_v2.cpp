///////////////////////////////////////////////////////////////////////////////
// FILE:          Buffer_v2.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     MMCore
//-----------------------------------------------------------------------------
// DESCRIPTION:   Generic implementation of a buffer for storing image data and
//                metadata. Provides thread-safe access for reading and writing
//                with configurable overflow behavior.
////
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
#include <thread>   // for std::this_thread::yield if needed

///////////////////////////////////////////////////////////////////////////////
// DataBuffer Implementation
///////////////////////////////////////////////////////////////////////////////

DataBuffer::DataBuffer(unsigned int memorySizeMB, const std::string& name) {
    // Convert MB to bytes for internal allocation
    AllocateBuffer(memorySizeMB, name);
}

DataBuffer::~DataBuffer() {
    // Cleanup
    delete[] buffer_;
}

/**
 * Allocate a character buffer
 * @param memorySizeMB The size (in MB) of the buffer to allocate.
 * @param name The name of the buffer.
 * @return Error code (0 on success).
 */
int DataBuffer::AllocateBuffer(unsigned int memorySizeMB, const std::string& name) {
    // Convert MB to bytes (1 MB = 1048576 bytes)
    size_t numBytes = static_cast<size_t>(memorySizeMB) * (1ULL << 20);
    buffer_ = new char[numBytes];
    bufferSize_ = numBytes;
    bufferName_ = name;
    return DEVICE_OK;
}

/**
 * Release the buffer if it matches the given name.
 * @param name The name of the buffer to release.
 * @return Error code (0 on success, error if buffer not found or already released).
 */
int DataBuffer::ReleaseBuffer(const std::string& name) {
    if (buffer_ != nullptr && bufferName_ == name) {
        delete[] buffer_;
        buffer_ = nullptr;
        bufferSize_ = 0;
        bufferName_.clear();
        return DEVICE_OK;
    }
    // TODO: Handle errors if other parts of the system still hold pointers.
    return DEVICE_ERR;
}

/**
 * @brief Copy data into the next available slot in the buffer.
 * 
 * Returns the size of the copied data through dataSize.
 * TODO: Implementing code should check the device type of the caller, and ensure that 
 * all required metadata for interpreting its image data is there.
 * Note: this can be implemented in terms of Get/Release slot + memcopy.
 * 
 * @param caller The device calling this function.
 * @param data The data to be copied.
 * @param dataSize Size of the data to copy.
 * @param serializedMetadata The associated metadata.
 * @return Error code (0 on success).
 */
int DataBuffer::InsertData(const MM::Device *caller, const void* data, 
                           size_t dataSize, const std::string& serializedMetadata) {
    // TODO: Create a slot, copy the data into it, then release write access on the slot.
    // Also, ensure that a slot is not garbage-collected while data remains available.
    return DEVICE_OK;
}


/**
 * Check if a new slot has been fully written in this buffer
 * @return true if new data is ready, false otherwise
 */
bool DataBuffer::IsNewDataReady() {
    // TODO: Implement checking logic based on the slot state.
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
    // Basic implementation:
    // TODO: Use slot management to return data from the next available slot.
    return DEVICE_OK;
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
    overwriteWhenFull_ = overwrite;
    return DEVICE_OK;
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
int DataBuffer::GetWritingSlot(const MM::Device *caller, void** slot,
                           size_t slotSize, const std::string& serializedMetadata) {
    // TODO: For now, we simply return a pointer into the buffer; a full implementation
    // would create and manage a BufferSlot and assign it with exclusive write access.
    if (slot == nullptr || slotSize > bufferSize_)
        return DEVICE_ERR;
        // TODO: understnad this pointer stuff
    *slot = static_cast<void*>(buffer_);
    return DEVICE_OK;
}

/**
 * @brief Release a data slot after writing is complete.
 * 
 * @param caller The device calling this function.
 * @param buffer The buffer to be released.
 * @return Error code (0 on success).
 */
int DataBuffer::ReleaseWritingSlot(const MM::Device *caller, void* buffer) {
    // TODO
    return DEVICE_OK;
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
    // TODO: Implement camera-specific metadata creation.
    return DEVICE_OK;
}

unsigned int DataBuffer::GetMemorySizeMB() const {
    // Convert bytes to MB (1 MB = 1048576 bytes)
    return static_cast<unsigned int>(bufferSize_ >> 20);
}

///////////////////////////////////////////////////////////////////////////////
// BufferSlot Implementation
///////////////////////////////////////////////////////////////////////////////

/**
 * Constructor.
 * Initializes the slot with the specified starting byte offset and length.
 * Also initializes atomic variables that track reader and writer access.
 */
BufferSlot::BufferSlot(std::size_t start, std::size_t length)
    : start_(start), length_(length),
      readAccessCountAtomicInt_(0),
      writeAtomicBool_(false)
{
    // No readers are active and the write lock is free upon construction.
}

/**
 * Destructor.
 * Currently no dynamic memory is used inside BufferSlot, so nothing needs to be cleaned up.
 */
BufferSlot::~BufferSlot() {
    // No explicit cleanup required here.
}

/**
 * Returns the start offset (in bytes) of the slot within the main buffer.
 */
std::size_t BufferSlot::GetStart() const {
    return start_;
}

/**
 * Returns the length (in bytes) of the slot.
 */
std::size_t BufferSlot::GetLength() const {
    return length_;
}

/**
 * Sets a detail for this slot using the provided key and value.
 * Typically used to store metadata information (e.g. width, height).
 */
void BufferSlot::SetDetail(const std::string &key, std::size_t value) {
    details_[key] = value;
}

/**
 * Retrieves a previously set detail.
 * Returns 0 if the key is not found.
 */
std::size_t BufferSlot::GetDetail(const std::string &key) const {
    auto it = details_.find(key);
    return (it != details_.end()) ? it->second : 0;
}

/**
 * Clears all additional details associated with this slot.
 */
void BufferSlot::ClearDetails() {
    details_.clear();
}

/**
 * Attempts to acquire exclusive write access.
 * This method first attempts to set the write flag atomically.
 * If it fails, that indicates another writer holds the lock.
 * Next, it attempts to confirm that no readers are active.
 * If there are active readers, it reverts the write flag and returns false.
 */
bool BufferSlot::AcquireWriteAccess() {
    bool expected = false;
    // Attempt to atomically set the write flag.
    if (!writeAtomicBool_.compare_exchange_strong(expected, true, std::memory_order_acquire)) {
        // A writer is already active.
        return false;
    }
    // Ensure no readers are active by checking the read counter.
    int expectedReaders = 0;
    if (!readAccessCountAtomicInt_.compare_exchange_strong(expectedReaders, 0, std::memory_order_acquire)) {
        // Active readers are present; revert the write lock.
        writeAtomicBool_.store(false, std::memory_order_release);
        return false;
    }
    // Exclusive write access has been acquired.
    return true;
}

/**
 * Releases exclusive write access.
 * The writer flag is cleared, and waiting readers are notified so that
 * they may acquire shared read access once the write is complete.
 */
void BufferSlot::ReleaseWriteAccess() {
    // Publish all writes by releasing the writer flag.
    writeAtomicBool_.store(false, std::memory_order_release);
    // Notify waiting readers (using the condition variable)
    // that the slot is now available for read access.
    std::lock_guard<std::mutex> lock(writeCompleteConditionMutex_);
    writeCompleteCondition_.notify_all();
}

/**
 * Acquires shared read access.
 * This is a blocking operation – if a writer is active,
 * the calling thread will wait until the writer releases its lock.
 * Once unlocked, the method increments the reader count.
 */
bool BufferSlot::AcquireReadAccess() {
    // Acquire the mutex associated with the condition variable.
    std::unique_lock<std::mutex> lock(writeCompleteConditionMutex_);
    // Block until no writer is active.
    writeCompleteCondition_.wait(lock, [this]() {
         return !writeAtomicBool_.load(std::memory_order_acquire);
    });
    // Now that there is no writer, increment the reader counter.
    readAccessCountAtomicInt_.fetch_add(1, std::memory_order_acquire);
    return true;
}

/**
 * Releases shared read access.
 * The reader count is decremented using release semantics to ensure that all
 * prior read operations complete before the decrement is visible to other threads.
 */
void BufferSlot::ReleaseReadAccess() {
    readAccessCountAtomicInt_.fetch_sub(1, std::memory_order_release);
}

/**
 * Checks if the slot is available for acquiring write access.
 * A slot is available for writing if there are no active readers and no writer.
 */
bool BufferSlot::IsAvailableForWriting() const {
    return (readAccessCountAtomicInt_.load(std::memory_order_acquire) == 0) &&
           (!writeAtomicBool_.load(std::memory_order_acquire));
}

/**
 * Checks if the slot is available for acquiring read access.
 * A slot is available for reading if no writer currently holds the lock.
 */
bool BufferSlot::IsAvailableForReading() const {
    return !writeAtomicBool_.load(std::memory_order_acquire);
}
