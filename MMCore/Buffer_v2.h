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
#include <deque>

/**
 * BufferSlot represents a contiguous slot in the DataBuffer that holds image
 * data and metadata. It supports exclusive write access with shared read access,
 * using atomics, a mutex, and a condition variable.
 */
class BufferSlot {
public:
    /**
     * Constructor.
     * Initializes the slot with the specified starting byte offset and length.
     * Also initializes atomic variables that track reader and writer access.
     *
     * @param start The starting offset (in bytes) within the buffer.
     * @param length The length (in bytes) of the slot.
     */
    BufferSlot(std::size_t start, std::size_t length);

    /**
     * Destructor.
     */
    ~BufferSlot();

    /**
     * Returns the starting offset (in bytes) of the slot in the buffer.
     *
     * @return The slot's start offset.
     */
    std::size_t GetStart() const;

    /**
     * Returns the length (in bytes) of the slot.
     *
     * @return The slot's length.
     */
    std::size_t GetLength() const;

    /**
     * Stores a detail (for example, image width or height) associated with the slot.
     *
     * @param key The name of the detail.
     * @param value The value of the detail.
     */
    void SetDetail(const std::string &key, std::size_t value);

    /**
     * Retrieves a previously stored detail.
     * Returns 0 if the key is not found.
     *
     * @param key The detail key.
     * @return The stored value or 0 if not found.
     */
    std::size_t GetDetail(const std::string &key) const;

    /**
     * Clears all stored details from the slot.
     */
    void ClearDetails();

    /**
     * Attempts to acquire exclusive write access.
     * It first tries to atomically set the write flag.
     * If another writer is active or if there are active readers, 
     * the write lock is not acquired.
     *
     * @return True if the slot is locked for writing; false otherwise.
     */
    bool AcquireWriteAccess();

    /**
     * Releases exclusive write access.
     * Clears the write flag and notifies waiting readers.
     */
    void ReleaseWriteAccess();

    /**
     * Acquires shared read access.
     * This is a blocking operation — if a writer is active, the caller waits
     * until the writer releases its lock. Once available, the reader count is incremented.
     *
     * @return True when read access has been successfully acquired.
     */
    bool AcquireReadAccess();

    /**
     * Releases shared read access.
     * Decrements the reader count.
     *
     * @return True if this call released the last active reader; false otherwise.
     */
    bool ReleaseReadAccess();

    /**
     * Checks if the slot is currently available for writing.
     * A slot is available if there are no active readers and no active writer.
     *
     * @return True if available for writing, false otherwise.
     */
    bool IsAvailableForWriting() const;

    /**
     * Checks if the slot is available for acquiring read access.
     * A slot is available for reading if no writer presently holds the lock.
     *
     * @return True if available for reading.
     */
    bool IsAvailableForReading() const;

private:
    std::size_t start_;
    std::size_t length_;
    std::map<std::string, std::size_t> details_;
    std::atomic<int> readAccessCountAtomicInt_;
    std::atomic<bool> writeAtomicBool_;
    mutable std::mutex writeCompleteConditionMutex_;
    mutable std::condition_variable writeCompleteCondition_;
};


/**
 * DataBuffer manages a contiguous block of memory divided into BufferSlot objects
 * for storing image data and metadata. It provides thread-safe access for both
 * reading and writing operations and supports configurable overflow behavior.
 *
 * Two data access patterns are provided:
 *  1. Copy-based access.
 *  2. Direct pointer access with an explicit release.
 *
 * Reference counting is used to ensure that memory is managed safely. A slot
 * is recycled when all references (readers and writers) have been released.
 */
class DataBuffer {
public:
    /**
     * Maximum number of released slots to track.
     */
    static const size_t MAX_RELEASED_SLOTS = 50;

    /**
     * Constructor.
     * Initializes the DataBuffer with a specified memory size in MB.
     *
     * @param memorySizeMB The size (in megabytes) of the buffer.
     */
    DataBuffer(unsigned int memorySizeMB);

    /**
     * Destructor.
     */
    ~DataBuffer();

    /**
     * Allocates a contiguous block of memory for the buffer.
     *
     * @param memorySizeMB The amount of memory (in MB) to allocate.
     * @return DEVICE_OK on success.
     */
    int AllocateBuffer(unsigned int memorySizeMB);

    /**
     * Releases the allocated buffer.
     *
     * @return DEVICE_OK on success, or an error if the buffer is already released.
     */
    int ReleaseBuffer();

    /**
     * Copies data into the next available slot in the buffer along with its metadata.
     * The copy-based approach is implemented using a slot acquisition, memory copy, and then
     * slot release.
     *
     * @param data Pointer to the data to be inserted.
     * @param dataSize The size of data (in bytes) being inserted.
     * @param pMd Pointer to the metadata associated with the data.
     * @return DEVICE_OK on success.
     */
    int InsertData(const void* data, size_t dataSize, const Metadata* pMd);

    /**
     * Copies data and metadata from the next available slot in the buffer into the provided destination.
     *
     * @param dataDestination Destination buffer into which data will be copied.
     * @param dataSize On success, returns the size of the copied data.
     * @param md Metadata object to be populated with the data's metadata.
     * @param waitForData If true, block until data becomes available.
     * @return DEVICE_OK on success.
     */
    int CopyNextDataAndMetadata(void* dataDestination, size_t* dataSize, Metadata &md, bool waitForData);

    /**
     * Sets whether the buffer should overwrite old data when it is full.
     * If true, the buffer will recycle the oldest slot when no free slot is available; 
     * if false, an error is returned when writing new data fails due to a full buffer.
     *
     * @param overwrite True to enable overwriting, false to disable.
     * @return DEVICE_OK on success.
     */
    int SetOverwriteData(bool overwrite);

    /**
     * Acquires a pointer to a free slot in the buffer for writing purposes.
     * The caller must later call ReleaseDataWriteSlot after finishing writing.
     *
     * @param slotSize The required size of the write slot.
     * @param slotPointer On success, receives a pointer within the buffer where data can be written.
     * @return DEVICE_OK on success.
     */
    int GetDataWriteSlot(size_t slotSize, void** slotPointer);

    /**
     * Releases the write slot after data writing is complete.
     * This clears the write lock and notifies any waiting reader threads.
     *
     * @param slotPointer Pointer previously obtained from GetDataWriteSlot.
     * @return DEVICE_OK on success.
     */
    int ReleaseDataWriteSlot(void** slotPointer);

    /**
     * Releases read access on a data slot after its contents have been completely read.
     * This makes the slot available for recycling.
     *
     * @param slotPointer Pointer previously obtained from GetNextDataReadPointer.
     * @return DEVICE_OK on success.
     */
    int ReleaseDataReadPointer(void** slotPointer);

    /**
     * Retrieves and removes (consumes) the next available data slot for reading.
     * This method advances the internal reading index.
     *
     * @param dataSize On success, returns the size of the data.
     * @param md Associated metadata for the data.
     * @param waitForData If true, block until data becomes available.
     * @return Pointer to the next available data in the buffer.
     */
    const unsigned char* PopNextDataReadPointer(size_t* dataSize, Metadata &md, bool waitForData);


    /**
     * Peeks at the next unread data slot without consuming it.
     * The slot remains available for subsequent acquisitions.
     *
     * @param slotPointer On success, receives the pointer to the data.
     * @param dataSize On success, returns the size of the data.
     * @param md Associated metadata for the data.
     * @return DEVICE_OK on success, or an error code if no data is available.
     */
    int PeekNextDataReadPointer(void** slotPointer, size_t* dataSize, Metadata &md);

    /**
     * Peeks at the nth unread data slot without consuming it.
     * (n = 0 is equivalent to PeekNextDataReadPointer.)
     *
     * @param n The index of the unread slot.
     * @param dataSize On success, returns the size of the data.
     * @param md Associated metadata for the data.
     * @return const pointer to the data.
     */
    const unsigned char* PeekDataReadPointerAtIndex(size_t n, size_t* dataSize, Metadata &md);


    /**
     * Releases the read access that was acquired by a peek operation.
     * This method releases the temporary read access without consuming the slot.
     *
     * @param slotPointer Pointer previously obtained from a peek method.
     * @return DEVICE_OK on success.
     */
    int ReleasePeekDataReadPointer(void** slotPointer);

    /**
     * Returns the total memory size of the buffer in megabytes.
     *
     * @return The buffer size in MB.
     */
    unsigned int GetMemorySizeMB() const;

    /**
     * Returns the number of currently occupied slots in the buffer.
     *
     * @return The number of occupied slots.
     */
    size_t GetOccupiedSlotCount() const;

    /**
     * Returns the total occupied memory (in bytes) within the buffer.
     *
     * @return The sum of the lengths of all active slots.
     */
    size_t GetOccupiedMemory() const;

    /**
     * Returns the amount of free memory (in bytes) remaining in the buffer.
     *
     * @return The number of free bytes available for new data.
     */
    size_t GetFreeMemory() const;

    /**
     * Returns whether the buffer has been overflowed (i.e. an attempt to
     * allocate a write slot failed because there was no available space).
     */
    bool Overflow() const;

    /**
     * Returns the number of unread data slots in the buffer.
     *
     * @return The number of unread data slots.
     */
    long GetRemainingImageCount() const;

    /**
     * Reinitialize the DataBuffer by clearing all internal data structures,
     * releasing the current buffer, and reallocating a new one.
     * This method uses the existing slotManagementMutex_ to ensure thread safety.
     *
     * @param memorySizeMB New size (in MB) for the buffer.
     * @return DEVICE_OK on success.
     * @throws std::runtime_error if any slot is still actively being read or written.
     */
    int ReinitializeBuffer(unsigned int memorySizeMB);

private:
    // Pointer to the allocated buffer memory.
    char* buffer_;
    // Total size (in bytes) of the allocated buffer.
    size_t bufferSize_;

    // Whether the buffer should overwrite older data when full.
    bool overwriteWhenFull_;

    // New: overflow indicator (set to true if an insert fails because of buffer full)
    bool overflow_;

    // Data structures for tracking active slot usage.
    std::vector<BufferSlot> activeSlotsVector_;
    std::map<size_t, BufferSlot*> activeSlotsByStart_;
    std::vector<size_t> releasedSlots_;

    // The next available offset for a new data slot.
    size_t nextAllocOffset_;

    // Tracks the current slot index for read operations.
    size_t currentSlotIndex_;

    // Synchronization primitives for managing slot access.
    std::condition_variable dataCV_;
    mutable std::mutex slotManagementMutex_;
};
