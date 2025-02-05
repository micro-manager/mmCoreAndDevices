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
#include <memory>

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
 * Each slot begins with a header (BufferSlotRecord) that stores:
 *    - The image data length
 *    - The serialized metadata length (which might be zero)
 *
 * The user-visible routines (e.g. InsertData and CopyNextDataAndMetadata)
 * automatically pack and unpack the header so that the caller need not worry
 * about the extra bytes.
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
     * Inserts data into the next available slot.
     * The data is stored together with its metadata and is arranged as:
     *   [BufferSlotRecord header][image data][serialized metadata]
     *
     * @param data Pointer to the raw image data.
     * @param dataSize The image data byte count.
     * @param pMd Pointer to the metadata. If null, no metadata is stored.
     * @return DEVICE_OK on success.
     */
    int InsertData(const unsigned char* data, size_t dataSize, const Metadata* pMd);

    /**
     * Copies data and metadata from the next available slot in the buffer.
     * The routine examines the header to determine the image byte count
     * and the length of the stored metadata.
     *
     * @param dataDestination Destination buffer where image data is copied.
     * @param imageDataSize On success, returns the image data size (in bytes).
     * @param md Metadata object to be populated (via deserialization of the stored blob).
     * @param waitForData If true, block until data becomes available.
     * @return DEVICE_OK on success.
     */
    int CopyNextDataAndMetadata(unsigned char* dataDestination, size_t* imageDataSize, Metadata &md, bool waitForData);

    /**
     * Sets whether the buffer should overwrite older data when full.
     *
     * @param overwrite True to enable overwriting, false otherwise.
     * @return DEVICE_OK on success.
     */
    int SetOverwriteData(bool overwrite);

    /**
     * Acquires a write slot large enough to hold the image data and metadata.
     * On success, provides two pointers: one to the image data region and one to the metadata region.
     * 
     * The metadataSize parameter specifies the maximum size to reserve for metadata if the exact
     * size is not known at call time. When the slot is released, the metadata will be automatically
     * null-terminated at its actual length, which must not exceed the reserved size.
     *
     * @param imageSize The number of bytes allocated for image data.
     * @param metadataSize The maximum number of bytes to reserve for metadata.
     * @param imageDataPointer On success, receives a pointer to the image data region.
     * @param metadataPointer On success, receives a pointer to the metadata region.
     * @return DEVICE_OK on success.
     */
    int GetDataWriteSlot(size_t imageSize, size_t metadataSize, unsigned char** imageDataPointer, unsigned char** metadataPointer);

    /**
     * Releases a write slot after data has been written.
     *
     * @param imageDataPointer Pointer previously obtained from GetDataWriteSlot.
     *                         This pointer references the start of the image data region.
     * @param actualMetadataBytes Optionally, the actual number of metadata bytes written.
     *         If provided and less than the maximum metadata size reserved, this value
     *         is used to update the header's metadataSize field.
     *         Defaults to -1, which means no update is performed.
     * @return DEVICE_OK on success.
     */
    int ReleaseDataWriteSlot(unsigned char** imageDataPointer, int actualMetadataBytes = -1);

    /**
     * Releases read access for the image data region after its content has been read.
     *
     * @param imageDataPointer Pointer previously obtained from reading routines.
     * @return DEVICE_OK on success.
     */
    int ReleaseDataReadPointer(const unsigned char** imageDataPointer);

    /**
     * Retrieves and removes (consumes) the next available data entry for reading,
     * and populates the provided Metadata object with the associated metadata.
     * The returned pointer points to the beginning of the image data region,
     * immediately after the header.
     *
     * @param md Metadata object to be populated from the stored blob.
     * @param imageDataSize On success, returns the image data size (in bytes).
     * @param waitForData If true, block until data becomes available.
     * @return Pointer to the start of the image data region, or nullptr if none available.
     */
    const unsigned char* PopNextDataReadPointer(Metadata &md, size_t* imageDataSize, bool waitForData);

    /**
     * Peeks at the next unread data entry without consuming it.
     * The header is examined so that the actual image data size (excluding header)
     * is returned.
     *
     * @param imageDataPointer On success, receives a pointer to the image data region.
     * @param imageDataSize On success, returns the image data size (in bytes).
     * @param md Metadata object populated from the stored metadata blob.
     * @return DEVICE_OK on success, error code otherwise.
     */
    int PeekNextDataReadPointer(const unsigned char** imageDataPointer, size_t* imageDataSize, Metadata &md);

    /**
     * Peeks at the nth unread data entry without consuming it.
     * (n = 0 is equivalent to PeekNextDataReadPointer.)
     *
     * @param n The index of the data entry to peek at (0 is next available).
     * @param imageDataSize On success, returns the image data size (in bytes).
     * @param md Metadata object populated from the stored metadata blob.
     * @return Pointer to the start of the image data region.
     */
    const unsigned char* PeekDataReadPointerAtIndex(size_t n, size_t* imageDataSize, Metadata &md);

    /**
     * Releases read access that was acquired by a peek.
     *
     * @param imageDataPointer Pointer previously obtained from a peek.
     * @return DEVICE_OK on success.
     */
    int ReleasePeekDataReadPointer(const unsigned char** imageDataPointer);

    /**
     * Returns the total buffer memory size (in MB).
     *
     * @return Buffer size in MB.
     */
    unsigned int GetMemorySizeMB() const;

    /**
     * Returns the number of occupied slots in the buffer.
     *
     * @return Occupied slot count.
     */
    size_t GetOccupiedSlotCount() const;

    /**
     * Returns the total occupied memory (in bytes).
     *
     * @return Sum of active slot lengths.
     */
    size_t GetOccupiedMemory() const;

    /**
     * Returns the amount of free memory (in bytes) remaining.
     *
     * @return Free byte count.
     */
    size_t GetFreeMemory() const;

    /**
     * Indicates whether a buffer overflow occurred (i.e. an insert failed because
     * no appropriate slot was available).
     *
     * @return True if overflow has happened, false otherwise.
     */
    bool Overflow() const;

    /**
     * Returns the number of unread slots in the buffer.
     *
     * @return Unread slot count.
     */
    long GetRemainingImageCount() const;

    /**
     * Reinitializes the DataBuffer by clearing its structures, releasing the current
     * buffer, and allocating a new one.
     *
     * @param memorySizeMB New buffer size (in MB).
     * @return DEVICE_OK on success.
     * @throws std::runtime_error if any slot is still actively in use.
     */
    int ReinitializeBuffer(unsigned int memorySizeMB);

private:
    // Pointer to the allocated block.
    unsigned char* buffer_;
    // Total allocated size in bytes.
    size_t bufferSize_;

    // Whether to overwrite old data when full.
    bool overwriteWhenFull_;

    // Overflow flag (set if insert fails due to full buffer).
    bool overflow_;

    // Data structures used to track active slots.
    std::vector<std::unique_ptr<BufferSlot>> activeSlotsVector_;
    std::map<size_t, BufferSlot*> activeSlotsByStart_;
    std::vector<size_t> releasedSlots_;

    // Next free offset within the buffer.
    size_t nextAllocOffset_;

    // Index tracking the next slot for read.
    size_t currentSlotIndex_;

    // Synchronization primitives for slot management.
    std::condition_variable dataCV_;
    mutable std::mutex slotManagementMutex_;
};
