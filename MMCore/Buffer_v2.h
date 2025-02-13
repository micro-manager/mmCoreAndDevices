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
// delivered using const pointers and is tracked via RAII-based synchronization.
// Write access is protected via an exclusive lock. This ensures that once a
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
#include <shared_mutex>   
#include <string>
#include <map>
#include <cstddef>
#include <vector>
#include <condition_variable>
#include <deque>
#include <memory>
#include "TaskSet_CopyMemory.h"

/**
 * BufferSlot represents a contiguous slot in the DataBuffer that holds image
 * data and metadata. It uses RAII-based locking with std::shared_timed_mutex to
 * support exclusive write access and concurrent shared read access.
 */
class BufferSlot {
public:
    /**
     * Constructor.
     * @param start The starting offset (in bytes) within the buffer.
     * @param length The length (in bytes) of the slot.
     */
    BufferSlot(std::size_t start, std::size_t length);

    /**
     * Destructor.
     */
    ~BufferSlot();

    /**
     * Returns the starting offset (in bytes) of the slot.
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
     * Attempts to acquire exclusive write access without blocking.
     * @return True if the write lock was acquired; false otherwise.
     */
    bool TryAcquireWriteAccess();

    /**
     * Acquires exclusive write access (blocking call).
     * This method will block until exclusive access is granted.
     */
    void AcquireWriteAccess();

    /**
     * Releases exclusive write access.
     */
    void ReleaseWriteAccess();

    /**
     * Acquires shared read access (blocking).
     */
    void AcquireReadAccess();

    /**
     * Releases shared read access.
     */
    void ReleaseReadAccess();

    /**
     * Checks if the slot is currently available for writing.
     * A slot is available if no thread holds either a write lock or any read lock.
     * @return True if available for writing.
     */
    bool IsAvailableForWriting() const;

    /**
     * Checks if the slot is available for acquiring read access.
     * @return True if available for reading.
     */
    bool IsAvailableForReading() const;

private:
    std::size_t start_;
    std::size_t length_;
    // RAII-based locking using std::shared_timed_mutex.
    mutable std::shared_timed_mutex rwMutex_;
};


/**
 * DataBuffer manages a contiguous block of memory divided into BufferSlot objects
 * for storing image data and metadata. It ensures thread-safe access for both
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
    static const size_t MAX_RELEASED_SLOTS = 50;

    /**
     * Constructor.
     * @param memorySizeMB The size (in megabytes) of the buffer.
     */
    DataBuffer(unsigned int memorySizeMB);

    /**
     * Destructor.
     */
    ~DataBuffer();

    /**
     * Allocates the memory buffer.
     * @param memorySizeMB Size in megabytes.
     * @return DEVICE_OK on success.
     */
    int AllocateBuffer(unsigned int memorySizeMB);

    /**
     * Releases the memory buffer.
     * @return DEVICE_OK on success.
     */
    int ReleaseBuffer();

    /**
     * Inserts image data along with metadata into the buffer.
     * @param data Pointer to the raw image data.
     * @param dataSize The image data size in bytes.
     * @param pMd Pointer to the metadata (can be null if not applicable).
     * @return DEVICE_OK on success.
     */
    int InsertData(const unsigned char* data, size_t dataSize, const Metadata* pMd);

    /**
     * Sets whether the buffer should overwrite old data when full.
     * @param overwrite True to enable overwriting.
     * @return DEVICE_OK on success.
     */
    int SetOverwriteData(bool overwrite);

    /**
     * Acquires a write slot large enough to hold the image data and metadata.
     * On success, returns pointers for the image data and metadata regions.
     *     *
     * @param imageSize The number of bytes reserved for image data.
     * @param metadataSize The maximum number of bytes reserved for metadata.
     * @param imageDataPointer On success, receives a pointer to the image data region.
     * @param metadataPointer On success, receives a pointer to the metadata region.
     * @return DEVICE_OK on success.
     */
    int AcquireWriteSlot(size_t imageSize, size_t metadataSize,
                         unsigned char** imageDataPointer, unsigned char** metadataPointer);

    /**
     * Finalizes (releases) a write slot after data has been written.
     *     *
     * @param imageDataPointer Pointer previously obtained from AcquireWriteSlot.
     * @param actualMetadataBytes Optionally, the actual number of metadata bytes written.
     *                            Defaults to -1 (no update).
     * @return DEVICE_OK on success.
     */
    int FinalizeWriteSlot(unsigned char* imageDataPointer, int actualMetadataBytes = -1);

    /**
     * Releases read access for the image data after reading.
     * @param imageDataPointer Pointer previously obtained from reading routines.
     * @return DEVICE_OK on success.
     */
    int ReleaseDataReadPointer(const unsigned char* imageDataPointer);

    /**
     * Retrieves and consumes the next available data entry for reading,
     * populating the provided Metadata object.
     * @param md Metadata object to populate.
     * @param imageDataSize On success, returns the image data size in bytes.
     * @param waitForData If true, blocks until data is available.
     * @return Pointer to the image data region, or nullptr if none available.
     */
    const unsigned char* PopNextDataReadPointer(Metadata &md, size_t *imageDataSize, bool waitForData);

    /**
     * Peeks at the next unread data entry without consuming it.
     * @param imageDataPointer On success, receives a pointer to the image data region.
     * @param imageDataSize On success, returns the image data size in bytes.
     * @param md Metadata object populated from the stored metadata.
     * @return DEVICE_OK on success.
     */
    int PeekNextDataReadPointer(const unsigned char** imageDataPointer, size_t* imageDataSize, Metadata &md);

    /**
     * Peeks at the nth unread data entry without consuming it.
     * (n = 0 is equivalent to PeekNextDataReadPointer).
     * @param n Index of the data entry to peek at (0 for next available).
     * @param imageDataSize On success, returns the image data size in bytes.
     * @param md Metadata object populated from the stored metadata.
     * @return Pointer to the start of the image data region.
     */
    const unsigned char* PeekDataReadPointerAtIndex(size_t n, size_t* imageDataSize, Metadata &md);

    /**
     * Releases read access that was acquired by a peek.
     * @param imageDataPointer Pointer previously obtained from a peek.
     * @return DEVICE_OK on success.
     */
    int ReleasePeekDataReadPointer(const unsigned char** imageDataPointer);

    /**
     * Returns the total buffer memory size (in MB).
     * @return Buffer size in MB.
     */
    unsigned int GetMemorySizeMB() const;

    /**
     * Returns the number of occupied buffer slots.
     * @return Occupied slot count.
     */
    size_t GetOccupiedSlotCount() const;

    /**
     * Returns the total occupied memory in bytes.
     * @return Sum of active slot lengths.
     */
    size_t GetOccupiedMemory() const;

    /**
     * Returns the amount of free memory remaining in bytes.
     * @return Free byte count.
     */
    size_t GetFreeMemory() const;

    /**
     * Indicates whether a buffer overflow has occurred.
     * @return True if an insert failed (buffer full), false otherwise.
     */
    bool Overflow() const;

    /**
     * Returns the number of unread slots in the buffer.
     * @return Unread slot count.
     */
    long GetActiveSlotCount() const;

    /**
     * Reinitializes the DataBuffer, clearing its structures and allocating a new buffer.
     * @param memorySizeMB New buffer size (in MB).
     * @return DEVICE_OK on success.
     * @throws std::runtime_error if any slot is still in use.
     */
    int ReinitializeBuffer(unsigned int memorySizeMB);

    /**
     * Returns the image width from the metadata stored with the image data.
     *
     * @param imageDataPtr Pointer to the image data.
     * @return Image width.
     */
    unsigned GetImageWidth(const unsigned char* imageDataPtr) const;

    /**
     * Returns the image height from the metadata stored with the image data.
     *
     * @param imageDataPtr Pointer to the image data.
     * @return Image height.
     */
    unsigned GetImageHeight(const unsigned char* imageDataPtr) const;

    /**
     * Returns the bytes per pixel from the metadata stored with the image data.
     *
     * @param imageDataPtr Pointer to the image data.
     * @return Bytes per pixel.
     */
    unsigned GetBytesPerPixel(const unsigned char* imageDataPtr) const;

    /**
     * Returns the image bit depth from the metadata stored with the image data.
     *
     * @param imageDataPtr Pointer to the image data.
     * @return Image bit depth.
     */
    unsigned GetImageBitDepth(const unsigned char* imageDataPtr) const;

    /**
     * Returns the number of components in the image data from the metadata stored with the image data.
     *
     * @param imageDataPtr Pointer to the image data.
     * @return Number of components.
     */
    unsigned GetNumberOfComponents(const unsigned char* imageDataPtr) const;

    /**
     * Returns the image buffer size from the metadata stored with the image data.
     *
     * @param imageDataPtr Pointer to the image data.
     * @return Image buffer size.
     */
    long GetImageBufferSize(const unsigned char* imageDataPtr) const;

private:
    /**
     * Removes a slot from active tracking and adds it to the free region list.
     * Caller must hold slotManagementMutex_.
     *
     * @param offset The buffer offset of the slot to remove.
     * @param it Iterator to the slot in activeSlotsByStart_.
     */
    void RemoveSlotFromActiveTracking(size_t offset, std::map<size_t, BufferSlot*>::iterator it);

    /**
     * Creates a new BufferSlot at an allocated region, registers it in the internal
     * tracking structures, initializes its header, and sets the output pointers for image
     * data and metadata.
     *
     * Allocation is performed differently based on the overwrite mode:
     *
     *  - Overwrite mode:
     *       Uses nextAllocOffset_ with wrap-around.
     *
     *  - Non-overwrite mode:
     *       First, we try to reuse recycled slots in order of their release. If none are 
     *       available or usable, we then check if there is a free region large enough to hold
     *       the slot. If there is, we use that region. If there is no suitable region, an exception is 
     *       thrown.
     *
     * @param candidateStart The starting offset candidate for the slot.
     * @param totalSlotSize The total size (in bytes) of the slot, including header, image data, and metadata.
     * @param imageDataSize The size (in bytes) to reserve for the image data.
     * @param imageDataPointer Output pointer for the caller to access the image data region.
     * @param metadataPointer Output pointer for the caller to access the metadata region.
     * @param fromFreeRegion If true, indicates that the candidate was selected from a free region.
     * @return DEVICE_OK on success, or an error code if allocation fails.
     */
    int CreateAndRegisterNewSlot(size_t candidateStart, size_t totalSlotSize, size_t imageDataSize,
                                  unsigned char** imageDataPointer, unsigned char** metadataPointer,
                                  bool fromFreeRegion);

    /**
     * Inserts a free region into the freeRegions_ list, merging with adjacent regions if necessary.
     *
     * @param offset The starting offset of the freed region.
     * @param size The size (in bytes) of the freed region.
     */
    void InsertFreeRegion(size_t offset, size_t size);

    // Memory managed by the DataBuffer.
    unsigned char* buffer_;
    size_t bufferSize_;

    // Whether to overwrite old data when full.
    bool overwriteWhenFull_;

    // Overflow flag (set if insert fails due to full buffer).
    bool overflow_;

    // Tracking of active slots and free memory regions.
    std::vector<std::unique_ptr<BufferSlot>> activeSlotsVector_;
    std::map<size_t, BufferSlot*> activeSlotsByStart_;

    // Free region list for non-overwrite mode.
    // Map from starting offset -> region size (in bytes).
    std::map<size_t, size_t> freeRegions_;
    std::vector<size_t> releasedSlots_;

    // Next free offset within the buffer.
    // In overwrite mode, new allocations will come from this pointer.
    size_t nextAllocOffset_;

    // Index tracking the next slot for read.
    size_t currentSlotIndex_;

    // Synchronization for slot management.
    std::condition_variable dataCV_;
    mutable std::mutex slotManagementMutex_;

    // Members for multithreaded copying.
    std::shared_ptr<ThreadPool> threadPool_;
    std::shared_ptr<TaskSet_CopyMemory> tasksMemCopy_;
};
