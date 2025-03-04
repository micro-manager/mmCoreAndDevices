///////////////////////////////////////////////////////////////////////////////
// FILE:          NewDataBuffer.h
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
#include <unordered_map>
#include <cstddef>
#include <vector>
#include <condition_variable>
#include <deque>
#include <memory>
#include "TaskSet_CopyMemory.h"
#include <cassert>
#include <atomic>

/**
 * BufferSlot represents a contiguous slot in the DataBuffer that holds image
 * data and metadata. It uses RAII-based locking with std::shared_timed_mutex to
 * support exclusive write access and concurrent shared read access.
 */
class BufferSlot {
public:
    /**
     * Constructs a BufferSlot with all sizes specified up front.
     *
     * @param start The starting offset (in bytes) within the buffer.
     * @param totalLength The total length (in bytes) reserved for this slot, typically
     *                    an aligned size (which includes image data and metadata).
     * @param imageSize The exact number of bytes for the image data.
     * @param metadataSize The exact number of bytes for the metadata.
     */
    BufferSlot()
        : start_(0),
          length_(0),
          imageSize_(0),
          initialMetadataSize_(0),
          additionalMetadataSize_(0),
          deviceLabel_(),
          rwMutex_()
    {
    }

    /**
     * Destructor.
     */
    ~BufferSlot() {};

    /**
     * Returns the starting offset (in bytes) of the slot.
     * @return The slot's start offset.
     */
    std::size_t GetStart() const { return start_; }

    /**
     * Returns the length (in bytes) of the slot.
     *
     * @return The slot's length.
     */
    std::size_t GetLength() const { return length_; }

    /**
     * Acquires exclusive (write) access (blocking).
     */
    void AcquireWriteAccess() {
        rwMutex_.lock();
    }

    /**
     * Releases exclusive write access.
     */
    void ReleaseWriteAccess() {
        rwMutex_.unlock();
    }

    /**
     * Acquires shared read access (blocking).
     */
    void AcquireReadAccess() {
        rwMutex_.lock_shared();
    }

    /**
     * Releases shared read access.
     */
    void ReleaseReadAccess() {
        rwMutex_.unlock_shared();
    }

    /**
     * Checks if the slot is completely free (no readers, no writers).
     * We do this by attempting to lock it exclusively:
     * if we succeed, there are no concurrent locks.
     */
    bool IsFree() const {
        std::unique_lock<std::shared_timed_mutex> lk(rwMutex_, std::try_to_lock);
        return lk.owns_lock();
    }

    /**
     * Resets the slot with new parameters.  
     * The assertion uses IsFree() as a best-effort check before reinitializing.
     */
    void Reset(size_t start,
               size_t length,
               size_t imageSize,
               size_t initialMetadataSize,
               size_t additionalMetadataSize,
               const std::string& deviceLabel)
    {
        // If this fails, there's likely an active read or write lock on the slot.
        assert(IsFree() &&
               "BufferSlot mutex still locked during Reset - indicates a bug!");

        start_ = start;
        length_ = length;
        imageSize_ = imageSize;
        initialMetadataSize_ = initialMetadataSize;
        additionalMetadataSize_ = initialMetadataSize + additionalMetadataSize;
        deviceLabel_ = deviceLabel;
    }

    /**
     * Updates the metadata size after writing is complete.
     * @param newSize The actual size of the written metadata.
     */
    void UpdateAdditionalMetadataSize(size_t newSize) {
        additionalMetadataSize_ = newSize;
    }

    /**
     * Record the number of bytes of the initial metadata that have been written to this slot.
     */
    void SetInitialMetadataSize(size_t initialSize) {
        initialMetadataSize_ = initialSize;
    }

    /**
     * Returns the size of the image data in bytes.
     * @return The image data size.
     */
    std::size_t GetDataSize() const {
        return imageSize_;
    }

    /**
     * Returns the size of the initial metadata in bytes.
     * @return The initial metadata size.
     */
    std::size_t GetInitialMetadataSize() const {
        return initialMetadataSize_;
    }

    /**
     * Returns the size of the additional metadata in bytes.
     * @return The additional metadata size.
     */
    std::size_t GetAdditionalMetadataSize() const {
        return additionalMetadataSize_;
    }

    const std::string& GetDeviceLabel() const {
        return deviceLabel_;
    }

private:
    std::size_t start_;
    std::size_t length_;
    size_t imageSize_;
    size_t initialMetadataSize_;
    size_t additionalMetadataSize_;
    std::string deviceLabel_;

    mutable std::shared_timed_mutex rwMutex_;
};

/**
 * DataBuffer manages a contiguous block of memory divided into BufferSlot objects
 * for storing image data and metadata. Each slot in memory holds
 * only the image data (followed immediately by metadata), while header information
 * is maintained in the BufferSlot objects.
 */
class DataBuffer {
public:

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
     * Inserts image data along with metadata into the buffer.
     * @param data Pointer to the raw image data.
     * @param dataSize The image data size in bytes.
     * @param pMd Pointer to the metadata (can be null if not applicable).
     * @param deviceLabel The label of the device that is the source of the data.
     * @return DEVICE_OK on success.
     */
    int InsertData(const void* data, size_t dataSize, const Metadata* pMd, const std::string& deviceLabel);

    /**
     * Sets whether the buffer should overwrite old data when full.
     * @param overwrite True to enable overwriting.
     * @return DEVICE_OK on success.
     */
    int SetOverwriteData(bool overwrite);

    /**
     * Returns whether the buffer should overwrite old data when full.
     * @return True if overwriting is enabled, false otherwise.
     */
    bool GetOverwriteData() const;

    /**
     * Acquires a write slot large enough to hold the image data and metadata.
     * On success, returns pointers for the image data and metadata regions.
     *
     * @param dataSize The number of bytes reserved for data.
     * @param additionalMetadataSize The maximum number of bytes reserved for additional metadata.
     * @param dataPointer On success, receives a pointer to the data region.
     * @param additionalMetadataPointer On success, receives a pointer to the additional metadata region.
     * @param serializedInitialMetadata Optional string containing initial metadata to write.
     * @return DEVICE_OK on success.
     */
    int AcquireWriteSlot(size_t dataSize, size_t additionalMetadataSize,
                         void** dataPointer, 
                         void** additionalMetadataPointer,
                         const std::string& serializedInitialMetadata,
                         const std::string& deviceLabel);

    /**
     * Finalizes (releases) a write slot after data has been written.
     * Requires the actual number of metadata bytes written.
     *
     * @param imageDataPointer Pointer previously obtained from AcquireWriteSlot.
     * @param actualMetadataBytes The actual number of metadata bytes written.
     * @return DEVICE_OK on success.
     */
    int FinalizeWriteSlot(const void* dataPointer, size_t actualMetadataBytes);

    /**
     * Releases read access for the image data after reading.
     * @param imageDataPointer Pointer previously obtained from reading routines.
     * @return DEVICE_OK on success.
     */
    int ReleaseDataReadPointer(const void* dataPointer);

    /**
     * Retrieves and consumes the next available data entry for reading,
     * populating the provided Metadata object.
     * @param md Metadata object to populate.
     * @param waitForData If true, blocks until data is available.
     * @return Pointer to the image data region, or nullptr if none available.
     */
    const void* PopNextDataReadPointer(Metadata &md, bool waitForData);

    /**
     * Peeks at the most recently added data entry.
     * @param md Metadata object populated from the stored metadata.
     * @return Pointer to the start of the data region.
     */
    const void* PeekLastDataReadPointer(Metadata &md);

    /**
     * Peeks at the nth unread data entry without consuming it.
     * (n = 0 is equivalent to PeekNextDataReadPointer).
     * @param n Index of the data entry to peek at (0 for next available).
     * @param md Metadata object populated from the stored metadata.
     * @return Pointer to the start of the data region.
     */
    const void* PeekDataReadPointerAtIndex(size_t n, Metadata &md);

    /**
     * Get the last image inserted by a specific device.
     * @param deviceLabel The label of the device to get the image from.
     * @param md Metadata object to populate.
     * @return Pointer to the image data, or nullptr if not found.
     */
    const void* PeekLastDataReadPointerFromDevice(const std::string& deviceLabel, Metadata& md);

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
     * Extracts metadata for a given image data pointer.
     * Thread-safe method that acquires necessary locks to lookup metadata location.
     *
     * @param dataPtr Pointer to the (usuallyimage data.
     * @param md Metadata object to populate.
     * @return DEVICE_OK on success, or an error code if extraction fails.
     */
    int ExtractCorrespondingMetadata(const void* dataPtr, Metadata &md);

    /**
     * Returns the size of the data portion of the slot in bytes.
     * @param dataPointer Pointer to the data portion of a slot.
     * @return Size in bytes of the data portion, or 0 if pointer is invalid.
     */
    size_t GetDatumSize(const void* dataPointer);

    /**
     * Check if a pointer is within the buffer's memory range.
     * @param ptr The pointer to check.
     * @return true if the pointer is within the buffer, false otherwise.
     */
    bool IsPointerInBuffer(const void* ptr);

    /**
     * Checks if there are any outstanding slots in the buffer. If so, it 
     * is unsafe to destroy the buffer.
     * @return true if there are outstanding slots, false otherwise.
     */
    int NumOutstandingSlots() const;

    /**
     * Reinitializes the DataBuffer, clearing its structures and allocating a new buffer.
     * @param memorySizeMB New buffer size (in MB).
     * @param forceReset If true, the buffer will be reset even if there are outstanding active slots.
     * This is a dangerous operation operation becuase there may be pointers into the buffer's memory
     * that are not valid anymore. It can be used to reset the buffer without having to restart the
     * application, but it indicates a bug in the application or device adapter that is not properly
     * releasing the buffer's memory.
     * @return DEVICE_OK on success.
     * @throws std::runtime_error if any slot is still in use.
     */
    int ReinitializeBuffer(unsigned int memorySizeMB, bool forceReset);

    /**
     * Clears the buffer, discarding all data that does not have outstanding pointers.
     */
    void Clear();

    

private:

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
     * Internal helper function that finds the slot for a given pointer.
     * Returns non-const pointer since slots need to be modified for locking.
     *
     * @param dataPtr Pointer to the data.
     * @return Pointer to the corresponding BufferSlot, or nullptr if not found.
     */
    BufferSlot* FindSlotForPointer(const void* dataPtr);

    // Memory managed by the DataBuffer.
    void* buffer_;
    size_t bufferSize_;

    // Whether to overwrite old data when full.
    bool overwriteWhenFull_;

    // Overflow flag (set if insert fails due to full buffer).
    bool overflow_;

    // Active slots and their mapping.
    std::vector<BufferSlot*> activeSlotsVector_;
    std::unordered_map<size_t, BufferSlot*> activeSlotsByStart_;

    // Free region list for non-overwrite mode.
    // Map from starting offset -> region size (in bytes).
    std::map<size_t, size_t> freeRegions_;

    // Cached cursor for scanning free regions in non-overwrite mode.
    size_t freeRegionCursor_;

    // Instead of ownership via unique_ptr, store raw pointers
    std::deque<BufferSlot*> unusedSlots_;

    // This container holds the ownership; they live for the lifetime of the buffer.
    std::vector<BufferSlot*> slotPool_;

    // Next free offset within the buffer.
    // In overwrite mode, new allocations will come from this pointer.
    std::atomic<size_t> nextAllocOffset_;

    // Index tracking the next slot for read.
    size_t currentSlotIndex_;

    // Synchronization for slot management.
    std::condition_variable dataCV_;
    mutable std::mutex slotManagementMutex_;

    // Members for multithreaded copying.
    std::shared_ptr<ThreadPool> threadPool_;
    std::shared_ptr<TaskSet_CopyMemory> tasksMemCopy_;

    void DeleteSlot(size_t offset, std::unordered_map<size_t, BufferSlot*>::iterator it);

    void MergeFreeRegions(size_t newRegionStart, size_t newRegionEnd);
    void RemoveFromActiveTracking(size_t offset, std::unordered_map<size_t, BufferSlot*>::iterator it);

       void UpdateFreeRegions(size_t candidateStart, size_t totalSlotSize);

    BufferSlot* GetSlotFromPool(size_t start, size_t totalLength, 
                               size_t dataSize, size_t initialMetadataSize,
                               size_t additionalMetadataSize, const std::string& deviceLabel);


    /**
     * Creates a new slot with the specified parameters.
     * Caller must hold slotManagementMutex_.
     */
    int CreateSlot(size_t candidateStart, size_t totalSlotSize, 
                   size_t dataSize, size_t additionalMetadataSize,
                   void** dataPointer, 
                   void** subsequentMetadataPointer,
                   bool fromFreeRegion, 
                   const std::string& serializedInitialMetadata,
                   const std::string& deviceLabel);


    void ReturnSlotToPool(BufferSlot* slot);

    int ExtractMetadata(const void* dataPointer, 
                       BufferSlot* slot,
                       Metadata &md);

};
