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
  - Writing is complete (via Insert or GetDataWriteSlot+Release)
  - All readers have released their references

Metadata Handling:
- Devices must specify PixelType when adding data
- Device-specific metadata requirements (e.g. image dimensions) are handled at the
  device API level rather than in the buffer API to maintain clean separation
*/


#include "Buffer_v2.h"
#include <cstring>
#include <thread>   // for std::this_thread::yield if needed
#include <deque>
#include <algorithm>
#include <cassert>
#include <map>
#include <vector>
#include <memory>

// New internal header that precedes every slot's data.
struct BufferSlotRecord {
    size_t imageSize;
    size_t metadataSize;
};

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
      writeAtomicBool_(true)   // The slot is created with write access held by default.
{
    // No readers are active and the slot starts with write access.
}

BufferSlot::~BufferSlot() {
    // No explicit cleanup required here.
}

/**
 * Returns the start offset (in bytes) of the slot from the start of the buffer.
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
bool BufferSlot::ReleaseReadAccess() {
    // fetch_sub returns the previous value. If that value was 1,
    // then this call decrements the active reader count to zero.
    int prevCount = readAccessCountAtomicInt_.fetch_sub(1, std::memory_order_release);
    return (prevCount == 1);
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



///////////////////////////////////////////////////////////////////////////////
// DataBuffer Implementation
///////////////////////////////////////////////////////////////////////////////

DataBuffer::DataBuffer(unsigned int memorySizeMB)
    : buffer_(nullptr),
      bufferSize_(0),
      overwriteWhenFull_(false),
      nextAllocOffset_(0),
      currentSlotIndex_(0),
      overflow_(false)
{
    AllocateBuffer(memorySizeMB);
}

DataBuffer::~DataBuffer() {
    delete[] buffer_;
}

/**
 * Allocate a character buffer
 * @param memorySizeMB The size (in MB) of the buffer to allocate.
 * @return Error code (0 on success).
 */
int DataBuffer::AllocateBuffer(unsigned int memorySizeMB) {
    // Convert MB to bytes (1 MB = 1048576 bytes)
    size_t numBytes = static_cast<size_t>(memorySizeMB) * (1ULL << 20);
    buffer_ = new unsigned char[numBytes];
    bufferSize_ = numBytes;
    overflow_ = false;
    return DEVICE_OK;
}

/**
 * Release the buffer.
 * @return Error code (0 on success, error if buffer not found or already released).
 */
int DataBuffer::ReleaseBuffer() {
    if (buffer_ != nullptr) {
        delete[] buffer_;
        buffer_ = nullptr;
        bufferSize_ = 0;
        return DEVICE_OK;
    }
    // TODO: Handle errors if other parts of the system still hold pointers.
    return DEVICE_ERR;
}

/**
 * Pack the data as [BufferSlotRecord][image data][serialized metadata]
 */
int DataBuffer::InsertData(const unsigned char* data, size_t dataSize, const Metadata* pMd) {
    size_t metaSize = 0;
    std::string metaStr;
    if (pMd) {
        metaStr = pMd->Serialize();
        metaSize = metaStr.size();
    }
    // Total size is header + image data + metadata
    size_t totalSize = sizeof(BufferSlotRecord) + dataSize + metaSize;
    unsigned char* imageDataPointer = nullptr;
    // TOFO: handle metadata pointer
    int result = GetDataWriteSlot(totalSize, metaSize, &imageDataPointer, nullptr);
    if (result != DEVICE_OK)
        return result;

    // The externally returned imageDataPointer points to the image data.
    // Write out the header by subtracting the header size.
    BufferSlotRecord* headerPointer = reinterpret_cast<BufferSlotRecord*>(imageDataPointer - sizeof(BufferSlotRecord));
    headerPointer->imageSize = dataSize;
    headerPointer->metadataSize = metaSize;

    // Copy the image data into the allocated slot (imageDataPointer is already at the image data).
    std::memcpy(imageDataPointer, data, dataSize);

    // If metadata is available, copy it right after the image data.
    if (metaSize > 0) {
        unsigned char* metaPtr = imageDataPointer + dataSize;
        std::memcpy(metaPtr, metaStr.data(), metaSize);
    }

    // Release the write slot
    return ReleaseDataWriteSlot(&imageDataPointer, metaSize > 0 ? static_cast<int>(metaSize) : -1);
}

/**
 * Reads the header from the slot, then copies the image data into the destination and
 * uses the metadata blob (if any) to populate 'md'.
 */
int DataBuffer::CopyNextDataAndMetadata(unsigned char* dataDestination, size_t* imageDataSize, Metadata &md, bool waitForData) {
    const unsigned char* imageDataPointer = PopNextDataReadPointer(md, imageDataSize, waitForData);
    if (imageDataPointer == nullptr)
        return DEVICE_ERR;

    const BufferSlotRecord* headerPointer = reinterpret_cast<const BufferSlotRecord*>(imageDataPointer - sizeof(BufferSlotRecord));
    *imageDataSize = headerPointer->imageSize;
    // imageDataPointer already points to the image data.
    std::memcpy(dataDestination, imageDataPointer, headerPointer->imageSize);

    // Extract the metadata (if any) following the image data.
    std::string metaStr;
    if (headerPointer->metadataSize > 0) {
        const char* metaDataStart = reinterpret_cast<const char*>(imageDataPointer + headerPointer->imageSize);
        metaStr.assign(metaDataStart, headerPointer->metadataSize);
    } 
    // Restore the metadata
    // This is analogous to what is done in FrameBuffer.cpp:
    md.Restore(metaStr.c_str());

    return ReleaseDataReadPointer(&imageDataPointer);
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
 * Get a pointer to the next available data slot in the buffer for writing.
 * 
 * The caller must release the slot using ReleaseDataSlot after writing is complete.
 */
int DataBuffer::GetDataWriteSlot(size_t imageSize, size_t metadataSize, unsigned char** imageDataPointer, unsigned char** metadataPointer) {
       // AllocateNextSlot allocates a slot for writing new data of variable size.
    //
    // First, it checks if there is a recently released slot start (from releasedSlots_).
    // For each candidate, it uses the activeSlotMap_ mechanism to
    // verify that the candidate start yields a gap large enough to allocate slotSize bytes.
    // This is done so to prefer recently released slots, in order to get performance
    // boosts from reusing recently freed memory.
    //
    // If no released slot fits, then it falls back to using nextAllocOffset_ and similar
    // collision checks. In overwrite mode, wrap-around is supported.
    // Lock to ensure exclusive allocation.
    std::lock_guard<std::mutex> lock(slotManagementMutex_);

    // Total slot size is the header plus the image and metadata lengths.
    size_t totalSlotSize = sizeof(BufferSlotRecord) + imageSize + metadataSize;

    // First, try using a released slot candidate (FILO order)
    for (int i = static_cast<int>(releasedSlots_.size()) - 1; i >= 0; i--) {
        size_t candidateStart = releasedSlots_[i];
        size_t localCandidate = candidateStart;

        // Find first slot at or after our candidate position
        auto nextIt = activeSlotsByStart_.lower_bound(localCandidate);

        // If a previous slot exists, adjust to avoid overlap.
        if (nextIt != activeSlotsByStart_.begin()) {
            size_t prevSlotEnd = std::prev(nextIt)->first + std::prev(nextIt)->second->GetLength();
            // If our candidate region [candidateStart, candidateStart+slotSize) overlaps the previous slot,
            // bump candidateStart to the end of the conflicting slot and try again.
            if (prevSlotEnd > localCandidate) {
                localCandidate = prevSlotEnd;
            }
        }
        
        // Check if there's space before the next active slot
        nextIt = activeSlotsByStart_.lower_bound(localCandidate);
        bool candidateValid = true;
        if (nextIt != activeSlotsByStart_.end()) {
            // Case 1: There is a next slot
            // Check if our proposed slot would overlap with the next slot
            candidateValid = (localCandidate + totalSlotSize <= nextIt->first);
        } else if (localCandidate + totalSlotSize > bufferSize_) {
             // Case 2: No next slot, but we'd exceed buffer size
            if (!overwriteWhenFull_) {
                candidateValid = false;
            } else {
                // Try wrapping around to start of buffer
                localCandidate = 0;
                nextIt = activeSlotsByStart_.lower_bound(localCandidate);
                
                // If there are any slots, ensure we don't overlap with the last one
                if (nextIt != activeSlotsByStart_.begin()) {
                    auto prevIt = std::prev(nextIt);
                    size_t prevSlotEnd = prevIt->first + prevIt->second->GetLength();
                    if (prevSlotEnd > localCandidate) {
                        localCandidate = prevSlotEnd;
                    }
                }
                
                // Check if wrapped position would overlap with first slot
                if (nextIt != activeSlotsByStart_.end()) {
                    candidateValid = (localCandidate + totalSlotSize <= nextIt->first);
                }
            }
        }

        if (candidateValid) {
            // Remove the candidate from releasedSlots_ (it was taken from the "back" if available).
            releasedSlots_.erase(releasedSlots_.begin() + i);
            activeSlotsVector_.push_back(std::make_unique<BufferSlot>(localCandidate, totalSlotSize));
            BufferSlot* slot = activeSlotsVector_.back().get();
            activeSlotsByStart_[localCandidate] = slot;
            *imageDataPointer = buffer_ + slot->GetStart() + sizeof(BufferSlotRecord);
            *metadataPointer = *imageDataPointer + imageSize;
            return DEVICE_OK;
        }
    }

    // If no released candidate fits, fall back to nextAllocOffset_.
    size_t candidateStart = nextAllocOffset_;
    if (candidateStart + totalSlotSize > bufferSize_) {
        // Not enough space in the buffer: if we are not allowed to overwrite then set our overflow flag.
        if (!overwriteWhenFull_) {
            overflow_ = true;
            *imageDataPointer = nullptr;
            *metadataPointer = nullptr;
            return DEVICE_ERR;
        }
        candidateStart = 0;  // Reset to start of buffer
        
        // Since we're starting at position 0, remove any slots that start before our requested size
        auto it = activeSlotsByStart_.begin();
        while (it != activeSlotsByStart_.end() && it->first < totalSlotSize) {
            BufferSlot* slot = it->second;
            if (!slot->IsAvailableForWriting() || !slot->IsAvailableForReading()) {
                throw std::runtime_error("Cannot overwrite slot that is currently being accessed (has active readers or writers)");
            }
            
            // Remove from both tracking structures
            activeSlotsVector_.erase(
                std::remove_if(activeSlotsVector_.begin(), activeSlotsVector_.end(),
                               [targetStart = it->first](const std::unique_ptr<BufferSlot>& slot) { 
                    return slot->GetStart() == targetStart; 
                }),
                activeSlotsVector_.end());
            it = activeSlotsByStart_.erase(it);
        }
    }

    activeSlotsVector_.push_back(std::make_unique<BufferSlot>(candidateStart, totalSlotSize));
    BufferSlot* newSlot = activeSlotsVector_.back().get();
    activeSlotsByStart_[candidateStart] = newSlot;
    nextAllocOffset_ = candidateStart + totalSlotSize;
    if (nextAllocOffset_ >= bufferSize_) {
        nextAllocOffset_ = 0;
    }
    *imageDataPointer = buffer_ + newSlot->GetStart() + sizeof(BufferSlotRecord);
    *metadataPointer = *imageDataPointer + imageSize;
    return DEVICE_OK;
}

/**
 * @brief Release a data slot after writing is complete.
 * 
 * @param caller The device calling this function.
 * @param buffer The buffer to be released.
 * @return Error code (0 on success).
 */
int DataBuffer::ReleaseDataWriteSlot(unsigned char** imageDataPointer, int actualMetadataBytes) {
    if (imageDataPointer == nullptr || *imageDataPointer == nullptr)
        return DEVICE_ERR;

    std::lock_guard<std::mutex> lock(slotManagementMutex_);

    // Convert the externally provided imageDataPointer (which points to the image data)
    // to the true slot start (header) by subtracting sizeof(BufferSlotRecord).
    unsigned char* headerPointer = *imageDataPointer - sizeof(BufferSlotRecord);
    size_t offset = headerPointer - buffer_;

    // Locate the slot using the true header offset.
    auto it = activeSlotsByStart_.find(offset);
    if (it == activeSlotsByStart_.end())
        return DEVICE_ERR; // Slot not found

    // Release the write access
    BufferSlot* slot = it->second;
    slot->ReleaseWriteAccess();

    // If a valid actual metadata byte count is provided (i.e. not -1),
    // update the header->metadataSize to the actual metadata length if it is less.
    if (actualMetadataBytes != -1) {
        BufferSlotRecord* hdr = reinterpret_cast<BufferSlotRecord*>(headerPointer);
        if (static_cast<size_t>(actualMetadataBytes) < hdr->metadataSize) {
            hdr->metadataSize = actualMetadataBytes;
        }
    }

    // Clear the externally provided image data pointer.
    *imageDataPointer = nullptr;

    // Notify any waiting threads that new data is available.
    dataCV_.notify_all();

    return DEVICE_OK;
}


/**
 * ReleaseSlot is called after a slot's content has been fully read.
 *
 * This implementation pushes only the start of the released slot onto the FILO
 * (releasedSlots_) and removes the slot from the active slot map and activeSlots_.
 */
int DataBuffer::ReleaseDataReadPointer(const unsigned char** imageDataPointer) {
    if (imageDataPointer == nullptr || *imageDataPointer == nullptr)
        return DEVICE_ERR;
  
    std::unique_lock<std::mutex> lock(slotManagementMutex_);
  
    // Compute the header pointer by subtracting the header size.
    const unsigned char* headerPointer = *imageDataPointer - sizeof(BufferSlotRecord);
    size_t offset = headerPointer - buffer_;

    // Find the slot in activeSlotsByStart_
    auto it = activeSlotsByStart_.find(offset);
    if (it != activeSlotsByStart_.end()) {
        BufferSlot* slot = it->second;
        // Release the previously acquired read access.
        slot->ReleaseReadAccess();
  
        // Now check if the slot is not being accessed 
        // (i.e. this was the last/readers and no writer holds it)
        if (slot->IsAvailableForWriting() && slot->IsAvailableForReading()) {
            // Ensure we do not exceed the maximum number of released slots.
            if (releasedSlots_.size() >= MAX_RELEASED_SLOTS)
                releasedSlots_.erase(releasedSlots_.begin());
            releasedSlots_.push_back(offset);
    
            // Remove slot from the active tracking structures.
            activeSlotsByStart_.erase(it);
            for (auto vecIt = activeSlotsVector_.begin(); vecIt != activeSlotsVector_.end(); ++vecIt) {
                if (vecIt->get()->GetStart() == offset) {
                    // Determine the index being removed.
                    size_t indexDeleted = std::distance(activeSlotsVector_.begin(), vecIt);
                    activeSlotsVector_.erase(vecIt);
                    // Adjust currentSlotIndex_:
                    // If the deleted slot was before the current index, decrement it.
                    if (currentSlotIndex_ > indexDeleted)
                        currentSlotIndex_--;
                    break;
                }
            }
        }
    } else {
        throw std::runtime_error("Cannot release slot that is not in the buffer.");
    }
    *imageDataPointer = nullptr;
    return DEVICE_OK;
}

const unsigned char* DataBuffer::PopNextDataReadPointer(Metadata &md, size_t *imageDataSize, bool waitForData)
{
    std::unique_lock<std::mutex> lock(slotManagementMutex_);
    
    // Wait until there is data available if requested.
    // (Here we check whether activeSlotsVector_ has an unread slot.
    //  Adjust the condition as appropriate for your implementation.)
    while (activeSlotsVector_.empty()) {
        if (!waitForData)
            return nullptr;
        dataCV_.wait(lock);
    }
    
    // Assume that the next unread slot is at index currentSlotIndex_.
    // (Depending on your data structure you might pop from a deque or update an iterator.)
    BufferSlot* slot = activeSlotsVector_[currentSlotIndex_].get();
    // Get the starting offset for this slot.
    size_t slotStart = slot->GetStart();
    
    // The header is stored at the beginning of the slot.
    const BufferSlotRecord* header = reinterpret_cast<const BufferSlotRecord*>(buffer_ + slotStart);
    
    // The image data region starts right after the header.
    const unsigned char* imageDataPointer = buffer_ + slotStart + sizeof(BufferSlotRecord);
    
    // Set the output image data size from the header.
    *imageDataSize = header->imageSize;
    
    // Populate the metadata.
    if (header->metadataSize > 0) {
        const char* metaDataStart = reinterpret_cast<const char*>(imageDataPointer + header->imageSize);
        md.Restore(metaDataStart);
    } else {
        // If no metadata is available, clear the metadata object.
        md.Clear();
    }
    
    // Consume this slot by advancing the index.
    currentSlotIndex_ = (currentSlotIndex_ + 1) % activeSlotsVector_.size();
    
    // Unlock and return the pointer to the image data region.
    return imageDataPointer;
}

unsigned int DataBuffer::GetMemorySizeMB() const {
    // Convert bytes to MB (1 MB = 1048576 bytes)
    return static_cast<unsigned int>(bufferSize_ >> 20);
}

int DataBuffer::PeekNextDataReadPointer(const unsigned char** imageDataPointer, size_t* imageDataSize,
                                          Metadata &md) {
    // Immediately check if there is an unread slot without waiting.
    std::unique_lock<std::mutex> lock(slotManagementMutex_);
    if (activeSlotsVector_.empty() || currentSlotIndex_ >= activeSlotsVector_.size()) {
        return DEVICE_ERR; // No unread data available.
    }

    // Obtain the next available slot *without* advancing currentSlotIndex_.
    BufferSlot& currentSlot = *activeSlotsVector_[currentSlotIndex_];
    if (!currentSlot.AcquireReadAccess())
        return DEVICE_ERR;

    *imageDataPointer = buffer_ + currentSlot.GetStart() + sizeof(BufferSlotRecord);
    const BufferSlotRecord* headerPointer = reinterpret_cast<const BufferSlotRecord*>( (*imageDataPointer) - sizeof(BufferSlotRecord) );
    *imageDataSize = headerPointer->imageSize;

    // Populate the Metadata object from the stored metadata blob.
    std::string metaStr;
    if (headerPointer->metadataSize > 0) {
        const char* metaDataStart = reinterpret_cast<const char*>(*imageDataPointer + headerPointer->imageSize);
        metaStr.assign(metaDataStart, headerPointer->metadataSize);
    }
    // Restore the metadata
    // This is analogous to what is done in FrameBuffer.cpp:
    md.Restore(metaStr.c_str());

    return DEVICE_OK;
}

const unsigned char* DataBuffer::PeekDataReadPointerAtIndex(size_t n, size_t* imageDataSize, Metadata &md) {
    std::unique_lock<std::mutex> lock(slotManagementMutex_);
    if (activeSlotsVector_.empty() || (currentSlotIndex_ + n) >= activeSlotsVector_.size()) {
        throw std::runtime_error("Not enough unread data available.");
    }
    
    // Access the nth slot (without advancing the read index)
    BufferSlot& slot = *activeSlotsVector_[currentSlotIndex_ + n];
    if (!slot.AcquireReadAccess())
        throw std::runtime_error("Failed to acquire read access for the selected slot.");
    
    // Obtain the pointer to the image data (skip the header)
    const unsigned char* imageDataPointer = buffer_ + slot.GetStart() + sizeof(BufferSlotRecord);
    const BufferSlotRecord* headerPointer = reinterpret_cast<const BufferSlotRecord*>(imageDataPointer - sizeof(BufferSlotRecord));
    
    // Return the image size via the pointer parameter
    if (imageDataSize != nullptr) {
        *imageDataSize = headerPointer->imageSize;
    }
    
    // Retrieve the serialized metadata from the slot.
    std::string metaStr;
    if (headerPointer->metadataSize > 0) {
        const char* metaDataStart = reinterpret_cast<const char*>(imageDataPointer + headerPointer->imageSize);
        metaStr.assign(metaDataStart, headerPointer->metadataSize);
    }
    
    // Restore the metadata
    // This is analogous to what is done in FrameBuffer.cpp:
    //   metadata_.Restore(md.Serialize().c_str());
    md.Restore(metaStr.c_str());
    
    // Return a pointer to the image data only.
    return imageDataPointer;
}

/**
 * Releases the read access that was acquired by a peek.
 * This is similar to ReleaseDataReadPointer except that it does not
 * remove the slot from the active list. This should be used when the
 * overwriteWhenFull_ flag is true and the caller wants to release the
 * peeked slot for reuse.
 */
int DataBuffer::ReleasePeekDataReadPointer(const unsigned char** imageDataPointer) {
    if (imageDataPointer == nullptr || *imageDataPointer == nullptr)
        return DEVICE_ERR;

    std::lock_guard<std::mutex> lock(slotManagementMutex_);
    const unsigned char* headerPointer = *imageDataPointer - sizeof(BufferSlotRecord);
    size_t offset = headerPointer - buffer_;

    // Look up the corresponding slot by its buffer offset.
    auto it = activeSlotsByStart_.find(offset);
    if (it == activeSlotsByStart_.end())
        return DEVICE_ERR;  // Slot not found

    BufferSlot* slot = it->second;
    // Release the read access (this does NOT remove the slot from the active list)
    slot->ReleaseReadAccess();

    *imageDataPointer = nullptr;
    return DEVICE_OK;
}

size_t DataBuffer::GetOccupiedSlotCount() const {
    std::lock_guard<std::mutex> lock(slotManagementMutex_);
    return activeSlotsVector_.size();
}

size_t DataBuffer::GetOccupiedMemory() const {
    std::lock_guard<std::mutex> lock(slotManagementMutex_);
    size_t usedMemory = 0;
    for (const auto& slot : activeSlotsVector_) {
        usedMemory += slot->GetLength();
    }
    return usedMemory;
}

size_t DataBuffer::GetFreeMemory() const {
    std::lock_guard<std::mutex> lock(slotManagementMutex_);
    // Free memory is the total buffer size minus the sum of all occupied memory.
    size_t usedMemory = 0;
    for (const auto& slot : activeSlotsVector_) {
        usedMemory += slot->GetLength();
    }
    return (bufferSize_ > usedMemory) ? (bufferSize_ - usedMemory) : 0;
}

bool DataBuffer::Overflow() const {
    std::lock_guard<std::mutex> lock(slotManagementMutex_);
    return overflow_;
}

/**
 * Reinitialize the DataBuffer by clearing all internal data structures,
 * releasing the current buffer, and reallocating a new one.
 * This method uses the existing slotManagementMutex_ to ensure thread-safety.
 *
 * @param memorySizeMB New size (in MB) for the buffer.
 * @return DEVICE_OK on success.
 * @throws std::runtime_error if any slot is still actively being read or written.
 */
int DataBuffer::ReinitializeBuffer(unsigned int memorySizeMB) {
   std::lock_guard<std::mutex> lock(slotManagementMutex_);

   // Check that there are no outstanding readers or writers.
   for (const std::unique_ptr<BufferSlot>& slot : activeSlotsVector_) {
      if (!slot->IsAvailableForReading() || !slot->IsAvailableForWriting()) {
         throw std::runtime_error("Cannot reinitialize DataBuffer: outstanding active slot detected.");
      }
   }

   // Clear internal data structures.
   activeSlotsVector_.clear();
   activeSlotsByStart_.clear();
   releasedSlots_.clear();
   currentSlotIndex_ = 0;
   nextAllocOffset_ = 0;
   overflow_ = false;

   // Release the old buffer.
   if (buffer_ != nullptr) {
      delete[] buffer_;
      buffer_ = nullptr;
      bufferSize_ = 0;
   }

   // Allocate a new buffer using the provided memory size.
   AllocateBuffer(memorySizeMB);

   return DEVICE_OK;
}

long DataBuffer::GetRemainingImageCount() const {
    return static_cast<long>(activeSlotsVector_.size());
}
