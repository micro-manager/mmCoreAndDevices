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
  - Slots are recycled when all references are released (In non-overwriting mode)

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
#include "TaskSet_CopyMemory.h"
#include <string>


///////////////////////////////////////////////////////////////////////////////
// DataBuffer Implementation
///////////////////////////////////////////////////////////////////////////////

namespace {
    inline size_t AlignTo(size_t value, size_t alignment) {
        return ((value + alignment - 1) / alignment) * alignment;
    }
}

DataBuffer::DataBuffer(unsigned int memorySizeMB)
    : buffer_(nullptr),
      bufferSize_(0),
      overwriteWhenFull_(false),
      nextAllocOffset_(0),
      currentSlotIndex_(0),
      overflow_(false),
      threadPool_(std::make_shared<ThreadPool>()),
      tasksMemCopy_(std::make_shared<TaskSet_CopyMemory>(threadPool_))
{
    // Pre-allocate slots (1 per MB)
    for (unsigned int i = 0; i < memorySizeMB; i++) {
        unusedSlots_.push_back(std::make_unique<BufferSlot>(0, 0, 0, 0));
    }
    
    AllocateBuffer(memorySizeMB);
}

DataBuffer::~DataBuffer() {
    if (buffer_) {
        #ifdef _WIN32
            VirtualFree(buffer_, 0, MEM_RELEASE);
        #else
            munmap(buffer_, bufferSize_);
        #endif
        buffer_ = nullptr;
    }
}

/**
 * Allocate a character buffer
 * @param memorySizeMB The size (in MB) of the buffer to allocate.
 * @return Error code (0 on success).
 */
int DataBuffer::AllocateBuffer(unsigned int memorySizeMB) {
    // Convert MB to bytes (1 MB = 1048576 bytes)
    size_t numBytes = static_cast<size_t>(memorySizeMB) * (1ULL << 20);
    
    #ifdef _WIN32
        buffer_ = (unsigned char*)VirtualAlloc(nullptr, numBytes,
                                             MEM_RESERVE | MEM_COMMIT,
                                             PAGE_READWRITE);
        if (!buffer_) {
            return DEVICE_ERR;
        }
    #else
        buffer_ = (unsigned char*)mmap(nullptr, numBytes,
                                     PROT_READ | PROT_WRITE,
                                     MAP_PRIVATE | MAP_ANONYMOUS,
                                     -1, 0);
        if (buffer_ == MAP_FAILED) {
            buffer_ = nullptr;
            return DEVICE_ERR;
        }
    #endif
    
    bufferSize_ = numBytes;
    overflow_ = false;
    freeRegions_.clear();
    freeRegions_[0] = bufferSize_;
    return DEVICE_OK;
}

/**
 * Release the buffer.
 * @return Error code (0 on success, error if buffer not found or already released).
 */
int DataBuffer::ReleaseBuffer() {
    if (buffer_ != nullptr) {
        #ifdef _WIN32
            VirtualFree(buffer_, 0, MEM_RELEASE);
        #else
            munmap(buffer_, bufferSize_);
        #endif
        buffer_ = nullptr;
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
    size_t totalSlotSize = dataSize + metaSize;
    unsigned char* imageDataPointer = nullptr;
    unsigned char* metadataPointer = nullptr;
    
    // Pass the actual image and metadata sizes to AcquireWriteSlot.
    int result = AcquireWriteSlot(dataSize, metaSize, &imageDataPointer, &metadataPointer);
    if (result != DEVICE_OK)
        return result;

    // Copy the image data.
    tasksMemCopy_->MemCopy((void*)imageDataPointer, data, dataSize);

    // Copy the metadata.
    if (metaSize > 0) {
        std::memcpy(metadataPointer, metaStr.data(), metaSize);
    }
    
    // Finalize the write slot.
    return FinalizeWriteSlot(imageDataPointer, metaSize);
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
int DataBuffer::AcquireWriteSlot(size_t imageSize, size_t metadataSize,
                                 unsigned char** imageDataPointer, unsigned char** metadataPointer)
{
    std::lock_guard<std::mutex> lock(slotManagementMutex_);
    if (buffer_ == nullptr) {
        return DEVICE_ERR;
    }


    size_t alignment = alignof(std::max_align_t);
    // Total size is the image data plus metadata.
    size_t rawTotalSize = imageSize + metadataSize;
    size_t totalSlotSize = AlignTo(rawTotalSize, alignment);
    size_t candidateStart = 0;

    if (!overwriteWhenFull_) {
        // FIRST: Look for a fit in recycled slots (releasedSlots_)
        for (int i = static_cast<int>(releasedSlots_.size()) - 1; i >= 0; i--) {
            // Align the candidate start position.
            candidateStart = AlignTo(releasedSlots_[i], alignment);
            
            // Find the free region that contains this position.
            auto it = freeRegions_.upper_bound(candidateStart);
            if (it != freeRegions_.begin()) {
                --it;
                size_t freeStart = it->first;
                size_t freeEnd = freeStart + it->second;
                
                // Check if the position is actually within this free region.
                if (candidateStart >= freeStart && candidateStart < freeEnd && 
                    candidateStart + totalSlotSize <= freeEnd) {
                    releasedSlots_.erase(releasedSlots_.begin() + i);
                    return CreateSlot(candidateStart, totalSlotSize, imageSize, metadataSize,
                                                     imageDataPointer, metadataPointer, true);
                }
            }
            
            // If we get here, this released slot position isn't in a free region,
            // so remove it as it's no longer valid.
            releasedSlots_.erase(releasedSlots_.begin() + i);
        }

        // SECOND: Look in the free-region list as fallback.
        for (auto it = freeRegions_.begin(); it != freeRegions_.end(); ++it) {
            // Align the free region start.
            size_t alignedCandidate = AlignTo(it->first, alignment);
            // Check if the free region has enough space after alignment.
            if (it->first + it->second >= alignedCandidate + totalSlotSize) {
                candidateStart = alignedCandidate;
                return CreateSlot(candidateStart, totalSlotSize, imageSize, metadataSize,
                                                 imageDataPointer, metadataPointer, true);
            }
        }
        // No recycled slot or free region can satisfy the allocation.
        overflow_ = true;
        *imageDataPointer = nullptr;
        *metadataPointer = nullptr;
        return DEVICE_ERR;
    } else {
        // Overwrite mode: use nextAllocOffset_. Ensure it is aligned.
        candidateStart = AlignTo(nextAllocOffset_, alignment);
        if (candidateStart + totalSlotSize > bufferSize_) {
            candidateStart = 0;  // Wrap around.
        }
        nextAllocOffset_ = candidateStart + totalSlotSize;
        // Register a new slot using the selected candidateStart.
        return CreateSlot(candidateStart, totalSlotSize, imageSize, metadataSize,
                                         imageDataPointer, metadataPointer, false);
    }
}

/**
 * @brief Release a data slot after writing is complete.
 * 
 * @param caller The device calling this function.
 * @param buffer The buffer to be released.
 * @return Error code (0 on success).
 */
int DataBuffer::FinalizeWriteSlot(unsigned char* imageDataPointer, size_t actualMetadataBytes) {
    if (imageDataPointer == nullptr)
        return DEVICE_ERR;

    // Use a unique_lock so that we can pass it to FindSlotForPointer
    std::unique_lock<std::mutex> lock(slotManagementMutex_);
    BufferSlot* slot = const_cast<BufferSlot*>(FindSlotForPointer(imageDataPointer));
    if (!slot)
        return DEVICE_ERR; // Slot not found

    slot->ReleaseWriteAccess();

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
int DataBuffer::ReleaseDataReadPointer(const unsigned char* imageDataPointer) {
    if (imageDataPointer == nullptr)
        return DEVICE_ERR;

    std::unique_lock<std::mutex> lock(slotManagementMutex_);

    // The slot starts directly at imageDataPointer.
    size_t offset = imageDataPointer - buffer_;

    auto it = activeSlotsByStart_.find(offset);
    if (it == activeSlotsByStart_.end())
        return DEVICE_ERR;  // Slot not found

    BufferSlot* slot = it->second;
    // Release the read access (this does NOT remove the slot from the active list)
    slot->ReleaseReadAccess();

    if (!overwriteWhenFull_) {
        // Now check if the slot is not being accessed.
        // (i.e. this was the last reader and no writer holds it)
        if (slot->IsAvailableForWriting() && slot->IsAvailableForReading()) {
            DeleteSlot(offset, it);
        } else {
            throw std::runtime_error("TESTING: this should not happen when copying data");
        }
    }

    return DEVICE_OK;
}

const unsigned char* DataBuffer::PopNextDataReadPointer(Metadata &md, size_t *imageDataSize, bool waitForData)
{
    BufferSlot* slot = nullptr;
    {
        std::unique_lock<std::mutex> lock(slotManagementMutex_);
        while (activeSlotsVector_.empty()) {
            if (!waitForData)
                return nullptr;
            dataCV_.wait(lock);
        }
        // Atomically take the slot and advance the index.
        slot = activeSlotsVector_[currentSlotIndex_].get();
        currentSlotIndex_ = (currentSlotIndex_ + 1) % activeSlotsVector_.size();
    } // Release lock

    // Now get read access outside the slot of the global mutex.
    slot->AcquireReadAccess();

    // Process the slot.
    size_t slotStart = slot->GetStart();
    const unsigned char* imageDataPointer = buffer_ + slotStart;
    *imageDataSize = slot->GetImageSize();

    // Use the dedicated metadata extraction function.
    this->ExtractMetadata(imageDataPointer, md);

    return imageDataPointer;
}

int DataBuffer::PeekNextDataReadPointer(const unsigned char** imageDataPointer, size_t* imageDataSize,
                                          Metadata &md) {
    // Immediately check if there is an unread slot without waiting.
    BufferSlot* currentSlot = nullptr;
    {
        // Lock the global slot management mutex to safely access the active slots.
        std::unique_lock<std::mutex> lock(slotManagementMutex_);
        if (activeSlotsVector_.empty() || currentSlotIndex_ >= activeSlotsVector_.size()) {
            return DEVICE_ERR; // No unread data available.
        }
        currentSlot = activeSlotsVector_[currentSlotIndex_].get();
    } // Release slotManagementMutex_ before acquiring read access!

    // Now safely get read access without holding the global lock.
    currentSlot->AcquireReadAccess();

    std::unique_lock<std::mutex> lock(slotManagementMutex_);
    *imageDataPointer = buffer_ + currentSlot->GetStart();
    size_t imgSize = currentSlot->GetImageSize();
    *imageDataSize = imgSize;
    
    // Use the dedicated metadata extraction function.
    this->ExtractMetadata(*imageDataPointer, md);
    
    return DEVICE_OK;
}

const unsigned char* DataBuffer::PeekDataReadPointerAtIndex(size_t n, size_t* imageDataSize, Metadata &md) {
    BufferSlot* currentSlot = nullptr;
    {
        // Lock the global slot management mutex to safely access the active slots.
        std::unique_lock<std::mutex> lock(slotManagementMutex_);
        if (activeSlotsVector_.empty() || n >= activeSlotsVector_.size()) {
            return nullptr;
        }
        
        // Instead of looking ahead from currentSlotIndex_, we look back from the end.
        // For n==0, return the most recent slot; for n==1, the one before it; etc.
        size_t index = activeSlotsVector_.size() - n - 1;
        currentSlot = activeSlotsVector_[index].get();
    }
    
    currentSlot->AcquireReadAccess();
    
    const unsigned char* imageDataPointer = buffer_ + currentSlot->GetStart();
    size_t imgSize = currentSlot->GetImageSize();
    *imageDataSize = imgSize;
    

    this->ExtractMetadata(imageDataPointer, md);
    
    // Return a pointer to the image data only.
    return imageDataPointer;
}

unsigned int DataBuffer::GetMemorySizeMB() const {
    // Convert bytes to MB (1 MB = 1048576 bytes)
    return static_cast<unsigned int>(bufferSize_ >> 20);
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
   
    // Check that there are no outstanding readers or writers
    for (const std::unique_ptr<BufferSlot>& slot : activeSlotsVector_) {
        if (!slot->IsAvailableForReading() || !slot->IsAvailableForWriting()) {
            throw std::runtime_error("Cannot reinitialize DataBuffer: outstanding active slot detected.");
        }
    }
   
    // Clear internal data structures
    activeSlotsVector_.clear();
    activeSlotsByStart_.clear();
    releasedSlots_.clear();
    currentSlotIndex_ = 0;
    nextAllocOffset_ = 0;
    overflow_ = false;
   
    // Reset the slot pool
    unusedSlots_.clear();
    
    // Pre-allocate new slots (1 per MB)
    for (unsigned int i = 0; i < memorySizeMB; i++) {
        unusedSlots_.push_back(std::make_unique<BufferSlot>(0, 0, 0, 0));
    }
   
    // Release and reallocate the buffer
    if (buffer_ != nullptr) {
        #ifdef _WIN32
            VirtualFree(buffer_, 0, MEM_RELEASE);
        #else
            munmap(buffer_, bufferSize_);
        #endif
        buffer_ = nullptr;
    }
   
    return AllocateBuffer(memorySizeMB);
}

long DataBuffer::GetActiveSlotCount() const {
    return static_cast<long>(activeSlotsVector_.size());
}

int DataBuffer::ExtractMetadata(const unsigned char* imageDataPtr, Metadata &md) const {
    std::unique_lock<std::mutex> lock(slotManagementMutex_);
    const BufferSlot* slot = FindSlotForPointer(imageDataPtr);
    if (!slot)
        return DEVICE_ERR; // Slot not found or invalid pointer

    // If no metadata is stored in this slot, clear the metadata object.
    if (slot->GetMetadataSize() == 0) {
        md.Clear();
    } else {
        // Metadata is stored immediately after the image data.
        const char* metaDataStart = reinterpret_cast<const char*>(imageDataPtr + slot->GetImageSize());
        // Create a temporary string from the metadata region.
        std::string metaStr(metaDataStart, slot->GetMetadataSize());
        // Restore the metadata from the serialized string.
        md.Restore(metaStr.c_str());
    }
    return DEVICE_OK;
}

// NOTE: Caller must hold slotManagementMutex_ for thread safety.
const BufferSlot* DataBuffer::FindSlotForPointer(const unsigned char* imageDataPtr) const {
    if (buffer_ == nullptr)
        return nullptr;
    std::size_t offset = imageDataPtr - buffer_;
    auto it = activeSlotsByStart_.find(offset);
    return (it != activeSlotsByStart_.end()) ? it->second : nullptr;
}

void DataBuffer::AddToReleasedSlots(size_t offset) {
    if (!overwriteWhenFull_) {
        if (releasedSlots_.size() >= MAX_RELEASED_SLOTS)
            releasedSlots_.erase(releasedSlots_.begin());
        releasedSlots_.push_back(offset);
    }
}

void DataBuffer::MergeFreeRegions(size_t newRegionStart, size_t newRegionEnd, size_t freedRegionSize) {
    // Find the free region that starts at or after newEnd
    auto right = freeRegions_.lower_bound(newRegionEnd);

    // Check if there is a free region immediately preceding the new region
    auto left = freeRegions_.lower_bound(newRegionStart);
    if (left != freeRegions_.begin()) {
        auto prev = std::prev(left);
        // If the previous region's end matches the new region's start...
        if (prev->first + prev->second == newRegionStart) {
            newRegionStart = prev->first;
            freeRegions_.erase(prev);
        }
    }

    // Check if the region immediately to the right can be merged
    if (right != freeRegions_.end() && right->first == newRegionEnd) {
        newRegionEnd = right->first + right->second;
        freeRegions_.erase(right);
    }

    // Insert the merged (or standalone) free region
    size_t newRegionSize = (newRegionEnd > newRegionStart ? newRegionEnd - newRegionStart : 0);
    if (newRegionSize > 0) {
        freeRegions_[newRegionStart] = newRegionSize;
    }
}

void DataBuffer::RemoveFromActiveTracking(size_t offset, std::map<size_t, BufferSlot*>::iterator it) {
    activeSlotsByStart_.erase(it);
    for (auto vecIt = activeSlotsVector_.begin(); vecIt != activeSlotsVector_.end(); ++vecIt) {
        if (vecIt->get()->GetStart() == offset) {
            // Determine the index being removed.
            size_t indexDeleted = std::distance(activeSlotsVector_.begin(), vecIt);
            activeSlotsVector_.erase(vecIt);
            // Adjust the currentSlotIndex_; if the deleted slot was before it, decrement.
            if (currentSlotIndex_ > indexDeleted)
                currentSlotIndex_--;
            break;
        }
    }
}

void DataBuffer::DeleteSlot(size_t offset, std::map<size_t, BufferSlot*>::iterator it) {
    assert(!slotManagementMutex_.try_lock() && "Caller must hold slotManagementMutex_");

    AddToReleasedSlots(offset);

    size_t freedRegionSize = it->second->GetLength();
    size_t newRegionStart = offset;
    size_t newRegionEnd = offset + freedRegionSize;

    // Return the slot to the pool before removing from active tracking
    ReturnSlotToPool(it->second);

    MergeFreeRegions(newRegionStart, newRegionEnd, freedRegionSize);
    RemoveFromActiveTracking(offset, it);
}

BufferSlot* DataBuffer::InitializeNewSlot(size_t candidateStart, size_t totalSlotSize, 
                                        size_t imageDataSize, size_t metadataSize) {
    BufferSlot* newSlot = new BufferSlot(candidateStart, totalSlotSize, imageDataSize, metadataSize);
    newSlot->AcquireWriteAccess();
    activeSlotsVector_.push_back(std::unique_ptr<BufferSlot>(newSlot));
    activeSlotsByStart_[candidateStart] = newSlot;
    return newSlot;
}

void DataBuffer::SetupSlotPointers(BufferSlot* newSlot, unsigned char** imageDataPointer, 
                                 unsigned char** metadataPointer) {
    *imageDataPointer = buffer_ + newSlot->GetStart();
    *metadataPointer = *imageDataPointer + newSlot->GetImageSize();
}

void DataBuffer::UpdateFreeRegions(size_t candidateStart, size_t totalSlotSize) {
    auto it = freeRegions_.upper_bound(candidateStart);
    if (it != freeRegions_.begin()) {
        --it;
        size_t freeRegionStart = it->first;
        size_t freeRegionSize = it->second;
        size_t freeRegionEnd = freeRegionStart + freeRegionSize;

        if (candidateStart >= freeRegionStart && 
            candidateStart + totalSlotSize <= freeRegionEnd) {
            freeRegions_.erase(it);
            
            if (candidateStart > freeRegionStart) {
                size_t gap = candidateStart - freeRegionStart;
                if (gap > 0)
                    freeRegions_.insert({freeRegionStart, gap});
            }
            
            if (freeRegionEnd > candidateStart + totalSlotSize) {
                size_t gap = freeRegionEnd - (candidateStart + totalSlotSize);
                if (gap > 0)
                    freeRegions_.insert({candidateStart + totalSlotSize, gap});
            }
        }
    }
}

int DataBuffer::CreateSlot(size_t candidateStart, size_t totalSlotSize, 
                         size_t imageDataSize, size_t metadataSize,
                         unsigned char** imageDataPointer, unsigned char** metadataPointer,
                         bool fromFreeRegion) {
    assert(!slotManagementMutex_.try_lock() && "Caller must hold slotManagementMutex_");

    BufferSlot* newSlot = GetSlotFromPool(candidateStart, totalSlotSize, imageDataSize, metadataSize);
    newSlot->AcquireWriteAccess();
    activeSlotsVector_.push_back(std::unique_ptr<BufferSlot>(newSlot));
    activeSlotsByStart_[candidateStart] = newSlot;

    SetupSlotPointers(newSlot, imageDataPointer, metadataPointer);

    if (fromFreeRegion) {
        UpdateFreeRegions(candidateStart, totalSlotSize);
    }

    return DEVICE_OK;
}

BufferSlot* DataBuffer::GetSlotFromPool(size_t start, size_t totalLength, 
                                      size_t imageSize, size_t metadataSize) {
    std::lock_guard<std::mutex> lock(slotManagementMutex_);
    
    if (unusedSlots_.empty()) {
        // Optionally grow the pool if needed
        unusedSlots_.push_back(std::make_unique<BufferSlot>(start, totalLength, imageSize, metadataSize));
    }
    
    BufferSlot* slot = unusedSlots_.back().release();  // Transfer ownership
    unusedSlots_.pop_back();
    slot->Reset(start, totalLength, imageSize, metadataSize);
    return slot;
}

void DataBuffer::ReturnSlotToPool(BufferSlot* slot) {
    std::lock_guard<std::mutex> lock(slotManagementMutex_);
    unusedSlots_.push_back(std::unique_ptr<BufferSlot>(slot));
}