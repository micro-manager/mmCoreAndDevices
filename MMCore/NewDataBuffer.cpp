///////////////////////////////////////////////////////////////////////////////
// FILE:          NewDataBuffer.cpp
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


#include "NewDataBuffer.h"
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
    // Get system page size at runtime
    inline size_t GetPageSize() {
        #ifdef _WIN32
            SYSTEM_INFO si;
            GetSystemInfo(&si);
            return si.dwPageSize;
        #else
            return sysconf(_SC_PAGESIZE);
        #endif
    }

    // Cache the page size.
    const size_t PAGE_SIZE = GetPageSize();

    // Inline alignment function using bitwise operations.
    // For a power-of-two alignment, this computes the smallest multiple
    // of 'alignment' that is at least as large as 'value'.
    inline size_t Align(size_t value) {
        // Use PAGE_SIZE if value is large enough; otherwise use the sizeof(max_align_t)
        size_t alignment = (value >= PAGE_SIZE) ? PAGE_SIZE : alignof(std::max_align_t);
        return (value + alignment - 1) & ~(alignment - 1);
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
    ReinitializeBuffer(memorySizeMB, false);
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

    std::lock_guard<std::mutex> lock(slotManagementMutex_);
    for (BufferSlot* bs : slotPool_) {
        delete bs;
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
        
        // Advise the kernel that we will need this memory soon.
        madvise(buffer_, numBytes, MADV_WILLNEED);

    #endif
    
    bufferSize_ = numBytes;
    overflow_ = false;
    freeRegions_.clear();
    freeRegions_[0] = bufferSize_;
    freeRegionCursor_ = 0;
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
    return DEVICE_ERR;
}

/**
 * Pack the data as [BufferSlotRecord][image data][serialized metadata]
 */
int DataBuffer::InsertData(const void* data, size_t dataSize, const Metadata* pMd, const std::string& deviceLabel) {
    
    void* dataPointer = nullptr;
    void* additionalMetadataPointer = nullptr;
    
    // Convert metadata to serialized string if provided
    std::string serializedMetadata;
    if (pMd != nullptr) {
        serializedMetadata = pMd->Serialize();
    }
    // Initial metadata is all metadata because the image and metadata are already complete
    int result = AcquireWriteSlot(dataSize, 0, &dataPointer, &additionalMetadataPointer, 
                                serializedMetadata, deviceLabel);
    if (result != DEVICE_OK) {
        return result;       
    }

    tasksMemCopy_->MemCopy((void*)dataPointer, data, dataSize);
    
    // Finalize the write slot.
    return FinalizeWriteSlot(dataPointer, 0);
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
    if (overwriteWhenFull_ == overwrite) {
        return DEVICE_OK;
    }

    // You can't change modes when code holds pointers into the buffer
    if (GetActiveSlotCount() > 0) {
        return DEVICE_ERR;
    }
    
    overwriteWhenFull_ = overwrite;
    return DEVICE_OK;
}

/**
 * Get whether the buffer should overwrite old data when full.
 * @return True if overwriting is enabled, false otherwise.
 */
bool DataBuffer::GetOverwriteData() const {
    return overwriteWhenFull_;
}

/**
 * Get a pointer to the next available data slot in the buffer for writing.
 * 
 * The caller must release the slot using ReleaseDataSlot after writing is complete.
 */
int DataBuffer::AcquireWriteSlot(size_t dataSize, size_t additionalMetadataSize,
                                void** dataPointer, 
                                void** additionalMetadataPointer,
                                const std::string& serializedInitialMetadata,
                                const std::string& deviceLabel)
{
    if (buffer_ == nullptr) {
        return DEVICE_ERR;
    }

    // Total size includes data, initial metadata, and any additional metadata space
    size_t rawTotalSize = dataSize + serializedInitialMetadata.size() + additionalMetadataSize;
    size_t totalSlotSize = Align(rawTotalSize);
    size_t candidateStart = 0;

    if (!overwriteWhenFull_) {
        std::lock_guard<std::mutex> lock(slotManagementMutex_);
        // Look in the free-region list as fallback using a cached cursor.
        {
            bool found = false;
            size_t newCandidate = 0;
            // Start search from freeRegionCursor_
            auto it = freeRegions_.lower_bound(freeRegionCursor_);
            // Loop over free regions at most once (wrapping around if necessary).
            for (size_t count = 0, sz = freeRegions_.size(); count < sz; count++) {
                if (it == freeRegions_.end())
                    it = freeRegions_.begin();
                size_t alignedCandidate = Align(it->first);
                if (it->first + it->second >= alignedCandidate + totalSlotSize) {
                    newCandidate = alignedCandidate;
                    found = true;
                    break;
                }
                ++it;
            }
            if (found) {
                candidateStart = newCandidate;
                // Update the cursor so that next search can start here.
                freeRegionCursor_ = candidateStart + totalSlotSize;
                return CreateSlot(candidateStart, totalSlotSize, dataSize, additionalMetadataSize,
                                                 dataPointer, additionalMetadataPointer,
                                                 true, serializedInitialMetadata, deviceLabel);
            }
        }

        // No recycled slot or free region can satisfy the allocation.
        overflow_ = true;
        *dataPointer = nullptr;
        *additionalMetadataPointer = nullptr;
        return DEVICE_ERR;
    } else {
        // Overwrite mode 
        size_t prevOffset, newOffset;
        do {
            prevOffset = nextAllocOffset_.load(std::memory_order_relaxed);
            candidateStart = Align(prevOffset);
            if (candidateStart + totalSlotSize > bufferSize_)
                candidateStart = 0;  // Wrap around if needed.
            newOffset = candidateStart + totalSlotSize;
        } while (!nextAllocOffset_.compare_exchange_weak(prevOffset, newOffset));
        
        // Only now grab the lock to register the new slot.
        {
            std::lock_guard<std::mutex> lock(slotManagementMutex_);
            return CreateSlot(candidateStart, totalSlotSize, dataSize, additionalMetadataSize,
                              dataPointer, additionalMetadataPointer, false, serializedInitialMetadata, deviceLabel);
        }
    }
}

/**
 * @brief Release a data slot after writing is complete.
 * 
 * @param caller The device calling this function.
 * @param buffer The buffer to be released.
 * @return Error code (0 on success).
 */
int DataBuffer::FinalizeWriteSlot(const void* dataPointer, size_t actualMetadataBytes) {
    if (dataPointer == nullptr)
        return DEVICE_ERR;

    BufferSlot* slot = nullptr;
    {
        std::lock_guard<std::mutex> lock(slotManagementMutex_);
        slot = FindSlotForPointer(dataPointer);
        if (!slot)
            return DEVICE_ERR;
            
        // Update the slot with actual metadata size
        slot->UpdateAdditionalMetadataSize(actualMetadataBytes);
    }

    slot->ReleaseWriteAccess();

    // Notify waiting threads under a brief lock
    {
        std::lock_guard<std::mutex> lock(slotManagementMutex_);
        dataCV_.notify_all();
    }
    
    return DEVICE_OK;
}

/**
 * ReleaseSlot is called after a slot's content has been fully read.
 *
 * This implementation pushes only the start of the released slot onto the FILO
 * (releasedSlots_) and removes the slot from the active slot map and activeSlots_.
 */
int DataBuffer::ReleaseDataReadPointer(const void* dataPointer) {
    if (dataPointer == nullptr)
        return DEVICE_ERR;

    // First find the slot without the global lock
    BufferSlot* slot = nullptr;
    {
        std::lock_guard<std::mutex> lock(slotManagementMutex_);
        slot = FindSlotForPointer(dataPointer);
        if (!slot)
            return DEVICE_ERR;
    }
    const size_t offset = static_cast<const unsigned char*>(dataPointer) - 
                                static_cast<const unsigned char*>(buffer_);

    // Release the read access outside the global lock
    slot->ReleaseReadAccess();

    if (!overwriteWhenFull_) {
        std::lock_guard<std::mutex> lock(slotManagementMutex_);
        // Now check if the slot is not being accessed
        if (slot->IsFree()) {
            auto it = activeSlotsByStart_.find(offset);
            DeleteSlot(offset, it);
        }
    }

    return DEVICE_OK;
}

const void* DataBuffer::PopNextDataReadPointer(Metadata &md, bool waitForData)
{
    if (overwriteWhenFull_) {
        throw std::runtime_error("PopNextDataReadPointer is not available in overwrite mode");
    }

    BufferSlot* slot = nullptr;
    size_t slotStart = 0;

    // First, get the slot under the global lock
    {
        std::unique_lock<std::mutex> lock(slotManagementMutex_);
        while (activeSlotsVector_.empty()) {
            if (!waitForData)
                return nullptr;
            dataCV_.wait(lock);
        }
        // Atomically take the slot and advance the index
        slot = activeSlotsVector_[currentSlotIndex_];
        slotStart = slot->GetStart();
        currentSlotIndex_ = (currentSlotIndex_ + 1) % activeSlotsVector_.size();
    } // Release global lock

    // Now acquire read access outside the global lock
    slot->AcquireReadAccess();
    
    const unsigned char* dataPointer = static_cast<const unsigned char*>(buffer_) + slotStart;
    this->ExtractMetadata(dataPointer, slot, md);

    return dataPointer;
}

const void* DataBuffer::PeekLastDataReadPointer(Metadata &md) {
    if (!overwriteWhenFull_) {
        throw std::runtime_error("PeekLastDataReadPointer is only available in overwrite mode");
    }

    BufferSlot* currentSlot = nullptr;
    {   
        std::unique_lock<std::mutex> lock(slotManagementMutex_);
        if (activeSlotsVector_.empty()) {
            return nullptr;
        }
        
        // Get the most recent slot (last in vector)
        currentSlot = activeSlotsVector_.back();
    }

    currentSlot->AcquireReadAccess();
    
    const void* result = static_cast<const unsigned char*>(buffer_) + currentSlot->GetStart();
    
    if (ExtractMetadata(result, currentSlot, md) != DEVICE_OK) {
        currentSlot->ReleaseReadAccess();
        return nullptr;
    }
    
    return result;
}

const void* DataBuffer::PeekDataReadPointerAtIndex(size_t n, Metadata &md) {
    if (!overwriteWhenFull_) {
        throw std::runtime_error("PeekDataReadPointerAtIndex is only available in overwrite mode");
    }

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
        currentSlot = activeSlotsVector_[index];
    }
    
    currentSlot->AcquireReadAccess();
    
    const unsigned char* dataPointer = static_cast<const unsigned char*>(buffer_) + currentSlot->GetStart();
    this->ExtractMetadata(dataPointer, currentSlot, md);
    
    return dataPointer;
}

const void* DataBuffer::PeekLastDataReadPointerFromDevice(const std::string& deviceLabel, Metadata& md) {
    if (!overwriteWhenFull_) {
        throw std::runtime_error("PeekLastDataReadPointerFromDevice is only available in overwrite mode");
    }

    BufferSlot* matchingSlot = nullptr;
    {
        std::unique_lock<std::mutex> lock(slotManagementMutex_);
        
        // Search backwards through activeSlotsVector_ to find most recent matching slot
        for (auto it = activeSlotsVector_.rbegin(); it != activeSlotsVector_.rend(); ++it) {
            if ((*it)->GetDeviceLabel() == deviceLabel) {
                matchingSlot = *it;
                break;
            }
        }
        
        if (!matchingSlot) {
            return nullptr;
        }
    }

    // Acquire read access and get data pointer
    matchingSlot->AcquireReadAccess();
    
    const void* result = static_cast<const unsigned char*>(buffer_) + matchingSlot->GetStart();
    
    if (ExtractMetadata(result, matchingSlot, md) != DEVICE_OK) {
        matchingSlot->ReleaseReadAccess();
        return nullptr;
    }
    
    return result;
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
 * @param forceReset If true, the buffer will be reset even if there are outstanding active slots.
 * This is a dangerous operation operation becuase there may be pointers into the buffer's memory
 * that are not valid anymore. It can be used to reset the buffer without having to restart the
 * application, but it indicates a bug in the application or device adapter that is not properly
 * releasing the buffer's memory.
 * @return DEVICE_OK on success.
 * @throws std::runtime_error if any slot is still actively being read or written.
 */
int DataBuffer::ReinitializeBuffer(unsigned int memorySizeMB, bool forceReset) {
    std::lock_guard<std::mutex> lock(slotManagementMutex_);

    // Ensure no active readers/writers exist.
    if (!forceReset) {
        for (BufferSlot* slot : activeSlotsVector_) {
            if (!slot->IsFree()) {
                throw std::runtime_error("Cannot reinitialize DataBuffer: outstanding active slot detected.");
            }
        }
    }
   
    // Clear internal data structures
    activeSlotsVector_.clear();
    activeSlotsByStart_.clear();
    currentSlotIndex_ = 0;
    nextAllocOffset_ = 0;
    overflow_ = false;
   
    // Pre-allocate slots (one per MB) and store in both slotPool_ and unusedSlots_
    slotPool_.clear();
    unusedSlots_.clear();
    for (unsigned int i = 0; i < memorySizeMB; i++) {
        BufferSlot* bs = new BufferSlot();
        slotPool_.push_back(bs);
        unusedSlots_.push_back(bs);
    }
    
   
    // Release and reallocate the buffer
    if (buffer_ != nullptr) {
        ReleaseBuffer(); 
    }
   
    return AllocateBuffer(memorySizeMB);
}

void DataBuffer::Clear() {
    if (NumOutstandingSlots() > 0) {
        throw std::runtime_error("Cannot clear DataBuffer: outstanding active slot detected.");
    }
    std::lock_guard<std::mutex> lock(slotManagementMutex_);
    activeSlotsVector_.clear();
    activeSlotsByStart_.clear();
    currentSlotIndex_ = 0;
    nextAllocOffset_ = 0;
    // reset the unused slot pool
    unusedSlots_.clear();
    for (BufferSlot* bs : slotPool_) {
        unusedSlots_.push_back(bs);
    }
    // Rest freee regions to whole buffer
    freeRegions_.clear();
    freeRegions_[0] = bufferSize_;
    freeRegionCursor_ = 0;
}

long DataBuffer::GetActiveSlotCount() const {
    return static_cast<long>(activeSlotsVector_.size());
}

int DataBuffer::ExtractMetadata(const void* dataPointer, BufferSlot* slot, Metadata &md) {
    // No lock is required here because we assume the slot is already locked

    if (!dataPointer || !slot)
        return DEVICE_ERR; // Invalid pointer

    // Calculate metadata pointers and sizes from the slot
    const unsigned char* initialMetadataPtr = static_cast<const unsigned char*>(dataPointer) + slot->GetDataSize();
    size_t initialMetadataSize = slot->GetInitialMetadataSize();
    const unsigned char* additionalMetadataPtr = initialMetadataPtr + initialMetadataSize;
    size_t additionalMetadataSize = slot->GetAdditionalMetadataSize();

    // Handle initial metadata if present
    if (initialMetadataSize > 0) {
        Metadata initialMd;
        std::string initialMetaStr(reinterpret_cast<const char*>(initialMetadataPtr), initialMetadataSize);
        initialMd.Restore(initialMetaStr.c_str());
        md.Merge(initialMd);
    }

    // Handle additional metadata if present
    if (additionalMetadataSize > 0) {
        Metadata additionalMd;
        std::string additionalMetaStr(reinterpret_cast<const char*>(additionalMetadataPtr), additionalMetadataSize);
        additionalMd.Restore(additionalMetaStr.c_str());
        md.Merge(additionalMd);
    }

    return DEVICE_OK;
}

// NOTE: Caller must hold slotManagementMutex_ for thread safety.
BufferSlot* DataBuffer::FindSlotForPointer(const void* dataPointer) {
    assert(!slotManagementMutex_.try_lock() && "Caller must hold slotManagementMutex_");
    if (buffer_ == nullptr)
        return nullptr;
    std::size_t offset = static_cast<const unsigned char*>(dataPointer) - 
                        static_cast<const unsigned char*>(buffer_);
    auto it = activeSlotsByStart_.find(offset);
    return (it != activeSlotsByStart_.end()) ? it->second : nullptr;
}

void DataBuffer::MergeFreeRegions(size_t newRegionStart, size_t newRegionEnd) {
    assert(!slotManagementMutex_.try_lock() && "Caller must hold slotManagementMutex_");
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

void DataBuffer::RemoveFromActiveTracking(size_t offset, std::unordered_map<size_t, BufferSlot*>::iterator it) {
    assert(!slotManagementMutex_.try_lock() && "Caller must hold slotManagementMutex_");
    activeSlotsByStart_.erase(it);
    for (auto vecIt = activeSlotsVector_.begin(); vecIt != activeSlotsVector_.end(); ++vecIt) {
        if ((*vecIt)->GetStart() == offset) {
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

void DataBuffer::DeleteSlot(size_t offset, std::unordered_map<size_t, BufferSlot*>::iterator it) {
    assert(!slotManagementMutex_.try_lock() && "Caller must hold slotManagementMutex_");

    size_t newRegionStart = offset;
    size_t newRegionEnd = offset + it->second->GetLength();

    // Return the slot to the pool before removing from active tracking
    ReturnSlotToPool(it->second);

    MergeFreeRegions(newRegionStart, newRegionEnd);
    RemoveFromActiveTracking(offset, it);
}

void DataBuffer::UpdateFreeRegions(size_t candidateStart, size_t totalSlotSize) {
    assert(!slotManagementMutex_.try_lock() && "Caller must hold slotManagementMutex_");
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
                          size_t dataSize, size_t additionalMetadataSize,
                          void** dataPointer, void** additionalMetadataPointer,
                          bool fromFreeRegion, const std::string& serializedInitialMetadata,
                          const std::string& deviceLabel)
{
    assert(!slotManagementMutex_.try_lock() && "Caller must hold slotManagementMutex_");
    BufferSlot* newSlot = GetSlotFromPool(candidateStart, totalSlotSize, 
                                         dataSize, serializedInitialMetadata.size(),  
                                         additionalMetadataSize, deviceLabel);    

    newSlot->AcquireWriteAccess();

    // Initialize the data pointer before using it.
    *dataPointer = static_cast<unsigned char*>(buffer_) + newSlot->GetStart();

    if (!serializedInitialMetadata.empty()) {
        std::memcpy(static_cast<unsigned char*>(*dataPointer) + dataSize, serializedInitialMetadata.data(), 
            serializedInitialMetadata.size());
        newSlot->SetInitialMetadataSize(serializedInitialMetadata.size());
    }

    *additionalMetadataPointer = static_cast<unsigned char*>(*dataPointer) + newSlot->GetDataSize() + newSlot->GetInitialMetadataSize();

    if (fromFreeRegion) {
        UpdateFreeRegions(candidateStart, totalSlotSize);
    }

    return DEVICE_OK;
}

BufferSlot* DataBuffer::GetSlotFromPool(size_t start, size_t totalLength, 
                                       size_t dataSize, size_t initialMetadataSize,
                                       size_t additionalMetadataSize,
                                       const std::string& deviceLabel) {
    assert(!slotManagementMutex_.try_lock() && "Caller must hold slotManagementMutex_");
    
    // Grow the pool if needed.
    if (unusedSlots_.empty()) {
        BufferSlot* newSlot = new BufferSlot();
        slotPool_.push_back(newSlot);
        unusedSlots_.push_back(newSlot);
    }
    
    // Get a slot from the front of the deque.
    BufferSlot* slot = unusedSlots_.front();
    unusedSlots_.pop_front();
    slot->Reset(start, totalLength, dataSize, initialMetadataSize, additionalMetadataSize, deviceLabel);
    
    // Add to active tracking.
    activeSlotsVector_.push_back(slot);
    activeSlotsByStart_[start] = slot;
    return slot;
}

void DataBuffer::ReturnSlotToPool(BufferSlot* slot) {
    assert(!slotManagementMutex_.try_lock() && "Caller must hold slotManagementMutex_");
    unusedSlots_.push_back(slot);
}

int DataBuffer::ExtractCorrespondingMetadata(const void* dataPointer, Metadata &md) {
    BufferSlot* slot = nullptr;
    {
        std::lock_guard<std::mutex> lock(slotManagementMutex_);
        slot = FindSlotForPointer(dataPointer);
        if (!slot) {
            return DEVICE_ERR;
        }
    }
    
    // Extract metadata (internal method doesn't need lock)
    return ExtractMetadata(dataPointer, slot, md);
}

size_t DataBuffer::GetDatumSize(const void* dataPointer) {
    std::lock_guard<std::mutex> lock(slotManagementMutex_);
    BufferSlot* slot = FindSlotForPointer(dataPointer);
    if (!slot) {
        throw std::runtime_error("DataBuffer::GetDatumSize: pointer not found in buffer");
    }
    return slot->GetDataSize();
}

bool DataBuffer::IsPointerInBuffer(const void* ptr) {
    if (buffer_ == nullptr || ptr == nullptr) {
        return false;
    }
    // get the mutex
    std::lock_guard<std::mutex> lock(slotManagementMutex_);
    // find the slot
    BufferSlot* slot = FindSlotForPointer(ptr);
    return slot != nullptr;
}

int DataBuffer::NumOutstandingSlots() const {
    std::lock_guard<std::mutex> lock(slotManagementMutex_);
    int numOutstanding = 0;
    for (const BufferSlot* slot : activeSlotsVector_) {
        if (!slot->IsFree()) {
            numOutstanding++;
        }
    }
    return numOutstanding;
}
