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
    buffer_ = new char[numBytes];
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
 * Copy data into the next available slot in the buffer.
 * 
 * Returns the size of the copied data through dataSize.
 * all required metadata for interpreting its image data is there.
 * Note: this can be implemented in terms of Get/Release slot + memcopy.
 * 
 * @param caller The device calling this function.
 * @param data The data to be copied.
 * @param dataSize Size of the data to copy.
 * @param serializedMetadata The associated metadata.
 * @return Error code (0 on success).
 */
int DataBuffer::InsertData(const void* data, 
                           size_t dataSize, const Metadata* pMd) {
    // Get a write slot of the required size
    void* slotPointer = nullptr;
    int result = GetDataWriteSlot(dataSize, &slotPointer);
    if (result != DEVICE_OK)
        return result;

    // Copy the data into the slot
    std::memcpy(slotPointer, data, dataSize);
    if (pMd) {
        // md.Serialize().c_str()
        // TODO: Need a metadata lock and perhaps a map from buffer offset to metadata?
    }

    // Release the write slot
    return ReleaseDataWriteSlot(&slotPointer);
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
    void* sourcePtr = nullptr;
    int result = PopNextDataReadPointer(&sourcePtr, dataSize, md, waitForData);
    if (result != DEVICE_OK)
        return result;
    
    // Copy the data from the slot into the user's destination buffer
    std::memcpy(dataDestination, sourcePtr, *dataSize);
    
    // Release the read pointer (this will handle cleanup and index management)
    return ReleaseDataReadPointer(&sourcePtr);
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
int DataBuffer::GetDataWriteSlot(size_t slotSize, void** slotPointer) {
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

    // Ensure that only one thread can allocate a slot at a time.
    std::lock_guard<std::mutex> lock(slotManagementMutex_); 

    // First, try using a released slot candidate (FILO order)
    for (int i = static_cast<int>(releasedSlots_.size()) - 1; i >= 0; i--) {
        size_t candidateStart = releasedSlots_[i];
        size_t localCandidate = candidateStart;

        // Find first slot at or after our candidate position
        auto nextIt = activeSlotsByStart_.lower_bound(localCandidate);

        // If there's a previous slot, ensure we don't overlap with it
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
            candidateValid = (localCandidate + slotSize <= nextIt->first);
        } else if (localCandidate + slotSize > bufferSize_) {
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
                    candidateValid = (localCandidate + slotSize <= nextIt->first);
                }
            }
        }

        if (candidateValid) {
            // Remove the candidate from releasedSlots_ (it was taken from the "back" if available).
            releasedSlots_.erase(releasedSlots_.begin() + i);
            activeSlotsVector_.push_back(BufferSlot(localCandidate, slotSize));
            BufferSlot* slot = &activeSlotsVector_.back();
            activeSlotsByStart_[localCandidate] = slot;
            *slotPointer = buffer_ + slot->GetStart();
            return DEVICE_OK;
        }
    }
    // If no released candidate fits, use nextAllocOffset_ fallback.
    size_t candidateStart = nextAllocOffset_;
    if (candidateStart + slotSize > bufferSize_) {
        // Not enough space in the buffer: if we are not allowed to overwrite then set our overflow flag.
        if (!overwriteWhenFull_) {
            overflow_ = true;    // <-- mark that overflow has happened
            *slotPointer = nullptr;
            return DEVICE_ERR;
        }
        candidateStart = 0;  // Reset to start of buffer
        
        // Since we're starting at position 0, remove any slots that start before our requested size
        auto it = activeSlotsByStart_.begin();
        while (it != activeSlotsByStart_.end() && it->first < slotSize) {
            BufferSlot* slot = it->second;
            if (!slot->IsAvailableForWriting() || !slot->IsAvailableForReading()) {
                throw std::runtime_error("Cannot overwrite slot that is currently being accessed (has active readers or writers)");
            }
            
            // Remove from both tracking structures
            activeSlotsVector_.erase(
                std::remove_if(activeSlotsVector_.begin(), activeSlotsVector_.end(),
                    [targetStart = it->first](const BufferSlot& slot) { 
                        return slot.GetStart() == targetStart; 
                    }),
                activeSlotsVector_.end());
            it = activeSlotsByStart_.erase(it);
        }
    }

    // Create and track the new slot
    activeSlotsVector_.push_back(BufferSlot(candidateStart, slotSize));
    BufferSlot* newSlot = &activeSlotsVector_.back();
    activeSlotsByStart_[candidateStart] = newSlot;
    
    // Update nextAllocOffset_ for next allocation
    nextAllocOffset_ = candidateStart + slotSize;
    if (nextAllocOffset_ >= bufferSize_) {
        nextAllocOffset_ = 0;
    }
    
    *slotPointer = buffer_ + newSlot->GetStart();
    return DEVICE_OK;
}

/**
 * @brief Release a data slot after writing is complete.
 * 
 * @param caller The device calling this function.
 * @param buffer The buffer to be released.
 * @return Error code (0 on success).
 */
int DataBuffer::ReleaseDataWriteSlot(void** slotPointer) {
    if (slotPointer == nullptr || *slotPointer == nullptr)
        return DEVICE_ERR;

    std::lock_guard<std::mutex> lock(slotManagementMutex_);

    // Calculate the offset from the buffer start to find the corresponding slot
    char* ptr = static_cast<char*>(*slotPointer);
    size_t offset = ptr - buffer_;

    // Find the slot in activeSlotsByStart_
    auto it = activeSlotsByStart_.find(offset);
    if (it == activeSlotsByStart_.end()) {
        return DEVICE_ERR; // Slot not found
    }

    // Release the write access
    BufferSlot* slot = it->second;
    slot->ReleaseWriteAccess();

    // Clear the pointer
    *slotPointer = nullptr;

    // Notify any waiting readers that new data is available
    dataCV_.notify_all();

    return DEVICE_OK;
}


/**
 * ReleaseSlot is called after a slot's content has been fully read.
 * It assumes the caller has already released its read access (the slot is free).
 *
 * This implementation pushes only the start of the released slot onto the FILO
 * (releasedSlots_) and removes the slot from the active slot map and activeSlots_.
 */
int DataBuffer::ReleaseDataReadPointer(void** slotPointer) {
    if (slotPointer == nullptr || *slotPointer == nullptr)
        return DEVICE_ERR;
  
    std::unique_lock<std::mutex> lock(slotManagementMutex_);

    // Compute the slot's start offset.
    char* ptr = static_cast<char*>(*slotPointer);
    size_t offset = ptr - buffer_;
  
    // Find the slot in activeSlotMap_.
    auto it = activeSlotsByStart_.find(offset);
    if (it != activeSlotsByStart_.end()) {
        BufferSlot* slot = it->second;
        // Check if the slot is being accessed by any readers or writers.
        if (!slot->IsAvailableForWriting() || !slot->IsAvailableForReading()) {
            // TODO: right way to handle exceptions?
            throw std::runtime_error("Cannot release slot that is currently being accessed");
        }
  
        // If we've reached max size, remove the oldest element (front of vector).
        if (releasedSlots_.size() >= MAX_RELEASED_SLOTS) {
            releasedSlots_.erase(releasedSlots_.begin());
        }
        releasedSlots_.push_back(offset);
  
        // Remove slot from active structures.
        activeSlotsByStart_.erase(it);
        for (auto vecIt = activeSlotsVector_.begin(); vecIt != activeSlotsVector_.end(); ++vecIt) {
            if (vecIt->GetStart() == offset) {
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
    *slotPointer = nullptr;
    return DEVICE_OK;
}

const unsigned char* DataBuffer::PopNextDataReadPointer(size_t* dataSize, Metadata &md, bool waitForData) {
    while (true) {
        std::unique_lock<std::mutex> lock(slotManagementMutex_);
        
        // If data is available, process it.
        if (!activeSlotsVector_.empty() && currentSlotIndex_ < activeSlotsVector_.size()) {
            BufferSlot& currentSlot = activeSlotsVector_[currentSlotIndex_];
            
            if (!currentSlot.AcquireReadAccess()) {
                throw std::runtime_error("Failed to acquire read access for the current slot.");
            }
            
            const unsigned char* slotPointer = reinterpret_cast<const unsigned char*>(buffer_ + currentSlot.GetStart());
            if (dataSize != nullptr) {
                *dataSize = currentSlot.GetLength();
            }
            
            currentSlotIndex_++;
            return slotPointer;
        }
        
        // No data available.
        if (!waitForData) {
            throw std::runtime_error("No data available to read.");
        }
        
        // Wait for notification of new data.
        dataCV_.wait(lock);
        // When we wake up, the while loop will check again if data is available
    }
}

unsigned int DataBuffer::GetMemorySizeMB() const {
    // Convert bytes to MB (1 MB = 1048576 bytes)
    return static_cast<unsigned int>(bufferSize_ >> 20);
}

int DataBuffer::PeekNextDataReadPointer(void** slotPointer, size_t* dataSize,
                                          Metadata &md) {
    // Immediately check if there is an unread slot without waiting.
    std::unique_lock<std::mutex> lock(slotManagementMutex_);
    if (activeSlotsVector_.empty() || currentSlotIndex_ >= activeSlotsVector_.size()) {
        return DEVICE_ERR; // No unread data available.
    }

    // Obtain the next available slot *without* advancing currentSlotIndex_.
    BufferSlot& currentSlot = activeSlotsVector_[currentSlotIndex_];
    if (!currentSlot.AcquireReadAccess())
        return DEVICE_ERR;

    *slotPointer = buffer_ + currentSlot.GetStart();
    *dataSize = currentSlot.GetLength();
    // (If metadata is stored per slot, populate md here.)
    return DEVICE_OK;
}

const unsigned char* DataBuffer::PeekDataReadPointerAtIndex(size_t n, size_t* dataSize, Metadata &md) {

    std::unique_lock<std::mutex> lock(slotManagementMutex_);
    if (activeSlotsVector_.empty() || (currentSlotIndex_ + n) >= activeSlotsVector_.size()) {
        throw std::runtime_error("Not enough unread data available.");
    }
    
    BufferSlot& slot = activeSlotsVector_[currentSlotIndex_ + n];
    if (!slot.AcquireReadAccess())
        throw std::runtime_error("Failed to acquire read access for the selected slot.");
    
    // Assign the size from the slot.
    if (dataSize != nullptr) {
        *dataSize = slot.GetLength();
    }
    
    // Return a constant pointer to the data.
    return reinterpret_cast<const unsigned char*>(buffer_ + slot.GetStart());
}

/**
 * Releases the read access that was acquired by a peek.
 * This is similar to ReleaseDataReadPointer except that it does not
 * remove the slot from the active list. This should be used when the
 * overwriteWhenFull_ flag is true and the caller wants to release the
 * peeked slot for reuse.
 */

int DataBuffer::ReleasePeekDataReadPointer(void** slotPointer) {
    if (slotPointer == nullptr || *slotPointer == nullptr)
        return DEVICE_ERR;


    std::lock_guard<std::mutex> lock(slotManagementMutex_);
    char* ptr = static_cast<char*>(*slotPointer);
    size_t offset = ptr - buffer_;

    // Look up the corresponding slot by its buffer offset.
    auto it = activeSlotsByStart_.find(offset);
    if (it == activeSlotsByStart_.end())
        return DEVICE_ERR;  // Slot not found

    BufferSlot* slot = it->second;
    // Release the read access (this does NOT remove the slot from the active list)
    slot->ReleaseReadAccess();

    *slotPointer = nullptr;
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
        usedMemory += slot.GetLength();
    }
    return usedMemory;
}

size_t DataBuffer::GetFreeMemory() const {
    std::lock_guard<std::mutex> lock(slotManagementMutex_);
    // Free memory is the total buffer size minus the sum of all occupied memory.
    size_t usedMemory = 0;
    for (const auto& slot : activeSlotsVector_) {
        usedMemory += slot.GetLength();
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
   for (const BufferSlot &slot : activeSlotsVector_) {
      if (!slot.IsAvailableForReading() || !slot.IsAvailableForWriting()) {
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
    std::lock_guard<std::mutex> lock(slotManagementMutex_);
    // Return the number of unread slots.
    // currentSlotIndex_ tracks the next slot to read,
    // so unread count is the total slots minus this index.
    return (activeSlotsVector_.size() > currentSlotIndex_) ? 
           static_cast<long>(activeSlotsVector_.size() - currentSlotIndex_) : 0;
}
