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
 */
BufferSlot::BufferSlot(std::size_t start, std::size_t length)
    : start_(start), length_(length)
{
    // Using RAII-based locking with std::shared_timed_mutex.
}

/**
 * Destructor.
 */
BufferSlot::~BufferSlot() {
    // No explicit cleanup required.
}

/**
 * Returns the starting offset (in bytes) of the slot.
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
 * Attempts to acquire exclusive write access without blocking.
 *
 * @return True if the exclusive lock was acquired, false otherwise.
 */
bool BufferSlot::TryAcquireWriteAccess() {
    return rwMutex_.try_lock();
}

/**
 * Acquires exclusive write access (blocking call).
 * This method will block until exclusive access is granted.
 */
void BufferSlot::AcquireWriteAccess() {
    rwMutex_.lock();
}

/**
 * Releases exclusive write access.
 */
void BufferSlot::ReleaseWriteAccess() {
    rwMutex_.unlock();
}

/**
 * Acquires shared read access (blocking).
 */
void BufferSlot::AcquireReadAccess() {
    rwMutex_.lock_shared();
}

/**
 * Releases shared read access.
 */
void BufferSlot::ReleaseReadAccess() {
    rwMutex_.unlock_shared();
}

/**
 * Checks if the slot is available for writing.
 * The slot is considered available if no thread holds either an exclusive or a shared lock.
 *
 * @return True if available for writing.
 */
bool BufferSlot::IsAvailableForWriting() const {
    // If we can acquire the lock exclusively, then no readers or writer are active.
    if (rwMutex_.try_lock()) {
        rwMutex_.unlock();
        return true;
    }
    return false;
}

/**
 * Checks if the slot is available for reading.
 * A slot is available for reading if no exclusive lock is held (readers might already be active).
 *
 * @return True if available for reading.
 */
bool BufferSlot::IsAvailableForReading() const {
    // If we can acquire a shared lock, then no writer is active.
    if (rwMutex_.try_lock_shared()) {
        rwMutex_.unlock_shared();
        return true;
    }
    return false;
}



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
    // Total size is header + image data + metadata
    size_t totalSize = sizeof(BufferSlotRecord) + dataSize + metaSize;
    unsigned char* imageDataPointer = nullptr;
    unsigned char* metadataPointer = nullptr;
    int result = GetDataWriteSlot(totalSize, metaSize, &imageDataPointer, &metadataPointer);
    if (result != DEVICE_OK)
        return result;

    
    // The externally returned imageDataPointer points to the image data.
    // Write out the header by subtracting the header size.
    BufferSlotRecord* headerPointer = reinterpret_cast<BufferSlotRecord*>(imageDataPointer - sizeof(BufferSlotRecord));
    headerPointer->imageSize = dataSize;
    headerPointer->metadataSize = metaSize;

    // Copy the image data into the allocated slot (imageDataPointer is already at the image data).
    tasksMemCopy_->MemCopy((void*)imageDataPointer, data, dataSize);

    // If metadata is available, copy it right after the image data.
    if (metaSize > 0) {
        unsigned char* metaPtr = imageDataPointer + dataSize;
        std::memcpy(metaPtr, metaStr.data(), metaSize);
    }

    // Release the write slot
    return ReleaseDataWriteSlot(imageDataPointer, metaSize > 0 ? static_cast<int>(metaSize) : -1);
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
int DataBuffer::GetDataWriteSlot(size_t imageSize, size_t metadataSize,
                                 unsigned char** imageDataPointer, unsigned char** metadataPointer)
{
    std::lock_guard<std::mutex> lock(slotManagementMutex_);
    if (buffer_ == nullptr) {
        return DEVICE_ERR;
    }

    // Determine the required alignment based on our header.
    size_t alignment = alignof(BufferSlotRecord);
    // Compute the raw total size (header + image data + metadata) and then round up.
    size_t rawTotalSize = sizeof(BufferSlotRecord) + imageSize + metadataSize;
    size_t totalSlotSize = AlignTo(rawTotalSize, alignment);
    size_t candidateStart = 0;

    if (!overwriteWhenFull_) {
        // FIRST: Look for a fit in recycled slots (releasedSlots_)
        for (int i = static_cast<int>(releasedSlots_.size()) - 1; i >= 0; i--) {
            // Align the candidate start position
            candidateStart = AlignTo(releasedSlots_[i], alignment);
            
            // Find the free region that contains this position
            auto it = freeRegions_.upper_bound(candidateStart);
            if (it != freeRegions_.begin()) {
                --it;
                size_t freeStart = it->first;
                size_t freeEnd = freeStart + it->second;
                
                // Check if the position is actually within this free region
                if (candidateStart >= freeStart && candidateStart < freeEnd && 
                    candidateStart + totalSlotSize <= freeEnd) {
                    releasedSlots_.erase(releasedSlots_.begin() + i);
                    return CreateAndRegisterNewSlot(candidateStart, totalSlotSize, imageSize,
                                                  imageDataPointer, metadataPointer, true);
                }
            }
            
            // If we get here, this released slot position isn't in a free region
            // so remove it as it's no longer valid
            releasedSlots_.erase(releasedSlots_.begin() + i);
        }


        // SECOND: Look in the free-region list as fallback.
        for (auto it = freeRegions_.begin(); it != freeRegions_.end(); ++it) {
            // Align the free region start.
            size_t alignedCandidate = AlignTo(it->first, alignment);
            // Check if the free region has enough space after alignment.
            if (it->first + it->second >= alignedCandidate + totalSlotSize) {
                candidateStart = alignedCandidate;
                return CreateAndRegisterNewSlot(candidateStart, totalSlotSize, imageSize,
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
        return CreateAndRegisterNewSlot(candidateStart, totalSlotSize, imageSize,
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
int DataBuffer::ReleaseDataWriteSlot(unsigned char* imageDataPointer, int actualMetadataBytes) {
    if (imageDataPointer == nullptr)
        return DEVICE_ERR;

    std::lock_guard<std::mutex> lock(slotManagementMutex_);

    // Convert the externally provided imageDataPointer (which points to the image data)
    // to the true slot start (header) by subtracting sizeof(BufferSlotRecord).
    unsigned char* headerPointer = imageDataPointer - sizeof(BufferSlotRecord);
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
    if (imageDataPointer == nullptr )
        return DEVICE_ERR;
  
    std::unique_lock<std::mutex> lock(slotManagementMutex_);
  
    // Compute the header pointer by subtracting the header size.
    const unsigned char* headerPointer = imageDataPointer - sizeof(BufferSlotRecord);
    size_t offset = headerPointer - buffer_;

    auto it = activeSlotsByStart_.find(offset);
    if (it == activeSlotsByStart_.end())
        return DEVICE_ERR;  // Slot not found

    BufferSlot* slot = it->second;
    // Release the read access (this does NOT remove the slot from the active list)
    slot->ReleaseReadAccess();
  
    if (!overwriteWhenFull_) {
        // Now check if the slot is not being accessed 
        // (i.e. this was the last/readers and no writer holds it)
        if (slot->IsAvailableForWriting() && slot->IsAvailableForReading()) {
            RemoveSlotFromActiveTracking(offset, it);
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
    const BufferSlotRecord* header = reinterpret_cast<const BufferSlotRecord*>(buffer_ + slotStart);
    const unsigned char* imageDataPointer = buffer_ + slotStart + sizeof(BufferSlotRecord);
    *imageDataSize = header->imageSize;

    if (header->metadataSize > 0) {
        const char* metaDataStart = reinterpret_cast<const char*>(imageDataPointer + header->imageSize);
        md.Restore(metaDataStart);
    } else {
        md.Clear();
    }

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
    *imageDataPointer = buffer_ + currentSlot->GetStart() + sizeof(BufferSlotRecord);
    const BufferSlotRecord* headerPointer = reinterpret_cast<const BufferSlotRecord*>( (*imageDataPointer) - sizeof(BufferSlotRecord) );
    *imageDataSize = headerPointer->imageSize;
    
    // Populate the metadata from the slot.
    std::string metaStr;
    if (headerPointer->metadataSize > 0) {
        const char* metaDataStart = reinterpret_cast<const char*>(*imageDataPointer + headerPointer->imageSize);
        metaStr.assign(metaDataStart, headerPointer->metadataSize);
    }
    md.Restore(metaStr.c_str());    
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
    
    // Obtain pointer to the image data by skipping over the header.
    const unsigned char* imageDataPointer = buffer_ + currentSlot->GetStart() + sizeof(BufferSlotRecord);
    const BufferSlotRecord* headerPointer = reinterpret_cast<const BufferSlotRecord*>(imageDataPointer - sizeof(BufferSlotRecord));
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
      #ifdef _WIN32
          VirtualFree(buffer_, 0, MEM_RELEASE);
      #else
          munmap(buffer_, bufferSize_);
      #endif
      buffer_ = nullptr;
   }
   
   // Allocate a new buffer using the provided memory size.
   AllocateBuffer(memorySizeMB);

   return DEVICE_OK;
}

long DataBuffer::GetActiveSlotCount() const {
    return static_cast<long>(activeSlotsVector_.size());
}

unsigned DataBuffer::GetImageWidth(const unsigned char* imageDataPtr) const {
    const BufferSlotRecord* header = reinterpret_cast<const BufferSlotRecord*>(imageDataPtr - sizeof(BufferSlotRecord));
    if (header->metadataSize > 0) {
        Metadata md;
        md.Restore(reinterpret_cast<const char*>(imageDataPtr + header->imageSize));
        return static_cast<unsigned>(atoi(md.GetSingleTag(MM::g_Keyword_Metadata_Width).GetValue().c_str()));
    }
    throw std::runtime_error("No metadata available for image width");
}

unsigned DataBuffer::GetImageHeight(const unsigned char* imageDataPtr) const {
    const BufferSlotRecord* header = reinterpret_cast<const BufferSlotRecord*>(imageDataPtr - sizeof(BufferSlotRecord));
    if (header->metadataSize > 0) {
        Metadata md;
        md.Restore(reinterpret_cast<const char*>(imageDataPtr + header->imageSize));
        return static_cast<unsigned>(atoi(md.GetSingleTag(MM::g_Keyword_Metadata_Height).GetValue().c_str()));
    }
    throw std::runtime_error("No metadata available for image height");
}

unsigned DataBuffer::GetBytesPerPixel(const unsigned char* imageDataPtr) const {
    const BufferSlotRecord* header = reinterpret_cast<const BufferSlotRecord*>(imageDataPtr - sizeof(BufferSlotRecord));
    if (header->metadataSize > 0) {
        Metadata md;
        md.Restore(reinterpret_cast<const char*>(imageDataPtr + header->imageSize));
        std::string pixelType = md.GetSingleTag(MM::g_Keyword_PixelType).GetValue();
        if (pixelType == MM::g_Keyword_PixelType_GRAY8)
            return 1;
        else if (pixelType == MM::g_Keyword_PixelType_GRAY16)
            return 2;
        else if (pixelType == MM::g_Keyword_PixelType_GRAY32)
            return 4;
        else if (pixelType == MM::g_Keyword_PixelType_RGB32)
            return 4;
        else if (pixelType == MM::g_Keyword_PixelType_RGB64)
            return 8;
    }
    throw std::runtime_error("No metadata available for bytes per pixel");
}


unsigned DataBuffer::GetImageBitDepth(const unsigned char* imageDataPtr) const {
    const BufferSlotRecord* header = reinterpret_cast<const BufferSlotRecord*>(imageDataPtr - sizeof(BufferSlotRecord));
    if (header->metadataSize > 0) {
        Metadata md;
        md.Restore(reinterpret_cast<const char*>(imageDataPtr + header->imageSize));
        std::string pixelType = md.GetSingleTag(MM::g_Keyword_PixelType).GetValue();
        if (pixelType == MM::g_Keyword_PixelType_GRAY8)
            return 8;
        else if (pixelType == MM::g_Keyword_PixelType_GRAY16)
            return 16;
        else if (pixelType == MM::g_Keyword_PixelType_GRAY32)
            return 32;
        else if (pixelType == MM::g_Keyword_PixelType_RGB32)
            return 32;
        else if (pixelType == MM::g_Keyword_PixelType_RGB64)
            return 64;
    }
    throw std::runtime_error("No metadata available for bit depth");
}


unsigned DataBuffer::GetNumberOfComponents(const unsigned char* imageDataPtr) const {
    const BufferSlotRecord* header = reinterpret_cast<const BufferSlotRecord*>(imageDataPtr - sizeof(BufferSlotRecord));
    if (header->metadataSize > 0) {
        Metadata md;
        md.Restore(reinterpret_cast<const char*>(imageDataPtr + header->imageSize));
        std::string pixelType = md.GetSingleTag(MM::g_Keyword_PixelType).GetValue();
        if (pixelType == MM::g_Keyword_PixelType_GRAY8 ||
            pixelType == MM::g_Keyword_PixelType_GRAY16 ||
            pixelType == MM::g_Keyword_PixelType_GRAY32)
            return 1;
        else if (pixelType == MM::g_Keyword_PixelType_RGB32 ||
                 pixelType == MM::g_Keyword_PixelType_RGB64)
            return 4;
    }
    throw std::runtime_error("No metadata available for number of components");
}

long DataBuffer::GetImageBufferSize(const unsigned char* imageDataPtr) const {
    const BufferSlotRecord* header = reinterpret_cast<const BufferSlotRecord*>(imageDataPtr - sizeof(BufferSlotRecord));
    return header->imageSize;
}

void DataBuffer::RemoveSlotFromActiveTracking(size_t offset, std::map<size_t, BufferSlot*>::iterator it) {
    // Assert that caller holds the mutex.
    // This assertion checks that the mutex is already locked by ensuring try_lock() fails.
    assert(!slotManagementMutex_.try_lock() && "Caller must hold slotManagementMutex_");

    // Only add to released slots in non-overwrite mode.
    if (!overwriteWhenFull_) {
        if (releasedSlots_.size() >= MAX_RELEASED_SLOTS)
            releasedSlots_.erase(releasedSlots_.begin());
        releasedSlots_.push_back(offset);
    }

    // Update the freeRegions_ list to include the freed region and merge with adjacent regions
    size_t freedRegionSize = it->second->GetLength();
    size_t newRegionStart = offset;
    size_t newRegionEnd = offset + freedRegionSize;

    // Find the free region that starts at or after newEnd.
    auto right = freeRegions_.lower_bound(newRegionEnd);

    // Check if there is a free region immediately preceding the new region.
    auto left = freeRegions_.lower_bound(newRegionStart);
    if (left != freeRegions_.begin()) {
        auto prev = std::prev(left);
        // If the previous region's end matches the new region's start...
        if (prev->first + prev->second == newRegionStart) {
            newRegionStart = prev->first;
            freeRegions_.erase(prev);
        }
    }

    // Check if the region immediately to the right can be merged.
    if (right != freeRegions_.end() && right->first == newRegionEnd) {
        newRegionEnd = right->first + right->second;
        freeRegions_.erase(right);
    }

    // Insert the merged (or standalone) free region.
    size_t newRegionSize = (newRegionEnd > newRegionStart ? newRegionEnd - newRegionStart : 0);
    if (newRegionSize > 0) {
        freeRegions_[newRegionStart] = newRegionSize;
    }

    // Remove the slot from the active tracking structures.
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

int DataBuffer::CreateAndRegisterNewSlot(size_t candidateStart, 
                                         size_t totalSlotSize, 
                                         size_t imageDataSize,
                                         unsigned char** imageDataPointer,
                                         unsigned char** metadataPointer,
                                         bool fromFreeRegion) {
    // Assert that caller holds the mutex.
    // This assertion checks that the mutex is already locked by ensuring try_lock() fails.
    assert(!slotManagementMutex_.try_lock() && "Caller must hold slotManagementMutex_");

    // Use the provided candidateStart to create and register a new slot.
    BufferSlot* newSlot = new BufferSlot(candidateStart, totalSlotSize);
    // Acquire the write access
    newSlot->AcquireWriteAccess();
    activeSlotsVector_.push_back(std::unique_ptr<BufferSlot>(newSlot));
    activeSlotsByStart_[candidateStart] = newSlot;

    // Initialize header (the metadata sizes will be updated later).
    BufferSlotRecord* headerPointer = reinterpret_cast<BufferSlotRecord*>(buffer_ + candidateStart);
    headerPointer->imageSize = imageDataSize;
    headerPointer->metadataSize = totalSlotSize - imageDataSize - sizeof(BufferSlotRecord);

    // Set the output pointers
    *imageDataPointer = buffer_ + candidateStart + sizeof(BufferSlotRecord);
    *metadataPointer = *imageDataPointer + imageDataSize;

    // If the candidate comes from a free region, update the freeRegions_ map.
    if (fromFreeRegion) {
        // Adjust the free region list to remove the allocated block.
        auto it = freeRegions_.upper_bound(candidateStart);
        if (it != freeRegions_.begin()) {
            --it;
            size_t freeRegionStart = it->first;
            size_t freeRegionSize = it->second;
            size_t freeRegionEnd = freeRegionStart + freeRegionSize;
            // candidateStart must lie within [freeRegionStart, freeRegionEnd).
            if (candidateStart >= freeRegionStart &&
                candidateStart + totalSlotSize <= freeRegionEnd)
            {
                freeRegions_.erase(it);
                // If there is a gap before the candidate slot, add that as a free region.
                if (candidateStart > freeRegionStart) {
                    size_t gap = candidateStart - freeRegionStart;
                    if (gap > 0)
                        freeRegions_.insert({freeRegionStart, gap});
                }
                // If there is a gap after the candidate slot, add that as well.
                if (freeRegionEnd > candidateStart + totalSlotSize) {
                    size_t gap = freeRegionEnd - (candidateStart + totalSlotSize);
                    if (gap > 0)
                        freeRegions_.insert({candidateStart + totalSlotSize, gap});
                }
            }
        }
    }

    return DEVICE_OK;
}
