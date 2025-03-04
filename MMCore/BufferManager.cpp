///////////////////////////////////////////////////////////////////////////////
// FILE:          BufferManager.cpp
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


#include "BufferManager.h"
#include <mutex>


BufferManager::BufferManager(bool useNewDataBuffer, unsigned int memorySizeMB)
   : useNewDataBuffer_(useNewDataBuffer), circBuffer_(nullptr), newDataBuffer_(nullptr)
{
   if (useNewDataBuffer_.load()) {
      newDataBuffer_ = new DataBuffer(memorySizeMB);
   } else {
      circBuffer_ = new CircularBuffer(memorySizeMB);
   }
}

BufferManager::~BufferManager()
{
   if (useNewDataBuffer_.load()) {
      if (newDataBuffer_) {
         delete newDataBuffer_;
      }
   } else {
      if (circBuffer_) {
         delete circBuffer_;
      }
   }
}

void BufferManager::Clear() {
   if (useNewDataBuffer_.load()) {
      newDataBuffer_->Clear();
   } else {
      circBuffer_->Clear();
   }
}

void BufferManager::ForceReset() {
    if (useNewDataBuffer_.load()) {
      // This is dangerous with the NewDataBuffer because there may be pointers into the buffer's memory
      newDataBuffer_->ReinitializeBuffer(GetMemorySizeMB(), true);
    } else {
      // This is not dangerous with the circular buffer because it does not give out pointers to its memory
      circBuffer_->Initialize(circBuffer_->NumChannels(), circBuffer_->Width(), 
                                circBuffer_->Height(), circBuffer_->Depth());
    }
}

bool BufferManager::InitializeCircularBuffer(unsigned int numChannels, unsigned int width, unsigned int height, unsigned int depth) {
   if (!useNewDataBuffer_.load()) {
      return circBuffer_->Initialize(numChannels, width, height, depth);
   }
   return false;
}

void BufferManager::ReallocateBuffer(unsigned int memorySizeMB) {
   if (useNewDataBuffer_.load()) {
      int numOutstanding = newDataBuffer_->NumOutstandingSlots();   
      if (numOutstanding > 0) {
         throw CMMError("Cannot reallocate NewDataBuffer: " + std::to_string(numOutstanding) + " outstanding active slot(s) detected.");
      }
      delete newDataBuffer_;
      newDataBuffer_ = new DataBuffer(memorySizeMB);
   } else {
      delete circBuffer_;
      circBuffer_ = new CircularBuffer(memorySizeMB);
   }
}

const void* BufferManager::GetLastData()
{
   if (useNewDataBuffer_.load()) {
      Metadata dummyMetadata;
      // NOTE: ensure calling code releases the slot after use
      return newDataBuffer_->PeekDataReadPointerAtIndex(0, dummyMetadata);
   } else {
      return circBuffer_->GetTopImage();
   }
}

const void* BufferManager::PopNextData()
{
   if (useNewDataBuffer_.load()) {
      Metadata dummyMetadata;
      // NOTE: ensure calling code releases the slot after use
      return newDataBuffer_->PopNextDataReadPointer(dummyMetadata, false);
   } else {
      return circBuffer_->PopNextImage();
   }
}

long BufferManager::GetRemainingDataCount() const
{
   if (useNewDataBuffer_.load()) {
      return newDataBuffer_->GetActiveSlotCount();
   } else {
      return circBuffer_->GetRemainingImageCount();
   }
}

unsigned BufferManager::GetMemorySizeMB() const {
   if (useNewDataBuffer_.load()) {
      return newDataBuffer_->GetMemorySizeMB();
   } else {
      return circBuffer_->GetMemorySizeMB();
   }
}

unsigned BufferManager::GetFreeSizeMB() const {
   if (useNewDataBuffer_.load()) {
      return (unsigned) newDataBuffer_->GetFreeMemory() / 1024 / 1024;
   } else {
      return circBuffer_->GetFreeSize() * circBuffer_->GetImageSizeBytes() / 1024 / 1024;
   }
}

bool BufferManager::Overflow() const
{
   if (useNewDataBuffer_.load()) {
      return newDataBuffer_->Overflow();
   } else {
      return circBuffer_->Overflow();
   }
}

/**
 * @deprecated Use InsertData() instead
 */
int BufferManager::InsertImage(const char* callerLabel, const unsigned char* buf, unsigned width, unsigned height, 
                             unsigned byteDepth, Metadata* pMd) {
    return InsertMultiChannel(callerLabel, buf, 1, width, height, byteDepth, pMd);
}

/**
 * @deprecated Use InsertData() instead
 */
int BufferManager::InsertMultiChannel(const char* callerLabel, const unsigned char* buf,
    unsigned numChannels, unsigned width, unsigned height, unsigned byteDepth, Metadata* pMd) {
    
    //  Initialize metadata with either provided metadata or create empty
    Metadata md = (pMd != nullptr) ? *pMd : Metadata();
    
    if (useNewDataBuffer_.load()) {
        // All the data needed to interpret the image is in the metadata
        // This function will copy data and metadata into the buffer
      return newDataBuffer_->InsertData(buf, width * height * byteDepth * numChannels, &md, callerLabel);
    } else {
        return circBuffer_->InsertMultiChannel(buf, numChannels, width, height, 
            byteDepth, &md) ? DEVICE_OK : DEVICE_BUFFER_OVERFLOW;
    } 
}

int BufferManager::InsertData(const char* callerLabel, const unsigned char* buf, size_t dataSize, Metadata* pMd) {
    //  Initialize metadata with either provided metadata or create empty
    Metadata md = (pMd != nullptr) ? *pMd : Metadata();
    
    if (!useNewDataBuffer_.load()) {
        throw CMMError("InsertData() not supported with circular buffer. Must use NewDataBuffer.");
    }
    // All the data needed to interpret the image should be in the metadata
    // This function will copy data and metadata into the buffer
    return newDataBuffer_->InsertData(buf, dataSize, &md, callerLabel);    
}


const void* BufferManager::GetLastDataMD(Metadata& md) const  
{
   return GetLastDataMD(0, 0, md); // single channel size doesnt matter here
}

const void* BufferManager::GetLastDataMD(unsigned channel, unsigned singleChannelSizeBytes, Metadata& md) const throw (CMMError)
{
   if (useNewDataBuffer_.load()) {
      const void* basePtr = newDataBuffer_->PeekLastDataReadPointer(md);
      if (basePtr == nullptr)
         throw CMMError("NewDataBuffer is empty.", MMERR_CircularBufferEmpty);
      // Add multiples of the number of bytes to get the channel pointer
      basePtr = static_cast<const unsigned char*>(basePtr) + channel * singleChannelSizeBytes;
      return basePtr;
   } else {
      const mm::ImgBuffer* pBuf = circBuffer_->GetTopImageBuffer(channel);
      if (pBuf != nullptr) {
         md = pBuf->GetMetadata();
         return pBuf->GetPixels();
      } else {
         throw CMMError("Circular buffer is empty.", MMERR_CircularBufferEmpty);
      }
   }
}

const void* BufferManager::GetNthDataMD(unsigned long n, Metadata& md) const throw (CMMError)
{
   if (useNewDataBuffer_.load()) {
      // NOTE: make sure calling code releases the slot after use.
      return newDataBuffer_->PeekDataReadPointerAtIndex(n, md);
   } else {
      const mm::ImgBuffer* pBuf = circBuffer_->GetNthFromTopImageBuffer(n);
      if (pBuf != nullptr) {
         md = pBuf->GetMetadata();
         return pBuf->GetPixels();
      } else {
         throw CMMError("Circular buffer is empty.", MMERR_CircularBufferEmpty);
      }
   }
}

const void* BufferManager::PopNextDataMD(Metadata& md) throw (CMMError)
{
   return PopNextDataMD(0, 0, md);
}  

/**
 * @deprecated Use PopNextDataMD() without channel parameter instead.
 *             The NewDataBuffer is data type agnostic
 */
const void* BufferManager::PopNextDataMD(unsigned channel, 
         unsigned singleChannelSizeBytes, Metadata& md) throw (CMMError)
{
   if (useNewDataBuffer_.load()) {
      const void* basePtr = newDataBuffer_->PopNextDataReadPointer(md, false);
      if (basePtr == nullptr)
         throw CMMError("NewDataBuffer is empty.", MMERR_CircularBufferEmpty);

      // Add multiples of the number of bytes to get the channel pointer
      basePtr = static_cast<const unsigned char*>(basePtr) + channel * singleChannelSizeBytes;
      
      return basePtr;
   } else {
      const mm::ImgBuffer* pBuf = circBuffer_->GetNextImageBuffer(channel);
      if (pBuf != nullptr) {
         md = pBuf->GetMetadata();
         return pBuf->GetPixels();
      } else {
         throw CMMError("Circular buffer is empty.", MMERR_CircularBufferEmpty);
      }
   }
}

int BufferManager::EnableNewDataBuffer(bool enable) {
    // Don't do anything if we're already in the requested state
    if (enable == useNewDataBuffer_.load()) {
        return DEVICE_OK;
    }

    // Create new buffer of requested type with same memory size
    unsigned memorySizeMB = GetMemorySizeMB();
    
    try {
        if (enable) {
            // Switch to V2 buffer
            DataBuffer* newBuffer = new DataBuffer(memorySizeMB);
            delete circBuffer_;
            circBuffer_ = nullptr;
            newDataBuffer_ = newBuffer;
        } else {
            // Switch to circular buffer
            int numOutstanding = newDataBuffer_->NumOutstandingSlots();
            if (numOutstanding > 0) {
                throw CMMError("Cannot switch to circular buffer: " + std::to_string(numOutstanding) + " outstanding active slot(s) detected.");
            }
            CircularBuffer* newBuffer = new CircularBuffer(memorySizeMB);
            delete newDataBuffer_;
            newDataBuffer_ = nullptr;
            circBuffer_ = newBuffer;
        }

        
        useNewDataBuffer_.store(enable);
        return DEVICE_OK;
    } catch (const std::exception&) {
        // If allocation fails, keep the existing buffer
        return DEVICE_ERR;
    }
}

bool BufferManager::IsUsingNewDataBuffer() const {
   return useNewDataBuffer_.load();
}

int BufferManager::ReleaseReadAccess(const void* ptr) {
    if (useNewDataBuffer_.load() && ptr) {
        return newDataBuffer_->ReleaseDataReadPointer(ptr);
    }
    return DEVICE_ERR;
}

unsigned BufferManager::GetDataSize(const void* ptr) const {
   if (!useNewDataBuffer_.load()) 
      return circBuffer_->GetImageSizeBytes();
   else
      return static_cast<long>(newDataBuffer_->GetDatumSize(ptr));
}

int BufferManager::SetOverwriteData(bool overwrite) {
    if (useNewDataBuffer_.load()) {
        return newDataBuffer_->SetOverwriteData(overwrite);
    } else {
        return circBuffer_->SetOverwriteData(overwrite);
    }
}

int BufferManager::AcquireWriteSlot(const char* deviceLabel, size_t dataSize, size_t additionalMetadataSize,
    void** dataPointer, void** additionalMetadataPointer, Metadata* pInitialMetadata) {
   if (!useNewDataBuffer_.load()) {
      // Not supported for circular buffer
      return DEVICE_ERR;
   }

   // Initialize metadata with either provided metadata or create empty
   Metadata md = (pInitialMetadata != nullptr) ? *pInitialMetadata : Metadata();
   
   std::string serializedMetadata = md.Serialize();
   int ret = newDataBuffer_->AcquireWriteSlot(dataSize, additionalMetadataSize,
      dataPointer, additionalMetadataPointer, serializedMetadata, deviceLabel);
   return ret;
}

int BufferManager::FinalizeWriteSlot(const void* imageDataPointer, size_t actualMetadataBytes) {
    if (!useNewDataBuffer_.load()) {
        // Not supported for circular buffer
        return DEVICE_ERR;
    }
    return newDataBuffer_->FinalizeWriteSlot(imageDataPointer, actualMetadataBytes);
}

void BufferManager::ExtractMetadata(const void* dataPtr, Metadata& md) const {
    if (!useNewDataBuffer_.load()) {
        throw CMMError("ExtractMetadata is only supported with NewDataBuffer enabled");
    }
    
    if (newDataBuffer_ == nullptr) {
        throw CMMError("NewDataBuffer is null");
    }

    int result = newDataBuffer_->ExtractCorrespondingMetadata(dataPtr, md);
    if (result != DEVICE_OK) {
        throw CMMError("Failed to extract metadata");
    }
}

const void* BufferManager::GetLastDataFromDevice(const std::string& deviceLabel) throw (CMMError) {
    if (!useNewDataBuffer_.load()) {
        throw CMMError("NewDataBuffer must be enabled for device-specific data access");
    }
    Metadata md;
    return GetLastDataMDFromDevice(deviceLabel, md);
}

const void* BufferManager::GetLastDataMDFromDevice(const std::string& deviceLabel, Metadata& md) throw (CMMError) {
    if (!useNewDataBuffer_.load()) {
        throw CMMError("NewDataBuffer must be enabled for device-specific data access");
    }
    
    const void* basePtr = newDataBuffer_->PeekLastDataReadPointerFromDevice(deviceLabel, md);
    if (basePtr == nullptr) {
        throw CMMError("No data found for device: " + deviceLabel, MMERR_InvalidContents);
    }
    return basePtr;
}

bool BufferManager::IsPointerInNewDataBuffer(const void* ptr) const {
    if (!useNewDataBuffer_.load()) {
        return false;
    }
    
    if (newDataBuffer_ == nullptr) {
        return false;
    }

    return newDataBuffer_->IsPointerInBuffer(ptr);
}

bool BufferManager::GetOverwriteData() const {
    if (useNewDataBuffer_.load()) {
        return newDataBuffer_->GetOverwriteData();
    } else {
        return circBuffer_->GetOverwriteData();
    }
}
