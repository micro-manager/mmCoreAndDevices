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


BufferManager::BufferManager(bool useV2Buffer, unsigned int memorySizeMB)
   : useV2_(useV2Buffer), circBuffer_(nullptr), v2Buffer_(nullptr)
{
   if (useV2_.load()) {
      v2Buffer_ = new DataBuffer(memorySizeMB);
   } else {
      circBuffer_ = new CircularBuffer(memorySizeMB);
   }
}

BufferManager::~BufferManager()
{
   if (useV2_.load()) {
      if (v2Buffer_) {
         delete v2Buffer_;
      }
   } else {
      if (circBuffer_) {
         delete circBuffer_;
      }
   }
}

void BufferManager::ReallocateBuffer(unsigned int memorySizeMB) {
   if (useV2_.load()) {
      int numOutstanding = v2Buffer_->NumOutstandingSlots();   
      if (numOutstanding > 0) {
         throw CMMError("Cannot reallocate V2 buffer: " + std::to_string(numOutstanding) + " outstanding active slot(s) detected.");
      }
      delete v2Buffer_;
      v2Buffer_ = new DataBuffer(memorySizeMB);
   } else {
      delete circBuffer_;
      circBuffer_ = new CircularBuffer(memorySizeMB);
   }
}

const void* BufferManager::GetLastData()
{
   if (useV2_.load()) {
      Metadata dummyMetadata;
      // NOTE: ensure calling code releases the slot after use
      return v2Buffer_->PeekDataReadPointerAtIndex(0, dummyMetadata);
   } else {
      return circBuffer_->GetTopImage();
   }
}

const void* BufferManager::PopNextData()
{
   if (useV2_.load()) {
      Metadata dummyMetadata;
      // NOTE: ensure calling code releases the slot after use
      return v2Buffer_->PopNextDataReadPointer(dummyMetadata, false);
   } else {
      return circBuffer_->PopNextImage();
   }
}

long BufferManager::GetRemainingDataCount() const
{
   if (useV2_.load()) {
      return v2Buffer_->GetActiveSlotCount();
   } else {
      return circBuffer_->GetRemainingImageCount();
   }
}

unsigned BufferManager::GetMemorySizeMB() const {
   if (useV2_.load()) {
      return v2Buffer_->GetMemorySizeMB();
   } else {
      return circBuffer_->GetMemorySizeMB();
   }
}

unsigned BufferManager::GetFreeSizeMB() const {
   if (useV2_.load()) {
      return (unsigned) v2Buffer_->GetFreeMemory() / 1024 / 1024;
   } else {
      return circBuffer_->GetFreeSize() * circBuffer_->GetImageSizeBytes() / 1024 / 1024;
   }
}

bool BufferManager::Overflow() const
{
   if (useV2_.load()) {
      return v2Buffer_->Overflow();
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
    
    if (useV2_.load()) {
        // All the data needed to interpret the image is in the metadata
        // This function will copy data and metadata into the buffer
      return v2Buffer_->InsertData(buf, width * height * byteDepth * numChannels, &md, callerLabel);
    } else {
        return circBuffer_->InsertMultiChannel(buf, numChannels, width, height, 
            byteDepth, &md) ? DEVICE_OK : DEVICE_BUFFER_OVERFLOW;
    } 
}

int BufferManager::InsertData(const char* callerLabel, const unsigned char* buf, size_t dataSize, Metadata* pMd) {
    //  Initialize metadata with either provided metadata or create empty
    Metadata md = (pMd != nullptr) ? *pMd : Metadata();
    
    if (!useV2_.load()) {
        throw CMMError("InsertData() not supported with circular buffer. Must use V2 buffer.");
    }
    // All the data needed to interpret the image should be in the metadata
    // This function will copy data and metadata into the buffer
    return v2Buffer_->InsertData(buf, dataSize, &md, callerLabel);    
}


const void* BufferManager::GetLastDataMD(Metadata& md) const throw (CMMError)
{
   return GetLastDataMD(0, 0, md); // single channel size doesnt matter here
}

const void* BufferManager::GetLastDataMD(unsigned channel, unsigned singleChannelSizeBytes, Metadata& md) const throw (CMMError)
{
   if (useV2_.load()) {
      const void* basePtr = v2Buffer_->PeekLastDataReadPointer(md);
      if (basePtr == nullptr)
         throw CMMError("V2 buffer is empty.", MMERR_CircularBufferEmpty);
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
   if (useV2_.load()) {
      // NOTE: make sure calling code releases the slot after use.
      return v2Buffer_->PeekDataReadPointerAtIndex(n, md);
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
 *             The V2 buffer is data type agnostic
 */
const void* BufferManager::PopNextDataMD(unsigned channel, 
         unsigned singleChannelSizeBytes, Metadata& md) throw (CMMError)
{
   if (useV2_.load()) {
      const void* basePtr = v2Buffer_->PopNextDataReadPointer(md, false);
      if (basePtr == nullptr)
         throw CMMError("V2 buffer is empty.", MMERR_CircularBufferEmpty);

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

int BufferManager::EnableV2Buffer(bool enable) {
    // Don't do anything if we're already in the requested state
    if (enable == useV2_.load()) {
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
            v2Buffer_ = newBuffer;
        } else {
            // Switch to circular buffer
            int numOutstanding = v2Buffer_->NumOutstandingSlots();
            if (numOutstanding > 0) {
                throw CMMError("Cannot switch to circular buffer: " + std::to_string(numOutstanding) + " outstanding active slot(s) detected.");
            }
            CircularBuffer* newBuffer = new CircularBuffer(memorySizeMB);
            delete v2Buffer_;
            v2Buffer_ = nullptr;
            circBuffer_ = newBuffer;
        }

        useV2_.store(enable);
        return DEVICE_OK;
    } catch (const std::exception&) {
        // If allocation fails, keep the existing buffer
        return DEVICE_ERR;
    }
}

bool BufferManager::IsUsingV2Buffer() const {
   return useV2_.load();
}

int BufferManager::ReleaseReadAccess(const void* ptr) {
    if (useV2_.load() && ptr) {
        return v2Buffer_->ReleaseDataReadPointer(ptr);
    }
    return DEVICE_ERR;
}

unsigned BufferManager::GetDataSize(const void* ptr) const {
   if (!useV2_.load()) 
      return circBuffer_->GetImageSizeBytes();
   else
      return static_cast<long>(v2Buffer_->GetDatumSize(ptr));
}

int BufferManager::SetOverwriteData(bool overwrite) {
    if (useV2_.load()) {
        return v2Buffer_->SetOverwriteData(overwrite);
    } else {
        return circBuffer_->SetOverwriteData(overwrite);
    }
}

int BufferManager::AcquireWriteSlot(const char* deviceLabel, size_t dataSize, size_t additionalMetadataSize,
    void** dataPointer, void** additionalMetadataPointer, Metadata* pInitialMetadata) {
   if (!useV2_.load()) {
      // Not supported for circular buffer
      return DEVICE_ERR;
   }

   // Initialize metadata with either provided metadata or create empty
   Metadata md = (pInitialMetadata != nullptr) ? *pInitialMetadata : Metadata();
   
   std::string serializedMetadata = md.Serialize();
   int ret = v2Buffer_->AcquireWriteSlot(dataSize, additionalMetadataSize,
      dataPointer, additionalMetadataPointer, serializedMetadata, deviceLabel);
   return ret;
}

int BufferManager::FinalizeWriteSlot(const void* imageDataPointer, size_t actualMetadataBytes) {
    if (!useV2_.load()) {
        // Not supported for circular buffer
        return DEVICE_ERR;
    }
    return v2Buffer_->FinalizeWriteSlot(imageDataPointer, actualMetadataBytes);
}

void BufferManager::ExtractMetadata(const void* dataPtr, Metadata& md) const {
    if (!useV2_.load()) {
        throw CMMError("ExtractMetadata is only supported with V2 buffer enabled");
    }
    
    if (v2Buffer_ == nullptr) {
        throw CMMError("V2 buffer is null");
    }

    int result = v2Buffer_->ExtractCorrespondingMetadata(dataPtr, md);
    if (result != DEVICE_OK) {
        throw CMMError("Failed to extract metadata");
    }
}

const void* BufferManager::GetLastDataFromDevice(const std::string& deviceLabel) throw (CMMError) {
    if (!useV2_.load()) {
        throw CMMError("V2 buffer must be enabled for device-specific data access");
    }
    Metadata md;
    return GetLastDataMDFromDevice(deviceLabel, md);
}

const void* BufferManager::GetLastDataMDFromDevice(const std::string& deviceLabel, Metadata& md) throw (CMMError) {
    if (!useV2_.load()) {
        throw CMMError("V2 buffer must be enabled for device-specific data access");
    }
    
    const void* basePtr = v2Buffer_->PeekLastDataReadPointerFromDevice(deviceLabel, md);
    if (basePtr == nullptr) {
        throw CMMError("No data found for device: " + deviceLabel, MMERR_InvalidContents);
    }
    return basePtr;
}

bool BufferManager::IsPointerInV2Buffer(const void* ptr) const throw (CMMError) {
    if (!useV2_.load()) {
        return false;
    }
    
    if (v2Buffer_ == nullptr) {
        return false;
    }

    return v2Buffer_->IsPointerInBuffer(ptr);
}

bool BufferManager::GetOverwriteData() const {
    if (useV2_.load()) {
        return v2Buffer_->GetOverwriteData();
    } else {
        return circBuffer_->GetOverwriteData();
    }
}

void BufferManager::Reset() {
    if (useV2_.load()) {
        v2Buffer_->Reset();
    } else {
        circBuffer_->Clear();
    }
}

