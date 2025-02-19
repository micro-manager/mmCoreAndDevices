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
   if (useV2_) {
      v2Buffer_ = new DataBuffer(memorySizeMB);
   } else {
      circBuffer_ = new CircularBuffer(memorySizeMB);
   }
}

BufferManager::~BufferManager()
{
   if (useV2_) {
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
   if (useV2_) {
      delete v2Buffer_;
      v2Buffer_ = new DataBuffer(memorySizeMB);
   } else {
      delete circBuffer_;
      circBuffer_ = new CircularBuffer(memorySizeMB);
   }
}

const void* BufferManager::GetLastData()
{
   if (useV2_) {
      Metadata dummyMetadata;
      // NOTE: ensure calling code releases the slot after use
      return v2Buffer_->PeekDataReadPointerAtIndex(0, dummyMetadata);
   } else {
      return circBuffer_->GetTopImage();
   }
}

const void* BufferManager::PopNextData()
{
   if (useV2_) {
      Metadata dummyMetadata;
      // NOTE: ensure calling code releases the slot after use
      return v2Buffer_->PopNextDataReadPointer(dummyMetadata, false);
   } else {
      return circBuffer_->PopNextImage();
   }
}

long BufferManager::GetRemainingDataCount() const
{
   if (useV2_) {
      return v2Buffer_->GetActiveSlotCount();
   } else {
      return circBuffer_->GetRemainingImageCount();
   }
}

unsigned BufferManager::GetMemorySizeMB() const {
   if (useV2_) {
      return v2Buffer_->GetMemorySizeMB();
   } else {
      return circBuffer_->GetMemorySizeMB();
   }
}

unsigned BufferManager::GetFreeSizeMB() const {
   if (useV2_) {
      return (unsigned) v2Buffer_->GetFreeMemory() / 1024 / 1024;
   } else {
      return circBuffer_->GetFreeSize() * circBuffer_->GetImageSizeBytes() / 1024 / 1024;
   }
}

bool BufferManager::Overflow() const
{
   if (useV2_) {
      return v2Buffer_->Overflow();
   } else {
      return circBuffer_->Overflow();
   }
}

/**
 * @deprecated Use InsertData() instead
 */
bool BufferManager::InsertImage(const char* callerLabel, const unsigned char *buf, unsigned width, unsigned height, 
                               unsigned byteDepth, Metadata *pMd) {
   return InsertMultiChannel(callerLabel, buf, 1, width, height, byteDepth, pMd);
}

/**
 * @deprecated Use InsertData() instead
 */
bool BufferManager::InsertMultiChannel(const char* callerLabel, const unsigned char* buf,
    unsigned numChannels, unsigned width, unsigned height, unsigned byteDepth, Metadata* pMd) {
    
    //  Initialize metadata with either provided metadata or create empty
    Metadata md = (pMd != nullptr) ? *pMd : Metadata();
    
    if (useV2_) {
        // All the data needed to interpret the image is in the metadata
        // This function will copy data and metadata into the buffer
        int ret = v2Buffer_->InsertData(buf, width * height * byteDepth * numChannels, &md, callerLabel);
        return ret == DEVICE_OK;
    } else {
        return circBuffer_->InsertMultiChannel(buf, numChannels, width, height, 
            byteDepth, &md);
    } 
}

bool BufferManager::InsertData(const char* callerLabel, const unsigned char* buf, size_t dataSize, Metadata* pMd) {
    
    //  Initialize metadata with either provided metadata or create empty
    Metadata md = (pMd != nullptr) ? *pMd : Metadata();
    
    if (!useV2_) {
        throw CMMError("InsertData() not supported with circular buffer. Must use V2 buffer.");
    }
    // All the data needed to interpret the image should be in the metadata
    // This function will copy data and metadata into the buffer
    return v2Buffer_->InsertData(buf, dataSize, &md, callerLabel);    
}


const void* BufferManager::GetLastDataMD(Metadata& md) const throw (CMMError)
{
   return GetLastDataMD(0, md);
}

const void* BufferManager::GetLastDataMD(unsigned channel, Metadata& md) const throw (CMMError)
{
   if (useV2_) {
      if (channel != 0) {
         throw CMMError("V2 buffer does not support channels.", MMERR_InvalidContents);
      }
      const void* basePtr = v2Buffer_->PeekLastDataReadPointer(md);
      if (basePtr == nullptr)
         throw CMMError("V2 buffer is empty.", MMERR_CircularBufferEmpty);
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
   if (useV2_) {
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
   return PopNextDataMD(0, md);
}  

const void* BufferManager::PopNextDataMD(unsigned channel, Metadata& md) throw (CMMError)
{
   if (useV2_) {
      if (channel != 0) {
         throw CMMError("V2 buffer does not support channels.", MMERR_InvalidContents);
      }
      const void* basePtr = v2Buffer_->PopNextDataReadPointer(md, false);
      if (basePtr == nullptr)
         throw CMMError("V2 buffer is empty.", MMERR_CircularBufferEmpty);
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

bool BufferManager::EnableV2Buffer(bool enable) {
    // Don't do anything if we're already in the requested state
    if (enable == useV2_) {
        return true;
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
            CircularBuffer* newBuffer = new CircularBuffer(memorySizeMB);
            delete v2Buffer_;
            v2Buffer_ = nullptr;
            circBuffer_ = newBuffer;
            // Require it to be initialized manually, which doesn't actually matter
            // because it gets initialized before sequence acquisition starts anyway.
        }

        useV2_ = enable;
        return true;
    } catch (const std::exception&) {
        // If allocation fails, keep the existing buffer
        return false;
    }
}

bool BufferManager::IsUsingV2Buffer() const {
   return useV2_;
}

bool BufferManager::ReleaseReadAccess(const void* ptr) {
   if (useV2_ && ptr) {
      return v2Buffer_->ReleaseDataReadPointer(ptr) == DEVICE_OK;
   }
   return false;
}

unsigned BufferManager::GetDataSize(const void* ptr) const {
   if (!useV2_) 
      return circBuffer_->GetImageSizeBytes();
   else
      return static_cast<long>(v2Buffer_->GetDatumSize(ptr));
}

bool BufferManager::SetOverwriteData(bool overwrite) {
    if (useV2_) {
        return v2Buffer_->SetOverwriteData(overwrite) == DEVICE_OK;
    } else {
        // CircularBuffer doesn't have this functionality
        return false;
    }
}

bool BufferManager::AcquireWriteSlot(const char* deviceLabel, size_t dataSize, size_t additionalMetadataSize,
    void** dataPointer, void** additionalMetadataPointer, Metadata* pInitialMetadata) {
   if (!useV2_) {
      // Not supported for circular buffer
      return false;
   }

   // Initialize metadata with either provided metadata or create empty
   Metadata md = (pInitialMetadata != nullptr) ? *pInitialMetadata : Metadata();
   
   std::string serializedMetadata = md.Serialize();
   int ret = v2Buffer_->AcquireWriteSlot(dataSize, additionalMetadataSize,
      dataPointer, additionalMetadataPointer, serializedMetadata, deviceLabel);
   return ret == DEVICE_OK;
}

bool BufferManager::FinalizeWriteSlot(void* imageDataPointer, size_t actualMetadataBytes)
{
    if (!useV2_) {
        // Not supported for circular buffer
        return false;
    }
    
    int ret = v2Buffer_->FinalizeWriteSlot(imageDataPointer, actualMetadataBytes);
    return ret == DEVICE_OK;
}

void BufferManager::ExtractMetadata(const void* dataPtr, Metadata& md) const {
    if (!useV2_) {
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
    if (!useV2_) {
        throw CMMError("V2 buffer must be enabled for device-specific data access");
    }
    Metadata md;
    return GetLastDataMDFromDevice(deviceLabel, md);
}

const void* BufferManager::GetLastDataMDFromDevice(const std::string& deviceLabel, Metadata& md) throw (CMMError) {
    if (!useV2_) {
        throw CMMError("V2 buffer must be enabled for device-specific data access");
    }
    
    const void* basePtr = v2Buffer_->PeekLastDataReadPointerFromDevice(deviceLabel, md);
    if (basePtr == nullptr) {
        throw CMMError("No data found for device: " + deviceLabel, MMERR_InvalidContents);
    }
    return basePtr;
}

bool BufferManager::IsPointerInV2Buffer(const void* ptr) const throw (CMMError) {
    if (!useV2_) {
        return false;
    }
    
    if (v2Buffer_ == nullptr) {
        return false;
    }

    return v2Buffer_->IsPointerInBuffer(ptr);
}

bool BufferManager::GetOverwriteData() const {
    if (useV2_) {
        return v2Buffer_->GetOverwriteData();
    } else {
        return circBuffer_->GetOverwriteData();
    }
}

void BufferManager::Reset() {
    if (useV2_) {
        v2Buffer_->Reset();
    } else {
        circBuffer_->Clear();
    }
}

