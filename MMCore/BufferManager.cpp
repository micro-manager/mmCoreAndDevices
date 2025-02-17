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

const void* BufferManager::GetLastImage()
{
   if (useV2_) {
      Metadata dummyMetadata;
      // NOTE: ensure calling code releases the slot after use
      return v2Buffer_->PeekDataReadPointerAtIndex(0, dummyMetadata);
   } else {
      return circBuffer_->GetTopImage();
   }
}

const void* BufferManager::PopNextImage()
{
   if (useV2_) {
      Metadata dummyMetadata;
      // NOTE: ensure calling code releases the slot after use
      return v2Buffer_->PopNextDataReadPointer(dummyMetadata, false);
   } else {
      return circBuffer_->PopNextImage();
   }
}

bool BufferManager::Initialize(unsigned numChannels, unsigned width, unsigned height, unsigned bytesPerPixel)
{
   if (useV2_) {
      // This in not required for v2 buffer because it can interleave multiple data types/image sizes
   } else {
      return circBuffer_->Initialize(numChannels, width, height, bytesPerPixel);
   }
}

unsigned BufferManager::GetMemorySizeMB() const
{
   if (useV2_) {
      return v2Buffer_->GetMemorySizeMB();
   } else {
      return circBuffer_->GetMemorySizeMB();
   }
}

long BufferManager::GetRemainingImageCount() const
{
   if (useV2_) {
      return v2Buffer_->GetActiveSlotCount();
   } else {
      return circBuffer_->GetRemainingImageCount();
   }
}

void BufferManager::Clear()
{
   if (useV2_) {
      // This has no effect on v2 buffer, because devices do not have authority to clear the buffer
      // since higher level code may hold pointers to data in the buffer.
      // It seems to be mostly used in live mode, where data is overwritten by default anyway.
   } else {
      circBuffer_->Clear();
   }
}

long BufferManager::GetSize(long imageSize) const
{
   if (useV2_) {
      unsigned int mb = v2Buffer_->GetMemorySizeMB();
      size_t totalBytes = static_cast<size_t>(mb) * 1024 * 1024;
      size_t num_images = totalBytes / static_cast<size_t>(imageSize);
      return static_cast<long>(num_images);
   } else {
      return circBuffer_->GetSize();
   }

}

long BufferManager::GetFreeSize(long imageSize) const
{
   if (useV2_) {
      unsigned int mb = v2Buffer_->GetFreeMemory();
      return static_cast<long>(mb) / imageSize;
   } else {
      return circBuffer_->GetFreeSize();
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

void BufferManager::PopulateMetadata(Metadata& md, const char* deviceLabel, 
      unsigned width, unsigned height, unsigned byteDepth, unsigned nComponents) {
    // Add the device label (can be used to route different devices to different buffers)
    md.put(MM::g_Keyword_Metadata_DataSourceDeviceLabel, deviceLabel);
    
    // Add essential image metadata needed for interpreting the image:
    md.PutImageTag(MM::g_Keyword_Metadata_Width, width);
    md.PutImageTag(MM::g_Keyword_Metadata_Height, height);
    
    if (byteDepth == 1)
         md.PutImageTag(MM::g_Keyword_PixelType, MM::g_Keyword_PixelType_GRAY8);
    else if (byteDepth == 2)
         md.PutImageTag(MM::g_Keyword_PixelType, MM::g_Keyword_PixelType_GRAY16);
    else if (byteDepth == 4) {
         if (nComponents == 1)
            md.PutImageTag(MM::g_Keyword_PixelType, MM::g_Keyword_PixelType_GRAY32);
         else
            md.PutImageTag(MM::g_Keyword_PixelType, MM::g_Keyword_PixelType_RGB32);
    }
    else if (byteDepth == 8)
         md.PutImageTag(MM::g_Keyword_PixelType, MM::g_Keyword_PixelType_RGB64);
    else
         md.PutImageTag(MM::g_Keyword_PixelType, MM::g_Keyword_PixelType_Unknown);
}

bool BufferManager::InsertImage(const char* callerLabel, const unsigned char* buf, 
      unsigned width, unsigned height, unsigned byteDepth, Metadata* pMd) {
   return InsertMultiChannel(callerLabel, buf, 1, width, height, byteDepth, 1, pMd);
}

bool BufferManager::InsertImage(const char* callerLabel, const unsigned char *buf, unsigned width, unsigned height, 
                               unsigned byteDepth, unsigned nComponents, Metadata *pMd) {
   return InsertMultiChannel(callerLabel, buf, 1, width, height, byteDepth, nComponents, pMd);
}


bool BufferManager::InsertMultiChannel(const char* callerLabel, const unsigned char *buf,
          unsigned numChannels, unsigned width, unsigned height, unsigned byteDepth, Metadata *pMd) {
   return InsertMultiChannel(callerLabel, buf, numChannels, width, height, byteDepth, 1, pMd);
}

bool BufferManager::InsertMultiChannel(const char* callerLabel, const unsigned char* buf,
    unsigned numChannels, unsigned width, unsigned height, unsigned byteDepth, unsigned nComponents, Metadata* pMd) {
    
    //  Initialize metadata with either provided metadata or create empty
    Metadata md = (pMd != nullptr) ? *pMd : Metadata();
    
    //  Add required and useful metadata. 
    PopulateMetadata(md, callerLabel, width, height, byteDepth, nComponents);

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

const void* BufferManager::GetLastImageMD(Metadata& md) const throw (CMMError)
{
   return GetLastImageMD(0, md);
}

const void* BufferManager::GetLastImageMD(unsigned channel, Metadata& md) const throw (CMMError)
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

const void* BufferManager::GetNthImageMD(unsigned long n, Metadata& md) const throw (CMMError)
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

const void* BufferManager::PopNextImageMD(Metadata& md) throw (CMMError)
{
   return PopNextImageMD(0, md);
}  

const void* BufferManager::PopNextImageMD(unsigned channel, Metadata& md) throw (CMMError)
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
    unsigned int memorySizeMB = GetMemorySizeMB();
    
    try {
        if (enable) {
            // Switch to V2 buffer
            DataBuffer* newBuffer = new DataBuffer(memorySizeMB);
            delete circBuffer_;
            circBuffer_ = nullptr;
            v2Buffer_ = newBuffer;
            v2Buffer_->ReinitializeBuffer(memorySizeMB);
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
        Clear(); // Reset the new buffer
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

// TODO: these methods each copy and create the metadata object. Since 
// they are called together, this could be made more efficient by 
// returning a struct with the metadata and the values.
unsigned BufferManager::GetImageWidth(const void* ptr) const {
   if (!useV2_) 
      throw CMMError("GetImageWidth(ptr) only supported with V2 buffer");
   Metadata md;
   if (v2Buffer_->ExtractCorrespondingMetadata(ptr, md) != DEVICE_OK)
      throw CMMError("Failed to extract metadata for image width");
   std::string sVal = md.GetSingleTag(MM::g_Keyword_Metadata_Width).GetValue();
   return static_cast<unsigned>(atoi(sVal.c_str()));
}

unsigned BufferManager::GetImageHeight(const void* ptr) const {
   if (!useV2_) 
      throw CMMError("GetImageHeight(ptr) only supported with V2 buffer");
   Metadata md;
   if (v2Buffer_->ExtractCorrespondingMetadata(ptr, md) != DEVICE_OK)
      throw CMMError("Failed to extract metadata for image height");
   std::string sVal = md.GetSingleTag(MM::g_Keyword_Metadata_Height).GetValue();
   return static_cast<unsigned>(atoi(sVal.c_str()));
}

unsigned BufferManager::GetBytesPerPixelFromType(const std::string& pixelType) const {
   if (pixelType == MM::g_Keyword_PixelType_GRAY8)
      return 1;
   else if (pixelType == MM::g_Keyword_PixelType_GRAY16)
      return 2;
   else if (pixelType == MM::g_Keyword_PixelType_GRAY32 ||
            pixelType == MM::g_Keyword_PixelType_RGB32)
      return 4;
   else if (pixelType == MM::g_Keyword_PixelType_RGB64)
      return 8;
   throw CMMError("Unknown pixel type for bytes per pixel");
}

unsigned BufferManager::GetComponentsFromType(const std::string& pixelType) const {
   if (pixelType == MM::g_Keyword_PixelType_GRAY8 ||
       pixelType == MM::g_Keyword_PixelType_GRAY16 ||
       pixelType == MM::g_Keyword_PixelType_GRAY32)
      return 1;
   else if (pixelType == MM::g_Keyword_PixelType_RGB32 ||
            pixelType == MM::g_Keyword_PixelType_RGB64)
      return 4;
   throw CMMError("Unknown pixel type for number of components");
}

unsigned BufferManager::GetBytesPerPixel(const void* ptr) const {
   if (!useV2_) 
      throw CMMError("GetBytesPerPixel(ptr) only supported with V2 buffer");
   Metadata md;
   if (v2Buffer_->ExtractCorrespondingMetadata(ptr, md) != DEVICE_OK)
      throw CMMError("Failed to extract metadata for bytes per pixel");
   std::string pixelType = md.GetSingleTag(MM::g_Keyword_PixelType).GetValue();
   return GetBytesPerPixelFromType(pixelType);
}

unsigned BufferManager::GetNumberOfComponents(const void* ptr) const {
   if (!useV2_) 
      throw CMMError("GetNumberOfComponents(ptr) only supported with V2 buffer");
   Metadata md;
   if (v2Buffer_->ExtractCorrespondingMetadata(ptr, md) != DEVICE_OK)
      throw CMMError("Failed to extract metadata for number of components");
   std::string pixelType = md.GetSingleTag(MM::g_Keyword_PixelType).GetValue();
   return GetComponentsFromType(pixelType);
}

long BufferManager::GetImageBufferSize(const void* ptr) const {
   if (!useV2_) 
      throw CMMError("GetImageBufferSize(ptr) only supported with V2 buffer");
   return static_cast<long>(v2Buffer_->GetDataSize(ptr));
}

bool BufferManager::SetOverwriteData(bool overwrite) {
    if (useV2_) {
        return v2Buffer_->SetOverwriteData(overwrite) == DEVICE_OK;
    } else {
        // CircularBuffer doesn't have this functionality
        return false;
    }
}

bool BufferManager::AcquireWriteSlot(const char* deviceLabel, size_t dataSize, unsigned width, unsigned height, 
    unsigned byteDepth, unsigned nComponents, size_t additionalMetadataSize,
    void** dataPointer, void** additionalMetadataPointer,
    Metadata* pInitialMetadata) {
   if (!useV2_) {
      // Not supported for circular buffer
      return false;
   }

   // Initialize metadata with either provided metadata or create empty
   Metadata md = (pInitialMetadata != nullptr) ? *pInitialMetadata : Metadata();
   
   // Add in width, height, byteDepth, and nComponents to the metadata so that when 
   // images are retrieved from the buffer, the data can be interpreted correctly
   PopulateMetadata(md, deviceLabel, width, height, byteDepth, nComponents);
   
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

const void* BufferManager::GetLastImageFromDevice(const std::string& deviceLabel) throw (CMMError) {
    if (!useV2_) {
        throw CMMError("V2 buffer must be enabled for device-specific image access");
    }
    Metadata md;
    return GetLastImageMDFromDevice(deviceLabel, md);
}

const void* BufferManager::GetLastImageMDFromDevice(const std::string& deviceLabel, Metadata& md) throw (CMMError) {
    if (!useV2_) {
        throw CMMError("V2 buffer must be enabled for device-specific image access");
    }
    
    const void* basePtr = v2Buffer_->PeekLastDataReadPointerFromDevice(deviceLabel, md);
    if (basePtr == nullptr) {
        throw CMMError("No image found for device: " + deviceLabel, MMERR_InvalidContents);
    }
    return basePtr;
}

bool BufferManager::IsPointerInBuffer(const void* ptr) const throw (CMMError) {
    if (!useV2_) {
        throw CMMError("IsPointerInBuffer is only supported with V2 buffer enabled");
    }
    
    if (v2Buffer_ == nullptr) {
        throw CMMError("V2 buffer is null");
    }

    return v2Buffer_->IsPointerInBuffer(ptr);
}

DataBuffer* BufferManager::GetV2Buffer() const {
    if (!useV2_) {
        return nullptr;
    }
    return v2Buffer_;
}
