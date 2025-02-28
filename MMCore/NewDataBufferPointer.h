///////////////////////////////////////////////////////////////////////////////
// FILE:          NewDataBufferPointer.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     MMCore
//-----------------------------------------------------------------------------
// DESCRIPTION:   A read-only wrapper class for accessing image data and metadata
//                from a buffer slot. Provides safe access to image data by
//                automatically releasing read access when the object is destroyed.
//                Includes methods for retrieving pixel data and associated
//                metadata..
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
// AUTHOR:        Henry Pinkard,  2/16/2025
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "MMCore.h"
#include "../MMDevice/ImageMetadata.h"
#include <mutex>
#include <random>

// This is needed for SWIG Java wrapping to differentiate its void*
// from the void* that MMCore uses for returning data
typedef const void* BufferDataPointerVoidStar; 

/// A read-only wrapper for accessing image data and metadata from a buffer slot.
/// Automatically releases the read access when destroyed.
class BufferDataPointer {

public:

   BufferDataPointer(BufferManager* bufferManager, DataPtr ptr)
      : bufferManager_(bufferManager), ptr_(ptr), mutex_()
   {
      // check for null pointer
      if (!ptr_) {
         throw CMMError("Pointer is null");
      }
      // check for v2 buffer use
      if (!bufferManager_->IsUsingNewDataBuffer()) {
         throw CMMError("V2 buffer must be enabled for BufferDataPointer");
      }
      // throw an error if the pointer is not in the buffer
      if (!bufferManager_->IsPointerInNewDataBuffer(ptr_)) {
         throw CMMError("Pointer is not in the buffer");
      }
   }

   // Returns a pointer to the pixel data (read-only)
   BufferDataPointerVoidStar getData() const {
      if (!ptr_) {
         return nullptr;
      }
      return ptr_;
   }

   // Same as the above method, but this get wrapped by SWIG differently
   // to return the actual
   DataPtr getDataPointer() const {
      if (!ptr_) {
         return nullptr;
      }
      return ptr_;
   }

   void getImageProperties(int& width, int& height, int& byteDepth, int& nComponents) throw (CMMError) {
      if (!bufferManager_ || !ptr_) {
         throw CMMError("Invalid buffer manager or pointer");
      }
      Metadata md;
      bufferManager_->ExtractMetadata(ptr_, md);
      CMMCore::parseImageMetadata(md, width, height, byteDepth, nComponents);
   }

   // Fills the provided Metadata object with metadata extracted from the pointer.
   // It encapsulates calling the core API function that copies metadata from the buffer.
   void getMetadata(Metadata &md) const {
      if (bufferManager_ && ptr_) {
         bufferManager_->ExtractMetadata(ptr_, md);
      }
   }

   // Destructor: releases the read access to the pointer if not already released
   ~BufferDataPointer() {
      release();
   }

  
   // Explicitly release the pointer before destruction if needed
   void release() {
      std::lock_guard<std::mutex> lock(mutex_);
      if (bufferManager_ && ptr_) {
         
         int ret = bufferManager_->ReleaseReadAccess(ptr_);
         if (ret != DEVICE_OK) {
            throw CMMError("Failed to release read access to buffer");
         }
         ptr_ = nullptr;  // Mark as released
         
      }
   }

   unsigned getSizeBytes() const {
      if (bufferManager_ && ptr_) {
         return bufferManager_->GetDataSize(ptr_);
      }
      return 0;
   }

private:
   // Disable copy semantics to avoid double releasing the pointer
   BufferDataPointer(const BufferDataPointer&);
   BufferDataPointer& operator=(const BufferDataPointer&);

   BufferManager* bufferManager_; 
   const void* ptr_; 
   mutable std::mutex mutex_;
};
