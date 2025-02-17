///////////////////////////////////////////////////////////////////////////////
// FILE:          DataPointer.h
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

/// A read-only wrapper for accessing image data and metadata from a buffer slot.
/// Automatically releases the read access when destroyed.
class BufferDataPointer {

public:

   BufferDataPointer(DataBuffer* bufferv2, const void* ptr)
      : bufferv2_(bufferv2), ptr_(ptr)
   {
   }

   // Returns a pointer to the pixel data (read-only)
   const void* getPixels() const {
      if (!ptr_) {
         return nullptr;
      }
      return ptr_;
   }

   // Fills the provided Metadata object with metadata extracted from the pointer.
   // It encapsulates calling the core API function that copies metadata from the buffer.
   void getMetadata(Metadata &md) const {
      if (bufferv2_ && ptr_) {
         bufferv2_->ExtractCorrespondingMetadata(ptr_, md);
      }
   }

   // Explicitly release the pointer before destruction if needed
   void release() {
      std::lock_guard<std::mutex> lock(mutex_);
      if (bufferv2_ && ptr_) {
         try {
            bufferv2_->ReleaseDataReadPointer(ptr_);
            ptr_ = nullptr;  // Mark as released
         } catch (...) {
            // Release must not throw
         }
      }
   }

   // Destructor: releases the read access to the pointer if not already released
   ~BufferDataPointer() {
      release();
   }

   // Disable copy semantics to avoid double releasing the pointer.
   BufferDataPointer(const BufferDataPointer&) = delete;
   BufferDataPointer& operator=(const BufferDataPointer&) = delete;

private:
   DataBuffer* bufferv2_; 
   const void* ptr_; 
   mutable std::mutex mutex_;
}; 