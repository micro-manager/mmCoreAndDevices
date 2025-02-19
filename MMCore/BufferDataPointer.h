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

// This is needed for SWIG Java wrapping to differentiate its void*
// from the void* that MMCore uses for returning data
typedef const void* BufferDataPointerVoidStar; 

/// A read-only wrapper for accessing image data and metadata from a buffer slot.
/// Automatically releases the read access when destroyed.
class BufferDataPointer {

public:

   BufferDataPointer(BufferManager* bufferManager, DataPtr ptr)
      : bufferManager_(bufferManager), ptr_(ptr)
   {
      // check for v2 buffer use
      if (!bufferManager_->IsUsingV2Buffer()) {
         throw CMMError("V2 buffer must be enabled for BufferDataPointer");
      }
   }

   // Returns a pointer to the pixel data (read-only)
   BufferDataPointerVoidStar getData() const {
      if (!ptr_) {
         return nullptr;
      }
      return ptr_;
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
         try {
            bufferManager_->ReleaseReadAccess(ptr_);
            ptr_ = nullptr;  // Mark as released
         } catch (...) {
            // Release must not throw
         }
      }
   }

   // TODO if an when these are needed, make them a call a single functions that reads width and height together
   // unsigned getImageWidth() const {
   //    if (bufferManager_ && ptr_) {
   //       return bufferManager_->GetImageWidth(ptr_);
   //    }
   //    return 0;
   // }

   // unsigned getImageHeight() const {
   //    if (bufferManager_ && ptr_) {
   //       return bufferManager_->GetImageHeight(ptr_);
   //    }
   //    return 0;
   // }

   // unsigned getBytesPerPixel() const {
   //    if (bufferManager_ && ptr_) {
   //       return bufferManager_->GetBytesPerPixel(ptr_);
   //    }
   //    return 0;
   // }
   //
   // unsigned getNumberOfComponents() const {
   //    if (bufferManager_ && ptr_) {
   //       return bufferManager_->GetNumberOfComponents(ptr_);
   //    }
   //    return 0;
   // }

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