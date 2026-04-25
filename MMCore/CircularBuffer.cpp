///////////////////////////////////////////////////////////////////////////////
// FILE:          CircularBuffer.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     MMCore
//-----------------------------------------------------------------------------
// DESCRIPTION:   Generic implementation of the circular buffer. The buffer
//                allows only one thread to enter at a time by using a mutex lock.
//                This makes the buffer susceptible to race conditions if the
//                calling threads are mutually dependent.
//              
// COPYRIGHT:     University of California, San Francisco, 2007,
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
// AUTHOR:        Nenad Amodaj, nenad@amodaj.com, 01/05/2007
// 
#include "CircularBuffer.h"
#include "CoreUtils.h"

#include "TaskSet_CopyMemory.h"
#include "ThreadPool.h"

#include "DeviceUtils.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <new>

namespace mmcore {
namespace internal {

constexpr std::size_t bytesInMB = 1 << 20;
constexpr std::size_t adjustThreshold = std::numeric_limits<std::size_t>::max() / 2;

// Maximum number of images allowed in the buffer. This arbitrary limit is code
// smell, but kept for now until careful checks for integer overflow and
// division by zero can be added.
constexpr std::size_t maxCBSize = 10000000;

CircularBuffer::CircularBuffer(std::size_t memorySizeMB) :
   frameSize_(0),
   insertIndex_(0),
   saveIndex_(0),
   overflow_(false),
   overwriteData_(false),
   memorySizeMB_(memorySizeMB),
   threadPool_(std::make_shared<ThreadPool>()),
   tasksMemCopy_(std::make_shared<TaskSet_CopyMemory>(threadPool_))
{
}

CircularBuffer::~CircularBuffer() {}

int CircularBuffer::SetOverwriteData(bool overwrite) {
   std::lock_guard<std::mutex> guard(bufferLock_);
   overwriteData_ = overwrite;
   return DEVICE_OK;
}

bool CircularBuffer::Initialize(std::size_t frameSize)
{
   std::lock_guard<std::mutex> guard(bufferLock_);

   overflow_ = false;
   insertIndex_ = 0;
   saveIndex_ = 0;

   try
   {
      if (frameSize == 0)
      {
         frameSize_ = 0;
         return false;
      }

      const std::size_t cbSize = std::min(maxCBSize,
         (memorySizeMB_ * bytesInMB) / frameSize);

      if (cbSize == 0)
      {
         frameSize_ = frameSize;
         frameArray_.resize(0);
         return false; // memory footprint too small
      }

      if (frameSize == frameSize_ && frameArray_.size() == cbSize)
         return true;

      frameSize_ = frameSize;
      frameArray_.resize(cbSize);
      for (auto& frameBuf : frameArray_)
         frameBuf.Resize(frameSize_);
      return true;
   }
   catch (std::bad_alloc&)
   {
      frameArray_.resize(0);
      return false;
   }
}

void CircularBuffer::Clear()
{
   std::lock_guard<std::mutex> guard(bufferLock_);
   ClearLocked();
}

void CircularBuffer::ClearLocked()
{
   insertIndex_=0;
   saveIndex_=0;
   overflow_ = false;
}

std::size_t CircularBuffer::GetSize() const
{
   std::lock_guard<std::mutex> guard(bufferLock_);
   return frameArray_.size();
}

std::size_t CircularBuffer::GetFreeSize() const
{
   std::lock_guard<std::mutex> guard(bufferLock_);
   return frameArray_.size() - (insertIndex_ - saveIndex_);
}

std::size_t CircularBuffer::GetRemainingImageCount() const
{
   std::lock_guard<std::mutex> guard(bufferLock_);
   return insertIndex_ - saveIndex_;
}

/**
* Inserts a single image in the buffer.
*/
bool CircularBuffer::InsertImage(const unsigned char* pixArray,
   std::size_t frameSize,
   std::string_view serializedMetadata) MMCORE_LEGACY_THROW(CMMError)
{
    std::lock_guard<std::mutex> insertGuard(insertLock_);

    FrameBuffer* pImg;

    {
       std::lock_guard<std::mutex> guard(bufferLock_);

       if (overflow_)
          return false;

       if (frameSize != frameSize_)
          throw CMMError("Incompatible image size in the circular buffer", MMERR_CircularBufferIncompatibleImage);

       bool overflowed = (insertIndex_ - saveIndex_) >= frameArray_.size();
       if (overflowed) {
         if (overwriteData_) {
            ClearLocked();
         } else {
            overflow_ = true;
            return false;
         }
       }

       pImg = &frameArray_[insertIndex_ % frameArray_.size()];
    }

   pImg->SetSerializedMetadata(serializedMetadata);

   // TODO: In MMCore the FrameBuffer::GetPixels() returns const pointer.
   //       It would be better to have something like FrameBuffer::GetPixelsRW() in MMDevice.
   //       Or even better - pass tasksMemCopy_ to FrameBuffer constructor
   //       and utilize parallel copy also in single snap acquisitions.
   tasksMemCopy_->MemCopy((void*)pImg->GetPixels(), pixArray, frameSize);

   {
      std::lock_guard<std::mutex> guard(bufferLock_);

      insertIndex_++;
      // Periodically rebase indices to keep them from growing without bound.
      if (insertIndex_ > frameArray_.size() + adjustThreshold &&
          saveIndex_  > frameArray_.size() + adjustThreshold)
      {
         insertIndex_ -= adjustThreshold;
         saveIndex_   -= adjustThreshold;
      }
   }

   return true;
}
 

const unsigned char* CircularBuffer::GetTopImage() const
{
   const FrameBuffer* img = GetNthFromTopImageBuffer(0);
   if (!img)
      return nullptr;
   return img->GetPixels();
}

const FrameBuffer* CircularBuffer::GetTopImageBuffer() const
{
   return GetNthFromTopImageBuffer(0);
}

const FrameBuffer* CircularBuffer::GetNthFromTopImageBuffer(std::size_t n) const
{
   std::lock_guard<std::mutex> guard(bufferLock_);

   const std::size_t availableImages = insertIndex_ - saveIndex_;
   if (n >= availableImages)
      return nullptr;

   const std::size_t targetIndex = (insertIndex_ - n - 1) % frameArray_.size();
   return &frameArray_[targetIndex];
}

const unsigned char* CircularBuffer::GetNextImage()
{
   const FrameBuffer* img = GetNextImageBuffer();
   if (!img)
      return nullptr;
   return img->GetPixels();
}

const FrameBuffer* CircularBuffer::GetNextImageBuffer()
{
   std::lock_guard<std::mutex> guard(bufferLock_);

   if (insertIndex_ == saveIndex_)
      return nullptr;

   const std::size_t targetIndex = saveIndex_ % frameArray_.size();
   ++saveIndex_;
   return &frameArray_[targetIndex];
}

} // namespace internal
} // namespace mmcore
