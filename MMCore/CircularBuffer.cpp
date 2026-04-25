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

#include <memory>

namespace mmcore {
namespace internal {

const long long bytesInMB = 1 << 20;
const long adjustThreshold = LONG_MAX / 2;

// Maximum number of images allowed in the buffer. This arbitrary limit is code
// smell, but kept for now until careful checks for integer overflow and
// division by zero can be added.
const unsigned long maxCBSize = 10000000;

CircularBuffer::CircularBuffer(unsigned int memorySizeMB) :
   width_(0), 
   height_(0), 
   pixDepth_(0), 
   imageCounter_(0), 
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

bool CircularBuffer::Initialize(unsigned int w, unsigned int h, unsigned int pixDepth)
{
   std::lock_guard<std::mutex> guard(bufferLock_);

   bool ret = true;
   try
   {
      if (w == 0 || h==0 || pixDepth == 0)
         return false; // does not make sense

      if (w == width_ && height_ == h && pixDepth_ == pixDepth)
         if (frameArray_.size() > 0)
            return true; // nothing to change

      width_ = w;
      height_ = h;
      pixDepth_ = pixDepth;

      insertIndex_ = 0;
      saveIndex_ = 0;
      overflow_ = false;

      // calculate the size of the entire buffer array once all images get allocated
      // the actual size at the time of the creation is going to be less, because
      // images are not allocated until pixels become available
      unsigned long frameSizeBytes = width_ * height_ * pixDepth_;
      unsigned long cbSize = (unsigned long) ((memorySizeMB_ * bytesInMB) / frameSizeBytes);

      if (cbSize == 0) 
      {
         frameArray_.resize(0);
         return false; // memory footprint too small
      }

      // set a reasonable limit to circular buffer capacity 
      if (cbSize > maxCBSize)
         cbSize = maxCBSize; 

      // TODO: verify if we have enough RAM to satisfy this request

      // allocate buffers  - could conceivably throw an out-of-memory exception
      frameArray_.resize(cbSize);
      for (unsigned long i=0; i<frameArray_.size(); i++)
         frameArray_[i].Resize(w, h, pixDepth);
   }

   catch( ... /* std::bad_alloc& ex */)
   {
      frameArray_.resize(0);
      ret = false;
   }
   return ret;
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

unsigned long CircularBuffer::GetSize() const
{
   std::lock_guard<std::mutex> guard(bufferLock_);
   return (unsigned long)frameArray_.size();
}

unsigned long CircularBuffer::GetFreeSize() const
{
   std::lock_guard<std::mutex> guard(bufferLock_);
   long freeSize = (long)frameArray_.size() - (insertIndex_ - saveIndex_);
   if (freeSize < 0)
      return 0;
   else
      return (unsigned long)freeSize;
}

unsigned long CircularBuffer::GetRemainingImageCount() const
{
   std::lock_guard<std::mutex> guard(bufferLock_);
   return (unsigned long)(insertIndex_ - saveIndex_);
}

/**
* Inserts a single image, possibly with multiple components, in the buffer.
*/
bool CircularBuffer::InsertImage(const unsigned char* pixArray,
   unsigned int width, unsigned int height, unsigned int byteDepth, unsigned int nComponents,
   std::string_view serializedMetadata) MMCORE_LEGACY_THROW(CMMError)
{
    (void)nComponents;
    std::lock_guard<std::mutex> insertGuard(insertLock_);

    FrameBuffer* pImg;
    unsigned long singleChannelSize = (unsigned long)width * height * byteDepth;

    {
       std::lock_guard<std::mutex> guard(bufferLock_);

       if (overflow_)
          return false;

       // check image dimensions
       if (width != width_ || height != height_ || byteDepth != pixDepth_)
          throw CMMError("Incompatible image dimensions in the circular buffer", MMERR_CircularBufferIncompatibleImage);

       bool overflowed = (insertIndex_ - saveIndex_) >= static_cast<long>(frameArray_.size());
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
   tasksMemCopy_->MemCopy((void*)pImg->GetPixels(),
         pixArray, singleChannelSize);

   {
      std::lock_guard<std::mutex> guard(bufferLock_);

      imageCounter_++;
      insertIndex_++;
      if ((insertIndex_ - (long)frameArray_.size()) > adjustThreshold && (saveIndex_- (long)frameArray_.size()) > adjustThreshold)
      {
         // adjust buffer indices to avoid overflowing integer size
         insertIndex_ -= adjustThreshold;
         saveIndex_ -= adjustThreshold;
      }
   }

   return true;
}
 

const unsigned char* CircularBuffer::GetTopImage() const
{
   const FrameBuffer* img = GetNthFromTopImageBuffer(0);
   if (!img)
      return 0;
   return img->GetPixels();
}

const FrameBuffer* CircularBuffer::GetTopImageBuffer() const
{
   return GetNthFromTopImageBuffer(0);
}

const FrameBuffer* CircularBuffer::GetNthFromTopImageBuffer(unsigned long n) const
{
   std::lock_guard<std::mutex> guard(bufferLock_);

   const long ln = static_cast<long>(n);
   long availableImages = insertIndex_ - saveIndex_;
   if (ln + 1 > availableImages)
      return 0;

   long targetIndex = insertIndex_ - ln - 1L;
   while (targetIndex < 0)
      targetIndex += (long) frameArray_.size();
   targetIndex %= frameArray_.size();

   return &frameArray_[targetIndex];
}

const unsigned char* CircularBuffer::GetNextImage()
{
   const FrameBuffer* img = GetNextImageBuffer();
   if (!img)
      return 0;
   return img->GetPixels();
}

const FrameBuffer* CircularBuffer::GetNextImageBuffer()
{
   std::lock_guard<std::mutex> guard(bufferLock_);

   long availableImages = insertIndex_ - saveIndex_;
   if (availableImages < 1)
      return 0;

   long targetIndex = saveIndex_ % frameArray_.size();
   ++saveIndex_;
   return &frameArray_[targetIndex];
}

} // namespace internal
} // namespace mmcore
