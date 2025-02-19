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

#include "../MMDevice/DeviceUtils.h"

#include <chrono>
#include <cstdio>
#include <ctime>
#include <memory>
#include <string>

#ifdef _MSC_VER
#pragma warning(disable: 4290) // 'C++ exception specification ignored'
#endif

#if defined(__GNUC__) && !defined(__clang__)
// 'dynamic exception specifications are deprecated in C++11 [-Wdeprecated]'
#pragma GCC diagnostic ignored "-Wdeprecated"
#endif

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
   memorySizeMB_(memorySizeMB), 
   overflow_(false),
   overwriteData_(false),
   threadPool_(std::make_shared<ThreadPool>()),
   tasksMemCopy_(std::make_shared<TaskSet_CopyMemory>(threadPool_))
{
}

CircularBuffer::~CircularBuffer() {}

bool CircularBuffer::Initialize(unsigned channels, unsigned int w, unsigned int h, unsigned int pixDepth)
{
   MMThreadGuard guard(g_bufferLock);

   bool ret = true;
   try
   {
      if (w == 0 || h==0 || pixDepth == 0 || channels == 0)
         return false; // does not make sense

      if (w == width_ && height_ == h && pixDepth_ == pixDepth && channels == numChannels_)
         if (frameArray_.size() > 0)
            return true; // nothing to change

      width_ = w;
      height_ = h;
      pixDepth_ = pixDepth;
      numChannels_ = channels;

      insertIndex_ = 0;
      saveIndex_ = 0;
      overflow_ = false;

      // calculate the size of the entire buffer array once all images get allocated
      // the actual size at the time of the creation is going to be less, because
      // images are not allocated until pixels become available
      unsigned long frameSizeBytes = GetImageSizeBytes();
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

      for (unsigned long i=0; i<frameArray_.size(); i++)
         frameArray_[i].Clear();

      // allocate buffers  - could conceivably throw an out-of-memory exception
      frameArray_.resize(cbSize);
      for (unsigned long i=0; i<frameArray_.size(); i++)
      {
         frameArray_[i].Resize(w, h, pixDepth);
         frameArray_[i].Preallocate(numChannels_);
      }
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
   MMThreadGuard guard(g_bufferLock); 
   insertIndex_=0; 
   saveIndex_=0; 
   overflow_ = false;
}

unsigned long CircularBuffer::GetSize() const
{
   MMThreadGuard guard(g_bufferLock);
   return (unsigned long)frameArray_.size();
}

unsigned long CircularBuffer::GetFreeSize() const
{
   MMThreadGuard guard(g_bufferLock);
   long freeSize = (long)frameArray_.size() - (insertIndex_ - saveIndex_);
   if (freeSize < 0)
      return 0;
   else
      return (unsigned long)freeSize;
}

unsigned long CircularBuffer::GetRemainingImageCount() const
{
   MMThreadGuard guard(g_bufferLock);
   return (unsigned long)(insertIndex_ - saveIndex_);
}

 
/**
* Inserts a multi-channel frame in the buffer.
*/
bool CircularBuffer::InsertMultiChannel(const unsigned char* pixArray, unsigned int numChannels, unsigned int width, unsigned int height, 
      unsigned int byteDepth, const Metadata* pMd) throw (CMMError)
{
    MMThreadGuard insertGuard(g_insertLock);
 
    mm::ImgBuffer* pImg;
    unsigned long singleChannelSize = (unsigned long)width * height * byteDepth;
 
    {
       MMThreadGuard guard(g_bufferLock);
 
       // check image dimensions
       if (width != width_ || height != height_ || byteDepth != pixDepth_)
          throw CMMError("Incompatible image dimensions in the circular buffer", MMERR_CircularBufferIncompatibleImage);
 
       bool overflowed = (insertIndex_ - saveIndex_) >= static_cast<long>(frameArray_.size());
       if (overflowed) {
         if (!overwriteData_) {
            Clear();
         } else {
            overflow_ = true;
            return false;
         }
       }
    }
 
    for (unsigned i=0; i<numChannels; i++)
    {
       {
          MMThreadGuard guard(g_bufferLock);
          // we assume that all buffers are pre-allocated
          pImg = frameArray_[insertIndex_ % frameArray_.size()].FindImage(i);
          if (!pImg)
             return false;
 
          if (pMd)
          {
             // Store the metadata as-is without modification
             pImg->SetMetadata(*pMd);
          }
       }
       //pImg->SetPixels(pixArray + i * singleChannelSize);
      // TODO: In MMCore the ImgBuffer::GetPixels() returns const pointer.
      //       It would be better to have something like ImgBuffer::GetPixelsRW() in MMDevice.
      //       Or even better - pass tasksMemCopy_ to ImgBuffer constructor
      //       and utilize parallel copy also in single snap acquisitions.
       tasksMemCopy_->MemCopy((void*)pImg->GetPixels(),
            pixArray + i * singleChannelSize, singleChannelSize);
   }

   {
      MMThreadGuard guard(g_bufferLock);

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
   const mm::ImgBuffer* img = GetNthFromTopImageBuffer(0, 0);
   if (!img)
      return 0;
   return img->GetPixels();
}

const mm::ImgBuffer* CircularBuffer::GetTopImageBuffer(unsigned channel) const
{
   return GetNthFromTopImageBuffer(0, channel);
}

const mm::ImgBuffer* CircularBuffer::GetNthFromTopImageBuffer(unsigned long n) const
{
   return GetNthFromTopImageBuffer(static_cast<long>(n), 0);
}

const mm::ImgBuffer* CircularBuffer::GetNthFromTopImageBuffer(long n,
      unsigned channel) const
{
   MMThreadGuard guard(g_bufferLock);

   long availableImages = insertIndex_ - saveIndex_;
   if (n + 1 > availableImages)
      return 0;

   long targetIndex = insertIndex_ - n - 1L;
   while (targetIndex < 0)
      targetIndex += (long) frameArray_.size();
   targetIndex %= frameArray_.size();

   return frameArray_[targetIndex].FindImage(channel);
}

const unsigned char* CircularBuffer::PopNextImage()
{
   const mm::ImgBuffer* img = GetNextImageBuffer(0);
   if (!img)
      return 0;
   return img->GetPixels();
}

const mm::ImgBuffer* CircularBuffer::GetNextImageBuffer(unsigned channel)
{
   MMThreadGuard guard(g_bufferLock);

   long availableImages = insertIndex_ - saveIndex_;
   if (availableImages < 1)
      return 0;

   long targetIndex = saveIndex_ % frameArray_.size();
   ++saveIndex_;
   return frameArray_[targetIndex].FindImage(channel);
}
