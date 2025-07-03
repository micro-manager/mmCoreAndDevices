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
   threadPool_(std::make_shared<ThreadPool>()),
   tasksMemCopy_(std::make_shared<TaskSet_CopyMemory>(threadPool_))
{
}

CircularBuffer::~CircularBuffer() {}

bool CircularBuffer::Initialize(unsigned channels, unsigned int w, unsigned int h, unsigned int pixDepth)
{
   MMThreadGuard guard(g_bufferLock);
   imageNumbers_.clear();
   startTime_ = std::chrono::steady_clock::now();

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
      unsigned long frameSizeBytes = width_ * height_ * pixDepth_ * numChannels_;
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
   startTime_ = std::chrono::steady_clock::now();
   imageNumbers_.clear();
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

static std::string FormatLocalTime(std::chrono::time_point<std::chrono::system_clock> tp) {
   using namespace std::chrono;
   auto us = duration_cast<microseconds>(tp.time_since_epoch());
   auto secs = duration_cast<seconds>(us);
   auto whole = duration_cast<microseconds>(secs);
   auto frac = static_cast<int>((us - whole).count());

   // As of C++14/17, it is simpler (and probably faster) to use C functions for
   // date-time formatting

   std::time_t t(secs.count()); // time_t is seconds on platforms we support
   std::tm *ptm;
#ifdef _WIN32 // Windows localtime() is documented thread-safe
   ptm = std::localtime(&t);
#else // POSIX has localtime_r()
   std::tm tmstruct;
   ptm = localtime_r(&t, &tmstruct);
#endif

   // Format as "yyyy-mm-dd hh:mm:ss.uuuuuu" (26 chars)
   const char *timeFmt = "%Y-%m-%d %H:%M:%S";
   char buf[32];
   std::size_t len = std::strftime(buf, sizeof(buf), timeFmt, ptm);
   std::snprintf(buf + len, sizeof(buf) - len, ".%06d", frac);
   return buf;
}

/**
* Inserts a single image in the buffer.
*/
bool CircularBuffer::InsertImage(const unsigned char* pixArray, unsigned int width, unsigned int height, unsigned int byteDepth, const Metadata* pMd) MMCORE_LEGACY_THROW(CMMError)
{
   return InsertMultiChannel(pixArray, 1, width, height, byteDepth, pMd);
}

/**
* Inserts a single image, possibly with multiple channels, but with 1 component, in the buffer.
*/
bool CircularBuffer::InsertMultiChannel(const unsigned char* pixArray, unsigned int numChannels, unsigned int width, unsigned int height, unsigned int byteDepth, const Metadata* pMd) MMCORE_LEGACY_THROW(CMMError)
{
   return InsertMultiChannel(pixArray, numChannels, width, height, byteDepth, 1, pMd);
}

/**
* Inserts a single image, possibly with multiple components, in the buffer.
*/
bool CircularBuffer::InsertImage(const unsigned char* pixArray, unsigned int width, unsigned int height, unsigned int byteDepth, unsigned int nComponents, const Metadata* pMd) MMCORE_LEGACY_THROW(CMMError)
{
    return InsertMultiChannel(pixArray, 1, width, height, byteDepth, nComponents, pMd);
}
 
/**
* Inserts a multi-channel frame in the buffer.
*/
bool CircularBuffer::InsertMultiChannel(const unsigned char* pixArray, unsigned int numChannels, unsigned int width, unsigned int height, unsigned int byteDepth, unsigned int nComponents, const Metadata* pMd) MMCORE_LEGACY_THROW(CMMError)
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
          overflow_ = true;
          return false;
       }
    }
 
    for (unsigned i=0; i<numChannels; i++)
    {
       Metadata md;
       {
          MMThreadGuard guard(g_bufferLock);
          // we assume that all buffers are pre-allocated
          pImg = frameArray_[insertIndex_ % frameArray_.size()].FindImage(i);
          if (!pImg)
             return false;
 
          if (pMd)
          {
             // TODO: the same metadata is inserted for each channel ???
             // Perhaps we need to add specific tags to each channel
             md = *pMd;
          }

         std::string cameraName = md.GetSingleTag(MM::g_Keyword_Metadata_CameraLabel).GetValue();
         if (imageNumbers_.end() == imageNumbers_.find(cameraName))
         {
            imageNumbers_[cameraName] = 0;
         }

         // insert image number. 
         md.put(MM::g_Keyword_Metadata_ImageNumber, CDeviceUtils::ConvertToString(imageNumbers_[cameraName]));
         ++imageNumbers_[cameraName];
      }

      if (!md.HasTag(MM::g_Keyword_Elapsed_Time_ms))
      {
         // if time tag was not supplied by the camera insert current timestamp
         using namespace std::chrono;
         auto elapsed = steady_clock::now() - startTime_;
         md.PutImageTag(MM::g_Keyword_Elapsed_Time_ms,
            std::to_string(duration_cast<milliseconds>(elapsed).count()));
      }

      // Note: It is not ideal to use local time. I think this tag is rarely
      // used. Consider replacing with UTC (micro)seconds-since-epoch (with
      // different tag key) after addressing current usage.
      auto now = std::chrono::system_clock::now();
      md.PutImageTag(MM::g_Keyword_Metadata_TimeInCore, FormatLocalTime(now));

      md.PutImageTag(MM::g_Keyword_Metadata_Width, width);
      md.PutImageTag(MM::g_Keyword_Metadata_Height, height);
      if (byteDepth == 1)
         md.PutImageTag(MM::g_Keyword_PixelType, MM::g_Keyword_PixelType_GRAY8);
      else if (byteDepth == 2)
         md.PutImageTag(MM::g_Keyword_PixelType, MM::g_Keyword_PixelType_GRAY16);
      else if (byteDepth == 4)
      {
         if (nComponents == 1)
            md.PutImageTag(MM::g_Keyword_PixelType, MM::g_Keyword_PixelType_GRAY32);
         else
            md.PutImageTag(MM::g_Keyword_PixelType, MM::g_Keyword_PixelType_RGB32);
      }
      else if (byteDepth == 8)
         md.PutImageTag(MM::g_Keyword_PixelType, MM::g_Keyword_PixelType_RGB64);
      else
         md.PutImageTag(MM::g_Keyword_PixelType, MM::g_Keyword_PixelType_Unknown);

      pImg->SetMetadata(md);
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

const unsigned char* CircularBuffer::GetNextImage()
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
