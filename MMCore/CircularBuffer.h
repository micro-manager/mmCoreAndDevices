///////////////////////////////////////////////////////////////////////////////
// FILE:          CircularBuffer.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     MMCore
//-----------------------------------------------------------------------------
// DESCRIPTION:   Generic implementation of the circular buffer
//              
// COPYRIGHT:     University of California, San Francisco, 2007,
//                100X Imaging Inc, 2008
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

#pragma once

#include "Error.h"
#include "ErrorCodes.h"
#include "FrameBuffer.h"

#include "MMDevice.h"

#include <chrono>
#include <memory>
#include <mutex>
#include <vector>

namespace mmcore {
namespace internal {

class ThreadPool;
class TaskSet_CopyMemory;


class CircularBuffer
{
public:
   CircularBuffer(unsigned int memorySizeMB);
   ~CircularBuffer();

   int SetOverwriteData(bool overwrite);

   unsigned GetMemorySizeMB() const { return memorySizeMB_; }

   bool Initialize(unsigned int xSize, unsigned int ySize, unsigned int pixDepth);
   unsigned long GetSize() const;
   unsigned long GetFreeSize() const;
   unsigned long GetRemainingImageCount() const;

   unsigned int Width() const {std::lock_guard<std::mutex> guard(bufferLock_); return width_;}
   unsigned int Height() const {std::lock_guard<std::mutex> guard(bufferLock_); return height_;}
   unsigned int Depth() const {std::lock_guard<std::mutex> guard(bufferLock_); return pixDepth_;}

   bool InsertImage(const unsigned char* pixArray,
      unsigned int width, unsigned int height, unsigned int byteDepth, unsigned int nComponents,
      const Metadata* pMd) MMCORE_LEGACY_THROW(CMMError);
   const unsigned char* GetTopImage() const;
   const unsigned char* GetNextImage();
   const ImgBuffer* GetTopImageBuffer(unsigned channel) const;
   const ImgBuffer* GetNthFromTopImageBuffer(unsigned long n) const;
   const ImgBuffer* GetNthFromTopImageBuffer(long n, unsigned channel) const;
   const ImgBuffer* GetNextImageBuffer(unsigned channel);
   void Clear(); 

   bool Overflow() {std::lock_guard<std::mutex> guard(bufferLock_); return overflow_;}

private:
   void ClearLocked();

   // Serializes InsertImage calls so that the pixel copy can occur
   // without holding bufferLock_.
   mutable std::mutex insertLock_;

   // Guards all mutable state below except where noted.
   mutable std::mutex bufferLock_;

   unsigned int width_;
   unsigned int height_;
   unsigned int pixDepth_;
   long imageCounter_;
   std::chrono::time_point<std::chrono::steady_clock> startTime_;
   std::map<std::string, long> imageNumbers_;

   // Invariants:
   // 0 <= saveIndex_ <= insertIndex_
   // insertIndex_ - saveIndex_ <= frameArray_.size()
   long insertIndex_;
   long saveIndex_;

   bool overflow_;
   bool overwriteData_;
   std::vector<FrameBuffer> frameArray_;

   // Effectively const after construction.
   unsigned long memorySizeMB_;
   std::shared_ptr<ThreadPool> threadPool_;
   std::shared_ptr<TaskSet_CopyMemory> tasksMemCopy_;
};

} // namespace internal
} // namespace mmcore
