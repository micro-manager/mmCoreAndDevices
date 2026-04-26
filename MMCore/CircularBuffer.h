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

#include <cstddef>
#include <memory>
#include <mutex>
#include <string_view>
#include <vector>

namespace mmcore {
namespace internal {

class ThreadPool;
class TaskSet_CopyMemory;


class CircularBuffer
{
public:
   CircularBuffer(std::size_t memorySizeMB);
   ~CircularBuffer();

   int SetOverwriteData(bool overwrite);

   std::size_t GetMemorySizeMB() const { return memorySizeMB_; }

   bool Initialize(std::size_t frameSize);
   std::size_t GetSize() const;
   std::size_t GetFreeSize() const;
   std::size_t GetRemainingImageCount() const;

   bool InsertImage(const unsigned char* pixArray, std::size_t frameSize,
      std::string_view serializedMetadata) MMCORE_LEGACY_THROW(CMMError);
   const unsigned char* GetTopImage() const;
   const unsigned char* GetNextImage();
   const FrameBuffer* GetTopImageBuffer() const;
   const FrameBuffer* GetNthFromTopImageBuffer(std::size_t n) const;
   const FrameBuffer* GetNextImageBuffer();
   void Clear();

   bool Overflow() const {std::lock_guard<std::mutex> guard(bufferLock_); return overflow_;}

private:
   void ClearLocked();

   // Serializes InsertImage calls so that the pixel copy can occur
   // without holding bufferLock_.
   mutable std::mutex insertLock_;

   // Guards all mutable state below except where noted.
   mutable std::mutex bufferLock_;

   std::size_t frameSize_;

   // Invariants:
   // 0 <= saveIndex_ <= insertIndex_
   // insertIndex_ - saveIndex_ <= frameArray_.size()
   std::size_t insertIndex_;
   std::size_t saveIndex_;

   bool overflow_;
   bool overwriteData_;
   std::vector<FrameBuffer> frameArray_;

   // Effectively const after construction.
   std::size_t memorySizeMB_;
   std::shared_ptr<ThreadPool> threadPool_;
   std::shared_ptr<TaskSet_CopyMemory> tasksMemCopy_;
};

} // namespace internal
} // namespace mmcore
