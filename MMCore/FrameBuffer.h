// AUTHOR:        Nenad Amodaj, nenad@amodaj.com
//
// COPYRIGHT:     2005 Nenad Amodaj
//                2005-2015 Regents of the University of California
//                2017 Open Imaging, Inc.
//
// LICENSE:       This file is free for use, modification and distribution and
//                is distributed under terms specified in the BSD license
//                This file is distributed in the hope that it will be useful,
//                but WITHOUT ANY WARRANTY; without even the implied warranty
//                of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//                IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//                CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//                INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.
//
// NOTE:          Imported from ADVI for use in Micro-Manager

#pragma once

#include <memory>
#include <string>
#include <string_view>

namespace mmcore {
namespace internal {

class FrameBuffer
{
   std::unique_ptr<unsigned char[]> pixels_;
   unsigned int width_ = 0;
   unsigned int height_ = 0;
   unsigned int pixDepth_ = 0;
   std::string serializedMetadata_;

public:
   FrameBuffer() = default;
   FrameBuffer(unsigned xSize, unsigned ySize, unsigned pixDepth);

   FrameBuffer(FrameBuffer&&) = default;
   FrameBuffer& operator=(FrameBuffer&&) = default;
   FrameBuffer(const FrameBuffer&) = delete;
   FrameBuffer& operator=(const FrameBuffer&) = delete;

   void SetPixels(const void* pixArray);
   const unsigned char* GetPixels() const;

   void Resize(unsigned xSize, unsigned ySize, unsigned pixDepth);

   void SetSerializedMetadata(std::string_view serialized);
   const std::string& GetSerializedMetadata() const {
      return serializedMetadata_;
   }
};

} // namespace internal
} // namespace mmcore
