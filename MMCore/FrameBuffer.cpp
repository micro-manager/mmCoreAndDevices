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

#include "FrameBuffer.h"

#include <cmath>
#include <cstring>

namespace mmcore {
namespace internal {

FrameBuffer::FrameBuffer(std::size_t size) :
   size_(size),
   pixels_(new unsigned char[size]())
{
}

const unsigned char* FrameBuffer::GetPixels() const
{
   return pixels_.get();
}

void FrameBuffer::SetPixels(const void* pix)
{
   memcpy(pixels_.get(), pix, size_);
}

void FrameBuffer::Resize(std::size_t size)
{
   // re-allocate internal buffer if it is not big enough
   if (size_ < size)
   {
      pixels_.reset(new unsigned char[size]());
   }
   size_ = size;
}

void FrameBuffer::SetSerializedMetadata(std::string_view serialized)
{
   serializedMetadata_.assign(serialized);
}

} // namespace internal
} // namespace mmcore
