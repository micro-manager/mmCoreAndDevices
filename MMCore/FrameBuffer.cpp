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

ImgBuffer::ImgBuffer(unsigned xSize, unsigned ySize, unsigned pixDepth) :
   pixels_(0), width_(xSize), height_(ySize), pixDepth_(pixDepth)
{
   pixels_ = new unsigned char[xSize * ySize * pixDepth];
   memset(pixels_, 0, xSize * ySize * pixDepth);
}

ImgBuffer::~ImgBuffer()
{
   delete[] pixels_;
}

const unsigned char* ImgBuffer::GetPixels() const
{
   return pixels_;
}

void ImgBuffer::SetPixels(const void* pix)
{
   memcpy((void*)pixels_, pix, width_ * height_ * pixDepth_);
}

void ImgBuffer::Resize(unsigned xSize, unsigned ySize, unsigned pixDepth)
{
   // re-allocate internal buffer if it is not big enough
   if (width_ * height_ * pixDepth_ < xSize * ySize * pixDepth)
   {
      delete[] pixels_;
      pixels_ = new unsigned char [xSize * ySize * pixDepth];
   }

   width_ = xSize;
   height_ = ySize;
   pixDepth_ = pixDepth;
}

void ImgBuffer::Resize(unsigned xSize, unsigned ySize)
{
   // re-allocate internal buffer if it is not big enough
   if (width_ * height_ < xSize * ySize)
   {
      delete[] pixels_;
      pixels_ = new unsigned char[xSize * ySize * pixDepth_];
   }

   width_ = xSize;
   height_ = ySize;

   memset(pixels_, 0, width_ * height_ * pixDepth_);
}

void ImgBuffer::SetMetadata(const Metadata& md)
{
   //metadata_ = md;
   // Serialize/Restore instead of =operator used to avoid object new/delete
   // issues across the DLL boundary (on Windows)
   // TODO: this is inefficient and should be revised
    metadata_.Restore(md.Serialize().c_str());
}


///////////////////////////////////////////////////////////////////////////////
// FrameBuffer class
///////////////////////////////////////////////////////////////////////////////

FrameBuffer::FrameBuffer(unsigned xSize, unsigned ySize, unsigned byteDepth)
{
   width_ = xSize;
   height_ = ySize;
   depth_ = byteDepth;
}

FrameBuffer::FrameBuffer()
{
   width_ = 0;
   height_ = 0;
   depth_ = 0;
}

void FrameBuffer::Clear()
{
   buffer_.reset();
}

void FrameBuffer::Preallocate()
{
   if (!buffer_) {
      buffer_ = std::make_unique<ImgBuffer>(width_, height_, depth_);
   }
}

void FrameBuffer::Resize(unsigned xSize, unsigned ySize, unsigned byteDepth)
{
   Clear();
   width_ = xSize;
   height_ = ySize;
   depth_ = byteDepth;
}

ImgBuffer* FrameBuffer::FindImage(unsigned channel) const
{
   if (channel > 0)
      return nullptr;
   return buffer_.get();
}

} // namespace internal
} // namespace mmcore
