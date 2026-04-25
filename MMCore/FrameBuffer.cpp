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
   pixels_(new unsigned char[xSize * ySize * pixDepth]()),
   width_(xSize), height_(ySize), pixDepth_(pixDepth)
{
}

const unsigned char* ImgBuffer::GetPixels() const
{
   return pixels_.get();
}

void ImgBuffer::SetPixels(const void* pix)
{
   memcpy((void*)pixels_.get(), pix, width_ * height_ * pixDepth_);
}

void ImgBuffer::Resize(unsigned xSize, unsigned ySize, unsigned pixDepth)
{
   // re-allocate internal buffer if it is not big enough
   if (width_ * height_ * pixDepth_ < xSize * ySize * pixDepth)
   {
      pixels_.reset(new unsigned char[xSize * ySize * pixDepth]);
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
      pixels_.reset(new unsigned char[xSize * ySize * pixDepth_]);
   }

   width_ = xSize;
   height_ = ySize;

   memset(pixels_.get(), 0, width_ * height_ * pixDepth_);
}

void ImgBuffer::SetSerializedMetadata(std::string_view serialized)
{
   serializedMetadata_.assign(serialized);
}

} // namespace internal
} // namespace mmcore
