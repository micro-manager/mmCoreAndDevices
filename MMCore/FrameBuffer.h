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

#include "ImageMetadata.h"

#include <memory>

namespace mm {

class ImgBuffer
{
   unsigned char* pixels_;
   unsigned int width_;
   unsigned int height_;
   unsigned int pixDepth_;
   Metadata metadata_;

public:
   ImgBuffer(unsigned xSize, unsigned ySize, unsigned pixDepth);
   ~ImgBuffer();

   unsigned int Width() const {return width_;}
   unsigned int Height() const {return height_;}
   unsigned int Depth() const {return pixDepth_;}
   void SetPixels(const void* pixArray);
   const unsigned char* GetPixels() const;

   void Resize(unsigned xSize, unsigned ySize, unsigned pixDepth);
   void Resize(unsigned xSize, unsigned ySize);

   void SetMetadata(const Metadata& md);
   const Metadata& GetMetadata() const {return metadata_;}

private:
   ImgBuffer& operator=(const ImgBuffer&);
};

// The FrameBuffer class wraps ImgBuffer (which is part of the MMCore API) for
// internal use. It was also previously part of a never-completed scheme to
// support multi-channel frames.
class FrameBuffer
{
   std::unique_ptr<ImgBuffer> buffer_; // May be empty
   unsigned int width_;
   unsigned int height_;
   unsigned int depth_;

public:
   FrameBuffer(unsigned xSize, unsigned ySize, unsigned byteDepth);
   FrameBuffer();

   void Resize(unsigned xSize, unsigned ySize, unsigned pixDepth);
   void Clear();
   void Preallocate();

   ImgBuffer* FindImage(unsigned channel) const;
   unsigned Width() const {return width_;}
   unsigned Height() const {return height_;}
   unsigned Depth() const {return depth_;}
};

} // namespace mm
