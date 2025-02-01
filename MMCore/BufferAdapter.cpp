#include "BufferAdapter.h"

// For demonstration, we assume DEVICE_OK and DEVICE_ERR macros are defined in MMCore.h or an included error header.
#ifndef DEVICE_OK
   #define DEVICE_OK 0
#endif
#ifndef DEVICE_ERR
   #define DEVICE_ERR -1
#endif

const char* const BufferAdapter::DEFAULT_V2_BUFFER_NAME = "DEFAULT_BUFFER";

BufferAdapter::BufferAdapter(bool useV2Buffer, unsigned int memorySizeMB)
   : useV2_(useV2Buffer), circBuffer_(nullptr), v2Buffer_(nullptr)
{
   if (useV2_) {
      // Create a new v2 buffer directly with MB
      v2Buffer_ = new DataBuffer(memorySizeMB, DEFAULT_V2_BUFFER_NAME);
   } else {
      circBuffer_ = new CircularBuffer(memorySizeMB);
   }
}

BufferAdapter::~BufferAdapter()
{
   if (useV2_) {
      if (v2Buffer_) {
         delete v2Buffer_;
      }
   } else {
      if (circBuffer_) {
         delete circBuffer_;
      }
   }
}

bool BufferAdapter::InsertImage(const unsigned char* buf, 
      unsigned width, unsigned height, unsigned byteDepth, Metadata* pMd) {
   if (useV2_) {
      // Implement logic for v2Buffer if available
      return false; // Placeholder
   } else {
      return circBuffer_->InsertImage(buf, width, height, byteDepth, pMd);
   }
}

bool BufferAdapter::InsertImage(const unsigned char *buf, unsigned width, unsigned height, 
                               unsigned byteDepth, unsigned nComponents, Metadata *pMd) {
   if (useV2_) {
      // Implement logic for v2Buffer if available
      return false; // Placeholder
   } else {
      return circBuffer_->InsertImage(buf, width, height, byteDepth, nComponents, pMd);
   }
}

const unsigned char* BufferAdapter::GetTopImage() const
{
   if (useV2_) {
      // Minimal support: the v2Buffer currently does not expose GetTopImage.
      return nullptr;
   } else {
      return circBuffer_->GetTopImage();
   }
}

const unsigned char* BufferAdapter::GetNextImage()
{
   if (useV2_) {
      // Minimal support: return nullptr since v2Buffer does not provide next image retrieval.
      return nullptr;
   } else {
      return circBuffer_->GetNextImage();
   }
}

const mm::ImgBuffer* BufferAdapter::GetNthFromTopImageBuffer(unsigned long n) const
{
   if (useV2_) {
      // Implement logic for v2Buffer if available
      return nullptr; // Placeholder
   } else {
      return circBuffer_->GetNthFromTopImageBuffer(n);
   }
}

const mm::ImgBuffer* BufferAdapter::GetNextImageBuffer(unsigned channel)
{
   if (useV2_) {
      // Implement logic for v2Buffer if available
      return nullptr; // Placeholder
   } else {
      return circBuffer_->GetNextImageBuffer(channel);
   }
}

bool BufferAdapter::Initialize(unsigned numChannels, unsigned width, unsigned height, unsigned bytesPerPixel)
{
   if (useV2_) {
      // Implement initialization logic for v2Buffer
      return true; // Placeholder
   } else {
      return circBuffer_->Initialize(numChannels, width, height, bytesPerPixel);
   }
}

unsigned BufferAdapter::GetMemorySizeMB() const
{
   if (useV2_) {
      return v2Buffer_->GetMemorySizeMB();
   } else {
      return circBuffer_->GetMemorySizeMB();
   }
}

long BufferAdapter::GetRemainingImageCount() const
{
   if (useV2_) {
      return 0; // TODO: need to implement this
   } else {
      return circBuffer_->GetRemainingImageCount();
   }
}

void BufferAdapter::Clear()
{
   if (useV2_) {
      v2Buffer_->ReleaseBuffer(DEFAULT_V2_BUFFER_NAME);
   } else {
      circBuffer_->Clear();
   }
}

bool BufferAdapter::InsertMultiChannel(const unsigned char *buf, unsigned numChannels, unsigned width, 
                                       unsigned height, unsigned byteDepth, Metadata *pMd) {
   if (useV2_) {
      // Implement logic for v2Buffer if available
      return false; // Placeholder
   } else {
      return circBuffer_->InsertMultiChannel(buf, numChannels, width, height, byteDepth, pMd);
   }
}

bool BufferAdapter::InsertMultiChannel(const unsigned char *buf, unsigned numChannels, unsigned width, 
                                       unsigned height, unsigned byteDepth, unsigned nComponents, Metadata *pMd) {
   if (useV2_) {
      // Implement logic for v2Buffer if available
      return false; // Placeholder
   } else {
      return circBuffer_->InsertMultiChannel(buf, numChannels, width, height, byteDepth, nComponents, pMd);
   }
}

long BufferAdapter::GetSize() const
{
   if (useV2_) {
      // Implement logic for v2Buffer if available
      return 0; // Placeholder
   } else {
      return circBuffer_->GetSize();
   }
}

long BufferAdapter::GetFreeSize() const
{
   if (useV2_) {
      // Implement logic for v2Buffer if available
      return 0; // Placeholder
   } else {
      return circBuffer_->GetFreeSize();
   }
}

bool BufferAdapter::Overflow() const
{
   if (useV2_) {
      // Implement logic for v2Buffer if available
      return false; // Placeholder
   } else {
      return circBuffer_->Overflow();
   }
}

const mm::ImgBuffer* BufferAdapter::GetTopImageBuffer(unsigned channel) const
{
   if (useV2_) {
      // Implement logic for v2Buffer if available
      return nullptr; // Placeholder
   } else {
      return circBuffer_->GetTopImageBuffer(channel);
   }
} 