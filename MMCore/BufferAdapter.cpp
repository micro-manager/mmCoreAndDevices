#include "BufferAdapter.h"

// For demonstration, we assume DEVICE_OK and DEVICE_ERR macros are defined in MMCore.h or an included error header.
#ifndef DEVICE_OK
   #define DEVICE_OK 0
#endif
#ifndef DEVICE_ERR
   #define DEVICE_ERR -1
#endif

BufferAdapter::BufferAdapter(bool useV2Buffer, unsigned int memorySizeMB)
   : useV2_(useV2Buffer), circBuffer_(nullptr), v2Buffer_(nullptr)
{
   if (useV2_) {
      // Create a new v2 buffer with a total size of memorySizeMB megabytes.
      // Multiply by (1 << 20) to convert megabytes to bytes.
      size_t bytes = memorySizeMB * (1 << 20);
      v2Buffer_ = new DataBuffer(bytes, "DEFAULT");
   } else {
      circBuffer_ = new CircularBuffer(memorySizeMB);
      // Optionally, perform any necessary initialization for the circular buffer.
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

bool BufferAdapter::InsertImage(const MM::Device *caller, const unsigned char* buf, 
      unsigned width, unsigned height, unsigned byteDepth, unsigned nComponents, const char* serializedMetadata)
{
   if (useV2_) {
      // For demonstration, assume that for a simple image the number of bytes is width * height * byteDepth * nComponents.
      size_t dataSize = width * height * byteDepth * nComponents;
      int res = v2Buffer_->InsertData(caller, buf, dataSize, serializedMetadata);
      return (res == 0);
   } else {
      // For now, pass a null metadata pointer.
      return circBuffer_->InsertImage(buf, width, height, byteDepth, nComponents, nullptr);
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
      return 0; // TODO: need to implement this
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
      // In this basic implementation, we call ReleaseBuffer with the known buffer name.
      v2Buffer_->ReleaseBuffer("DEFAULT");
   } else {
      circBuffer_->Clear();
   }
} 