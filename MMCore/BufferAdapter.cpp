#include "BufferAdapter.h"
#include <mutex>

// For demonstration, we assume DEVICE_OK and DEVICE_ERR macros are defined in MMCore.h or an included error header.
#ifndef DEVICE_OK
   #define DEVICE_OK 0
#endif
#ifndef DEVICE_ERR
   #define DEVICE_ERR -1
#endif


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


BufferAdapter::BufferAdapter(bool useV2Buffer, unsigned int memorySizeMB)
   : useV2_(useV2Buffer), circBuffer_(nullptr), v2Buffer_(nullptr)
{
   if (useV2_) {
      v2Buffer_ = new DataBuffer(memorySizeMB);
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

const unsigned char* BufferAdapter::GetLastImage() const
{
   if (useV2_) {
      Metadata dummyMetadata;
      return v2Buffer_->PeekDataReadPointerAtIndex(0, nullptr, dummyMetadata);
      // TODO: ensure calling code releases the slot after use
   } else {
      return circBuffer_->GetTopImage();
   }

}

const unsigned char* BufferAdapter::PopNextImage()
{
   if (useV2_) {
      Metadata dummyMetadata;
      return v2Buffer_->PopNextDataReadPointer(dummyMetadata, nullptr, false);
      // TODO: ensure calling code releases the slot after use
   } else {
      return circBuffer_->PopNextImage();
   }
}

bool BufferAdapter::Initialize(unsigned numChannels, unsigned width, unsigned height, unsigned bytesPerPixel)
{
   startTime_ = std::chrono::steady_clock::now();  // Initialize start time
   imageNumbers_.clear();
   if (useV2_) {
      try {
         // Reinitialize the v2Buffer using its current allocated memory size.
         int ret = v2Buffer_->ReinitializeBuffer(v2Buffer_->GetMemorySizeMB());
         if (ret != DEVICE_OK)
            return false;
      } catch (const std::exception&) {
         // Optionally log the exception
         return false;
      }
      return true;
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
      return v2Buffer_->GetRemainingImageCount();
   } else {
      return circBuffer_->GetRemainingImageCount();
   }
}

void BufferAdapter::Clear()
{
   if (useV2_) {
      v2Buffer_->ReleaseBuffer();
   } else {
      circBuffer_->Clear();
   }
   // Reset image counters when buffer is cleared
   imageNumbers_.clear();
}

long BufferAdapter::GetSize(long imageSize) const
{
   if (useV2_) {
      return v2Buffer_->GetMemorySizeMB() * 1024 * 1024 / imageSize;
   } else {
      return circBuffer_->GetSize();
   }

}

long BufferAdapter::GetFreeSize(long imageSize) const
{
   if (useV2_) {
      return static_cast<long>(v2Buffer_->GetFreeMemory()) / imageSize;
   } else {
      return circBuffer_->GetFreeSize();
   }
}

bool BufferAdapter::Overflow() const
{
   if (useV2_) {
      return v2Buffer_->Overflow();
   } else {
      return circBuffer_->Overflow();
   }
}

void BufferAdapter::ProcessMetadata(Metadata& md, unsigned width, unsigned height, 
    unsigned byteDepth, unsigned nComponents) {
    // Track image numbers per camera
    {
        std::lock_guard<std::mutex> lock(imageNumbersMutex_);
        std::string cameraName = md.GetSingleTag(MM::g_Keyword_Metadata_CameraLabel).GetValue();
        if (imageNumbers_.end() == imageNumbers_.find(cameraName))
        {
            imageNumbers_[cameraName] = 0;
        }

        // insert image number
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
}

bool BufferAdapter::InsertImage(const unsigned char* buf, 
      unsigned width, unsigned height, unsigned byteDepth, Metadata* pMd) {
   return InsertMultiChannel(buf, 1, width, height, byteDepth, 1, pMd);
}

bool BufferAdapter::InsertImage(const unsigned char *buf, unsigned width, unsigned height, 
                               unsigned byteDepth, unsigned nComponents, Metadata *pMd) {
   return InsertMultiChannel(buf, 1, width, height, byteDepth, nComponents, pMd);
}


bool BufferAdapter::InsertMultiChannel(const unsigned char *buf, unsigned numChannels, unsigned width, 
                                       unsigned height, unsigned byteDepth, Metadata *pMd) {
   return InsertMultiChannel(buf, numChannels, width, height, byteDepth, 1, pMd);
}

bool BufferAdapter::InsertMultiChannel(const unsigned char* buf, unsigned numChannels, 
    unsigned width, unsigned height, unsigned byteDepth, unsigned nComponents, Metadata* pMd) {
    
    // Initialize metadata with either provided metadata or create empty
    Metadata md = (pMd != nullptr) ? *pMd : Metadata();

    // Process common metadata
    ProcessMetadata(md, width, height, byteDepth, nComponents);

    if (useV2_) {
      // All the data needed to interpret the image is in the metadata
      // This function will copy data and metadata into the buffer
      int ret = v2Buffer_->InsertData(buf, width * height * byteDepth *numChannels, &md);
      return ret == DEVICE_OK;
    } else {
        return circBuffer_->InsertMultiChannel(buf, numChannels, width, height, 
            byteDepth, &md);
    }

}

void* BufferAdapter::GetLastImageMD(unsigned channel, Metadata& md) const throw (CMMError)
{
   if (useV2_) {
      // In v2, we use PeekNextDataReadPointer (which does not advance the internal pointer)
      // Note: the v2 buffer is not channel aware, so the 'channel' parameter is ignored.
      // TODO implement the channel aware version
      const unsigned char* ptr = nullptr;
      size_t imageDataSize = 0;
      int ret = v2Buffer_->PeekNextDataReadPointer(&ptr, &imageDataSize, md);
      if (ret != DEVICE_OK || ptr == nullptr)
         throw CMMError("V2 buffer is empty.", MMERR_CircularBufferEmpty);
      return const_cast<unsigned char*>(ptr);
      // TODO: make sure calling code releases the slot after use
   } else {
      const mm::ImgBuffer* pBuf = circBuffer_->GetTopImageBuffer(channel);
      if (pBuf != nullptr) {
         md = pBuf->GetMetadata();
         return const_cast<unsigned char*>(pBuf->GetPixels());
      } else {
         throw CMMError("Circular buffer is empty.", MMERR_CircularBufferEmpty);
      }
   }
}

void* BufferAdapter::GetNthImageMD(unsigned long n, Metadata& md) const throw (CMMError)
{
   if (useV2_) {
      size_t dataSize = 0;
      const unsigned char* ptr = v2Buffer_->PeekDataReadPointerAtIndex(n, &dataSize, md);
      if (ptr == nullptr)
         throw CMMError("V2 buffer does not contain enough data.", MMERR_CircularBufferEmpty);
      // Return a non-const pointer (caller must be careful with the const_cast)
      return const_cast<unsigned char*>(ptr);
      // TODO: make sure calling code releases the slot after use
   } else {
      const mm::ImgBuffer* pBuf = circBuffer_->GetNthFromTopImageBuffer(n);
      if (pBuf != nullptr) {
         md = pBuf->GetMetadata();
         return const_cast<unsigned char*>(pBuf->GetPixels());
      } else {
         throw CMMError("Circular buffer is empty.", MMERR_CircularBufferEmpty);
      }
   }
}

void* BufferAdapter::PopNextImageMD(unsigned channel, Metadata& md) throw (CMMError)
{
   if (useV2_) {
      // For v2, consume the data by calling PopNextDataReadPointer,
      // which returns a const unsigned char* or throws an exception on error.
      // The caller is expected to call ReleaseDataReadPointer on the returned pointer once done.
      // TODO: make channel aware
      size_t dataSize = 0;
      const unsigned char* ptr = v2Buffer_->PopNextDataReadPointer(md, &dataSize, false);
      if (ptr == nullptr)
         throw CMMError("V2 buffer is empty.", MMERR_CircularBufferEmpty);
      return const_cast<unsigned char*>(ptr);
      // TODO: ensure that calling code releases the read pointer after use.
   } else {
      const mm::ImgBuffer* pBuf = circBuffer_->GetNextImageBuffer(channel);
      if (pBuf != nullptr) {
         md = pBuf->GetMetadata();
         return const_cast<unsigned char*>(pBuf->GetPixels());
      } else {
         throw CMMError("Circular buffer is empty.", MMERR_CircularBufferEmpty);
      }
   }
}
