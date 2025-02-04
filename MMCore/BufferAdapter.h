#ifndef BUFFERADAPTER_H
#define BUFFERADAPTER_H

#include "CircularBuffer.h"
#include "Buffer_v2.h"
#include "../MMDevice/MMDevice.h"
#include <chrono>
#include <map>
#include <mutex>

// BufferAdapter provides a common interface for buffer operations
// used by MMCore. It currently supports only a minimal set of functions.
class BufferAdapter {
public:
   static const char* const DEFAULT_V2_BUFFER_NAME;

   /**
    * Constructor.
    * @param useV2Buffer Set to true to use the new DataBuffer (v2); false to use CircularBuffer.
    * @param memorySizeMB Memory size for the buffer (in megabytes).
    */
   BufferAdapter(bool useV2Buffer, unsigned int memorySizeMB);
   ~BufferAdapter();

   /**
    * Get a pointer to the top (most recent) image.
    * @return Pointer to image data, or nullptr if unavailable.
    */
   const unsigned char* GetLastImage() const;

   /**
    * Get a pointer to the next image from the buffer.
    * @return Pointer to image data, or nullptr if unavailable.
    */
   const unsigned char* PopNextImage();

   /**
    * Get a pointer to the nth image from the top of the buffer.
    * @param n The index from the top.
    * @return Pointer to image data, or nullptr if unavailable.
    */
   const mm::ImgBuffer* GetNthFromTopImageBuffer(unsigned long n) const;

   /**
    * Get a pointer to the next image buffer for a specific channel.
    * @param channel The channel number.
    * @return Pointer to image data, or nullptr if unavailable.
    */
   const mm::ImgBuffer* GetNextImageBuffer(unsigned channel);

   /**
    * Initialize the buffer with the given parameters.
    * @param numChannels Number of channels.
    * @param width Image width.
    * @param height Image height.
    * @param bytesPerPixel Bytes per pixel.
    * @return true on success, false on error.
    */
   bool Initialize(unsigned numChannels, unsigned width, unsigned height, unsigned bytesPerPixel);

   /**
    * Get the memory size of the buffer in megabytes.
    * @return Memory size in MB.
    */
   unsigned GetMemorySizeMB() const;

   /**
    * Get the remaining image count in the buffer.
    * @return Number of remaining images.
    */
   long GetRemainingImageCount() const;

   /**
    * Clear the entire image buffer.
    */
   void Clear();

   /**
    * Insert an image into the buffer.
    * @param buf The image data.
    * @param width Image width.
    * @param height Image height.
    * @param byteDepth Bytes per pixel.
    * @param pMd Metadata associated with the image.
    * @return true on success, false on error.
    */
   bool InsertImage(const unsigned char *buf, unsigned width, unsigned height, 
                    unsigned byteDepth, Metadata *pMd);

   /**
    * Insert an image into the buffer with specified number of components.
    * @param buf The image data.
    * @param width Image width.
    * @param height Image height.
    * @param byteDepth Bytes per pixel.
    * @param nComponents Number of components in the image.
    * @param pMd Metadata associated with the image.
    * @return true on success, false on error.
    */
   bool InsertImage(const unsigned char *buf, unsigned width, unsigned height, 
                    unsigned byteDepth, unsigned nComponents, Metadata *pMd);

   /**
    * Insert a multi-channel image into the buffer.
    * @param buf The image data.
    * @param numChannels Number of channels in the image.
    * @param width Image width.
    * @param height Image height.
    * @param byteDepth Bytes per pixel.
    * @param pMd Metadata associated with the image.
    * @return true on success, false on error.
    */
   bool InsertMultiChannel(const unsigned char *buf, unsigned numChannels, unsigned width, 
                           unsigned height, unsigned byteDepth, Metadata *pMd);

   /**
    * Insert a multi-channel image into the buffer with specified number of components.
    * @param buf The image data.
    * @param numChannels Number of channels in the image.
    * @param width Image width.
    * @param height Image height.
    * @param byteDepth Bytes per pixel.
    * @param nComponents Number of components in the image.
    * @param pMd Metadata associated with the image.
    * @return true on success, false on error.
    */
   bool InsertMultiChannel(const unsigned char *buf, unsigned numChannels, unsigned width, 
                           unsigned height, unsigned byteDepth, unsigned nComponents, Metadata *pMd);

   /**
    * Get the total capacity of the buffer.
    * @return Total capacity of the buffer.
    */
   long GetSize(long imageSize) const;

   /**
    * Get the free capacity of the buffer.
    * @param imageSize Size of a single image in bytes.
    * @return Number of images that can be added without overflowing.
    */
   long GetFreeSize(long imageSize) const;

   /**
    * Check if the buffer is overflowed.
    * @return True if overflowed, false otherwise.
    */
   bool Overflow() const;

   /**
    * Get a pointer to the top image buffer for a specific channel.
    * @param channel The channel number.
    * @return Pointer to image data, or nullptr if unavailable.
    */
   const mm::ImgBuffer* GetTopImageBuffer(unsigned channel) const;

   void* GetLastImageMD(unsigned channel, Metadata& md) const throw (CMMError);
   void* GetNthImageMD(unsigned long n, Metadata& md) const throw (CMMError);
   void* PopNextImageMD(unsigned channel, Metadata& md) throw (CMMError);

private:
   bool useV2_; // if true use DataBuffer, otherwise use CircularBuffer.
   CircularBuffer* circBuffer_;
   DataBuffer* v2Buffer_;
   
   std::chrono::steady_clock::time_point startTime_;
   std::map<std::string, long> imageNumbers_;  // Track image numbers per camera
   std::mutex imageNumbersMutex_;  // Mutex to protect access to imageNumbers_

   void ProcessMetadata(Metadata& md, unsigned width, unsigned height, 
       unsigned byteDepth, unsigned nComponents);
};

#endif // BUFFERADAPTER_H 