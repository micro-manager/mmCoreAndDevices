///////////////////////////////////////////////////////////////////////////////
// FILE:          BufferAdapter.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     MMCore
//-----------------------------------------------------------------------------
// DESCRIPTION:   Generic implementation of a buffer for storing image data and
//                metadata. Provides thread-safe access for reading and writing
//                with configurable overflow behavior.
////
// COPYRIGHT:     Henry Pinkard, 2025
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
// AUTHOR:        Henry Pinkard,  01/31/2025


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
    * Enable or disable v2 buffer usage.
    * @param enable Set to true to use v2 buffer, false to use circular buffer.
    * @return true if the switch was successful, false otherwise.
    */
   bool EnableV2Buffer(bool enable);

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

   /**
    * Check if this adapter is using the V2 buffer implementation.
    * @return true if using V2 buffer, false if using circular buffer.
    */
   bool IsUsingV2Buffer() const;

   /**
    * Release a pointer obtained from the buffer.
    * This is required when using the V2 buffer implementation.
    * @param ptr The pointer to release.
    */
   void ReleaseReadAccess(const unsigned char* ptr);

   // Methods for the v2 buffer where width and heigh must be gotton on a per-image basis
   unsigned GetImageWidth(const unsigned char* ptr) const;
   unsigned GetImageHeight(const unsigned char* ptr) const;
   unsigned GetBytesPerPixel(const unsigned char* ptr) const;
   unsigned GetImageBitDepth(const unsigned char* ptr) const;
   unsigned GetNumberOfComponents(const unsigned char* ptr) const;
   long GetImageBufferSize(const unsigned char* ptr) const;

   /**
    * Configure whether to overwrite old data when buffer is full.
    * @param overwrite If true, overwrite old data when buffer is full.
    * @return true on success, false on error.
    */
   bool SetOverwriteData(bool overwrite);

   /**
    * Acquires a write slot large enough to hold the image data and metadata.
    * @param dataSize The number of bytes reserved for image or other primary data.
    * @param width Image width.
    * @param height Image height.
    * @param byteDepth Bytes per pixel.
    * @param nComponents Number of components in the image.
    * @param additionalMetadataSize The maximum number of bytes reserved for metadata.
    * @param dataPointer On success, receives a pointer to the image data region.
    * @param additionalMetadataPointer On success, receives a pointer to the metadata region.
    * @param pInitialMetadata Optionally, a pointer to a metadata object whose contents should be pre‐written. Defaults to nullptr.
    * @return true on success, false on error.
    */
   bool AcquireWriteSlot(size_t dataSize, unsigned width, unsigned height, 
       unsigned byteDepth, unsigned nComponents, size_t additionalMetadataSize,
       unsigned char** dataPointer, unsigned char** additionalMetadataPointer,
       Metadata* pInitialMetadata = nullptr);

   /**
    * Finalizes (releases) a write slot after data has been written.
    * @param imageDataPointer Pointer previously obtained from AcquireWriteSlot.
    * @param actualMetadataBytes The actual number of metadata bytes written.
    * @return true on success, false on error.
    */
   bool FinalizeWriteSlot(unsigned char* imageDataPointer, size_t actualMetadataBytes);

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