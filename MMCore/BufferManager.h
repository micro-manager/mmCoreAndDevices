///////////////////////////////////////////////////////////////////////////////
// FILE:          BufferManager.h
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


#ifndef BUFFERMANAGER_H
#define BUFFERMANAGER_H

#include "CircularBuffer.h"
#include "Buffer_v2.h"
#include "../MMDevice/MMDevice.h"
#include <chrono>
#include <map>
#include <mutex>

// BufferManager provides a common interface for buffer operations
// used by MMCore. It currently supports only a minimal set of functions.
class BufferManager {
public:
   static const char* const DEFAULT_V2_BUFFER_NAME;

   /**
    * Constructor.
    * @param useV2Buffer Set to true to use the new DataBuffer (v2); false to use CircularBuffer.
    * @param memorySizeMB Memory size for the buffer (in megabytes).
    */
   BufferManager(bool useV2Buffer, unsigned int memorySizeMB);
   ~BufferManager();

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
   const void* GetLastImage();

   /**
    * Get a pointer to the next image from the buffer.
    * @return Pointer to image data, or nullptr if unavailable.
    */
   const void* PopNextImage();


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
    * @param caller The device inserting the image.
    * @param buf The image data.
    * @param width Image width.
    * @param height Image height.
    * @param byteDepth Bytes per pixel.
    * @param pMd Metadata associated with the image.
    * @return true on success, false on error.
    */
   bool InsertImage(const char* deviceLabel, const unsigned char *buf, 
                    unsigned width, unsigned height, unsigned byteDepth, Metadata *pMd);

   /**
    * Insert an image into the buffer with specified number of components.
    * @param caller The device inserting the image.
    * @param buf The image data.
    * @param width Image width.
    * @param height Image height.
    * @param byteDepth Bytes per pixel.
    * @param nComponents Number of components in the image.
    * @param pMd Metadata associated with the image.
    * @return true on success, false on error.
    */
   bool InsertImage(const char* deviceLabel, const unsigned char *buf, unsigned width, 
                    unsigned height, unsigned byteDepth, unsigned nComponents, Metadata *pMd);

   /**
    * Insert a multi-channel image into the buffer.
    * @param caller The device inserting the image.
    * @param buf The image data.
    * @param numChannels Number of channels in the image.
    * @param width Image width.
    * @param height Image height.
    * @param byteDepth Bytes per pixel.
    * @param pMd Metadata associated with the image.
    * @return true on success, false on error.
    */
   bool InsertMultiChannel(const char* deviceLabel, const unsigned char *buf, 
                           unsigned numChannels, unsigned width, unsigned height, 
                           unsigned byteDepth, Metadata *pMd);

   /**
    * Insert a multi-channel image into the buffer with specified number of components.
    * @param caller The device inserting the image.
    * @param buf The image data.
    * @param numChannels Number of channels in the image.
    * @param width Image width.
    * @param height Image height.
    * @param byteDepth Bytes per pixel.
    * @param nComponents Number of components in the image.
    * @param pMd Metadata associated with the image.
    * @return true on success, false on error.
    */
   bool InsertMultiChannel(const char* deviceLabel, const unsigned char *buf, 
                           unsigned numChannels, unsigned width, unsigned height,
                           unsigned byteDepth, unsigned nComponents, Metadata *pMd);

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



   const void* GetNthImageMD(unsigned long n, Metadata& md) const throw (CMMError);

   // Channels are not directly supported in v2 buffer, these are for backwards compatibility
   // with circular buffer
   const void* GetLastImageMD(unsigned channel, Metadata& md) const throw (CMMError);
   const void* PopNextImageMD(unsigned channel, Metadata& md) throw (CMMError);

   const void* GetLastImageMD(Metadata& md) const throw (CMMError);
   const void* PopNextImageMD(Metadata& md) throw (CMMError);

   /**
    * Check if this manager is using the V2 buffer implementation.
    * @return true if using V2 buffer, false if using circular buffer.
    */
   bool IsUsingV2Buffer() const;

   /**
    * Release a pointer obtained from the buffer.
    * This is required when using the V2 buffer implementation.
    * @param ptr The pointer to release.
    * @return true on success, false on error.
    */
   bool ReleaseReadAccess(const void* ptr);

   // Methods for the v2 buffer where width and heigh must be gotton on a per-image basis
   unsigned GetImageWidth(const void* ptr) const;
   unsigned GetImageHeight(const void* ptr) const;
   unsigned GetBytesPerPixel(const void* ptr) const;
   unsigned GetNumberOfComponents(const void* ptr) const;
   long GetImageBufferSize(const void* ptr) const;

   /**
    * Configure whether to overwrite old data when buffer is full.
    * @param overwrite If true, overwrite old data when buffer is full.
    * @return true on success, false on error.
    */
   bool SetOverwriteData(bool overwrite);

   /**
    * Acquires a write slot large enough to hold the image data and metadata.
    * @param deviceLabel The label of the device requesting the write slot
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
   bool AcquireWriteSlot(const char* deviceLabel, size_t dataSize, unsigned width, unsigned height, 
       unsigned byteDepth, unsigned nComponents, size_t additionalMetadataSize,
       void** dataPointer, void** additionalMetadataPointer,
       Metadata* pInitialMetadata = nullptr);

   /**
    * Finalizes (releases) a write slot after data has been written.
    * @param imageDataPointer Pointer previously obtained from AcquireWriteSlot.
    * @param actualMetadataBytes The actual number of metadata bytes written.
    * @return true on success, false on error.
    */
   bool FinalizeWriteSlot(void* imageDataPointer, size_t actualMetadataBytes);

   /**
    * Extracts metadata for a given image data pointer.
    * @param dataPtr Pointer to the image data.
    * @param md Metadata object to populate.
    * @throws CMMError if V2 buffer is not enabled or extraction fails.
    */
   void ExtractMetadata(const void* dataPtr, Metadata& md) const;

   /**
    * Add basic metadata tags for the data source device.
    * @param deviceLabel The label of the device inserting the metadata
    * @param pMd Optional pointer to existing metadata to merge with
    * @return Metadata object containing merged metadata
    */
   Metadata AddDeviceLabel(const char* deviceLabel, const Metadata* pMd);

   /**
    * Get the last image inserted by a specific device.
    * @param deviceLabel The label of the device to get the image from.
    * @return Pointer to the image data.
    * @throws CMMError if no image is found or V2 buffer is not enabled.
    */
   const void* GetLastImageFromDevice(const std::string& deviceLabel) throw (CMMError);

   /**
    * Get the last image and metadata inserted by a specific device.
    * @param deviceLabel The label of the device to get the image from.
    * @param md Metadata object to populate.
    * @return Pointer to the image data.
    * @throws CMMError if no image is found or V2 buffer is not enabled.
    */
   const void* GetLastImageMDFromDevice(const std::string& deviceLabel, Metadata& md) throw (CMMError);

   /**
    * Check if a pointer is currently managed by the buffer.
    * @param ptr The pointer to check.
    * @return true if the pointer is in the buffer, false otherwise.
    * @throws CMMError if V2 buffer is not enabled.
    */
   bool IsPointerInBuffer(const void* ptr) const throw (CMMError);

   /**
    * Get a pointer to the V2 buffer.
    * @return Pointer to the V2 buffer, or nullptr if V2 buffer is not enabled.
    */
   DataBuffer* GetV2Buffer() const;

private:
   unsigned GetBytesPerPixelFromType(const std::string& pixelType) const;
   unsigned GetComponentsFromType(const std::string& pixelType) const;
   
   bool useV2_;
   CircularBuffer* circBuffer_;
   DataBuffer* v2Buffer_;
   
    /**
    * Add essential metadata tags required for interpreting stored data and routing it
    * if multiple buffers are used. Only minimal parameters (width, height, pixel type)
    * are added. 
    * 
    * Future data-producer devices (ie those that dont produce conventional images) may
    * need alternative versions of this function.
    */
   void PopulateMetadata(Metadata& md, const char* deviceLabel, 
          unsigned width, unsigned height, unsigned byteDepth, unsigned nComponents);


   /**
    * Get the metadata tags attached to device caller, and merge them with metadata
    * in pMd (if not null). Returns a metadata object.
    * @param caller The device inserting the metadata
    * @param pMd Optional pointer to existing metadata to merge with
    * @return Metadata object containing merged metadata
    */
   Metadata AddCallerMetadata(const MM::Device* caller, const Metadata* pMd);
};

#endif // BUFFERMANAGER_H 