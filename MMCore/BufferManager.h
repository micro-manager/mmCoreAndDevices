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
#include "NewDataBuffer.h"
#include "../MMDevice/MMDevice.h"
#include <chrono>
#include <map>
#include <mutex>
#include <atomic>

/**
 * BufferManager provides a generic interface for managing data buffers in MMCore.
 * 
 * This class is designed to handle arbitrary data in a format-agnostic way, with
 * metadata for interpretation handled separately. Data can be stored and retrieved
 * with associated metadata that describes how to interpret the raw bytes.
 * 
 * The implementation supports two buffer types:
 * - The newer NewDataBuffer that handles generic data
 * - The legacy circular buffer (for backwards compatibility) that assumes image data
 * 
 * While the preferred usage is through the generic data methods (InsertData, 
 * GetLastData, etc.), legacy image-specific methods are maintained for compatibility
 * with existing code that assumes images captured on a camera.
 */
class BufferManager {
public:
   static const char* const DEFAULT_NEW_DATA_BUFFER_NAME;

   /**
    * Constructor.
    * @param useNewDataBuffer Set to true to use the new DataBuffer (NewDataBuffer); false to use CircularBuffer.
    * @param memorySizeMB Memory size for the buffer (in megabytes).
    */
   BufferManager(bool useNewDataBuffer, unsigned int memorySizeMB);
   ~BufferManager();

   /**
    * Reinitialize the buffer.
    * @param memorySizeMB Memory size for the buffer (in megabytes).
    */
   void ReallocateBuffer(unsigned int memorySizeMB);

   /**
    * Enable or disable NewDataBuffer usage.
    * @param enable Set to true to use NewDataBuffer, false to use CircularBuffer.
    * @return true if the switch was successful, false otherwise.
    */
   int EnableNewDataBuffer(bool enable);

   /**
    * Get a pointer to the top (most recent) image.
    * @return Pointer to image data, or nullptr if unavailable.
    */
   const void* GetLastData();

   /**
    * Get a pointer to the next image from the buffer.
    * @return Pointer to image data, or nullptr if unavailable.
    */
   const void* PopNextData();

   /**
    * Get the memory size of the buffer in megabytes.
    * @return Memory size in MB.
    */
   unsigned GetMemorySizeMB() const;

   /**
    * Get the free capacity of the buffer.
    * @return Free capacity in MB.
    */
   unsigned GetFreeSizeMB() const;

   /**
    * Get the remaining data entry (e.g. image) count in the buffer.
    * @return Number of remaining data entries.
    */
   long GetRemainingDataCount() const;

   /**
    * Insert an image into the buffer
    * @param caller The device inserting the image.
    * @param buf The image data.
    * @param width Image width.
    * @param height Image height.
    * @param byteDepth Bytes per pixel.
    * @param pMd Metadata associated with the image.
    * @return DEVICE_OK on success, DEVICE_ERR on error.
    * @deprecated This method assumes specific image data format. It is provided for backwards 
    * compatibility with with the circular buffer, which assumes images captured on a camera.
    * Use InsertData() instead, which provides format-agnostic data handling with metadata for interpretation.
    */
   int InsertImage(const char* deviceLabel, const unsigned char *buf, 
                   unsigned width, unsigned height, unsigned byteDepth, 
                   Metadata *pMd);


   /**
    * Insert a multi-channel image into the buffer
    * @param caller The device inserting the image.
    * @param buf The image data.
    * @param numChannels Number of channels in the image.
    * @param width Image width.
    * @param height Image height.
    * @param byteDepth Bytes per pixel.
    * @param pMd Metadata associated with the image.
    * @return DEVICE_OK on success, DEVICE_ERR on error.
    * @deprecated This method is not preferred for the NewDataBuffer. Use InsertData() instead.
    *             This method assumes specific image data format. It is provided for backwards 
    *             compatibility with with the circular buffer, which assumes images captured on a camera.
    */
   int InsertMultiChannel(const char* deviceLabel, const unsigned char *buf, 
                           unsigned numChannels, unsigned width, unsigned height,
                           unsigned byteDepth, Metadata *pMd);

   /**
    * Insert data into the buffer. This method is agnostic to the format of the data
    * It is the caller's responsibility to ensure that appropriate metadata is provided
    * for interpretation of the data (e.g. for an image: width, height, byteDepth, nComponents)
    * 
    * @param callerLabel The label of the device inserting the data.
    * @param buf The data to insert.
    * @param dataSize The size of the data to insert.
    * @param pMd Metadata associated with the data.
    * @return DEVICE_OK on success, DEVICE_ERR on error.
    */
   int InsertData(const char* callerLabel, const unsigned char* buf, size_t dataSize, Metadata *pMd);


   /**
    * Check if the buffer is overflowed.
    * @return True if overflowed, false otherwise.
    */
   bool Overflow() const;



   const void* GetNthDataMD(unsigned long n, Metadata& md) const;

   // Channels are not directly supported in NewDataBuffer, these are for backwards compatibility
   // with circular buffer
   
   /**
    * @deprecated This method is not preferred for the NewDataBuffer. Use GetLastDataMD() without channel parameter instead.
    *             The NewDataBuffer is data type agnostic
    */
   const void* GetLastDataMD(unsigned channel, unsigned singleChannelSizeBytes, Metadata& md) const;
   /**
    * @deprecated This method is not preferred for the NewDataBuffer. Use PopNextDataMD() without channel parameter instead.
    *             The NewDataBuffer is data type agnostic
    */
   const void* PopNextDataMD(unsigned channel, unsigned singleChannelSizeBytes,Metadata& md);

   const void* GetLastDataMD(Metadata& md) const;
   const void* PopNextDataMD(Metadata& md);

   /**
    * Check if this manager is using the NewDataBuffer implementation.
    * @return true if using NewDataBuffer, false if using circular buffer.
    */
   bool IsUsingNewDataBuffer() const;

   /**
    * Release a pointer obtained from the buffer.
    * This is required when using the NewDataBuffer implementation.
    * @param ptr The pointer to release.
    * @return DEVICE_OK on success, DEVICE_ERR on error.
    */
   int ReleaseReadAccess(const void* ptr);

   // Get the size of just the data in this slot
   unsigned GetDataSize(const void* ptr) const;

   /**
    * Configure whether to overwrite old data when buffer is full.
    * @param overwrite If true, overwrite old data when buffer is full.
    * @return DEVICE_OK on success, DEVICE_ERR on error.
    */
   int SetOverwriteData(bool overwrite);

   /**
    * Acquires a write slot large enough to hold the data and metadata.
    * @param deviceLabel The label of the device requesting the write slot
    * @param dataSize The number of bytes reserved for image or other primary data.
    * @param additionalMetadataSize The maximum number of bytes reserved for metadata.
    * @param dataPointer On success, receives a pointer to the image data region.
    * @param additionalMetadataPointer On success, receives a pointer to the metadata region.
    * @param pInitialMetadata Optionally, a pointer to a metadata object whose contents should be pre‐written
    * @return DEVICE_OK on success, DEVICE_ERR on error.
    */
   int AcquireWriteSlot(const char* deviceLabel, size_t dataSize, size_t additionalMetadataSize,
       void** dataPointer, void** additionalMetadataPointer, Metadata* pInitialMetadata);

   /**
    * Finalizes (releases) a write slot after data has been written.
    * @param dataPointer Pointer previously obtained from AcquireWriteSlot.
    * @param actualMetadataBytes The actual number of metadata bytes written.
    * @return DEVICE_OK on success, DEVICE_ERR on error.
    */
   int FinalizeWriteSlot(const void* imageDataPointer, size_t actualMetadataBytes);

   /**
    * Extracts metadata for a given data pointer.
    * @param dataPtr Pointer to the data.
    * @param md Metadata object to populate.
    * @throws CMMError if NewDataBuffer is not enabled or extraction fails.
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
    * Get the last data inserted by a specific device.
    * @param deviceLabel The label of the device to get the data from.
    * @return Pointer to the data.
    * @throws CMMError if no data is found or NewDataBuffer is not enabled.
    */
   const void* GetLastDataFromDevice(const std::string& deviceLabel);

   /**
    * Get the last data and metadata inserted by a specific device.
    * @param deviceLabel The label of the device to get the data from.
    * @param md Metadata object to populate.
    * @return Pointer to the data.
    * @throws CMMError if no data is found or NewDataBuffer is not enabled.
    */
   const void* GetLastDataMDFromDevice(const std::string& deviceLabel, Metadata& md);

   /**
    * Check if a pointer is currently managed by the buffer.
    * @param ptr The pointer to check.
    * @return true if the pointer is in the buffer, false otherwise.
    */
   bool IsPointerInNewDataBuffer(const void* ptr) const;

   /**
    * Get whether the buffer is in overwrite mode.
    * @return true if buffer overwrites old data when full, false otherwise.
    */
   bool GetOverwriteData() const;

   /**
    * Get the underlying CircularBuffer pointer.
    * This method is provided for backwards compatibility only.
    * @return Pointer to CircularBuffer if using legacy buffer, nullptr if using NewDataBuffer
    * @deprecated This method exposes implementation details and should be avoided in new code
    */
   CircularBuffer* GetCircularBuffer() { return circBuffer_; }

   /**
    * Reset the buffer, discarding all data that is not currently held externally.
    */
   void Reset();

private:
   
   std::atomic<bool> useNewDataBuffer_;
   CircularBuffer* circBuffer_;
   DataBuffer* newDataBuffer_;
   

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