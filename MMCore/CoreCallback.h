///////////////////////////////////////////////////////////////////////////////
// FILE:          CoreCallback.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     MMCore
//-----------------------------------------------------------------------------
// DESCRIPTION:   Callback object for MMCore device interface. Encapsulates
//                (bottom) internal API for calls going from devices to the 
//                core.
//              
// AUTHOR:        Nenad Amodaj, nenad@amodaj.com, 01/23/2006

// COPYRIGHT:     University of California, San Francisco, 2006-2014
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

#pragma once

#include "Devices/DeviceInstances.h"
#include "CoreUtils.h"
#include "MMCore.h"
#include "MMEventCallback.h"
#include "../MMDevice/DeviceUtils.h"

namespace mm
{
   class DeviceManager;
}


///////////////////////////////////////////////////////////////////////////////
// CoreCallback class
// ------------------

class CoreCallback : public MM::Core
{
public:
   CoreCallback(CMMCore* c);
   ~CoreCallback();

   int GetDeviceProperty(const char* deviceName, const char* propName, char* value);
   int SetDeviceProperty(const char* deviceName, const char* propName, const char* value);

   /**
    * Writes a message to the Micro-Manager log file.
    */
   int LogMessage(const MM::Device* caller, const char* msg,
         bool debugOnly) const;

   /**
    * Returns a direct pointer to the device with the specified name.
    */
   MM::Device* GetDevice(const MM::Device* caller, const char* label);

   MM::PortType GetSerialPortType(const char* portName) const;
 
   int SetSerialProperties(const char* portName,
                           const char* answerTimeout,
                           const char* baudRate,
                           const char* delayBetweenCharsMs,
                           const char* handshaking,
                           const char* parity,
                           const char* stopBits);

   int WriteToSerial(const MM::Device* caller, const char* portName, const unsigned char* buf, unsigned long length);
   int ReadFromSerial(const MM::Device* caller, const char* portName, unsigned char* buf, unsigned long bufLength, unsigned long &bytesRead);
   int PurgeSerial(const MM::Device* caller, const char* portName);
   int SetSerialCommand(const MM::Device*, const char* portName, const char* command, const char* term);
   int GetSerialAnswer(const MM::Device*, const char* portName, unsigned long ansLength, char* answerTxt, const char* term);

   /*Deprecated*/ unsigned long GetClockTicksUs(const MM::Device* caller);

   MM::MMTime GetCurrentMMTime();

   void Sleep(const MM::Device* caller, double intervalMs);

   // continuous acquisition support
   /*Deprecated*/ int InsertImage(const MM::Device* caller, const ImgBuffer& imgBuf); // Note: _not_ mm::ImgBuffer
   int InsertImage(const MM::Device* caller, const unsigned char* buf, unsigned width, unsigned height, unsigned byteDepth, const char* serializedMetadata, const bool doProcess = true);
   int InsertImage(const MM::Device* caller, const unsigned char* buf, unsigned width, unsigned height, unsigned byteDepth, unsigned nComponents, const char* serializedMetadata, const bool doProcess = true);

   /*Deprecated*/ int InsertImage(const MM::Device* caller, const unsigned char* buf, unsigned width, unsigned height, unsigned byteDepth, const Metadata* pMd = 0, const bool doProcess = true);
   /*Deprecated*/ int InsertImage(const MM::Device* caller, const unsigned char* buf, unsigned width, unsigned height, unsigned byteDepth, unsigned nComponents, const Metadata* pMd = 0, const bool doProcess = true);

   /*Deprecated*/ int InsertMultiChannel(const MM::Device* caller, const unsigned char* buf, unsigned numChannels, unsigned width, unsigned height, unsigned byteDepth, Metadata* pMd = 0);
   
   // Method for inserting generic non-image data into the buffer
   int InsertData(const MM::Device* caller, const unsigned char* buf, size_t dataSize, Metadata* pMd = 0);

// TODO: enable these when we know what to do about backwards compatibility for circular buffer
//    /**
//     * Direct writing into V2 buffer instead of using InsertImage (which has to copy the data again).
//     * Version with essential parameters for camera devices.
//     * @param caller Camera device making the call
//     * @param dataSize Size of the image data in bytes
//     * @param metadataSize Size of metadata in bytes
//     * @param dataPointer Pointer that will be set to allocated image buffer. The caller should write 
//     *   data to this buffer.
//     * @param metadataPointer Pointer that will be set to allocated metadata buffer. The caller should write 
//     *   serialized metadata to this buffer.
//     * @param width Width of the image in pixels
//     * @param height Height of the image in pixels
//     * @param byteDepth Number of bytes per pixel
//     * @param nComponents Number of components per pixel
//     * @return DEVICE_OK on success
//     */
//    int AcquireImageWriteSlot(const MM::Camera* caller, size_t dataSize, size_t metadataSize,
//                         unsigned char** dataPointer, unsigned char** metadataPointer,
//                         unsigned width, unsigned height, unsigned byteDepth, unsigned nComponents);

//    /**
//     * Generic version for inserting data into the buffer.
//     * @param caller Device making the call
//     * @param dataSize Size of the data in bytes
//     * @param metadataSize Size of metadata in bytes
//     * @param dataPointer Pointer that will be set to allocated data buffer. The caller should write   
//     *   data to this buffer.
//     * @param metadataPointer Pointer that will be set to allocated metadata buffer. The caller should write 
//     *   serialized metadata to this buffer.
//     * @return DEVICE_OK on success
//     */
//    int AcquireDataWriteSlot(const MM::Device* caller, size_t dataSize, size_t metadataSize,
//                         unsigned char** dataPointer, unsigned char** metadataPointer);

//    /**
//     * Finalizes a write slot after data has been written.
//     * @param dataPointer Pointer to the data buffer that was written to
//     * @param actualMetadataBytes Actual number of metadata bytes written
//     * @return DEVICE_OK on success
//     */
//    int FinalizeWriteSlot(unsigned char* dataPointer, size_t actualMetadataBytes);

   // @deprecated This method was previously used by many camera adapters to enforce an overwrite mode on the circular buffer
   // -- making it wrap around when running a continuous sequence acquisition. Now we've added an option to do this into the 
   // circular buffer itself, so this method is no longer required and is not used. higher level code will control when this
   // option is activated, making it simpler to develop camera adatpers and giving more consistent behavior.
   void ClearImageBuffer(const MM::Device* /*caller*/) {};
   // @deprecated This method is not required for the V2 buffer and is called by higher level code for circular buffer
   bool InitializeImageBuffer(unsigned channels, unsigned slices, unsigned int w, unsigned int h, unsigned int pixDepth);

   int AcqFinished(const MM::Device* caller, int statusCode);
   int PrepareForAcq(const MM::Device* caller);

   // autofocus support
   const char* GetImage();
   int GetImageDimensions(int& width, int& height, int& depth);
   int GetFocusPosition(double& pos);
   int SetFocusPosition(double pos);
   int MoveFocus(double v);
   int SetXYPosition(double x, double y);
   int GetXYPosition(double& x, double& y);
   int MoveXYStage(double vX, double vY);
   int SetExposure(double expMs);
   int GetExposure(double& expMs);
   int SetConfig(const char* group, const char* name);
   int GetCurrentConfig(const char* group, int bufLen, char* name);
   int GetChannelConfig(char* channelConfigName, const unsigned int channelConfigIterator);

   // notification handlers
   int OnPropertiesChanged(const MM::Device* caller);
   int OnPropertyChanged(const MM::Device* device, const char* propName, const char* value);
   int OnStagePositionChanged(const MM::Device* device, double pos);
   int OnXYStagePositionChanged(const MM::Device* device, double xpos, double ypos);
   int OnExposureChanged(const MM::Device* device, double newExposure);
   int OnSLMExposureChanged(const MM::Device* device, double newExposure);
   int OnMagnifierChanged(const MM::Device* device);


   void NextPostedError(int& errorCode, char* pMessage, int maxlen, int& messageLength);
   void PostError(const int errorCode, const char* pMessage);
   void ClearPostedErrors();


   MM::ImageProcessor* GetImageProcessor(const MM::Device* caller);
   MM::State* GetStateDevice(const MM::Device* caller, const char* label);
   MM::SignalIO* GetSignalIODevice(const MM::Device* caller,
         const char* label);
   MM::AutoFocus* GetAutoFocus(const MM::Device* caller);
   MM::Hub* GetParentHub(const MM::Device* caller) const;
   void GetLoadedDeviceOfType(const MM::Device* caller, MM::DeviceType devType,
         char* deviceName, const unsigned int deviceIterator);



private:
   CMMCore* core_;
   MMThreadLock* pValueChangeLock_;

   int OnConfigGroupChanged(const char* groupName, const char* newConfigName);
   int OnPixelSizeChanged(double newPixelSizeUm);
   int OnPixelSizeAffineChanged(std::vector<double> newPixelSizeAffine);
};
