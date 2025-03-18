// Micro-Manager IIDC Device Adapter
//
// AUTHOR:        Mark A. Tsuchida
//
// COPYRIGHT:     2014-2015, Regents of the University of California
//                2016, Open Imaging, Inc.
//
// LICENSE:       This file is distributed under the BSD license.
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

#include "IIDCInterface.h"
#include "IIDCCamera.h"

#include "DeviceBase.h"

#include <boost/thread.hpp>
#include <boost/shared_array.hpp>
#include <boost/shared_ptr.hpp>

#include <utility>
#include <vector>


class MMIIDCHub; // Not a Micro-Manager "Hub" device


class MMIIDCCamera : public CLegacyCameraBase<MMIIDCCamera>
{
   boost::shared_ptr<MMIIDCHub> hub_;

   boost::shared_ptr<IIDC::Camera> iidcCamera_;
   std::vector< boost::shared_ptr<IIDC::VideoMode> > videoModes_;

   // Cache the video mode info, because getting it during a capture can hang
   // (observed on OpenSUSE 12.3, libdc1394 2.2.0).
   boost::shared_ptr<IIDC::VideoMode> currentVideoMode_;

   // Unchanging settings (set once from pre-init properties)
   double shutterUsPerUnit_, shutterOffsetUs_;
   bool absoluteGainIsReadOnly_;

   // Cached settings and properties
   unsigned cachedBitsPerSample_; // Depends on video mode
   double cachedFramerate_; // Depends on video mode and Format_7 packet size
   uint32_t cachedPacketSize_;
   double cachedExposure_; // User settable but may also depend on video mode

   boost::mutex sampleProcessingMutex_;
   bool rightShift16BitSamples_; // Guarded by sampleProcessingMutex_

   bool stopOnOverflow_; // Set by StartSequenceAcquisition(), read by SequenceCallback()

   boost::mutex timebaseMutex_;
   uint32_t timebaseUs_; // 0 at start of sequence acquisition; timestamp of first frame

   int nextAdHocErrorCode_;

   /*
    * ROI state
    * We have a "hard" ROI (implemented in camera and IIDC) and "soft" ROI
    * (implemented in this device adapter), so that we can always set the ROI
    * at 1-pixel resolution.
    */
   unsigned roiLeft_, roiTop_; // As presented to MMCore
   unsigned roiWidth_, roiHeight_; // As presented to MMCore
   unsigned softROILeft_, softROITop_; // Relative to hard ROI

   /*
    * Keep snapped image in our own buffer
    */
   boost::shared_array<const unsigned char> snappedPixels_;
   size_t snappedWidth_, snappedHeight_, snappedBytesPerPixel_;

public:
   MMIIDCCamera();
   virtual ~MMIIDCCamera();

   /*
    * Device methods
    */

   virtual int Initialize();
   virtual int Shutdown();

   virtual bool Busy();
   virtual void GetName(char* name) const;

   /*
    * Camera methods
    */

   virtual int SnapImage();
   virtual const unsigned char* GetImageBuffer() { return GetImageBuffer(0); }
   virtual const unsigned char* GetImageBuffer(unsigned chan);

   // virtual const unsigned int* GetImageBufferAsRGB32(); // TODO
   // virtual unsigned GetNumberOfComponents() const;
   // virtual int GetComponentName(unsigned component, char* name);

   virtual long GetImageBufferSize() const;
   virtual unsigned GetImageWidth() const;
   virtual unsigned GetImageHeight() const;
   virtual unsigned GetImageBytesPerPixel() const;
   virtual unsigned GetNumberOfComponents() const;
   virtual unsigned GetBitDepth() const;

   virtual int GetBinning() const { return 1; }
   virtual int SetBinning(int) { return DEVICE_UNSUPPORTED_COMMAND; }
   virtual void SetExposure(double milliseconds);
   virtual double GetExposure() const { return cachedExposure_; }

   virtual int SetROI(unsigned x, unsigned y, unsigned xSize, unsigned ySize);
   virtual int GetROI(unsigned& x, unsigned& y, unsigned& xSize, unsigned& ySize);
   virtual int ClearROI();

   virtual int StartSequenceAcquisition(long count, double intervalMs, bool stopOnOverflow);
   virtual int StartSequenceAcquisition(double intervalMs)
   { return StartSequenceAcquisition(LONG_MAX, intervalMs, false); }
   virtual int StopSequenceAcquisition();
   virtual bool IsCapturing();

   virtual int IsExposureSequenceable(bool& f) const { f = false; return DEVICE_OK; }

private:
   /*
    * Property action handlers
    */

   int OnMaximumFramerate(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnRightShift16BitSamples(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnFormat7PacketSizeNegativeDelta(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnFormat7PacketSize(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnFormat7PacketSizeMode(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnVideoMode(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnExposure(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnBrightness(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnGain(MM::PropertyBase* pProp, MM::ActionType eAct);
   int OnReadOnlyAbsoluteGain(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
   /*
    * Internal functions
    */

   int InitializeInformationalProperties();
   int InitializeBehaviorTweakProperties();
   int InitializeVideoMode();
   int InitializeVideoModeDependentState();
   int InitializeFeatureProperties();

   // Set the framerate appropriately after parameters affecting possible frame
   // rate have changed
   int UpdateFramerate(bool forceKeepPacketSize = false);
   // Update properties that might be affected after a video mode switch
   int VideoModeDidChange();

   void SetExposureImpl(double exposure);
   double GetExposureUncached();
   std::pair<double, double> GetExposureLimits();

   // Note: in the next bunch of functions, the uint32_t timestampUs is
   // currently the only pass-through metadata. If anything more is added (such
   // as other fields of dc1394video_frame_t), we should change this into an
   // object (with unique_ptr semantics).

   void ProcessImage(const void* source, bool ownResultBuffer,
         IIDC::PixelFormat sourceFormat,
         size_t sourceWidth, size_t sourceHeight,
         size_t destLeft, size_t destTop,
         size_t destWidth, size_t destHeight,
         uint32_t timestampUs,
         boost::function<void (const void*, size_t, uint32_t)> resultCallback);

   void SnapCallback(const void* pixels, size_t width, size_t height,
         IIDC::PixelFormat format, uint32_t timestampUs);
   void SequenceCallback(const void* pixels, size_t width, size_t height,
         IIDC::PixelFormat format, uint32_t timestampUs);
   void ProcessedSnapCallback(const void* pixels, size_t width, size_t height,
         size_t bytesPerPixel, uint32_t timestampUs);
   void ProcessedSequenceCallback(const void* pixels, size_t width, size_t height,
         size_t bytesPerPixel, uint32_t timestampUs);
   void SequenceFinishCallback();

   void ResetTimebase();
   double ComputeRelativeTimestampMs(uint32_t rawTimeStampUs);

   int AdHocErrorCode(const std::string& message);

   void LogIIDCMessage(const std::string& message, bool isDebug);
};
