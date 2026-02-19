///////////////////////////////////////////////////////////////////////////////
// FILE:          MMDevice.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     MMDevice - Device adapter kit
//-----------------------------------------------------------------------------
// DESCRIPTION:   The interface to the Micro-Manager devices. Defines the
//                plugin API for all devices.
//
// AUTHOR:        Nenad Amodaj, nenad@amodaj.com, 06/08/2005
//
// COPYRIGHT:     University of California, San Francisco, 2006-2014
//                100X Imaging Inc, 2008
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

///////////////////////////////////////////////////////////////////////////////
// Header version
// If any of the class definitions changes, the interface version
// must be incremented
#define DEVICE_INTERFACE_VERSION 74
///////////////////////////////////////////////////////////////////////////////

// N.B.
//
// Never add parameters or return values that are not POD
// (http://stackoverflow.com/a/146454) to any method of class Device and its
// derived classes defined in this file. For example, a std::string parameter
// is not acceptable (use const char*). This is to prevent inter-DLL
// incompatibilities.

#include "MMDeviceConstants.h"
#include "DeviceUtils.h"
#include "DeviceThreads.h"

#include <climits>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>


namespace MM {

   // forward declaration for the MMCore callback class
   class Core;

   /**
    * @brief Utility class used both MMCore and devices to maintain time
    * intervals in the uniform, platform independent way.
    */
   class MMTime
   {
         long long microseconds_;

      public:
         MMTime() : microseconds_(0LL) {}

         explicit MMTime(double uSecTotal) :
            microseconds_(static_cast<long long>(uSecTotal))
         {}

         explicit MMTime(long sec, long uSec) :
            microseconds_(sec * 1'000'000LL + uSec)
         {}

         static MMTime fromUs(long long us)
         {
            // Work around our lack of a constructor that directly sets the
            // internal representation.
            // (Note that we cannot add a constructor from 'long long' because
            // many existing uses would then get an error (ambiguous with the
            // 'double' overload).)
            MMTime ret;
            ret.microseconds_ = us;
            return ret;
         }

         static MMTime fromMs(double ms)
         {
            return MMTime(ms * 1000.0);
         }

         static MMTime fromSeconds(long secs)
         {
            return MMTime(secs, 0);
         }

         MMTime operator+(const MMTime &other) const
         {
            return fromUs(microseconds_ + other.microseconds_);
         }

         MMTime operator-(const MMTime &other) const
         {
            return fromUs(microseconds_ - other.microseconds_);
         }

         bool operator>(const MMTime &other) const
         {
            return microseconds_ > other.microseconds_;
         }

         bool operator>=(const MMTime &other) const
         {
            return microseconds_ >= other.microseconds_;
         }

         bool operator<(const MMTime &other) const
         {
            return microseconds_ < other.microseconds_;
         }

         bool operator<=(const MMTime &other) const
         {
            return microseconds_ <= other.microseconds_;
         }

         bool operator==(const MMTime &other) const
         {
            return microseconds_ == other.microseconds_;
         }

         bool operator!=(const MMTime &other) const
         {
            return !(*this == other);
         }

         double getMsec() const
         {
            return microseconds_ / 1000.0;
         }

         double getUsec() const
         {
            return static_cast<double>(microseconds_);
         }

         std::string toString() const {
            long long absUs = std::abs(microseconds_);
            long long seconds = absUs / 1'000'000LL;
            long long fracUs = absUs - seconds * 1'000'000LL;
            const char *sign = microseconds_ < 0 ? "-" : "";

            using namespace std;
            ostringstream s;
            s << sign << seconds << '.' <<
                  setfill('0') << right << setw(6) << fracUs;
            return s.str();
         }
   };


   /**
    * @brief Timeout utility class.
    */
   class TimeoutMs
   {
   public:
      // arguments:  MMTime start time, millisecond interval time
      explicit TimeoutMs(const MMTime startTime, const unsigned long intervalMs) :
         startTime_(startTime),
         interval_(0, 1000*intervalMs)
      {
      }
      explicit TimeoutMs(const MMTime startTime, const MMTime interval) :
         startTime_(startTime),
         interval_(interval)
      {
      }

      bool expired(const MMTime tnow)
      {
         MMTime elapsed = tnow - startTime_;
         return ( interval_ < elapsed );
      }

   private:
      MMTime startTime_; // start time
      MMTime interval_; // interval in milliseconds
   };


   /**
    * @brief Generic device interface.
    */
   class Device {
   public:
      Device() {}
      virtual ~Device() {}

      virtual unsigned GetNumberOfProperties() const = 0;
      virtual int GetProperty(const char* name, char* value) const = 0;
      virtual int SetProperty(const char* name, const char* value) = 0;
      virtual bool HasProperty(const char* name) const = 0;
      virtual bool GetPropertyName(unsigned idx, char* name) const = 0;
      virtual int GetPropertyReadOnly(const char* name, bool& readOnly) const = 0;
      virtual int GetPropertyInitStatus(const char* name, bool& preInit) const = 0;
      virtual int HasPropertyLimits(const char* name, bool& hasLimits) const = 0;
      virtual int GetPropertyLowerLimit(const char* name, double& lowLimit) const = 0;
      virtual int GetPropertyUpperLimit(const char* name, double& hiLimit) const = 0;
      virtual int GetPropertyType(const char* name, MM::PropertyType& pt) const = 0;
      virtual unsigned GetNumberOfPropertyValues(const char* propertyName) const = 0;
      virtual bool GetPropertyValueAt(const char* propertyName, unsigned index, char* value) const = 0;
      /**
       * @brief Check whether the given property can be used with sequences.
       *
       * Sequences can be used for fast acquisitions, synchronized by TTLs rather
       * than computer commands. Sequences of states can be uploaded to the
       * device. The device will cycle through the uploaded list of states
       * (triggered by an external trigger - most often coming from the camera).
       * If the device is capable (and ready) to do so, isSequenceable will be
       * true.
       */
      virtual int IsPropertySequenceable(const char* name, bool& isSequenceable) const = 0;
      /**
       * @brief Return the largest sequence that can be stored in the device.
       */
      virtual int GetPropertySequenceMaxLength(const char* propertyName, long& nrEvents) const = 0;
      /**
       * @brief Start execution of the sequence.
       */
      virtual int StartPropertySequence(const char* propertyName) = 0;
      /**
       * @brief Stop execution of the sequence.
       */
      virtual int StopPropertySequence(const char* propertyName) = 0;
      /**
       * @brief Remove previously added sequence.
       */
      virtual int ClearPropertySequence(const char* propertyName) = 0;
      /**
       * @brief Add one value to the sequence.
       */
      virtual int AddToPropertySequence(const char* propertyName, const char* value) = 0;
      /**
       * @brief Signal that we are done sending sequence values so that the
       * adapter can send the whole sequence to the device.
       */
      virtual int SendPropertySequence(const char* propertyName) = 0;

      virtual bool GetErrorText(int errorCode, char* errMessage) const = 0;
      virtual bool Busy() = 0;
      virtual double GetDelayMs() const = 0;
      virtual void SetDelayMs(double delay) = 0;
      virtual bool UsesDelay() = 0;

      virtual void SetLabel(const char* label) = 0;
      virtual void GetLabel(char* name) const = 0;
      virtual void SetModuleName(const char* moduleName) = 0;
      virtual void GetModuleName(char* moduleName) const = 0;
      virtual void SetDescription(const char* description) = 0;
      virtual void GetDescription(char* description) const = 0;

      virtual int Initialize() = 0;
      /**
       * @brief Shut down (unload) the device.
       *
       * Required by the MM::Device API.
       * Ideally this method will completely unload the device and release all resources.
       * Shutdown() may be called multiple times in a row.
       * After Shutdown() we should be allowed to call Initialize() again to load the device
       * without causing problems.
       */
      virtual int Shutdown() = 0;

      virtual DeviceType GetType() const = 0;
      virtual void GetName(char* name) const = 0;
      virtual void SetCallback(Core* callback) = 0;

      //device discovery API
      virtual bool SupportsDeviceDetection(void) = 0;
      virtual MM::DeviceDetectionStatus DetectDevice(void) = 0;

      // hub-peripheral relationship
      virtual void SetParentID(const char* parentId) = 0;
      virtual void GetParentID(char* parentID) const = 0;
      // virtual void SetID(const char* id) = 0;
      // virtual void GetID(char* id) const = 0;
   };

   /**
    * @brief Generic Device.
    */
   class Generic : public Device
   {
   public:
      virtual DeviceType GetType() const { return Type; }
      static const DeviceType Type;
   };

   /**
    * @brief Camera API.
    */
   class Camera : public Device {
   public:
      Camera() {}
      virtual ~Camera() {}

      virtual DeviceType GetType() const { return Type; }
      static const DeviceType Type;

      // Camera API
      /**
       * @brief Perform exposure and grab a single image.
       *
       * Required by the MM::Camera API.
       *
       * SnapImage should start the image exposure in the camera and block until
       * the exposure is finished.  It should not wait for read-out and transfer of data.
       * Return DEVICE_OK on success, error code otherwise.
       */
      virtual int SnapImage() = 0;
      /**
       * @brief Return pixel data.
       *
       * Required by the MM::Camera API.
       * GetImageBuffer will be called shortly after SnapImage returns.
       * Use it to wait for camera read-out and transfer of data into memory
       * Return a pointer to a buffer containing the image data
       * The calling program will assume the size of the buffer based on the values
       * obtained from GetImageBufferSize(), which in turn should be consistent with
       * values returned by GetImageWidth(), GetImageHeight(), GetImageBytesPerPixel(),
       * and getNumberOfComponents().
       * The calling program also assumes that camera never changes the size of
       * the pixel buffer on its own. In other words, the buffer can change only if
       * appropriate properties are set (such as binning, pixel type, etc.)
       * Multi-Channel cameras should return the content of the first channel in this call.
       *
       * Supported data types are byte (8 bits per pixel, 1 component), short
       * (16 bits per pixel, 1 component), float (32 bits per pixel, 1 component, not
       * supported by the UI yet), RGB_32 (8 bits per component, 4 components), RGB_64
       * (16 bits per component, 4 components, not supported by UI yet).
       * RGB buffers are expected to be in big endian ARGB format (ARGB8888), which means that
       * on little endian format (currently most/all? code is compiled for little endian
       * architecture), the format is BGRA888 (see: https://en.wikipedia.org/wiki/RGBA_color_model).
       */
      virtual const unsigned char* GetImageBuffer() = 0;
      /**
       * @brief Return pixel data for cameras with multiple channels.
       *
       * See description for GetImageBuffer() for details.
       * Use this overloaded version for cameras with multiple channels
       * When calling this function for a single channel camera, this function
       * should return the content of the imagebuffer as returned by the function
       * GetImageBuffer().  This behavior is implemented in the DeviceBase.
       * When GetImageBuffer() is called for a multi-channel camera, the
       * camera adapter should return the ImageBuffer for the first channel
       *
       * @param channelNr Number of the channel for which the image data are requested.
       */
      virtual const unsigned char* GetImageBuffer(unsigned channelNr) = 0;
      /**
       * @brief Return pixel data with interleaved RGB pixels in 32 bpp format.
       */
      virtual const unsigned int* GetImageBufferAsRGB32() = 0;
      /**
       * @brief Return the number of components in this image.
       *
       * This is '1' for grayscale cameras, and '4' for RGB cameras.
       */
      virtual unsigned GetNumberOfComponents() const = 0;
      /**
       * @brief Return the name for each component.
       */
      virtual int GetComponentName(unsigned component, char* name) = 0;
      /**
       * @brief Return the number of simultaneous channels that camera is capable of.
       *
       * This should be used by devices capable of generating multiple channels of imagedata simultaneously.
       * Note: this should not be used by color cameras (use getNumberOfComponents instead).
       */
      virtual int unsigned GetNumberOfChannels() const = 0;
      /**
       * @brief Return the name for each Channel.
       *
       * An implementation of this function is provided in DeviceBase.h.  It will return an empty string
       */
      virtual int GetChannelName(unsigned channel, char* name) = 0;
      /**
       * @brief Return the size in bytes of the image buffer.
       *
       * Required by the MM::Camera API.
       * For multi-channel cameras, return the size of a single channel
       */
      virtual long GetImageBufferSize() const = 0;
      /**
       * @brief Return image buffer X-size in pixels.
       *
       * Required by the MM::Camera API.
       */
      virtual unsigned GetImageWidth() const = 0;
      /**
       * @brief Return image buffer Y-size in pixels.
       *
       * Required by the MM::Camera API.
       */
      virtual unsigned GetImageHeight() const = 0;
      /**
       * @brief Return image buffer pixel depth in bytes.
       *
       * Required by the MM::Camera API.
       */
      virtual unsigned GetImageBytesPerPixel() const = 0;
      /**
       * @brief Return the bit depth (dynamic range) of the pixel.
       *
       * This does not affect the buffer size, it just gives the client application
       * a guideline on how to interpret pixel values.
       * Required by the MM::Camera API.
       */
      virtual unsigned GetBitDepth() const = 0;
      /**
       * @brief Unused and slated for removal. Implemented in DeviceBase.h.
       */
      virtual double GetPixelSizeUm() const = 0;
      /**
       * @brief Return the current binning factor.
       */
      virtual int GetBinning() const = 0;
      /**
       * @brief Set binning factor.
       */
      virtual int SetBinning(int binSize) = 0;
      /**
       * @brief Set exposure in milliseconds.
       */
      virtual void SetExposure(double exp_ms) = 0;
      /**
       * @brief Return the current exposure setting in milliseconds.
       */
      virtual double GetExposure() const = 0;
      /**
       * @brief Set the camera Region Of Interest.
       *
       * Required by the MM::Camera API.
       * This command will change the dimensions of the image.
       * Depending on the hardware capabilities the camera may not be able to configure the
       * exact dimensions requested - but should try do as close as possible.
       * If the hardware does not have this capability the software should simulate the ROI by
       * appropriately cropping each frame.
       *
       * @param x top-left corner coordinate
       * @param y top-left corner coordinate
       * @param xSize width
       * @param ySize height
       */
      virtual int SetROI(unsigned x, unsigned y, unsigned xSize, unsigned ySize) = 0;
      /**
       * @brief Return the actual dimensions of the current ROI.
       */
      virtual int GetROI(unsigned& x, unsigned& y, unsigned& xSize, unsigned& ySize) = 0;
      /**
       * @brief Reset the Region of Interest to full frame.
       */
      virtual int ClearROI() = 0;
      virtual bool SupportsMultiROI() = 0;
      virtual bool IsMultiROISet() = 0;
      virtual int GetMultiROICount(unsigned& count) = 0;
      virtual int SetMultiROI(const unsigned* xs, const unsigned* ys,
              const unsigned* widths, const unsigned* heights,
              unsigned numROIs) = 0;
      virtual int GetMultiROI(unsigned* xs, unsigned* ys, unsigned* widths,
              unsigned* heights, unsigned* length) = 0;
      /**
       * @brief Start continuous acquisition.
       */
      virtual int StartSequenceAcquisition(long numImages, double interval_ms, bool stopOnOverflow) = 0;
      /**
       * @brief Start Sequence Acquisition with given interval.
       *
       * Most camera adapters will ignore this number.
       */
      virtual int StartSequenceAcquisition(double interval_ms) = 0;
      /**
       * @brief Stop an ongoing sequence acquisition.
       */
      virtual int StopSequenceAcquisition() = 0;
      /**
       * @brief Set up the camera so that Sequence acquisition can start without delay.
       */
      virtual int PrepareSequenceAcqusition() = 0;
      /**
       * @brief Indicate whether sequence acquisition is currently running.
       *
       * Returns true when sequence acquisition is active, false otherwise.
       */
      virtual bool IsCapturing() = 0;

      /**
       * @brief Get the metadata tags stored in this device.
       *
       * These tags will automatically be add to the metadata of an image inserted
       * into the circular buffer.
       */
      virtual void GetTags(char* serializedMetadata) = 0;

      /**
       * @brief Add new tag or modify the value of an existing one.
       *
       * These will automatically be added to images inserted into the circular buffer.
       * Use this mechanism for tags that do not change often.  For metadata that
       * change often, create an instance of metadata yourself and add to one of
       * the versions of the InsertImage function.
       */
      virtual void AddTag(const char* key, const char* deviceLabel, const char* value) = 0;

      /**
       * @brief Remove an existing tag from the metadata associated with this device.
       *
       * These tags will automatically be add to the metadata of an image inserted
       * into the circular buffer.
       */
      virtual void RemoveTag(const char* key) = 0;

      /**
       * @brief Return whether a camera's exposure time can be sequenced.
       *
       * If returning true, then a Camera adapter class should also inherit
       * the SequenceableExposure class and implement its methods.
       */
      virtual int IsExposureSequenceable(bool& isSequenceable) const = 0;

      // Sequence functions
      // Sequences can be used for fast acquisitions, synchronized by TTLs rather than
      // computer commands.
      // Sequences of exposures can be uploaded to the camera.  The camera will cycle through
      // the uploaded list of exposures (triggered by either an internal or
      // external trigger).  If the device is capable (and ready) to do so isSequenceable will
      // be true. If your device can not execute this (true for most cameras)
      // simply set IsExposureSequenceable to false
      virtual int GetExposureSequenceMaxLength(long& nrEvents) const = 0;
      virtual int StartExposureSequence() = 0;
      virtual int StopExposureSequence() = 0;
      // Remove all values in the sequence
      virtual int ClearExposureSequence() = 0;
      // Add one value to the sequence
      virtual int AddToExposureSequence(double exposureTime_ms) = 0;
      // Signal that we are done sending sequence values so that the adapter can send the whole sequence to the device
      virtual int SendExposureSequence() const = 0;
   };

   /**
    * @brief Shutter API.
    */
   class Shutter : public Device
   {
   public:
      Shutter() {}
      virtual ~Shutter() {}

      // Device API
      virtual DeviceType GetType() const { return Type; }
      static const DeviceType Type;

      // Shutter API
      virtual int SetOpen(bool open = true) = 0;
      virtual int GetOpen(bool& open) = 0;
      /**
       * @brief Open the shutter for the given duration, then close it again.
       *
       * Currently not implemented in any shutter adapters.
       */
      virtual int Fire(double deltaT) = 0;
   };

   /**
    * @brief Single axis stage API.
    */
   class Stage : public Device
   {
   public:
      Stage() {}
      virtual ~Stage() {}

      // Device API
      virtual DeviceType GetType() const { return Type; }
      static const DeviceType Type;

      // Stage API
      virtual int SetPositionUm(double pos) = 0;
      virtual int SetRelativePositionUm(double d) = 0;
      virtual int Move(double velocity) = 0;
      virtual int Stop() = 0;
      virtual int Home() = 0;
      virtual int SetAdapterOriginUm(double d) = 0;
      virtual int GetPositionUm(double& pos) = 0;
      virtual int SetPositionSteps(long steps) = 0;
      virtual int GetPositionSteps(long& steps) = 0;
      virtual int SetOrigin() = 0;
      virtual int GetLimits(double& lower, double& upper) = 0;

      /**
       * @brief Return the focus direction.
       *
       * Indicates whether increasing position corresponds to movement in the
       * direction that brings the objective and sample closer together.
       *
       * Unless the direction is known for sure (does not depend on e.g. how
       * the hardware is installed), this function must return
       * FocusDirectionUnknown (the application can allow the user to specify
       * the direction as needed).
       *
       * Non-focus single-axis stages must also return FocusDirectionUnknown.
       */
      virtual int GetFocusDirection(FocusDirection& direction) = 0;

      /**
       * @brief Indicate whether the stage can be sequenced (synchronized by
       * TTLs).
       *
       * If true, the following methods must be implemented:
       * GetStageSequenceMaxLength(), StartStageSequence(), StopStageSequence(),
       * ClearStageSequence(), AddToStageSequence(), and SendStageSequence().
       */
      virtual int IsStageSequenceable(bool& isSequenceable) const = 0;

      /**
       * @brief Indicate whether the stage can perform linear TTL sequencing.
       *
       * Linear sequencing uses a delta and count instead of an arbitrary list
       * of positions. If true, the following methods must be implemented:
       * SetStageLinearSequence(), StartStageSequence(), StopStageSequence().
       */
      virtual int IsStageLinearSequenceable(bool& isSequenceable) const = 0;

      // Check if a stage has continuous focusing capability (positions can be set while continuous focus runs).
      virtual bool IsContinuousFocusDrive() const = 0;

      // Sequence functions
      // Sequences can be used for fast acquisitions, synchronized by TTLs rather than
      // computer commands.
      // Sequences of positions can be uploaded to the stage.  The device will cycle through
      // the uploaded list of states (triggered by an external trigger - most often coming
      // from the camera).  If the device is capable (and ready) to do so isSequenceable will
      // be true. If your device can not execute this (true for most stages)
      // simply set isSequenceable to false
      virtual int GetStageSequenceMaxLength(long& nrEvents) const = 0;
      virtual int StartStageSequence() = 0;
      virtual int StopStageSequence() = 0;
      /**
       * @brief Remove all values in the sequence.
       */
      virtual int ClearStageSequence() = 0;
      /**
       * @brief Add one value to the sequence.
       */
      virtual int AddToStageSequence(double position) = 0;
      /**
       * @brief Signal that we are done sending sequence values so that the
       * adapter can send the whole sequence to the device.
       */
      virtual int SendStageSequence() = 0;

      /**
       * @brief Set up to perform an equally-spaced triggered Z stack.
       *
       * After calling this function, StartStageSequence() must cause the stage
       * to step by dZ_um on each trigger. On the Nth trigger, the stage must
       * return to the position where it was when StartStageSequence() was
       * called.
       */
      virtual int SetStageLinearSequence(double dZ_um, long nSlices) = 0;
   };

   /**
    * @brief Dual axis stage API.
    */
   class XYStage : public Device
   {
   public:
      XYStage() {}
      virtual ~XYStage() {}

      // Device API
      virtual DeviceType GetType() const { return Type; }
      static const DeviceType Type;

      // XYStage API
      // it is recommended that device adapters implement the  "Steps" methods
      // taking long integers but leave the default implementations (in
      // DeviceBase.h) for the "Um" methods taking doubles. The latter utilize
      // directionality and origin settings set by user and operate via the
      // "Steps" methods. The step size is the inherent minimum distance/step
      // and should be defined by the adapter.
      virtual int SetPositionUm(double x, double y) = 0;
      virtual int SetRelativePositionUm(double dx, double dy) = 0;
      virtual int SetAdapterOriginUm(double x, double y) = 0;
      virtual int GetPositionUm(double& x, double& y) = 0;
      virtual int GetLimitsUm(double& xMin, double& xMax, double& yMin, double& yMax) = 0;
      virtual int Move(double vx, double vy) = 0;

      virtual int SetPositionSteps(long x, long y) = 0;
      virtual int GetPositionSteps(long& x, long& y) = 0;
      virtual int SetRelativePositionSteps(long x, long y) = 0;
      virtual int Home() = 0;
      virtual int Stop() = 0;

      /**
       * @brief Define the current position as the (hardware) origin (0, 0).
       */
      virtual int SetOrigin() = 0;

      /**
       * @brief Define the current position as X = 0 (in hardware if possible).
       *
       * Do not alter the Y coordinates.
       */
      virtual int SetXOrigin() = 0;

      /**
       * @brief Define the current position as Y = 0 (in hardware if possible).
       *
       * Do not alter the X coordinates.
       */
      virtual int SetYOrigin() = 0;

      virtual int GetStepLimits(long& xMin, long& xMax, long& yMin, long& yMax) = 0;
      virtual double GetStepSizeXUm() = 0;
      virtual double GetStepSizeYUm() = 0;
      /**
       * @brief Return whether a stage can be sequenced (synchronized by TTLs).
       *
       * If returning true, then an XYStage class should also inherit
       * the SequenceableXYStage class and implement its methods.
       */
      virtual int IsXYStageSequenceable(bool& isSequenceable) const = 0;
      // Sequence functions
      // Sequences can be used for fast acquisitions, synchronized by TTLs rather than
      // computer commands.
      // Sequences of positions can be uploaded to the XY stage.  The device will cycle through
      // the uploaded list of states (triggered by an external trigger - most often coming
      // from the camera).  If the device is capable (and ready) to do so isSequenceable will
      // be true. If your device can not execute this (true for most XY stages
      // simply set isSequenceable to false
      virtual int GetXYStageSequenceMaxLength(long& nrEvents) const = 0;
      virtual int StartXYStageSequence() = 0;
      virtual int StopXYStageSequence() = 0;
      /**
       * @brief Remove all values in the sequence.
       */
      virtual int ClearXYStageSequence() = 0;
      /**
       * @brief Add one value to the sequence.
       */
      virtual int AddToXYStageSequence(double positionX, double positionY) = 0;
      /**
       * @brief Signal that we are done sending sequence values so that the
       * adapter can send the whole sequence to the device.
       */
      virtual int SendXYStageSequence() = 0;

   };

   /**
    * @brief State device API, e.g. filter wheel, objective turret, etc.
    */
   class State : public Device
   {
   public:
      State() {}
      virtual ~State() {}

      // MMDevice API
      virtual DeviceType GetType() const { return Type; }
      static const DeviceType Type;

      // MMStateDevice API
      virtual int SetPosition(long pos) = 0;
      virtual int SetPosition(const char* label) = 0;
      virtual int GetPosition(long& pos) const = 0;
      virtual int GetPosition(char* label) const = 0;
      virtual int GetPositionLabel(long pos, char* label) const = 0;
      virtual int GetLabelPosition(const char* label, long& pos) const = 0;
      virtual int SetPositionLabel(long pos, const char* label) = 0;
      virtual unsigned long GetNumberOfPositions() const = 0;
      virtual int SetGateOpen(bool open = true) = 0;
      virtual int GetGateOpen(bool& open) = 0;
   };

   /**
    * @brief Serial port API.
    */
   class Serial : public Device
   {
   public:
      Serial() {}
      virtual ~Serial() {}

      // MMDevice API
      virtual DeviceType GetType() const { return Type; }
      static const DeviceType Type;

      // Serial API
      virtual PortType GetPortType() const = 0;
      virtual int SetCommand(const char* command, const char* term) = 0;
      virtual int GetAnswer(char* txt, unsigned maxChars, const char* term) = 0;
      virtual int Write(const unsigned char* buf, unsigned long bufLen) = 0;
      virtual int Read(unsigned char* buf, unsigned long bufLen, unsigned long& charsRead) = 0;
      virtual int Purge() = 0;
   };

   /**
    * @brief Auto-focus device API.
    */
   class AutoFocus : public Device
   {
   public:
      AutoFocus() {}
      virtual ~AutoFocus() {}

      // MMDevice API
      virtual DeviceType GetType() const { return Type; }
      static const DeviceType Type;

      // AutoFocus API
      virtual int SetContinuousFocusing(bool state) = 0;
      virtual int GetContinuousFocusing(bool& state) = 0;
      virtual bool IsContinuousFocusLocked() = 0;
      virtual int FullFocus() = 0;
      virtual int IncrementalFocus() = 0;
      virtual int GetLastFocusScore(double& score) = 0;
      virtual int GetCurrentFocusScore(double& score) = 0;
      virtual int AutoSetParameters() = 0;
      virtual int GetOffset(double &offset) = 0;
      virtual int SetOffset(double offset) = 0;
   };

   /**
    * @brief Image processor API.
    */
   class ImageProcessor : public Device
   {
      public:
         ImageProcessor() {}
         virtual ~ImageProcessor() {}

      // MMDevice API
      virtual DeviceType GetType() const { return Type; }
      static const DeviceType Type;

      // image processor API
      virtual int Process(unsigned char* buffer, unsigned width, unsigned height, unsigned byteDepth) = 0;


   };

   /**
    * @brief ADC and DAC interface.
    */
   class SignalIO : public Device
   {
   public:
      SignalIO() {}
      virtual ~SignalIO() {}

      // MMDevice API
      virtual DeviceType GetType() const { return Type; }
      static const DeviceType Type;

      // signal io API
      virtual int SetGateOpen(bool open = true) = 0;
      virtual int GetGateOpen(bool& open) = 0;
      virtual int SetSignal(double volts) = 0;
      virtual int GetSignal(double& volts) = 0;
      virtual int GetLimits(double& minVolts, double& maxVolts) = 0;

      /**
       * @brief Indicate whether or not this DA device accepts sequences.
       *
       * If the device is sequenceable, it is usually best to add a property
       * through which the user can set "isSequenceable", since only the user
       * knows whether the device is actually connected to a trigger source.
       * If isDASequenceable returns true, the device adapter must also inherit
       * the SequenceableDA class and provide method implementations.
       *
       * @param isSequenceable signals whether other sequence functions will
       *        work
       *
       * @return errorcode (DEVICE_OK if no error)
       */
      virtual int IsDASequenceable(bool& isSequenceable) const = 0;

      // Sequence functions
      // Sequences can be used for fast acquisitions, synchronized by TTLs rather than
      // computer commands.
      // Sequences of voltages can be uploaded to the DA.  The device will cycle through
      // the uploaded list of voltages (triggered by an external trigger - most often coming
      // from the camera).  If the device is capable (and ready) to do so isSequenceable will
      // be true. If your device can not execute this simply set isSequenceable to false
      /**
       * @brief Return the maximum length of a sequence that the hardware can store.
       *
       * @param nrEvents max length of sequence
       * @return errorcode (DEVICE_OK if no error)
       */
      virtual int GetDASequenceMaxLength(long& nrEvents) const = 0;
      /**
       * @brief Start running a sequence (i.e., start switching between voltages
       * sent previously, triggered by a TTL).
       *
       * @return errorcode (DEVICE_OK if no error)
       */
      virtual int StartDASequence() = 0;
      /**
       * @brief Stop running the sequence.
       *
       * @return errorcode (DEVICE_OK if no error)
       */
      virtual int StopDASequence() = 0;
      /**
       * @brief Clear the DA sequence from the device and the adapter.
       *
       * If this functions is not called in between running
       * two sequences, it is expected that the same sequence will run twice.
       * To upload a new sequence, first call this functions, then call AddToDASequence(double
       * voltage) as often as needed.
       *
       * @return errorcode (DEVICE_OK if no error)
       */
      virtual int ClearDASequence() = 0;

      /**
       * @brief Add a new data point (voltage) to the sequence.
       *
       * The data point can either be added to a representation of the sequence in the
       * adapter, or it can be directly written to the device.
       *
       * @return errorcode (DEVICE_OK if no error)
       */
      virtual int AddToDASequence(double voltage) = 0;
      /**
       * @brief Send the complete sequence to the device.
       *
       * If the individual data points were already sent to the device, there is
       * nothing to be done.
       *
       * @return errorcode (DEVICE_OK if no error)
       */
      virtual int SendDASequence() = 0;

   };

   /**
    * @brief Devices that can change magnification of the system.
    */
   class Magnifier : public Device
   {
   public:
      Magnifier() {}
      virtual ~Magnifier() {}

      // MMDevice API
      virtual DeviceType GetType() const { return Type; }
      static const DeviceType Type;

      virtual double GetMagnification() = 0;
   };


   /**
    * @brief Spatial Light Modulator (SLM) API.
    *
    * An SLM is a device that can display images.
    * It is expected to represent a rectangular grid (i.e. it has width and height)
    * of pixels that can be either 8 bit or 32 bit.  Illumination (light source on
    * or off) is logically independent of displaying the image.  Likely the most
    * widely used implementation is the GenericSLM.
    */
   class SLM : public Device
   {
   public:
      SLM() {}
      virtual ~SLM() {}

      virtual DeviceType GetType() const { return Type; }
      static const DeviceType Type;

      // SLM API
      /**
       * @brief Load the image into the SLM device adapter.
       */
      virtual int SetImage(unsigned char * pixels) = 0;

      /**
       * @brief Load a 32-bit image into the SLM device adapter.
       */
      virtual int SetImage(unsigned int * pixels) = 0;

      /**
       * @brief Command the SLM to display the loaded image.
       */
      virtual int DisplayImage() = 0;

      /**
       * @brief Command the SLM to display one 8-bit intensity.
       */
      virtual int SetPixelsTo(unsigned char intensity) = 0;

      /**
       * @brief Command the SLM to display one 32-bit color.
       */
      virtual int SetPixelsTo(unsigned char red, unsigned char green, unsigned char blue) = 0;

      /**
       * @brief Command the SLM to turn off after a specified interval.
       */
      virtual int SetExposure(double interval_ms) = 0;

      /**
       * @brief Get the exposure interval of an SLM.
       */
      virtual double GetExposure() = 0;

      /**
       * @brief Get the SLM width in pixels.
       */
      virtual unsigned GetWidth() = 0;

      /**
       * @brief Get the SLM height in pixels.
       */
      virtual unsigned GetHeight() = 0;

      /**
       * @brief Get the SLM number of components (colors).
       */
      virtual unsigned GetNumberOfComponents() = 0;

      /**
       * @brief Get the SLM number of bytes per pixel.
       */
      virtual unsigned GetBytesPerPixel() = 0;

      // SLM Sequence functions
      // Sequences can be used for fast acquisitions, synchronized by TTLs rather than
      // computer commands.
      // Sequences of images can be uploaded to the SLM.  The SLM will cycle through
      // the uploaded list of images (perhaps triggered by an external trigger or by
      // an internal clock.
      // If the device is capable (and ready) to do so IsSLMSequenceable will return
      // be true. If your device can not execute sequences, IsSLMSequenceable returns false.

      /**
       * @brief Indicate whether or not this SLM device accepts sequences.
       *
       * If the device is sequenceable, it is usually best to add a property
       * through which the user can set "isSequenceable", since only the user
       * knows whether the device is actually connected to a trigger source.
       * If IsSLMSequenceable returns true, the device adapter must also
       * implement the sequencing functions for the SLM.
       *
       * @param isSequenceable signals whether other sequence functions will
       *        work
       *
       * @return errorcode (DEVICE_OK if no error)
       */
      virtual int IsSLMSequenceable(bool& isSequenceable) const = 0;

      /**
       * @brief Return the maximum length of a sequence that the hardware can store.
       *
       * @param nrEvents max length of sequence
       * @return errorcode (DEVICE_OK if no error)
       */
      virtual int GetSLMSequenceMaxLength(long& nrEvents) const = 0;

      /**
       * @brief Start running a sequence (i.e., start switching between images
       * sent previously, triggered by a TTL or internal clock).
       *
       * @return errorcode (DEVICE_OK if no error)
       */
      virtual int StartSLMSequence() = 0;

      /**
       * @brief Stop running the sequence.
       *
       * @return errorcode (DEVICE_OK if no error)
       */
      virtual int StopSLMSequence() = 0;

      /**
       * @brief Clear the SLM sequence from the device and the adapter.
       *
       * If this function is not called in between running
       * two sequences, it is expected that the same sequence will run twice.
       * To upload a new sequence, first call this function, then call
       * AddToSLMSequence(image)
       * as often as needed.
       *
       * @return errorcode (DEVICE_OK if no error)
       */
      virtual int ClearSLMSequence() = 0;

      /**
       * @brief Add a new 8-bit projection image to the sequence.
       *
       * The image can either be added to a representation of the sequence in the
       * adapter, or it can be directly written to the device.
       *
       * @param pixels An array of 8-bit pixels whose length matches that expected by the SLM.
       * @return errorcode (DEVICE_OK if no error)
       */
      virtual int AddToSLMSequence(const unsigned char * const pixels) = 0;

      /**
       * @brief Add a new 32-bit (RGB) projection image to the sequence.
       *
       * The image can either be added to a representation of the sequence in the
       * adapter, or it can be directly written to the device.
       *
       * @param pixels An array of 32-bit RGB pixels whose length matches that expected by the SLM.
       * @return errorcode (DEVICE_OK if no error)
       */
      virtual int AddToSLMSequence(const unsigned int * const pixels) = 0;

      /**
       * @brief Send the complete sequence to the device.
       *
       * If the individual images were already sent to the device, there is
       * nothing to be done.
       *
       * @return errorcode (DEVICE_OK if no error)
       */
      virtual int SendSLMSequence() = 0;

   };

   /**
    * @brief Galvo API.
    *
    * A Galvo in Micro-Manager is a two-axis (conveniently labeled x and y) that can illuminate
    * a sample in the microscope.  It therefore also has the capability to switch a light source
    * on and off (note that this functionality can be offloaded to a shutter device
    * that can be obtained through a callback).  Galvos can illuminate a point, or
    * possibly be directed to illuminate a polygon by scanning the two axis and controlling
    * the light source so that only the area with the polygon is illuminated.
    * Currently known implementations are Utilities-DAGalvo (which uses two DAs to
    * control a Galvo), Democamera-Galvo, ASITiger-ASIScanner, and Rapp.
    * There is no integration with a detector as would be needed for a confocal microscope,
    * and there is also no support for waveforms.
    */
   class Galvo : public Device
   {
   public:
      Galvo() {}
      virtual ~Galvo() {}

      virtual DeviceType GetType() const { return Type; }
      static const DeviceType Type;

   //Galvo API:

      /**
       * @brief Move the galvo devices to the requested position, activate the
       * light source, wait for the specified amount of time (in microseconds),
       * and deactivate the light source.
       *
       * @return errorcode (DEVICE_OK if no error)
       */
      virtual int PointAndFire(double x, double y, double time_us) = 0;
      /**
       * @brief Set the spot interval time.
       *
       * This function seems to be misnamed.  Its name suggest that it is the
       * interval between illuminating two consecutive spots, but in practice it
       * is used to set the time a single spot is illuminated (and the time
       * to move between two spots is usually extremely short).
       *
       * @return errorcode (DEVICE_OK if no error)
       */
      virtual int SetSpotInterval(double pulseInterval_us) = 0;
      /**
       * @brief Set the position of the two axes of the Galvo device in native
       * unit (usually through a voltage that controls the galvo position).
       *
       * @return errorcode (DEVICE_OK if no error)
       */
      virtual int SetPosition(double x, double y) = 0;
      /**
       * @brief Return the current position of the two axes (usually the last
       * position that was set, although this may be different for Galvo devices
       * that also can be controlled through another source).
       *
       * @return errorcode (DEVICE_OK if no error)
       */
      virtual int GetPosition(double& x, double& y) = 0;
      /**
       * @brief Switch the light source under control of this device on or off.
       *
       * If light control through a Shutter device is desired, a property should
       * be added that can be set to the name of the lightsource.
       *
       * @return errorcode (DEVICE_OK if no error)
       */
      virtual int SetIlluminationState(bool on) = 0;
      /**
       * @brief Return the X range of the device in native units.
       */
      virtual double GetXRange() = 0;
      /**
       * @brief Return the minimum X value for the device in native units.
       *
       * Must be implemented if it is not 0.0.
       */
      virtual double GetXMinimum() = 0;
      /**
       * @brief Return the Y range of the device in native units.
       */
      virtual double GetYRange() = 0;
      /**
       * @brief Return the minimum Y value for the device in native units.
       *
       * Must be implemented if it is not 0.0.
       */
      virtual double GetYMinimum() = 0;
      /**
       * @brief Add a vertex point to a polygon.
       *
       * A galvo device in principle can draw arbitrary polygons.  Polygons are
       * added added here point by point.  There is nothing in the API that prevents
       * adding polygons in random order, but most implementations so far
       * do not deal with that well (i.e. expect polygons to be added in incremental
       * order).  Vertex points are added in order and can not be modified through the API
       * after adding (only way is to delete all polygons and start anew).
       *
       * @return errorcode (DEVICE_OK if no error)
       */
      virtual int AddPolygonVertex(int polygonIndex, double x, double y) = 0;
      /**
       * @brief Delete all polygons previously stored in the device adapter.
       *
       * @return errorcode (DEVICE_OK if no error)
       */
      virtual int DeletePolygons() = 0;
      /**
       * @brief Run a TTL-triggered polygon sequence.
       *
       * Presumably the idea of this function is to have the Galvo draw the
       * each polygon in the pre-loaded sequence after its controller receives
       * a TTL trigger.  This is not likely to be supported by all Galvo devices.
       * There currently is no API method to query whether Sequences are supported.
       * When the number of TTLs exceeds the number of polygons, the desired behavior
       * is to repeat the sequence from the beginning.
       *
       * @return errorcode (DEVICE_OK if no error)
       */
      virtual int RunSequence() = 0;
      /**
       * @brief Transfer the polygons from the device adapter memory to the Galvo controller.
       *
       * Should be called before RunPolygons() or RunSequence(). This is mainly an
       * optimization so that the device adapter does not need to transfer each vertex
       * individually.  Some Galvo device adapters will do nothing in this function.
       *
       * @return errorcode (DEVICE_OK if no error)
       */
      virtual int LoadPolygons() = 0;
      /**
       * @brief Set the number of times the polygons should be displayed in the
       * RunPolygons function.
       *
       * @return errorcode (DEVICE_OK if no error)
       */
      virtual int SetPolygonRepetitions(int repetitions) = 0;
      /**
       * @brief Display each pre-loaded polygon in sequence, each illuminated for
       * pulseinterval_us micro-seconds.
       *
       * @return errorcode (DEVICE_OK if no error)
       */
      virtual int RunPolygons() = 0;
      /**
       * @brief Stop the TTL triggered transitions of drawing polygons started
       * in RunSequence().
       *
       * @return errorcode (DEVICE_OK if no error)
       */
      virtual int StopSequence() = 0;
      /**
       * @brief TODO-BRIEF
       *
       * It is completely unclear what this function is supposed to do.  Deprecate?
       *
       * @return errorcode (DEVICE_OK if no error)
       */
      virtual int GetChannel(char* channelName) = 0;
   };

   /**
    * @brief HUB device.
    *
    * Used for complex uber-device functionality in microscope stands
    * and managing auto-configuration (discovery) of other devices.
    */
   class Hub : public Device
   {
   public:
      Hub() {}
      virtual ~Hub() {}

      // MMDevice API
      virtual DeviceType GetType() const { return Type; }
      static const DeviceType Type;

      /**
       * @brief Instantiate all available child peripheral devices.
       *
       * The implementation must instantiate all available child devices and
       * register them by calling AddInstalledDevice() (currently in HubBase).
       *
       * Instantiated peripherals are owned by the Core, and will be destroyed
       * by calling the usual ModuleInterface DeleteDevice() function.
       *
       * The result of calling this function more than once for a given hub
       * instance is undefined.
       */
      virtual int DetectInstalledDevices() = 0;

      /**
       * @brief Remove all Device instances that were created by
       * DetectInstalledDevices(). Not used.
       *
       * Note: Device adapters traditionally call this function at the
       * beginning of their DetectInstalledDevices implementation. This is not
       * necessary but is permissible.
       */
      virtual void ClearInstalledDevices() = 0;

      /**
       * @brief Return the number of child Devices after DetectInstalledDevices
       * was called.
       *
       * Must not be called from device adapters.
       */
      virtual unsigned GetNumberOfInstalledDevices() = 0;

      /**
       * @brief Return a pointer to the Device with index devIdx.
       *
       * 0 <= devIdx < GetNumberOfInstalledDevices().
       *
       * Must not be called from device adapters.
       */
      virtual Device* GetInstalledDevice(int devIdx) = 0;
   };

   /**
    * @brief Pressure Pump API.
    */
   class PressurePump : public Device
   {
   public:
       PressurePump() {}
       virtual ~PressurePump() {}

       // MMDevice API
       virtual DeviceType GetType() const { return Type; }
       static const DeviceType Type;

       /**
        * @brief Stop the pump.
        *
        * The implementation should halt any dispensing/withdrawal,
        * and make the pump available again (make Busy() return false).
        *
        * Required function of PressurePump API.
        */
       virtual int Stop() = 0;

       /**
        * @brief Calibrate the pressure controller.
        *
        * If no internal calibration is supported, just return
        * DEVICE_UNSUPPORTED_COMMAND.
        *
        * Optional function of PressurePump API.
        */
       virtual int Calibrate() = 0;

       /**
        * @brief Return whether the pressure controller is functional before
        * calibration, or it needs to undergo internal calibration before any
        * commands can be executed.
        *
        * Required function of PressurePump API.
        */
       virtual bool RequiresCalibration() = 0;

       /**
        * @brief Set the pressure of the pressure controller.
        *
        * The provided value will be in kPa. The implementation should convert
        * the unit from kPa to the desired unit by the device.
        *
        * Required function of PressurePump API.
        */
       virtual int SetPressureKPa(double pressureKPa) = 0;

       /**
        * @brief Get the pressure of the pressure controller.
        *
        * The returned value has to be in kPa. The implementation, therefore,
        * should convert the value provided by the pressure controller to kPa.
        *
        * Required function of PressurePump API.
        */
       virtual int GetPressureKPa(double& pressureKPa) = 0;
   };

   /**
    * @brief Volumetric Pump API.
    */
   class VolumetricPump : public Device
   {
   public:
       VolumetricPump() {}
       virtual ~VolumetricPump() {}

       // MMDevice API
       virtual DeviceType GetType() const { return Type; }
       static const DeviceType Type;

       /**
        * @brief Home the pump.
        *
        * If no homing is supported, just return DEVICE_UNSUPPORTED_COMMAND.
        *
        * Optional function of VolumetricPump API.
        */
       virtual int Home() = 0;

       /**
        * @brief Stop the pump.
        *
        * The implementation should halt any dispensing/withdrawal,
        * and make the pump available again (make Busy() return false).
        *
        * Required function of VolumetricPump API.
        */
       virtual int Stop() = 0;

       /**
        * @brief Check whether the pump requires homing before being operational.
        *
        * Required function of VolumetricPump API.
        */
       virtual bool RequiresHoming() = 0;

       /**
        * @brief Set the direction of the pump.
        *
        * Certain pumps (e.g. peristaltic and DC pumps) don't have an apriori
        * forward-reverse direction, as it depends on how it is connected. This
        * function allows you to switch forward and reverse.
        *
        * If the pump is uni-directional, this function does not need to be
        * implemented (return DEVICE_UNSUPPORTED_COMMAND).
        *
        * Optional function of VolumetricPump API.
        */
       virtual int InvertDirection(bool inverted) = 0;

       /**
        * @brief Check whether the direction of the pump is inverted.
        *
        * Certain pumps (e.g. peristaltic and DC pumps) don't have an apriori
        * forward-reverse direction, as it depends on how it is connected.
        *
        * When the pump is uni-directional, this function should always assign
        * false to `inverted`.
        *
        * Required function of VolumetricPump API.
        */
       virtual int IsDirectionInverted(bool& inverted) = 0;

       /**
        * @brief Set the current volume of the pump in microliters (uL).
        *
        * Required function of VolumetricPump API.
        */
       virtual int SetVolumeUl(double volUl) = 0;

       /**
        * @brief Get the current volume of the pump in microliters (uL).
        *
        * Required function of VolumetricPump API.
        */
       virtual int GetVolumeUl(double& volUl) = 0;

       /**
        * @brief Set the maximum volume of the pump in microliters (uL).
        *
        * Required function of VolumetricPump API.
        */
       virtual int SetMaxVolumeUl(double volUl) = 0;

       /**
        * @brief Get the maximum volume of the pump in microliters (uL).
        *
        * Required function of VolumetricPump API.
        */
       virtual int GetMaxVolumeUl(double& volUl) = 0;

       /**
        * @brief Set the flowrate in microliter (uL) per second.
        *
        * The implementation should convert the provided flowrate to whichever
        * unit the pump desires (steps/s, mL/h, V).
        *
        * Required function of VolumetricPump API.
        */
       virtual int SetFlowrateUlPerSecond(double flowrate) = 0;

       /**
        * @brief Get the flowrate in microliter (uL) per second.
        *
        * Required function of VolumetricPump API.
        */
       virtual int GetFlowrateUlPerSecond(double& flowrate) = 0;

       /**
        * @brief Start dispensing/withdrawing until the minimum or maximum volume
        * has been reached, or the pumping is manually stopped.
        *
        * Required function of VolumetricPump API.
        */
       virtual int Start() = 0;

       /**
        * @brief Dispense/withdraw for the provided time.
        *
        * Uses the flowrate provided by GetFlowrate_uLperMin.
        * Dispensing for an undetermined amount of time can be done with DBL_MAX.
        * During the dispensing/withdrawal, Busy() should return "true".
        *
        * Required function of VolumetricPump API.
        */
       virtual int DispenseDurationSeconds(double durSec) = 0;

       /**
        * @brief Dispense/withdraw the provided volume.
        *
        * The implementation should cause positive volumes to be dispensed, whereas
        * negative volumes should be withdrawn. The implementation should prevent
        * the volume to go negative (i.e. stop the pump once the syringe is empty),
        * or to go over the maximum volume (i.e. stop the pump once it is full).
        * This automatically allows for dispensing/withdrawal for an undetermined
        * amount of time by providing DBL_MAX for dispense, and DBL_MIN for
        * withdraw.
        *
        * During the dispensing/withdrawal, Busy() should return "true".
        *
        * Required function of VolumetricPump API.
        */
       virtual int DispenseVolumeUl(double volUl) = 0;
   };


   /**
    * @brief Callback API to the core control module.
    *
    * Devices use this abstract interface to use Core services.
    */
   class Core
   {
   public:
      Core() {}
      virtual ~Core() {}

      /**
       * @brief Log a message (msg) in the Corelog output, labeled with the device
       * name (derived from caller).
       *
       * If debugOnly flag is true, the output will only be logged if the
       * general system has been set to output debug logging.
       */
      virtual int LogMessage(const Device* caller, const char* msg, bool debugOnly) const = 0;
      /**
       * @brief Get a pointer to another device.
       *
       * Be aware of potential threading issues. Provide a valid label for the
       * device and receive a pointer to the desired device.
       */
      virtual Device* GetDevice(const Device* caller, const char* label) = 0;
      virtual int GetDeviceProperty(const char* deviceName, const char* propName, char* value) = 0;
      virtual int SetDeviceProperty(const char* deviceName, const char* propName, const char* value) = 0;

      /**
       * @brief Get the names of currently loaded devices of a given type.
       *
       * If deviceIterator exceeds or is equal to the number of currently
       * loaded devices of type devType, an empty string is returned.
       *
       * @param caller the calling device
       * @param devType the device type
       * @param pDeviceName buffer in which device name will be returned
       * @param deviceIterator index of device (within the given type)
       */
      virtual void GetLoadedDeviceOfType(const Device* caller, MM::DeviceType devType, char* pDeviceName, const unsigned int deviceIterator) = 0;

      virtual int SetSerialProperties(const char* portName,
                                      const char* answerTimeout,
                                      const char* baudRate,
                                      const char* delayBetweenCharsMs,
                                      const char* handshaking,
                                      const char* parity,
                                      const char* stopBits) = 0;
      virtual int SetSerialCommand(const Device* caller, const char* portName, const char* command, const char* term) = 0;
      virtual int GetSerialAnswer(const Device* caller, const char* portName, unsigned long ansLength, char* answer, const char* term) = 0;
      virtual int WriteToSerial(const Device* caller, const char* port, const unsigned char* buf, unsigned long length) = 0;
      virtual int ReadFromSerial(const Device* caller, const char* port, unsigned char* buf, unsigned long length, unsigned long& read) = 0;
      virtual int PurgeSerial(const Device* caller, const char* portName) = 0;
      virtual MM::PortType GetSerialPortType(const char* portName) const = 0;

      virtual int OnPropertiesChanged(const Device* caller) = 0;
      /**
       * @brief Signal the UI that a property changed.
       *
       * The Core will check if groups or pixel size changed as a consequence of
       * the change of this property and inform the UI.
       */
      virtual int OnPropertyChanged(const Device* caller, const char* propName, const char* propValue) = 0;
      /**
       * @brief Signal the UI when the stage has reached a new position.
       */
      virtual int OnStagePositionChanged(const Device* caller, double pos) = 0;
      /**
       * @brief Signal the UI when the XY stage has reached a new position.
       */
      virtual int OnXYStagePositionChanged(const Device* caller, double xPos, double yPos) = 0;
      /**
       * @brief Inform the UI when the exposure time has changed.
       */
      virtual int OnExposureChanged(const Device* caller, double newExposure) = 0;
      /**
       * @brief Inform the UI when the SLM exposure time has changed.
       */
      virtual int OnSLMExposureChanged(const Device* caller, double newExposure) = 0;
      /**
       * @brief Signal changes in magnification.
       */
      virtual int OnMagnifierChanged(const Device* caller) = 0;
      /**
       * @brief Signal that the shutter opened or closed.
       */
      virtual int OnShutterOpenChanged(const Device* caller, bool open) = 0;

      // Deprecated: Return value overflows in ~72 minutes on Windows.
      // Prefer std::chrono::steady_clock for time delta measurements.
      virtual unsigned long GetClockTicksUs(const Device* caller) = 0;

      // Returns monotonic MMTime suitable for time delta measurements.
      // Time zero is not fixed and may change on every launch.
      // Prefer std::chrono::steady_clock::now() in new code.
      virtual MM::MMTime GetCurrentMMTime() = 0;

      // sequence acquisition
      virtual int AcqFinished(const Device* caller, int statusCode) = 0;
      virtual int PrepareForAcq(const Device* caller) = 0;

      /**
       * @brief Send a frame to the Core during sequence acquisition.
       *
       * Cameras must call this function during sequence acquisition to send
       * each frame to the Core.
       *
       * byteDepth: 1 or 2 for grayscale images; 4 for BGR_
       *
       * nComponents: 1 for grayscale; 4 for BGR_ (_: unused component)
       *
       * serializedMetadata: must be the result of md.serialize().c_str() (md
       *                     being an instance of Metadata)
       *
       * doProcess: must be true, except for the case mentioned below
       *
       * Legacy note: Previously, cameras were required to perform special
       * handling when InsertImage() returns DEVICE_BUFFER_OVERFLOW and
       * stopOnOverflow == false. However, InsertImage() no longer ever
       * returns that particular error when stopOnOverflow == false. So
       * cameras should always just stop the acquisition if InsertImage()
       * returns any error.
       */
      virtual int InsertImage(const Device* caller, const unsigned char* buf, unsigned width, unsigned height, unsigned byteDepth, unsigned nComponents, const char* serializedMetadata, const bool doProcess = true) = 0;

      /**
       * @brief Send a grayscale frame to the Core during sequence acquisition.
       *
       * Same as the overload with the added nComponents parameter.
       * Assumes nComponents == 1 (grayscale).
       */
      virtual int InsertImage(const Device* caller, const unsigned char* buf, unsigned width, unsigned height, unsigned byteDepth, const char* serializedMetadata = nullptr, const bool doProcess = true) = 0;

      /**
       * @brief Prepare the sequence buffer for the given image size and pixel format.
       *
       * Cameras normally do not need to call this explicitly.
       * 'channels' is ignored (should be 1) and 'slices' must be 1.
       */
      virtual bool InitializeImageBuffer(unsigned channels, unsigned slices, unsigned int w, unsigned int h, unsigned int pixDepth) = 0;

      // These functions violate the separation between device adapters and
      // will be removed as soon as we remove all uses. Never use in new code.
      MMDEVICE_DEPRECATED virtual int GetFocusPosition(double& pos) = 0;
      MMDEVICE_DEPRECATED virtual MM::SignalIO* GetSignalIODevice(const MM::Device* caller, const char* deviceName) = 0;

      virtual MM::Hub* GetParentHub(const MM::Device* caller) const = 0;
   };

} // namespace MM
