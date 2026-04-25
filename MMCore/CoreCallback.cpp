///////////////////////////////////////////////////////////////////////////////
// FILE:          CoreCallback.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     MMCore
//-----------------------------------------------------------------------------
// DESCRIPTION:   Callback object for MMCore device interface. Encapsulates
//                (bottom) internal API for calls going from devices to the 
//                core.
//
//                This class is essentially an extension of the CMMCore class
//                and has full access to CMMCore private members.
//              
// AUTHOR:        Nenad Amodaj, nenad@amodaj.com, 01/05/2007
//
// COPYRIGHT:     University of California, San Francisco, 2007-2014
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

#include "CircularBuffer.h"
#include "CoreCallback.h"
#include "DeviceManager.h"
#include "Notification.h"
#include "SerializedMetadata.h"
#include "SynchronizedConfiguration.h"

#include "DeviceUtils.h"

#include <cassert>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <string>
#include <vector>
#include <algorithm>

namespace notif = mmcore::internal::notification;

namespace mmcore {
namespace internal {


CoreCallback::CoreCallback(CMMCore* c) :
   core_(c),
   startTime_(std::chrono::steady_clock::now())
{
   assert(core_);
}


void CoreCallback::ResetImageInsertionState()
{
   std::lock_guard<std::mutex> guard(imageInsertionStateMutex_);
   imageNumbers_.clear();
   startTime_ = std::chrono::steady_clock::now();
}


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


CoreCallback::~CoreCallback() = default;


int
CoreCallback::LogMessage(const MM::Device* caller, const char* msg,
      bool debugOnly) const
{
   std::shared_ptr<DeviceInstance> device;
   try
   {
      device = core_->deviceManager_->GetDevice(caller);
   }
   catch (const CMMError&)
   {
      LOG_ERROR(core_->coreLogger_) <<
         "Attempt to log message from unregistered device: " << msg;
      return DEVICE_OK;
   }
   return device->LogMessage(msg, debugOnly);
}


MM::Device*
CoreCallback::GetDevice(const MM::Device* caller, const char* label)
{
   if (!caller || !label)
      return 0;

   try
   {
      MM::Device* pDevice = core_->deviceManager_->GetDevice(label)->GetRawPtr();
      if (pDevice == caller)
         return 0;
      return pDevice;
   }
   catch (const CMMError&)
   {
      return 0;
   }
}


MM::PortType
CoreCallback::GetSerialPortType(const char* portName) const
{
   std::shared_ptr<SerialInstance> pSerial;
   try
   {
      pSerial = core_->deviceManager_->GetDeviceOfType<SerialInstance>(portName);
   }
   catch (...)
   {
      return MM::InvalidPort;
   }

   return pSerial->GetPortType();
}


MM::ImageProcessor*
CoreCallback::GetImageProcessor(const MM::Device*)
{
   std::shared_ptr<ImageProcessorInstance> imageProcessor =
      core_->currentImageProcessor_.lock();
   if (imageProcessor)
   {
      return imageProcessor->GetRawPtr();
   }
   return 0;
}


MM::SignalIO*
CoreCallback::GetSignalIODevice(const MM::Device*, const char* label)
{
   try {
      return core_->deviceManager_->
         GetDeviceOfType<SignalIOInstance>(label)->GetRawPtr();
   }
   catch (const CMMError&)
   {
      return 0;
   }
}


MM::Hub*
CoreCallback::GetParentHub(const MM::Device* caller) const
{
   if (caller == 0)
      return 0;

   std::shared_ptr<HubInstance> hubDevice;
   try
   {
      hubDevice = core_->deviceManager_->GetParentDevice(core_->deviceManager_->GetDevice(caller));
   }
   catch (const CMMError&)
   {
      return 0;
   }
   if (hubDevice)
      return hubDevice->GetRawPtr();
   return 0;
}


void
CoreCallback::GetLoadedDeviceOfType(const MM::Device*, MM::DeviceType devType,
      char* deviceName, const unsigned int deviceIterator)
{
   deviceName[0] = 0;
   std::vector<std::string> v = core_->getLoadedDevicesOfType(devType);
   if( deviceIterator < v.size())
      strncpy( deviceName, v.at(deviceIterator).c_str(), MM::MaxStrLength);
   return;
}


void
CoreCallback::Sleep(const MM::Device*, double intervalMs)
{
   CDeviceUtils::SleepMs((long)(0.5 + intervalMs));
}


/**
 * Append the metadata tags attached to device caller to md.
 */
void
CoreCallback::AddCameraMetadata(const MM::Device* caller,
      SerializedMetadata& md)
{
   std::shared_ptr<CameraInstance> camera =
      std::static_pointer_cast<CameraInstance>(
            core_->deviceManager_->GetDevice(caller));

   std::string label = camera->GetLabel();
   md.AddTag(MM::g_Keyword_Metadata_CameraLabel, label);

   std::string serializedMD;
   try
   {
      serializedMD = camera->GetTags();
   }
   catch (const CMMError&)
   {
      return;
   }

   md.AppendSerialized(serializedMD.c_str());
}

int CoreCallback::InsertImage(const MM::Device* caller, const unsigned char* buf,
   unsigned width, unsigned height, unsigned bytesPerPixel,
   const char* serializedMetadata)
{
   return InsertImage(caller, buf, width, height, bytesPerPixel, 1, serializedMetadata);
}

SerializedMetadata
CoreCallback::BuildSequenceImageMetadata(const MM::Device* caller,
   unsigned width, unsigned height,
   unsigned byteDepth, unsigned nComponents,
   const char* origSerializedMd)
{
   SerializedMetadata md(origSerializedMd);
   AddCameraMetadata(caller, md);

   md.AddTag(MM::g_Keyword_Metadata_Width, width);
   md.AddTag(MM::g_Keyword_Metadata_Height, height);

   const char* pixelType = MM::g_Keyword_PixelType_Unknown;
   if (byteDepth == 1)
      pixelType = MM::g_Keyword_PixelType_GRAY8;
   else if (byteDepth == 2)
      pixelType = MM::g_Keyword_PixelType_GRAY16;
   else if (byteDepth == 4)
      pixelType = (nComponents == 1) ? MM::g_Keyword_PixelType_GRAY32
                                     : MM::g_Keyword_PixelType_RGB32;
   else if (byteDepth == 8)
      pixelType = MM::g_Keyword_PixelType_RGB64;
   md.AddTag(MM::g_Keyword_PixelType, pixelType);

   // Note: It is not ideal to use local time. I think this tag is rarely
   // used. Consider replacing with UTC (micro)seconds-since-epoch (with
   // different tag key) after addressing current usage.
   md.AddTag(MM::g_Keyword_Metadata_TimeInCore,
         FormatLocalTime(std::chrono::system_clock::now()));

   {
      std::lock_guard<std::mutex> guard(imageInsertionStateMutex_);
      if (!md.HasTag(MM::g_Keyword_Elapsed_Time_ms))
      {
         using namespace std::chrono;
         auto elapsed = steady_clock::now() - startTime_;
         md.AddTag(MM::g_Keyword_Elapsed_Time_ms,
            std::to_string(duration_cast<milliseconds>(elapsed).count()));
      }

      auto cameraLabel = md.GetTag(MM::g_Keyword_Metadata_CameraLabel);
      assert(cameraLabel.has_value());
      long& counter = imageNumbers_[std::string(*cameraLabel)];
      md.AddTag(MM::g_Keyword_Metadata_ImageNumber,
         CDeviceUtils::ConvertToString(counter));
      ++counter;
   }

   return md;
}

int CoreCallback::InsertImage(const MM::Device* caller, const unsigned char* buf,
   unsigned width, unsigned height, unsigned bytesPerPixel, unsigned nComponents,
   const char* serializedMetadata)
{
   try
   {
      SerializedMetadata md = BuildSequenceImageMetadata(
         caller, width, height, bytesPerPixel, nComponents, serializedMetadata);

      MM::ImageProcessor* ip = GetImageProcessor(caller);
      if (ip != nullptr)
      {
         ip->Process(const_cast<unsigned char*>(buf), width, height, bytesPerPixel);
      }
      if (core_->cbuf_->InsertImage(buf, width, height, bytesPerPixel, nComponents,
            md.View()))
         return DEVICE_OK;
      else
         return DEVICE_BUFFER_OVERFLOW;
   }
   catch (CMMError& /*e*/)
   {
      return DEVICE_INCOMPATIBLE_IMAGE;
   }
}

bool CoreCallback::InitializeImageBuffer(unsigned channels, unsigned slices,
      unsigned int w, unsigned int h, unsigned int pixDepth)
{
   // Multi-channel images were never implemented so 'channels' should be 1,
   // but some cameras confuse it with color components and pass 4.
   (void)channels;

   // Support for multi-slice images has not been implemented
   if (slices != 1)
      return false;

   return core_->cbuf_->Initialize(w, h, pixDepth);
}

int CoreCallback::AcqFinished(const MM::Device* caller, int /*statusCode*/)
{
   std::shared_ptr<DeviceInstance> camera;
   try
   {
      camera = core_->deviceManager_->GetDevice(caller);
   }
   catch (const CMMError&)
   {
      LOG_ERROR(core_->coreLogger_) <<
         "AcqFinished() called from unregistered device";
      return DEVICE_ERR;
   }

   std::shared_ptr<DeviceInstance> currentCamera =
      core_->currentCameraDevice_.lock();

   if (core_->autoShutter_)
   {
      std::shared_ptr<ShutterInstance> shutter =
         core_->currentShutterDevice_.lock();
      if (shutter)
      {
         // We need to lock the shutter's module for thread safety, but there's
         // a case where deadlock would result.
         int sret = DEVICE_ERR;
         if (camera->GetAdapterModule() == shutter->GetAdapterModule())
         {
            // This is a nasty hack to allow the case where the shutter and
            // camera live in the same module. It is not safe, but this is how
            // _all_ cases used to be implemented, and I can't immediately
            // think of a fully safe fix that is reasonably simple.
            sret = shutter->SetOpen(false);
         }
         else if (currentCamera && currentCamera->GetAdapterModule() ==
               shutter->GetAdapterModule())
         {
            // Likewise, we might be called as a result of a call to
            // StopSequenceAcquisition() on a virtual wrapper camera device
            // (such as Multi Camera), in which case we would get a deadlock if
            // the shutter is in the same module as the virtual camera.
            // This is an even nastier hack in that it ignores the possibility
            // of StopSequenceAcquisition() being called on a camera other than
            // currentCamera, but such cases are rare.
            sret = shutter->SetOpen(false);
         }
         else
         {
            // If the shutter is in a different device adapter, it is safe to
            // lock that adapter.
            DeviceModuleLockGuard g(shutter);
            sret = shutter->SetOpen(false);

            // We could wait for the shutter to close here, but the
            // implementation has always returned without waiting. The camera
            // doesn't care, so let's keep the behavior. Thus,
            // stopSequenceAcquisition() does not wait for the shutter before
            // returning.
         }
         if (sret == DEVICE_OK)
            core_->postNotification(notif::ShutterOpenChanged{
               shutter->GetLabel(), false});
      }
   }

   core_->postNotification(
      notif::SequenceAcquisitionStopped{camera->GetLabel()});

   return DEVICE_OK;
}

int CoreCallback::PrepareForAcq(const MM::Device* caller)
{
   if (core_->autoShutter_)
   {
      std::shared_ptr<ShutterInstance> shutter =
         core_->currentShutterDevice_.lock();
      if (shutter)
      {
         int sret;
         {
            DeviceModuleLockGuard g(shutter);
            sret = shutter->SetOpen(true);
         }
         if (sret == DEVICE_OK)
            core_->postNotification(notif::ShutterOpenChanged{
               shutter->GetLabel(), true});
         core_->waitForDevice(shutter);
      }
   }

   char label[MM::MaxStrLength];
   caller->GetLabel(label);
   core_->postNotification(notif::SequenceAcquisitionStarted{label});

   return DEVICE_OK;
}

/**
 * Handler for the property change event from the device.
 */
int CoreCallback::OnPropertiesChanged(const MM::Device* /* caller */)
{
   core_->postNotification(notif::PropertiesChanged{});

   // TODO It is inconsistent that we do not update the system state cache in
   // this case. However, doing so would be time-consuming (if not unsafe).

   return DEVICE_OK;
}

/**
 * Device signals that a specific property changed and reports the new value
 */
int CoreCallback::OnPropertyChanged(const MM::Device* device, const char* propName, const char* value)
{
   std::lock_guard<std::mutex> g(onPropertyChangedLock_);
   char label[MM::MaxStrLength];
   device->GetLabel(label);
   bool readOnly;
   device->GetPropertyReadOnly(propName, readOnly);
   const PropertySetting ps(label, propName, value, readOnly);
   core_->stateCache_->addSetting(ps);
   core_->postNotification(
      notif::PropertyChanged{label, propName, value});

   // Find all configs that contain this property and notify that the
   // config group changed.
   // TODO: Assess whether performance is better by maintaining a map tying
   // property to configurations
   for (const auto& group : core_->getAvailableConfigGroups()) {
      for (const auto& config : core_->getAvailableConfigs(group.c_str())) {
         Configuration configData =
            core_->getConfigData(group.c_str(), config.c_str());
         if (configData.isPropertyIncluded(label, propName)) {
            std::string currentConfig =
               core_->getCurrentConfigFromCache(group.c_str());
            core_->postNotification(
               notif::ConfigGroupChanged{group, currentConfig});
            break;
         }
      }
   }

   // Check if pixel size was potentially affected. If so, update from cache.
   for (const auto& psConfig : core_->getAvailablePixelSizeConfigs()) {
      Configuration pixelSizeConfig =
         core_->getPixelSizeConfigData(psConfig.c_str());
      if (pixelSizeConfig.isPropertyIncluded(label, propName)) {
         double pixSizeUm;
         try {
            pixSizeUm = core_->getPixelSizeUm(true);
            std::vector<double> affine = core_->getPixelSizeAffine(true);
            if (affine.size() == 6) {
               core_->postNotification(notif::PixelSizeAffineChanged{
                  affine[0], affine[1], affine[2],
                  affine[3], affine[4], affine[5]});
            }
         }
         catch (const CMMError&) {
            pixSizeUm = 0.0;
         }
         core_->postNotification(notif::PixelSizeChanged{pixSizeUm});
         break;
      }
   }

   return DEVICE_OK;
}


/**
 * Handler for Stage position update
 */
int CoreCallback::OnStagePositionChanged(const MM::Device* device, double pos)
{
   char label[MM::MaxStrLength];
   device->GetLabel(label);
   core_->postNotification(notif::StagePositionChanged{label, pos});
   return DEVICE_OK;
}

/**
 * Handler for XYStage position update
 */
int CoreCallback::OnXYStagePositionChanged(const MM::Device* device, double xPos, double yPos)
{
   char label[MM::MaxStrLength];
   device->GetLabel(label);
   core_->postNotification(notif::XYStagePositionChanged{label, xPos, yPos});
   return DEVICE_OK;
}

/**
 * Handler for exposure update
 * 
 */
int CoreCallback::OnExposureChanged(const MM::Device* device, double newExposure)
{
   char label[MM::MaxStrLength];
   device->GetLabel(label);
   core_->postNotification(notif::ExposureChanged{label, newExposure});
   return DEVICE_OK;
}

/**
 * Handler for SLM exposure update
 * 
 */
int CoreCallback::OnSLMExposureChanged(const MM::Device* device, double newExposure)
{
   char label[MM::MaxStrLength];
   device->GetLabel(label);
   core_->postNotification(notif::SLMExposureChanged{label, newExposure});
   return DEVICE_OK;
}

/**
 * Handler for magnifier changer
 * 
 */
int CoreCallback::OnMagnifierChanged(const MM::Device* /* device */)
{
   double pixSizeUm;
   try {
      pixSizeUm = core_->getPixelSizeUm(true);
      std::vector<double> affine = core_->getPixelSizeAffine(true);
      if (affine.size() == 6) {
         core_->postNotification(notif::PixelSizeAffineChanged{
            affine[0], affine[1], affine[2],
            affine[3], affine[4], affine[5]});
      }
   }
   catch (const CMMError&) {
      pixSizeUm = 0.0;
   }
   core_->postNotification(notif::PixelSizeChanged{pixSizeUm});
   return DEVICE_OK;
}

/**
 * Handler for Shutter State changes.
 * 
 */
int CoreCallback::OnShutterOpenChanged(const MM::Device* device, bool state)
{
   char label[MM::MaxStrLength];
   device->GetLabel(label);
   core_->postNotification(notif::ShutterOpenChanged{label, state});
   return DEVICE_OK;
}


int CoreCallback::SetSerialProperties(const char* portName,
                                      const char* answerTimeout,
                                      const char* baudRate,
                                      const char* delayBetweenCharsMs,
                                      const char* handshaking,
                                      const char* parity,
                                      const char* stopBits)
{
   try
   {
      core_->setSerialProperties(portName, answerTimeout, baudRate,
         delayBetweenCharsMs, handshaking, parity, stopBits);
   }
   catch (CMMError& e)
   {
      return e.getCode();
   }

   return DEVICE_OK;
}

/**
 * Sends an array of bytes to the port.
 */
int CoreCallback::WriteToSerial(const MM::Device* caller, const char* portName, const unsigned char* buf, unsigned long length)
{
   std::shared_ptr<SerialInstance> pSerial;
   try
   {
      pSerial = core_->deviceManager_->GetDeviceOfType<SerialInstance>(portName);
   }
   catch (CMMError& err)
   {
      return err.getCode();    
   }
   catch (...)
   {
      return DEVICE_SERIAL_COMMAND_FAILED;
   }

   // don't allow self reference
   if (pSerial->GetRawPtr() == caller)
      return DEVICE_SELF_REFERENCE;

   return pSerial->Write(buf, length);
}
   
/**
  * Reads bytes form the port, up to the buffer length.
  */
int CoreCallback::ReadFromSerial(const MM::Device* caller, const char* portName, unsigned char* buf, unsigned long bufLength, unsigned long &bytesRead)
{
   std::shared_ptr<SerialInstance> pSerial;
   try
   {
      pSerial = core_->deviceManager_->GetDeviceOfType<SerialInstance>(portName);
   }
   catch (CMMError& err)
   {
      return err.getCode();    
   }
   catch (...)
   {
      return DEVICE_SERIAL_COMMAND_FAILED;
   }

   // don't allow self reference
   if (pSerial->GetRawPtr() == caller)
      return DEVICE_SELF_REFERENCE;

   return pSerial->Read(buf, bufLength, bytesRead);
}

/**
 * Clears port buffers.
 */
int CoreCallback::PurgeSerial(const MM::Device* caller, const char* portName)
{
   std::shared_ptr<SerialInstance> pSerial;
   try
   {
      pSerial = core_->deviceManager_->GetDeviceOfType<SerialInstance>(portName);
   }
   catch (CMMError& err)
   {
      return err.getCode();    
   }
   catch (...)
   {
      return DEVICE_SERIAL_COMMAND_FAILED;
   }

   // don't allow self reference
   if (pSerial->GetRawPtr() == caller)
      return DEVICE_SELF_REFERENCE;

   return pSerial->Purge();
}

/**
 * Sends an ASCII command terminated by the specified character sequence.
 */
int CoreCallback::SetSerialCommand(const MM::Device*, const char* portName, const char* command, const char* term)
{
   try {
      core_->setSerialPortCommand(portName, command, term);
   }
   catch (...)
   {
      // trap all exceptions and return generic serial error
      return DEVICE_SERIAL_COMMAND_FAILED;
   }
   return DEVICE_OK;
}

/**
 * Receives an ASCII string terminated by the specified character sequence.
 * The terminator string is stripped of the answer. If the termination code is not
 * received within the com port timeout and error will be flagged.
 */
int CoreCallback::GetSerialAnswer(const MM::Device*, const char* portName, unsigned long ansLength, char* answerTxt, const char* term)
{
   std::string answer;
   try {
      answer = core_->getSerialPortAnswer(portName, term);
      if (answer.length() >= ansLength)
         return DEVICE_SERIAL_BUFFER_OVERRUN;
   }
   catch (...)
   {
      // trap all exceptions and return generic serial error
      return DEVICE_SERIAL_COMMAND_FAILED;
   }
   strcpy(answerTxt, answer.c_str());
   return DEVICE_OK;
}

int CoreCallback::GetFocusPosition(double& pos)
{
   std::shared_ptr<StageInstance> focus = core_->currentFocusDevice_.lock();
   if (focus)
   {
      return focus->GetPositionUm(pos);
   }
   pos = 0.0;
   return DEVICE_CORE_FOCUS_STAGE_UNDEF;
}

int CoreCallback::GetDeviceProperty(const char* deviceName, const char* propName, char* value)
{
   try
   {
      std::string propVal = core_->getProperty(deviceName, propName);
      CDeviceUtils::CopyLimitedString(value, propVal.c_str());
   }
   catch(CMMError& e)
   {
      return e.getCode();
   }

   return DEVICE_OK;
}

int CoreCallback::SetDeviceProperty(const char* deviceName, const char* propName, const char* value)
{
   try
   {
      std::string propVal(value);
      core_->setProperty(deviceName, propName, propVal.c_str());
   }
   catch(CMMError& e)
   {
      return e.getCode();
   }

   return DEVICE_OK;
}

static long long SteadyMicroseconds()
{
   using namespace std::chrono;
   auto now = steady_clock::now().time_since_epoch();
   auto usec = duration_cast<microseconds>(now);
   return usec.count();
}

/**
 * Returns the number of microsecond tick
 * N.B. an unsigned long microsecond count rolls over in just over an hour!!!!
 *
 * This method is obsolete and deprecated.
 * Prefer std::chrono::steady_clock for time delta measurements
 */
unsigned long CoreCallback::GetClockTicksUs(const MM::Device* /*caller*/)
{
   return static_cast<unsigned long>(SteadyMicroseconds());
}

MM::MMTime CoreCallback::GetCurrentMMTime()
{
   return MM::MMTime::fromUs(SteadyMicroseconds());
}

} // namespace internal
} // namespace mmcore
