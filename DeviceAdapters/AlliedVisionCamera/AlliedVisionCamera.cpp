/*=============================================================================
  Copyright (C) 2012 - 2023 Allied Vision Technologies.  All Rights Reserved.

  Redistribution of this header file, in original or modified form, without
  prior written consent of Allied Vision Technologies is prohibited.

  THIS SOFTWARE IS PROVIDED BY THE AUTHOR "AS IS" AND ANY EXPRESS OR IMPLIED
  WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF TITLE,
  NON-INFRINGEMENT, MERCHANTABILITY AND FITNESS FOR A PARTICULAR  PURPOSE ARE
  DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
  INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
  AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
  TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

=============================================================================*/
#include "AlliedVisionCamera.h"

#include <Windows.h>

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>

#include "ModuleInterface.h"
#include "VmbC/VmbC.h"

///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////
MODULE_API void InitializeModuleData() {
  g_api = std::make_unique<VimbaXApi>();
  assert(g_api != nullptr);
  auto err = g_api->VmbStartup_t(nullptr);
  assert(err == VmbErrorSuccess);

  err = AlliedVisionCamera::getCamerasList();
  if (err != VmbErrorSuccess) {
    // TODO Handle error
  }

  g_api->VmbShutdown_t();
}

MODULE_API MM::Device* CreateDevice(const char* deviceName) {
  if (deviceName == nullptr) {
    return nullptr;
  }

  return new AlliedVisionCamera(deviceName);
}

MODULE_API void DeleteDevice(MM::Device* pDevice) { delete pDevice; }

///////////////////////////////////////////////////////////////////////////////
// AlliedVisionCamera
///////////////////////////////////////////////////////////////////////////////

AlliedVisionCamera::~AlliedVisionCamera() {
  m_handle = nullptr;

  for (size_t i = 0; i < MAX_FRAMES; i++) {
    delete[] m_buffer[i];
  }
}

AlliedVisionCamera::AlliedVisionCamera(const char* deviceName)
    : CCameraBase<AlliedVisionCamera>(),
      m_handle{nullptr},
      m_cameraName{deviceName},
      m_frames{},
      m_buffer{},
      m_bufferSize{0},
      m_imageWidth{},
      m_imageHeight{},
      m_isAcquisitionRunning{false} {
  // [Rule] Create properties here (pre-init only)
  InitializeDefaultErrorMessages();
  setApiErrorMessages();
}

int AlliedVisionCamera::Initialize() {
  // [Rule] Implement communication here
  LogMessage("Initializing Vimba X API...");
  VmbError_t err = g_api->VmbStartup_t(nullptr);
  if (err != VmbErrorSuccess) {
    return err;
  }

  VmbVersionInfo_t ver;
  err = g_api->VmbVersionQuery_t(&ver, sizeof(ver));
  if (err != VmbErrorSuccess) {
    return err;
  }
  std::string v = std::to_string(ver.major) + "." + std::to_string(ver.minor) +
                  "." + std::to_string(ver.patch);
  LogMessage("SDK version:" + v);

  LogMessage("Opening camera: " + m_cameraName);
  err = g_api->VmbCameraOpen_t(m_cameraName.c_str(),
                               VmbAccessModeType::VmbAccessModeFull, &m_handle);
  if (err != VmbErrorSuccess || m_handle == nullptr) {
    return err;
  }

  // Init properties and buffer
  setupProperties();
  resizeImageBuffer();

  return DEVICE_OK;
}

int AlliedVisionCamera::Shutdown() {
  // [Rule] Implement disconnection here
  LogMessage("Shutting down camera: " + m_cameraName);
  if (m_handle != nullptr) {
    VmbError_t err = g_api->VmbCameraClose_t(m_handle);
    if (err != VmbErrorSuccess) {
      return err;
    }
  }

  g_api->VmbShutdown_t();
  return DEVICE_OK;
}

const unsigned char* AlliedVisionCamera::GetImageBuffer() {
  return reinterpret_cast<VmbUint8_t*>(m_buffer[0]);
}

unsigned AlliedVisionCamera::GetImageWidth() const { return m_imageWidth; }

unsigned AlliedVisionCamera::GetImageHeight() const { return m_imageHeight; }

unsigned AlliedVisionCamera::GetImageBytesPerPixel() const {
  // TODO implement
  return 1;
}

int AlliedVisionCamera::SnapImage() {
  if (m_isAcquisitionRunning) {
    return DEVICE_CAMERA_BUSY_ACQUIRING;
  }
  resizeImageBuffer();

  VmbFrame_t frame;
  frame.buffer = m_buffer[0];
  frame.bufferSize = m_bufferSize;

  VmbError_t err =
      g_api->VmbFrameAnnounce_t(m_handle, &frame, sizeof(VmbFrame_t));
  if (err != VmbErrorSuccess) {
    return err;
  }

  err = g_api->VmbCaptureStart_t(m_handle);
  if (err != VmbErrorSuccess) {
    return err;
  }

  err = g_api->VmbCaptureFrameQueue_t(m_handle, &frame, nullptr);
  if (err != VmbErrorSuccess) {
    return err;
  }

  err = g_api->VmbFeatureCommandRun_t(m_handle, "AcquisitionStart");
  if (err != VmbErrorSuccess) {
    return err;
  }

  err = g_api->VmbCaptureFrameWait_t(m_handle, &frame, 3000);
  if (err != VmbErrorSuccess) {
    return err;
  }

  err = g_api->VmbFeatureCommandRun_t(m_handle, "AcquisitionStop");
  if (err != VmbErrorSuccess) {
    return err;
  }

  err = g_api->VmbCaptureEnd_t(m_handle);
  if (err != VmbErrorSuccess) {
    return err;
  }

  err = g_api->VmbCaptureQueueFlush_t(m_handle);
  if (err != VmbErrorSuccess) {
    return err;
  }

  err = g_api->VmbFrameRevokeAll_t(m_handle);
  if (err != VmbErrorSuccess) {
    return err;
  }

  return err;
}

long AlliedVisionCamera::GetImageBufferSize() const { return m_bufferSize; }
unsigned AlliedVisionCamera::GetBitDepth() const {
  // TODO implement
  return 8;
}
int AlliedVisionCamera::GetBinning() const {
  // TODO implement
  return 1;
}
int AlliedVisionCamera::SetBinning(int binSize) {
  // TODO implement
  return VmbErrorSuccess;
}
void AlliedVisionCamera::SetExposure(double exp_ms) {
  // TODO implement
}
double AlliedVisionCamera::GetExposure() const {
  // TODO implement
  return 8058.96;
}
int AlliedVisionCamera::SetROI(unsigned x, unsigned y, unsigned xSize,
                               unsigned ySize) {
  // TODO implement
  return VmbErrorSuccess;
}
int AlliedVisionCamera::GetROI(unsigned& x, unsigned& y, unsigned& xSize,
                               unsigned& ySize) {
  // TODO implement
  return VmbErrorSuccess;
}
int AlliedVisionCamera::ClearROI() {
  // TODO implement
  return 0;
}
int AlliedVisionCamera::IsExposureSequenceable(bool& isSequenceable) const {
  // TODO implement
  return VmbErrorSuccess;
}
void AlliedVisionCamera::GetName(char* name) const {
  CDeviceUtils::CopyLimitedString(name, m_cameraName.c_str());
}
bool AlliedVisionCamera::IsCapturing() { return m_isAcquisitionRunning; }

void AlliedVisionCamera::setApiErrorMessages() {
  SetErrorText(VmbErrorApiNotStarted, "Vimba X API not started");
  SetErrorText(VmbErrorNotFound, "Device cannot be found");
  SetErrorText(VmbErrorDeviceNotOpen, "Device cannot be opened");
  SetErrorText(VmbErrorBadParameter,
               "Invalid parameter passed to the function");
  SetErrorText(VmbErrorNotImplemented, "Feature not implemented");
  SetErrorText(VmbErrorNotSupported, "Feature not supported");
  SetErrorText(VmbErrorUnknown, "Unknown error");
}

VmbError_t AlliedVisionCamera::getCamerasList() {
  VmbUint32_t camNum;

  // Get the number of connected cameras first
  VmbError_t err = g_api->VmbCamerasList_t(nullptr, 0, &camNum, 0);
  if (VmbErrorSuccess == err) {
    VmbCameraInfo_t* camInfo = new VmbCameraInfo_t[camNum];

    // Get the cameras
    err = g_api->VmbCamerasList_t(camInfo, camNum, &camNum, sizeof *camInfo);

    if (err == VmbErrorSuccess) {
      for (VmbUint32_t i = 0; i < camNum; ++i) {
        RegisterDevice(camInfo[i].cameraIdString, MM::CameraDevice,
                       camInfo[i].cameraName);
      }
    }

    delete[] camInfo;
  }

  return err;
}

VmbError_t AlliedVisionCamera::resizeImageBuffer() {
  auto err = g_api->VmbFeatureIntGet_t(m_handle, "Width", &m_imageWidth);
  if (err != VmbErrorSuccess) {
    return err;
  }

  err = g_api->VmbFeatureIntGet_t(m_handle, "Height", &m_imageHeight);
  if (err != VmbErrorSuccess) {
    return err;
  }

  err = g_api->VmbPayloadSizeGet_t(m_handle, &m_bufferSize);
  if (err != VmbErrorSuccess) {
    return err;
  }

  for (size_t i = 0; i < MAX_FRAMES; i++) {
    delete[] m_buffer[i];
    m_buffer[i] = new VmbUint8_t[m_bufferSize];
  }

  return VmbErrorSuccess;
}

int AlliedVisionCamera::OnPixelTypeChanged(MM::PropertyBase* pProp,
                                           MM::ActionType eAct) {
  // TODO implement
  resizeImageBuffer();
  return 0;
}

int AlliedVisionCamera::OnBinningChanged(MM::PropertyBase* pProp,
                                         MM::ActionType eAct) {
  // TODO implement
  resizeImageBuffer();
  return 0;
}

VmbError_t AlliedVisionCamera::createPropertyFromFeature(
    const VmbFeatureInfo_t* feature) {
  // TODO
  // Implemnet onProperyChanged for some properties and buffer resize
  // Implement readOnly/WriteOnly reading
  if (feature == nullptr) {
    return VmbErrorInvalidValue;
  }

  VmbError_t err = VmbErrorSuccess;
  switch (feature->featureDataType) {
    case VmbFeatureDataBool: {
      VmbBool_t value;
      err = g_api->VmbFeatureBoolGet_t(m_handle, feature->name, &value);
      if (VmbErrorSuccess == err) {
        CreateIntegerProperty(feature->name, value, true, nullptr);
      }
      break;
    }
    case VmbFeatureDataEnum: {
      char const* value = nullptr;
      err = g_api->VmbFeatureEnumGet_t(m_handle, feature->name, &value);
      if (VmbErrorSuccess == err) {
        CreateStringProperty(feature->name, value, true, nullptr);
      }
      break;
    }
    case VmbFeatureDataFloat: {
      double value;
      err = g_api->VmbFeatureFloatGet_t(m_handle, feature->name, &value);
      if (err == VmbErrorSuccess) {
        CreateFloatProperty(feature->name, value, true, nullptr);
      }
      break;
    }
    case VmbFeatureDataInt: {
      VmbInt64_t value;
      err = g_api->VmbFeatureIntGet_t(m_handle, feature->name, &value);
      if (err == VmbErrorSuccess) {
        CreateIntegerProperty(feature->name, value, true, nullptr);
      }
      break;
    }
    case VmbFeatureDataString: {
      VmbUint32_t size = 0;
      err = g_api->VmbFeatureStringGet_t(m_handle, feature->name, nullptr, 0,
                                         &size);
      if (VmbErrorSuccess == err && size > 0) {
        std::shared_ptr<char> buff = std::shared_ptr<char>(new char[size]);
        err = g_api->VmbFeatureStringGet_t(m_handle, feature->name, buff.get(),
                                           size, &size);
        if (VmbErrorSuccess == err) {
          CreateStringProperty(feature->name, buff.get(), true, nullptr);
        }
      }
      break;
    }
    case VmbFeatureDataCommand:
    case VmbFeatureDataUnknown:
    case VmbFeatureDataRaw:
    case VmbFeatureDataNone:
    default:
      err = VmbErrorFeaturesUnavailable;
      break;
  }

  return err;
}

VmbError_t AlliedVisionCamera::setupProperties() {
  VmbUint32_t featureCount = 0;
  VmbError_t err = g_api->VmbFeaturesList_t(m_handle, NULL, 0, &featureCount,
                                            sizeof(VmbFeatureInfo_t));
  if (err != VmbErrorSuccess || !featureCount) {
    return err;
  }

  std::shared_ptr<VmbFeatureInfo_t> features =
      std::shared_ptr<VmbFeatureInfo_t>(new VmbFeatureInfo_t[featureCount]);

  err = g_api->VmbFeaturesList_t(m_handle, features.get(), featureCount,
                                 &featureCount, sizeof(VmbFeatureInfo_t));

  if (err != VmbErrorSuccess) {
    return err;
  }

  const VmbFeatureInfo_t* end = features.get() + featureCount;
  for (VmbFeatureInfo_t* feature = features.get(); feature != end; ++feature) {
    std::stringstream ss;
    ss << "/// Feature Name: " << feature->name << "\n";
    ss << "/// Display Name: " << feature->displayName << "\n";
    ss << "/// Tooltip: " << feature->tooltip << "\n";
    ss << "/// Description: " << feature->description << "\n";
    ss << "/// SNFC Namespace: " << feature->sfncNamespace << "\n";
    LogMessage(ss.str().c_str());
    createPropertyFromFeature(feature);
  }
}

int AlliedVisionCamera::StartSequenceAcquisition(long numImages,
                                                 double interval_ms,
                                                 bool stopOnOverflow) {
  if (m_isAcquisitionRunning) {
    return DEVICE_CAMERA_BUSY_ACQUIRING;
  }

  int err = GetCoreCallback()->PrepareForAcq(this);
  if (err != DEVICE_OK) {
    return err;
  }

  err = resizeImageBuffer();
  if (err != VmbErrorSuccess) {
    return err;
  }

  for (size_t i = 0; i < MAX_FRAMES; i++) {
    // Setup frame with buffer
    m_frames[i].buffer = new uint8_t[m_bufferSize];
    m_frames[i].bufferSize = m_bufferSize;
    m_frames[i].context[0] = this;  //<! Pointer to camera
    m_frames[i].context[1] =
        reinterpret_cast<void*>(i);  //<! Pointer to frame index

    err =
        g_api->VmbFrameAnnounce_t(m_handle, &(m_frames[i]), sizeof(VmbFrame_t));
    if (err != VmbErrorSuccess) {
      return err;
    }

    err = g_api->VmbCaptureFrameQueue_t(
        m_handle, &(m_frames[i]),
        [](const VmbHandle_t cameraHandle, const VmbHandle_t streamHandle,
           VmbFrame_t* frame) {
          reinterpret_cast<AlliedVisionCamera*>(frame->context[0])
              ->insertFrame(frame);
        });
    if (err != VmbErrorSuccess) {
      return err;
    }
  }

  err = g_api->VmbCaptureStart_t(m_handle);
  if (err != VmbErrorSuccess) {
    return err;
  }

  err = g_api->VmbFeatureCommandRun_t(m_handle, "AcquisitionStart");
  if (err != VmbErrorSuccess) {
    return err;
  }

  m_isAcquisitionRunning = true;
  return err;
}

int AlliedVisionCamera::StartSequenceAcquisition(double interval_ms) {
  return StartSequenceAcquisition(LONG_MAX, interval_ms, true);
}
int AlliedVisionCamera::StopSequenceAcquisition() {
  if (m_isAcquisitionRunning) {
    auto err = g_api->VmbFeatureCommandRun_t(m_handle, "AcquisitionStop");
    if (err != VmbErrorSuccess) {
      return err;
    }

    m_isAcquisitionRunning = false;
  }

  auto err = g_api->VmbCaptureEnd_t(m_handle);
  if (err != VmbErrorSuccess) {
    return err;
  }

  err = g_api->VmbCaptureQueueFlush_t(m_handle);
  if (err != VmbErrorSuccess) {
    return err;
  }

  err = g_api->VmbFrameRevokeAll_t(m_handle);
  if (err != VmbErrorSuccess) {
    return err;
  }

  return err;
}

void AlliedVisionCamera::insertFrame(VmbFrame_t* frame) {
  if (frame != nullptr && frame->receiveStatus == VmbFrameStatusComplete) {
    VmbUint8_t* buffer = reinterpret_cast<VmbUint8_t*>(frame->buffer);

    // TODO implement metadata
    Metadata md;
    md.put("Camera", m_cameraName);

    // TODO implement parameters
    auto err = GetCoreCallback()->InsertImage(this, buffer, m_imageWidth,
                                              m_imageHeight, 1, 1,
                                              md.Serialize().c_str());

    if (err == DEVICE_BUFFER_OVERFLOW) {
      GetCoreCallback()->ClearImageBuffer(this);
      // TODO implement parameters
      err = GetCoreCallback()->InsertImage(this, buffer, frame->width,
                                           frame->height, 1, 1,
                                           md.Serialize().c_str(), false);
    }

    if (m_isAcquisitionRunning) {
      g_api->VmbCaptureFrameQueue_t(
          m_handle, frame,
          [](const VmbHandle_t cameraHandle, const VmbHandle_t streamHandle,
             VmbFrame_t* frame) {
            reinterpret_cast<AlliedVisionCamera*>(frame->context[0])
                ->insertFrame(frame);
          });
    }
  }
}