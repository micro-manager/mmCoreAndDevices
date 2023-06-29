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
#define NOMINMAX  //<! Remove WINDOWS MACROS for MIN and MAX

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

unsigned AlliedVisionCamera::GetImageWidth() const {
  char value[MM::MaxStrLength];
  int ret = GetProperty(g_Width, value);
  if (ret != DEVICE_OK) {
    return 0;
  }

  return atoi(value);
}

unsigned AlliedVisionCamera::GetImageHeight() const {
  char value[MM::MaxStrLength];
  int ret = GetProperty(g_Height, value);
  if (ret != DEVICE_OK) {
    return 0;
  }

  return atoi(value);
}

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
  char value[MM::MaxStrLength];
  int ret = GetProperty(MM::g_Keyword_Binning, value);
  if (ret != DEVICE_OK) {
    return 0;
  }

  return atoi(value);
}

int AlliedVisionCamera::SetBinning(int binSize) {
  return SetProperty(MM::g_Keyword_Binning,
                     CDeviceUtils::ConvertToString(binSize));
}

int AlliedVisionCamera::OnBinning(MM::PropertyBase* pProp,
                                  MM::ActionType eAct) {
  // Get horizonal binning
  VmbFeatureInfo_t featureInfoHorizontal;
  VmbError_t err = g_api->VmbFeatureInfoQuery_t(
      m_handle, g_BinningHorizontalFeature, &featureInfoHorizontal,
      sizeof(featureInfoHorizontal));
  if (VmbErrorSuccess != err) {
    return err;
  }

  // Get vertical binning
  VmbFeatureInfo_t featureInfoVertical;
  err = g_api->VmbFeatureInfoQuery_t(m_handle, g_BinningVerticalFeature,
                                     &featureInfoVertical,
                                     sizeof(featureInfoVertical));
  if (VmbErrorSuccess != err) {
    return err;
  }

  VmbInt64_t minHorizontal, maxHorizontal, minVertical, maxVertical, min, max;
  std::string value, valueHorizontal, valueVertical;
  bool rMode, wMode;
  std::vector<std::string> strValues;

  // Cast to Property to have an access to setting read/write mode
  MM::Property* pChildProperty = (MM::Property*)pProp;

  // Get read/write mode - assume binning horizontal and vertical has the same
  // mode
  err = g_api->VmbFeatureAccessQuery_t(m_handle, g_BinningVerticalFeature,
                                       &rMode, &wMode);
  if (VmbErrorSuccess != err) {
    return err;
  }

  switch (eAct) {
    case MM::ActionType::BeforeGet:
      // Get horizontal binning value
      err = getFeatureValue(&featureInfoHorizontal, g_BinningHorizontalFeature,
                            valueHorizontal);
      if (VmbErrorSuccess != err) {
        return err;
      }
      // Get vertical binning value
      err = getFeatureValue(&featureInfoVertical, g_BinningVerticalFeature,
                            valueVertical);
      if (VmbErrorSuccess != err) {
        return err;
      }

      // Get min value from these two
      min =
          std::min(atoi(valueHorizontal.c_str()), atoi(valueVertical.c_str()));
      pProp->Set(std::to_string(min).c_str());

      // Update binning limits
      err = g_api->VmbFeatureIntRangeQuery_t(
          m_handle, g_BinningHorizontalFeature, &minHorizontal, &maxHorizontal);
      if (VmbErrorSuccess != err) {
        return err;
      }

      err = g_api->VmbFeatureIntRangeQuery_t(m_handle, g_BinningVerticalFeature,
                                             &minVertical, &maxVertical);
      if (VmbErrorSuccess != err) {
        return err;
      }
      min = std::max(minHorizontal, minVertical);
      max = std::min(maxHorizontal, maxVertical);

      for (VmbInt64_t i = min; i <= max; i++) {
        strValues.push_back(std::to_string(i));
      }
      SetAllowedValues(pProp->GetName().c_str(), strValues);
      // Update access mode
      pChildProperty->SetReadOnly(rMode && !wMode);
      break;
    case MM::ActionType::AfterSet:
      pProp->Get(value);
      err = setFeatureValue(&featureInfoHorizontal, g_BinningHorizontalFeature,
                            value);
      if (VmbErrorSuccess != err) {
        return err;
      }
      err = setFeatureValue(&featureInfoVertical, g_BinningVerticalFeature,
                            value);
      break;
    default:
      // nothing
      break;
  }

  // TODO return uManager error
  return err;
}

double AlliedVisionCamera::GetExposure() const {
  char strExposure[MM::MaxStrLength];
  int ret = GetProperty(MM::g_Keyword_Exposure, strExposure);
  if (ret != DEVICE_OK) {
    return 0.0;
  }

  return strtod(strExposure, nullptr);
}

void AlliedVisionCamera::SetExposure(double exp_ms) {
  SetProperty(MM::g_Keyword_Exposure, CDeviceUtils::ConvertToString(exp_ms));
  GetCoreCallback()->OnExposureChanged(this, exp_ms);
}

int AlliedVisionCamera::OnExposure(MM::PropertyBase* pProp,
                                   MM::ActionType eAct) {
  const auto propertyName = pProp->GetName();
  VmbFeatureInfo_t featureInfo;
  VmbError_t err = g_api->VmbFeatureInfoQuery_t(
      m_handle, g_ExposureFeature, &featureInfo, sizeof(featureInfo));
  if (VmbErrorSuccess != err) {
    return err;
  }

  std::string value;
  bool rMode, wMode;
  err = g_api->VmbFeatureAccessQuery_t(m_handle, propertyName.c_str(), &rMode,
                                       &wMode);
  MM::Property* pChildProperty = (MM::Property*)pProp;

  switch (eAct) {
    case MM::ActionType::BeforeGet:
      // Update limits
      setAllowedValues(&featureInfo);
      // Update access mode
      pChildProperty->SetReadOnly(rMode && !wMode);
      // Update value
      err = getFeatureValue(&featureInfo, g_ExposureFeature, value);
      pProp->Set(value.c_str());
      break;
    case MM::ActionType::AfterSet:
      pProp->Get(value);
      err = setFeatureValue(&featureInfo, g_ExposureFeature, value);
      break;
    default:
      // nothing
      break;
  }

  // TODO return uManager error
  return err;
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
  VmbError_t err = g_api->VmbPayloadSizeGet_t(m_handle, &m_bufferSize);
  if (err != VmbErrorSuccess) {
    return err;
  }

  for (size_t i = 0; i < MAX_FRAMES; i++) {
    delete[] m_buffer[i];
    m_buffer[i] = new VmbUint8_t[m_bufferSize];
  }

  return err;
}

int AlliedVisionCamera::OnPixelType(MM::PropertyBase* pProp,
                                    MM::ActionType eAct) {
  // TODO implement
  return 0;
}

int AlliedVisionCamera::onProperty(MM::PropertyBase* pProp,
                                   MM::ActionType eAct) {
  const auto propertyName = pProp->GetName();
  VmbFeatureInfo_t featureInfo;
  VmbError_t err = g_api->VmbFeatureInfoQuery_t(
      m_handle, propertyName.c_str(), &featureInfo, sizeof(featureInfo));
  if (VmbErrorSuccess != err) {
    return err;
  }

  std::string value{};
  bool rMode, wMode;
  err = g_api->VmbFeatureAccessQuery_t(m_handle, propertyName.c_str(), &rMode,
                                       &wMode);
  MM::Property* pChildProperty = (MM::Property*)pProp;
  switch (eAct) {
    case MM::ActionType::BeforeGet:
      // Update limits
      setAllowedValues(&featureInfo);
      // Update access mode
      pChildProperty->SetReadOnly(rMode && !wMode);
      // Update value
      //TODO error handling
      getFeatureValue(&featureInfo, propertyName.c_str(), value);
      pProp->Set(value.c_str());
      break;
    case MM::ActionType::AfterSet:
      // Update value
      pProp->Get(value);
      // TODO error handling
      setFeatureValue(&featureInfo, propertyName.c_str(), value);
      break;
    default:
      // nothing
      break;
  }

  return err;
}

VmbError_t AlliedVisionCamera::getFeatureValue(VmbFeatureInfo_t* featureInfo,
                                               const char* featureName,
                                               std::string& value) {
  VmbError_t err = VmbErrorSuccess;
  switch (featureInfo->featureDataType) {
    case VmbFeatureDataBool: {
      VmbBool_t out;
      err = g_api->VmbFeatureBoolGet_t(m_handle, featureName, &out);
      if (err != VmbErrorSuccess) {
        break;
      }
      value = std::to_string(out);
      break;
    }
    case VmbFeatureDataEnum: {
      const char* out = nullptr;
      err = g_api->VmbFeatureEnumGet_t(m_handle, featureName, &out);
      if (err != VmbErrorSuccess) {
        break;
      }
      value = std::string(out);
      break;
    }
    case VmbFeatureDataFloat: {
      double out;
      err = g_api->VmbFeatureFloatGet_t(m_handle, featureName, &out);
      if (err != VmbErrorSuccess) {
        break;
      }
      value = std::to_string(out);
      break;
    }
    case VmbFeatureDataInt: {
      VmbInt64_t out;
      err = g_api->VmbFeatureIntGet_t(m_handle, featureName, &out);
      if (err != VmbErrorSuccess) {
        break;
      }
      value = std::to_string(out);
      break;
    }
    case VmbFeatureDataString: {
      VmbUint32_t size = 0;
      err = g_api->VmbFeatureStringGet_t(m_handle, featureName, nullptr, 0,
                                         &size);
      if (VmbErrorSuccess == err && size > 0) {
        std::shared_ptr<char> buff = std::shared_ptr<char>(new char[size]);
        err = g_api->VmbFeatureStringGet_t(m_handle, featureName, buff.get(),
                                           size, &size);
        if (err != VmbErrorSuccess) {
          break;
        }
        value = std::string(buff.get());
      }
      break;
    }
    case VmbFeatureDataCommand:
    case VmbFeatureDataUnknown:
    case VmbFeatureDataRaw:
    case VmbFeatureDataNone:
    default:
      // nothing
      break;
  }
  return err;
}

VmbError_t AlliedVisionCamera::setFeatureValue(VmbFeatureInfo_t* featureInfo,
                                               const char* featureName,
                                               std::string& value) {
  VmbError_t err = VmbErrorSuccess;
  std::stringstream ss(value);

  switch (featureInfo->featureDataType) {
    case VmbFeatureDataBool: {
      VmbBool_t out;
      ss >> out;
      err = g_api->VmbFeatureBoolSet_t(m_handle, featureName, out);
      break;
    }
    case VmbFeatureDataEnum: {
      err = g_api->VmbFeatureEnumSet_t(m_handle, featureName, value.c_str());
      break;
    }
    case VmbFeatureDataFloat: {
      double out;
      ss >> out;
      err = g_api->VmbFeatureFloatSet_t(m_handle, featureName, out);
      break;
    }
    case VmbFeatureDataInt: {
      VmbInt64_t out;
      ss >> out;
      err = g_api->VmbFeatureIntSet_t(m_handle, featureName, out);
      break;
    }
    case VmbFeatureDataString: {
      err = g_api->VmbFeatureStringSet_t(m_handle, featureName, value.c_str());
      break;
    }
    case VmbFeatureDataCommand:
    case VmbFeatureDataUnknown:
    case VmbFeatureDataRaw:
    case VmbFeatureDataNone:
    default:
      // nothing
      break;
  }
  return err;
}

VmbError_t AlliedVisionCamera::createPropertyFromFeature(
    const VmbFeatureInfo_t* feature, MM::ActionFunctor* callback,
    const char* propertyName, bool skipVmbCallback) {
  if (feature == nullptr) {
    return VmbErrorInvalidValue;
  }

  auto featureName = feature->name;
  auto propName = (propertyName != nullptr) ? propertyName : featureName;
  VmbError_t err = VmbErrorSuccess;

  if (!skipVmbCallback) {
    // Vmb callback for given feature
    auto vmbCallback = [](VmbHandle_t handle, const char* name,
                          void* userContext) {
      AlliedVisionCamera* camera =
          reinterpret_cast<AlliedVisionCamera*>(userContext);
      camera->UpdateProperty(name);
    };
    // Register VMb callback
    err = g_api->VmbFeatureInvalidationRegister_t(m_handle, featureName,
                                                  vmbCallback, this);
    if (err != VmbErrorSuccess) {
      return err;
    }
  }

  if (HasProperty(propName)) {
    // Already exist
    return err;
  }

  switch (feature->featureDataType) {
    case VmbFeatureDataBool: {
      CreateIntegerProperty(propName, 0, true, callback);
      break;
    }
    case VmbFeatureDataEnum: {
      CreateStringProperty(propName, "", true, callback);
      break;
    }
    case VmbFeatureDataFloat: {
      CreateFloatProperty(propName, 0.0, true, callback);
      break;
    }
    case VmbFeatureDataInt: {
      CreateIntegerProperty(propName, 0, true, callback);
      break;
    }
    case VmbFeatureDataString: {
      CreateStringProperty(propName, "", true, callback);
      break;
    }
    case VmbFeatureDataCommand:
    case VmbFeatureDataUnknown:
    case VmbFeatureDataRaw:
    case VmbFeatureDataNone:
    default:
      // nothing
      break;
  }

  return err;
}

VmbError_t AlliedVisionCamera::createCoreProperties() {
  VmbError_t err = VmbErrorSuccess;
  //=== Create PIXEL_TYPE from PIXEL_FORMAT
  {
    VmbFeatureInfo_t feature;
    err = g_api->VmbFeatureInfoQuery_t(m_handle, g_PixelFormatFeature, &feature,
                                       sizeof(VmbFeatureInfo_t));
    if (err != VmbErrorSuccess) {
      return err;
    }
    // uManager callback
    CPropertyAction* callback =
        new CPropertyAction(this, &AlliedVisionCamera::OnPixelType);
    err = createPropertyFromFeature(&feature, callback, MM::g_Keyword_PixelType,
                                    true);
    if (err != VmbErrorSuccess) {
      return err;
    }

    err = setAllowedValues(&feature, MM::g_Keyword_PixelType);
    if (err != VmbErrorSuccess) {
      return err;
    }

    // Vmb callback
    auto vmbCallback = [](VmbHandle_t handle, const char* name,
                          void* userContext) {
      AlliedVisionCamera* camera =
          reinterpret_cast<AlliedVisionCamera*>(userContext);
      camera->UpdateProperty(MM::g_Keyword_PixelType);
    };
    err = g_api->VmbFeatureInvalidationRegister_t(
        m_handle, g_PixelFormatFeature, vmbCallback, this);
    if (err != VmbErrorSuccess) {
      return err;
    }
  }

  //=== Create EXPOSURE from EXPOSURE_TIME
  {
    VmbFeatureInfo_t feature;
    err = g_api->VmbFeatureInfoQuery_t(m_handle, g_ExposureFeature, &feature,
                                       sizeof(VmbFeatureInfo_t));
    if (err != VmbErrorSuccess) {
      return err;
    }

    // uManager callback
    CPropertyAction* callback =
        new CPropertyAction(this, &AlliedVisionCamera::OnExposure);
    err = createPropertyFromFeature(&feature, callback, MM::g_Keyword_Exposure,
                                    true);
    if (err != VmbErrorSuccess) {
      return err;
    }

    err = setAllowedValues(&feature, MM::g_Keyword_Exposure);
    if (err != VmbErrorSuccess) {
      return err;
    }

    // Vmb callback
    auto vmbCallback = [](VmbHandle_t handle, const char* name,
                          void* userContext) {
      AlliedVisionCamera* camera =
          reinterpret_cast<AlliedVisionCamera*>(userContext);
      camera->UpdateProperty(MM::g_Keyword_Exposure);
    };

    err = g_api->VmbFeatureInvalidationRegister_t(m_handle, g_ExposureFeature,
                                                  vmbCallback, this);
    if (err != VmbErrorSuccess) {
      return err;
    }
  }

  //=== Create BINNING from BINNING_HORIZONTAL and BINNING_VERTICAL
  {
    VmbFeatureInfo_t feature;
    err = g_api->VmbFeatureInfoQuery_t(m_handle, g_BinningHorizontalFeature,
                                       &feature, sizeof(VmbFeatureInfo_t));
    if (err != VmbErrorSuccess) {
      return err;
    }

    // uManager callback
    CPropertyAction* callback =
        new CPropertyAction(this, &AlliedVisionCamera::OnBinning);
    err = createPropertyFromFeature(&feature, callback, MM::g_Keyword_Binning,
                                    true);
    if (err != VmbErrorSuccess) {
      return err;
    }

    // Set limits for BINNING
    VmbInt64_t minHorizontal, maxHorizontal, minVertical, maxVertical;
    err = g_api->VmbFeatureIntRangeQuery_t(m_handle, g_BinningHorizontalFeature,
                                           &minHorizontal, &maxHorizontal);
    if (VmbErrorSuccess != err) {
      return err;
    }

    err = g_api->VmbFeatureIntRangeQuery_t(m_handle, g_BinningVerticalFeature,
                                           &minVertical, &maxVertical);
    if (VmbErrorSuccess != err) {
      return err;
    }
    auto min = std::max(minHorizontal, minVertical);
    auto max = std::min(maxHorizontal, maxVertical);

    std::vector<std::string> strValues;
    for (VmbInt64_t i = min; i <= max; i++) {
      strValues.push_back(std::to_string(i));
    }

    SetAllowedValues(MM::g_Keyword_Binning, strValues);

    // Vmb callback for horizontal binning
    auto vmbCallbackHorizontal = [](VmbHandle_t handle, const char* name,
                                    void* userContext) {
      AlliedVisionCamera* camera =
          reinterpret_cast<AlliedVisionCamera*>(userContext);
      camera->UpdateProperty(MM::g_Keyword_Binning);
    };

    err = g_api->VmbFeatureInvalidationRegister_t(
        m_handle, g_BinningHorizontalFeature, vmbCallbackHorizontal, this);
    if (err != VmbErrorSuccess) {
      return err;
    }

    // Vmb callback for vertical binning
    auto vmbCallbackVertical = [](VmbHandle_t handle, const char* name,
                                  void* userContext) {
      AlliedVisionCamera* camera =
          reinterpret_cast<AlliedVisionCamera*>(userContext);
      camera->UpdateProperty(MM::g_Keyword_Binning);
    };

    err = g_api->VmbFeatureInvalidationRegister_t(
        m_handle, g_BinningVerticalFeature, vmbCallbackVertical, this);
    if (err != VmbErrorSuccess) {
      return err;
    }
  }

  return err;
}

VmbError_t AlliedVisionCamera::setAllowedValues(const VmbFeatureInfo_t* feature,
                                                const char* propertyName) {
  if (feature == nullptr) {
    return VmbErrorInvalidValue;
  }

  auto propName = (propertyName != nullptr) ? propertyName : feature->name;
  VmbError_t err = VmbErrorSuccess;

  switch (feature->featureDataType) {
    case VmbFeatureDataBool: {
      // TODO check if possible values needs to be set here
      break;
    }
    case VmbFeatureDataFloat: {
      double min = 0;
      double max = 0;
      err = g_api->VmbFeatureFloatRangeQuery_t(m_handle, feature->name, &min,
                                               &max);
      if (VmbErrorSuccess != err) {
        return err;
      }

      err = SetPropertyLimits(propName, min, max);
      break;
    }
    case VmbFeatureDataEnum: {
      std::array<const char*, MM::MaxStrLength> values;
      std::vector<std::string> strValues;
      VmbUint32_t valuesNum = 0;
      err = g_api->VmbFeatureEnumRangeQuery_t(
          m_handle, feature->name, values.data(), MM::MaxStrLength, &valuesNum);
      if (VmbErrorSuccess != err) {
        return err;
      }

      for (size_t i = 0; i < valuesNum; i++) {
        strValues.push_back(values[i]);
      }
      SetAllowedValues(propName, strValues);

      break;
    }
    case VmbFeatureDataInt: {
      VmbInt64_t min = 0;
      VmbInt64_t max = 0;
      err =
          g_api->VmbFeatureIntRangeQuery_t(m_handle, feature->name, &min, &max);
      if (VmbErrorSuccess != err) {
        return err;
      }

      err = SetPropertyLimits(propName, min, max);
      break;
    }
    case VmbFeatureDataString: {
      break;
    }
    case VmbFeatureDataRaw:
    case VmbFeatureDataCommand:
    case VmbFeatureDataNone:
    default:
      // nothing
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

  // TODO handle error
  err = createCoreProperties();

  const VmbFeatureInfo_t* end = features.get() + featureCount;
  for (VmbFeatureInfo_t* feature = features.get(); feature != end; ++feature) {
    auto featureName = std::string(feature->name);
    // Skip these features as they are mapped to the Core Properties
    if (featureName == std::string(g_PixelFormatFeature) ||
        featureName == std::string(g_BinningHorizontalFeature) ||
        featureName == std::string(g_BinningVerticalFeature) ||
        featureName == std::string(g_ExposureFeature)) {
      continue;
    }

    // uManager callback
    CPropertyAction* callback =
        new CPropertyAction(this, &AlliedVisionCamera::onProperty);
    // TODO handle error
    err = createPropertyFromFeature(feature, callback);
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
    auto err = GetCoreCallback()->InsertImage(this, buffer, GetImageWidth(),
                                              GetImageHeight(), 1, 1,
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