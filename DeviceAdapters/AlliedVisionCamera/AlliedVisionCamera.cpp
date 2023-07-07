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
// STATIC VALUES
///////////////////////////////////////////////////////////////////////////////
const std::unordered_map<std::string, std::string>
    AlliedVisionCamera::m_featureToProperty = {
        {g_PixelFormatFeature, MM::g_Keyword_PixelType},
        {g_ExposureFeature, MM::g_Keyword_Exposure},
        {g_BinningHorizontalFeature, MM::g_Keyword_Binning},
        {g_BinningVerticalFeature, MM::g_Keyword_Binning}};

const std::unordered_multimap<std::string, std::string>
    AlliedVisionCamera::m_propertyToFeature = {
        {MM::g_Keyword_PixelType, g_PixelFormatFeature},
        {MM::g_Keyword_Exposure, g_ExposureFeature},
        {MM::g_Keyword_Binning, g_BinningHorizontalFeature},
        {MM::g_Keyword_Binning, g_BinningVerticalFeature}};

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
  // TODO handle error
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

void AlliedVisionCamera::setApiErrorMessages() {
  SetErrorText(VmbErrorApiNotStarted, "Vimba X API not started");
  SetErrorText(VmbErrorNotFound, "Device cannot be found");
  SetErrorText(VmbErrorDeviceNotOpen, "Device cannot be opened");
  SetErrorText(VmbErrorBadParameter,
               "Invalid parameter passed to the function");
  SetErrorText(VmbErrorNotImplemented, "Feature not implemented");
  SetErrorText(VmbErrorNotSupported, "Feature not supported");
  SetErrorText(VmbErrorUnknown, "Unknown error");
  SetErrorText(VmbErrorInvalidValue,
               "The value is not valid: either out of bounds or not an "
               "increment of the minimum");
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
    // uManager callback
    CPropertyAction* callback =
        new CPropertyAction(this, &AlliedVisionCamera::onProperty);

    err = createPropertyFromFeature(feature, callback);
    if (err != VmbErrorSuccess) {
      LogMessageCode(err);
      continue;
    }
  }
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

VmbError_t AlliedVisionCamera::createPropertyFromFeature(
    const VmbFeatureInfo_t* feature, MM::ActionFunctor* callback) {
  if (feature == nullptr) {
    return VmbErrorInvalidValue;
  }

  auto featureName = feature->name;
  VmbError_t err = VmbErrorSuccess;
  std::string propName = {};
  mapFeatureNameToPropertyName(featureName, propName);

  // Vimba callback
  auto vmbCallback = [](VmbHandle_t handle, const char* name,
                        void* userContext) {
    AlliedVisionCamera* camera =
        reinterpret_cast<AlliedVisionCamera*>(userContext);
    std::string propertyName;
    camera->mapFeatureNameToPropertyName(name, propertyName);
    camera->UpdateProperty(propertyName.c_str());
  };

  // Add property to the list
  m_propertyItems.insert({propName, {propName}});

  // Register VMb callback
  err = g_api->VmbFeatureInvalidationRegister_t(m_handle, featureName,
                                                vmbCallback, this);
  if (err != VmbErrorSuccess) {
    return err;
  }

  switch (feature->featureDataType) {
    case VmbFeatureDataInt: {
      err = CreateIntegerProperty(propName.c_str(), 0, true, callback);
      break;
    }
    case VmbFeatureDataBool:
    case VmbFeatureDataEnum:
    case VmbFeatureDataCommand:
    case VmbFeatureDataString: {
      err = CreateStringProperty(propName.c_str(), "", true, callback);
      break;
    }
    case VmbFeatureDataFloat: {
      err = CreateFloatProperty(propName.c_str(), 0.0, true, callback);
      break;
    }
    case VmbFeatureDataUnknown:
    case VmbFeatureDataRaw:
    case VmbFeatureDataNone:
    default:
      // nothing
      break;
  }

  return err;
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

  // Get read/write mode - assume binning horizontal and vertical has the same
  // mode
  bool rMode, wMode, readOnly, featureAvailable;
  err = g_api->VmbFeatureAccessQuery_t(m_handle, g_BinningVerticalFeature,
                                       &rMode, &wMode);
  if (VmbErrorSuccess != err) {
    return err;
  }

  featureAvailable = (rMode || wMode);
  if (!featureAvailable) {
    return err;
  }

  readOnly = featureAvailable && (rMode && !wMode);

  // Get values of property and features
  std::string propertyValue, featureHorizontalValue, featureVerticalValue;
  pProp->Get(propertyValue);
  err = getFeatureValue(&featureInfoHorizontal, g_BinningHorizontalFeature,
                        featureHorizontalValue);
  if (VmbErrorSuccess != err) {
    return err;
  }
  err = getFeatureValue(&featureInfoVertical, g_BinningVerticalFeature,
                        featureVerticalValue);
  if (VmbErrorSuccess != err) {
    return err;
  }

  std::string featureValue =
      std::to_string(std::min(atoi(featureHorizontalValue.c_str()),
                              atoi(featureVerticalValue.c_str())));

  MM::Property* pChildProperty = (MM::Property*)pProp;
  std::vector<std::string> strValues;
  VmbInt64_t minHorizontal, maxHorizontal, minVertical, maxVertical, min, max;

  switch (eAct) {
    case MM::ActionType::BeforeGet:
      // Update property
      if (propertyValue != featureValue) {
        pProp->Set(featureValue.c_str());
      }
      // Update property's access mode
      pChildProperty->SetReadOnly(readOnly);
      // Update property's limits and allowed values
      if (!readOnly) {
        // Update binning limits
        err = g_api->VmbFeatureIntRangeQuery_t(m_handle,
                                               g_BinningHorizontalFeature,
                                               &minHorizontal, &maxHorizontal);
        if (VmbErrorSuccess != err) {
          return err;
        }

        err = g_api->VmbFeatureIntRangeQuery_t(
            m_handle, g_BinningVerticalFeature, &minVertical, &maxVertical);
        if (VmbErrorSuccess != err) {
          return err;
        }

        //[IMPORTANT] For binning, increment step is ignored

        min = std::max(minHorizontal, minVertical);
        max = std::min(maxHorizontal, maxVertical);

        for (VmbInt64_t i = min; i <= max; i++) {
          strValues.push_back(std::to_string(i));
        }
        err = SetAllowedValues(pProp->GetName().c_str(), strValues);
      }
      break;
    case MM::ActionType::AfterSet:
      if (propertyValue != featureValue) {
        VmbError_t errHor = setFeatureValue(
            &featureInfoHorizontal, g_BinningHorizontalFeature, propertyValue);
        VmbError_t errVer = setFeatureValue(
            &featureInfoVertical, g_BinningVerticalFeature, propertyValue);
        if (VmbErrorSuccess != errHor || VmbErrorSuccess != errVer) {
          //[IMPORTANT] For binning, adjust value is ignored
        }
      }
      break;
    default:
      // nothing
      break;
  }

  //// TODO return uManager error
  return err;
}

int AlliedVisionCamera::OnPixelType(MM::PropertyBase* pProp,
                                    MM::ActionType eAct) {
  // TODO implement
  return 0;
}

int AlliedVisionCamera::onProperty(MM::PropertyBase* pProp,
                                   MM::ActionType eAct) {
  // Init
  std::vector<std::string> featureNames = {};
  VmbError_t err = VmbErrorSuccess;
  MM::Property* pChildProperty = (MM::Property*)pProp;
  const auto propertyName = pProp->GetName();

  // Check property mapping
  mapPropertyNameToFeatureNames(propertyName.c_str(), featureNames);

  if (propertyName == std::string(MM::g_Keyword_Binning)) {
    // Binning requires special handling and combining two features into one
    // property
    OnBinning(pProp, eAct);
  } else {
    // Retrive each feature
    for (const auto& featureName : featureNames) {
      // Get Feature Info
      VmbFeatureInfo_t featureInfo;
      err = g_api->VmbFeatureInfoQuery_t(m_handle, featureName.c_str(),
                                         &featureInfo, sizeof(featureInfo));
      if (VmbErrorSuccess != err) {
        return err;
      }

      // Get Access Mode
      bool rMode, wMode, readOnly, featureAvailable;
      err = g_api->VmbFeatureAccessQuery_t(m_handle, featureName.c_str(),
                                           &rMode, &wMode);
      if (VmbErrorSuccess != err) {
        return err;
      }

      featureAvailable = (rMode || wMode);
      if (!featureAvailable) {
        return err;
      }

      readOnly = featureAvailable && (rMode && !wMode);

      // Get values
      std::string propertyValue, featureValue;
      pProp->Get(propertyValue);
      err = getFeatureValue(&featureInfo, featureName.c_str(), featureValue);
      if (VmbErrorSuccess != err) {
        return err;
      }

      // Handle property value change
      switch (eAct) {
        case MM::ActionType::BeforeGet:  //!< Update property from feature
          // Update property
          if (propertyValue != featureValue) {
            pProp->Set(featureValue.c_str());
          }

          // Update property's access mode
          pChildProperty->SetReadOnly(readOnly);
          // Update property's limits and allowed values
          if (!readOnly) {
            err = setAllowedValues(&featureInfo, propertyName.c_str());
            if (VmbErrorSuccess != err) {
              return err;
            }
          }
          break;
        case MM::ActionType::AfterSet:  //!< Update feature from property
          if (propertyValue != featureValue) {
            err = setFeatureValue(&featureInfo, featureName.c_str(),
                                  propertyValue);
            if (err != VmbErrorSuccess) {
              if (featureInfo.featureDataType ==
                      VmbFeatureDataType::VmbFeatureDataFloat ||
                  featureInfo.featureDataType ==
                      VmbFeatureDataType::VmbFeatureDataInt) {
                auto propertyItem = m_propertyItems.at(propertyName);
                std::string adjustedValue =
                    adjustValue(propertyItem.m_min, propertyItem.m_max,
                                propertyItem.m_step, std::stod(propertyValue));
                pProp->Set(adjustedValue.c_str());
                (void)setFeatureValue(&featureInfo, featureName.c_str(),
                                      adjustedValue);
              }
            }
          }
          break;
        default:
          // nothing
          break;
      }
    }
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
      value = (out ? g_True : g_False);
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
      // TODO
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
      VmbBool_t out = (value == g_True ? true : false);
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
      // TODO
    case VmbFeatureDataUnknown:
    case VmbFeatureDataRaw:
    case VmbFeatureDataNone:
    default:
      // nothing
      break;
  }
  return err;
}

void AlliedVisionCamera::mapFeatureNameToPropertyName(
    const char* feature, std::string& property) const {
  property = std::string(feature);
  auto search = m_featureToProperty.find(property);
  if (search != m_featureToProperty.end()) {
    property = search->second;
  }
}
void AlliedVisionCamera::mapPropertyNameToFeatureNames(
    const char* property, std::vector<std::string>& featureNames) const {
  // Check property mapping
  auto searchRange = m_propertyToFeature.equal_range(property);
  if (searchRange.first != m_propertyToFeature.end()) {
    // Features that are mapped from property
    for (auto it = searchRange.first; it != searchRange.second; ++it) {
      featureNames.push_back(it->second);
    }
  } else {
    // for rest
    featureNames.push_back(property);
  }
}

std::string AlliedVisionCamera::adjustValue(double min, double max, double step,
                                            double propertyValue) const {
  if (propertyValue > max) {
    return std::to_string(max);
  }
  if (propertyValue < min) {
    return std::to_string(min);
  }

  VmbInt64_t factor = static_cast<VmbInt64_t>((propertyValue - min) / step);
  double prev = min + factor * step;
  double next = min + (factor + 1) * step;

  double prevDiff = abs(propertyValue - prev);
  double nextDiff = abs(next - propertyValue);

  return (nextDiff < prevDiff) ? std::to_string(next) : std::to_string(prev);
}

VmbError_t AlliedVisionCamera::setAllowedValues(const VmbFeatureInfo_t* feature,
                                                const char* propertyName) {
  if (feature == nullptr || propertyName == nullptr) {
    return VmbErrorInvalidValue;
  }

  VmbError_t err = VmbErrorSuccess;
  auto search = m_propertyItems.find(propertyName);
  if (search == m_propertyItems.end()) {
    LogMessage("Cannot find propery on internal list");
    return VmbErrorInvalidValue;
  }

  switch (feature->featureDataType) {
    case VmbFeatureDataBool: {
      AddAllowedValue(propertyName, g_False);
      AddAllowedValue(propertyName, g_True);
      break;
    }
    case VmbFeatureDataFloat: {
      double min = 0;
      double max = 0;
      err = g_api->VmbFeatureFloatRangeQuery_t(m_handle, feature->name, &min,
                                               &max);
      if (VmbErrorSuccess != err || min == max) {
        return err;
      }

      double step = 0.0;
      bool isIncremental = false;
      err = g_api->VmbFeatureFloatIncrementQuery_t(m_handle, feature->name,
                                                   &isIncremental, &step);
      if (VmbErrorSuccess != err) {
        return err;
      }

      PropertyItem tempProp{propertyName, static_cast<double>(min),
                            static_cast<double>(max),
                            static_cast<double>(step)};
      if (tempProp == search->second) {
        break;
      }

      m_propertyItems.at(propertyName).m_min = static_cast<double>(min);
      m_propertyItems.at(propertyName).m_max = static_cast<double>(max);
      m_propertyItems.at(propertyName).m_step = static_cast<double>(step);

      err = SetPropertyLimits(propertyName, min, max);
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
      err = SetAllowedValues(propertyName, strValues);

      break;
    }
    case VmbFeatureDataInt: {
      VmbInt64_t min, max, step;
      std::vector<std::string> strValues;

      err =
          g_api->VmbFeatureIntRangeQuery_t(m_handle, feature->name, &min, &max);
      if (VmbErrorSuccess != err || min == max) {
        return err;
      }

      err =
          g_api->VmbFeatureIntIncrementQuery_t(m_handle, feature->name, &step);
      if (VmbErrorSuccess != err) {
        return err;
      }

      PropertyItem tempProp{propertyName, static_cast<double>(min),
                            static_cast<double>(max),
                            static_cast<double>(step)};
      if (tempProp == search->second) {
        break;
      }

      m_propertyItems.at(propertyName).m_min = static_cast<double>(min);
      m_propertyItems.at(propertyName).m_max = static_cast<double>(max);
      m_propertyItems.at(propertyName).m_step = static_cast<double>(step);

      err = SetPropertyLimits(propertyName, min, max);
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
    VmbUint8_t* buffer = reinterpret_cast<VmbUint8_t*>(frame->imageData);

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