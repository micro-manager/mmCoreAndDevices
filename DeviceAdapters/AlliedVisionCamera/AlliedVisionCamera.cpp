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

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>

#include "AlliedVisionHub.h"
#include "ModuleInterface.h"
#include "VmbC/VmbC.h"

///////////////////////////////////////////////////////////////////////////////
// STATIC VALUES
///////////////////////////////////////////////////////////////////////////////
const std::unordered_map<std::string, std::string>
    AlliedVisionCamera::m_featureToProperty = {
        {g_PixelFormatFeature, MM::g_Keyword_PixelType},
        {g_ExposureFeature, MM::g_Keyword_Exposure}};

const std::unordered_multimap<std::string, std::string>
    AlliedVisionCamera::m_propertyToFeature = {
        {MM::g_Keyword_PixelType, g_PixelFormatFeature},
        {MM::g_Keyword_Exposure, g_ExposureFeature}};

///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////
MODULE_API void InitializeModuleData() {
  RegisterDevice(g_hubName, MM::HubDevice, "Allied Vision Hub");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName) {
  if (deviceName == nullptr) {
    return nullptr;
  }

  if (std::string(deviceName) == std::string(g_hubName)) {
    return new AlliedVisionHub();
  } else {
    return new AlliedVisionCamera(deviceName);
  }
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
    : m_sdk(nullptr),
      m_handle{nullptr},
      m_cameraName{deviceName},
      m_frames{},
      m_buffer{},
      m_bufferSize{0},
      m_payloadSize{0},
      m_isAcquisitionRunning{false},
      m_currentPixelFormat{} {
  CreateHubIDProperty();
  // Binning property is a Core Property, we will have a dummy one
  CreateProperty(MM::g_Keyword_Binning, "N/A", MM::String, true, nullptr);
  AddAllowedValue(MM::g_Keyword_Binning, "N/A");
}

int AlliedVisionCamera::Initialize() {
  auto parentHub = dynamic_cast<AlliedVisionHub*>(GetParentHub());
  if (parentHub == nullptr) {
    LOG_ERROR(VmbErrorBadParameter, "Parent HUB not found!");
    return VmbErrorBadParameter;
  }
  m_sdk = parentHub->getSDK();

  LogMessage("Opening camera: " + m_cameraName);
  VmbError_t err = m_sdk->VmbCameraOpen_t(
      m_cameraName.c_str(), VmbAccessModeType::VmbAccessModeFull, &m_handle);
  if (err != VmbErrorSuccess || m_handle == nullptr) {
    LOG_ERROR(err, "Error while opening camera or handle is NULL!");
    return err;
  }

  // Ignore errors from setting up properties
  (void)setupProperties();
  return resizeImageBuffer();
}

int AlliedVisionCamera::Shutdown() {
  LogMessage("Shutting down camera: " + m_cameraName);
  VmbError_t err = VmbErrorSuccess;
  if (m_sdk != nullptr && m_sdk->isInitialized()) {
    if (m_handle != nullptr) {
      err = m_sdk->VmbCameraClose_t(m_handle);
    }
  }

  return err;
}

VmbError_t AlliedVisionCamera::setupProperties() {
  VmbUint32_t featureCount = 0;
  VmbError_t err = m_sdk->VmbFeaturesList_t(m_handle, NULL, 0, &featureCount,
                                            sizeof(VmbFeatureInfo_t));
  if (err != VmbErrorSuccess) {
    LOG_ERROR(err, "Error occurred when obtaining features count!");
    return err;
  }

  std::shared_ptr<VmbFeatureInfo_t> features =
      std::shared_ptr<VmbFeatureInfo_t>(new VmbFeatureInfo_t[featureCount]);
  err = m_sdk->VmbFeaturesList_t(m_handle, features.get(), featureCount,
                                 &featureCount, sizeof(VmbFeatureInfo_t));
  if (err != VmbErrorSuccess) {
    LOG_ERROR(err, "Error occurred when obtaining features!");
    return err;
  }

  const VmbFeatureInfo_t* end = features.get() + featureCount;
  for (VmbFeatureInfo_t* feature = features.get(); feature != end; ++feature) {
    err = createPropertyFromFeature(feature);
    if (err != VmbErrorSuccess) {
      LOG_ERROR(err,
                "Error while creating property" + std::string(feature->name));
      continue;
    }
  }

  return VmbErrorSuccess;
}

VmbError_t AlliedVisionCamera::resizeImageBuffer() {
  VmbError_t err = m_sdk->VmbPayloadSizeGet_t(m_handle, &m_payloadSize);
  if (err != VmbErrorSuccess) {
    LOG_ERROR(err, "Error while reading payload size");
    return err;
  }

  m_bufferSize = std::max(GetImageWidth() * GetImageHeight() *
                              m_currentPixelFormat.getBytesPerPixel(),
                          m_payloadSize);

  for (size_t i = 0; i < MAX_FRAMES; i++) {
    delete[] m_buffer[i];
    m_buffer[i] = new VmbUint8_t[m_bufferSize];
  }

  return VmbErrorSuccess;
}

VmbError_t AlliedVisionCamera::createPropertyFromFeature(
    const VmbFeatureInfo_t* feature) {
  if (feature == nullptr) {
    LogMessage("Cannot create feature. It is NULL");
    return VmbErrorInvalidValue;
  }

  VmbError_t err = VmbErrorSuccess;
  // Skip Event, Chunk, RAW features
  std::string featureCategory = feature->category;
  if (featureCategory.find(g_EventCategory) != std::string::npos ||
      featureCategory.find(g_ChunkCategory) != std::string::npos ||
      feature->featureDataType == VmbFeatureDataRaw) {
    // Skip
    return err;
  }

  // Map feature to property name
  std::string propertyName = {};
  mapFeatureNameToPropertyName(feature->name, propertyName);

  // uManager callback
  CPropertyAction* uManagerCallback =
      new CPropertyAction(this, &AlliedVisionCamera::onProperty);

  // Vimba callback
  auto vmbCallback = [](VmbHandle_t handle, const char* name,
                        void* userContext) {
    (void)handle;
    AlliedVisionCamera* camera =
        reinterpret_cast<AlliedVisionCamera*>(userContext);
    std::string propertyName;
    camera->mapFeatureNameToPropertyName(name, propertyName);
    auto err = camera->UpdateProperty(propertyName.c_str());
    if (err != VmbErrorSuccess) {
      camera->LOG_ERROR(err, "Property: " + propertyName + " update failed");
    }
  };

  // Register VMB callback for given feature
  err = m_sdk->VmbFeatureInvalidationRegister_t(m_handle, feature->name,
                                                vmbCallback, this);
  if (err != VmbErrorSuccess) {
    LOG_ERROR(err, "Error while registering invalidation callback for " +
                       std::string(feature->name));
    return err;
  }

  switch (feature->featureDataType) {
    case VmbFeatureDataInt: {
      err = CreateIntegerProperty(propertyName.c_str(), 0, false,
                                  uManagerCallback);
      break;
    }
    case VmbFeatureDataBool: {
      err = CreateStringProperty(propertyName.c_str(), g_False, false,
                                 uManagerCallback);
      AddAllowedValue(propertyName.c_str(), g_False);
      AddAllowedValue(propertyName.c_str(), g_True);
      break;
    }
    case VmbFeatureDataCommand: {
      err = CreateStringProperty(propertyName.c_str(), g_Command, false,
                                 uManagerCallback);
      AddAllowedValue(propertyName.c_str(), g_Command);
      AddAllowedValue(propertyName.c_str(), g_Execute);
      break;
    }
    case VmbFeatureDataEnum:
    case VmbFeatureDataString: {
      err = CreateStringProperty(propertyName.c_str(), "", false,
                                 uManagerCallback);
      break;
    }
    case VmbFeatureDataFloat: {
      err = CreateFloatProperty(propertyName.c_str(), 0.0, false,
                                uManagerCallback);
      break;
    }
    case VmbFeatureDataUnknown:
    case VmbFeatureDataRaw:
    case VmbFeatureDataNone:
    default:
      // nothing
      break;
  }

  if (err != VmbErrorSuccess) {
    LOG_ERROR(err,
              "Error while creating property " + std::string(feature->name));
  }

  return err;
}

const unsigned char* AlliedVisionCamera::GetImageBuffer() {
  return reinterpret_cast<VmbUint8_t*>(m_buffer[0]);
}

unsigned AlliedVisionCamera::GetImageWidth() const {
  std::string value{};
  auto ret = getFeatureValue(g_Width, value);
  if (ret != VmbErrorSuccess) {
    LOG_ERROR(ret, "Error while getting image width!");
    return 0;
  }

  return atoi(value.c_str());
}

unsigned AlliedVisionCamera::GetImageHeight() const {
  std::string value{};
  auto ret = getFeatureValue(g_Height, value);
  if (ret != VmbErrorSuccess) {
    LOG_ERROR(ret, "Error while getting image height!");
    return 0;
  }

  return atoi(value.c_str());
}

unsigned AlliedVisionCamera::GetImageBytesPerPixel() const {
  return m_currentPixelFormat.getBytesPerPixel();
}

long AlliedVisionCamera::GetImageBufferSize() const { return m_bufferSize; }

unsigned AlliedVisionCamera::GetBitDepth() const {
  return m_currentPixelFormat.getBitDepth();
}

unsigned AlliedVisionCamera::GetNumberOfComponents() const {
  return m_currentPixelFormat.getNumberOfComponents();
}

int AlliedVisionCamera::GetBinning() const {
  // Binning not supported. We support BinningVertical/Horizontal
  return 1;
}

int AlliedVisionCamera::SetBinning(int binSize) {
  // Binning not supported. We support BinningVertical/Horizontal
  return DEVICE_ERR;
}

double AlliedVisionCamera::GetExposure() const {
  std::string value{};
  auto ret = getFeatureValue(g_ExposureFeature, value);
  if (ret != VmbErrorSuccess) {
    LOG_ERROR(ret, "Error while getting exposure!");
    return 0;
  }

  return strtod(value.c_str(), nullptr);
}

void AlliedVisionCamera::SetExposure(double exp_ms) {
  SetProperty(MM::g_Keyword_Exposure, CDeviceUtils::ConvertToString(exp_ms));
  GetCoreCallback()->OnExposureChanged(this, exp_ms);
}

int AlliedVisionCamera::SetROI(unsigned x, unsigned y, unsigned xSize,
                               unsigned ySize) {
  auto width = GetImageWidth();
  auto height = GetImageHeight();
  VmbError_t err = VmbErrorSuccess;

  std::function<VmbError_t(unsigned)> setOffsetXProperty = [this](long x) {
    auto err = SetProperty(g_OffsetX, CDeviceUtils::ConvertToString(x));
    if (err) {
      LOG_ERROR(err, "Error while ROI Offset X!");
    }
    return err;
  };
  std::function<VmbError_t(unsigned)> setOffsetYProperty = [this](long y) {
    auto err = SetProperty(g_OffsetY, CDeviceUtils::ConvertToString(y));
    if (err) {
      LOG_ERROR(err, "Error while ROI Offset Y!");
    }
    return err;
  };
  std::function<VmbError_t(unsigned)> setWidthProperty = [this](long xSize) {
    auto err = SetProperty(g_Width, CDeviceUtils::ConvertToString(xSize));
    if (err) {
      LOG_ERROR(err, "Error while ROI X!");
    }
    return err;
  };
  std::function<VmbError_t(unsigned)> setHeightProperty = [this](long ySize) {
    auto err = SetProperty(g_Height, CDeviceUtils::ConvertToString(ySize));
    if (err) {
      LOG_ERROR(err, "Error while ROI Y!");
    }
    return err;
  };

  if (xSize > width) {
    err = setOffsetXProperty(x) || setWidthProperty(xSize);
  } else {
    err = setWidthProperty(xSize) || setOffsetXProperty(x);
  }

  if (ySize > height) {
    err = setOffsetYProperty(y) || setHeightProperty(ySize);
  } else {
    err = setHeightProperty(ySize) || setOffsetYProperty(y);
  }

  if (err != VmbErrorSuccess) {
    return err;
  }

  return resizeImageBuffer();
}

int AlliedVisionCamera::GetROI(unsigned& x, unsigned& y, unsigned& xSize,
                               unsigned& ySize) {
  std::map<const char*, unsigned> fields = {
      {g_OffsetX, x}, {g_OffsetY, y}, {g_Width, xSize}, {g_Height, ySize}};

  VmbError_t err = VmbErrorSuccess;
  for (auto& field : fields) {
    std::string value{};
    err = getFeatureValue(field.first, value);
    if (err != VmbErrorSuccess) {
      LOG_ERROR(err, "Error while getting ROI!");
      break;
    }
    field.second = atoi(value.data());
  }

  return err;
}

int AlliedVisionCamera::ClearROI() {
  std::string maxWidth, maxHeight;
  VmbError_t err = getFeatureValue(g_WidthMax, maxWidth) |
                   getFeatureValue(g_HeightMax, maxHeight);
  if (VmbErrorSuccess != err) {
    LOG_ERROR(err, "Error while clearing ROI!");
    return err;
  }

  // Keep the order of the fields
  std::vector<std::pair<const char*, std::string>> fields = {
      {g_OffsetX, "0"},
      {g_OffsetY, "0"},
      {g_Width, maxWidth},
      {g_Height, maxHeight}};

  for (auto& field : fields) {
    err = setFeatureValue(field.first, field.second);
    if (err != VmbErrorSuccess) {
      LOG_ERROR(err, "Error while clearing ROI!");
      break;
    }
  }

  return resizeImageBuffer();
}

int AlliedVisionCamera::IsExposureSequenceable(bool& isSequenceable) const {
  // TODO implement
  return VmbErrorSuccess;
}

void AlliedVisionCamera::GetName(char* name) const {
  CDeviceUtils::CopyLimitedString(name, m_cameraName.c_str());
}

bool AlliedVisionCamera::IsCapturing() { return m_isAcquisitionRunning; }

int AlliedVisionCamera::onProperty(MM::PropertyBase* pProp,
                                   MM::ActionType eAct) {
  // Init
  std::vector<std::string> featureNames = {};
  VmbError_t err = VmbErrorSuccess;
  MM::Property* pChildProperty = (MM::Property*)pProp;
  const auto propertyName = pProp->GetName();

  // Check property mapping
  mapPropertyNameToFeatureNames(propertyName.c_str(), featureNames);

  // Retrive each feature
  for (const auto& featureName : featureNames) {
    // Get Feature Info and Access Mode
    VmbFeatureInfo_t featureInfo;
    bool rMode{}, wMode{}, readOnly{}, featureAvailable{};
    err = m_sdk->VmbFeatureInfoQuery_t(m_handle, featureName.c_str(),
                                       &featureInfo, sizeof(featureInfo)) |
          m_sdk->VmbFeatureAccessQuery_t(m_handle, featureName.c_str(), &rMode,
                                         &wMode);
    if (VmbErrorSuccess != err) {
      LOG_ERROR(err, "Error while getting info or access query!");
      return err;
    }

    readOnly = (rMode && !wMode);
    featureAvailable = rMode || wMode;
   
    // Get values
    std::string propertyValue{}, featureValue{};
    pProp->Get(propertyValue);

    // Handle property value change
    switch (eAct) {
      case MM::ActionType::BeforeGet:  //!< Update property from feature
        
        // Update feature range
        if (featureAvailable)
        {
          err = setAllowedValues(&featureInfo, propertyName.c_str());
        }
        // Feature not available -> clear value and range
        else {
          switch (featureInfo.featureDataType) {
            case VmbFeatureDataInt:
            case VmbFeatureDataFloat:
              err = SetPropertyLimits(propertyName.c_str(), 0.0, 0.0);
              pProp->Set("0");
              break;
            case VmbFeatureDataEnum:
            case VmbFeatureDataString:
            case VmbFeatureDataBool:
            case VmbFeatureDataCommand:
              ClearAllowedValues(propertyName.c_str());
              pProp->Set("");
              break;
            case VmbFeatureDataRaw:
            case VmbFeatureDataNone:
            default:
              // feature type not supported
              break;
          }
        }
          
        if (rMode) {
          err =
              getFeatureValue(&featureInfo, featureName.c_str(), featureValue);
          if (VmbErrorSuccess != err) {
            LOG_ERROR(err, "Error while getting feature value " + featureName);
            return err;
          }

          // Update property
          if (propertyValue != featureValue) {
            pProp->Set(featureValue.c_str());
            err = GetCoreCallback()->OnPropertyChanged(
                this, propertyName.c_str(), featureValue.c_str());
            if (propertyName == MM::g_Keyword_PixelType) {
              handlePixelFormatChange(featureValue);
            }

            if (VmbErrorSuccess != err) {
              LOG_ERROR(err,
                        "Error while calling OnPropertyChanged callback for " +
                            featureName);
              return err;
            }
          }
        }

        // Set property to readonly (grey out in GUI) if it is readonly or unavailable
        pChildProperty->SetReadOnly(readOnly || !featureAvailable);

        break;
      case MM::ActionType::AfterSet:  //!< Update feature from property
        err = setFeatureValue(&featureInfo, featureName.c_str(), propertyValue);
        if (err == VmbErrorInvalidValue) {
          // Update limits first to have latest min and max
          err = setAllowedValues(&featureInfo, propertyName.c_str());
          if (VmbErrorSuccess != err) {
            LOG_ERROR(err, "Error while setting allowed values for feature " +
                               featureName);
            return err;
          }

          // Adjust value
          double min{}, max{};
          err = GetPropertyLowerLimit(propertyName.c_str(), min) |
                GetPropertyUpperLimit(propertyName.c_str(), max);
          if (VmbErrorSuccess != err) {
            LOG_ERROR(err, "Error while getting limits for " + propertyName);
            return err;
          }
          std::string adjustedValue =
              adjustValue(featureInfo, min, max, std::stod(propertyValue));
          err =
              setFeatureValue(&featureInfo, featureName.c_str(), adjustedValue);
        }

        if (propertyName == MM::g_Keyword_PixelType) {
          handlePixelFormatChange(propertyValue);
        }
        break;
      default:
        // nothing
        break;
    }
  }

  if (VmbErrorSuccess != err) {
    LOG_ERROR(err, "Error while updating property " + propertyName);
  }

  return err;
}

void AlliedVisionCamera::handlePixelFormatChange(const std::string& pixelType) {
  m_currentPixelFormat.setPixelType(pixelType);
}

VmbError_t AlliedVisionCamera::getFeatureValue(VmbFeatureInfo_t* featureInfo,
                                               const char* featureName,
                                               std::string& value) const {
  VmbError_t err = VmbErrorSuccess;
  switch (featureInfo->featureDataType) {
    case VmbFeatureDataBool: {
      VmbBool_t out;
      err = m_sdk->VmbFeatureBoolGet_t(m_handle, featureName, &out);
      if (err != VmbErrorSuccess) {
        break;
      }
      value = (out ? g_True : g_False);
      break;
    }
    case VmbFeatureDataEnum: {
      const char* out = nullptr;
      err = m_sdk->VmbFeatureEnumGet_t(m_handle, featureName, &out);
      if (err != VmbErrorSuccess) {
        break;
      }
      value = std::string(out);
      break;
    }
    case VmbFeatureDataFloat: {
      double out;
      err = m_sdk->VmbFeatureFloatGet_t(m_handle, featureName, &out);
      if (err != VmbErrorSuccess) {
        break;
      }
      value = std::to_string(out);
      break;
    }
    case VmbFeatureDataInt: {
      VmbInt64_t out;
      err = m_sdk->VmbFeatureIntGet_t(m_handle, featureName, &out);
      if (err != VmbErrorSuccess) {
        break;
      }
      value = std::to_string(out);
      break;
    }
    case VmbFeatureDataString: {
      VmbUint32_t size = 0;
      err = m_sdk->VmbFeatureStringGet_t(m_handle, featureName, nullptr, 0,
                                         &size);
      if (VmbErrorSuccess == err && size > 0) {
        std::shared_ptr<char> buff = std::shared_ptr<char>(new char[size]);
        err = m_sdk->VmbFeatureStringGet_t(m_handle, featureName, buff.get(),
                                           size, &size);
        if (err != VmbErrorSuccess) {
          break;
        }
        value = std::string(buff.get());
      }
      break;
    }
    case VmbFeatureDataCommand:
      value = std::string(g_Command);
      break;
    case VmbFeatureDataUnknown:
    case VmbFeatureDataRaw:
    case VmbFeatureDataNone:
    default:
      // nothing
      break;
  }

  if (VmbErrorSuccess != err) {
    LOG_ERROR(err,
              "Error while getting feature value " + std::string(featureName));
  }

  return err;
}

VmbError_t AlliedVisionCamera::getFeatureValue(const char* featureName,
                                               std::string& value) const {
  VmbFeatureInfo_t featureInfo;
  VmbError_t err = m_sdk->VmbFeatureInfoQuery_t(
      m_handle, featureName, &featureInfo, sizeof(featureInfo));
  if (VmbErrorSuccess != err) {
    LOG_ERROR(err,
              "Error while getting feature value " + std::string(featureName));
    return err;
  }

  return getFeatureValue(&featureInfo, featureName, value);
}

VmbError_t AlliedVisionCamera::setFeatureValue(VmbFeatureInfo_t* featureInfo,
                                               const char* featureName,
                                               std::string& value) {
  VmbError_t err = VmbErrorSuccess;
  std::stringstream ss(value);
  bool isDone = false;
  VmbUint32_t maxLen = 0;
  std::string property{};

  switch (featureInfo->featureDataType) {
    case VmbFeatureDataBool: {
      VmbBool_t out = (value == g_True ? true : false);
      err = m_sdk->VmbFeatureBoolSet_t(m_handle, featureName, out);
      break;
    }
    case VmbFeatureDataEnum: {
      err = m_sdk->VmbFeatureEnumSet_t(m_handle, featureName, value.c_str());
      break;
    }
    case VmbFeatureDataFloat: {
      double out;
      ss >> out;
      err = m_sdk->VmbFeatureFloatSet_t(m_handle, featureName, out);
      break;
    }
    case VmbFeatureDataInt: {
      VmbInt64_t out;
      ss >> out;
      err = m_sdk->VmbFeatureIntSet_t(m_handle, featureName, out);
      break;
    }
    case VmbFeatureDataString: {
      err = m_sdk->VmbFeatureStringMaxlengthQuery_t(m_handle, featureName,
                                                    &maxLen);
      if (err != VmbErrorSuccess) {
        break;
      }
      if (value.size() > maxLen) {
        err = VmbErrorInvalidValue;
      } else {
        err =
            m_sdk->VmbFeatureStringSet_t(m_handle, featureName, value.c_str());
      }
      break;
    }
    case VmbFeatureDataCommand:
      if (value == g_Execute) {
        mapFeatureNameToPropertyName(featureName, property);
        if (!property.empty()) {
          err = m_sdk->VmbFeatureCommandRun_t(m_handle, featureName);
          if (err != VmbErrorSuccess) {
            break;
          }
          while (!isDone) {
            err = m_sdk->VmbFeatureCommandIsDone_t(m_handle, featureName,
                                                   &isDone);
            if (err != VmbErrorSuccess) {
              break;
            }
          }
          // Set back property to "Command"
          err = SetProperty(property.c_str(), g_Command);
          GetCoreCallback()->OnPropertyChanged(this, property.c_str(),
                                               g_Command);
          if (err != VmbErrorSuccess) {
            break;
          }
        } else {
          err = VmbErrorInvalidValue;
        }
      }
      break;
    case VmbFeatureDataUnknown:
    case VmbFeatureDataRaw:
    case VmbFeatureDataNone:
    default:
      // nothing
      break;
  }

  if (VmbErrorSuccess != err) {
    LOG_ERROR(err,
              "Error while setting feature value " + std::string(featureName));
  }

  return err;
}

VmbError_t AlliedVisionCamera::setFeatureValue(const char* featureName,
                                               std::string& value) {
  VmbFeatureInfo_t featureInfo;
  VmbError_t err = m_sdk->VmbFeatureInfoQuery_t(
      m_handle, featureName, &featureInfo, sizeof(featureInfo));
  if (VmbErrorSuccess != err) {
    LOG_ERROR(err,
              "Error while setting feature value " + std::string(featureName));
    return err;
  }

  return setFeatureValue(&featureInfo, featureName, value);
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

std::string AlliedVisionCamera::adjustValue(VmbFeatureInfo_t& featureInfo,
                                            double min, double max,
                                            double propertyValue) const {
  VmbError_t err = VmbErrorSuccess;
  double step = 1.0;
  VmbInt64_t stepI = 1;
  bool isIncremental = true;
  switch (featureInfo.featureDataType) {
    case VmbFeatureDataFloat:
      err = m_sdk->VmbFeatureFloatIncrementQuery_t(m_handle, featureInfo.name,
                                                   &isIncremental, &step);
      break;
    case VmbFeatureDataInt:
      err = m_sdk->VmbFeatureIntIncrementQuery_t(m_handle, featureInfo.name,
                                                 &stepI);
      step = static_cast<double>(stepI);
      break;
    default:
      // nothing
      break;
  }
  if (VmbErrorSuccess != err) {
    LOG_ERROR(err, "Error while getting increment query for feature " +
                       std::string(featureInfo.name));
    return std::to_string(propertyValue);
  }

  if (!isIncremental) {
    return std::to_string(propertyValue);
  }

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

  switch (feature->featureDataType) {
    case VmbFeatureDataBool: {
      // Already set in creation, but maybe reset when become unavailable
      if (GetNumberOfPropertyValues(propertyName) == 0) {
          AddAllowedValue(propertyName, g_False);
          AddAllowedValue(propertyName, g_True);
      }
      break;
    }
    case VmbFeatureDataCommand: {
      // Already set in creation, but maybe reset when become unavailable
      if (GetNumberOfPropertyValues(propertyName) == 0) {
        AddAllowedValue(propertyName, g_Command);
        AddAllowedValue(propertyName, g_Execute);
      }
      break;
    }
    case VmbFeatureDataFloat: {
      double min, max;
      err = m_sdk->VmbFeatureFloatRangeQuery_t(m_handle, feature->name, &min,
                                               &max);
      if (VmbErrorSuccess != err || min == max) {
        break;
      }

      err = SetPropertyLimits(propertyName, min, max);
      break;
    }
    case VmbFeatureDataEnum: {
      std::array<const char*, MM::MaxStrLength> values;
      std::vector<std::string> strValues;
      VmbUint32_t valuesNum = 0;
      err = m_sdk->VmbFeatureEnumRangeQuery_t(
          m_handle, feature->name, values.data(), MM::MaxStrLength, &valuesNum);
      if (VmbErrorSuccess != err) {
        break;
      }

      for (size_t i = 0; i < valuesNum; i++) {
        strValues.push_back(values[i]);
      }
      err = SetAllowedValues(propertyName, strValues);

      break;
    }
    case VmbFeatureDataInt: {
      VmbInt64_t min, max;
      std::vector<std::string> strValues;

      err =
          m_sdk->VmbFeatureIntRangeQuery_t(m_handle, feature->name, &min, &max);
      if (VmbErrorSuccess != err || min == max) {
        break;
      }

      err = SetPropertyLimits(propertyName, static_cast<double>(min),
                              static_cast<double>(max));
      break;
    }
    case VmbFeatureDataString:
    case VmbFeatureDataRaw:
    case VmbFeatureDataNone:
    default:
      // nothing
      break;
  }

  if (VmbErrorSuccess != err) {
    LOG_ERROR(err, "Error while setting allowed values for feature " +
                       std::string(feature->name));
  }

  return err;
}

int AlliedVisionCamera::SnapImage() {
  if (IsCapturing()) {
    return DEVICE_CAMERA_BUSY_ACQUIRING;
  }
  resizeImageBuffer();

  VmbFrame_t frame;
  frame.buffer = m_buffer[0];
  frame.bufferSize = m_payloadSize;

  VmbError_t err = VmbErrorSuccess;
  err = m_sdk->VmbFrameAnnounce_t(m_handle, &frame, sizeof(VmbFrame_t));
  if (err != VmbErrorSuccess) {
    LOG_ERROR(err, "Error while snapping image!");
    (void)StopSequenceAcquisition();
    return err;
  }
  err = m_sdk->VmbCaptureStart_t(m_handle);
  if (err != VmbErrorSuccess) {
    LOG_ERROR(err, "Error while snapping image!");
    (void)StopSequenceAcquisition();
    return err;
  }
  err = m_sdk->VmbCaptureFrameQueue_t(m_handle, &frame, nullptr);
  if (err != VmbErrorSuccess) {
    LOG_ERROR(err, "Error while snapping image!");
    (void)StopSequenceAcquisition();
    return err;
  }
  err = m_sdk->VmbFeatureCommandRun_t(m_handle, g_AcquisitionStart);
  if (err != VmbErrorSuccess) {
    LOG_ERROR(err, "Error while snapping image!");
    (void)StopSequenceAcquisition();
    return err;
  }
  m_isAcquisitionRunning = true;
  err = m_sdk->VmbCaptureFrameWait_t(m_handle, &frame, 3000);
  if (err != VmbErrorSuccess) {
    LOG_ERROR(err, "Error while snapping image!");
    (void)StopSequenceAcquisition();
    return err;
  }

  err = transformImage(&frame);
  if (err != VmbErrorSuccess) {
    LOG_ERROR(err, "Error while snapping image - cannot transform image!");
    (void)StopSequenceAcquisition();
    return err;
  }

  (void)StopSequenceAcquisition();
  return err;
}

int AlliedVisionCamera::StartSequenceAcquisition(long numImages,
                                                 double interval_ms,
                                                 bool stopOnOverflow) {
  (void)stopOnOverflow;
  (void)interval_ms;
  (void)numImages;

  if (IsCapturing()) {
    return DEVICE_CAMERA_BUSY_ACQUIRING;
  }

  int err = GetCoreCallback()->PrepareForAcq(this);
  if (err != VmbErrorSuccess) {
    LOG_ERROR(err, "Error while preparing for acquisition!");
    return err;
  }

  err = resizeImageBuffer();
  if (err != VmbErrorSuccess) {
    LOG_ERROR(err, "Error during frame praparation for continous acquisition!");
    return err;
  }

  for (size_t i = 0; i < MAX_FRAMES; i++) {
    m_frames[i].buffer = m_buffer[i];
    m_frames[i].bufferSize = m_payloadSize;
    m_frames[i].context[0] = this;  //<! Pointer to camera
    m_frames[i].context[1] =
        reinterpret_cast<void*>(i);  //<! Pointer to frame index

    auto frameCallback = [](const VmbHandle_t cameraHandle,
                            const VmbHandle_t streamHandle, VmbFrame_t* frame) {
      (void)cameraHandle;
      (void)streamHandle;
      reinterpret_cast<AlliedVisionCamera*>(frame->context[0])
          ->insertFrame(frame);
    };

    err =
        m_sdk->VmbFrameAnnounce_t(m_handle, &(m_frames[i]), sizeof(VmbFrame_t));
    if (err != VmbErrorSuccess) {
      LOG_ERROR(err,
                "Error during frame praparation for continous acquisition!");
      return err;
    }

    err =
        m_sdk->VmbCaptureFrameQueue_t(m_handle, &(m_frames[i]), frameCallback);
    if (err != VmbErrorSuccess) {
      LOG_ERROR(err,
                "Error during frame praparation for continous acquisition!");
      return err;
    }
  }

  err = m_sdk->VmbCaptureStart_t(m_handle);
  if (err != VmbErrorSuccess) {
    LOG_ERROR(err, "Error during frame praparation for continous acquisition!");
    return err;
  }

  err = m_sdk->VmbFeatureCommandRun_t(m_handle, g_AcquisitionStart);
  m_isAcquisitionRunning = true;

  if (err != VmbErrorSuccess) {
    LOG_ERROR(err, "Error during start acquisition!");
    return err;
  }

  return UpdateStatus();
}

int AlliedVisionCamera::StartSequenceAcquisition(double interval_ms) {
  return StartSequenceAcquisition(LONG_MAX, interval_ms, true);
}
int AlliedVisionCamera::StopSequenceAcquisition() {
  // This method shall never return any error
  VmbError_t err = VmbErrorSuccess;
  if (IsCapturing()) {
    err = m_sdk->VmbFeatureCommandRun_t(m_handle, g_AcquisitionStop);
    m_isAcquisitionRunning = false;
    if (err != VmbErrorSuccess) {
      LOG_ERROR(err, "Error during stopping acquisition command!");
      return err;
    }
  }

  err = m_sdk->VmbCaptureEnd_t(m_handle);
  if (err != VmbErrorSuccess) {
    LOG_ERROR(err, "Error during stop acquisition!");
    return err;
  }

  err = m_sdk->VmbCaptureQueueFlush_t(m_handle);
  if (err != VmbErrorSuccess) {
    LOG_ERROR(err, "Error during stop acquisition!");
    return err;
  }

  err = m_sdk->VmbFrameRevokeAll_t(m_handle);
  if (err != VmbErrorSuccess) {
    LOG_ERROR(err, "Error during stop acquisition!");
    return err;
  }

  return UpdateStatus();
}

VmbError_t AlliedVisionCamera::transformImage(VmbFrame_t* frame) {
  VmbError_t err = VmbErrorSuccess;
  VmbImage src{}, dest{};
  VmbTransformInfo info{};
  std::shared_ptr<VmbUint8_t> tempBuff =
      std::shared_ptr<VmbUint8_t>(new VmbUint8_t[m_bufferSize]);
  auto srcBuff = reinterpret_cast<VmbUint8_t*>(frame->buffer);

  src.Data = srcBuff;
  src.Size = sizeof(src);

  dest.Data = tempBuff.get();
  dest.Size = sizeof(dest);

  err = m_sdk->VmbSetImageInfoFromPixelFormat_t(
      frame->pixelFormat, frame->width, frame->height, &src);
  if (err != VmbErrorSuccess) {
    LOG_ERROR(err, "Cannot set image info from pixel format!");
    return err;
  }

  err = m_sdk->VmbSetImageInfoFromPixelFormat_t(
      m_currentPixelFormat.getVmbFormat(), frame->width, frame->height, &dest);
  if (err != VmbErrorSuccess) {
    LOG_ERROR(err, "Cannot set image info from pixel format!");
    return err;
  }

  err = m_sdk->VmbImageTransform_t(&src, &dest, &info, 0);
  if (err != VmbErrorSuccess) {
    LOG_ERROR(err, "Cannot transform image!");
    return err;
  }

  memcpy(srcBuff, tempBuff.get(), m_bufferSize);
  return err;
}

void AlliedVisionCamera::insertFrame(VmbFrame_t* frame) {
  if (frame != nullptr && frame->receiveStatus == VmbFrameStatusComplete) {
    VmbError_t err = VmbErrorSuccess;

    err = transformImage(frame);
    if (err != VmbErrorSuccess) {
      // Error logged in transformImage
      return;
    }

    // TODO implement metadata
    Metadata md;
    md.put("Camera", m_cameraName);

    VmbUint8_t* buffer = reinterpret_cast<VmbUint8_t*>(frame->buffer);
    err = GetCoreCallback()->InsertImage(
        this, buffer, GetImageWidth(), GetImageHeight(),
        m_currentPixelFormat.getBytesPerPixel(),
        m_currentPixelFormat.getNumberOfComponents(), md.Serialize().c_str());

    if (err == DEVICE_BUFFER_OVERFLOW) {
      GetCoreCallback()->ClearImageBuffer(this);
      err = GetCoreCallback()->InsertImage(
          this, buffer, GetImageWidth(), GetImageHeight(),
          m_currentPixelFormat.getBytesPerPixel(),
          m_currentPixelFormat.getNumberOfComponents(), md.Serialize().c_str(),
          false);
    }

    if (IsCapturing()) {
      m_sdk->VmbCaptureFrameQueue_t(
          m_handle, frame,
          [](const VmbHandle_t cameraHandle, const VmbHandle_t streamHandle,
             VmbFrame_t* frame) {
            (void)cameraHandle;
            (void)streamHandle;
            reinterpret_cast<AlliedVisionCamera*>(frame->context[0])
                ->insertFrame(frame);
          });
    }
  }
}