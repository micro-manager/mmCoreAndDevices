// Copyright (c) 2026, Dr. Volodymyr Cherkas
// SPDX-License-Identifier: BSD-3-Clause

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
#include <iphlpapi.h>
#endif

#include "EmergentCamera.h"

#include "CameraImageMetadata.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>

using Emergent::CEmergentFrame;

namespace
{
const char* const kPropertyIPAddress = "Camera IP Address";
const char* const kPropertyCameraSelection = "Camera";
const char* const kPropertyDiscoveredCameras = "Camera Info - Discovered Cameras";
const char* const kPropertyModel = "Camera Info - Model";
const char* const kPropertySerial = "Camera Info - Serial Number";
const char* const kPropertyFirmware = "Camera Info - Firmware";
const char* const kPropertyConnectionSpeed = "Camera Info - Connection Speed";
const char* const kPropertyConnectedIPAddress = "Camera Info - IP Address";
const char* const kPropertySensorTemperature = "Temperature - Sensor (C)";
const char* const kPropertyCameraTemperature = "Temperature - Camera (C)";
const char* const kPropertyPhyTemperature = "Temperature - Network PHY (C)";
const char* const kPropertyPtpOffset = "PTP - Offset (ns)";
const char* const kPropertyCameraTimestamp = "PTP - Camera Clock (ns)";

const char* const kPropertyExposure = "Exposure - Time (ms)";
const char* const kPropertyADC = "Readout - ADC Mode";
const char* const kPropertyPixelType = "Readout - Pixel Type";
const char* const kPropertyFrameRateLimit =
   "Acquisition - Frame Rate Limit";
const char* const kPropertyFrameRate =
   "Acquisition - Frame Rate (fps)";
const char* const kFrameRateLimitOff = "Off";
const char* const kFrameRateLimitOn = "On";

const char* const kPropertyGain = "Gain - Analog (dB)";
const char* const kPropertyAutoGain = "Gain - Auto";

// Micro-Manager's main window and Core binning API require this exact keyword.
const char* const kPropertyBinning = MM::g_Keyword_Binning;
const char* const kPropertyRoiX = "ROI - Offset X";
const char* const kPropertyRoiY = "ROI - Offset Y";
const char* const kPropertyRoiWidth = "ROI - Width";
const char* const kPropertyRoiHeight = "ROI - Height";
const char* const kPropertyRoiCenterX = "ROI - Center X";
const char* const kPropertyRoiCenterY = "ROI - Center Y";
const char* const kRoiCenterIdle = "Idle";
const char* const kRoiCenterNow = "Center now";

const char* const kPropertyTriggerMode = "Trigger - Mode";
const char* const kPropertyTriggerInput = "Trigger - Start Input";
const char* const kPropertyTriggerInputEdge = "Trigger - Start Edge";
const char* const kPropertyTriggerEndMode = "Trigger - End Exposure Input";
const char* const kPropertyTriggerEndEdge = "Trigger - End Exposure Edge";
const char* const kPropertyTriggerDelay = "Trigger - Delay (us)";
const char* const kPropertyOutput0Source = "Trigger - GPIO 0 - Output Signal";
const char* const kPropertyOutput0Polarity = "Trigger - GPIO 0 - Output Polarity";
const char* const kPropertyOutput1Source = "Trigger - GPIO 1 - Output Signal";
const char* const kPropertyOutput1Polarity = "Trigger - GPIO 1 - Output Polarity";
const char* const kManualIpSelection = "Manual IP";
const char* const kTriggerInternal = "Internal";
const char* const kTriggerSoftware = "Software";
const char* const kTriggerExternal = "External";
const char* const kPolarityNormal = "Active high";
const char* const kPolarityInverted = "Active low";

struct UInt32FeatureDefinition
{
   const char* propertyName;
   const char* sdkName;
};

const UInt32FeatureDefinition kUInt32Features[] = {
   {"Acquisition - Frame Count", "AcquisitionFrameCount"},
   {"Gain - PGA", "PGAGain"},
   {"Gain - Offset", "Offset"},
   {"Gain - Auto Target", "AutoGainSet"},
   {"Gain - Auto I Gain", "AutoGainIGain"},
   {"Exposure - Auto Target", "AutoExpSet"},
   {"Exposure - Auto I Gain", "AutoExpIGain"},
   {"Exposure - Auto Maximum (us)", "AutoExpMaxExp"},
   {"Auto Control ROI - Width", "AutoSizeX"},
   {"Auto Control ROI - Height", "AutoSizeY"},
   {"Auto Control ROI - Offset X", "AutoOffsetX"},
   {"Auto Control ROI - Offset Y", "AutoOffsetY"},
   {"Image LUT - Index", "LUTIndex"},
   {"Image LUT - Value", "LUTValue"},
   {"Trigger - GPIO 2 - Input Debounce Count", "GPI_2_Debounce_Count"},
   {"Trigger - GPIO 3 - Input Debounce Count", "GPI_3_Debounce_Count"},
   {"Trigger - GPIO - Test Generator Frame Time (us)", "TG_Frame_Time"}
};

struct BoolFeatureDefinition
{
   const char* propertyName;
   const char* sdkName;
   const char* falseValue;
   const char* trueValue;
   bool restartStream;
};

const BoolFeatureDefinition kBoolFeatures[] = {
   {"Image Orientation - Reverse X", "ReverseX", "Off", "On", true},
   {"Image Orientation - Reverse Y", "ReverseY", "Off", "On", true},
   {"Exposure - Auto", "AutoExposure", "Off", "On", false},
   {"Acquisition - Fast Trigger", "FastTrigger", "Off", "On", false},
   {"Image LUT - Enabled", "LUTEnable", "Off", "On", false},
   {"Trigger - GPIO 2 - Input State", "GPI_2_Polarity", "Low", "High", false},
   {"Trigger - GPIO 3 - Input State", "GPI_3_Polarity", "Low", "High", false}
};

struct EnumFeatureDefinition
{
   const char* propertyName;
   const char* sdkName;
   bool restartStream;
};

const EnumFeatureDefinition kEnumFeatures[] = {
   {"Acquisition - Mode", "AcquisitionMode", false},
   {"PTP - Mode", "PtpMode", false},
   {"PTP - Status", "PtpStatus", false},
   {"Trigger - GPIO 2 - Input Mode", "GPI_2_Mode", false},
   {"Trigger - GPIO 3 - Input Mode", "GPI_3_Mode", false}
};
const unsigned kDiscoveryCapacity = 16;
const unsigned kFrameBufferCount = 8;
const int kFramePollTimeoutMs = 100;
const int kExtraAcquisitionTimeoutMs = 5000;
const int kExternalTriggerTimeoutMs = 60000;

std::string GetCameraString(Emergent::CEmergentCamera* camera, const char* name)
{
   char value[256] = {};
   unsigned long size = 0;
   if (EVT_CameraGetStringParam(camera, name, value, sizeof(value), &size) != EVT_SUCCESS)
      return std::string();
   value[sizeof(value) - 1] = '\0';
   return value;
}

// MM sorts numeric-leading choices by their numeric prefix in the browser.
std::string BitDepthDisplay(const std::string& native)
{
   const size_t digit = native.find_first_of("0123456789");
   if (digit == std::string::npos)
      return native;
   const size_t end = native.find_first_not_of("0123456789", digit);
   return native.substr(digit, end - digit) + "-bit (" + native + ")";
}

std::string BitDepthNative(const std::string& display)
{
   const size_t open = display.find('(');
   if (open == std::string::npos || display.back() != ')')
      return display;
   return display.substr(open + 1, display.size() - open - 2);
}

std::vector<std::string> BitDepthDisplays(const std::vector<std::string>& nativeValues)
{
   std::vector<std::string> values;
   for (const auto& native : nativeValues)
      values.push_back(BitDepthDisplay(native));
   return values;
}
}

EmergentCamera::EmergentCamera() :
   ipAddress_("192.168.207.207"),
   cameraSelection_(kManualIpSelection),
   pixelFormat_("Mono8"),
   triggerMode_(kTriggerInternal),
   initialized_(false),
   propertiesCreated_(false),
   cameraOpen_(false),
   streamOpen_(false),
   acquisitionRunning_(false),
   softwareTriggerPending_(false),
   softwareTriggerPerFrame_(true),
   hardwareBinningSupported_(false),
   frameRateSupported_(false),
   frameRateLimitEnabled_(false),
   rememberSettings_(true),
   restoringSettings_(false),
   settingsFileHealthy_(true),
   settingsFile_(DefaultSettingsFile()),
   settingsStatus_("No settings loaded"),
   sensorWidth_(0),
   sensorHeight_(0),
   width_(0),
   height_(0),
   offsetX_(0),
   offsetY_(0),
   widthMin_(1),
   heightMin_(1),
   widthInc_(1),
   heightInc_(1),
   offsetXInc_(1),
   offsetYInc_(1),
   bytesPerPixel_(1),
   bitDepth_(8),
   binning_(1),
   exposureMs_(1.0),
   gainDb_(0.0),
   exposureMinimumMs_(0.0),
   exposureMaximumMs_(0.0),
   requestedFrameRateFps_(1),
   frameRateFps_(1),
   frameRateMinimumFps_(1),
   frameRateMaximumFps_(1),
   sequenceBurstSize_(0),
   frameTimestampNs_(0),
   frameId_(0)
{
   SetErrorText(ERR_CAMERA_NOT_FOUND,
      "No Emergent camera was found at the configured IP address");
   SetErrorText(ERR_ESDK_CALL,
      "An Emergent eSDK call failed; see the Micro-Manager CoreLog for the numeric eSDK error");
   SetErrorText(ERR_UNSUPPORTED_PIXEL_FORMAT,
      "The camera does not expose a supported unpacked monochrome pixel format");
   SetErrorText(ERR_ACQUISITION_TIMEOUT, "Timed out waiting for an image from the camera");
   SetErrorText(ERR_INVALID_ROI, "The requested ROI is outside the camera sensor");
   SetErrorText(ERR_FRAME_FORMAT, "The received image does not match the configured image format");
   SetErrorText(ERR_ACQUISITION_STOPPED, "Image acquisition was stopped");
   SetErrorText(ERR_SETTINGS_FILE, "Cannot read or write Emergent.ini; see Settings - Status and CoreLog");

   CreateProperty(MM::g_Keyword_Description,
      "Emergent camera via eSDK; Dr. Volodymyr Cherkas; Advanced Optical Devices (www.ao-devices.com)",
      MM::String, true);
   CreateProperty("Camera Info - Developer", "Dr. Volodymyr Cherkas", MM::String, true);
   CreateProperty("Camera Info - Company", "Advanced Optical Devices", MM::String, true);
   CreateProperty("Camera Info - Developer Website", "www.ao-devices.com", MM::String, true);
   CreateProperty("Camera Info - Manufacturer", "Emergent", MM::String, true);
   CreateProperty("Camera Info - Manufacturer Website", "www.emergentvisiontec.com", MM::String, true);

   DiscoverCameras();
   CPropertyAction* cameraAction =
      new CPropertyAction(this, &EmergentCamera::OnCameraSelection);
   CreateProperty(kPropertyCameraSelection, cameraSelection_.c_str(), MM::String,
                  false, cameraAction, true);
   std::vector<std::string> cameraChoices;
   cameraChoices.push_back(kManualIpSelection);
   std::ostringstream discoverySummary;
   for (size_t i = 0; i < discoveredCameras_.size(); ++i)
   {
      cameraChoices.push_back(discoveredCameras_[i].first);
      if (i > 0)
         discoverySummary << "; ";
      discoverySummary << discoveredCameras_[i].first;
   }
   SetAllowedValues(kPropertyCameraSelection, cameraChoices);
   CreateProperty(kPropertyDiscoveredCameras,
      discoveredCameras_.empty() ? "None" : discoverySummary.str().c_str(),
      MM::String, true, nullptr, true);

   CPropertyAction* ipAction = new CPropertyAction(this, &EmergentCamera::OnIPAddress);
   CreateProperty(kPropertyIPAddress, ipAddress_.c_str(), MM::String, false, ipAction, true);
   CreateProperty("Settings - File", settingsFile_.c_str(), MM::String, false,
      new CPropertyAction(this, &EmergentCamera::OnSettingsFile), true);
   CreateProperty("Settings - Remember", "On", MM::String, false,
      new CPropertyAction(this, &EmergentCamera::OnRememberSettings), true);
   std::vector<std::string> rememberValues = {"Off", "On"};
   SetAllowedValues("Settings - Remember", rememberValues);

   sequenceThread_.reset(new EmergentSequenceThread(this));
}

EmergentCamera::~EmergentCamera()
{
   Shutdown();
}

void EmergentCamera::GetName(char* name) const
{
   CDeviceUtils::CopyLimitedString(name, g_EmergentCameraName);
}

int EmergentCamera::Initialize()
{
   if (initialized_)
      return DEVICE_OK;

   GigEVisionDeviceInfo found{};
   int ret = FindCameraByIp(found);
   if (ret != DEVICE_OK)
      return ret;

   {
      MMThreadGuard guard(cameraLock_);
      deviceInfo_ = found;

      EVT_ERROR error = EVT_CameraOpen(&camera_, &deviceInfo_);
      if (error != EVT_SUCCESS)
         return SdkError("EVT_CameraOpen", error);
      cameraOpen_ = true;

      ret = ConfigureCameraLocked();
      if (ret != DEVICE_OK)
      {
         CloseStreamLocked();
         EVT_CameraClose(&camera_);
         cameraOpen_ = false;
         return ret;
      }

      ret = CreateRuntimeProperties();
      if (ret != DEVICE_OK)
      {
         CloseStreamLocked();
         EVT_CameraClose(&camera_);
         cameraOpen_ = false;
         return ret;
      }

      // Restore only after every runtime property exists. Notifications are
      // suppressed until initialization has finished.
      if (rememberSettings_)
         RestoreSettingsLocked();
      else
         settingsStatus_ = "Disabled";
      initialized_ = true;
   }

   ret = UpdateStatus();
   if (ret != DEVICE_OK)
      return ret;

   std::ostringstream message;
   message << "Opened Emergent " << model_ << " serial " << serial_
           << " at " << ipAddress_ << " using " << pixelFormat_
           << " (" << width_ << " x " << height_ << ")";
   LogMessage(message.str().c_str());
   return DEVICE_OK;
}

int EmergentCamera::Shutdown()
{
   if (sequenceThread_)
   {
      sequenceThread_->Stop();
      sequenceThread_->wait();
   }

   MMThreadGuard guard(cameraLock_);
   if (acquisitionRunning_)
      StopAcquisitionLocked();
   if (initialized_ && cameraOpen_ && rememberSettings_ && settingsFileHealthy_)
      SaveSettingsLocked();
   CloseStreamLocked();
   if (cameraOpen_)
   {
      EVT_CameraClose(&camera_);
      cameraOpen_ = false;
   }
   initialized_ = false;
   return DEVICE_OK;
}

void EmergentCamera::DiscoverCameras()
{
   discoveredCameras_.clear();
   GigEVisionDeviceInfo devices[kDiscoveryCapacity] = {};
   unsigned capacity = kDiscoveryCapacity;
   unsigned count = 0;
   const EVT_ERROR error = Emergent::EVT_ListDevices(devices, &capacity, &count);
   if (error != EVT_SUCCESS)
      return;

   const unsigned available = std::min(count, kDiscoveryCapacity);
   for (unsigned i = 0; i < available; ++i)
   {
      std::ostringstream label;
      label << devices[i].modelName << " | SN " << devices[i].serialNumber
            << " | " << devices[i].currentIp;
      discoveredCameras_.push_back(std::make_pair(label.str(), devices[i].currentIp));
      if (ipAddress_ == devices[i].currentIp)
         cameraSelection_ = label.str();
   }
}

int EmergentCamera::FindCameraByIp(GigEVisionDeviceInfo& result)
{
   GigEVisionDeviceInfo devices[kDiscoveryCapacity] = {};
   unsigned capacity = kDiscoveryCapacity;
   unsigned count = 0;
   ListDevicesSettings settings;
   settings.timeout = 1500;
   settings.cameraIP = ipAddress_.c_str();

   EVT_ERROR error = Emergent::EVT_ListDevices(devices, &capacity, &count, &settings);
   if (error != EVT_SUCCESS || count == 0)
   {
      // Some NIC/camera combinations do not answer unicast discovery. Fall back
      // to the normal subnet discovery, but still require an exact IP match.
      capacity = kDiscoveryCapacity;
      count = 0;
      error = Emergent::EVT_ListDevices(devices, &capacity, &count);
   }
   if (error != EVT_SUCCESS)
      return SdkError("EVT_ListDevices", error);

   const unsigned available = std::min(count, kDiscoveryCapacity);
   for (unsigned i = 0; i < available; ++i)
   {
      if (ipAddress_ == devices[i].currentIp)
      {
         result = devices[i];
         return DEVICE_OK;
      }
   }

   std::ostringstream message;
   message << "No camera found at " << ipAddress_ << "; eSDK discovered "
           << count << " camera(s)";
   LogMessage(message.str().c_str(), false);
   return ERR_CAMERA_NOT_FOUND;
}

int EmergentCamera::ConfigureCameraLocked()
{
   EVT_ERROR error = EVT_CameraSetEnumParam(&camera_, "AcquisitionMode", "Continuous");
   if (error != EVT_SUCCESS)
      return SdkError("set AcquisitionMode=Continuous", error);

   error = EVT_CameraSetEnumParam(&camera_, "TriggerMode", "Off");
   if (error != EVT_SUCCESS)
      return SdkError("set TriggerMode=Off", error);
   triggerMode_ = kTriggerInternal;

   int ret = ReadCameraIdentityLocked();
   if (ret != DEVICE_OK)
      return ret;
   ret = ReadPixelFormatsLocked();
   if (ret != DEVICE_OK)
      return ret;
   ret = RefreshGeometryLocked();
   if (ret != DEVICE_OK)
      return ret;
   ret = ConfigureGpioDefaultsLocked();
   if (ret != DEVICE_OK)
      return ret;

   unsigned exposureUs = 0;
   error = EVT_CameraGetUInt32Param(&camera_, "Exposure", &exposureUs);
   if (error != EVT_SUCCESS)
      return SdkError("get Exposure", error);
   exposureMs_ = exposureUs / 1000.0;

   image_.Resize(width_, height_, bytesPerPixel_);
   return OpenStreamLocked();
}

int EmergentCamera::ReadCameraIdentityLocked()
{
   model_ = GetCameraString(&camera_, "DeviceModelName");
   serial_ = GetCameraString(&camera_, "DeviceSerialNumber");
   firmware_ = GetCameraString(&camera_, "DeviceFirmwareVersion");
   if (model_.empty())
      model_ = deviceInfo_.modelName;
   if (serial_.empty())
      serial_ = deviceInfo_.serialNumber;
   return DEVICE_OK;
}

int EmergentCamera::ConfigureGpioDefaultsLocked()
{
   // The Eros HE camera exposes fixed GPI2/GPI3 and GPO0/GPO1 directions.
   // Route the user's chosen default input and exposure-active output, while
   // leaving Trigger Mode internal until external triggering is selected.
   const std::pair<const char*, const char*> defaults[] = {
      {"GPI_2_Mode", "Exposure"},
      {"GPI_Start_Exp_Mode", "GPI_2"},
      {"GPI_Start_Exp_Event", "Rising_Edge"},
      {"GPI_End_Exp_Mode", "Internal"},
      {"GPO_1_Mode", "Exposure"}
   };
   for (const auto& setting : defaults)
   {
      EvtParamAttribute attribute{};
      if (EVT_CameraGetParamAttr(&camera_, setting.first, &attribute) != EVT_SUCCESS ||
          attribute.accessMode != ERW)
         continue;
      const EVT_ERROR error = EVT_CameraSetEnumParam(&camera_, setting.first, setting.second);
      if (error != EVT_SUCCESS)
         return SdkError(setting.first, error);
   }
   EvtParamAttribute attribute{};
   if (EVT_CameraGetParamAttr(&camera_, "GPO_1_Polarity", &attribute) == EVT_SUCCESS &&
       attribute.accessMode == ERW)
   {
      const EVT_ERROR error = EVT_CameraSetBoolParam(&camera_, "GPO_1_Polarity", true);
      if (error != EVT_SUCCESS)
         return SdkError("default GPO1 active-high polarity", error);
   }
   return DEVICE_OK;
}

int EmergentCamera::ReadPixelFormatsLocked()
{
   unsigned long needed = 0;
   EVT_ERROR error = EVT_CameraGetEnumParamRange(&camera_, "PixelFormat", nullptr, 0, &needed);
   if (error != EVT_SUCCESS && error != EVT_ERROR_FBIG)
      return SdkError("get PixelFormat range size", error);
   if (needed == 0)
      needed = 2048;

   std::vector<char> values(needed + 1, 0);
   error = EVT_CameraGetEnumParamRange(&camera_, "PixelFormat", values.data(),
                                      static_cast<unsigned long>(values.size()), &needed);
   if (error != EVT_SUCCESS)
      return SdkError("get PixelFormat range", error);

   pixelFormats_.clear();
   const std::vector<std::string> allFormats = SplitEnumValues(values.data());
   for (const std::string& format : allFormats)
   {
      if (IsSupportedMonoFormat(format))
         pixelFormats_.push_back(format);
   }
   if (pixelFormats_.empty())
      return ERR_UNSUPPORTED_PIXEL_FORMAT;

   char current[128] = {};
   unsigned long currentSize = 0;
   error = EVT_CameraGetEnumParam(&camera_, "PixelFormat", current, sizeof(current), &currentSize);
   if (error != EVT_SUCCESS)
      return SdkError("get PixelFormat", error);

   const std::string currentFormat(current);
   if (IsSupportedMonoFormat(currentFormat))
      pixelFormat_ = currentFormat;
   else
   {
      const auto mono8 = std::find(pixelFormats_.begin(), pixelFormats_.end(), "Mono8");
      pixelFormat_ = mono8 != pixelFormats_.end() ? *mono8 : pixelFormats_.front();
      error = EVT_CameraSetEnumParam(&camera_, "PixelFormat", pixelFormat_.c_str());
      if (error != EVT_SUCCESS)
         return SdkError("set PixelFormat", error);
   }

   bytesPerPixel_ = PixelBytes(pixelFormat_);
   bitDepth_ = PixelBits(pixelFormat_);
   return DEVICE_OK;
}

int EmergentCamera::RefreshGeometryLocked()
{
   EVT_ERROR error = EVT_SUCCESS;
   if (sensorWidth_ == 0 || sensorHeight_ == 0)
   {
      unsigned currentWidth = 0;
      unsigned currentHeight = 0;
      unsigned maximumOffsetX = 0;
      unsigned maximumOffsetY = 0;
      error = EVT_CameraGetUInt32Param(&camera_, "Width", &currentWidth);
      if (error != EVT_SUCCESS)
         return SdkError("get Width for sensor extent", error);
      error = EVT_CameraGetUInt32Param(&camera_, "Height", &currentHeight);
      if (error != EVT_SUCCESS)
         return SdkError("get Height for sensor extent", error);
      error = EVT_CameraGetUInt32ParamMax(&camera_, "OffsetX", &maximumOffsetX);
      if (error != EVT_SUCCESS)
         return SdkError("get OffsetX max for sensor extent", error);
      error = EVT_CameraGetUInt32ParamMax(&camera_, "OffsetY", &maximumOffsetY);
      if (error != EVT_SUCCESS)
         return SdkError("get OffsetY max for sensor extent", error);
      sensorWidth_ = currentWidth + maximumOffsetX;
      sensorHeight_ = currentHeight + maximumOffsetY;
   }

   EVT_CameraGetUInt32ParamMin(&camera_, "Width", &widthMin_);
   EVT_CameraGetUInt32ParamMin(&camera_, "Height", &heightMin_);
   EVT_CameraGetUInt32ParamInc(&camera_, "Width", &widthInc_);
   EVT_CameraGetUInt32ParamInc(&camera_, "Height", &heightInc_);
   EVT_CameraGetUInt32ParamInc(&camera_, "OffsetX", &offsetXInc_);
   EVT_CameraGetUInt32ParamInc(&camera_, "OffsetY", &offsetYInc_);
   widthInc_ = std::max(1u, widthInc_);
   heightInc_ = std::max(1u, heightInc_);
   offsetXInc_ = std::max(1u, offsetXInc_);
   offsetYInc_ = std::max(1u, offsetYInc_);

   error = EVT_CameraGetUInt32Param(&camera_, "Width", &width_);
   if (error != EVT_SUCCESS)
      return SdkError("get Width", error);
   error = EVT_CameraGetUInt32Param(&camera_, "Height", &height_);
   if (error != EVT_SUCCESS)
      return SdkError("get Height", error);
   error = EVT_CameraGetUInt32Param(&camera_, "OffsetX", &offsetX_);
   if (error != EVT_SUCCESS)
      return SdkError("get OffsetX", error);
   error = EVT_CameraGetUInt32Param(&camera_, "OffsetY", &offsetY_);
   if (error != EVT_SUCCESS)
      return SdkError("get OffsetY", error);
   RefreshGeometryLimitsLocked();
   return DEVICE_OK;
}

void EmergentCamera::RefreshGeometryLimitsLocked()
{
   if (HasProperty(kPropertyRoiX))
      SetPropertyLimits(kPropertyRoiX, 0, sensorWidth_ - widthMin_);
   if (HasProperty(kPropertyRoiY))
      SetPropertyLimits(kPropertyRoiY, 0, sensorHeight_ - heightMin_);
   if (HasProperty(kPropertyRoiWidth))
      SetPropertyLimits(kPropertyRoiWidth, widthMin_, sensorWidth_);
   if (HasProperty(kPropertyRoiHeight))
      SetPropertyLimits(kPropertyRoiHeight, heightMin_, sensorHeight_);
   for (const auto& feature : kUInt32Features)
   {
      if (!HasProperty(feature.propertyName))
         continue;
      unsigned minimum = 0, maximum = 0;
      if (EVT_CameraGetUInt32ParamMin(&camera_, feature.sdkName, &minimum) == EVT_SUCCESS &&
          EVT_CameraGetUInt32ParamMax(&camera_, feature.sdkName, &maximum) == EVT_SUCCESS &&
          maximum >= minimum)
      {
         maximum = UsableUInt32Maximum(feature.sdkName, maximum);
         if (maximum > minimum)
            SetPropertyLimits(feature.propertyName, minimum, maximum);
         else
            ClearPropertyLimits(feature.propertyName);
      }
   }
}

unsigned EmergentCamera::UsableUInt32Maximum(const char* sdkName, unsigned maximum) const
{
   // HE-25000 firmware 3.35 advertises 1000 and reads it back, but a 1000-frame
   // burst produces 232 frames (1000 modulo 256). Keep its hardware bursts
   // within the working 8-bit range; MM sequences can span multiple bursts.
   if (std::strcmp(sdkName, "AcquisitionFrameCount") == 0 &&
       model_ == "HE-25000SBL-T-M" && firmware_ == "3.35")
      return std::min(maximum, 255u);
   return maximum;
}

void EmergentCamera::PublishGeometryState()
{
   if (!initialized_ || restoringSettings_)
      return;
   OnPropertyChanged(kPropertyBinning, CDeviceUtils::ConvertToString(binning_));
   OnPropertyChanged(kPropertyRoiX, CDeviceUtils::ConvertToString(static_cast<long>(offsetX_)));
   OnPropertyChanged(kPropertyRoiY, CDeviceUtils::ConvertToString(static_cast<long>(offsetY_)));
   OnPropertyChanged(kPropertyRoiWidth, CDeviceUtils::ConvertToString(static_cast<long>(width_)));
   OnPropertyChanged(kPropertyRoiHeight, CDeviceUtils::ConvertToString(static_cast<long>(height_)));
   PublishFrameRateState();
   OnPropertiesChanged();
}

std::vector<std::string> EmergentCamera::ReadEnumValuesLocked(const char* name)
{
   unsigned long needed = 0;
   EVT_ERROR error = EVT_CameraGetEnumParamRange(&camera_, name, nullptr, 0, &needed);
   if (error != EVT_SUCCESS && error != EVT_ERROR_FBIG)
      return std::vector<std::string>();
   if (needed == 0)
      needed = 2048;
   std::vector<char> values(needed + 1, 0);
   error = EVT_CameraGetEnumParamRange(&camera_, name, values.data(),
                                      static_cast<unsigned long>(values.size()), &needed);
   return error == EVT_SUCCESS ? SplitEnumValues(values.data()) :
                                 std::vector<std::string>();
}

int EmergentCamera::CreateRuntimeProperties()
{
   if (propertiesCreated_)
      return DEVICE_OK;

   int ret = CreateProperty(kPropertyModel, model_.c_str(), MM::String, true);
   if (ret != DEVICE_OK)
      return ret;
   ret = CreateProperty(kPropertySerial, serial_.c_str(), MM::String, true);
   if (ret != DEVICE_OK)
      return ret;
   ret = CreateProperty(kPropertyFirmware, firmware_.c_str(), MM::String, true);
   if (ret != DEVICE_OK)
      return ret;
   ret = CreateProperty(kPropertyConnectedIPAddress, deviceInfo_.currentIp, MM::String, true);
   if (ret != DEVICE_OK)
      return ret;
   ret = CreateProperty("Camera Info - Network Interface", deviceInfo_.nic.friendlyName, MM::String, true);
   if (ret != DEVICE_OK)
      return ret;
   ret = CreateProperty(kPropertyConnectionSpeed, "Unknown", MM::String, true,
      new CPropertyAction(this, &EmergentCamera::OnConnectionSpeed));
   if (ret != DEVICE_OK)
      return ret;

   CPropertyAction* pixelAction = new CPropertyAction(this, &EmergentCamera::OnPixelType);
   ret = CreateProperty(kPropertyPixelType, BitDepthDisplay(pixelFormat_).c_str(), MM::String,
                        false, pixelAction);
   if (ret != DEVICE_OK)
      return ret;
   auto pixelChoices = BitDepthDisplays(pixelFormats_);
   ret = SetAllowedValues(kPropertyPixelType, pixelChoices);
   if (ret != DEVICE_OK)
      return ret;

   CPropertyAction* exposureAction = new CPropertyAction(this, &EmergentCamera::OnExposure);
   ret = CreateProperty(kPropertyExposure,
                        CDeviceUtils::ConvertToString(exposureMs_),
                        MM::Float, false, exposureAction);
   if (ret != DEVICE_OK)
      return ret;

   unsigned minUs = 0;
   unsigned maxUs = 0;
   if (EVT_CameraGetUInt32ParamMin(&camera_, "Exposure", &minUs) == EVT_SUCCESS &&
       EVT_CameraGetUInt32ParamMax(&camera_, "Exposure", &maxUs) == EVT_SUCCESS)
   {
      exposureMinimumMs_ = minUs / 1000.0;
      exposureMaximumMs_ = maxUs / 1000.0;
      SetPropertyLimits(kPropertyExposure,
                        exposureMinimumMs_, exposureMaximumMs_);
   }

   EvtParamAttribute attribute{};
   std::vector<std::string> allowedValues;
   if (EVT_CameraGetParamAttr(&camera_, "FrameRate", &attribute) == EVT_SUCCESS &&
       attribute.dataType == EDataTypeUInt32 && attribute.accessMode == ERW)
   {
      frameRateSupported_ = true;
      unsigned currentFrameRate = 1;
      EVT_CameraGetUInt32Param(&camera_, "FrameRate", &currentFrameRate);
      requestedFrameRateFps_ = std::max(1u, currentFrameRate);
      int frameRateRet = ApplyFrameRatePolicyLocked(false);
      if (frameRateRet != DEVICE_OK)
         return frameRateRet;

      CPropertyAction* frameRateLimitAction =
         new CPropertyAction(this, &EmergentCamera::OnFrameRateLimit);
      ret = CreateProperty(kPropertyFrameRateLimit, kFrameRateLimitOff,
                           MM::String, false, frameRateLimitAction);
      if (ret != DEVICE_OK)
         return ret;
      allowedValues = {kFrameRateLimitOff, kFrameRateLimitOn};
      SetAllowedValues(kPropertyFrameRateLimit, allowedValues);

      CPropertyAction* frameRateAction =
         new CPropertyAction(this, &EmergentCamera::OnFrameRate);
      ret = CreateProperty(kPropertyFrameRate,
         CDeviceUtils::ConvertToString(static_cast<long>(frameRateFps_)),
         MM::Integer, false, frameRateAction);
      if (ret != DEVICE_OK)
         return ret;
      SetPropertyLimits(kPropertyFrameRate,
                        frameRateMinimumFps_, frameRateMaximumFps_);
   }

   if (EVT_CameraGetParamAttr(&camera_, "Gain", &attribute) == EVT_SUCCESS &&
       attribute.accessMode != ENI && attribute.accessMode != ENA)
   {
      float gain = 0.0f;
      EVT_CameraGetFloatParam(&camera_, "Gain", &gain);
      gainDb_ = gain;
      CPropertyAction* gainAction = new CPropertyAction(this, &EmergentCamera::OnGain);
      ret = CreateProperty(kPropertyGain,
                           CDeviceUtils::ConvertToString(gainDb_), MM::Float,
                           attribute.accessMode != ERW, gainAction);
      if (ret != DEVICE_OK)
         return ret;
      float gainMin = 0.0f;
      float gainMax = 0.0f;
      if (EVT_CameraGetFloatParamMin(&camera_, "Gain", &gainMin) == EVT_SUCCESS &&
          EVT_CameraGetFloatParamMax(&camera_, "Gain", &gainMax) == EVT_SUCCESS)
         SetPropertyLimits(kPropertyGain, gainMin, gainMax);
   }

   if (EVT_CameraGetParamAttr(&camera_, "AutoGain", &attribute) == EVT_SUCCESS &&
       attribute.accessMode != ENI && attribute.accessMode != ENA)
   {
      CPropertyAction* autoGainAction =
         new CPropertyAction(this, &EmergentCamera::OnAutoGain);
      ret = CreateProperty(kPropertyAutoGain, "Off", MM::String,
                           attribute.accessMode != ERW, autoGainAction);
      if (ret != DEVICE_OK)
         return ret;
      allowedValues = {"Off", "On"};
      SetAllowedValues(kPropertyAutoGain, allowedValues);
   }

   hardwareBinningSupported_ =
      EVT_CameraGetParamAttr(&camera_, "Bin", &attribute) == EVT_SUCCESS &&
      attribute.dataType == EDataTypeEnumeration &&
      attribute.accessMode == ERW;
   allowedValues = {"1"};
   if (hardwareBinningSupported_)
   {
      const auto modes = ReadEnumValuesLocked("Bin");
      if (std::find(modes.begin(), modes.end(), "B2x2") != modes.end())
         allowedValues.push_back("2");
   }
   {
      CPropertyAction* binAction = new CPropertyAction(this, &EmergentCamera::OnBinning);
      ret = CreateProperty(kPropertyBinning, "1", MM::Integer, !hardwareBinningSupported_, binAction);
      if (ret != DEVICE_OK)
         return ret;
      ret = SetAllowedValues(kPropertyBinning, allowedValues);
      if (ret != DEVICE_OK)
         return ret;
   }

   if (EVT_CameraGetParamAttr(&camera_, "ADC", &attribute) == EVT_SUCCESS &&
       attribute.accessMode != ENI && attribute.accessMode != ENA)
   {
      char adc[64] = {};
      unsigned long adcSize = 0;
      EVT_CameraGetEnumParam(&camera_, "ADC", adc, sizeof(adc), &adcSize);
      CPropertyAction* adcAction = new CPropertyAction(this, &EmergentCamera::OnADC);
      ret = CreateProperty(kPropertyADC, BitDepthDisplay(adc).c_str(), MM::String,
                           attribute.accessMode != ERW, adcAction);
      if (ret != DEVICE_OK)
         return ret;
      allowedValues = BitDepthDisplays(ReadEnumValuesLocked("ADC"));
      SetAllowedValues(kPropertyADC, allowedValues);
   }

   CPropertyAction* roiXAction = new CPropertyAction(this, &EmergentCamera::OnRoiX);
   ret = CreateProperty(kPropertyRoiX,
                        CDeviceUtils::ConvertToString(static_cast<long>(offsetX_)),
                        MM::Integer, false, roiXAction);
   if (ret != DEVICE_OK)
      return ret;
   SetPropertyLimits(kPropertyRoiX, 0, sensorWidth_ - 1);

   CPropertyAction* roiYAction = new CPropertyAction(this, &EmergentCamera::OnRoiY);
   ret = CreateProperty(kPropertyRoiY,
                        CDeviceUtils::ConvertToString(static_cast<long>(offsetY_)),
                        MM::Integer, false, roiYAction);
   if (ret != DEVICE_OK)
      return ret;
   SetPropertyLimits(kPropertyRoiY, 0, sensorHeight_ - 1);

   CPropertyAction* roiWidthAction =
      new CPropertyAction(this, &EmergentCamera::OnRoiWidth);
   ret = CreateProperty(kPropertyRoiWidth,
                        CDeviceUtils::ConvertToString(static_cast<long>(width_)),
                        MM::Integer, false, roiWidthAction);
   if (ret != DEVICE_OK)
      return ret;
   SetPropertyLimits(kPropertyRoiWidth, widthMin_, sensorWidth_);

   CPropertyAction* roiHeightAction =
      new CPropertyAction(this, &EmergentCamera::OnRoiHeight);
   ret = CreateProperty(kPropertyRoiHeight,
                        CDeviceUtils::ConvertToString(static_cast<long>(height_)),
                        MM::Integer, false, roiHeightAction);
   if (ret != DEVICE_OK)
      return ret;
   SetPropertyLimits(kPropertyRoiHeight, heightMin_, sensorHeight_);

   CPropertyAction* roiCenterXAction =
      new CPropertyAction(this, &EmergentCamera::OnRoiCenterX);
   ret = CreateProperty(kPropertyRoiCenterX, kRoiCenterIdle,
                        MM::String, false, roiCenterXAction);
   if (ret != DEVICE_OK)
      return ret;
   allowedValues = {kRoiCenterIdle, kRoiCenterNow};
   SetAllowedValues(kPropertyRoiCenterX, allowedValues);

   CPropertyAction* roiCenterYAction =
      new CPropertyAction(this, &EmergentCamera::OnRoiCenterY);
   ret = CreateProperty(kPropertyRoiCenterY, kRoiCenterIdle,
                        MM::String, false, roiCenterYAction);
   if (ret != DEVICE_OK)
      return ret;
   allowedValues = {kRoiCenterIdle, kRoiCenterNow};
   SetAllowedValues(kPropertyRoiCenterY, allowedValues);

   CPropertyAction* triggerModeAction =
      new CPropertyAction(this, &EmergentCamera::OnTriggerMode);
   ret = CreateProperty(kPropertyTriggerMode, triggerMode_.c_str(), MM::String,
                        false, triggerModeAction);
   if (ret != DEVICE_OK)
      return ret;
   allowedValues = {kTriggerInternal, kTriggerSoftware, kTriggerExternal};
   SetAllowedValues(kPropertyTriggerMode, allowedValues);

   CPropertyAction* triggerInputAction =
      new CPropertyAction(this, &EmergentCamera::OnTriggerInput);
   ret = CreateProperty(kPropertyTriggerInput, "GPI_2", MM::String,
                        false, triggerInputAction);
   if (ret != DEVICE_OK)
      return ret;
   allowedValues = ReadEnumValuesLocked("GPI_Start_Exp_Mode");
   SetAllowedValues(kPropertyTriggerInput, allowedValues);

   CPropertyAction* triggerInputEdgeAction =
      new CPropertyAction(this, &EmergentCamera::OnTriggerInputEdge);
   ret = CreateProperty(kPropertyTriggerInputEdge, "Rising_Edge", MM::String,
                        false, triggerInputEdgeAction);
   if (ret != DEVICE_OK)
      return ret;
   allowedValues = ReadEnumValuesLocked("GPI_Start_Exp_Event");
   SetAllowedValues(kPropertyTriggerInputEdge, allowedValues);

   CPropertyAction* triggerEndModeAction =
      new CPropertyAction(this, &EmergentCamera::OnTriggerEndMode);
   ret = CreateProperty(kPropertyTriggerEndMode, "Internal", MM::String,
                        false, triggerEndModeAction);
   if (ret != DEVICE_OK)
      return ret;
   allowedValues = ReadEnumValuesLocked("GPI_End_Exp_Mode");
   SetAllowedValues(kPropertyTriggerEndMode, allowedValues);

   CPropertyAction* triggerEndEdgeAction =
      new CPropertyAction(this, &EmergentCamera::OnTriggerEndEdge);
   ret = CreateProperty(kPropertyTriggerEndEdge, "Falling_Edge", MM::String,
                        false, triggerEndEdgeAction);
   if (ret != DEVICE_OK)
      return ret;
   allowedValues = ReadEnumValuesLocked("GPI_End_Exp_Event");
   SetAllowedValues(kPropertyTriggerEndEdge, allowedValues);

   unsigned triggerDelay = 0;
   EVT_CameraGetUInt32Param(&camera_, "Trigger_Delay", &triggerDelay);
   CPropertyAction* triggerDelayAction =
      new CPropertyAction(this, &EmergentCamera::OnTriggerDelay);
   ret = CreateProperty(kPropertyTriggerDelay,
                        CDeviceUtils::ConvertToString(static_cast<long>(triggerDelay)), MM::Integer,
                        false, triggerDelayAction);
   if (ret != DEVICE_OK)
      return ret;
   unsigned triggerDelayMin = 0;
   unsigned triggerDelayMax = 0;
   if (EVT_CameraGetUInt32ParamMin(&camera_, "Trigger_Delay", &triggerDelayMin) == EVT_SUCCESS &&
       EVT_CameraGetUInt32ParamMax(&camera_, "Trigger_Delay", &triggerDelayMax) == EVT_SUCCESS)
      SetPropertyLimits(kPropertyTriggerDelay, triggerDelayMin, triggerDelayMax);

   CPropertyAction* output0Action =
      new CPropertyAction(this, &EmergentCamera::OnOutput0Source);
   ret = CreateProperty(kPropertyOutput0Source, "Exposure", MM::String,
                        false, output0Action);
   if (ret != DEVICE_OK)
      return ret;
   allowedValues = ReadEnumValuesLocked("GPO_0_Mode");
   SetAllowedValues(kPropertyOutput0Source, allowedValues);

   CPropertyAction* output0PolarityAction =
      new CPropertyAction(this, &EmergentCamera::OnOutput0Polarity);
   ret = CreateProperty(kPropertyOutput0Polarity, kPolarityNormal, MM::String,
                        false, output0PolarityAction);
   if (ret != DEVICE_OK)
      return ret;
   allowedValues = {kPolarityNormal, kPolarityInverted};
   SetAllowedValues(kPropertyOutput0Polarity, allowedValues);

   CPropertyAction* output1Action =
      new CPropertyAction(this, &EmergentCamera::OnOutput1Source);
   ret = CreateProperty(kPropertyOutput1Source, "Exposure", MM::String,
                        false, output1Action);
   if (ret != DEVICE_OK)
      return ret;
   allowedValues = ReadEnumValuesLocked("GPO_1_Mode");
   SetAllowedValues(kPropertyOutput1Source, allowedValues);

   CPropertyAction* output1PolarityAction =
      new CPropertyAction(this, &EmergentCamera::OnOutput1Polarity);
   ret = CreateProperty(kPropertyOutput1Polarity, kPolarityNormal, MM::String,
                        false, output1PolarityAction);
   if (ret != DEVICE_OK)
      return ret;
   allowedValues = {kPolarityNormal, kPolarityInverted};
   SetAllowedValues(kPropertyOutput1Polarity, allowedValues);

   CPropertyAction* sensorTemperatureAction =
      new CPropertyAction(this, &EmergentCamera::OnSensorTemperature);
   ret = CreateProperty(kPropertySensorTemperature, "0", MM::Integer, true,
                        sensorTemperatureAction);
   if (ret != DEVICE_OK)
      return ret;

   CPropertyAction* cameraTemperatureAction =
      new CPropertyAction(this, &EmergentCamera::OnCameraTemperature);
   ret = CreateProperty(kPropertyCameraTemperature, "0", MM::Integer, true,
                        cameraTemperatureAction);
   if (ret != DEVICE_OK)
      return ret;

   CPropertyAction* phyTemperatureAction =
      new CPropertyAction(this, &EmergentCamera::OnPhyTemperature);
   ret = CreateProperty(kPropertyPhyTemperature, "0", MM::Integer, true,
                        phyTemperatureAction);
   if (ret != DEVICE_OK)
      return ret;

   ret = CreateAdditionalFeatureProperties();
   if (ret != DEVICE_OK)
      return ret;

   for (unsigned line = 0; line < 4; ++line)
   {
      const bool output = line < 2;
      const std::string node = std::string(output ? "GPO_" : "GPI_") +
         std::to_string(line) + "_Mode";
      if (EVT_CameraGetParamAttr(&camera_, node.c_str(), &attribute) != EVT_SUCCESS ||
          attribute.accessMode == ENI || attribute.accessMode == ENA)
         continue;
      const std::string name = "Trigger - GPIO " + std::to_string(line) + " - Direction";
      ret = CreateProperty(name.c_str(), output ? "Output (fixed)" : "Input (fixed)", MM::String, true);
      if (ret != DEVICE_OK)
         return ret;
   }

   if (EVT_CameraGetParamAttr(&camera_, "PtpOffset", &attribute) == EVT_SUCCESS &&
       attribute.dataType == EDataTypeInt32 && attribute.accessMode == ERO)
   {
      ret = CreateProperty(kPropertyPtpOffset, "0", MM::Integer, true,
         new CPropertyAction(this, &EmergentCamera::OnPtpOffset));
      if (ret != DEVICE_OK)
         return ret;
   }
   if (EVT_CameraGetParamAttr(&camera_, "GevTimestampControlLatch", &attribute) == EVT_SUCCESS &&
       attribute.dataType == EDataTypeCommand && attribute.accessMode != ENI &&
       attribute.accessMode != ENA)
   {
      // Decimal text preserves every bit of the 64-bit nanosecond clock.
      ret = CreateProperty(kPropertyCameraTimestamp, "0", MM::String, true,
         new CPropertyAction(this, &EmergentCamera::OnCameraTimestamp));
      if (ret != DEVICE_OK)
         return ret;
   }

   ret = CreateProperty("Settings - Active File", settingsFile_.c_str(), MM::String, true);
   if (ret != DEVICE_OK)
      return ret;
   ret = CreateProperty("Settings - Status", settingsStatus_.c_str(), MM::String, true,
      new CPropertyAction(this, &EmergentCamera::OnSettingsStatus));
   if (ret != DEVICE_OK)
      return ret;
   ret = CreateProperty("Settings - Save", "Idle", MM::String, false,
      new CPropertyAction(this, &EmergentCamera::OnSaveSettings));
   if (ret != DEVICE_OK)
      return ret;
   std::vector<std::string> saveValues = {"Idle", "Save now"};
   SetAllowedValues("Settings - Save", saveValues);
   propertiesCreated_ = true;
   return DEVICE_OK;
}

int EmergentCamera::CreateAdditionalFeatureProperties()
{
   EvtParamAttribute attribute{};
   for (size_t i = 0; i < sizeof(kUInt32Features) / sizeof(kUInt32Features[0]); ++i)
   {
      const UInt32FeatureDefinition& feature = kUInt32Features[i];
      if (EVT_CameraGetParamAttr(&camera_, feature.sdkName, &attribute) != EVT_SUCCESS ||
          attribute.dataType != EDataTypeUInt32 || attribute.accessMode == ENI ||
          attribute.accessMode == ENA)
         continue;

      unsigned value = 0;
      EVT_CameraGetUInt32Param(&camera_, feature.sdkName, &value);
      CPropertyActionEx* action = new CPropertyActionEx(
         this, &EmergentCamera::OnUInt32Feature, static_cast<long>(i));
      int ret = CreateProperty(feature.propertyName,
         CDeviceUtils::ConvertToString(static_cast<long>(value)), MM::Integer,
         attribute.accessMode != ERW, action);
      if (ret != DEVICE_OK)
         return ret;

      unsigned minimum = 0;
      unsigned maximum = 0;
      if (EVT_CameraGetUInt32ParamMin(&camera_, feature.sdkName, &minimum) == EVT_SUCCESS &&
          EVT_CameraGetUInt32ParamMax(&camera_, feature.sdkName, &maximum) == EVT_SUCCESS &&
          maximum > minimum)
         SetPropertyLimits(feature.propertyName, minimum, UsableUInt32Maximum(feature.sdkName, maximum));
   }

   for (size_t i = 0; i < sizeof(kBoolFeatures) / sizeof(kBoolFeatures[0]); ++i)
   {
      const BoolFeatureDefinition& feature = kBoolFeatures[i];
      if (EVT_CameraGetParamAttr(&camera_, feature.sdkName, &attribute) != EVT_SUCCESS ||
          attribute.dataType != EDataTypeBoolean || attribute.accessMode == ENI ||
          attribute.accessMode == ENA)
         continue;

      bool value = false;
      EVT_CameraGetBoolParam(&camera_, feature.sdkName, &value);
      CPropertyActionEx* action = new CPropertyActionEx(
         this, &EmergentCamera::OnBoolFeature, static_cast<long>(i));
      int ret = CreateProperty(feature.propertyName,
         value ? feature.trueValue : feature.falseValue, MM::String,
         attribute.accessMode != ERW, action);
      if (ret != DEVICE_OK)
         return ret;
      if (attribute.accessMode == ERW)
      {
         std::vector<std::string> allowedValues =
            {feature.falseValue, feature.trueValue};
         SetAllowedValues(feature.propertyName, allowedValues);
      }
   }

   for (size_t i = 0; i < sizeof(kEnumFeatures) / sizeof(kEnumFeatures[0]); ++i)
   {
      const EnumFeatureDefinition& feature = kEnumFeatures[i];
      if (EVT_CameraGetParamAttr(&camera_, feature.sdkName, &attribute) != EVT_SUCCESS ||
          attribute.dataType != EDataTypeEnumeration || attribute.accessMode == ENI ||
          attribute.accessMode == ENA)
         continue;

      char value[128] = {};
      unsigned long size = 0;
      EVT_CameraGetEnumParam(&camera_, feature.sdkName, value, sizeof(value), &size);
      CPropertyActionEx* action = new CPropertyActionEx(
         this, &EmergentCamera::OnEnumFeature, static_cast<long>(i));
      int ret = CreateProperty(feature.propertyName, value, MM::String,
                               attribute.accessMode != ERW, action);
      if (ret != DEVICE_OK)
         return ret;
      if (attribute.accessMode == ERW)
      {
         std::vector<std::string> allowedValues =
            ReadEnumValuesLocked(feature.sdkName);
         SetAllowedValues(feature.propertyName, allowedValues);
      }
   }
   return DEVICE_OK;
}

int EmergentCamera::OpenStreamLocked()
{
   if (streamOpen_)
      return DEVICE_OK;

   EVT_ERROR error = EVT_CameraOpenStream(&camera_);
   if (error != EVT_SUCCESS)
      return SdkError("EVT_CameraOpenStream", error);
   streamOpen_ = true;

   for (unsigned i = 0; i < kFrameBufferCount; ++i)
   {
      std::unique_ptr<CEmergentFrame> frame(new CEmergentFrame());
      frame->size_x = width_;
      frame->size_y = height_;
      frame->pixel_type = PixelFormatCode(pixelFormat_);

      error = EVT_AllocateFrameBuffer(&camera_, frame.get(), EVT_FRAME_BUFFER_ZERO_COPY);
      if (error != EVT_SUCCESS)
      {
         SdkError("EVT_AllocateFrameBuffer", error);
         CloseStreamLocked();
         return ERR_ESDK_CALL;
      }
      error = EVT_CameraQueueFrame(&camera_, frame.get());
      if (error != EVT_SUCCESS)
      {
         EVT_ReleaseFrameBuffer(&camera_, frame.get());
         SdkError("EVT_CameraQueueFrame", error);
         CloseStreamLocked();
         return ERR_ESDK_CALL;
      }
      frameBuffers_.push_back(std::move(frame));
   }
   return DEVICE_OK;
}

void EmergentCamera::CloseStreamLocked()
{
   if (!streamOpen_)
      return;
   if (acquisitionRunning_)
      StopAcquisitionLocked();
   for (auto& frame : frameBuffers_)
      EVT_ReleaseFrameBuffer(&camera_, frame.get());
   frameBuffers_.clear();
   EVT_CameraCloseStream(&camera_);
   streamOpen_ = false;
}

int EmergentCamera::StartAcquisitionLocked()
{
   if (acquisitionRunning_)
      return DEVICE_OK;
   DrainCompletedFramesLocked();
   char mode[64] = {};
   unsigned long size = 0;
   EVT_ERROR modeError = EVT_CameraGetEnumParam(&camera_, "AcquisitionMode", mode, sizeof(mode), &size);
   if (modeError != EVT_SUCCESS)
      return SdkError("get acquisition trigger pacing", modeError);
   softwareTriggerPerFrame_ = std::strcmp(mode, "MultiFrame") != 0;
   EVT_ERROR error = EVT_CameraExecuteCommand(&camera_, "AcquisitionStart");
   if (error != EVT_SUCCESS)
      return SdkError("AcquisitionStart", error);
   acquisitionRunning_ = true;
   softwareTriggerPending_ = true;
   return DEVICE_OK;
}

void EmergentCamera::StopAcquisitionLocked()
{
   if (!acquisitionRunning_)
      return;
   const EVT_ERROR error = EVT_CameraExecuteCommand(&camera_, "AcquisitionStop");
   if (error != EVT_SUCCESS)
      SdkError("AcquisitionStop", error);
   acquisitionRunning_ = false;
   softwareTriggerPending_ = false;
   DrainCompletedFramesLocked();
}

void EmergentCamera::DrainCompletedFramesLocked()
{
   if (!streamOpen_)
      return;
   for (unsigned i = 0; i < kFrameBufferCount * 2; ++i)
   {
      CEmergentFrame received;
      const EVT_ERROR error = EVT_CameraGetFrame(&camera_, &received, 0);
      if (error != EVT_SUCCESS)
         break;
      EVT_CameraQueueFrame(&camera_, &received);
   }
}

int EmergentCamera::SnapImage()
{
   if (IsCapturing())
      return DEVICE_CAMERA_BUSY_ACQUIRING;

   MMThreadGuard guard(cameraLock_);
   // A snap must finish its entire MultiFrame burst before the next re-arm.
   char acquisitionMode[64] = {};
   unsigned long modeSize = 0;
   unsigned savedCount = 0;
   bool restoreCount = false;
   if (EVT_CameraGetEnumParam(&camera_, "AcquisitionMode", acquisitionMode,
                              sizeof(acquisitionMode), &modeSize) == EVT_SUCCESS &&
       std::strcmp(acquisitionMode, "MultiFrame") == 0)
   {
      EVT_ERROR error = EVT_CameraGetUInt32Param(&camera_, "AcquisitionFrameCount", &savedCount);
      if (error == EVT_SUCCESS)
         error = EVT_CameraSetUInt32Param(&camera_, "AcquisitionFrameCount", 1);
      if (error != EVT_SUCCESS)
         return SdkError("set snap AcquisitionFrameCount", error);
      restoreCount = true;
   }
   int ret = StartAcquisitionLocked();
   if (ret == DEVICE_OK)
      ret = FireSoftwareTriggerLocked();
   if (ret == DEVICE_OK)
      ret = AcquireOneFrameLocked(false);
   StopAcquisitionLocked();
   if (restoreCount)
   {
      const EVT_ERROR error = EVT_CameraSetUInt32Param(&camera_, "AcquisitionFrameCount", savedCount);
      if (ret == DEVICE_OK && error != EVT_SUCCESS)
         ret = SdkError("restore AcquisitionFrameCount after snap", error);
   }
   return ret;
}

int EmergentCamera::AcquireOneFrameLocked(bool interruptible)
{
   const double timeoutMs = triggerMode_ == kTriggerExternal ?
      kExternalTriggerTimeoutMs :
      std::max(1000.0, exposureMs_ + kExtraAcquisitionTimeoutMs);
   const auto deadline = std::chrono::steady_clock::now() +
      std::chrono::milliseconds(static_cast<long long>(std::ceil(timeoutMs)));

   for (;;)
   {
      if (interruptible && sequenceThread_->IsStopped())
         return ERR_ACQUISITION_STOPPED;

      CEmergentFrame received;
      const EVT_ERROR error = EVT_CameraGetFrame(&camera_, &received, kFramePollTimeoutMs);
      if (error == EVT_SUCCESS)
      {
         const int copyResult = CopyFrameToImageLocked(received);
         const EVT_ERROR queueError = EVT_CameraQueueFrame(&camera_, &received);
         if (copyResult != DEVICE_OK)
            return copyResult;
         if (queueError != EVT_SUCCESS)
            return SdkError("EVT_CameraQueueFrame", queueError);
         return DEVICE_OK;
      }
      if (error != EVT_ERROR_AGAIN && error != EVT_ERROR_TIMEDOUT)
         return SdkError("EVT_CameraGetFrame", error);
      if (std::chrono::steady_clock::now() >= deadline)
         return ERR_ACQUISITION_TIMEOUT;
   }
}

int EmergentCamera::CopyFrameToImageLocked(const CEmergentFrame& frame)
{
   if (frame.size_x != width_ || frame.size_y != height_ ||
       frame.pixel_type != PixelFormatCode(pixelFormat_) || frame.imagePtr == nullptr)
   {
      std::ostringstream message;
      message << "Unexpected frame: " << frame.size_x << " x " << frame.size_y
              << ", format " << frame.GetPixelFormatString();
      LogMessage(message.str().c_str(), false);
      return ERR_FRAME_FORMAT;
   }

   const size_t rowBytes = static_cast<size_t>(width_) * bytesPerPixel_;
   const size_t stride = frame.stride > 0 ? static_cast<size_t>(frame.stride) : rowBytes;
   if (stride < rowBytes || frame.bufferSize < stride * (height_ - 1) + rowBytes)
      return ERR_FRAME_FORMAT;

   MMThreadGuard imageGuard(imageLock_);
   frameTimestampNs_ = frame.timestamp;
   frameId_ = frame.frame_id;
   unsigned char* destination = image_.GetPixelsRW();
   for (unsigned row = 0; row < height_; ++row)
   {
      std::memcpy(destination + row * rowBytes,
                  frame.imagePtr + row * stride, rowBytes);
   }
   return DEVICE_OK;
}

const unsigned char* EmergentCamera::GetImageBuffer()
{
   MMThreadGuard guard(imageLock_);
   return image_.GetPixels();
}

long EmergentCamera::GetImageBufferSize() const
{
   return static_cast<long>(width_) * static_cast<long>(height_) * bytesPerPixel_;
}

double EmergentCamera::GetExposure() const
{
   // Auto exposure can change the hardware value without a property write.
   if (initialized_)
   {
      double actual = exposureMs_;
      if (const_cast<EmergentCamera*>(this)->GetProperty(kPropertyExposure, actual) == DEVICE_OK)
         return actual;
   }
   return exposureMs_;
}

void EmergentCamera::SetExposure(double exposureMs)
{
   if (!initialized_)
   {
      exposureMs_ = exposureMs;
      return;
   }
   SetProperty(kPropertyExposure,
               CDeviceUtils::ConvertToString(exposureMs));
}

int EmergentCamera::OnExposure(MM::PropertyBase* property, MM::ActionType action)
{
   if (action == MM::BeforeGet)
   {
      MMThreadGuard guard(cameraLock_);
      unsigned exposureUs = 0;
      const EVT_ERROR error = EVT_CameraGetUInt32Param(&camera_, "Exposure", &exposureUs);
      if (error != EVT_SUCCESS)
         return SdkError("get Exposure", error);
      exposureMs_ = exposureUs / 1000.0;
      property->Set(exposureMs_);
   }
   else if (action == MM::AfterSet)
   {
      if (IsCapturing())
         return DEVICE_CAMERA_BUSY_ACQUIRING;
      double requestedMs = 0.0;
      property->Get(requestedMs);
      const unsigned exposureUs = static_cast<unsigned>(std::llround(requestedMs * 1000.0));
      int ret = DEVICE_OK;
      {
         MMThreadGuard guard(cameraLock_);
         if (frameRateSupported_ && !frameRateLimitEnabled_)
         {
            ret = ApplyFrameRatePolicyLocked(false);
            if (ret != DEVICE_OK)
               return ret;
            const EVT_ERROR relaxError = EVT_CameraSetUInt32Param(
               &camera_, "FrameRate", frameRateMinimumFps_);
            if (relaxError != EVT_SUCCESS)
               return SdkError("relax FrameRate before Exposure", relaxError);
         }
         const EVT_ERROR error =
            EVT_CameraSetUInt32Param(&camera_, "Exposure", exposureUs);
         if (error != EVT_SUCCESS)
         {
            ApplyFrameRatePolicyLocked();
            return SdkError("set Exposure", error);
         }
         unsigned actualUs = exposureUs;
         const EVT_ERROR readError = EVT_CameraGetUInt32Param(&camera_, "Exposure", &actualUs);
         if (readError != EVT_SUCCESS)
            return SdkError("read back Exposure", readError);
         exposureMs_ = actualUs / 1000.0;
         property->Set(exposureMs_);
         ret = ApplyFrameRatePolicyLocked();
         if (ret == DEVICE_OK)
            ret = RefreshExposureLimitsLocked(!frameRateLimitEnabled_);
      }
      if (frameRateSupported_)
         PublishFrameRateState();
      if (ret == DEVICE_OK && initialized_ && !restoringSettings_)
         OnExposureChanged(exposureMs_);
      return ret;
   }
   return DEVICE_OK;
}

int EmergentCamera::ApplyFrameRatePolicyLocked(bool allowWrite)
{
   if (!frameRateSupported_)
      return DEVICE_OK;

   unsigned minimum = 0;
   unsigned maximum = 0;
   EVT_ERROR error = EVT_CameraGetUInt32ParamMin(&camera_, "FrameRate", &minimum);
   if (error != EVT_SUCCESS)
      return SdkError("get FrameRate minimum", error);
   error = EVT_CameraGetUInt32ParamMax(&camera_, "FrameRate", &maximum);
   if (error != EVT_SUCCESS)
      return SdkError("get FrameRate maximum", error);
   minimum = std::max(1u, minimum);
   maximum = std::max(minimum, maximum);
   frameRateMinimumFps_ = minimum;
   frameRateMaximumFps_ = maximum;

   if (allowWrite)
   {
      const unsigned target = frameRateLimitEnabled_ ?
         std::max(minimum, std::min(requestedFrameRateFps_, maximum)) : maximum;
      error = EVT_CameraSetUInt32Param(&camera_, "FrameRate", target);
      if (error != EVT_SUCCESS)
         return SdkError("set FrameRate policy", error);
   }

   unsigned actual = 0;
   error = EVT_CameraGetUInt32Param(&camera_, "FrameRate", &actual);
   if (error != EVT_SUCCESS)
      return SdkError("get FrameRate", error);
   frameRateFps_ = actual;
   return DEVICE_OK;
}

int EmergentCamera::RefreshExposureLimitsLocked(bool relaxFrameRate)
{
   int ret = DEVICE_OK;
   if (relaxFrameRate && frameRateSupported_)
   {
      ret = ApplyFrameRatePolicyLocked(false);
      if (ret != DEVICE_OK)
         return ret;
      const EVT_ERROR error = EVT_CameraSetUInt32Param(
         &camera_, "FrameRate", frameRateMinimumFps_);
      if (error != EVT_SUCCESS)
         return SdkError("relax FrameRate for Exposure range", error);
   }

   unsigned minimumUs = 0;
   unsigned maximumUs = 0;
   EVT_ERROR error = EVT_CameraGetUInt32ParamMin(
      &camera_, "Exposure", &minimumUs);
   if (error == EVT_SUCCESS)
      error = EVT_CameraGetUInt32ParamMax(&camera_, "Exposure", &maximumUs);

   const int restoreRet = relaxFrameRate ?
      ApplyFrameRatePolicyLocked() : DEVICE_OK;
   if (error != EVT_SUCCESS)
      return SdkError("get Exposure range", error);
   if (restoreRet != DEVICE_OK)
      return restoreRet;

   exposureMinimumMs_ = minimumUs / 1000.0;
   exposureMaximumMs_ = maximumUs / 1000.0;
   SetPropertyLimits(kPropertyExposure,
                     exposureMinimumMs_, exposureMaximumMs_);
   if (frameRateSupported_)
      SetPropertyLimits(kPropertyFrameRate,
                        frameRateMinimumFps_, frameRateMaximumFps_);
   return DEVICE_OK;
}

void EmergentCamera::PublishFrameRateState()
{
   if (!frameRateSupported_ || !initialized_ || restoringSettings_)
      return;
   SetPropertyLimits(kPropertyFrameRate,
                     frameRateMinimumFps_, frameRateMaximumFps_);
   SetPropertyLimits(kPropertyExposure,
                     exposureMinimumMs_, exposureMaximumMs_);
   OnPropertyChanged(kPropertyFrameRate,
      CDeviceUtils::ConvertToString(static_cast<long>(frameRateFps_)));
}

int EmergentCamera::OnFrameRateLimit(MM::PropertyBase* property,
                                     MM::ActionType action)
{
   if (action == MM::BeforeGet)
   {
      property->Set(frameRateLimitEnabled_ ?
                    kFrameRateLimitOn : kFrameRateLimitOff);
      return DEVICE_OK;
   }
   if (IsCapturing())
      return DEVICE_CAMERA_BUSY_ACQUIRING;

   std::string requested;
   property->Get(requested);
   if (requested != kFrameRateLimitOff && requested != kFrameRateLimitOn)
      return DEVICE_INVALID_INPUT_PARAM;

   int ret = DEVICE_OK;
   {
      MMThreadGuard guard(cameraLock_);
      frameRateLimitEnabled_ = requested == kFrameRateLimitOn;
      ret = ApplyFrameRatePolicyLocked();
      if (ret == DEVICE_OK)
         ret = RefreshExposureLimitsLocked(!frameRateLimitEnabled_);
   }
   property->Set(frameRateLimitEnabled_ ? kFrameRateLimitOn : kFrameRateLimitOff);
   PublishFrameRateState();
   return ret;
}

int EmergentCamera::OnFrameRate(MM::PropertyBase* property,
                                MM::ActionType action)
{
   if (action == MM::BeforeGet)
   {
      int ret = DEVICE_OK;
      {
         MMThreadGuard guard(cameraLock_);
         const bool allowWrite = !IsCapturing();
         ret = ApplyFrameRatePolicyLocked(allowWrite);
         if (ret == DEVICE_OK && allowWrite)
            ret = RefreshExposureLimitsLocked(!frameRateLimitEnabled_);
         property->Set(static_cast<long>(frameRateFps_));
      }
      SetPropertyLimits(kPropertyFrameRate,
                        frameRateMinimumFps_, frameRateMaximumFps_);
      return ret;
   }
   if (IsCapturing())
      return DEVICE_CAMERA_BUSY_ACQUIRING;

   long requested = 0;
   property->Get(requested);
   if (requested <= 0)
      return DEVICE_INVALID_INPUT_PARAM;

   int ret = DEVICE_OK;
   {
      MMThreadGuard guard(cameraLock_);
      requestedFrameRateFps_ = static_cast<unsigned>(requested);
      ret = ApplyFrameRatePolicyLocked();
      if (ret == DEVICE_OK)
         ret = RefreshExposureLimitsLocked(!frameRateLimitEnabled_);
      property->Set(static_cast<long>(frameRateFps_));
   }
   SetPropertyLimits(kPropertyFrameRate,
                     frameRateMinimumFps_, frameRateMaximumFps_);
   return ret;
}

int EmergentCamera::OnGain(MM::PropertyBase* property, MM::ActionType action)
{
   MMThreadGuard guard(cameraLock_);
   if (action == MM::BeforeGet)
   {
      float gain = 0.0f;
      const EVT_ERROR error = EVT_CameraGetFloatParam(&camera_, "Gain", &gain);
      if (error != EVT_SUCCESS)
         return SdkError("get Gain", error);
      gainDb_ = gain;
      property->Set(gainDb_);
   }
   else if (action == MM::AfterSet)
   {
      if (IsCapturing())
         return DEVICE_CAMERA_BUSY_ACQUIRING;
      double requested = 0.0;
      property->Get(requested);
      const EVT_ERROR error =
         EVT_CameraSetFloatParam(&camera_, "Gain", static_cast<float>(requested));
      if (error != EVT_SUCCESS)
         return SdkError("set Gain", error);
      float actual = 0.0f;
      if (EVT_CameraGetFloatParam(&camera_, "Gain", &actual) != EVT_SUCCESS)
         actual = static_cast<float>(requested);
      gainDb_ = actual;
      property->Set(gainDb_);
   }
   return DEVICE_OK;
}

int EmergentCamera::OnAutoGain(MM::PropertyBase* property, MM::ActionType action)
{
   MMThreadGuard guard(cameraLock_);
   if (action == MM::BeforeGet)
   {
      bool enabled = false;
      const EVT_ERROR error = EVT_CameraGetBoolParam(&camera_, "AutoGain", &enabled);
      if (error != EVT_SUCCESS)
         return SdkError("get AutoGain", error);
      property->Set(enabled ? "On" : "Off");
   }
   else if (action == MM::AfterSet)
   {
      if (IsCapturing())
         return DEVICE_CAMERA_BUSY_ACQUIRING;
      std::string value;
      property->Get(value);
      const EVT_ERROR error =
         EVT_CameraSetBoolParam(&camera_, "AutoGain", value == "On");
      if (error != EVT_SUCCESS)
         return SdkError("set AutoGain", error);
   }
   return DEVICE_OK;
}

int EmergentCamera::SetBinning(int binSize)
{
   if (binSize == binning_)
      return DEVICE_OK;
   if (!hardwareBinningSupported_ || (binSize != 1 && binSize != 2))
      return DEVICE_UNSUPPORTED_COMMAND;
   return SetProperty(kPropertyBinning,
                      CDeviceUtils::ConvertToString(binSize));
}

int EmergentCamera::OnBinning(MM::PropertyBase* property, MM::ActionType action)
{
   if (action == MM::BeforeGet)
   {
      MMThreadGuard guard(cameraLock_);
      if (!hardwareBinningSupported_)
      {
         property->Set(1L);
         return DEVICE_OK;
      }
      char value[64] = {};
      unsigned long size = 0;
      const EVT_ERROR error =
         EVT_CameraGetEnumParam(&camera_, "Bin", value, sizeof(value), &size);
      if (error != EVT_SUCCESS)
         return SdkError("get Bin", error);
      binning_ = std::strcmp(value, "B2x2") == 0 ? 2 : 1;
      property->Set(static_cast<long>(binning_));
      return DEVICE_OK;
   }
   if (IsCapturing())
      return DEVICE_CAMERA_BUSY_ACQUIRING;

   long requested = 1;
   property->Get(requested);
   if (requested != 1 && requested != 2)
      return DEVICE_INVALID_INPUT_PARAM;
   int ret = DEVICE_OK;
   {
      MMThreadGuard guard(cameraLock_);
      if (requested == binning_)
         return DEVICE_OK;
      CloseStreamLocked();
      const EVT_ERROR error = EVT_CameraSetEnumParam(
         &camera_, "Bin", requested == 2 ? "B2x2" : "B1x1");
      if (error != EVT_SUCCESS)
      {
         OpenStreamLocked();
         return SdkError("set Bin", error);
      }
      binning_ = static_cast<int>(requested);
      sensorWidth_ = 0;
      sensorHeight_ = 0;
      ret = ConfigureFullFrameLocked();
      if (ret == DEVICE_OK)
         ret = ApplyFrameRatePolicyLocked();
      if (ret == DEVICE_OK)
         ret = RefreshExposureLimitsLocked(!frameRateLimitEnabled_);
      image_.Resize(width_, height_, bytesPerPixel_);
      const int streamRet = OpenStreamLocked();
      if (ret == DEVICE_OK)
         ret = streamRet;
      property->Set(static_cast<long>(binning_));
   }
   if (ret == DEVICE_OK)
      PublishGeometryState();
   return ret;
}

int EmergentCamera::OnADC(MM::PropertyBase* property, MM::ActionType action)
{
   MMThreadGuard guard(cameraLock_);
   if (action == MM::BeforeGet)
   {
      char value[64] = {};
      unsigned long size = 0;
      const EVT_ERROR error =
         EVT_CameraGetEnumParam(&camera_, "ADC", value, sizeof(value), &size);
      if (error != EVT_SUCCESS)
         return SdkError("get ADC", error);
      property->Set(BitDepthDisplay(value).c_str());
      return DEVICE_OK;
   }
   if (IsCapturing())
      return DEVICE_CAMERA_BUSY_ACQUIRING;

   std::string requested;
   property->Get(requested);
   requested = BitDepthNative(requested);
   const unsigned savedOffsetX = offsetX_;
   const unsigned savedOffsetY = offsetY_;
   const unsigned savedWidth = width_;
   const unsigned savedHeight = height_;
   CloseStreamLocked();
   const EVT_ERROR error =
      EVT_CameraSetEnumParam(&camera_, "ADC", requested.c_str());
   if (error != EVT_SUCCESS)
   {
      OpenStreamLocked();
      return SdkError("set ADC", error);
   }

   int ret = ReadPixelFormatsLocked();
   if (ret == DEVICE_OK)
      ret = RefreshGeometryLocked();
   if (ret == DEVICE_OK)
      ret = ConfigureRoiLocked(savedOffsetX, savedOffsetY, savedWidth, savedHeight);
   if (ret == DEVICE_OK)
      ret = ApplyFrameRatePolicyLocked();
   if (ret == DEVICE_OK)
      ret = RefreshExposureLimitsLocked(!frameRateLimitEnabled_);
   image_.Resize(width_, height_, bytesPerPixel_);
   const int streamRet = OpenStreamLocked();
   if (ret == DEVICE_OK)
   {
      auto choices = BitDepthDisplays(pixelFormats_);
      SetAllowedValues(kPropertyPixelType, choices);
      property->Set(BitDepthDisplay(requested).c_str());
   }
   return ret != DEVICE_OK ? ret : streamRet;
}

int EmergentCamera::OnUInt32Feature(MM::PropertyBase* property,
                                    MM::ActionType action, long featureIndex)
{
   if (featureIndex < 0 || static_cast<size_t>(featureIndex) >=
         sizeof(kUInt32Features) / sizeof(kUInt32Features[0]))
      return DEVICE_INVALID_INPUT_PARAM;
   const UInt32FeatureDefinition& feature = kUInt32Features[featureIndex];
   MMThreadGuard guard(cameraLock_);
   if (action == MM::BeforeGet)
   {
      unsigned value = 0;
      const EVT_ERROR error =
         EVT_CameraGetUInt32Param(&camera_, feature.sdkName, &value);
      if (error != EVT_SUCCESS)
         return SdkError(feature.sdkName, error);
      property->Set(static_cast<long>(value));
      unsigned minimum = 0, maximum = 0;
      if (EVT_CameraGetUInt32ParamMin(&camera_, feature.sdkName, &minimum) == EVT_SUCCESS &&
          EVT_CameraGetUInt32ParamMax(&camera_, feature.sdkName, &maximum) == EVT_SUCCESS)
      {
         maximum = UsableUInt32Maximum(feature.sdkName, maximum);
         if (maximum > minimum)
            SetPropertyLimits(feature.propertyName, minimum, maximum);
         else
            ClearPropertyLimits(feature.propertyName);
      }
   }
   else if (action == MM::AfterSet)
   {
      if (IsCapturing())
         return DEVICE_CAMERA_BUSY_ACQUIRING;
      long value = 0;
      property->Get(value);
      if (value < 0)
         return DEVICE_INVALID_INPUT_PARAM;
      unsigned minimum = 0, maximum = 0, increment = 1;
      EVT_ERROR rangeError = EVT_CameraGetUInt32ParamMin(&camera_, feature.sdkName, &minimum);
      if (rangeError == EVT_SUCCESS)
         rangeError = EVT_CameraGetUInt32ParamMax(&camera_, feature.sdkName, &maximum);
      if (rangeError == EVT_SUCCESS)
         rangeError = EVT_CameraGetUInt32ParamInc(&camera_, feature.sdkName, &increment);
      if (rangeError != EVT_SUCCESS)
         return SdkError("read integer feature range/increment", rangeError);
      maximum = UsableUInt32Maximum(feature.sdkName, maximum);
      increment = std::max(1u, increment);
      unsigned aligned = std::max(minimum, std::min(static_cast<unsigned>(value), maximum));
      aligned = minimum + (aligned - minimum) / increment * increment;
      const EVT_ERROR error = EVT_CameraSetUInt32Param(
         &camera_, feature.sdkName, aligned);
      if (error != EVT_SUCCESS)
         return SdkError(feature.sdkName, error);
      unsigned actual = 0;
      if (EVT_CameraGetUInt32Param(&camera_, feature.sdkName, &actual) == EVT_SUCCESS)
         property->Set(static_cast<long>(actual));
      RefreshGeometryLimitsLocked();
   }
   return DEVICE_OK;
}

int EmergentCamera::OnBoolFeature(MM::PropertyBase* property,
                                  MM::ActionType action, long featureIndex)
{
   if (featureIndex < 0 || static_cast<size_t>(featureIndex) >=
         sizeof(kBoolFeatures) / sizeof(kBoolFeatures[0]))
      return DEVICE_INVALID_INPUT_PARAM;
   const BoolFeatureDefinition& feature = kBoolFeatures[featureIndex];
   MMThreadGuard guard(cameraLock_);
   if (action == MM::BeforeGet)
   {
      bool value = false;
      const EVT_ERROR error =
         EVT_CameraGetBoolParam(&camera_, feature.sdkName, &value);
      if (error != EVT_SUCCESS)
         return SdkError(feature.sdkName, error);
      property->Set(value ? feature.trueValue : feature.falseValue);
   }
   else if (action == MM::AfterSet)
   {
      if (IsCapturing())
         return DEVICE_CAMERA_BUSY_ACQUIRING;
      std::string value;
      property->Get(value);
      if (feature.restartStream)
         CloseStreamLocked();
      const EVT_ERROR error = EVT_CameraSetBoolParam(
         &camera_, feature.sdkName, value == feature.trueValue);
      const int streamRet = feature.restartStream ? OpenStreamLocked() : DEVICE_OK;
      if (error != EVT_SUCCESS)
         return SdkError(feature.sdkName, error);
      if (streamRet != DEVICE_OK)
         return streamRet;
   }
   return DEVICE_OK;
}

int EmergentCamera::OnEnumFeature(MM::PropertyBase* property,
                                  MM::ActionType action, long featureIndex)
{
   if (featureIndex < 0 || static_cast<size_t>(featureIndex) >=
         sizeof(kEnumFeatures) / sizeof(kEnumFeatures[0]))
      return DEVICE_INVALID_INPUT_PARAM;
   const EnumFeatureDefinition& feature = kEnumFeatures[featureIndex];
   MMThreadGuard guard(cameraLock_);
   if (action == MM::BeforeGet)
   {
      char value[128] = {};
      unsigned long size = 0;
      const EVT_ERROR error = EVT_CameraGetEnumParam(
         &camera_, feature.sdkName, value, sizeof(value), &size);
      if (error != EVT_SUCCESS)
         return SdkError(feature.sdkName, error);
      property->Set(value);
   }
   else if (action == MM::AfterSet)
   {
      if (IsCapturing())
         return DEVICE_CAMERA_BUSY_ACQUIRING;
      std::string value;
      property->Get(value);
      if (feature.restartStream)
         CloseStreamLocked();
      const EVT_ERROR error =
         EVT_CameraSetEnumParam(&camera_, feature.sdkName, value.c_str());
      const int streamRet = feature.restartStream ? OpenStreamLocked() : DEVICE_OK;
      if (error != EVT_SUCCESS)
         return SdkError(feature.sdkName, error);
      if (streamRet != DEVICE_OK)
         return streamRet;
   }
   return DEVICE_OK;
}

int EmergentCamera::OnCameraSelection(MM::PropertyBase* property,
                                      MM::ActionType action)
{
   if (action == MM::BeforeGet)
   {
      property->Set(cameraSelection_.c_str());
   }
   else if (action == MM::AfterSet)
   {
      property->Get(cameraSelection_);
      if (cameraSelection_ != kManualIpSelection)
      {
         for (const auto& discovered : discoveredCameras_)
         {
            if (discovered.first == cameraSelection_)
            {
               ipAddress_ = discovered.second;
               break;
            }
         }
      }
   }
   return DEVICE_OK;
}

int EmergentCamera::OnConnectionSpeed(MM::PropertyBase* property, MM::ActionType action)
{
   if (action != MM::BeforeGet)
      return DEVICE_OK;
#ifdef _WIN32
   wchar_t alias[NIC_FRIENDLY_NAME_LEN] = {};
   if (MultiByteToWideChar(CP_ACP, 0, deviceInfo_.nic.friendlyName, -1,
                           alias, NIC_FRIENDLY_NAME_LEN) > 0)
   {
      MIB_IF_ROW2 row{};
      if (ConvertInterfaceAliasToLuid(alias, &row.InterfaceLuid) == NO_ERROR &&
          GetIfEntry2(&row) == NO_ERROR)
      {
         if (row.MediaConnectState != MediaConnectStateConnected)
            property->Set("Disconnected (host NIC)");
         else if (row.TransmitLinkSpeed > 0)
         {
            std::ostringstream value;
            value << row.TransmitLinkSpeed / 1000000000.0 << " Gbit/s (host NIC)";
            property->Set(value.str().c_str());
         }
         else
            property->Set("Unknown (host NIC)");
         return DEVICE_OK;
      }
   }
#elif defined(__linux__)
   const std::string interfaceName(deviceInfo_.nic.friendlyName);
   if (!interfaceName.empty() && interfaceName != "." && interfaceName != ".." &&
       interfaceName.find('/') == std::string::npos)
   {
      const std::string interfacePath = "/sys/class/net/" + interfaceName + "/";
      std::ifstream stateFile(interfacePath + "operstate");
      std::string state;
      if (stateFile >> state)
      {
         if (state == "down" || state == "notpresent" || state == "lowerlayerdown")
         {
            property->Set("Disconnected (host NIC)");
            return DEVICE_OK;
         }
         std::ifstream speedFile(interfacePath + "speed");
         long long speedMbps = 0;
         if (speedFile >> speedMbps)
         {
            if (speedMbps > 0)
            {
               std::ostringstream value;
               value << speedMbps / 1000.0 << " Gbit/s (host NIC)";
               property->Set(value.str().c_str());
            }
            else
               property->Set("Unknown (host NIC)");
            return DEVICE_OK;
         }
      }
   }
#endif
   property->Set("Unavailable (host NIC)");
   return DEVICE_OK;
}

int EmergentCamera::OnIPAddress(MM::PropertyBase* property, MM::ActionType action)
{
   if (action == MM::BeforeGet)
      property->Set(ipAddress_.c_str());
   else if (action == MM::AfterSet)
   {
      property->Get(ipAddress_);
      cameraSelection_ = kManualIpSelection;
      for (const auto& discovered : discoveredCameras_)
      {
         if (discovered.second == ipAddress_)
         {
            cameraSelection_ = discovered.first;
            break;
         }
      }
   }
   return DEVICE_OK;
}

int EmergentCamera::OnPixelType(MM::PropertyBase* property, MM::ActionType action)
{
   MMThreadGuard guard(cameraLock_);
   if (action == MM::BeforeGet)
   {
      property->Set(BitDepthDisplay(pixelFormat_).c_str());
      return DEVICE_OK;
   }
   if (IsCapturing())
      return DEVICE_CAMERA_BUSY_ACQUIRING;

   std::string requested;
   property->Get(requested);
   const int ret = SetPixelFormatLocked(BitDepthNative(requested));
   if (ret == DEVICE_OK)
      property->Set(BitDepthDisplay(pixelFormat_).c_str());
   return ret;
}

int EmergentCamera::SetPixelFormatLocked(const std::string& format)
{
   if (!IsSupportedMonoFormat(format) ||
       std::find(pixelFormats_.begin(), pixelFormats_.end(), format) == pixelFormats_.end())
      return ERR_UNSUPPORTED_PIXEL_FORMAT;
   if (format == pixelFormat_)
      return DEVICE_OK;

   const unsigned savedOffsetX = offsetX_;
   const unsigned savedOffsetY = offsetY_;
   const unsigned savedWidth = width_;
   const unsigned savedHeight = height_;
   CloseStreamLocked();
   const EVT_ERROR error = EVT_CameraSetEnumParam(&camera_, "PixelFormat", format.c_str());
   if (error != EVT_SUCCESS)
   {
      OpenStreamLocked();
      return SdkError("set PixelFormat", error);
   }

   pixelFormat_ = format;
   bytesPerPixel_ = PixelBytes(pixelFormat_);
   bitDepth_ = PixelBits(pixelFormat_);
   int ret = ConfigureRoiLocked(savedOffsetX, savedOffsetY, savedWidth, savedHeight);
   if (ret == DEVICE_OK)
      ret = ApplyFrameRatePolicyLocked();
   if (ret == DEVICE_OK)
      ret = RefreshExposureLimitsLocked(!frameRateLimitEnabled_);
   image_.Resize(width_, height_, bytesPerPixel_);
   const int streamRet = OpenStreamLocked();
   return ret != DEVICE_OK ? ret : streamRet;
}

int EmergentCamera::OnEnumProperty(MM::PropertyBase* property,
                                   MM::ActionType action, const char* sdkName)
{
   MMThreadGuard guard(cameraLock_);
   if (action == MM::BeforeGet)
   {
      char value[128] = {};
      unsigned long size = 0;
      const EVT_ERROR error =
         EVT_CameraGetEnumParam(&camera_, sdkName, value, sizeof(value), &size);
      if (error != EVT_SUCCESS)
         return SdkError(sdkName, error);
      property->Set(value);
   }
   else if (action == MM::AfterSet)
   {
      if (IsCapturing())
         return DEVICE_CAMERA_BUSY_ACQUIRING;
      std::string value;
      property->Get(value);
      const EVT_ERROR error = EVT_CameraSetEnumParam(&camera_, sdkName, value.c_str());
      if (error != EVT_SUCCESS)
         return SdkError(sdkName, error);
   }
   return DEVICE_OK;
}

int EmergentCamera::OnBoolPolarityProperty(MM::PropertyBase* property,
                                           MM::ActionType action,
                                           const char* sdkName)
{
   MMThreadGuard guard(cameraLock_);
   if (action == MM::BeforeGet)
   {
      bool activeHigh = false;
      const EVT_ERROR error = EVT_CameraGetBoolParam(&camera_, sdkName, &activeHigh);
      if (error != EVT_SUCCESS)
         return SdkError(sdkName, error);
      property->Set(activeHigh ? kPolarityNormal : kPolarityInverted);
   }
   else if (action == MM::AfterSet)
   {
      if (IsCapturing())
         return DEVICE_CAMERA_BUSY_ACQUIRING;
      std::string value;
      property->Get(value);
      const EVT_ERROR error = EVT_CameraSetBoolParam(
         &camera_, sdkName, value == kPolarityNormal);
      if (error != EVT_SUCCESS)
         return SdkError(sdkName, error);
   }
   return DEVICE_OK;
}

int EmergentCamera::OnTemperatureProperty(MM::PropertyBase* property,
                                          MM::ActionType action,
                                          const char* sdkName)
{
   if (action != MM::BeforeGet)
      return DEVICE_OK;
   MMThreadGuard guard(cameraLock_);
   int temperature = 0;
   const EVT_ERROR error = EVT_CameraGetInt32Param(&camera_, sdkName, &temperature);
   if (error != EVT_SUCCESS)
      return SdkError(sdkName, error);
   property->Set(static_cast<long>(temperature));
   return DEVICE_OK;
}

int EmergentCamera::ConfigureTriggerModeLocked(const std::string& mode)
{
   EVT_ERROR error =
      EVT_CameraSetEnumParam(&camera_, "TriggerSelector", "AcquisitionStart");
   if (error != EVT_SUCCESS)
      return SdkError("set TriggerSelector=AcquisitionStart", error);

   if (mode == kTriggerInternal)
   {
      error = EVT_CameraSetEnumParam(&camera_, "TriggerMode", "Off");
   }
   else
   {
      const char* source = mode == kTriggerSoftware ? "Software" : "Hardware";
      error = EVT_CameraSetEnumParam(&camera_, "TriggerSource", source);
      if (error == EVT_SUCCESS)
         error = EVT_CameraSetEnumParam(&camera_, "TriggerMode", "On");
   }
   if (error != EVT_SUCCESS)
      return SdkError("configure trigger mode", error);
   triggerMode_ = mode;
   return DEVICE_OK;
}

int EmergentCamera::FireSoftwareTriggerLocked()
{
   if (triggerMode_ != kTriggerSoftware || !softwareTriggerPending_)
      return DEVICE_OK;
   const EVT_ERROR error = EVT_CameraExecuteCommand(&camera_, "TriggerSoftware");
   if (error == EVT_SUCCESS && !softwareTriggerPerFrame_)
      softwareTriggerPending_ = false;
   return error == EVT_SUCCESS ? DEVICE_OK : SdkError("TriggerSoftware", error);
}

int EmergentCamera::OnTriggerMode(MM::PropertyBase* property, MM::ActionType action)
{
   MMThreadGuard guard(cameraLock_);
   if (action == MM::BeforeGet)
   {
      property->Set(triggerMode_.c_str());
      return DEVICE_OK;
   }
   if (IsCapturing())
      return DEVICE_CAMERA_BUSY_ACQUIRING;
   std::string requested;
   property->Get(requested);
   if (requested != kTriggerInternal && requested != kTriggerSoftware &&
       requested != kTriggerExternal)
      return DEVICE_INVALID_INPUT_PARAM;
   const int ret = ConfigureTriggerModeLocked(requested);
   if (ret == DEVICE_OK)
      property->Set(triggerMode_.c_str());
   return ret;
}

int EmergentCamera::OnPtpOffset(MM::PropertyBase* property, MM::ActionType action)
{
   if (action != MM::BeforeGet)
      return DEVICE_OK;
   MMThreadGuard guard(cameraLock_);
   int offset = 0;
   const EVT_ERROR error = EVT_CameraGetInt32Param(&camera_, "PtpOffset", &offset);
   if (error != EVT_SUCCESS)
      return SdkError("get PtpOffset", error);
   property->Set(static_cast<long>(offset));
   return DEVICE_OK;
}

int EmergentCamera::OnCameraTimestamp(MM::PropertyBase* property, MM::ActionType action)
{
   if (action != MM::BeforeGet)
      return DEVICE_OK;
   MMThreadGuard guard(cameraLock_);
   unsigned high = 0, low = 0;
   EVT_ERROR error = EVT_CameraExecuteCommand(&camera_, "GevTimestampControlLatch");
   if (error == EVT_SUCCESS)
      error = EVT_CameraGetUInt32Param(&camera_, "GevTimestampValueHigh", &high);
   if (error == EVT_SUCCESS)
      error = EVT_CameraGetUInt32Param(&camera_, "GevTimestampValueLow", &low);
   if (error != EVT_SUCCESS)
      return SdkError("latch/read camera timestamp", error);
   const auto timestamp = (static_cast<unsigned long long>(high) << 32) | low;
   property->Set(std::to_string(timestamp).c_str());
   return DEVICE_OK;
}

int EmergentCamera::OnTriggerInput(MM::PropertyBase* property, MM::ActionType action)
{
   return OnEnumProperty(property, action, "GPI_Start_Exp_Mode");
}

int EmergentCamera::OnTriggerInputEdge(MM::PropertyBase* property,
                                       MM::ActionType action)
{
   return OnEnumProperty(property, action, "GPI_Start_Exp_Event");
}

int EmergentCamera::OnTriggerEndMode(MM::PropertyBase* property,
                                     MM::ActionType action)
{
   return OnEnumProperty(property, action, "GPI_End_Exp_Mode");
}

int EmergentCamera::OnTriggerEndEdge(MM::PropertyBase* property,
                                     MM::ActionType action)
{
   return OnEnumProperty(property, action, "GPI_End_Exp_Event");
}

int EmergentCamera::OnTriggerDelay(MM::PropertyBase* property,
                                   MM::ActionType action)
{
   MMThreadGuard guard(cameraLock_);
   if (action == MM::BeforeGet)
   {
      unsigned delay = 0;
      const EVT_ERROR error =
         EVT_CameraGetUInt32Param(&camera_, "Trigger_Delay", &delay);
      if (error != EVT_SUCCESS)
         return SdkError("get Trigger_Delay", error);
      property->Set(static_cast<long>(delay));
   }
   else if (action == MM::AfterSet)
   {
      if (IsCapturing())
         return DEVICE_CAMERA_BUSY_ACQUIRING;
      long delay = 0;
      property->Get(delay);
      if (delay < 0)
         return DEVICE_INVALID_INPUT_PARAM;
      const EVT_ERROR error = EVT_CameraSetUInt32Param(
         &camera_, "Trigger_Delay", static_cast<unsigned>(delay));
      if (error != EVT_SUCCESS)
         return SdkError("set Trigger_Delay", error);
   }
   return DEVICE_OK;
}

int EmergentCamera::OnOutput0Source(MM::PropertyBase* property,
                                    MM::ActionType action)
{
   return OnEnumProperty(property, action, "GPO_0_Mode");
}

int EmergentCamera::OnOutput0Polarity(MM::PropertyBase* property,
                                      MM::ActionType action)
{
   return OnBoolPolarityProperty(property, action, "GPO_0_Polarity");
}

int EmergentCamera::OnOutput1Source(MM::PropertyBase* property,
                                    MM::ActionType action)
{
   return OnEnumProperty(property, action, "GPO_1_Mode");
}

int EmergentCamera::OnOutput1Polarity(MM::PropertyBase* property,
                                      MM::ActionType action)
{
   return OnBoolPolarityProperty(property, action, "GPO_1_Polarity");
}

int EmergentCamera::OnSensorTemperature(MM::PropertyBase* property,
                                        MM::ActionType action)
{
   return OnTemperatureProperty(property, action, "SensTemp");
}

int EmergentCamera::OnCameraTemperature(MM::PropertyBase* property,
                                        MM::ActionType action)
{
   return OnTemperatureProperty(property, action, "FPGATemp");
}

int EmergentCamera::OnPhyTemperature(MM::PropertyBase* property,
                                     MM::ActionType action)
{
   return OnTemperatureProperty(property, action, "PHYTemp");
}

int EmergentCamera::ApplyRoiLocked(unsigned x, unsigned y, unsigned width,
                                   unsigned height)
{
   if (IsCapturing())
      return DEVICE_CAMERA_BUSY_ACQUIRING;
   CloseStreamLocked();
   int ret = ConfigureRoiLocked(x, y, width, height);
   if (ret == DEVICE_OK)
      ret = ApplyFrameRatePolicyLocked();
   if (ret == DEVICE_OK)
      ret = RefreshExposureLimitsLocked(!frameRateLimitEnabled_);
   image_.Resize(width_, height_, bytesPerPixel_);
   const int streamRet = OpenStreamLocked();
   return ret != DEVICE_OK ? ret : streamRet;
}

int EmergentCamera::OnRoiX(MM::PropertyBase* property, MM::ActionType action)
{
   MMThreadGuard guard(cameraLock_);
   if (action == MM::BeforeGet)
      property->Set(static_cast<long>(offsetX_));
   else
   {
      long value = 0;
      property->Get(value);
      if (value < 0)
         return DEVICE_INVALID_INPUT_PARAM;
      const int ret = ApplyRoiLocked(static_cast<unsigned>(value), offsetY_, width_, height_);
      property->Set(static_cast<long>(offsetX_));
      return ret;
   }
   return DEVICE_OK;
}

int EmergentCamera::OnRoiY(MM::PropertyBase* property, MM::ActionType action)
{
   MMThreadGuard guard(cameraLock_);
   if (action == MM::BeforeGet)
      property->Set(static_cast<long>(offsetY_));
   else
   {
      long value = 0;
      property->Get(value);
      if (value < 0)
         return DEVICE_INVALID_INPUT_PARAM;
      const int ret = ApplyRoiLocked(offsetX_, static_cast<unsigned>(value), width_, height_);
      property->Set(static_cast<long>(offsetY_));
      return ret;
   }
   return DEVICE_OK;
}

int EmergentCamera::OnRoiWidth(MM::PropertyBase* property, MM::ActionType action)
{
   MMThreadGuard guard(cameraLock_);
   if (action == MM::BeforeGet)
      property->Set(static_cast<long>(width_));
   else
   {
      long value = 0;
      property->Get(value);
      if (value <= 0)
         return DEVICE_INVALID_INPUT_PARAM;
      const int ret = ApplyRoiLocked(offsetX_, offsetY_, static_cast<unsigned>(value), height_);
      property->Set(static_cast<long>(width_));
      return ret;
   }
   return DEVICE_OK;
}

int EmergentCamera::OnRoiHeight(MM::PropertyBase* property, MM::ActionType action)
{
   MMThreadGuard guard(cameraLock_);
   if (action == MM::BeforeGet)
      property->Set(static_cast<long>(height_));
   else
   {
      long value = 0;
      property->Get(value);
      if (value <= 0)
         return DEVICE_INVALID_INPUT_PARAM;
      const int ret = ApplyRoiLocked(offsetX_, offsetY_, width_, static_cast<unsigned>(value));
      property->Set(static_cast<long>(height_));
      return ret;
   }
   return DEVICE_OK;
}

int EmergentCamera::OnRoiCenterX(MM::PropertyBase* property,
                                 MM::ActionType action)
{
   if (action == MM::BeforeGet)
   {
      property->Set(kRoiCenterIdle);
      return DEVICE_OK;
   }

   std::string command;
   property->Get(command);
   property->Set(kRoiCenterIdle);
   if (command != kRoiCenterNow)
      return DEVICE_OK;

   int ret = DEVICE_OK;
   unsigned updatedOffset = 0;
   {
      MMThreadGuard guard(cameraLock_);
      const unsigned centeredOffset =
         sensorWidth_ > width_ ? (sensorWidth_ - width_) / 2 : 0;
      ret = ApplyRoiLocked(centeredOffset, offsetY_, width_, height_);
      updatedOffset = offsetX_;
   }
   if (ret == DEVICE_OK)
      OnPropertyChanged(kPropertyRoiX,
         CDeviceUtils::ConvertToString(static_cast<long>(updatedOffset)));
   return ret;
}

int EmergentCamera::OnRoiCenterY(MM::PropertyBase* property,
                                 MM::ActionType action)
{
   if (action == MM::BeforeGet)
   {
      property->Set(kRoiCenterIdle);
      return DEVICE_OK;
   }

   std::string command;
   property->Get(command);
   property->Set(kRoiCenterIdle);
   if (command != kRoiCenterNow)
      return DEVICE_OK;

   int ret = DEVICE_OK;
   unsigned updatedOffset = 0;
   {
      MMThreadGuard guard(cameraLock_);
      const unsigned centeredOffset =
         sensorHeight_ > height_ ? (sensorHeight_ - height_) / 2 : 0;
      ret = ApplyRoiLocked(offsetX_, centeredOffset, width_, height_);
      updatedOffset = offsetY_;
   }
   if (ret == DEVICE_OK)
      OnPropertyChanged(kPropertyRoiY,
         CDeviceUtils::ConvertToString(static_cast<long>(updatedOffset)));
   return ret;
}

int EmergentCamera::SetROI(unsigned x, unsigned y, unsigned xSize, unsigned ySize)
{
   if (IsCapturing())
      return DEVICE_CAMERA_BUSY_ACQUIRING;
   if (xSize == 0 || ySize == 0 || x >= sensorWidth_ || y >= sensorHeight_)
      return ERR_INVALID_ROI;

   MMThreadGuard guard(cameraLock_);
   CloseStreamLocked();
   int ret = ConfigureRoiLocked(x, y, xSize, ySize);
   if (ret == DEVICE_OK)
      ret = ApplyFrameRatePolicyLocked();
   if (ret == DEVICE_OK)
      ret = RefreshExposureLimitsLocked(!frameRateLimitEnabled_);
   image_.Resize(width_, height_, bytesPerPixel_);
   const int streamRet = OpenStreamLocked();
   return ret != DEVICE_OK ? ret : streamRet;
}

int EmergentCamera::ConfigureRoiLocked(unsigned x, unsigned y, unsigned requestedWidth,
                                       unsigned requestedHeight)
{
   x = AlignDown(x, offsetXInc_);
   y = AlignDown(y, offsetYInc_);
   if (x >= sensorWidth_ || y >= sensorHeight_)
      return ERR_INVALID_ROI;

   const unsigned maxWidth = sensorWidth_ - x;
   const unsigned maxHeight = sensorHeight_ - y;
   const unsigned newWidth = ClampAndAlignSize(requestedWidth, widthMin_, maxWidth, widthInc_);
   const unsigned newHeight = ClampAndAlignSize(requestedHeight, heightMin_, maxHeight, heightInc_);
   if (newWidth == 0 || newHeight == 0)
      return ERR_INVALID_ROI;

   // Reset offsets first so expanding an existing ROI cannot violate the
   // camera's dynamic Width/Height maxima. Then set size before final offsets.
   EVT_ERROR error = EVT_CameraSetUInt32Param(&camera_, "OffsetX", 0);
   if (error == EVT_SUCCESS)
      error = EVT_CameraSetUInt32Param(&camera_, "OffsetY", 0);
   if (error == EVT_SUCCESS)
      error = EVT_CameraSetUInt32Param(&camera_, "Width", newWidth);
   if (error == EVT_SUCCESS)
      error = EVT_CameraSetUInt32Param(&camera_, "Height", newHeight);
   if (error == EVT_SUCCESS)
      error = EVT_CameraSetUInt32Param(&camera_, "OffsetX", x);
   if (error == EVT_SUCCESS)
      error = EVT_CameraSetUInt32Param(&camera_, "OffsetY", y);

   const int refreshRet = RefreshGeometryLocked();
   if (error != EVT_SUCCESS)
      return SdkError("configure ROI", error);
   return refreshRet;
}

int EmergentCamera::ConfigureFullFrameLocked()
{
   EVT_ERROR error = EVT_CameraSetUInt32Param(&camera_, "OffsetX", 0);
   if (error == EVT_SUCCESS)
      error = EVT_CameraSetUInt32Param(&camera_, "OffsetY", 0);
   if (error != EVT_SUCCESS)
      return SdkError("clear ROI offsets", error);

   error = EVT_CameraGetUInt32ParamMax(&camera_, "Width", &sensorWidth_);
   if (error == EVT_SUCCESS)
      error = EVT_CameraGetUInt32ParamMax(&camera_, "Height", &sensorHeight_);
   if (error == EVT_SUCCESS)
      error = EVT_CameraSetUInt32Param(&camera_, "Width", sensorWidth_);
   if (error == EVT_SUCCESS)
      error = EVT_CameraSetUInt32Param(&camera_, "Height", sensorHeight_);
   if (error != EVT_SUCCESS)
      return SdkError("configure full-frame ROI", error);
   return RefreshGeometryLocked();
}

int EmergentCamera::GetROI(unsigned& x, unsigned& y, unsigned& xSize, unsigned& ySize)
{
   MMThreadGuard guard(cameraLock_);
   x = offsetX_;
   y = offsetY_;
   xSize = width_;
   ySize = height_;
   return DEVICE_OK;
}

int EmergentCamera::ClearROI()
{
   if (IsCapturing())
      return DEVICE_CAMERA_BUSY_ACQUIRING;
   MMThreadGuard guard(cameraLock_);
   CloseStreamLocked();
   int ret = ConfigureFullFrameLocked();
   if (ret == DEVICE_OK)
      ret = ApplyFrameRatePolicyLocked();
   if (ret == DEVICE_OK)
      ret = RefreshExposureLimitsLocked(!frameRateLimitEnabled_);
   image_.Resize(width_, height_, bytesPerPixel_);
   const int streamRet = OpenStreamLocked();
   return ret != DEVICE_OK ? ret : streamRet;
}

int EmergentCamera::GetMultiROICount(unsigned& count)
{
   count = 0;
   return DEVICE_OK;
}

int EmergentCamera::SetMultiROI(const unsigned*, const unsigned*, const unsigned*,
                                const unsigned*, unsigned)
{
   return DEVICE_UNSUPPORTED_COMMAND;
}

int EmergentCamera::GetMultiROI(unsigned*, unsigned*, unsigned*, unsigned*, unsigned*)
{
   return DEVICE_UNSUPPORTED_COMMAND;
}

int EmergentCamera::StartSequenceAcquisition(double intervalMs)
{
   return StartSequenceAcquisition(std::numeric_limits<long>::max(), intervalMs, false);
}

int EmergentCamera::StartSequenceAcquisition(long numImages, double intervalMs,
                                             bool stopOnOverflow)
{
   if (IsCapturing())
      return DEVICE_CAMERA_BUSY_ACQUIRING;
   if (numImages <= 0)
      return DEVICE_INVALID_INPUT_PARAM;

   int ret = GetCoreCallback()->PrepareForAcq(this);
   if (ret != DEVICE_OK)
      return ret;

   {
      MMThreadGuard guard(cameraLock_);
      sequenceBurstSize_ = 0;
      char acquisitionMode[64] = {};
      unsigned long acquisitionModeSize = 0;
      if (EVT_CameraGetEnumParam(&camera_, "AcquisitionMode", acquisitionMode,
                                 sizeof(acquisitionMode), &acquisitionModeSize) == EVT_SUCCESS &&
          std::strcmp(acquisitionMode, "MultiFrame") == 0)
      {
         unsigned maximumFrameCount = 0;
         if (EVT_CameraGetUInt32ParamMax(&camera_, "AcquisitionFrameCount",
                                         &maximumFrameCount) != EVT_SUCCESS || maximumFrameCount == 0)
            ret = DEVICE_INVALID_INPUT_PARAM;
         else
         {
            maximumFrameCount = UsableUInt32Maximum("AcquisitionFrameCount", maximumFrameCount);
            sequenceBurstSize_ = static_cast<unsigned>(std::min(
               static_cast<unsigned long long>(numImages),
               static_cast<unsigned long long>(maximumFrameCount)));
            // Live mode must be interruptible between complete bursts; stopping
            // a large partial burst does not immediately disarm this camera.
            if (numImages == std::numeric_limits<long>::max())
               sequenceBurstSize_ = 1;
            const EVT_ERROR error = EVT_CameraSetUInt32Param(
               &camera_, "AcquisitionFrameCount", sequenceBurstSize_);
            if (error != EVT_SUCCESS)
               ret = SdkError("set AcquisitionFrameCount", error);
         }
      }
      if (ret == DEVICE_OK)
         ret = StartAcquisitionLocked();
   }
   if (ret != DEVICE_OK)
      return ret;

   sequenceStartTime_ = GetCurrentMMTime();
   sequenceThread_->Start(numImages, intervalMs, stopOnOverflow);
   return DEVICE_OK;
}

int EmergentCamera::StopSequenceAcquisition()
{
   sequenceThread_->Stop();
   sequenceThread_->wait();
   return DEVICE_OK;
}

bool EmergentCamera::IsCapturing()
{
   return sequenceThread_ && !sequenceThread_->IsStopped();
}

int EmergentCamera::SequenceLoop(long numImages, double intervalMs, bool stopOnOverflow)
{
   // Like most hardware camera adapters, acquisition is paced by the camera.
   // MMCore's interval argument is advisory and is intentionally ignored here.
   (void)intervalMs;
   int result = DEVICE_OK;
   unsigned framesInBurst = 0;
   for (long imageNumber = 0;
        imageNumber < numImages && !sequenceThread_->IsStopped(); ++imageNumber)
   {
      {
         MMThreadGuard guard(cameraLock_);
         if (sequenceBurstSize_ > 0 && framesInBurst == sequenceBurstSize_)
         {
            StopAcquisitionLocked();
            const unsigned nextBurst = static_cast<unsigned>(std::min(
               static_cast<unsigned long long>(numImages - imageNumber),
               static_cast<unsigned long long>(sequenceBurstSize_)));
            const EVT_ERROR error = EVT_CameraSetUInt32Param(&camera_, "AcquisitionFrameCount", nextBurst);
            result = error == EVT_SUCCESS ? StartAcquisitionLocked() :
               SdkError("set next AcquisitionFrameCount", error);
            framesInBurst = 0;
         }
         if (result == DEVICE_OK)
            result = FireSoftwareTriggerLocked();
         if (result == DEVICE_OK)
            result = AcquireOneFrameLocked(true);
      }
      if (result == ERR_ACQUISITION_STOPPED)
      {
         result = DEVICE_OK;
         break;
      }
      if (result != DEVICE_OK)
         break;
      ++framesInBurst;

      result = InsertCurrentImage(stopOnOverflow);
      if (result != DEVICE_OK)
         break;
   }

   {
      MMThreadGuard guard(cameraLock_);
      StopAcquisitionLocked();
   }
   return result;
}

int EmergentCamera::InsertCurrentImage(bool stopOnOverflow)
{
   (void)stopOnOverflow; // MMCore handles overflow policy for current adapters.
   MM::CameraImageMetadata metadata;
   char label[MM::MaxStrLength] = {};
   GetLabel(label);
   metadata.AddTag(MM::g_Keyword_Metadata_CameraLabel, label);
   metadata.AddTag(MM::g_Keyword_Elapsed_Time_ms,
      CDeviceUtils::ConvertToString((GetCurrentMMTime() - sequenceStartTime_).getMsec()));
   metadata.AddTag(MM::g_Keyword_Metadata_ROI_X,
      CDeviceUtils::ConvertToString(static_cast<long>(offsetX_)));
   metadata.AddTag(MM::g_Keyword_Metadata_ROI_Y,
      CDeviceUtils::ConvertToString(static_cast<long>(offsetY_)));
   metadata.AddTag("Emergent-HardwareTimestampNs", std::to_string(frameTimestampNs_).c_str());
   metadata.AddTag("Emergent-FrameId", std::to_string(frameId_).c_str());

   MMThreadGuard guard(imageLock_);
   return GetCoreCallback()->InsertImage(this, image_.GetPixels(), width_, height_,
                                          bytesPerPixel_, 1, metadata.Serialize());
}

void EmergentCamera::SequenceFinished(int status) throw()
{
   try
   {
      if (GetCoreCallback())
         GetCoreCallback()->AcqFinished(this, status);
   }
   catch (...)
   {
      LogMessage("Exception while completing Emergent sequence acquisition", false);
   }
}

int EmergentCamera::IsExposureSequenceable(bool& isSequenceable) const
{
   isSequenceable = false;
   return DEVICE_OK;
}

int EmergentCamera::GetExposureSequenceMaxLength(long& nrEvents) const
{
   nrEvents = 0;
   return DEVICE_UNSUPPORTED_COMMAND;
}

int EmergentCamera::StartExposureSequence() { return DEVICE_UNSUPPORTED_COMMAND; }
int EmergentCamera::StopExposureSequence() { return DEVICE_UNSUPPORTED_COMMAND; }
int EmergentCamera::ClearExposureSequence() { return DEVICE_UNSUPPORTED_COMMAND; }
int EmergentCamera::AddToExposureSequence(double) { return DEVICE_UNSUPPORTED_COMMAND; }
int EmergentCamera::SendExposureSequence() const { return DEVICE_UNSUPPORTED_COMMAND; }

int EmergentCamera::SdkError(const char* operation, EVT_ERROR error) const
{
   std::ostringstream message;
   message << operation << " failed with eSDK error " << static_cast<int>(error);
   LogMessage(message.str().c_str(), false);
   return ERR_ESDK_CALL;
}

std::vector<std::string> EmergentCamera::SplitEnumValues(const char* values)
{
   std::vector<std::string> result;
   std::stringstream input(values ? values : "");
   std::string value;
   while (std::getline(input, value, ','))
   {
      if (!value.empty())
         result.push_back(value);
   }
   return result;
}

bool EmergentCamera::IsSupportedMonoFormat(const std::string& value)
{
   return value == "Mono8" || value == "Mono10" || value == "Mono12";
}

PIXEL_FORMAT EmergentCamera::PixelFormatCode(const std::string& value)
{
   if (value == "Mono10")
      return GVSP_PIX_MONO10;
   if (value == "Mono12")
      return GVSP_PIX_MONO12;
   return GVSP_PIX_MONO8;
}

unsigned EmergentCamera::PixelBytes(const std::string& value)
{
   return value == "Mono8" ? 1u : 2u;
}

unsigned EmergentCamera::PixelBits(const std::string& value)
{
   if (value == "Mono10")
      return 10;
   if (value == "Mono12")
      return 12;
   return 8;
}

unsigned EmergentCamera::AlignDown(unsigned value, unsigned increment)
{
   increment = std::max(1u, increment);
   return value - value % increment;
}

unsigned EmergentCamera::ClampAndAlignSize(unsigned requested, unsigned minimum,
                                           unsigned maximum, unsigned increment)
{
   if (maximum < minimum)
      return 0;
   const unsigned clamped = std::max(minimum, std::min(requested, maximum));
   const unsigned aligned = AlignDown(clamped, increment);
   return aligned >= minimum ? aligned : minimum;
}

EmergentSequenceThread::EmergentSequenceThread(EmergentCamera* camera) :
   camera_(camera),
   stop_(true),
   numImages_(0),
   intervalMs_(0.0),
   stopOnOverflow_(false)
{}

void EmergentSequenceThread::Start(long numImages, double intervalMs, bool stopOnOverflow)
{
   // Join a naturally completed acquisition before reusing its thread state.
   wait();
   numImages_ = numImages;
   intervalMs_ = std::max(0.0, intervalMs);
   stopOnOverflow_ = stopOnOverflow;
   stop_.store(false);
   activate();
}

void EmergentSequenceThread::Stop()
{
   stop_.store(true);
}

bool EmergentSequenceThread::IsStopped() const
{
   return stop_.load();
}

int EmergentSequenceThread::svc() throw()
{
   int result = DEVICE_ERR;
   try
   {
      result = camera_->SequenceLoop(numImages_, intervalMs_, stopOnOverflow_);
   }
   catch (...)
   {
      camera_->LogMessage("Exception in Emergent acquisition thread", false);
   }
   camera_->SequenceFinished(result);
   stop_.store(true);
   return result;
}
