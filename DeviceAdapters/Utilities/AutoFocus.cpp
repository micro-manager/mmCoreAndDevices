///////////////////////////////////////////////////////////////////////////////
// FILE:          AutoFocus.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Various 'Meta-Devices' that add to or combine functionality of
//                physcial devices.
//
// AUTHOR:        Nico Stuurman, nico.stuurman@ucsf.edu 2025
// COPYRIGHT:     University of California, San Francisco, 2008-2025
//                2015-2016, Open Imaging, Inc.
//                Altos Labs, 2022
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
//

#ifdef _WIN32
// Prevent windows.h from defining min and max macros,
// which clash with std::min and std::max.
#define NOMINMAX
#endif

#include "Utilities.h"
#include <fstream>
#include <cstdlib>

const char* g_Camera = "Devices_Camera";
const char* g_Shutter = "Devices_Shutter";
const char* g_FocusStage = "Devices_FocusStage";
const char* g_Settings = "Settings";
const char* g_SettingsDescription = "SettingsDescription";
const char* g_Alg = "Algorithm";
const char* g_Alg_Standard = "Standard";

AutoFocus::AutoFocus() :
   initialized_(false),
   continuousFocusing_(false),
   offset_(0.0),
   algorithm_(g_Alg_Standard),
   roiX_(0),
   roiY_(0),
   roiWidth_(0),
   roiHeight_(0),
   binning_(1),
   exposureMs_(5.0),
   deviceSettings_(0),
   deviceSettingsDescription_(""),
   spotSelection_("Top"),
   precision_(0.1),
   maxZ_(10000.0),
   continuousFocusLocked_(false),
   stopThread_(false),
   status_("Off")
{
   InitializeDefaultErrorMessages();
   SetErrorText(ERR_NO_PHYSICAL_CAMERA, "No physical camera found.  Please select a valid camera in the Camera property.");
   SetErrorText(ERR_AUTOFOCUS_NOT_SUPPORTED, "The selected camera does not support AutoFocus.");
   SetErrorText(ERR_NO_SHUTTER_DEVICE_FOUND, "No Shutter device found.  Please select a valid shutter in the Shutter property.");
   SetErrorText(ERR_NO_AUTOFOCUS_DEVICE, "No AutoFocus Device selected");
   SetErrorText(ERR_NO_AUTOFOCUS_DEVICE_FOUND, "No AutoFocus Device loaded");
   SetErrorText(ERR_TAGET_TOO_HIGH, "Target position exceeds allowed maximum.  Likely error in autofocus");
   SetErrorText(ERR_NOT_CALIBRATED, "These calibration settings are not yet calibrated.  Calibrate first.");
   // Name
   CreateProperty(MM::g_Keyword_Name, "AutoFocus", MM::String, true);
   // Description
   CreateProperty(MM::g_Keyword_Description, "Hardware-based autofocus device that uses a shutter and a camera to determine the location/size of the reflection spot", MM::String, true);
}

AutoFocus::~AutoFocus()
{
   if (initialized_)
      Shutdown();
}

int AutoFocus::Shutdown()
{
   if (!initialized_)
      return DEVICE_OK;

   // Signal thread to stop
   stopThread_ = true;

   // Wake up thread if waiting
   continuousFocusCV_.notify_one();

   // Wait for thread to finish
   if (continuousFocusThread_.joinable())
   {
      continuousFocusThread_.join();
   }

   // Clear registered stages (defensive - they should have unregistered)
   {
      std::lock_guard<std::mutex> lock(registrationMutex_);
      registeredStages_.clear();
   }

   initialized_ = false;
   return DEVICE_OK;
}

void AutoFocus::GetName(char* name) const
{
   CDeviceUtils::CopyLimitedString(name, "AutoFocus");
}

int AutoFocus::Initialize()
{
   // get list with available shutter devices.
   char deviceName[MM::MaxStrLength];
   unsigned int deviceIterator = 0;
   for (;;)
   {
      GetLoadedDeviceOfType(MM::ShutterDevice, deviceName, deviceIterator++);
      if (0 < strlen(deviceName))
      {
         availableShutters_.push_back(std::string(deviceName));
      }
      else
         break;
   }
   CPropertyAction* pAct = new CPropertyAction(this, &AutoFocus::OnShutter);
   std::string defaultShutter = "Undefined";
   if (availableShutters_.size() >= 1)
      defaultShutter = availableShutters_[0];
   CreateProperty(g_Shutter, defaultShutter.c_str(), MM::String, false, pAct, false);
   if (availableShutters_.size() >= 1)
      SetAllowedValues(g_Shutter, availableShutters_);
   else
      return ERR_NO_SHUTTER_DEVICE_FOUND;
   // This is needed, otherwise Shutter_ is not always set resulting in crashes
   // This could lead to strange problems if multiple shutter devices are loaded
   SetProperty(g_Shutter, defaultShutter.c_str());

   // Get list with available physical cameras.
   deviceIterator = 0;
   for (;;)
   {
      GetLoadedDeviceOfType(MM::CameraDevice, deviceName, deviceIterator++);
      if (0 < strlen(deviceName))
      {
         availableCameras_.push_back(std::string(deviceName));
      }
      else
         break;
   }
   pAct = new CPropertyAction(this, &AutoFocus::OnCamera);
   std::string defaultCamera = "Undefined";
   CreateProperty(g_Camera, defaultCamera.c_str(), MM::String, false, pAct, false);
   AddAllowedValue(g_Camera, defaultCamera.c_str());
   for (int i = 0; i < availableCameras_.size(); i++)
   {
      AddAllowedValue(g_Camera, availableCameras_[i].c_str());
   }

   // Get list with available focus stage devices.
   deviceIterator = 0;
   for (;;)
   {
      GetLoadedDeviceOfType(MM::StageDevice, deviceName, deviceIterator++);
      if (0 < strlen(deviceName))
      {
         availableFocusStages_.push_back(std::string(deviceName));
      }
      else
         break;
   }
   pAct = new CPropertyAction(this, &AutoFocus::OnFocusStage);
   std::string defaultFocusStage = "Undefined";
   CreateProperty(g_FocusStage, defaultFocusStage.c_str(), MM::String, false, pAct, false);
   AddAllowedValue(g_FocusStage, defaultFocusStage.c_str());
   for (int i = 0; i < availableFocusStages_.size(); i++)
   {
      AddAllowedValue(g_FocusStage, availableFocusStages_[i].c_str());
   }

   pAct = new CPropertyAction(this, &AutoFocus::OnAlgorithm);
   CreateProperty(g_Alg, g_Alg_Standard, MM::String, false, pAct);
   AddAllowedValue(g_Alg, g_Alg_Standard);


   // Create ROI-X property
   pAct = new CPropertyAction(this, &AutoFocus::OnROI_X);
   CreateIntegerProperty("Camera_ROI-X", 0, false, pAct);

   // Create ROI-Y property
   pAct = new CPropertyAction(this, &AutoFocus::OnROI_Y);
   CreateIntegerProperty("Camera_ROI-Y", 65536, false, pAct);

   // Create ROI-Width property
   pAct = new CPropertyAction(this, &AutoFocus::OnROI_Width);
   CreateIntegerProperty("Camera_ROI-Width", 65536, false, pAct);

   // Create ROI-Height property
   pAct = new CPropertyAction(this, &AutoFocus::OnROI_Height);
   CreateIntegerProperty("Camera_ROI-Height", 65536, false, pAct);

   // Create Binning property
   pAct = new CPropertyAction(this, &AutoFocus::OnBinning);
   CreateIntegerProperty("Camera_Binning", binning_, false, pAct);

   // Create Exposure property
   pAct = new CPropertyAction(this, &AutoFocus::OnExposure);
   CreateFloatProperty("Camera_Exposure_Ms", exposureMs_, false, pAct);
   SetPropertyLimits("Camera_Exposure_Ms", 0.1, 10000.0);

   // Create SpotSelection property
   pAct = new CPropertyAction(this, &AutoFocus::OnSpotSelection);
   CreateStringProperty("Cal_SpotSelection", spotSelection_.c_str(), false, pAct);
   AddAllowedValue("Cal_SpotSelection", "Top");
   AddAllowedValue("Cal_SpotSelection", "Bottom");

   // Create Precision property
   pAct = new CPropertyAction(this, &AutoFocus::OnPrecision);
   CreateFloatProperty("Cal_Precision", precision_, false, pAct);
   SetPropertyLimits("Cal_Precision", 0.0, 100.0);

   // Create MaxZ property
   pAct = new CPropertyAction(this, &AutoFocus::OnMaxZ);
   CreateFloatProperty("Cal_MaxZ", maxZ_, false, pAct);
   SetPropertyLimits("Cal_MaxZ", 0.0, 100000.0);

   // Create Status property (read-only)
   pAct = new CPropertyAction(this, &AutoFocus::OnStatus);
   CreateStringProperty("Status", status_.c_str(), true, pAct);
   AddAllowedValue("Status", "Off");
   AddAllowedValue("Status", "Out of Range");
   AddAllowedValue("Status", "Focusing");
   AddAllowedValue("Status", "Locked");

   // Create DeviceSettings property
   pAct = new CPropertyAction(this, &AutoFocus::OnDeviceSettings);
   CreateIntegerProperty(g_Settings, deviceSettings_, false, pAct);

   // Create DeviceSettingsDescription property
   pAct = new CPropertyAction(this, &AutoFocus::OnDeviceSettingsDescription);
   CreateStringProperty(g_SettingsDescription, deviceSettingsDescription_.c_str(), false, pAct);

   // Create Calibrate action property
   pAct = new CPropertyAction(this, &AutoFocus::OnCalibrate);
   CreateStringProperty("Calibrate", "Idle", false, pAct);
   AddAllowedValue("Calibrate", "Idle");
   AddAllowedValue("Calibrate", "Start");

   // Create MeasureOffset action property
   pAct = new CPropertyAction(this, &AutoFocus::OnMeasureOffset);
   CreateStringProperty("MeasureOffset", "Idle", false, pAct);
   AddAllowedValue("MeasureOffset", "Idle");
   AddAllowedValue("MeasureOffset", "Measure");

   // Load calibration data from file
   LoadCalibrationData();

   // Load properties for current device setting if it exists
   if (calibrationMap_.find(deviceSettings_) != calibrationMap_.end())
   {
      CalibrationData& cal = calibrationMap_[deviceSettings_];
      deviceSettingsDescription_ = cal.description;
      spotSelection_ = cal.spotSelection;
      precision_ = cal.precision;
      maxZ_ = cal.maxZ;

      // Apply ROI and binning settings
      roiX_ = cal.roiX;
      roiY_ = cal.roiY;
      roiWidth_ = cal.roiWidth;
      roiHeight_ = cal.roiHeight;
      SetCameraROI();

      binning_ = cal.binning;
      SetCameraBinning();
   }

   // Start continuous focusing thread
   stopThread_ = false;
   continuousFocusLocked_ = false;
   continuousFocusThread_ = std::thread(&AutoFocus::ContinuousFocusThread, this);

   initialized_ = true;
   return DEVICE_OK;

 }

int AutoFocus::OnShutter(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet) {
      pProp->Set(shutter_.c_str());
   }
   else if (eAct == MM::AfterSet) {
      pProp->Get(shutter_);
   }
   return DEVICE_OK;
}

int AutoFocus::OnCamera(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet) {
      pProp->Set(camera_.c_str());
   }
   else if (eAct == MM::AfterSet) {
      std::string oldCamera = camera_;
      pProp->Get(camera_);
      if (camera_ != "" && camera_ != "Undefined")
      {
         int ret = SetCameraBinning();
         if (ret == DEVICE_OK)
            ret = SetCameraROI();
         if (ret != DEVICE_OK)
         {
            // Restore previous value
            camera_ = oldCamera;
            pProp->Set(camera_.c_str());
            return ERR_NO_PHYSICAL_CAMERA;
         }
      }

   }
   return DEVICE_OK;
}

int AutoFocus::OnFocusStage(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet) {
      pProp->Set(focusStage_.c_str());
   }
   else if (eAct == MM::AfterSet) {
      pProp->Get(focusStage_);
   }
   return DEVICE_OK;
}

int AutoFocus::OnAlgorithm(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet) {
      pProp->Set(algorithm_.c_str());
   }
   else if (eAct == MM::AfterSet) {
      pProp->Get(algorithm_);
   }
   return DEVICE_OK;
}

int AutoFocus::OnROI_X(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set((long)roiX_);
   }
   else if (eAct == MM::AfterSet)
   {
      long value;
      pProp->Get(value);
      roiX_ = (unsigned)value;
      return SetCameraROI();
   }
   return DEVICE_OK;
}

int AutoFocus::OnROI_Y(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set((long)roiY_);
   }
   else if (eAct == MM::AfterSet)
   {
      long value;
      pProp->Get(value);
      roiY_ = (unsigned)value;
      return SetCameraROI();
   }
   return DEVICE_OK;
}

int AutoFocus::OnROI_Width(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set((long)roiWidth_);
   }
   else if (eAct == MM::AfterSet)
   {
      long value;
      pProp->Get(value);
      roiWidth_ = (unsigned)value;
      return SetCameraROI();
   }
   return DEVICE_OK;
}

int AutoFocus::OnROI_Height(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set((long)roiHeight_);
   }
   else if (eAct == MM::AfterSet)
   {
      long value;
      pProp->Get(value);
      roiHeight_ = (unsigned)value;
      return SetCameraROI();
   }
   return DEVICE_OK;
}

int AutoFocus::OnBinning(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(binning_);
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(binning_);
      return SetCameraBinning();
   }
   return DEVICE_OK;
}

int AutoFocus::OnExposure(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(exposureMs_);
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(exposureMs_);
   }
   return DEVICE_OK;
}

int AutoFocus::OnDeviceSettings(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(deviceSettings_);
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(deviceSettings_);

      // Load all properties for this device setting if it exists in calibration map
      if (calibrationMap_.find(deviceSettings_) != calibrationMap_.end())
      {
         CalibrationData& cal = calibrationMap_[deviceSettings_];

         // Update description
         deviceSettingsDescription_ = cal.description;
         GetCoreCallback()->OnPropertyChanged(this, g_SettingsDescription, deviceSettingsDescription_.c_str());

         // Update spot selection
         spotSelection_ = cal.spotSelection;
         GetCoreCallback()->OnPropertyChanged(this, "Cal_SpotSelection", spotSelection_.c_str());

         // Update precision
         precision_ = cal.precision;
         GetCoreCallback()->OnPropertyChanged(this, "Cal_Precision", CDeviceUtils::ConvertToString(precision_));

         // Update maxZ
         maxZ_ = cal.maxZ;
         GetCoreCallback()->OnPropertyChanged(this, "Cal_MaxZ", CDeviceUtils::ConvertToString(maxZ_));

         // Update and apply ROI settings
         roiX_ = cal.roiX;
         roiY_ = cal.roiY;
         roiWidth_ = cal.roiWidth;
         roiHeight_ = cal.roiHeight;
         SetCameraROI();
         GetCoreCallback()->OnPropertyChanged(this, "Camera_ROI-X", CDeviceUtils::ConvertToString((long)roiX_));
         GetCoreCallback()->OnPropertyChanged(this, "Camera_ROI-Y", CDeviceUtils::ConvertToString((long)roiY_));
         GetCoreCallback()->OnPropertyChanged(this, "Camera_ROI-Width", CDeviceUtils::ConvertToString((long)roiWidth_));
         GetCoreCallback()->OnPropertyChanged(this, "Camera_ROI-Height", CDeviceUtils::ConvertToString((long)roiHeight_));

         // Update and apply binning
         binning_ = cal.binning;
         SetCameraBinning();
         GetCoreCallback()->OnPropertyChanged(this, "Camera_Binning", CDeviceUtils::ConvertToString(binning_));

         // Update exposure
         exposureMs_ = cal.exposureMs;
         GetCoreCallback()->OnPropertyChanged(this, "Camera_Exposure_Ms", CDeviceUtils::ConvertToString(exposureMs_));
      }
      else
      {
         // No calibration data for this setting - clear description but leave other properties
         deviceSettingsDescription_ = "";
         GetCoreCallback()->OnPropertyChanged(this, g_SettingsDescription, deviceSettingsDescription_.c_str());
      }
   }
   return DEVICE_OK;
}

int AutoFocus::OnSpotSelection(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(spotSelection_.c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(spotSelection_);
   }
   return DEVICE_OK;
}

int AutoFocus::OnPrecision(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(precision_);
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(precision_);
   }
   return DEVICE_OK;
}

int AutoFocus::OnMaxZ(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(maxZ_);
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(maxZ_);
   }
   return DEVICE_OK;
}

int AutoFocus::OnStatus(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(status_.c_str());
   }
   return DEVICE_OK;
}

int AutoFocus::OnDeviceSettingsDescription(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(deviceSettingsDescription_.c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      pProp->Get(deviceSettingsDescription_);
   }
   return DEVICE_OK;
}

int AutoFocus::OnCalibrate(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set("Idle");
   }
   else if (eAct == MM::AfterSet)
   {
      std::string value;
      pProp->Get(value);
      if (value == "Start")
      {
         int ret = PerformCalibration();
         if (ret == DEVICE_OK)
         {
            ret = SaveCalibrationData();
         }
         // Reset the property to Idle
         pProp->Set("Idle");
         GetCoreCallback()->OnPropertyChanged(this, "Calibrate", "Idle");
         return ret;
      }
   }
   return DEVICE_OK;
}

int AutoFocus::OnMeasureOffset(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set("Idle");
   }
   else if (eAct == MM::AfterSet)
   {
      std::string value;
      pProp->Get(value);
      if (value == "Measure")
      {
         int ret = PerformMeasureOffset();
         // Reset the property to Idle
         pProp->Set("Idle");
         GetCoreCallback()->OnPropertyChanged(this, "MeasureOffset", "Idle");
         return ret;
      }
   }
   return DEVICE_OK;
}

bool AutoFocus::Busy()
{
   return false;
}

int AutoFocus::SetContinuousFocusing(bool on)
{
   {
      std::lock_guard<std::mutex> lock(continuousFocusMutex_);
      continuousFocusing_ = on;
   }

   if (on)
   {
      // Starting continuous focus - wake up thread
      continuousFocusLocked_ = false;
      continuousFocusCV_.notify_one();
   }
   else
   {
      // Stopping continuous focus - clear locked status
      continuousFocusLocked_ = false;
      UpdateStatus("Off");
   }

   return DEVICE_OK;
}

int AutoFocus::GetContinuousFocusing(bool& on)
{
   on = continuousFocusing_;
   return DEVICE_OK;
}

bool AutoFocus::IsContinuousFocusLocked()
{
   return continuousFocusLocked_;
}

int AutoFocus::SetOffset(double offset)
{
   offset_ = offset;
   NotifyRegisteredStages();
   return DEVICE_OK;
}

int AutoFocus::GetOffset(double& offset)
{
   offset = offset_;
   return DEVICE_OK;
}

void AutoFocus::RegisterStage(MM::Stage* stage)
{
   if (stage == nullptr)
      return;

   std::lock_guard<std::mutex> lock(registrationMutex_);

   // Check if already registered
   for (size_t i = 0; i < registeredStages_.size(); ++i)
   {
      if (registeredStages_[i] == stage)
         return;  // Already registered
   }

   registeredStages_.push_back(stage);
}

void AutoFocus::UnregisterStage(MM::Stage* stage)
{
   if (stage == nullptr)
      return;

   std::lock_guard<std::mutex> lock(registrationMutex_);

   for (size_t i = 0; i < registeredStages_.size(); ++i)
   {
      if (registeredStages_[i] == stage)
      {
         registeredStages_.erase(registeredStages_.begin() + i);
         break;
      }
   }
}

void AutoFocus::NotifyRegisteredStages()
{
   std::lock_guard<std::mutex> lock(registrationMutex_);

   for (size_t i = 0; i < registeredStages_.size(); ++i)
   {
      MM::Stage* pStage = registeredStages_[i];
      if (pStage != nullptr)
      {
         GetCoreCallback()->OnStagePositionChanged(pStage, offset_);
      }
   }
}

int AutoFocus::FullFocus()
{
   MM::Stage* pStage = static_cast<MM::Stage*>(GetDevice(focusStage_.c_str()));
   if (pStage == nullptr)
      return ERR_NO_PHYSICAL_STAGE;

   double currentZ;
   int ret = pStage->GetPositionUm(currentZ);
   if (ret != DEVICE_OK)
      return ret;

   // Step 1: Capture image and analyze spot position
   ret = SnapAndAnalyze();
   if (ret != DEVICE_OK)
      return ret;

   // Step 2: Check if spot was detected
   if (lastSpotScore_ <= 0.0)
      return DEVICE_ERR;  // No spot detected

   // Step 3: Check if we have calibration data for current device settings
   if (calibrationMap_.find(deviceSettings_) == calibrationMap_.end())
      return ERR_NOT_CALIBRATED;  // No calibration data available

   CalibrationData cal = calibrationMap_[deviceSettings_];

   // Iterative focusing loop to achieve desired precision
   const int maxIterations = 10;  // Prevent infinite loops
   int iteration = 0;

   while (iteration < maxIterations)
   {
      // Step 4: Calculate target Z position using helper function
      double diffZ = CalculateTargetZDiff(cal, lastSpotX_, lastSpotY_);

      // Step 5: Apply user-defined offset
      diffZ += offset_;

      // Step 6: Check if we've achieved the desired precision
      if (fabs(diffZ) <= cal.precision)
      {
         // Within precision, we're done
         return DEVICE_OK;
      }

      // Step 7: Validate Z position against stage limits and maxZ
      double targetZ = currentZ + diffZ;
      if (targetZ > maxZ_)
         return ERR_TAGET_TOO_HIGH;  // Would exceed maximum Z limit

      ret = ValidateZPosition(targetZ);
      if (ret != DEVICE_OK)
         return ret;

      // Step 8: Get focus stage and move to target position
      if (focusStage_ == "" || focusStage_ == "Undefined")
         return ERR_NO_PHYSICAL_STAGE;

      ret = pStage->SetPositionUm(targetZ);
      if (ret != DEVICE_OK)
         return ret;

      // Step 9: Wait for stage to settle
      while (pStage->Busy())
      {
         CDeviceUtils::SleepMs(10);
      }

      // Step 10: Update current Z position
      ret = pStage->GetPositionUm(currentZ);
      if (ret != DEVICE_OK)
         return ret;

      // Step 11: Capture new image and analyze spot position
      ret = SnapAndAnalyze();
      if (ret != DEVICE_OK)
         return ret;

      // Check if spot is still detected
      if (lastSpotScore_ <= 0.0)
         return DEVICE_ERR;  // Lost the spot

      iteration++;
   }

   // If we reached max iterations without achieving precision, still return OK
   // (we tried our best)
   return DEVICE_OK;
}

int AutoFocus::IncrementalFocus()
{
   return SnapAndAnalyze();
}

void AutoFocus::ContinuousFocusThread()
{
   while (!stopThread_)
   {
      // Check if continuous focusing is enabled
      bool shouldFocus = false;
      {
         std::lock_guard<std::mutex> lock(continuousFocusMutex_);
         shouldFocus = continuousFocusing_;
      }

      if (!shouldFocus)
      {
         // Wait for signal to start continuous focusing
         std::unique_lock<std::mutex> lock(continuousFocusMutex_);
         continuousFocusCV_.wait_for(lock, std::chrono::milliseconds(100));
         continue;
      }

      // Perform one continuous focus iteration
      try
      {
         // Step 1: Capture and analyze
         int ret = SnapAndAnalyze();
         if (ret != DEVICE_OK || lastSpotScore_ <= 0.0)
         {
            continuousFocusLocked_ = false;
            UpdateStatus("Out of Range");
            CDeviceUtils::SleepMs(50);
            continue;
         }

         // Step 2: Check if we have calibration
         CalibrationData cal;
         {
            std::lock_guard<std::mutex> lock(continuousFocusMutex_);
            if (calibrationMap_.find(deviceSettings_) == calibrationMap_.end())
            {
               continuousFocusLocked_ = false;
               UpdateStatus("Out of Range");
               CDeviceUtils::SleepMs(50);
               continue;
            }
            cal = calibrationMap_[deviceSettings_];
         }

         // Step 3: Calculate Z correction
         double diffZ = CalculateTargetZDiff(cal, lastSpotX_, lastSpotY_);
         diffZ += offset_;

         // Step 4: Check if locked (within precision)
         if (fabs(diffZ) <= cal.precision)
         {
            continuousFocusLocked_ = true;
            UpdateStatus("Locked");
            CDeviceUtils::SleepMs(50);  // Locked, check less frequently
            continue;
         }

         // Step 5: Not locked, need to adjust
         continuousFocusLocked_ = false;
         UpdateStatus("Focusing");

         // Step 6: Get stage and move
         MM::Stage* pStage = static_cast<MM::Stage*>(GetDevice(focusStage_.c_str()));
         if (pStage == nullptr)
         {
            CDeviceUtils::SleepMs(50);
            continue;
         }

         double currentZ;
         ret = pStage->GetPositionUm(currentZ);
         if (ret != DEVICE_OK)
         {
            CDeviceUtils::SleepMs(50);
            continue;
         }

         double targetZ = currentZ + diffZ;

         // Check against maxZ limit
         if (targetZ > maxZ_)
         {
            CDeviceUtils::SleepMs(50);
            continue;
         }

         ret = ValidateZPosition(targetZ);
         if (ret != DEVICE_OK)
         {
            CDeviceUtils::SleepMs(50);
            continue;
         }

         ret = pStage->SetPositionUm(targetZ);
         if (ret == DEVICE_OK)
         {
            // Wait for stage to settle
            while (pStage->Busy())
            {
               CDeviceUtils::SleepMs(10);
            }
         }

         CDeviceUtils::SleepMs(20);  // Brief pause before next iteration
      }
      catch (...)
      {
         continuousFocusLocked_ = false;
         CDeviceUtils::SleepMs(100);
      }
   }
}

void AutoFocus::UpdateStatus(const std::string& newStatus)
{
   if (status_ != newStatus)
   {
      status_ = newStatus;
      OnPropertyChanged("Status", status_.c_str());
   }
}

int AutoFocus::GetLastFocusScore(double& score)
{
   score = lastSpotScore_;
   return DEVICE_OK;
}

int AutoFocus::GetCurrentFocusScore(double& score)
{
   // is this current?
   score = lastSpotScore_;
   return DEVICE_OK;
}

int AutoFocus::SnapAndAnalyze()
{
   // Get shutter device
   MM::Shutter* shutter = static_cast<MM::Shutter*>(GetDevice(shutter_.c_str()));
   if (shutter == nullptr)
      return ERR_NO_SHUTTER_DEVICE_FOUND;
   // Get camera device
   MM::Camera* camera = static_cast<MM::Camera*>(GetDevice(camera_.c_str()));
   if (camera == nullptr)
      return ERR_NO_PHYSICAL_CAMERA;

   // Set camera exposure
   camera->SetExposure(exposureMs_);

   // Close shutter to block IR light
   shutter->SetOpen(false);
   CDeviceUtils::SleepMs(10); // wait for shutter to close
   // Snap image with shutter closed
   camera->SnapImage();
   ImgBuffer darkImage;
   int ret = GetImageFromBuffer(darkImage);

   shutter->SetOpen(true);
   CDeviceUtils::SleepMs(10); // wait for shutter to open
   // Snap image with shutter open
   camera->SnapImage();
   ImgBuffer lightImage;
   ret = GetImageFromBuffer(lightImage);
   shutter->SetOpen(false);

   // Subtract darkImage from lightImage
   ImgBuffer resultImage(lightImage.Width(), lightImage.Height(), lightImage.Depth());
   const unsigned char* lightPixels = lightImage.GetPixels();
   const unsigned char* darkPixels = darkImage.GetPixels();
   unsigned int numPixels = lightImage.Width() * lightImage.Height();

   if (lightImage.Depth() == 1)
   {
      // 8-bit images
      std::vector<unsigned char> resultPixels(numPixels);
      for (unsigned int i = 0; i < numPixels; ++i)
      {
         int diff = (int)lightPixels[i] - (int)darkPixels[i];
         resultPixels[i] = (diff > 0) ? (unsigned char)diff : 0;
      }
      resultImage.SetPixels(resultPixels.data());
   }
   else if (lightImage.Depth() == 2)
   {
      // 16-bit images
      const unsigned short* lightPixels16 = (const unsigned short*)lightPixels;
      const unsigned short* darkPixels16 = (const unsigned short*)darkPixels;
      std::vector<unsigned short> resultPixels(numPixels);
      for (unsigned int i = 0; i < numPixels; ++i)
      {
         int diff = (int)lightPixels16[i] - (int)darkPixels16[i];
         resultPixels[i] = (diff > 0) ? (unsigned short)diff : 0;
      }
      resultImage.SetPixels((const unsigned char*)resultPixels.data());
   }

   double score1, x1, y1;
   double score2, x2, y2;
   AnalyzeImage(resultImage, score1, x1, y1, score2, x2, y2);

   // Select which spot to use based on calibration data
   // Default to spot 1 if no calibration data
   bool useSpot1 = true;

   if (calibrationMap_.find(deviceSettings_) != calibrationMap_.end())
   {
      CalibrationData cal = calibrationMap_[deviceSettings_];

      // we refer here to higher as higher in the image, i.e. higher x or y coordinate
      bool spot1IsHigher = (x1 > x2);
      if (cal.dominantAxis == 'Y')
         spot1IsHigher = (y1 > y2);

      // Use the spot selection that was stored during calibration
      if (cal.spotSelection == "Top")
      {
         // Use top surface spot
         useSpot1 = cal.slope1Dominant > 0 ? !spot1IsHigher : spot1IsHigher;
      }
      else if (cal.spotSelection == "Bottom")
      {
         // Use bottom surface spot
         useSpot1 = cal.slope1Dominant <= 0 ? !spot1IsHigher : spot1IsHigher;
      }
   }

   // Store the selected spot for use by FullFocus()
   if (useSpot1)
   {
      lastSpotX_ = x1;
      lastSpotY_ = y1;
      lastSpotScore_ = score1;
   }
   else if (score2 > 0)  // Make sure spot 2 exists
   {
      lastSpotX_ = x2;
      lastSpotY_ = y2;
      lastSpotScore_ = score2;
   }
   else
   {
      // Fallback to spot 1 if spot 2 doesn't exist
      lastSpotX_ = x1;
      lastSpotY_ = y1;
      lastSpotScore_ = score1;
   }

   // Here we would implement the logic to adjust focus based on scores
   // For now, we just return OK
   return DEVICE_OK;
}


int AutoFocus::GetImageFromBuffer(ImgBuffer& img)
{
   // Get camera device
   MM::Camera* camera = static_cast<MM::Camera*>(GetDevice(camera_.c_str()));
   if (camera == nullptr)
      return ERR_NO_PHYSICAL_CAMERA;

   const unsigned char* imgBuffer = camera->GetImageBuffer();
   if (imgBuffer == nullptr)
      return ERR_NO_PHYSICAL_CAMERA;

   unsigned int width = camera->GetImageWidth();
   unsigned int height = camera->GetImageHeight();
   unsigned int byteDepth = camera->GetImageBytesPerPixel();
   img = ImgBuffer(width, height, byteDepth);
   img.SetPixels(imgBuffer);

   return DEVICE_OK;
}

int AutoFocus::AnalyzeImage(ImgBuffer img, double& score1, double& x1, double& y1, double& score2, double& x2, double& y2)
{
   // Find up to two spots in the image, return their (x,y) coordinates and scores

   const unsigned int width = img.Width();
   const unsigned int height = img.Height();
   const unsigned int pixDepth = img.Depth();
   const unsigned char* pixels = img.GetPixels();

   if (pixels == nullptr || width == 0 || height == 0)
   {
      score1 = score2 = 0.0;
      x1 = y1 = x2 = y2 = 0.0;
      return DEVICE_OK;
   }

   // Calculate threshold using Otsu's method or use a simple percentile-based threshold
   // For bright spots on dark background, we'll use a high percentile threshold
   std::vector<unsigned char> intensities;
   intensities.reserve(width * height);

   if (pixDepth == 1)
   {
      // 8-bit images: copy directly
      for (unsigned int i = 0; i < width * height; ++i)
      {
         intensities.push_back(pixels[i]);
      }
   }
   else if (pixDepth == 2)
   {
      // 16-bit images: find max value and scale to 8-bit range
      const unsigned short* pixels16 = (const unsigned short*)pixels;
      unsigned short maxValue = 0;

      // First pass: find maximum value
      for (unsigned int i = 0; i < width * height; ++i)
      {
         if (pixels16[i] > maxValue)
            maxValue = pixels16[i];
      }

      // Second pass: scale to 8-bit range
      if (maxValue > 0)
      {
         double scale = 255.0 / maxValue;
         for (unsigned int i = 0; i < width * height; ++i)
         {
            unsigned char scaledValue = static_cast<unsigned char>(pixels16[i] * scale);
            intensities.push_back(scaledValue);
         }
      }
      else
      {
         // All pixels are zero - no spots to find
         score1 = score2 = 0.0;
         x1 = y1 = x2 = y2 = 0.0;
         return DEVICE_OK;
      }
   }

   // Find threshold as 95th percentile to isolate bright spots
   std::vector<unsigned char> sorted = intensities;
   std::sort(sorted.begin(), sorted.end());
   unsigned char threshold = sorted[static_cast<size_t>(sorted.size() * 0.99)];

   // Label connected components using flood fill
   std::vector<int> labels(width * height, -1);
   int currentLabel = 0;

   struct Spot {
      double sumX = 0.0;
      double sumY = 0.0;
      double sumIntensity = 0.0;
      double sumX2 = 0.0; // For calculating variance/spread
      double sumY2 = 0.0;
      int pixelCount = 0;
      unsigned char maxIntensity = 0;
   };

   std::vector<Spot> spots;

   // Flood fill to find connected components
   for (unsigned int y = 0; y < height; ++y)
   {
      for (unsigned int x = 0; x < width; ++x)
      {
         unsigned int idx = y * width + x;
         unsigned char intensity = intensities[idx];

         if (intensity >= threshold && labels[idx] == -1)
         {
            // Start new component
            spots.push_back(Spot());
            std::vector<std::pair<unsigned int, unsigned int>> stack;
            stack.push_back(std::make_pair(x, y));
            labels[idx] = currentLabel;

            while (!stack.empty())
            {
               std::pair<unsigned int, unsigned int> pos = stack.back();
               stack.pop_back();
               unsigned int px = pos.first;
               unsigned int py = pos.second;
               unsigned int pidx = py * width + px;
               unsigned char pIntensity = intensities[pidx];

               // Add to spot statistics
               spots[currentLabel].sumX += px * pIntensity;
               spots[currentLabel].sumY += py * pIntensity;
               spots[currentLabel].sumIntensity += pIntensity;
               spots[currentLabel].sumX2 += px * px * pIntensity;
               spots[currentLabel].sumY2 += py * py * pIntensity;
               spots[currentLabel].pixelCount++;
               if (pIntensity > spots[currentLabel].maxIntensity)
                  spots[currentLabel].maxIntensity = pIntensity;

               // Check 8-connected neighbors
               for (int dy = -1; dy <= 1; ++dy)
               {
                  for (int dx = -1; dx <= 1; ++dx)
                  {
                     if (dx == 0 && dy == 0) continue;

                     int nx = px + dx;
                     int ny = py + dy;

                     if (nx >= 0 && nx < (int)width && ny >= 0 && ny < (int)height)
                     {
                        unsigned int nidx = ny * width + nx;
                        if (intensities[nidx] >= threshold && labels[nidx] == -1)
                        {
                           labels[nidx] = currentLabel;
                           stack.push_back(std::make_pair(nx, ny));
                        }
                     }
                  }
               }
            }
            currentLabel++;
         }
      }
   }

   // Calculate centroid, variance, and score for each spot
   struct SpotResult {
      double x = 0.0;
      double y = 0.0;
      double score = 0.0;
      double totalIntensity = 0.0;
   };

   std::vector<SpotResult> results;
   for (size_t i = 0; i < spots.size(); ++i)
   {
      if (spots[i].sumIntensity > 0)
      {
         SpotResult result;
         result.x = spots[i].sumX / spots[i].sumIntensity;
         result.y = spots[i].sumY / spots[i].sumIntensity;
         result.totalIntensity = spots[i].sumIntensity;

         // Calculate variance (spread) of the spot
         double varX = (spots[i].sumX2 / spots[i].sumIntensity) - (result.x * result.x);
         double varY = (spots[i].sumY2 / spots[i].sumIntensity) - (result.y * result.y);
         double spread = sqrt(varX + varY);

         // Score: smaller spread = better focus = higher score
         // Use inverse of spread, normalized by total intensity
         if (spread > 0.0)
            result.score = spots[i].sumIntensity / (spread * spread);
         else
            result.score = spots[i].sumIntensity * 1000.0; // Very small spot

         results.push_back(result);
      }
   }

   // Sort spots by intensity
   std::sort(results.begin(), results.end(),
      [](const SpotResult& a, const SpotResult& b) {
         return a.totalIntensity > b.totalIntensity;
      });

   // Return top 2 spots, highest score first
   if (results.size() >= 2)
   {
      if (results[0].score >= results[1].score)
      {
         x1 = results[0].x;
         y1 = results[0].y;
         score1 = results[0].score;
         x2 = results[1].x;
         y2 = results[1].y;
         score2 = results[1].score;
      }
      else
      {
         x1 = results[1].x;
         y1 = results[1].y;
         score1 = results[1].score;
         x2 = results[0].x;
         y2 = results[0].y;
         score2 = results[0].score;
      }
   }
   else if (results.size() == 1)
   {
      x1 = results[0].x;
      y1 = results[0].y;
      score1 = results[0].score;
      x2 = y2 = score2 = 0.0;
   }
   else
   {
      x1 = y1 = score1 = 0.0;
   }

   return DEVICE_OK;
}

int AutoFocus::SetCameraBinning()
{
   if (camera_ == "" || camera_ == "Undefined")
   {
      // even though we are not setting binning, things get complicated if we return an error here
      return DEVICE_OK;
   }
   MM::Camera* pCam = static_cast<MM::Camera*>(GetDevice(camera_.c_str()));
   if (pCam == nullptr)
   {
      return ERR_NO_PHYSICAL_CAMERA;
   }
   // Apply current settings, if that fails, read them from the camera
   int ret = pCam->SetBinning(binning_);
   if (ret != DEVICE_OK)
   {
      // Get current binning
      binning_ = pCam->GetBinning();
      if (binning_ <= 0)
         binning_ = 1;
      GetCoreCallback()->OnPropertyChanged(this, "Binning", CDeviceUtils::ConvertToString(binning_));
   }
   return DEVICE_OK;
}


int AutoFocus::SetCameraROI()
{
   if (camera_ == "" || camera_ == "Undefined")
   {
      // even though we are not setting binning, things get complicated if we return an error here
      return DEVICE_OK;
   }
   MM::Camera* pCam = static_cast<MM::Camera*>(GetDevice(camera_.c_str()));
   if (pCam == nullptr)
   {
      return ERR_NO_PHYSICAL_CAMERA;
   }
   int ret = pCam->SetROI(roiX_, roiY_, roiWidth_, roiHeight_);
   if (ret != DEVICE_OK)
   {
      unsigned x, y, xSize, ySize;
      ret = pCam->GetROI(x, y, xSize, ySize);
      if (ret == DEVICE_OK)
      {
         // If no ROI is set, use full frame
         if (xSize == 0 || ySize == 0)
         {
            roiWidth_ = pCam->GetImageWidth();
            roiHeight_ = pCam->GetImageHeight();
            roiX_ = 0;
            roiY_ = 0;
         }
         else
         {
            roiX_ = x;
            roiY_ = y;
            roiWidth_ = xSize;
            roiHeight_ = ySize;

         }
         GetCoreCallback()->OnPropertyChanged(this, "ROI-X", CDeviceUtils::ConvertToString((long)roiX_));
         GetCoreCallback()->OnPropertyChanged(this, "ROI-Y", CDeviceUtils::ConvertToString((long)roiY_));
         GetCoreCallback()->OnPropertyChanged(this, "ROI-Width", CDeviceUtils::ConvertToString((long)roiWidth_));
         GetCoreCallback()->OnPropertyChanged(this, "ROI-Height", CDeviceUtils::ConvertToString((long)roiHeight_));
      }
   }
   return ret;
}

int AutoFocus::PerformCalibration()
{
   // Get focus stage device
   if (focusStage_ == "" || focusStage_ == "Undefined")
   {
      return ERR_NO_PHYSICAL_STAGE;
   }

   MM::Stage* pStage = static_cast<MM::Stage*>(GetDevice(focusStage_.c_str()));
   if (pStage == nullptr)
   {
      return ERR_NO_PHYSICAL_STAGE;
   }

   // Store the starting position
   double originalPos;
   int ret = pStage->GetPositionUm(originalPos);
   if (ret != DEVICE_OK)
      return ret;

   // Calibration parameters
   const int numSteps = 20;
   const double stepSize = precision_; // use precision property value
   std::vector<double> zPositions;
   std::vector<double> spotXPositions;
   std::vector<double> spotYPositions;
   // Track second spot as well
   std::vector<double> spot2XPositions;
   std::vector<double> spot2YPositions;
   double startPos = originalPos - (numSteps / 2) * stepSize;
   
   // Get shutter and camera
   MM::Shutter* shutter = static_cast<MM::Shutter*>(GetDevice(shutter_.c_str()));
   MM::Camera* camera = static_cast<MM::Camera*>(GetDevice(camera_.c_str()));
   if (shutter == nullptr || camera == nullptr)
   {
      pStage->SetPositionUm(startPos);
      return ERR_NO_PHYSICAL_CAMERA;
   }

   // Set camera exposure
   camera->SetExposure(exposureMs_);

   // Close shutter and take dark image
   shutter->SetOpen(false);
   CDeviceUtils::SleepMs(10);
   camera->SnapImage();
   ImgBuffer darkImage;
   ret = GetImageFromBuffer(darkImage);

   // Open shutter and take light image
   shutter->SetOpen(true);
   CDeviceUtils::SleepMs(10);
   camera->SnapImage();
   ImgBuffer lightImage;
   ret = GetImageFromBuffer(lightImage);

   // Subtract darkImage from lightImage
   ImgBuffer resultImage(lightImage.Width(), lightImage.Height(), lightImage.Depth());
   const unsigned char* lightPixels = lightImage.GetPixels();
   const unsigned char* darkPixels = darkImage.GetPixels();
   unsigned int numPixels = lightImage.Width() * lightImage.Height();

   if (lightImage.Depth() == 1)
   {
      // 8-bit images
      std::vector<unsigned char> resultPixels(numPixels);
      for (unsigned int i = 0; i < numPixels; ++i)
      {
         int diff = (int)lightPixels[i] - (int)darkPixels[i];
         resultPixels[i] = (diff > 0) ? (unsigned char)diff : 0;
      }
      resultImage.SetPixels(resultPixels.data());
   }
   else if (lightImage.Depth() == 2)
   {
      // 16-bit images
      const unsigned short* lightPixels16 = (const unsigned short*)lightPixels;
      const unsigned short* darkPixels16 = (const unsigned short*)darkPixels;
      std::vector<unsigned short> resultPixels(numPixels);
      for (unsigned int i = 0; i < numPixels; ++i)
      {
         int diff = (int)lightPixels16[i] - (int)darkPixels16[i];
         resultPixels[i] = (diff > 0) ? (unsigned short)diff : 0;
      }
      resultImage.SetPixels((const unsigned char*)resultPixels.data());
   }

   // Analyze to find spot position
   double score1, x1, y1, score2, x2, y2;
   ret = AnalyzeImage(resultImage, score1, x1, y1, score2, x2, y2);

   // Store reference spot position at the starting (in-focus) position
   double refSpotXMeasured = x1;
   double refSpotYMeasured = y1;

   // Initialize previous positions for tracking continuity
   double prevSpot1X = x1;
   double prevSpot1Y = y1;
   double prevSpot2X = x2;
   double prevSpot2Y = y2;
   bool hasSpot2 = (score2 > 0);

   // Collect calibration data
   for (int i = 0; i < numSteps; i++)
   {
      // Move to new position
      double targetPos = startPos + i * stepSize;
      ret = pStage->SetPositionUm(targetPos);
      if (ret != DEVICE_OK)
      {
         // Try to restore original position
         pStage->SetPositionUm(startPos);
         return ret;
      }

      // Wait for stage to settle
      while (pStage->Busy())
      {
         CDeviceUtils::SleepMs(10);
      }

      // Close shutter and take dark image
      shutter->SetOpen(false);
      CDeviceUtils::SleepMs(10);
      camera->SnapImage();
      ImgBuffer darkImageLoop;
      ret = GetImageFromBuffer(darkImageLoop);

      // Open shutter and take light image
      shutter->SetOpen(true);
      CDeviceUtils::SleepMs(10);
      camera->SnapImage();
      ImgBuffer lightImageLoop;
      ret = GetImageFromBuffer(lightImageLoop);

      // Subtract darkImage from lightImage
      ImgBuffer resultImageLoop(lightImageLoop.Width(), lightImageLoop.Height(), lightImageLoop.Depth());
      const unsigned char* lightPixelsLoop = lightImageLoop.GetPixels();
      const unsigned char* darkPixelsLoop = darkImageLoop.GetPixels();
      unsigned int numPixelsLoop = lightImageLoop.Width() * lightImageLoop.Height();

      if (lightImageLoop.Depth() == 1)
      {
         // 8-bit images
         std::vector<unsigned char> resultPixelsLoop(numPixelsLoop);
         for (unsigned int j = 0; j < numPixelsLoop; ++j)
         {
            int diff = (int)lightPixelsLoop[j] - (int)darkPixelsLoop[j];
            resultPixelsLoop[j] = (diff > 0) ? (unsigned char)diff : 0;
         }
         resultImageLoop.SetPixels(resultPixelsLoop.data());
      }
      else if (lightImageLoop.Depth() == 2)
      {
         // 16-bit images
         const unsigned short* lightPixels16Loop = (const unsigned short*)lightPixelsLoop;
         const unsigned short* darkPixels16Loop = (const unsigned short*)darkPixelsLoop;
         std::vector<unsigned short> resultPixelsLoop(numPixelsLoop);
         for (unsigned int j = 0; j < numPixelsLoop; ++j)
         {
            int diff = (int)lightPixels16Loop[j] - (int)darkPixels16Loop[j];
            resultPixelsLoop[j] = (diff > 0) ? (unsigned short)diff : 0;
         }
         resultImageLoop.SetPixels((const unsigned char*)resultPixelsLoop.data());
      }

      // Analyze to find spot position
      ret = AnalyzeImage(resultImageLoop, score1, x1, y1, score2, x2, y2);
      if (ret != DEVICE_OK)
      {
         pStage->SetPositionUm(startPos);
         return ret;
      }

      // Track spots based on spatial continuity (minimum distance)
      if (score1 > 0)
      {
         double track1X = 0.0;
         double track1Y = 0.0;
         double track2X = 0.0;
         double track2Y = 0.0;;
         bool haveTrack2 = false;

         if (hasSpot2 && score2 > 0)
         {
            // Have two spots - need to match them to previous positions
            // Calculate distances from each detected spot to each tracked spot
            double dist_1to1 = sqrt((x1 - prevSpot1X)*(x1 - prevSpot1X) +
                                    (y1 - prevSpot1Y)*(y1 - prevSpot1Y));
            double dist_1to2 = sqrt((x1 - prevSpot2X)*(x1 - prevSpot2X) +
                                    (y1 - prevSpot2Y)*(y1 - prevSpot2Y));
            double dist_2to1 = sqrt((x2 - prevSpot1X)*(x2 - prevSpot1X) +
                                    (y2 - prevSpot1Y)*(y2 - prevSpot1Y));
            double dist_2to2 = sqrt((x2 - prevSpot2X)*(x2 - prevSpot2X) +
                                    (y2 - prevSpot2Y)*(y2 - prevSpot2Y));

            // Match spots using minimum total distance
            // Option A: spot1->track1, spot2->track2
            double totalDistA = dist_1to1 + dist_2to2;
            // Option B: spot1->track2, spot2->track1
            double totalDistB = dist_1to2 + dist_2to1;

            if (totalDistA <= totalDistB)
            {
               // No swap: spot1 stays as track1, spot2 stays as track2
               track1X = x1; track1Y = y1;
               track2X = x2; track2Y = y2;
            }
            else
            {
               // Swap: spot1 becomes track2, spot2 becomes track1
               track1X = x2; track1Y = y2;
               track2X = x1; track2Y = y1;
            }
            haveTrack2 = true;
         }
         else
         {
            // Only one spot detected
            track1X = x1; track1Y = y1;
            haveTrack2 = false;
         }

         // Store tracked positions
         zPositions.push_back(targetPos);
         spotXPositions.push_back(track1X);
         spotYPositions.push_back(track1Y);

         if (haveTrack2)
         {
            spot2XPositions.push_back(track2X);
            spot2YPositions.push_back(track2Y);
         }

         // Update previous positions for next iteration
         prevSpot1X = track1X;
         prevSpot1Y = track1Y;
         if (haveTrack2)
         {
            prevSpot2X = track2X;
            prevSpot2Y = track2Y;
         }
      }
   }

   // Return to starting position
   pStage->SetPositionUm(originalPos);

   // Calculate linear fit using least squares
   if (zPositions.size() < 3)
   {
      return DEVICE_ERR; // Not enough data points
   }

   // Calculate means
   double meanZ = 0.0, meanX = 0.0, meanY = 0.0;
   for (size_t i = 0; i < zPositions.size(); i++)
   {
      meanZ += zPositions[i];
      meanX += spotXPositions[i];
      meanY += spotYPositions[i];
   }
   meanZ /= zPositions.size();
   meanX /= zPositions.size();
   meanY /= zPositions.size();

   // Calculate slopes using least squares
   double numeratorX = 0.0, numeratorY = 0.0, denominator = 0.0;
   for (size_t i = 0; i < zPositions.size(); i++)
   {
      double zDiff = zPositions[i] - meanZ;
      numeratorX += zDiff * (spotXPositions[i] - meanX);
      numeratorY += zDiff * (spotYPositions[i] - meanY);
      denominator += zDiff * zDiff;
   }

   if (denominator == 0.0)
   {
      return DEVICE_ERR; // No variation in Z
   }

   CalibrationData cal;
   cal.slopeX = numeratorX / denominator;
   cal.slopeY = numeratorY / denominator;
   cal.offsetX = meanX - cal.slopeX * meanZ;
   cal.offsetY = meanY - cal.slopeY * meanZ;

   // Calculate slopes for spot 2 if we have data
   double slope2X = 0.0, slope2Y = 0.0;
   if (spot2XPositions.size() >= 3)
   {
      double mean2X = 0.0, mean2Y = 0.0;
      for (size_t i = 0; i < spot2XPositions.size(); i++)
      {
         mean2X += spot2XPositions[i];
         mean2Y += spot2YPositions[i];
      }
      mean2X /= spot2XPositions.size();
      mean2Y /= spot2YPositions.size();

      double numerator2X = 0.0, numerator2Y = 0.0, denominator2 = 0.0;
      for (size_t i = 0; i < spot2XPositions.size(); i++)
      {
         double zDiff = zPositions[i] - meanZ;
         numerator2X += zDiff * (spot2XPositions[i] - mean2X);
         numerator2Y += zDiff * (spot2YPositions[i] - mean2Y);
         denominator2 += zDiff * zDiff;
      }

      if (denominator2 > 0.0)
      {
         slope2X = numerator2X / denominator2;
         slope2Y = numerator2Y / denominator2;
      }
   }

   // Determine dominant axis (which axis has larger movement)
   double absSlope1X = fabs(cal.slopeX);
   double absSlope1Y = fabs(cal.slopeY);
   double absSlope2X = fabs(slope2X);
   double absSlope2Y = fabs(slope2Y);

   // Average the slopes to determine which axis moves more
   double avgSlopeX = (absSlope1X + absSlope2X) / 2.0;
   double avgSlopeY = (absSlope1Y + absSlope2Y) / 2.0;

   // Establish dominant axis
   // Also determine which spot is top based on movement direction
   if (avgSlopeX > avgSlopeY)
   {
      // X is dominant axis
      cal.dominantAxis = 'X';
      cal.slope1Dominant = cal.slopeX;
      cal.slope2Dominant = slope2X;
   }
   else
   {
      // Y is dominant axis
      cal.dominantAxis = 'Y';
      cal.slope1Dominant = cal.slopeY;
      cal.slope2Dominant = slope2Y;
   }

   // Store reference Z position (where calibration started - in focus)
   cal.refZ = startPos;

   // Use the actual measured spot positions at the in-focus starting position
   // This is what we want the spot to return to during autofocus
   cal.refSpotX = refSpotXMeasured;
   cal.refSpotY = refSpotYMeasured;

   // Store the spot selection that was used during this calibration
   cal.spotSelection = spotSelection_;

   // Store the precision that was set during this calibration
   cal.precision = precision_;

   // Store the description that was set for this device setting
   cal.description = deviceSettingsDescription_;

   // Store the maximum Z position
   cal.maxZ = maxZ_;

   // Store the camera settings (ROI and binning) used during calibration
   cal.roiX = roiX_;
   cal.roiY = roiY_;
   cal.roiWidth = roiWidth_;
   cal.roiHeight = roiHeight_;
   cal.binning = binning_;
   cal.exposureMs = exposureMs_;

   // Store calibration for current device setting
   calibrationMap_[deviceSettings_] = cal;

   return DEVICE_OK;
}

int AutoFocus::PerformMeasureOffset()
{
   // Step 1: Check if we have calibration data
   if (calibrationMap_.find(deviceSettings_) == calibrationMap_.end())
   {
      return DEVICE_ERR;  // No calibration data available
   }

   CalibrationData cal = calibrationMap_[deviceSettings_];

   // Step 2: Get current Z position from stage
   if (focusStage_ == "" || focusStage_ == "Undefined")
   {
      return ERR_NO_PHYSICAL_STAGE;
   }

   MM::Stage* pStage = static_cast<MM::Stage*>(GetDevice(focusStage_.c_str()));
   if (pStage == nullptr)
   {
      return ERR_NO_PHYSICAL_STAGE;
   }

   double currentZ;
   int ret = pStage->GetPositionUm(currentZ);
   if (ret != DEVICE_OK)
   {
      return ret;
   }

   // Step 3: Snap and analyze image to get current spot position
   ret = SnapAndAnalyze();
   if (ret != DEVICE_OK)
   {
      return ret;
   }

   // Step 4: Check if spot was detected
   if (lastSpotScore_ <= 0.0)
   {
      return DEVICE_ERR;  // No spot detected
   }

   // Step 5: Calculate what Z correction would be needed
   double diffZ = CalculateTargetZDiff(cal, lastSpotX_, lastSpotY_);

   // Step 6: Set offset to negative of diffZ to make current position "in focus"
   // This works because: diffZ + offset = diffZ + (-diffZ) = 0
   offset_ = -diffZ;

   // Step 7: Notify the system of the new offset value
   GetCoreCallback()->OnPropertyChanged(this, "Offset", CDeviceUtils::ConvertToString(offset_));
   NotifyRegisteredStages();

   return DEVICE_OK;
}

double AutoFocus::CalculateTargetZDiff(const CalibrationData& cal, double spotX, double spotY)
{
   const double MIN_SLOPE = 1e-6;  // Threshold for effectively zero slope
   const double INVALID_Z = -1.0e10;  // Sentinel value

   // bool xValid = fabs(cal.slopeX) > MIN_SLOPE;
   bool xValid = fabs(cal.slopeX) > MIN_SLOPE;
   bool yValid = fabs(cal.slopeY) > MIN_SLOPE;

   if (!xValid && cal.dominantAxis == 'X')
      return INVALID_Z;  //  slope too small - unusable calibration
   if (!yValid && cal.dominantAxis == 'Y')
      return INVALID_Z;  //  slope too small - unusable calibration

   if (cal.dominantAxis == 'X')
      return -(spotX - cal.refSpotX) / cal.slopeX;
   else if (cal.dominantAxis == 'Y')
      return -(spotY - cal.refSpotY) / cal.slopeY;

   // error
   return INVALID_Z;
}

int AutoFocus::ValidateZPosition(double targetZ)
{
   if (focusStage_ == "" || focusStage_ == "Undefined")
      return ERR_NO_PHYSICAL_STAGE;

   MM::Stage* pStage = static_cast<MM::Stage*>(GetDevice(focusStage_.c_str()));
   if (pStage == nullptr)
      return ERR_NO_PHYSICAL_STAGE;

   double lowerLimit, upperLimit;
   int ret = pStage->GetLimits(lowerLimit, upperLimit);
   if (ret != DEVICE_OK)
      return DEVICE_OK;  // If can't get limits, assume OK

   if (targetZ < lowerLimit || targetZ > upperLimit)
      return ERR_POS_OUT_OF_RANGE;

   return DEVICE_OK;
}

int AutoFocus::SaveCalibrationData()
{
   // Simple JSON format without external library
   std::ofstream file("Util-Autofocus.json");
   if (!file.is_open())
   {
      return DEVICE_ERR;
   }

   file << "{\n";
   file << "  \"calibrations\": [\n";

   bool first = true;
   for (std::map<long, CalibrationData>::iterator it = calibrationMap_.begin();
        it != calibrationMap_.end(); ++it)
   {
      if (!first)
         file << ",\n";
      first = false;

      file << "    {\n";
      file << "      \"deviceSetting\": " << it->first << ",\n";
      file << "      \"offsetX\": " << it->second.offsetX << ",\n";
      file << "      \"offsetY\": " << it->second.offsetY << ",\n";
      file << "      \"slopeX\": " << it->second.slopeX << ",\n";
      file << "      \"slopeY\": " << it->second.slopeY << ",\n";
      file << "      \"refSpotX\": " << it->second.refSpotX << ",\n";
      file << "      \"refSpotY\": " << it->second.refSpotY << ",\n";
      file << "      \"refZ\": " << it->second.refZ << ",\n";
      file << "      \"dominantAxis\": \"" << it->second.dominantAxis << "\",\n";
      file << "      \"slope1Dominant\": " << it->second.slope1Dominant << ",\n";
      file << "      \"slope2Dominant\": " << it->second.slope2Dominant << ",\n";
      file << "      \"spotSelection\": \"" << it->second.spotSelection << "\",\n";
      file << "      \"precision\": " << it->second.precision << ",\n";

      // Escape quotes in description for JSON
      std::string escapedDesc = it->second.description;
      size_t pos = 0;
      while ((pos = escapedDesc.find("\"", pos)) != std::string::npos)
      {
         escapedDesc.replace(pos, 1, "\\\"");
         pos += 2;
      }
      file << "      \"description\": \"" << escapedDesc << "\",\n";
      file << "      \"maxZ\": " << it->second.maxZ << ",\n";

      // Save camera settings
      file << "      \"roiX\": " << it->second.roiX << ",\n";
      file << "      \"roiY\": " << it->second.roiY << ",\n";
      file << "      \"roiWidth\": " << it->second.roiWidth << ",\n";
      file << "      \"roiHeight\": " << it->second.roiHeight << ",\n";
      file << "      \"binning\": " << it->second.binning << ",\n";
      file << "      \"exposureMs\": " << it->second.exposureMs << "\n";
      file << "    }";
   }

   file << "\n  ]\n";
   file << "}\n";

   file.close();
   return DEVICE_OK;
}

int AutoFocus::LoadCalibrationData()
{
   std::ifstream file("Util-Autofocus.json");
   if (!file.is_open())
   {
      // File doesn't exist yet - not an error
      return DEVICE_OK;
   }

   calibrationMap_.clear();

   // Simple JSON parsing - look for closing bracket to know when object is complete
   std::string line;
   long currentDeviceSetting = -1;
   CalibrationData currentCal;

   while (std::getline(file, line))
   {
      size_t pos;

      // Look for deviceSetting
      if ((pos = line.find("\"deviceSetting\"")) != std::string::npos)
      {
         size_t colonPos = line.find(":", pos);
         if (colonPos != std::string::npos)
         {
            currentDeviceSetting = std::atol(line.substr(colonPos + 1).c_str());
         }
      }
      // Look for offsetX
      else if ((pos = line.find("\"offsetX\"")) != std::string::npos)
      {
         size_t colonPos = line.find(":", pos);
         if (colonPos != std::string::npos)
         {
            currentCal.offsetX = std::atof(line.substr(colonPos + 1).c_str());
         }
      }
      // Look for offsetY
      else if ((pos = line.find("\"offsetY\"")) != std::string::npos)
      {
         size_t colonPos = line.find(":", pos);
         if (colonPos != std::string::npos)
         {
            currentCal.offsetY = std::atof(line.substr(colonPos + 1).c_str());
         }
      }
      // Look for slopeX
      else if ((pos = line.find("\"slopeX\"")) != std::string::npos)
      {
         size_t colonPos = line.find(":", pos);
         if (colonPos != std::string::npos)
         {
            currentCal.slopeX = std::atof(line.substr(colonPos + 1).c_str());
         }
      }
      // Look for slopeY
      else if ((pos = line.find("\"slopeY\"")) != std::string::npos)
      {
         size_t colonPos = line.find(":", pos);
         if (colonPos != std::string::npos)
         {
            currentCal.slopeY = std::atof(line.substr(colonPos + 1).c_str());
         }
      }
      // Look for refSpotX
      else if ((pos = line.find("\"refSpotX\"")) != std::string::npos)
      {
         size_t colonPos = line.find(":", pos);
         if (colonPos != std::string::npos)
         {
            currentCal.refSpotX = std::atof(line.substr(colonPos + 1).c_str());
         }
      }
      // Look for refSpotY
      else if ((pos = line.find("\"refSpotY\"")) != std::string::npos)
      {
         size_t colonPos = line.find(":", pos);
         if (colonPos != std::string::npos)
         {
            currentCal.refSpotY = std::atof(line.substr(colonPos + 1).c_str());
         }
      }
      // Look for refZ
      else if ((pos = line.find("\"refZ\"")) != std::string::npos)
      {
         size_t colonPos = line.find(":", pos);
         if (colonPos != std::string::npos)
         {
            currentCal.refZ = std::atof(line.substr(colonPos + 1).c_str());
         }
      }
      // Look for dominantAxis
      else if ((pos = line.find("\"dominantAxis\"")) != std::string::npos)
      {
         size_t colonPos = line.find(":", pos);
         if (colonPos != std::string::npos)
         {
            size_t quotePos = line.find("\"", colonPos + 1);
            if (quotePos != std::string::npos)
            {
               currentCal.dominantAxis = line[quotePos + 1];
            }
         }
      }
      // Look for slope1Dominant
      else if ((pos = line.find("\"slope1Dominant\"")) != std::string::npos)
      {
         size_t colonPos = line.find(":", pos);
         if (colonPos != std::string::npos)
         {
            currentCal.slope1Dominant = std::atof(line.substr(colonPos + 1).c_str());
         }
      }
      // Look for slope2Dominant
      else if ((pos = line.find("\"slope2Dominant\"")) != std::string::npos)
      {
         size_t colonPos = line.find(":", pos);
         if (colonPos != std::string::npos)
         {
            currentCal.slope2Dominant = std::atof(line.substr(colonPos + 1).c_str());
         }
      }
      // Look for spotSelection
      else if ((pos = line.find("\"spotSelection\"")) != std::string::npos)
      {
         size_t colonPos = line.find(":", pos);
         if (colonPos != std::string::npos)
         {
            size_t quotePos1 = line.find("\"", colonPos + 1);
            if (quotePos1 != std::string::npos)
            {
               size_t quotePos2 = line.find("\"", quotePos1 + 1);
               if (quotePos2 != std::string::npos)
               {
                  currentCal.spotSelection = line.substr(quotePos1 + 1, quotePos2 - quotePos1 - 1);
               }
            }
         }
      }
      // Look for precision
      else if ((pos = line.find("\"precision\"")) != std::string::npos)
      {
         size_t colonPos = line.find(":", pos);
         if (colonPos != std::string::npos)
         {
            currentCal.precision = std::atof(line.substr(colonPos + 1).c_str());
         }
      }
      // Look for description
      else if ((pos = line.find("\"description\"")) != std::string::npos)
      {
         size_t colonPos = line.find(":", pos);
         if (colonPos != std::string::npos)
         {
            size_t quotePos1 = line.find("\"", colonPos + 1);
            if (quotePos1 != std::string::npos)
            {
               size_t quotePos2 = line.find("\"", quotePos1 + 1);
               if (quotePos2 != std::string::npos)
               {
                  currentCal.description = line.substr(quotePos1 + 1, quotePos2 - quotePos1 - 1);
               }
            }
         }
      }
      // Look for maxZ
      else if ((pos = line.find("\"maxZ\"")) != std::string::npos)
      {
         size_t colonPos = line.find(":", pos);
         if (colonPos != std::string::npos)
         {
            currentCal.maxZ = std::atof(line.substr(colonPos + 1).c_str());
         }
      }
      // Look for roiX
      else if ((pos = line.find("\"roiX\"")) != std::string::npos)
      {
         size_t colonPos = line.find(":", pos);
         if (colonPos != std::string::npos)
         {
            currentCal.roiX = std::atoi(line.substr(colonPos + 1).c_str());
         }
      }
      // Look for roiY
      else if ((pos = line.find("\"roiY\"")) != std::string::npos)
      {
         size_t colonPos = line.find(":", pos);
         if (colonPos != std::string::npos)
         {
            currentCal.roiY = std::atoi(line.substr(colonPos + 1).c_str());
         }
      }
      // Look for roiWidth
      else if ((pos = line.find("\"roiWidth\"")) != std::string::npos)
      {
         size_t colonPos = line.find(":", pos);
         if (colonPos != std::string::npos)
         {
            currentCal.roiWidth = std::atoi(line.substr(colonPos + 1).c_str());
         }
      }
      // Look for roiHeight
      else if ((pos = line.find("\"roiHeight\"")) != std::string::npos)
      {
         size_t colonPos = line.find(":", pos);
         if (colonPos != std::string::npos)
         {
            currentCal.roiHeight = std::atoi(line.substr(colonPos + 1).c_str());
         }
      }
      // Look for binning
      else if ((pos = line.find("\"binning\"")) != std::string::npos)
      {
         size_t colonPos = line.find(":", pos);
         if (colonPos != std::string::npos)
         {
            currentCal.binning = std::atol(line.substr(colonPos + 1).c_str());
         }
      }
      // Look for exposureMs
      else if ((pos = line.find("\"exposureMs\"")) != std::string::npos)
      {
         size_t colonPos = line.find(":", pos);
         if (colonPos != std::string::npos)
         {
            currentCal.exposureMs = std::atof(line.substr(colonPos + 1).c_str());
         }
      }
      // Look for closing bracket - this marks end of calibration object
      else if (line.find("}") != std::string::npos && currentDeviceSetting >= 0)
      {
         // Save the complete calibration data
         calibrationMap_[currentDeviceSetting] = currentCal;
         currentDeviceSetting = -1;
      }
   }

   file.close();
   return DEVICE_OK;
}


