///////////////////////////////////////////////////////////////////////////////
// FILE:          ReflectionFocus.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Hardware-based reflection spot tracking autofocus using a
//                camera and optional shutter.
//
// AUTHOR:        Nico Stuurman, nico@cmp.ucsf.edu, 11/07/2008
//                Nico Stuurman, nstuurman@altoslabs.com, 4/22/2022
// COPYRIGHT:     University of California, San Francisco, 2008
//                2015 Open Imaging, Inc.
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

#ifndef _REFLECTIONFOCUS_H_
#define _REFLECTIONFOCUS_H_

#include "MMDevice.h"
#include "DeviceBase.h"
#include "ImgBuffer.h"
#include <string>
#include <map>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

//////////////////////////////////////////////////////////////////////////////
// Device name constants
//
extern const char* g_DeviceNameReflectionFocus;
extern const char* g_DeviceNameReflectionFocusStage;

//////////////////////////////////////////////////////////////////////////////
// Error codes
//
#define ERR_INVALID_DEVICE_NAME            10001
#define ERR_NO_AUTOFOCUS_DEVICE            10008
#define ERR_NO_AUTOFOCUS_DEVICE_FOUND      10009
#define ERR_NO_PHYSICAL_CAMERA             10010
#define ERR_AUTOFOCUS_NOT_SUPPORTED        10012
#define ERR_NO_PHYSICAL_STAGE              10013
#define ERR_NO_SHUTTER_DEVICE_FOUND        10014
#define ERR_TARGET_TOO_HIGH                10015
#define ERR_NOT_CALIBRATED                 10016
#define ERR_POS_OUT_OF_RANGE               10017

/**
 * Treats a ReflectionFocus device as a Drive.
 * Can be used to make the ReflectionFocus offset appear in the position list
 */
class AutoFocusStage : public CStageBase<AutoFocusStage>
{
public:
   AutoFocusStage();
   ~AutoFocusStage();

   // Device API
   // ----------
   int Initialize();
   int Shutdown();

   void GetName(char* pszName) const;
   bool Busy();

   // Stage API
   // ---------
  int SetPositionUm(double pos);
  int GetPositionUm(double& pos);
  int SetPositionSteps(long steps);
  int GetPositionSteps(long& steps);
  int SetOrigin();
  int GetLimits(double& min, double& max);

  int IsStageSequenceable(bool& isSequenceable) const {isSequenceable = false; return DEVICE_OK;}
  bool IsContinuousFocusDrive() const {return true;}

   // action interface
   // ----------------
   int OnAutoFocusDevice(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
   std::vector<std::string> availableAutoFocusDevices_;
   std::string AutoFocusDeviceName_;
   bool initialized_;
};

/**
 * Hardware-based autofocus device that uses a shutter (to
  * switch IR light source on/off) and a camera to determine
  * the location/size of the reflection spot.  Since images
  * will be read out snap by snap, this will not be
  * extremely fast, but more a proof of concept.
  * Multiples algorithms can be used to determine the
  * location of the best focus.
  */
class AutoFocus : public CAutoFocusBase<AutoFocus>
{
   public:
      AutoFocus();
      ~AutoFocus();
      // Device API
      // ----------
      int Initialize();
      int Shutdown();
      void GetName(char* name) const;
      bool Busy();
      // AutoFocus API
      // -------------
      int SetContinuousFocusing(bool on);
      int GetContinuousFocusing(bool& on);
      bool IsContinuousFocusLocked();
      int FullFocus();
      int IncrementalFocus();
      int GetLastFocusScore(double& score);
      int GetCurrentFocusScore(double& score);
      int SetOffset(double offset);
      int GetOffset(double& offset);
      void RegisterStage(MM::Stage* stage);
      void UnregisterStage(MM::Stage* stage);

      // action interface
      int OnShutter(MM::PropertyBase* pProp, MM::ActionType eAct);
      int OnCamera(MM::PropertyBase* pProp, MM::ActionType eAct);
      int OnFocusStage(MM::PropertyBase* pProp, MM::ActionType eAct);
      int OnAlgorithm(MM::PropertyBase* pProp, MM::ActionType eAct);
      int OnROI_X(MM::PropertyBase* pProp, MM::ActionType eAct);
      int OnROI_Y(MM::PropertyBase* pProp, MM::ActionType eAct);
      int OnROI_Width(MM::PropertyBase* pProp, MM::ActionType eAct);
      int OnROI_Height(MM::PropertyBase* pProp, MM::ActionType eAct);
      int OnBinning(MM::PropertyBase* pProp, MM::ActionType eAct);
      int OnExposure(MM::PropertyBase* pProp, MM::ActionType eAct);
      int OnDeviceSettings(MM::PropertyBase* pProp, MM::ActionType eAct);
      int OnCalibrate(MM::PropertyBase* pProp, MM::ActionType eAct);
      int OnMeasureOffset(MM::PropertyBase* pProp, MM::ActionType eAct);
      int OnSpotSelection(MM::PropertyBase* pProp, MM::ActionType eAct);
      int OnPrecision(MM::PropertyBase* pProp, MM::ActionType eAct);
      int OnStatus(MM::PropertyBase* pProp, MM::ActionType eAct);
      int OnDeviceSettingsDescription(MM::PropertyBase* pProp, MM::ActionType eAct);
      int OnMaxZ(MM::PropertyBase* pProp, MM::ActionType eAct);

private:
      struct CalibrationData {
         double offsetX;
         double offsetY;
         double slopeX;
         double slopeY;
         double refSpotX;    // Reference spot X (fitted value at refZ)
         double refSpotY;    // Reference spot Y (fitted value at refZ)
         double refZ;        // Reference Z position (startPos during calibration)

         // Spot identification data
         char dominantAxis;     // 'X' or 'Y' - which axis has larger movement
         double slope1Dominant; // slope of spot 1 on dominant axis
         double slope2Dominant; // slope of spot 2 on dominant axis
         std::string spotSelection; // Spot selection used during calibration ("Top" or "Bottom")
         double precision;      // Precision in microns for iterative focusing
         std::string description;   // User description for these device settings
         double maxZ;           // Maximum Z position allowed

         // Camera settings
         unsigned roiX;
         unsigned roiY;
         unsigned roiWidth;
         unsigned roiHeight;
         long binning;
         double exposureMs;
      };

      int SnapAndAnalyze();
      int GetImageFromBuffer(ImgBuffer& img);
      int AnalyzeImage(ImgBuffer img, double& score1, double& x1, double& y1, double& score2, double& x2, double& y2);
      int SetCameraROI();
      int SetCameraBinning();
      int PerformCalibration();
      int PerformMeasureOffset();
      int SaveCalibrationData();
      int LoadCalibrationData();
      std::string GetCalibrationFilePath();
      double CalculateTargetZDiff(const CalibrationData& cal, double spotX, double spotY);
      int ValidateZPosition(double targetZ);
      void ContinuousFocusThread();
      void UpdateStatus(const std::string& newStatus);
      void NotifyRegisteredStages();

      std::vector<std::string> availableShutters_;
      std::string shutter_;
      std::vector<std::string> availableCameras_;
      std::string camera_;
      std::vector<std::string> availableFocusStages_;
      std::string focusStage_;
      bool initialized_;
      bool continuousFocusing_;
      double offset_;
      std::string algorithm_;
      ImgBuffer img_;
      // Camera ROI and binning settings
      unsigned roiX_;
      unsigned roiY_;
      unsigned roiWidth_;
      unsigned roiHeight_;
      long binning_;
      double exposureMs_;
      // Last spot measurement
      double lastSpotX_;
      double lastSpotY_;
      double lastSpotScore_;
      // Device settings and calibration
      long deviceSettings_;
      std::string deviceSettingsDescription_;
      std::string spotSelection_;
      double precision_;
      double maxZ_;
      std::map<long, CalibrationData> calibrationMap_;

      // Continuous focusing thread infrastructure
      std::thread continuousFocusThread_;
      std::mutex continuousFocusMutex_;
      std::condition_variable continuousFocusCV_;
      std::atomic<bool> continuousFocusLocked_;
      std::atomic<bool> stopThread_;
      std::string status_;

      // AutoFocusStage registration
      std::vector<MM::Stage*> registeredStages_;
      std::mutex registrationMutex_;
};


#endif //_REFLECTIONFOCUS_H_
