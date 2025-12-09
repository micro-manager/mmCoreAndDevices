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
#include <opencv2/opencv.hpp>

const char* g_Camera = "Camera";
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
   binning_(1)
{
   InitializeDefaultErrorMessages();
   SetErrorText(ERR_NO_PHYSICAL_CAMERA, "No physical camera found.  Please select a valid camera in the Camera property.");
   SetErrorText(ERR_AUTOFOCUS_NOT_SUPPORTED, "The selected camera does not support AutoFocus.");
   SetErrorText(ERR_NO_SHUTTER_DEVICE_FOUND, "No Shutter device found.  Please select a valid shutter in the Shutter property.");
   SetErrorText(ERR_NO_AUTOFOCUS_DEVICE, "No AutoFocus Device selected");
   SetErrorText(ERR_NO_AUTOFOCUS_DEVICE_FOUND, "No AutoFocus Device loaded");
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
   CreateProperty("Shutter", defaultShutter.c_str(), MM::String, false, pAct, false);
   if (availableShutters_.size() >= 1)
      SetAllowedValues("Shutter", availableShutters_);
   else
      return ERR_NO_SHUTTER_DEVICE_FOUND;
   // This is needed, otherwise Shutter_ is not always set resulting in crashes
   // This could lead to strange problems if multiple shutter devices are loaded
   SetProperty("Shutter", defaultShutter.c_str());
   // get list with available physical cameras.
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

   pAct = new CPropertyAction(this, &AutoFocus::OnAlgorithm);
   CreateProperty(g_Alg, g_Alg_Standard, MM::String, false, pAct);
   AddAllowedValue(g_Alg, g_Alg_Standard);

   // Query camera for current ROI and dimensions
   /*
   MM::Camera* camera = static_cast<MM::Camera*>(GetDevice(camera_.c_str()));
   if (camera != 0)
   {
      unsigned x, y, xSize, ySize;
      int ret = camera->GetROI(x, y, xSize, ySize);
      if (ret == DEVICE_OK)
      {
         // If no ROI is set, use full frame
         if (xSize == 0 || ySize == 0)
         {
            roiWidth_ = camera->GetImageWidth();
            roiHeight_ = camera->GetImageHeight();
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
      }

      // Get current binning
      binning_ = camera->GetBinning();
      if (binning_ <= 0)
         binning_ = 1;
   }
   */

   // Create ROI-X property
   pAct = new CPropertyAction(this, &AutoFocus::OnROI_X);
   CreateIntegerProperty("ROI-X", 0, false, pAct);
   SetPropertyLimits("ROI-X", 0, 65536);

   // Create ROI-Y property
   pAct = new CPropertyAction(this, &AutoFocus::OnROI_Y);
   CreateIntegerProperty("ROI-Y", 65536, false, pAct);
   SetPropertyLimits("ROI-Y", 0, 65536);

   // Create ROI-Width property
   pAct = new CPropertyAction(this, &AutoFocus::OnROI_Width);
   CreateIntegerProperty("ROI-Width", 65536, false, pAct);
   SetPropertyLimits("ROI-Width", 0, 65536);

   // Create ROI-Height property
   pAct = new CPropertyAction(this, &AutoFocus::OnROI_Height);
   CreateIntegerProperty("ROI-Height", 65536, false, pAct);
   SetPropertyLimits("ROI-Height", 0, 65536);

   // Create Binning property
   pAct = new CPropertyAction(this, &AutoFocus::OnBinning);
   CreateIntegerProperty("Binning", binning_, false, pAct);
   SetPropertyLimits("Binning", 1, 10);

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
      // Query camera for current ROI and dimensions
      MM::Camera* pCam = static_cast<MM::Camera*>(GetDevice(camera_.c_str()));
      if (pCam != 0)
      {
         unsigned x, y, xSize, ySize;
         int ret = pCam->GetROI(x, y, xSize, ySize);
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
         }

         // Get current binning
         binning_ = pCam->GetBinning();
         if (binning_ <= 0)
            binning_ = 1;
      }
      else
      {
         // Restore previous value
         camera_ = oldCamera;
         pProp->Set(camera_.c_str());
         return ERR_NO_PHYSICAL_CAMERA;
      }

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

      // Apply ROI to camera
      MM::Camera* camera = static_cast<MM::Camera*>(GetDevice(camera_.c_str()));
      if (camera != nullptr)
      {
         int ret = camera->SetROI(roiX_, roiY_, roiWidth_, roiHeight_);
         if (ret != DEVICE_OK)
         {
            // Restore previous value
            pProp->Set((long)roiX_);
            return ret;
         }
      }
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

      // Apply ROI to camera
      MM::Camera* camera = static_cast<MM::Camera*>(GetDevice(camera_.c_str()));
      if (camera != nullptr)
      {
         int ret = camera->SetROI(roiX_, roiY_, roiWidth_, roiHeight_);
         if (ret != DEVICE_OK)
         {
            // Restore previous value
            pProp->Set((long)roiY_);
            return ret;
         }
      }
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

      // Apply ROI to camera
      MM::Camera* camera = static_cast<MM::Camera*>(GetDevice(camera_.c_str()));
      if (camera != nullptr)
      {
         int ret = camera->SetROI(roiX_, roiY_, roiWidth_, roiHeight_);
         if (ret != DEVICE_OK)
         {
            // Restore previous value
            pProp->Set((long)roiWidth_);
            return ret;
         }
      }
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

      // Apply ROI to camera
      MM::Camera* camera = static_cast<MM::Camera*>(GetDevice(camera_.c_str()));
      if (camera != nullptr)
      {
         int ret = camera->SetROI(roiX_, roiY_, roiWidth_, roiHeight_);
         if (ret != DEVICE_OK)
         {
            // Restore previous value
            pProp->Set((long)roiHeight_);
            return ret;
         }
      }
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
      long value;
      pProp->Get(value);

      // Apply binning to camera
      MM::Camera* camera = static_cast<MM::Camera*>(GetDevice(camera_.c_str()));
      if (camera != nullptr)
      {
         int ret = camera->SetBinning((int)value);
         if (ret != DEVICE_OK)
         {
            // Restore previous value
            pProp->Set(binning_);
            return ret;
         }
         binning_ = value;
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
   continuousFocusing_ = on;
   return DEVICE_OK;
}

int AutoFocus::GetContinuousFocusing(bool& on)
{
   on = continuousFocusing_;
   return DEVICE_OK;
}

bool AutoFocus::IsContinuousFocusLocked()
{
   return false;
}

int AutoFocus::SetOffset(double offset)
{
   offset_ = offset;
   return DEVICE_OK;
}

int AutoFocus::GetOffset(double& offset)
{
   offset = offset_;
   return DEVICE_OK;
}

int AutoFocus::FullFocus()
{
   return SnapAndAnalyze();
}

int AutoFocus::IncrementalFocus()
{
   return SnapAndAnalyze();
}

int AutoFocus::GetLastFocusScore(double& score)
{
   //return AnalyzeImage(algorithm_ == g_Alg_Standard ? 0 : 0, score, score, score);
   return DEVICE_ERR;
}

int AutoFocus::GetCurrentFocusScore(double& score)
{
   //return AnalyzeImage(algorithm_ == g_Alg_Standard ? 0 : 0, score, score, score);
   return DEVICE_ERR;
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

   // Close shutter to block IR light
   shutter->SetOpen(false);
   CDeviceUtils::SleepMs(10); // wait for shutter to close
   // Snap image with shutter closed
   camera->SnapImage();
   // TODO: take dark image only once
   cv::Mat darkImage = GetImageFromBuffer();

   shutter->SetOpen(true);
   CDeviceUtils::SleepMs(10); // wait for shutter to open
   // Snap image with shutter open
   camera->SnapImage();
   cv::Mat lightImage = GetImageFromBuffer();

   cv::Mat resultImage;

   // Subtract image2 from image1
   cv::subtract(lightImage, darkImage, resultImage);

   double scoreOpen;
   double xOpen, yOpen;
   AnalyzeImage(resultImage, scoreOpen, xOpen, yOpen);

   // Here we would implement the logic to adjust focus based on scores
   // For now, we just return OK
   return DEVICE_OK;
}

cv::Mat AutoFocus::GetImageFromBuffer()
{
   // Get camera device
   MM::Camera* camera = static_cast<MM::Camera*>(GetDevice(camera_.c_str()));
   if (camera == nullptr)
      return cv::Mat();
   // Get image buffer
   const unsigned int width = camera->GetImageWidth();
   const unsigned int height = camera->GetImageHeight();
   const unsigned int bpp = camera->GetImageBytesPerPixel();
   const unsigned char* imgBuffer = camera->GetImageBuffer();
   if (imgBuffer == nullptr || width == 0 || height == 0)
      return cv::Mat();
   cv::Mat image(height, width, CV_16UC3, (void *) imgBuffer);
   return image;
}

int AutoFocus::AnalyzeImage(cv::Mat image, double& score, double& x, double& y)
{
   // convert grayscale to binary image
   cv::Mat thr;
   cv::threshold(image, thr, 100, 65000, cv::THRESH_BINARY);

   // find moments of the image
   cv::Moments m = moments(thr, true);
   cv::Point p(m.m10 / m.m00, m.m01 / m.m00);

   // coordinates of centroid
   std::ostringstream os;
   os << p.x << p.y ;
   this->LogMessage(os.str().c_str());
   

/*
   // Simple analysis: compute center of mass as focus point
   double sumX = 0.0;
   double sumY = 0.0;
   double sumIntensity = 0.0;
   for (unsigned int yPos = 0; yPos < height; ++yPos)
   {
      for (unsigned int xPos = 0; xPos < width; ++xPos)
      {
         unsigned char intensity = imgBuffer[yPos * width + xPos];
         sumX += xPos * intensity;
         sumY += yPos * intensity;
         sumIntensity += intensity;
      }
   }
   if (sumIntensity > 0)
   {
      x = sumX / sumIntensity;
      y = sumY / sumIntensity;
      score = sumIntensity / (width * height); // average intensity as score
   }
   else
   {
      x = 0.0;
      y = 0.0;
      score = 0.0;
   }
   */
   return DEVICE_OK;
}


