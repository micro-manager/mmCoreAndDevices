///////////////////////////////////////////////////////////////////////////////
// FILE:          MultiStage.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Various 'Meta-Devices' that add to or combine functionality of 
//                physcial devices.
//
// AUTHOR:        Nico Stuurman, nico@cmp.ucsf.edu, 11/07/2008
//                DAXYStage by Ed Simmon, 11/28/2011
//                Nico Stuurman, nstuurman@altoslabs.com, 4/22/2022
// COPYRIGHT:     University of California, San Francisco, 2008
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

#include <algorithm>

extern const char* g_DeviceNameMultiCamera;
extern const char* g_Undefined;


MultiCamera::MultiCamera() :
   imageBuffer_(0),
   nrCamerasInUse_(0),
   initialized_(false)
{
   InitializeDefaultErrorMessages();

   SetErrorText(ERR_INVALID_DEVICE_NAME, "Please select a valid camera");
   SetErrorText(ERR_NO_PHYSICAL_CAMERA, "No physical camera assigned");
   SetErrorText(ERR_NO_EQUAL_SIZE, "Cameras differ in image size");

   // Name                                                                   
   CreateProperty(MM::g_Keyword_Name, g_DeviceNameMultiCamera, MM::String, true);

   // Description                                                            
   CreateProperty(MM::g_Keyword_Description, "Combines multiple cameras into a single camera", MM::String, true);

   for (int i = 0; i < MAX_NUMBER_PHYSICAL_CAMERAS; i++) {
      usedCameras_.push_back(g_Undefined);
   }
}

MultiCamera::~MultiCamera()
{
   if (initialized_)
      Shutdown();
}

int MultiCamera::Shutdown()
{
   delete imageBuffer_;
   // Rely on the cameras to shut themselves down
   return DEVICE_OK;
}

int MultiCamera::Initialize()
{
   // get list with available Cameras.   
   std::vector<std::string> availableCameras;
   availableCameras.clear();
   char deviceName[MM::MaxStrLength];
   unsigned int deviceIterator = 0;
   for (;;)
   {
      GetLoadedDeviceOfType(MM::CameraDevice, deviceName, deviceIterator++);
      if (0 < strlen(deviceName))
      {
         availableCameras.push_back(std::string(deviceName));
      }
      else
         break;
   }

   availableCameras_.push_back(g_Undefined);
   std::vector<std::string>::iterator iter;
   for (iter = availableCameras.begin();
      iter != availableCameras.end();
      iter++)
   {
      MM::Device* camera = GetDevice((*iter).c_str());
      std::ostringstream os;
      os << this << " " << camera;
      LogMessage(os.str().c_str());
      if (camera && (this != camera))
         availableCameras_.push_back(*iter);
   }

   for (long i = 0; i < MAX_NUMBER_PHYSICAL_CAMERAS; i++)
   {
      CPropertyActionEx* pAct = new CPropertyActionEx(this, &MultiCamera::OnPhysicalCamera, i);
      std::ostringstream os;
      os << "Physical Camera " << i + 1;
      CreateProperty(os.str().c_str(), availableCameras_[0].c_str(), MM::String, false, pAct, false);
      SetAllowedValues(os.str().c_str(), availableCameras_);
   }

   CPropertyAction* pAct = new CPropertyAction(this, &MultiCamera::OnBinning);
   CreateProperty(MM::g_Keyword_Binning, "1", MM::Integer, false, pAct, false);

   initialized_ = true;

   return DEVICE_OK;
}

void MultiCamera::GetName(char* name) const
{
   CDeviceUtils::CopyLimitedString(name, g_DeviceNameMultiCamera);
}

int MultiCamera::SnapImage()
{
   if (nrCamerasInUse_ < 1)
      return ERR_NO_PHYSICAL_CAMERA;

   if (!ImageSizesAreEqual())
      return ERR_NO_EQUAL_SIZE;

   CameraSnapThread t[MAX_NUMBER_PHYSICAL_CAMERAS];
   for (unsigned int i = 0; i < usedCameras_.size(); i++)
   {
      MM::Camera* camera = (MM::Camera*)GetDevice(usedCameras_[i].c_str());
      if (camera != 0)
      {
         t[i].SetCamera(camera);
         t[i].Start();
      }
   }
   // I think that the CameraSnapThread destructor waits until the SnapImage function is done
   // So, we are likely to be waiting here until all cameras are done snapping

   return DEVICE_OK;
}

/**
 * return the ImageBuffer of the first physical camera
 */
const unsigned char* MultiCamera::GetImageBuffer()
{
   if (nrCamerasInUse_ < 1)
      return 0;

   return GetImageBuffer(0);
}

const unsigned char* MultiCamera::GetImageBuffer(unsigned channelNr)
{
   // We have a vector of physicalCameras, and a vector of Strings listing the cameras
   // we actually use.  
   int j = -1;
   unsigned height = GetImageHeight();
   unsigned width = GetImageWidth();
   unsigned pixDepth = GetImageBytesPerPixel();
   for (unsigned int i = 0; i < usedCameras_.size(); i++)
   {
      MM::Camera* camera = (MM::Camera*)GetDevice(usedCameras_[i].c_str());
      if (usedCameras_[i] != g_Undefined)
         j++;
      if (j == (int)channelNr)
      {
         unsigned thisHeight = camera->GetImageHeight();
         unsigned thisWidth = camera->GetImageWidth();
         if (height == thisHeight && width == thisWidth)
            return camera->GetImageBuffer();
         else
         {
            img_.Resize(width, height, pixDepth);
            img_.ResetPixels();
            if (width == thisWidth)
            {
               memcpy(img_.GetPixelsRW(), camera->GetImageBuffer(), thisHeight * thisWidth * pixDepth);
            }
            else
            {
               // we need to copy line by line
               const unsigned char* pixels = camera->GetImageBuffer();
               for (unsigned k = 0; k < thisHeight; k++)
               {
                  memcpy(img_.GetPixelsRW() + k * width, pixels + k * thisWidth, thisWidth);
               }
            }
            return img_.GetPixels();
         }
      }
   }
   return 0;
}

bool MultiCamera::IsCapturing()
{
   std::vector<std::string>::iterator iter;
   for (iter = usedCameras_.begin(); iter != usedCameras_.end(); iter++) {
      MM::Camera* camera = (MM::Camera*)GetDevice((*iter).c_str());
      if ((camera != 0) && camera->IsCapturing())
         return true;
   }

   return false;
}

/**
 * Returns the largest width of cameras used
 */
unsigned MultiCamera::GetImageWidth() const
{
   // TODO: should we use cached width?
   // If so, when do we cache?
   // Since this function is const, we can not cache the width found
   unsigned width = 0;
   unsigned int j = 0;
   while (j < usedCameras_.size())
   {
      MM::Camera* camera = (MM::Camera*)GetDevice(usedCameras_[j].c_str());
      if (camera != 0) {
         unsigned tmp = camera->GetImageWidth();
         if (tmp > width)
            width = tmp;
      }
      j++;
   }

   return width;
}

/**
 * Returns the largest height of cameras used
 */
unsigned MultiCamera::GetImageHeight() const
{
   unsigned height = 0;
   unsigned int j = 0;
   while (j < usedCameras_.size())
   {
      MM::Camera* camera = (MM::Camera*)GetDevice(usedCameras_[j].c_str());
      if (camera != 0)
      {
         unsigned tmp = camera->GetImageHeight();
         if (tmp > height)
            height = tmp;
      }
      j++;
   }

   return height;
}


/**
 * Returns true if image sizes of all available cameras are identical
 * false otherwise
 * edge case: if we have no or one camera, their sizes are equal
 */
bool MultiCamera::ImageSizesAreEqual() {
   unsigned height = 0;
   unsigned width = 0;
   for (unsigned int i = 0; i < usedCameras_.size(); i++) {
      MM::Camera* camera = (MM::Camera*)GetDevice(usedCameras_[i].c_str());
      if (camera != 0)
      {
         height = camera->GetImageHeight();
         width = camera->GetImageWidth();
      }
   }

   for (unsigned int i = 0; i < usedCameras_.size(); i++) {
      MM::Camera* camera = (MM::Camera*)GetDevice(usedCameras_[i].c_str());
      if (camera != 0)
      {
         if (height != camera->GetImageHeight())
            return false;
         if (width != camera->GetImageWidth())
            return false;
      }
   }
   return true;
}

unsigned MultiCamera::GetImageBytesPerPixel() const
{
   MM::Camera* camera0 = (MM::Camera*)GetDevice(usedCameras_[0].c_str());
   if (camera0 != 0)
   {
      unsigned bytes = camera0->GetImageBytesPerPixel();
      for (unsigned int i = 1; i < usedCameras_.size(); i++)
      {
         MM::Camera* camera = (MM::Camera*)GetDevice(usedCameras_[i].c_str());
         if (camera != 0)
            if (bytes != camera->GetImageBytesPerPixel())
               return 0;
      }
      return bytes;
   }
   return 0;
}

unsigned MultiCamera::GetBitDepth() const
{
   // Return the maximum bit depth found in all channels.
   MM::Camera* camera0 = (MM::Camera*)GetDevice(usedCameras_[0].c_str());
   if (camera0 != 0)
   {
      unsigned bitDepth = 0;
      for (unsigned int i = 0; i < usedCameras_.size(); i++)
      {
         MM::Camera* camera = (MM::Camera*)GetDevice(usedCameras_[i].c_str());
         if (camera != 0)
         {
            unsigned nextBitDepth = camera->GetBitDepth();
            if (nextBitDepth > bitDepth)
            {
               bitDepth = nextBitDepth;
            }
         }
      }
      return bitDepth;
   }
   return 0;
}

long MultiCamera::GetImageBufferSize() const
{
   long maxSize = 0;
   int unsigned counter = 0;
   for (unsigned int i = 0; i < usedCameras_.size(); i++)
   {
      MM::Camera* camera = (MM::Camera*)GetDevice(usedCameras_[i].c_str());
      if (camera != 0)
      {
         counter++;
         long tmp = camera->GetImageBufferSize();
         if (tmp > maxSize)
            maxSize = tmp;
      }
   }

   return counter * maxSize;
}

double MultiCamera::GetExposure() const
{
   MM::Camera* camera0 = (MM::Camera*)GetDevice(usedCameras_[0].c_str());
   if (camera0 != 0)
   {
      double exposure = camera0->GetExposure();
      for (unsigned int i = 1; i < usedCameras_.size(); i++)
      {
         MM::Camera* camera = (MM::Camera*)GetDevice(usedCameras_[i].c_str());
         if (camera != 0)
            if (exposure != camera->GetExposure())
               return 0;
      }
      return exposure;
   }
   return 0.0;
}

void MultiCamera::SetExposure(double exp)
{
   if (exp > 0.0)
   {
      for (unsigned int i = 0; i < usedCameras_.size(); i++)
      {
         MM::Camera* camera = (MM::Camera*)GetDevice(usedCameras_[i].c_str());
         if (camera != 0)
            camera->SetExposure(exp);
      }
   }
}

int MultiCamera::SetROI(unsigned x, unsigned y, unsigned xSize, unsigned ySize)
{
   for (unsigned int i = 0; i < usedCameras_.size(); i++)
   {
      MM::Camera* camera = (MM::Camera*)GetDevice(usedCameras_[i].c_str());
      // TODO: deal with case when CCD size are not identical
      if (camera != 0)
      {
         int ret = camera->SetROI(x, y, xSize, ySize);
         if (ret != DEVICE_OK)
            return ret;
      }
   }
   return DEVICE_OK;
}

int MultiCamera::GetROI(unsigned& x, unsigned& y, unsigned& xSize, unsigned& ySize)
{
   MM::Camera* camera0 = (MM::Camera*)GetDevice(usedCameras_[0].c_str());
   // TODO: check if ROI is same on all cameras
   if (camera0 != 0)
   {
      int ret = camera0->GetROI(x, y, xSize, ySize);
      if (ret != DEVICE_OK)
         return ret;
   }

   return DEVICE_OK;
}

int MultiCamera::ClearROI()
{
   for (unsigned int i = 0; i < usedCameras_.size(); i++)
   {
      MM::Camera* camera = (MM::Camera*)GetDevice(usedCameras_[i].c_str());
      if (camera != 0)
      {
         int ret = camera->ClearROI();
         if (ret != DEVICE_OK)
            return ret;
      }
   }

   return DEVICE_OK;
}

int MultiCamera::PrepareSequenceAcqusition()
{
   if (nrCamerasInUse_ < 1)
      return ERR_NO_PHYSICAL_CAMERA;

   for (unsigned int i = 0; i < usedCameras_.size(); i++)
   {
      MM::Camera* camera = (MM::Camera*)GetDevice(usedCameras_[i].c_str());
      if (camera != 0)
      {
         int ret = camera->PrepareSequenceAcqusition();
         if (ret != DEVICE_OK)
            return ret;
      }
   }

   return DEVICE_OK;
}

int MultiCamera::StartSequenceAcquisition(double interval)
{
   if (nrCamerasInUse_ < 1)
      return ERR_NO_PHYSICAL_CAMERA;

   if (!ImageSizesAreEqual())
      return ERR_NO_EQUAL_SIZE;

   for (unsigned int i = 0; i < usedCameras_.size(); i++)
   {
      MM::Camera* camera = (MM::Camera*)GetDevice(usedCameras_[i].c_str());
      if (camera != 0)
      {
         int ret = camera->StartSequenceAcquisition(interval);
         if (ret != DEVICE_OK)
            return ret;
      }
   }
   return DEVICE_OK;
}

int MultiCamera::StartSequenceAcquisition(long numImages, double interval_ms, bool stopOnOverflow)
{
   if (nrCamerasInUse_ < 1)
      return ERR_NO_PHYSICAL_CAMERA;

   for (unsigned int i = 0; i < usedCameras_.size(); i++)
   {
      MM::Camera* camera = (MM::Camera*)GetDevice(usedCameras_[i].c_str());
      if (camera != 0)
      {
         int ret = camera->StartSequenceAcquisition(numImages, interval_ms, stopOnOverflow);
         if (ret != DEVICE_OK)
            return ret;
      }
   }
   return DEVICE_OK;
}

int MultiCamera::StopSequenceAcquisition()
{
   for (unsigned int i = 0; i < usedCameras_.size(); i++)
   {
      MM::Camera* camera = (MM::Camera*)GetDevice(usedCameras_[i].c_str());
      if (camera != 0)
      {
         int ret = camera->StopSequenceAcquisition();
         if (ret != DEVICE_OK)
            return ret;
      }
   }
   return DEVICE_OK;
}

int MultiCamera::GetBinning() const
{
   MM::Camera* camera0 = (MM::Camera*)GetDevice(usedCameras_[0].c_str());
   int binning = 0;
   if (camera0 != 0)
      binning = camera0->GetBinning();
   for (unsigned int i = 0; i < usedCameras_.size(); i++)
   {
      MM::Camera* camera = (MM::Camera*)GetDevice(usedCameras_[i].c_str());
      if (camera != 0)
      {
         if (binning != camera->GetBinning())
            return 0;
      }
   }
   return binning;
}

int MultiCamera::SetBinning(int bS)
{
   for (unsigned int i = 0; i < usedCameras_.size(); i++)
   {
      MM::Camera* camera = (MM::Camera*)GetDevice(usedCameras_[i].c_str());
      if (camera != 0)
      {
         int ret = camera->SetBinning(bS);
         if (ret != DEVICE_OK)
            return ret;
      }
   }
   return DEVICE_OK;
}

int MultiCamera::IsExposureSequenceable(bool& isSequenceable) const
{
   isSequenceable = false;

   return DEVICE_OK;
}

unsigned MultiCamera::GetNumberOfComponents() const
{
   return 1;
}

unsigned MultiCamera::GetNumberOfChannels() const
{
   return nrCamerasInUse_;
}

int MultiCamera::GetChannelName(unsigned channel, char* name)
{
   CDeviceUtils::CopyLimitedString(name, "");
   int ch = Logical2Physical(channel);
   if (ch >= 0 && static_cast<unsigned>(ch) < usedCameras_.size())
   {
      CDeviceUtils::CopyLimitedString(name, usedCameras_[ch].c_str());
   }
   return DEVICE_OK;
}

int MultiCamera::Logical2Physical(int logical)
{
   int j = -1;
   for (unsigned int i = 0; i < usedCameras_.size(); i++)
   {
      if (usedCameras_[i] != g_Undefined)
         j++;
      if (j == logical)
         return i;
   }
   return -1;
}


int MultiCamera::OnPhysicalCamera(MM::PropertyBase* pProp, MM::ActionType eAct, long i)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set(usedCameras_[i].c_str());
   }

   else if (eAct == MM::AfterSet)
   {
      MM::Camera* camera = (MM::Camera*)GetDevice(usedCameras_[i].c_str());
      if (camera != 0)
      {
         camera->RemoveTag(MM::g_Keyword_CameraChannelName);
         camera->RemoveTag(MM::g_Keyword_CameraChannelIndex);
      }

      std::string cameraName;
      pProp->Get(cameraName);

      if (cameraName == g_Undefined) {
         usedCameras_[i] = g_Undefined;
      }
      else {
         camera = (MM::Camera*)GetDevice(cameraName.c_str());
         if (camera != 0) {
            usedCameras_[i] = cameraName;
            std::ostringstream os;
            os << i;
            char myName[MM::MaxStrLength];
            GetLabel(myName);
            camera->AddTag(MM::g_Keyword_CameraChannelName, myName, usedCameras_[i].c_str());
            camera->AddTag(MM::g_Keyword_CameraChannelIndex, myName, os.str().c_str());
         }
         else
            return ERR_INVALID_DEVICE_NAME;
      }
      nrCamerasInUse_ = 0;
      for (unsigned int usedCameraCounter = 0; usedCameraCounter < usedCameras_.size(); usedCameraCounter++)
      {
         if (usedCameras_[usedCameraCounter] != g_Undefined)
            nrCamerasInUse_++;
      }

      // TODO: Set allowed binning values correctly
      MM::Camera* camera0 = (MM::Camera*)GetDevice(usedCameras_[0].c_str());
      if (camera0 != 0)
      {
         ClearAllowedValues(MM::g_Keyword_Binning);
         int nr = camera0->GetNumberOfPropertyValues(MM::g_Keyword_Binning);
         for (int j = 0; j < nr; j++)
         {
            char value[MM::MaxStrLength];
            camera0->GetPropertyValueAt(MM::g_Keyword_Binning, j, value);
            AddAllowedValue(MM::g_Keyword_Binning, value);
         }
      }
   }

   return DEVICE_OK;
}

int MultiCamera::OnBinning(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::BeforeGet)
   {
      pProp->Set((long)GetBinning());
   }
   else if (eAct == MM::AfterSet)
   {
      long binning;
      pProp->Get(binning);
      int ret = SetBinning(binning);
      if (ret != DEVICE_OK)
         return ret;
   }
   return DEVICE_OK;
}


