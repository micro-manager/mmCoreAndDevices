#include "CameraPulser.h"
#include "ModuleInterface.h"
#include <sstream>
#include <cstdio>

#ifdef WIN32
   #define WIN32_LEAN_AND_MEAN
   #include <windows.h>
#endif


// Global info about the state of the Arduino.  
const uint32_t g_Min_MMVersion = 1;
const uint32_t g_Max_MMVersion = 1;
const char* g_versionProp = "Version";
const char* g_Undefined = "Undefined";
const char* g_IntervalBeyondExposure = "Interval-ms_on_top_of_exposure";
const char* g_WaitForInputMode = "Wait_for_Input";


const char* g_DeviceNameCameraPulser = "TeensySendsPulsesToCamera";

CameraPulser::CameraPulser() :
   pulseDuration_(1.0),
   intervalBeyondExposure_(5.0),
   waitForInput_(false),
   initialized_(false),
   version_(0),
   nrCamerasInUse_(0),
   teensyCom_(0)
{
   InitializeDefaultErrorMessages();

   SetErrorText(ERR_INVALID_DEVICE_NAME, "Please select a valid camera");
   SetErrorText(ERR_NO_PHYSICAL_CAMERA, "No physical camera assigned");
   SetErrorText(ERR_FIRMWARE_VERSION_TOO_NEW, "Firmware version is newer than expected");
   SetErrorText(ERR_FIRMWARE_VERSION_TOO_OLD, "Firmware version is older than expected");

   // Name                                                                   
   CreateProperty(MM::g_Keyword_Name, g_DeviceNameCameraPulser, MM::String, true);

   // Description                                                            
   CreateProperty(MM::g_Keyword_Description, "Use Camera in external trigger mode and provide triggers with Teensy", MM::String, true);

   for (int i = 0; i < MAX_NUMBER_PHYSICAL_CAMERAS; i++) {
      usedCameras_.push_back(g_Undefined);
   }

   CPropertyAction* pAct = new CPropertyAction(this, &CameraPulser::OnPort);
   CreateProperty(MM::g_Keyword_Port, "Undefined", MM::String, false, pAct, true);
}

CameraPulser::~CameraPulser()
{
   if (initialized_)
      Shutdown();
}

int CameraPulser::Shutdown()
{
   // Rely on the cameras to shut themselves down
   return DEVICE_OK;
}

int CameraPulser::Initialize()
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

   for (unsigned i = 0; i < MAX_NUMBER_PHYSICAL_CAMERAS; i++)
   {
      CPropertyActionEx* pAct = new CPropertyActionEx(this, &CameraPulser::OnPhysicalCamera, i);
      std::ostringstream os;
      os << "Triggered Camera-" << i;
      CreateProperty(os.str().c_str(), availableCameras_[0].c_str(), MM::String, false, pAct, false);
      SetAllowedValues(os.str().c_str(), availableCameras_);
   }

   CPropertyAction* pAct = new CPropertyAction(this, &CameraPulser::OnBinning);
   CreateProperty(MM::g_Keyword_Binning, "1", MM::Integer, false, pAct, false);

   // start Teensy
   teensyCom_ = new TeensyCom(GetCoreCallback(), this, port_.c_str());
   int ret = teensyCom_->GetVersion(version_);
   if (ret != DEVICE_OK)
      return ret;

   if (version_ < g_Min_MMVersion)
       return ERR_FIRMWARE_VERSION_TOO_OLD;
   if (version_ > g_Max_MMVersion)
       return ERR_FIRMWARE_VERSION_TOO_NEW;

   pAct = new CPropertyAction(this, &CameraPulser::OnVersion);
   CreateIntegerProperty(g_versionProp, version_, true, pAct);

   // Pulse Duration property
   uint32_t pulseDuration;
   ret = teensyCom_->GetPulseDuration(pulseDuration);
   if (ret != DEVICE_OK)
      return ret;
   pulseDuration_ = pulseDuration / 1000.0;
   pAct = new CPropertyAction(this, &CameraPulser::OnPulseDuration);
   CreateFloatProperty("PulseDuration-ms", pulseDuration_, false, pAct);

   // Interval property
   // At this point, GetExposure does not give us a good value, adjust interval
   // for exposure after camera was set 
   pAct = new CPropertyAction(this, &CameraPulser::OnIntervalBeyondExposure);
   CreateFloatProperty(g_IntervalBeyondExposure, intervalBeyondExposure_, false, pAct);

   // Trigger Mode property
   uint32_t waitForInput;
   ret = teensyCom_->GetWaitForInput(waitForInput);
   if (ret != DEVICE_OK)
      return ret;
   waitForInput_ = (bool) waitForInput;
   pAct = new CPropertyAction(this, &CameraPulser::OnWaitForInput);
   CreateProperty(g_WaitForInputMode, waitForInput_ ? "On" : "Off", MM::String, false, pAct);
   AddAllowedValue(g_WaitForInputMode, "Off");
   AddAllowedValue(g_WaitForInputMode, "On");

   initialized_ = true;

   return DEVICE_OK;
}

void CameraPulser::GetName(char* name) const
{
   CDeviceUtils::CopyLimitedString(name, g_DeviceNameCameraPulser);
}

int CameraPulser::SnapImage()
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
   // send one pulse, even when the cameras are not in external trigger mode, this should not hurt
   uint32_t response;
   int ret = teensyCom_->SetNumberOfPulses(1, response);
   if (ret != DEVICE_OK)
      return ret;
   if (response != 1)
      return ERR_COMMUNICATION;
   ret = teensyCom_->SetStart(response);
   if (ret != DEVICE_OK)
      return ret;

   // I think that the CameraSnapThread destructor waits until the SnapImage function is done
   // So, we are likely to be waiting here until all cameras are done snapping

   return DEVICE_OK;
}

/**
 * return the ImageBuffer of the first physical camera
 */
const unsigned char* CameraPulser::GetImageBuffer()
{
   if (nrCamerasInUse_ < 1)
      return 0;

   return GetImageBuffer(0);
}

const unsigned char* CameraPulser::GetImageBuffer(unsigned channelNr)
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

bool CameraPulser::IsCapturing()
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
unsigned CameraPulser::GetImageWidth() const
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
unsigned CameraPulser::GetImageHeight() const
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
bool CameraPulser::ImageSizesAreEqual() {
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

unsigned CameraPulser::GetImageBytesPerPixel() const
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

unsigned CameraPulser::GetBitDepth() const
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

long CameraPulser::GetImageBufferSize() const
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

double CameraPulser::GetExposure() const
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

void CameraPulser::SetExposure(double exp)
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

int CameraPulser::SetROI(unsigned x, unsigned y, unsigned xSize, unsigned ySize)
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

int CameraPulser::GetROI(unsigned& x, unsigned& y, unsigned& xSize, unsigned& ySize)
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

int CameraPulser::ClearROI()
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

int CameraPulser::PrepareSequenceAcqusition()
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

int CameraPulser::StartSequenceAcquisition(double interval)
{
   if (nrCamerasInUse_ < 1)
      return ERR_NO_PHYSICAL_CAMERA;

   if (!ImageSizesAreEqual())
      return ERR_NO_EQUAL_SIZE;

   uint32_t response;
   int ret = teensyCom_->SetNumberOfPulses(0, response);
   if (ret != DEVICE_OK)
      return ret;

   uint32_t tInterval = static_cast<uint32_t> ((GetExposure() + intervalBeyondExposure_) * 1000.0);
   ret = teensyCom_->SetInterval(tInterval, response);
   if (response != tInterval)
      return ERR_COMMUNICATION;

   for (unsigned int i = 0; i < usedCameras_.size(); i++)
   {
      MM::Camera* camera = (MM::Camera*)GetDevice(usedCameras_[i].c_str());
      if (camera != 0)
      {
         std::ostringstream os;
         os << i;
         camera->AddTag(MM::g_Keyword_CameraChannelName, usedCameras_[i].c_str(),
            usedCameras_[i].c_str());
         camera->AddTag(MM::g_Keyword_CameraChannelIndex, usedCameras_[i].c_str(),
            os.str().c_str());

         ret = camera->StartSequenceAcquisition(interval);
         if (ret != DEVICE_OK)
            return ret;
      }
   }
   // start pulses, should even when cameras are not in external trigger mode
   ret = teensyCom_->SetStart(response);
   if (ret != DEVICE_OK)
      return ret;

   return DEVICE_OK;
}

int CameraPulser::StartSequenceAcquisition(long numImages, double interval_ms, bool stopOnOverflow)
{
   if (nrCamerasInUse_ < 1)
      return ERR_NO_PHYSICAL_CAMERA;

   uint32_t response;
   int ret = teensyCom_->SetNumberOfPulses(numImages, response);
   if (response != (uint32_t) numImages)
      return ERR_COMMUNICATION;

   uint32_t interval = static_cast<uint32_t> ((GetExposure() + intervalBeyondExposure_) * 1000.0);
   ret = teensyCom_->SetInterval(interval, response);
   if (response != interval)
      return ERR_COMMUNICATION;

   // First start camera sequences, then start trigger
   for (unsigned int i = 0; i < usedCameras_.size(); i++)
   {
      MM::Camera* camera = (MM::Camera*)GetDevice(usedCameras_[i].c_str());
      if (camera != 0)
      {
         ret = camera->StartSequenceAcquisition(numImages, interval_ms, stopOnOverflow);
         if (ret != DEVICE_OK)
            return ret;
      }
   }

   ret = teensyCom_->SetStart(response);
   if (ret != DEVICE_OK) // TODO: Check response
      return ret;

   return DEVICE_OK;
}

int CameraPulser::StopSequenceAcquisition()
{
   uint32_t response;
   int ret = teensyCom_->SetStop(response);
   if (ret != DEVICE_OK)
      return ret;

   for (unsigned int i = 0; i < usedCameras_.size(); i++)
   {
      MM::Camera* camera = (MM::Camera*)GetDevice(usedCameras_[i].c_str());
      if (camera != 0)
      {
         ret = camera->StopSequenceAcquisition();
         if (ret != DEVICE_OK)
            return ret;

         std::ostringstream os;
         os << i;
         camera->AddTag(MM::g_Keyword_CameraChannelName, usedCameras_[i].c_str(),
            "");
         camera->AddTag(MM::g_Keyword_CameraChannelIndex, usedCameras_[i].c_str(),
            os.str().c_str());
      }
   }
   std::ostringstream os;
   os << "Stopped Sequence after sending " << response << " pulses.";
   GetCoreCallback()->LogMessage(this, os.str().c_str(), false);

   return DEVICE_OK;
}

int CameraPulser::GetBinning() const
{
   if (usedCameras_.empty())
   {
      return 1;
   }
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

int CameraPulser::SetBinning(int bS)
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

int CameraPulser::IsExposureSequenceable(bool& isSequenceable) const
{
   isSequenceable = false;

   return DEVICE_OK;
}

unsigned CameraPulser::GetNumberOfComponents() const
{
   return 1;
}

unsigned CameraPulser::GetNumberOfChannels() const
{
   return nrCamerasInUse_;
}

int CameraPulser::GetChannelName(unsigned channel, char* name)
{
   CDeviceUtils::CopyLimitedString(name, "");
   int ch = Logical2Physical(channel);
   if (ch >= 0 && static_cast<unsigned>(ch) < usedCameras_.size())
   {
      CDeviceUtils::CopyLimitedString(name, usedCameras_[ch].c_str());
   }
   return DEVICE_OK;
}

int CameraPulser::Logical2Physical(int logical)
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


int CameraPulser::OnPhysicalCamera(MM::PropertyBase* pProp, MM::ActionType eAct, long i)
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

int CameraPulser::OnBinning(MM::PropertyBase* pProp, MM::ActionType eAct)
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

int CameraPulser::OnPort(MM::PropertyBase* pProp, MM::ActionType pAct)
{
   if (pAct == MM::BeforeGet)
   {
      pProp->Set(port_.c_str());
   }
   else if (pAct == MM::AfterSet)
   {
      pProp->Get(port_);
   }
   return DEVICE_OK;
}


int CameraPulser::OnVersion(MM::PropertyBase* pProp, MM::ActionType pAct)
{
   if (pAct == MM::BeforeGet)
   {
      pProp->Set((long)version_);
   }
   return DEVICE_OK;
}

int CameraPulser::OnPulseDuration(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(pulseDuration_);
    }
    else if (eAct == MM::AfterSet)
    {
       pProp->Get(pulseDuration_);
        
       // Send pulse duration command if initialized
       if (initialized_)
       {
          uint32_t pulseDurationUs =  static_cast<uint32_t>(pulseDuration_ * 1000.0);
          uint32_t param;
          int ret = teensyCom_->SetPulseDuration(pulseDurationUs, param);
          if (ret != DEVICE_OK)
             return ret;
          if (param != pulseDurationUs)
          {
            GetCoreCallback()->LogMessage(this, "PulseDuration sent not the same as pulseDuration echoed back", false);
            return ERR_COMMUNICATION;

          }
       }
   }
   return DEVICE_OK;
}

int CameraPulser::OnIntervalBeyondExposure(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (eAct == MM::BeforeGet)
    {
        pProp->Set(intervalBeyondExposure_);
    }
    else if (eAct == MM::AfterSet)
    {
       pProp->Get(intervalBeyondExposure_);
        
       // Send interval command if initialized
       if (initialized_)
       {
          uint32_t interval = static_cast<uint32_t> ((GetExposure() + intervalBeyondExposure_) * 1000.0);
          uint32_t parm;
          int ret = teensyCom_->SetInterval(interval, parm);
          if (ret != DEVICE_OK)
             return ret;
          if (parm != interval)
          {
            GetCoreCallback()->LogMessage(this, "Interval sent not the same as interval echoed back", false);
            return ERR_COMMUNICATION;
          }
       }
    }
    return DEVICE_OK;
}

int CameraPulser::OnWaitForInput(MM::PropertyBase* pProp, MM::ActionType eAct)
{

   if (eAct == MM::BeforeGet)
   {
      pProp->Set(waitForInput_ ? "On" : "Off");
   }
   else if (eAct == MM::AfterSet)
   {
      std::string waitForInput;
      pProp->Get(waitForInput);
      waitForInput_ = (waitForInput == "On");
        
      // Send wait for input command if initialized
      if (initialized_)
      {
         uint32_t sp = waitForInput_ ? 1 : 0;
         uint32_t param;
         int ret = teensyCom_->SetWaitForInput(sp, param);
         if (ret != DEVICE_OK)
            return ret;
         if (param != sp)
         {
            GetCoreCallback()->LogMessage(this, "WaitforInput sent not the same as echoed back", false);
            return ERR_COMMUNICATION;
         }
      }
   }
   return DEVICE_OK;
}
