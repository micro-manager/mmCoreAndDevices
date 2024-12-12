///////////////////////////////////////////////////////////////////////////////
// FILE:          MMSCCam.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   The example implementation of the demo camera.
//                Simulates generic digital camera and associated automated
//                microscope devices and enables testing of the rest of the
//                system without the need to connect to the actual hardware. 
//                
// AUTHOR:        
//
// COPYRIGHT:     
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

#include "MMSCCam.h"
#include "ModuleInterface.h"
#include "SCDefines.h"


const double SCCamera::nominalPixelSizeUm_ = 1.0;

const char* gCameraDeviceName  = "Revealer";
const char* g_PixelType_8bit = "8bit";
const char* g_PixelType_16bit = "16bit";
const char* g_feature_xoffset = "OffsetX";
const char* g_feature_yoffset = "OffsetY";
const char* g_feature_width = "Width";
const char* g_feature_height = "Height";
const char* g_feature_pixelFormat = "PixelFormat";
const char* g_feature_sensorWidth = "SensorWidth";
const char* g_feature_sensorHeight = "SensorHeight";
const char* g_feature_binning = "BinningMode";
const char* g_feature_readoutMode = "ReadoutMode";
const char* g_feature_triggerInType = "TriggerInType";
const char* g_feature_triggerActivation = "TriggerActivation";
const char* g_feature_triggerDelay = "TriggerDelay";

const char* g_feature_triggerOutSelector = "TriggerOutSelector";
const char* g_feature_triggerOutType = "TriggerOutType";
const char* g_feature_triggerOutActivation = "TriggerOutActivation";
const char* g_feature_triggerOutDelay = "TriggerOutDelay";
const char* g_feature_triggerOutPulseWidth = "TriggerOutPulseWidth";
const char* NoHubError = "Parent Hub not defined.";

const char* g_trigger_off = "off";
const char* g_trigger_external_edge = "External_Edge_Trigger";
const char* g_trigger_external_level = "External_Level_Trigger";
const char* g_trigger_rising_edge = "RisingEdge";
const char* g_trigger_falling_edge = "FallEdge";
const char* g_trigger_level_high = "LevelHigh";
const char* g_trigger_level_low = "LevelLow";
const char* g_trigger_global_reset_external_edge = "Global_Reset_External_Edge_Trigger";
const char* g_trigger_global_reset_external_level = "Global_Reset_External_Level_Trigger";

const char* g_triggerout1 = "TriggerOut1";
const char* g_triggerout2 = "TriggerOut2";
const char* g_triggerout3 = "TriggerOut3";

//const char* g_property_triggerOutIFType1 = "TriggerOutIFType1";
//const char* g_property_triggerOutType1 = "TriggerOutType1";
//const char* g_property_triggerOutAct1 = "TriggerOutAct1";
//const char* g_property_triggerOutDelay1 = "TriggerOutAct1";
//const char* g_property_triggerOutPulseWidth1 = "TriggerOutPulseWidth1";
//const char* g_trigger_if_name1 = "triggerOut1";
//
//const char* g_property_triggerOutIFType2 = "TriggerOutIFType2";
//const char* g_property_triggerOutType2 = "TriggerOutType2";
//const char* g_property_triggerOutAct2 = "TriggerOutAct2";
//const char* g_property_triggerOutDelay2 = "TriggerOutAct2";
//const char* g_property_triggerOutPulseWidth2 = "TriggerOutPulseWidth2";
//const char* g_trigger_if_name2 = "triggerOut2";
//
//const char* g_property_triggerOutIFType3 = "TriggerOutIFType3";
//const char* g_property_triggerOutType3 = "TriggerOutType3";
//const char* g_property_triggerOutAct3 = "TriggerOutAct3";
//const char* g_property_triggerOutDelay3 = "TriggerOutAct3";
//const char* g_property_triggerOutPulseWidth3 = "TriggerOutPulseWidth3";
//const char* g_trigger_if_name3 = "triggerOut3";

const char* g_trigger_exposure_start = "Exposure_Start";
const char* g_trigger_vsync = "VSYNC";
const char* g_trigger_readout_end = "Readout_End";
const char* g_trigger_ready = "Trigger_Ready";
const char* g_trigger_global_exposure = "Global_Exposure";
const char* g_trigger_high = "High";
const char* g_trigger_low = "Low";

///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////

MODULE_API void InitializeModuleData()
{
   RegisterDevice(gCameraDeviceName, MM::CameraDevice, "Scientific Camera Adapter");
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
   if (deviceName == 0)
      return 0;

   // decide which device class to create based on the deviceName parameter
   if (strcmp(deviceName, gCameraDeviceName) == 0)
   {
      // create camera
      return new SCCamera();
   }

   // ...supplied name not recognized
   return 0;
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
   delete pDevice;
}


///////////////////////////////////////////////////////////////////////////////
// SCCamera implementation
// ~~~~~~~~~~~~~~~~~~~~~~~~~~

/**
* SCCamera constructor.
* Setup default all variables and create device properties required to exist
* before intialization. In this case, no such properties were required. All
* properties will be created in the Initialize() method.
*
* As a general guideline Micro-Manager devices do not access hardware in the
* the constructor. We should do as little as possible in the constructor and
* perform most of the initialization in the Initialize() method.
*/
SCCamera::SCCamera() :
    CCameraBase<SCCamera> (),
    insertCount_(0),
    initialized_(false),
    devHandle_(nullptr),
    isAcqusition_(false),
    xoffset_(0),
    yoffset_(0),
    depth_(0),
    rawBuffer_(nullptr),
    recvImage_(nullptr),
    thd_(nullptr),
    stopOnOverflow_(false),
    isSequenceable_(false),
    sequenceMaxLength_(100),
    sequenceRunning_(false),
    sequenceIndex_(0)
{
     CreateHubIDProperty();
}

SCCamera::~SCCamera()
{
    StopSequenceAcquisition();
    if (thd_) {
        delete thd_;
        thd_ = nullptr;
    }

    if (recvImage_) {
        delete recvImage_;
        recvImage_ = nullptr;
    }

}

/**
* Obtains device name.
* Required by the MM::Device API.
*/
void SCCamera::GetName(char* name) const
{
    CDeviceUtils::CopyLimitedString(name, gCameraDeviceName);
}

/**
* Intializes the hardware.
* Required by the MM::Device API.
* Typically we access and initialize hardware at this point.
* Device properties are typically created here as well, except
* the ones we need to use for defining initialization parameters.
* Such pre-initialization properties are created in the constructor.
* (This device does not have any pre-initialization properties)
*/
int SCCamera::Initialize()
{
    if (initialized_) return DEVICE_OK;
    DemoHub* pHub = static_cast<DemoHub*>(GetParentHub());
    if (pHub)
    {
       char hubLabel[MM::MaxStrLength];
       pHub->GetLabel(hubLabel);
       SetParentID(hubLabel); // for backward comp.
    }
    else
       LogMessage(NoHubError);


    LogMessage("init start");
    // init 
    SC_Init(SCLogLevel::eInfo);

    // enum device
    SC_DeviceList deviceInfoList;
    auto ret = SC_EnumDevices(&deviceInfoList, eInterfaceTypeAll, nullptr);
    if (ret != SC_OK) {
        LogMessage("enumDevice failed");
        return DEVICE_ERR;
    }
    
    if (deviceInfoList.nDevNum < 1) {
         LogMessage("on device online");
         return DEVICE_NOT_CONNECTED;
    }

    for (uint32_t i = 0; i < deviceInfoList.nDevNum; i++) {
        if (deviceInfoList.pDevInfo[i].nCameraType == eTypeVirtual) {
            continue;
        }
        ret = SC_CreateHandle(&devHandle_, eModeByCameraKey, deviceInfoList.pDevInfo[i].cameraKey);
        if (ret != SC_OK) {
            LogMessage("create device handle failed");
            return DEVICE_NOT_CONNECTED;
        }
        break;
    }

    ret = SC_OpenEx(devHandle_, SC_ECameraAccessPermission::accessPermissionMonitor);
    if (ret != SC_OK) {
        LogMessage("create device handle failed");
        return DEVICE_NOT_CONNECTED;
    }

   int nRet = CreateProperty(MM::g_Keyword_Name, gCameraDeviceName, MM::String, true);
   if (DEVICE_OK != nRet)
      return nRet;

   // Description
   nRet = CreateProperty(MM::g_Keyword_Description, "ZKJS Camera Device ", MM::String, true);
   if (DEVICE_OK != nRet)
      return nRet;

   // CameraName
   nRet = CreateProperty(MM::g_Keyword_CameraName, "ZKJS Camera", MM::String, true);
   assert(nRet == DEVICE_OK);

   // CameraID
   nRet = CreateProperty(MM::g_Keyword_CameraID, "V1.0", MM::String, true);
   assert(nRet == DEVICE_OK);

   // exposure
   double exp = 0;
   nRet = SC_GetFloatFeatureValue(devHandle_, "ExposureTime", &exp);
   nRet = CreateProperty(MM::g_Keyword_Exposure, std::to_string(exp/1000.0f).c_str(), MM::Float, false);
   assert(nRet == DEVICE_OK);

   // binning
   CPropertyAction *pAct = new CPropertyAction (this, &SCCamera::OnBinning);
   nRet = CreateProperty(MM::g_Keyword_Binning, "1", MM::Integer, false, pAct);
   assert(nRet == DEVICE_OK);

   std::vector<std::string> binValues;
   binValues.push_back("1");    // off

   nRet = SetAllowedValues(MM::g_Keyword_Binning, binValues);
   if (nRet != DEVICE_OK)
      return nRet;

   // pixelType
   pAct = new CPropertyAction (this, &SCCamera::OnPixelType);
   nRet = CreateProperty(MM::g_Keyword_PixelType, g_PixelType_16bit, MM::String, false, pAct);
   assert(nRet == DEVICE_OK);

   std::vector<std::string> pixelValues;
   pixelValues.push_back(g_PixelType_16bit);

   nRet = SetAllowedValues(MM::g_Keyword_PixelType, pixelValues);
   if (nRet != DEVICE_OK)
      return nRet;

   // trigger in type
   nRet = createTriggerProperty();
   if (nRet != DEVICE_OK)
      return nRet;

   nRet = createTriggerOutProperty();
   if (nRet != DEVICE_OK)
      return nRet;

   // readoutMode
   uint64_t readOutMode = 0;
   readOutModes_.clear();
   readOutModes_.push_back("bit11_HS_Low");   
   readOutModes_.push_back("bit11_HS_High");
   readOutModes_.push_back("bit11_HDR_Low");
   readOutModes_.push_back("bit11_HDR_High");
   readOutModes_.push_back("bit12_HDR_Low");
   readOutModes_.push_back("bit12_HDR_High");
   readOutModes_.push_back("bit12_CMS");
   readOutModes_.push_back("bit16_From11");
   readOutModes_.push_back("bit16_From12");
   //readOutModes_.push_back("bit11_HDR_2F");
   //readOutModes_.push_back("bit12_HDR_2F");
   //readOutModes_.push_back("CMS_2F");

   std::string value = "bit11_HS_Low";
   nRet = SC_GetEnumFeatureValue(devHandle_, g_feature_readoutMode, &readOutMode);
   if (nRet == SC_OK && readOutMode < readOutModes_.size()) {
       value = readOutModes_[readOutMode];
   }
   pAct = new CPropertyAction (this, &SCCamera::OnReadOutMode);
   nRet = CreateProperty(g_feature_readoutMode, value.c_str(), MM::String, false, pAct);
   assert(nRet == DEVICE_OK);

   nRet = SetAllowedValues(g_feature_readoutMode, readOutModes_);
   if (nRet != DEVICE_OK)
      return nRet;


   // Whether or not to use exposure time sequencing
   pAct = new CPropertyAction (this, &SCCamera::OnIsSequenceable);
   std::string propName = "UseExposureSequences";
   CreateStringProperty(propName.c_str(), "No", false, pAct);
   AddAllowedValue(propName.c_str(), "Yes");
   AddAllowedValue(propName.c_str(), "No");

   // resizeBuffer
   int64_t width = 0, height = 0;
   uint64_t pixelFormat = 0, binningMode = 0;
   SC_GetIntFeatureValue(devHandle_, g_feature_width, &width);
   SC_GetIntFeatureValue(devHandle_, g_feature_height, &height);
   SC_GetEnumFeatureValue(devHandle_, g_feature_pixelFormat, &pixelFormat);
   SC_GetEnumFeatureValue(devHandle_, g_feature_binning, &binningMode);
   auto binningValue = convertBinningModeToBinningValue((int)binningMode);
   auto depth = convertPixelFormatToDepth((SC_EPixelType)pixelFormat);
   ResizeImageBuffer((int)width, (int)height, depth/8, binningValue);

   nRet = UpdateStatus();
   if (nRet != DEVICE_OK)
	   return nRet;

   thd_ = new CameraRecvThread(this);
   recvImage_ = new SC_Frame();

   initialized_ = true;
   LogMessage("init end");
   return DEVICE_OK;
}

/**
* Shuts down (unloads) the device.
* Required by the MM::Device API.
* Ideally this method will completely unload the device and release all resources.
* Shutdown() may be called multiple times in a row.
* After Shutdown() we should be allowed to call Initialize() again to load the device
* without causing problems.
*/
int SCCamera::Shutdown()
{
    LogMessage("Shutdown");
    StopSequenceAcquisition();
    SC_Close(devHandle_);
    SC_DestroyHandle(devHandle_);
    SC_Release();
    return DEVICE_OK;
}


int SCCamera::SnapImage() 
{ 
    if (!isAcqusition_) {
        SC_StartGrabbingEx(devHandle_, 0, SC_EGrabStrategy::grabStrartegySequential);
    }
    isAcqusition_ = true;
  
	if (sequenceRunning_ && IsCapturing()){
		double exp = GetSequenceExposure();
        SetExposure(exp);
    }

    auto rslt = getNextFrame();
    if (rslt != DEVICE_OK) {
        LogMessage("getFrame failed");
        return rslt;
    }

    auto ret = SC_StopGrabbing(devHandle_);
    if (ret != SC_OK) {
        LogMessage("stop grabbing failed");
        return DEVICE_ERR;
    }
    isAcqusition_ = false;
    return DEVICE_OK;
};

const unsigned char* SCCamera::GetImageBuffer() 
{
    MMThreadGuard g(imgPixelsLock_);
    unsigned char *pB =  (unsigned char*)(image_.GetPixels());
    return pB;
}

unsigned SCCamera::GetImageWidth() const 
{
    return image_.Width();
}

unsigned SCCamera::GetImageHeight() const { 
    return image_.Height();
}

unsigned SCCamera::GetImageBytesPerPixel() const { 
    return image_.Depth();
}

unsigned SCCamera::GetBitDepth() const { 
    return 8 * GetImageBytesPerPixel(); 
}

long SCCamera::GetImageBufferSize() const { 
    auto size = image_.Width() * image_.Height() * image_.Depth();
    return size;
}

double SCCamera::GetExposure() const { 
    char buf[MM::MaxStrLength];
    int ret = GetProperty(MM::g_Keyword_Exposure, buf);
    if (ret != DEVICE_OK)
        return 0.0;

    std::ostringstream  msg;
    msg << "GetExposure:" <<  atof(buf);
    LogMessage(msg.str().c_str());
    return atof(buf);
}

void SCCamera::SetExposure(double exp) {
    double realExp = 0;
    SC_SetFloatFeatureValue(devHandle_, "ExposureTime", exp * 1000);
    SC_GetFloatFeatureValue(devHandle_, "ExposureTime", &realExp);
    SetProperty(MM::g_Keyword_Exposure, CDeviceUtils::ConvertToString(realExp/1000));
    GetCoreCallback()->OnExposureChanged(this, realExp/1000);
   
    std::ostringstream  msg;
    msg << "SetExposure:" <<  exp << "," << realExp/1000;
    LogMessage(msg.str().c_str());
    return; 
}

int SCCamera::SetROI(unsigned x, unsigned y, unsigned xSize, unsigned ySize) { 
    
    int32_t rslt = SC_ERROR;
    std::ostringstream  errMsg;
    unsigned actualX = x, actualY = y, actualXSize = xSize, actualYSize = ySize;

    // close binning
    auto origBinningValue = GetBinning();
    auto origBinningMode = convertBinningValueToBinningMode(origBinningValue);

    auto binningValue = 1;
    auto binningMode = convertBinningValueToBinningMode(binningValue);
    SC_SetEnumFeatureValue(devHandle_, g_feature_binning, binningMode);

    // TODO:
    getActualRoi(actualX, actualY, actualXSize, actualYSize);

    rslt = SC_SetIntFeatureValue(devHandle_, g_feature_xoffset, (uint64_t)actualX);
    if (rslt != SC_OK) {
        errMsg << "set xoffset failed:" << x << std::endl;
        LogMessage(errMsg.str().c_str());
        return DEVICE_INVALID_PROPERTY_VALUE;
    }

    rslt = SC_SetIntFeatureValue(devHandle_, g_feature_yoffset, (uint64_t)actualY);
    if (rslt != SC_OK) {
        errMsg << "set yoffset failed:" << y << std::endl;
        LogMessage(errMsg.str().c_str());
        return DEVICE_INVALID_PROPERTY_VALUE;
    }

    rslt = SC_SetIntFeatureValue(devHandle_, g_feature_width, (uint64_t)actualXSize);
    if (rslt != SC_OK) {
        errMsg << "set width failed:" << xSize << std::endl;
        LogMessage(errMsg.str().c_str());
        return DEVICE_INVALID_PROPERTY_VALUE;
    }

    rslt = SC_SetIntFeatureValue(devHandle_, g_feature_height, (uint64_t)actualYSize);
    if (rslt != SC_OK) {
        errMsg << "set height failed:" << ySize << std::endl;
        LogMessage(errMsg.str().c_str());
        return DEVICE_INVALID_PROPERTY_VALUE;
    }
    
    SC_SetEnumFeatureValue(devHandle_, g_feature_binning, origBinningMode);
    GetROI(actualX, actualY, actualXSize, actualYSize);
    xoffset_ = actualX;
    yoffset_ = actualY;

    char buf[MM::MaxStrLength];
	auto ret = GetProperty(MM::g_Keyword_PixelType, buf);
	if (ret != DEVICE_OK)
		return ret;

    int depth = 1;
    ret = getDepth(buf, depth);
    if (ret != DEVICE_OK) return ret;

    return ResizeImageBuffer(actualXSize, actualYSize, depth, 1);
}

int SCCamera::GetROI(unsigned& x, unsigned& y, unsigned& xSize, unsigned& ySize) 
{
    int64_t xoffset = x;
    int64_t yoffset = y;
    int64_t width = xSize;
    int64_t height = ySize;
    auto rslt = SC_GetIntFeatureValue(devHandle_, g_feature_xoffset, &xoffset);
	SC_GetIntFeatureValue(devHandle_, g_feature_yoffset, &yoffset);
	SC_GetIntFeatureValue(devHandle_, g_feature_width, &width);
	SC_GetIntFeatureValue(devHandle_, g_feature_height, &height);
    return DEVICE_OK; 
}

int SCCamera::ClearROI() { 
    int64_t width = 0;
    int64_t height = 0;
	SC_GetIntFeatureValue(devHandle_, g_feature_sensorWidth, &width);
	SC_GetIntFeatureValue(devHandle_, g_feature_sensorHeight, &height);
    SetROI(0, 0, width, height);
    return DEVICE_OK; 
}

double SCCamera::GetSequenceExposure() 
{
    if (exposureSequence_.size() == 0) {
        return this->GetExposure();
    }

    double exposure = exposureSequence_[sequenceIndex_];
    sequenceIndex_++;
    if (sequenceIndex_ >= exposureSequence_.size()) {
        sequenceIndex_ = 0;
    }
    return exposure;
}

int SCCamera::StartSequenceAcquisition(double interval) { 
    return StartSequenceAcquisition(LONG_MAX, interval, false);
}

int SCCamera::StartSequenceAcquisition(long numImages, double interval_ms, bool stopOnOverflow) { 
	if (IsCapturing()) {
        //return DEVICE_CAMERA_BUSY_ACQUIRING;
        StopSequenceAcquisition();
	}

    if (!isAcqusition_) {
        auto rslt = SC_StartGrabbingEx(devHandle_, numImages, SC_EGrabStrategy::grabStrartegySequential);
        if (rslt != SC_OK) {
            return rslt;
        }
    }

	stopOnOverflow_ = stopOnOverflow;
	int ret = GetCoreCallback()->PrepareForAcq(this);
	if (ret != DEVICE_OK)
      return ret;

    //auto bret = GetCoreCallback()->InitializeImageBuffer(10, 1, GetImageWidth(), GetImageHeight(), GetImageBytesPerPixel());
    //if (!bret) {
    //    LogMessage("InitializeImageBuffer failed");
    //}
    thd_->Start(numImages, interval_ms);
    isAcqusition_ = true;
    LogMessage("StartSequenceAcquisition");
    return DEVICE_OK;
}

int SCCamera::StopSequenceAcquisition() { 
    if (thd_ && !thd_->IsStopped()) {
        thd_->Stop();
        thd_->wait();
    }

    auto rslt = SC_StopGrabbing(devHandle_);
    if (rslt != SC_OK) {
        LogMessage("stop aquisition failed");
        return DEVICE_ERR;
    }
    isAcqusition_ = false;
    LogMessage("stop aquisition");
    return DEVICE_OK;
}




int SCCamera::ResizeImageBuffer(
    int imageSizeW,
    int imageSizeH,
    int byteDepth,
    int binSize) 
{
    std::ostringstream  msg;
    msg << "ResizeImageBuffer: (" << imageSizeW << "," << imageSizeH << "," << binSize << "," << byteDepth << ")" << std::endl;
    LogMessage(msg.str().c_str());

    image_.Resize(imageSizeW / binSize, imageSizeH / binSize, byteDepth);
    return DEVICE_OK;
}

void SCCamera::getActualRoi(
    unsigned& actualX,
    unsigned& actualY,
    unsigned& actualXSize,
    unsigned& actualYSize) 
{
    actualX = (actualX >> 6) << 6;
    actualY = (actualY >> 1) << 1;
    actualXSize = (actualXSize >> 6) << 6;
    actualYSize = (actualYSize >> 1) << 1;
    if (actualXSize == 0) actualXSize = (1 >> 6);
    if (actualYSize == 0) actualYSize = (1 >> 2);
}


int SCCamera::getDepth(const char* pixelType, int &byteDepth)
{
    std::ostringstream  errMsg;
	if (strcmp(pixelType, g_PixelType_8bit) == 0) {
		byteDepth = 1;
	}
	else if (strcmp(pixelType, g_PixelType_16bit) == 0) {
		byteDepth = 2;
	}
	else {
        errMsg << "invalid pixeltype : " << pixelType << std::endl;
        LogMessage(errMsg.str().c_str());
		return  DEVICE_UNSUPPORTED_DATA_FORMAT;
	}
	return DEVICE_OK;
}

int SCCamera::RunSequenceOnThread(MM::MMTime startTime) 
{
    auto rslt = getNextFrame();
    if (rslt != DEVICE_OK) {
        LogMessage("getNextFrame failed");
        return rslt;
    }
    LogMessage("GetOneFrame end");
    rslt = InsertImage();
    if (rslt != DEVICE_OK) {
        LogMessage("insertFrame failed");
    }

    ++insertCount_;
    std::ostringstream  msg;
    msg << "InsertOneFrame end ("  << GetImageWidth()  << ',' <<  GetImageHeight() << ',' << GetImageBytesPerPixel() << "," << insertCount_ << ')';
    LogMessage(msg.str().c_str());
    return rslt;
}

int SCCamera::InsertImage() 
{
    auto width = GetImageWidth();
    auto height = GetImageHeight();
    unsigned int bytePerPixel = GetImageBytesPerPixel();

    MM::MMTime timeStamp = this->GetCurrentMMTime();
    char label[MM::MaxStrLength];
    this->GetLabel(label);

    Metadata md;
    md.put(MM::g_Keyword_Metadata_CameraLabel, label);
    //md.put(MM::g_Keyword_Metadata_ROI_X, CDeviceUtils::ConvertToString( (long) xoffset_));
    //md.put(MM::g_Keyword_Metadata_ROI_Y, CDeviceUtils::ConvertToString( (long) yoffset_)); 
    MMThreadGuard g(imgPixelsLock_);
	const unsigned char* data = GetImageBuffer();

    auto coreCallback = GetCoreCallback();
    auto rslt = GetCoreCallback()->InsertImage(this, data, width, height, bytePerPixel,1, md.Serialize().c_str());
    if (!stopOnOverflow_ && rslt == DEVICE_BUFFER_OVERFLOW)
    {  
		// do not stop on overflow - just reset the buffer
		GetCoreCallback()->ClearImageBuffer(this);
		// don't process this same image again...
		rslt = GetCoreCallback()->InsertImage(this, data, width, height, bytePerPixel, 1, md.Serialize().c_str(), false);
	}
    return rslt;
}

int SCCamera::getNextFrame() {
    auto rslt = SC_GetFrame(devHandle_, recvImage_, 1000);
    if (rslt == SC_TIMEOUT) {
        LogMessage("recv frame timeout");
        return DEVICE_OK;
    }
    else if (rslt != SC_OK) {
        LogMessage("recv frame failed");
        return DEVICE_ERR;
    }

    auto height = recvImage_->frameInfo.height;
    auto width = recvImage_->frameInfo.width;
    auto pixelFormat = recvImage_->frameInfo.pixelFormat;
    auto depth = convertPixelFormatToDepth(pixelFormat);

	if (height == 0 || width == 0 || depth == 0)
        return DEVICE_OUT_OF_MEMORY;

    if (height != GetImageHeight() || width != GetImageWidth() || depth/8 != GetImageBytesPerPixel()) {
        ResizeImageBuffer(width, height, depth/8, 1);
    }

    std::ostringstream  msg;
    msg << "getFrame:(" << height << "," << width << "," << recvImage_->frameInfo.size << "," << GetImageBufferSize() <<"," << depth << ")" << std::endl;
    LogMessage(msg.str().c_str());

    auto dst = image_.GetPixelsRW();
    memcpy(dst, recvImage_->pData, recvImage_->frameInfo.size);
    SC_ReleaseFrame(devHandle_, recvImage_);

    return DEVICE_OK;
}

void SCCamera::OnThreadExiting() throw()
{ 
   try
   {
      LogMessage(g_Msg_SEQUENCE_ACQUISITION_THREAD_EXITING);
      if (GetCoreCallback()) GetCoreCallback()->AcqFinished(this,0);
   }
   catch(...)
   {
      LogMessage(g_Msg_EXCEPTION_IN_ON_THREAD_EXITING, false);
   }
}

bool SCCamera::IsCapturing(){ 
    if (!thd_) return false;
    std::ostringstream  msg;
    msg << "IsCapturing" << ":" << !thd_->IsStopped();
    LogMessage(msg.str().c_str());
    return !thd_->IsStopped(); 
}

double SCCamera::GetNominalPixelSizeUm() const{
    return nominalPixelSizeUm_;
}

double SCCamera::GetPixelSizeUm() const { 
    return nominalPixelSizeUm_ * GetBinning();
}

int SCCamera::GetBinning() const {
    uint64_t binning = 0;
    SC_GetEnumFeatureValue(devHandle_, "BinningMode", &binning);
    auto value = convertBinningModeToBinningValue((int)binning);
    return value;
}

int SCCamera::SetBinning(int bS) {
    std::ostringstream  msg;
    auto mode = convertBinningValueToBinningMode(bS);
    auto rslt = SC_SetEnumFeatureValue(devHandle_, g_feature_binning, mode);
    if (rslt != SC_OK) {
        msg << "SetBinning value:" << bS << ", mode:" << mode << "failed";
        LogMessage(msg.str().c_str());
    }
    LogMessage(std::string("setBinning ").append(std::to_string(mode)).data());
    return SC_OK;
    //return SetProperty(MM::g_Keyword_Binning, CDeviceUtils::ConvertToString(bS));
}

int SCCamera::IsExposureSequenceable(bool& isSequenceable) const {
    isSequenceable = isSequenceable_;
    LogMessage("IsExposureSequenceable");
    return DEVICE_OK;
}

int SCCamera::GetExposureSequenceMaxLength(long& length) const {
    if (!isSequenceable_) {
      return DEVICE_UNSUPPORTED_COMMAND;
    }

    length = sequenceMaxLength_;
    LogMessage("GetExposureSequenceMaxLength");
    return DEVICE_OK;
}

int SCCamera::StartExposureSequence() {
    if (!isSequenceable_) {
      return DEVICE_UNSUPPORTED_COMMAND;
    }

    sequenceRunning_ = true;
    LogMessage("StartExposureSequence");
    return DEVICE_OK;
}

int SCCamera::StopExposureSequence() {
    if (!isSequenceable_) 
    {
        return DEVICE_UNSUPPORTED_COMMAND;
    }

    sequenceRunning_ = false;
    sequenceIndex_ = 0;
    LogMessage("StopExposureSequence");
    return DEVICE_OK;
}

int SCCamera::ClearExposureSequence() {
	if (!isSequenceable_) {
		return DEVICE_UNSUPPORTED_COMMAND;
	}

	exposureSequence_.clear();
	std::ostringstream  msg;
    LogMessage("ClearExposureSequence");
    return DEVICE_OK;
}

int SCCamera::AddToExposureSequence(double exposureTime_ms) {
	if (!isSequenceable_) {
		return DEVICE_UNSUPPORTED_COMMAND;
	}

    exposureSequence_.push_back(exposureTime_ms);
    LogMessage("AddToExposureSequence");
    return DEVICE_OK;
}

int SCCamera::SendExposureSequence() const {
	if (!isSequenceable_) {
		return DEVICE_UNSUPPORTED_COMMAND;
	}
    LogMessage("SendExposureSequence");
    return DEVICE_OK;
}

unsigned  SCCamera::GetNumberOfComponents() const {
    LogMessage("GetNumberOfComponents");
    return 1;
}

int SCCamera::OnIsSequenceable(MM::PropertyBase* pProp, MM::ActionType eAct)
{
	std::string val = "Yes";
	if (eAct == MM::BeforeGet)
	{
		if (!isSequenceable_)
		{
			val = "No";
		}
		pProp->Set(val.c_str());
	}
	else if (eAct == MM::AfterSet)
	{
		isSequenceable_ = false;
		pProp->Get(val);
		if (val == "Yes")
		{
			isSequenceable_ = true;
		}
	}
	return DEVICE_OK;
}

int SCCamera::OnBinning(MM::PropertyBase* pProp, MM::ActionType eAct) 
{
    int ret = DEVICE_ERR;
    switch (eAct) 
    {
        case MM::AfterSet: 
        {
            if (IsCapturing()) {
                return DEVICE_CAMERA_BUSY_ACQUIRING;
            }
            long binFactor;
            pProp->Get(binFactor);
            if (binFactor == 1 || binFactor == 2 || binFactor == 4)
            {
                // calculate ROI using the previous bin settings
                ResizeImageBuffer(GetImageWidth(), GetImageHeight(), GetImageBytesPerPixel(), binFactor);

                std::ostringstream os;
                os << binFactor;
                OnPropertyChanged("Binning", os.str().c_str());
                ret = DEVICE_OK;
            }
            else {
                LogMessage("invalid binning value");
            }
            break;
        }

        case MM::BeforeGet: {
            ret = DEVICE_OK;
            break;
        }
    }
    return ret;
}

int SCCamera::OnReadOutMode(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    auto ret = DEVICE_ERR;
    switch (eAct) {
        case MM::AfterSet: 
        {
            if (IsCapturing()) {
                return DEVICE_CAN_NOT_SET_PROPERTY;
            }

            std::string readOutMode;
            pProp->Get(readOutMode);
            auto mode = getReadOutModeEnum(readOutMode.c_str());
            if (mode == bit12_HDR_Low || mode == bit12_HDR_High || mode == bit12_CMS) {
                AddAllowedValue(g_feature_triggerInType, g_trigger_global_reset_external_edge);
                AddAllowedValue(g_feature_triggerInType, g_trigger_global_reset_external_level);
            }
            ret = DEVICE_OK;
        }break;

        case MM::BeforeGet: {
            ret = DEVICE_OK;;
        }break;
               
    }
    return ret;
}

int SCCamera::OnTriggerSelector(MM::PropertyBase* pProp, MM::ActionType eAct)
{
     auto ret = DEVICE_ERR;
     switch (eAct)
     {
        case MM::AfterSet: {
            std::string selector;
            pProp->Get(selector);
            auto rslt = SC_SetEnumFeatureValue(devHandle_, g_feature_triggerOutSelector, convertTriggerOutSeletorToEnum(selector.c_str()));
            if (rslt != SC_OK) {
                LogMessage("set selector failed");
                return ret;
            }

            uint64_t triggerOutType = 0, triggerOutAct = 0;
            double delay = 0, pulseWidth = 0;
            SC_GetEnumFeatureValue(devHandle_, g_feature_triggerOutType, &triggerOutType);
            SetProperty(g_feature_triggerOutType, triggerOutTypes_[triggerOutType].c_str());

            SC_GetEnumFeatureValue(devHandle_, g_feature_triggerOutActivation, &triggerOutAct);
            SetProperty(g_feature_triggerOutActivation, triggerActivations_[triggerOutAct].c_str());

            SC_GetFloatFeatureValue(devHandle_, g_feature_triggerOutDelay, &delay);
            SetProperty(g_feature_triggerOutDelay, std::to_string(delay).c_str());

            SC_GetFloatFeatureValue(devHandle_, g_feature_triggerOutPulseWidth, &pulseWidth);
            SetProperty(g_feature_triggerOutPulseWidth, std::to_string(pulseWidth).c_str());
            ret = DEVICE_OK;
        }break;

	    case MM::BeforeGet:
	    {
	        ret = DEVICE_OK;
	    }break;
     }
     return ret;
}

int SCCamera::OnTriggerOut(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    auto ret = DEVICE_ERR;
    switch (eAct)
    {
        case MM::AfterSet: 
        {
            if (IsCapturing())
                return DEVICE_CAN_NOT_SET_PROPERTY;
            std::string type;
            pProp->Get(type);

            std::vector<std::string> values;
            if (type == g_trigger_exposure_start || type == g_trigger_vsync || type == g_trigger_readout_end) {
                values.push_back(g_trigger_falling_edge);
                values.push_back(g_trigger_rising_edge);
                SetAllowedValues(g_feature_triggerOutActivation, values);
                SetPropertyLimits(g_feature_triggerOutDelay, 0, 1e6); // us
                SetProperty(g_feature_triggerOutActivation, g_trigger_rising_edge);
            }
            else if (type == g_trigger_ready || type == g_trigger_global_exposure) {
                values.push_back(g_trigger_level_high);
                values.push_back(g_trigger_level_low);
                SetAllowedValues(g_feature_triggerOutActivation, values);
                SetPropertyLimits(g_feature_triggerOutDelay, 0, 1e6); // us
                SetProperty(g_feature_triggerOutActivation, g_trigger_level_high);
            }
            else {
                ClearAllowedValues(g_feature_triggerActivation);
            }
            ret = DEVICE_OK;
        } break;

        case MM::BeforeGet:
        {
            ret = DEVICE_OK;
        }break;
    }
    return ret;
}

int SCCamera::OnTriggerIn(MM::PropertyBase* pProp, MM::ActionType eAct) {
    auto ret = DEVICE_ERR;
    switch (eAct)
    {
        case MM::AfterSet: 
        {
            if (IsCapturing())
                return DEVICE_CAN_NOT_SET_PROPERTY;
            std::string type;
            pProp->Get(type);

            std::vector<std::string> values;
            if (type == g_trigger_external_edge) {
                values.push_back(g_trigger_falling_edge);
                values.push_back(g_trigger_rising_edge);
                SetAllowedValues(g_feature_triggerActivation, values);
                SetPropertyLimits(g_feature_triggerDelay, 0, 1e6); // us
                SetProperty(g_feature_triggerActivation, g_trigger_rising_edge);
            }
            else if (type == g_trigger_external_level) {
                values.push_back(g_trigger_level_high);
                values.push_back(g_trigger_level_low);
                SetAllowedValues(g_feature_triggerActivation, values);
                SetPropertyLimits(g_feature_triggerDelay, 0, 1e6); // us
                SetProperty(g_feature_triggerActivation, g_trigger_level_high);
            }
            else {
                ClearAllowedValues(g_feature_triggerActivation);
                SetPropertyLimits(g_feature_triggerDelay, 0, 0); // disable
            }
            ret = DEVICE_OK;
        } break;

        case MM::BeforeGet:
        {
            ret = DEVICE_OK;
        }break;
    }
    return ret;
}

int SCCamera::OnPixelType(MM::PropertyBase* pProp, MM::ActionType eAct) 
{
   int ret = DEVICE_ERR;
   switch(eAct)
   {
   case MM::AfterSet:
      {
         if(IsCapturing())
            return DEVICE_CAN_NOT_SET_PROPERTY;

         std::string pixelType;
         pProp->Get(pixelType);

         int depth;
         ret = getDepth(pixelType.c_str(), depth);
         if (ret == DEVICE_OK) {
            ret = ResizeImageBuffer(image_.Width(), image_.Height(), depth, 1);
         }
         else
         {
            // on error switch to default pixel type
            pProp->Set(g_PixelType_8bit);
            ResizeImageBuffer(image_.Width(), image_.Height(), 1);
            ret = DEVICE_UNSUPPORTED_DATA_FORMAT;
         }

      }break;
   case MM::BeforeGet:
      {
         ret=DEVICE_OK;
      }break;
   }
   return ret; 
}

int SCCamera::convertPixelFormatToDepth(SC_EPixelType pixel) {
    switch (pixel) {
    case SC_EPixelType::gvspPixelMono12:
    case SC_EPixelType::gvspPixelMono16:
        return 16;

    default:
        LogMessage("invalid pixel");
        return 0;
    }
    return 0;
}

int32_t SCCamera::convertTriggerOutSeletorToEnum(const char* selector)
{
    if (strcmp(selector, g_triggerout1) == 0) return 1;
    if (strcmp(selector, g_triggerout2) == 0) return 2;
    if (strcmp(selector, g_triggerout3) == 0) return 3;
    return 0;
}

std::string SCCamera::convertTriggerOutSelectorToVal(int32_t selector)
{
    if (selector == 1) return g_triggerout1;
    if (selector == 2) return g_triggerout2;
    if (selector == 3) return g_triggerout3;
    return "";
}

int32_t SCCamera::convertTriggerOutTypeToEnum(const char* type)
{
    for (int32_t i = 0; i < triggerOutTypes_.size(); ++i) {
        if (strcmp(triggerOutTypes_[i].c_str(), type) == 0) return i;
    }
    return -1;
}

int SCCamera::convertBinningModeToBinningValue(int binningMode) const
{
    if (binningMode == 0) return 1; // off
    if (binningMode == 1) return 2; // 2*2
    if (binningMode == 2) return 4; // 4*4
    return 1;
}

int SCCamera::convertBinningValueToBinningMode(int binningValue) const
{
    if (binningValue == 1) return 0; // off
    if (binningValue == 2) return 1; // 2*2
    if (binningValue == 4) return 2; // 4*4
    return 0;
}

uint32_t SCCamera::getReadOutModeEnum(const char* vvStr) const
{
    for (uint32_t i = 0; i < readOutModes_.size(); ++i) {
        if (strcmp(vvStr, readOutModes_[i].c_str()) == 0) {
            return i;
        }
    }
    return (uint32_t)readOutModes_.size();
}

uint32_t SCCamera::convertReadOutModeEnum(const char* name, const char* vvStr) const
{
    if (strcmp(vvStr, "bit11_HS_Low") == 0) return bit11_HS_Low;
	if (strcmp(vvStr, "bit11_HS_High") == 0) return bit11_HS_High;
	if (strcmp(vvStr, "bit11_HDR_Low") == 0) return bit11_HDR_Low;
	if (strcmp(vvStr, "bit11_HDR_High") == 0) return bit11_HDR_High;
	if (strcmp(vvStr, "bit12_HDR_Low") == 0) return bit12_HDR_Low;
	if (strcmp(vvStr, "bit12_HDR_High") == 0) return bit12_HDR_High;
	if (strcmp(vvStr, "bit12_CMS") == 0) return bit12_CMS;
	if (strcmp(vvStr, "bit16_From11") == 0) return bit16_From11;
	if (strcmp(vvStr, "bit16_From12") == 0) return bit16_From12;
	//if (strcmp(vvStr, "bit11_HDR_2F") == 0) return 9;
	//if (strcmp(vvStr, "bit12_HDR_2F") == 0) return 10;
	//if (strcmp(vvStr, "CMS_2F") == 0) return 11;
    return 0;
}

std::string SCCamera::convertReadOutModeStr(const char* name, uint64_t vv) const
{
    switch (vv) {
        case bit11_HS_Low:
            return "bit11_HS_Low";
        case bit11_HS_High:
            return "bit11_HS_High";
        case bit11_HDR_Low:
            return "bit11_HDR_Low";
        case bit11_HDR_High:
            return "bit11_HDR_High";
        case bit12_HDR_Low:
            return "bit12_HDR_Low";
        case bit12_HDR_High:
            return "bit12_HDR_High";
        case bit12_CMS:
            return "bit12_CMS";
        case bit16_From11:
            return "bit16_From11";
        case bit16_From12:
            return "bit16_From12";
        default:
            return "bit11_HS_Low";
    }
}

std::string SCCamera::convertTriggerTypeToValue(int32_t type)
{
    if (type == 0) return g_trigger_off;
    if (type == 1) return g_trigger_external_edge;
    if (type == 3) return g_trigger_external_level;
    return g_trigger_off;
}

int32_t SCCamera::convertTriggerValueToType(const char* value) {
    if (strcmp(value, g_trigger_off) == 0) return 0;
    if (strcmp(value, g_trigger_external_edge) == 0) return 1;
    if (strcmp(value, g_trigger_external_level) == 0) return 3;
    return 0;
}

int SCCamera::GetProperty(const char* name, char* value) const
{
    if (strcmp(g_feature_readoutMode, name) == 0) {
        uint64_t vv;
        auto rslt = SC_GetEnumFeatureValue(devHandle_, name, &vv);
        if (rslt != SC_OK) {
            LogMessage("getProperty failed");
            return DEVICE_OK;
        }
		auto valueStr = convertReadOutModeStr(name, vv);
		memcpy(value, valueStr.data(), valueStr.length());
		return DEVICE_OK;
    }
    return CDeviceBase::GetProperty(name, value);
}


int SCCamera::SetProperty(const char* name, const char* value)
{
	int32_t rslt = SC_ERROR;
	if (strcmp(g_feature_readoutMode, name) == 0) {
		auto mode = convertReadOutModeEnum(name, value);
		rslt = SC_SetEnumFeatureValue(devHandle_, name, mode);
	}

    else if (strcmp(g_feature_triggerOutSelector, name) == 0) {
        auto selector = convertTriggerOutSeletorToEnum(value);
		rslt = SC_SetEnumFeatureValue(devHandle_, name, selector);
    }

	else if (strcmp(g_feature_triggerInType, name) == 0) {
		auto type = convertTriggerValueToType(value);
		rslt = SC_SetEnumFeatureValue(devHandle_, name, type);
	}

	else if (strcmp(g_feature_triggerActivation, name) == 0 || strcmp(g_feature_triggerOutActivation, name) == 0) {
		for (uint32_t i = 0; i < triggerActivations_.size(); ++i) {
			if (strcmp(triggerActivations_[i].data(), value) == 0) {
				rslt = SC_SetEnumFeatureValue(devHandle_, name, i);
			}
		}
	}
	else if (strcmp(g_feature_triggerDelay, name) == 0 || strcmp(g_feature_triggerOutDelay, name) == 0) {
		double vv = atof(value);
		rslt = SC_SetFloatFeatureValue(devHandle_, name, vv);
	}

	else if (strcmp(g_feature_triggerOutPulseWidth, name) == 0) {
		double vv = atof(value);
		rslt = SC_SetFloatFeatureValue(devHandle_, name, vv);
	}
    else if (strcmp(g_feature_triggerOutType, name) == 0) {
        auto type = convertTriggerOutTypeToEnum(value);
        rslt = SC_SetEnumFeatureValue(devHandle_, name, type);
    }
    else if (strcmp("Binning", name) == 0) {
        int32_t vv = atoi(value);
        rslt = SetBinning(vv);
    }
    else {
        rslt = SC_OK;
    }

	if (rslt != SC_OK) {
		LogMessage("setProperty failed");
		return DEVICE_OK;
	}
    return CDeviceBase::SetProperty(name, value);
}

int32_t SCCamera::createTriggerOutProperty()
{
	uint64_t selector = 1;
    auto rslt = SC_SetEnumFeatureValue(devHandle_, g_feature_triggerOutSelector, selector);
    assert(rslt == SC_OK);
 
    // trigger out selector
    std::vector<std::string> selectors;
    std::string value = convertTriggerOutSelectorToVal((uint32_t)selector);
    auto pAct = new CPropertyAction(this, &SCCamera::OnTriggerSelector);
	auto nRet = CreateProperty(g_feature_triggerOutSelector, value.c_str(), MM::String, false, pAct);
	assert(nRet == DEVICE_OK);
    selectors.push_back(g_triggerout1);
    selectors.push_back(g_triggerout2);
    selectors.push_back(g_triggerout3);
    SetAllowedValues(g_feature_triggerOutSelector, selectors);

    // trigger out type
    uint64_t triggerOutType = 0;
    rslt = SC_GetEnumFeatureValue(devHandle_, g_feature_triggerOutType, &triggerOutType);
    assert(rslt == SC_OK);

	triggerOutTypes_.push_back(g_trigger_exposure_start);
	triggerOutTypes_.push_back(g_trigger_vsync);
	triggerOutTypes_.push_back(g_trigger_readout_end);
	triggerOutTypes_.push_back(g_trigger_ready);
	triggerOutTypes_.push_back(g_trigger_global_exposure);
	triggerOutTypes_.push_back(g_trigger_high);
	triggerOutTypes_.push_back(g_trigger_low);

	value = g_trigger_exposure_start;
	if (triggerOutType < triggerOutTypes_.size()) value = triggerOutTypes_[triggerOutType];
	pAct = new CPropertyAction(this, &SCCamera::OnTriggerOut);
    nRet = CreateProperty(g_feature_triggerOutType, value.c_str(), MM::String, false, pAct);
	assert(nRet == DEVICE_OK);

	nRet = SetAllowedValues(g_feature_triggerOutType, triggerOutTypes_);
	if (nRet != DEVICE_OK)
		return nRet;

	// trigger out activation
	uint64_t triggerActivation = 0;
	value = g_trigger_rising_edge;
	rslt = SC_GetEnumFeatureValue(devHandle_, g_feature_triggerOutActivation, &triggerActivation);
	if (rslt == SC_OK && triggerActivation < triggerActivations_.size()) {
		value = triggerActivations_[triggerActivation];
	}

	nRet = CreateProperty(g_feature_triggerOutActivation, value.c_str(), MM::String, false);
	assert(nRet == DEVICE_OK);

	// trigger out delay
    double delay = 0;
    rslt = SC_GetFloatFeatureValue(devHandle_, g_feature_triggerOutDelay, &delay);

	nRet = CreateProperty(g_feature_triggerOutDelay, std::to_string(delay).c_str(), MM::Float, false);
	assert(nRet == DEVICE_OK);

	// trigger out pulse width 
    double pulseWidth = 0;
	rslt = SC_GetFloatFeatureValue(devHandle_, g_feature_triggerOutPulseWidth, &delay);

	nRet = CreateProperty(g_feature_triggerOutPulseWidth, std::to_string(pulseWidth).c_str(), MM::Float, false);
	assert(nRet == DEVICE_OK);

    return DEVICE_OK;
    
}

int SCCamera::createTriggerProperty()
{
   uint64_t triggerInType = 0;
   auto rslt = SC_GetEnumFeatureValue(devHandle_, g_feature_triggerInType, &triggerInType);
   if (rslt != SC_OK) {
       LogMessage("set trigger off failed");
   }
   std::string value = convertTriggerTypeToValue((uint32_t)triggerInType);
   auto pAct = new CPropertyAction (this, &SCCamera::OnTriggerIn);
   auto nRet = CreateProperty(g_feature_triggerInType, value.c_str(), MM::String, false, pAct);
   assert(nRet == DEVICE_OK);

   std::vector<std::string> triggerInValues;
   triggerInValues.push_back(g_trigger_off);
   triggerInValues.push_back(g_trigger_external_edge);
   triggerInValues.push_back(g_trigger_external_level);

   nRet = SetAllowedValues(g_feature_triggerInType, triggerInValues);
   if (nRet != DEVICE_OK)
      return nRet;

   // trigger activation
   uint64_t triggerActivation = 0;
   value = g_trigger_rising_edge;
   triggerActivations_.push_back(g_trigger_rising_edge);
   triggerActivations_.push_back(g_trigger_falling_edge);
   triggerActivations_.push_back(g_trigger_level_high);
   triggerActivations_.push_back(g_trigger_level_low);
   rslt = SC_GetEnumFeatureValue(devHandle_, g_feature_triggerActivation, &triggerActivation);
   if (rslt == SC_OK && triggerActivation < triggerActivations_.size()) {
       value = triggerActivations_[triggerActivation];
   }

   nRet = CreateProperty(g_feature_triggerActivation, value.c_str(), MM::String, false);
   assert(nRet == DEVICE_OK);
   
   // trigger delay
   double delay = 0;
   rslt = SC_GetFloatFeatureValue(devHandle_, g_feature_triggerDelay, &delay);

   nRet = CreateProperty(g_feature_triggerDelay, std::to_string(delay).c_str(), MM::Float, false);
   assert(nRet == DEVICE_OK);
    
   return DEVICE_OK;
}

int SCCamera::ThreadRun() {
     LogMessage("ThreadRun");
     return 0;
}

CameraRecvThread::CameraRecvThread(SCCamera* pCam)
    :camera_(pCam)
	,stop_(true)
	,suspend_(false)
	,numImages_(default_numImages)
	,imageCounter_(0)
	,intervalMs_(default_intervalMS)
	,startTime_(0)
	,actualDuration_(0)
    ,lastFrameTime_(0) 
{
}

CameraRecvThread::~CameraRecvThread() {
}

void CameraRecvThread::Stop() {
	MMThreadGuard(this->stopLock_);
	stop_ = true;
}

void CameraRecvThread::Start(long numImages, double intervalMs)
{
	MMThreadGuard(this->stopLock_);
	MMThreadGuard(this->suspendLock_);
	numImages_ = numImages;
	intervalMs_ = intervalMs;
	imageCounter_ = 0;
	stop_ = false;
	suspend_ = false;
	activate();
	actualDuration_ = MM::MMTime{};
	startTime_ = camera_->GetCurrentMMTime();
	lastFrameTime_ = MM::MMTime{};
}

bool CameraRecvThread::IsStopped() {
	MMThreadGuard(this->stopLock_);
	return stop_;
}
void CameraRecvThread::Suspend() {
	MMThreadGuard(this->suspendLock_);
	suspend_ = true;
}
bool CameraRecvThread::IsSuspended() {
	MMThreadGuard(this->suspendLock_);
	return suspend_;
}
void CameraRecvThread::Resume() {
	MMThreadGuard(this->suspendLock_);
	suspend_ = false;
}

double CameraRecvThread::GetIntervalMs() { 
    return intervalMs_; 
}

void CameraRecvThread::SetLength(long images) { 
    numImages_ = images; 
}

long CameraRecvThread::GetLength() const { 
    return numImages_; 
}

long CameraRecvThread::GetImageCounter() { 
    return imageCounter_; 
}

MM::MMTime CameraRecvThread::GetStartTime() { 
    return startTime_; 
}

MM::MMTime CameraRecvThread::GetActualDuration() { 
    return actualDuration_; 
}

int CameraRecvThread::svc(void) throw()
{
	int ret = DEVICE_ERR;
	try
	{
		MM::MMTime startTime = camera_->GetCurrentMMTime();
		do
		{
          ret = camera_->RunSequenceOnThread(startTime);
		} while (DEVICE_OK == ret && !IsStopped() && imageCounter_++ < numImages_ - 1);

        if (IsStopped()) {
            camera_->LogMessage("SeqAcquisition interrupted by the user\n");
        }
    }
	catch (...) {
		camera_->LogMessage(g_Msg_EXCEPTION_IN_THREAD, false);
	}
	stop_ = true;
	actualDuration_ = camera_->GetCurrentMMTime() - startTime_;
	camera_->OnThreadExiting();
	return ret;
}
