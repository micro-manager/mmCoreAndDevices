#include "ClassGalaxy.h"
#include <condition_variable>

#include <iostream>

#include <exception>
#include <typeinfo>
#include <stdexcept>

using namespace std;

const char* g_CameraDeviceName = "DahengCamera";

static const char* g_PropertyChannel = "PropertyNAme";

static const char* g_PixelType_8bit = "8bit mono";
static const char* g_PixelType_10bit = "10bit mono";
static const char* g_PixelType_12bit = "12bit mono";
static const char* g_PixelType_16bit = "16bit mono";
static const char* g_PixelType_10packedbit = "10bit mono";
static const char* g_PixelType_12packedbit = "12bit mono";

static const char* g_PixelType_8bitRGBA = "8bitBGRA";
static const char* g_PixelType_8bitRGB =  "8bitRGB";
static const char* g_PixelType_8bitBGR =  "8bitBGR";



MODULE_API void InitializeModuleData()
{
    RegisterDevice(g_CameraDeviceName, MM::CameraDevice, "Daheng Camera");
}

//Camera Device
MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
    if (deviceName == 0)
        return 0;

    // decide which device class to create based on the deviceName parameter 比较结果
    if (strcmp(deviceName, g_CameraDeviceName) == 0) {
        // create camera
        return new ClassGalaxy();
    }
    // ...supplied name not recognized
    return 0;
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
    delete pDevice;
}

ClassGalaxy::ClassGalaxy() :
    CLegacyCameraBase<ClassGalaxy>(),
        ImageHandler_(0),
        Width_(0),
        Height_(0),
        imageBufferSize_(0),
        maxWidth_(0),
        maxHeight_(0),
        DeviceLinkThroughputLimit_(0),
        exposure_us_(0),
        exposureMax_(0),
        exposureMin_(0),
        gain_(0),
        gainMax_(0),
        gainMin_(0),
        bitDepth_(8),
        bytesPerPixel_(1),
        temperatureState_("Undefined"),
        reverseX_("0"),
        reverseY_("0"),
        imgBuffer_(NULL),
        // Buffer4ContinuesShot(NULL),
        colorCamera_(false),
        pixelType_("Undefined"),
        sensorReadoutMode_("Undefined"),
        shutterMode_("None"),
        imgBufferSize_(0),
        sequenceRunning_(false),
        initialized_(false),
        exposureTimeoutS_(5)
{
        // call the base class method to set-up default error codes/messages
        InitializeDefaultErrorMessages();
        SetErrorText(ERR_CAMERA_SDK, "Error while interacting with the Daheng Galaxy SDK");

        IGXFactory::GetInstance().Init();
        IGXFactory::GetInstance().UpdateDeviceList(1000, vectorDeviceInfo);
        
        CreateStringProperty("SerialNumber", "Undefined", false, 0, true);

        //pre-init properties

        if (vectorDeviceInfo.size() <= 0)
        {
            LogMessage("No camera present.");
        }

        vector<string> SnString;
        bool first = false;
        string serialNumberstr;
        if (true)
        {
            for (size_t i = 0; i < vectorDeviceInfo.size(); i++)
            {
                serialNumberstr = vectorDeviceInfo[i].GetSN().c_str();
                AddAllowedValue("SerialNumber", serialNumberstr.c_str());
                SnString.push_back(serialNumberstr);
                first = true;
            }
            if (first)
            {
                SetProperty("SerialNumber", SnString[0].c_str());
                first = false;
            }
        }
}

ClassGalaxy::~ClassGalaxy(void)
{
   if (initialized_)
   {
      Shutdown();
   }
}

int ClassGalaxy::Initialize()
{
    if (initialized_)
        return DEVICE_OK;

    try
    {
        // Before calling any Galaxy SDK methods, the runtime must be initialized. 
        IGXFactory::GetInstance().Init();

        if (1)
        {
           char serialNumber[MM::MaxStrLength];
           int ret = GetProperty("SerialNumber", serialNumber);
           if (ret != DEVICE_OK) {
              return DEVICE_NOT_CONNECTED;
           }
        }

        vectorDeviceInfo.clear();
        //枚举设备
        IGXFactory::GetInstance().UpdateDeviceList(1000, vectorDeviceInfo);

        //判断枚举到的设备是否大于零，如果不是则弹框提示
        if (vectorDeviceInfo.size() <= 0)
        {
            return DEVICE_NOT_CONNECTED;
        }
        //获取可执行程序的当前路径,默认开启第一个
        initialized_ = false;
        // This checks, among other things, that if the camera is already in use.
        // Without that check, the following CreateDevice() may crash on duplicate
        // serial number. Unfortunately, this call is slow. 默认打开第一个设备
        int index = 0;

        string serialNumberstr = vectorDeviceInfo[index].GetSN().c_str();

        const char* serialNumber = serialNumberstr.c_str();

        if (strlen(serialNumber) == 0 || strcmp(serialNumber, "Undefined") == 0)
            return 0;
        SetProperty("SerialNumber", serialNumber);
        //打开设备
        m_objDevicePtr = IGXFactory::GetInstance().OpenDeviceBySN(vectorDeviceInfo[index].GetSN(), GX_ACCESS_MODE::GX_ACCESS_EXCLUSIVE);

        m_objFeatureControlPtr = m_objDevicePtr->GetRemoteFeatureControl();

        //m_objFeatureControlPtr->GetEnumFeature("StreamBufferHandlingMode")->SetValue("NewestOnly");
        //判断设备流是否大于零，如果大于零则打开流

        int nStreamCount = m_objDevicePtr->GetStreamCount();
        //CPropertyAction* pAct;

        if (nStreamCount > 0)
        {
            m_objStreamPtr = m_objDevicePtr->OpenStream(0);
            //m_objStreamPtr->SetAcqusitionBufferNumber(1);
            m_objStreamFeatureControlPtr = m_objStreamPtr->GetFeatureControl();
            m_objStreamFeatureControlPtr->GetEnumFeature("StreamBufferHandlingMode")->SetValue("NewestOnly");
            initialized_ = true;
        }
        else
        {
            throw exception("未发现设备流!");
        }


        GX_DEVICE_CLASS_LIST objDeviceClass = m_objDevicePtr->GetDeviceInfo().GetDeviceClass();
        if (GX_DEVICE_CLASS_GEV == objDeviceClass)  // GigE devices
        {
            // 判断设备是否支持流通道数据包功能
            if (true == m_objFeatureControlPtr->IsImplemented("GevSCPSPacketSize"))
            {
                // 获取当前网络环境的最优包长值
                int nPacketSize = m_objStreamPtr->GetOptimalPacketSize();
                // 将最优包长值设置为当前设备的流通道包长值
                CIntFeaturePointer GevSCPD = m_objFeatureControlPtr->GetIntFeature("GevSCPSPacketSize");
                m_objFeatureControlPtr->GetIntFeature("GevSCPSPacketSize")->SetValue(nPacketSize);
                m_objFeatureControlPtr->GetIntFeature("GevHeartbeatTimeout")->SetValue(300000);
                ////Inter packet delay for GigE Camera
                CPropertyAction* pAct = new CPropertyAction(this, &ClassGalaxy::OnInterPacketDelay);
                int ret = CreateProperty("InterPacketDelay", CDeviceUtils::ConvertToString((long)nPacketSize), MM::Integer, false, pAct);
                SetPropertyLimits("InterPacketDelay", (double)GevSCPD->GetMin(), (double)GevSCPD->GetMax());
                assert(ret == DEVICE_OK);
            }
            //第二个参数为用户私有参数，用户可以在回调函数内部将其还原然使用，如果不需要则可传入 NULL 即可
            //hDeviceOffline = m_objDevicePtr->RegisterDeviceOfflineCallback(pDeviceOfflineEventHandler, this);
        }
        else if (GX_DEVICE_CLASS_U3V == objDeviceClass) // USB 3.0 devices
        {
            CIntFeaturePointer DeviceLinkThroughputLimit = m_objFeatureControlPtr->GetIntFeature("DeviceLinkThroughputLimit");
            if (1)
            {
                DeviceLinkThroughputLimit_ = DeviceLinkThroughputLimit->GetValue();
                CPropertyAction* pAct = new CPropertyAction(this, &ClassGalaxy::OnDeviceLinkThroughputLimit);

                int ret = CreateProperty("DeviceLinkThroughputLimit", CDeviceUtils::ConvertToString((long)DeviceLinkThroughputLimit_), MM::Integer, false, pAct);
                SetPropertyLimits("DeviceLinkThroughputLimit", (double)DeviceLinkThroughputLimit->GetMin(), (double)DeviceLinkThroughputLimit->GetMax());
                assert(ret == DEVICE_OK);
            }
        }

        //颜色判断
        gxstring strValue = "";
        if (m_objDevicePtr->GetRemoteFeatureControl()->IsImplemented("PixelColorFilter"))
        {
            strValue = m_objDevicePtr->GetRemoteFeatureControl()->GetEnumFeature("PixelColorFilter")->GetValue();

            if ("None" != strValue)
            {
                colorCamera_ = true;

            }
        }

        // Name
        stringstream msg;
        msg << "using camera " << m_objFeatureControlPtr->GetStringFeature("DeviceUserID")->GetValue();
        LogMessage(msg.str());

        int ret = CreateProperty(MM::g_Keyword_Name, g_CameraDeviceName, MM::String, true);
        if (DEVICE_OK != ret)
            return ret;

        // Description
        ret = CreateProperty(MM::g_Keyword_Description, "Daheng Camera device adapter", MM::String, true);
        if (DEVICE_OK != ret)
            return ret;

        // Serial Number
        ret = CreateProperty(MM::g_Keyword_CameraID, serialNumber, MM::String, true);
        if (DEVICE_OK != ret)
            return ret;

        //Get information about camera (e.g. height, width, byte depth)
        //check if given Camera support event. //Register Camera events
        //赵伟甫：注册相机事件，注册采集回调-未加 温度事件

        CIntFeaturePointer width = m_objFeatureControlPtr->GetIntFeature("Width");
        CIntFeaturePointer height = m_objFeatureControlPtr->GetIntFeature("Height");

        //总属性器
        if (1)
        {
            CPropertyAction* pAct = new CPropertyAction(this, &ClassGalaxy::OnWidth);
            ret = CreateProperty("SensorWidth", CDeviceUtils::ConvertToString((int)width->GetValue()), MM::Integer, false, pAct);
            SetPropertyLimits("SensorWidth", (double)width->GetMin(), (double)width->GetMax());
            assert(ret == DEVICE_OK);
        }

        if (1)
        {
            CPropertyAction* pAct = new CPropertyAction(this, &ClassGalaxy::OnHeight);
            ret = CreateProperty("SensorHeight", CDeviceUtils::ConvertToString((int)height->GetValue()), MM::Integer, false, pAct);
            SetPropertyLimits("SensorHeight", (double)height->GetMin(), (double)height->GetMax());
            assert(ret == DEVICE_OK);
        }

        maxWidth_ = (unsigned int) m_objFeatureControlPtr->GetIntFeature("WidthMax")->GetValue();
        maxHeight_ = (unsigned int) m_objFeatureControlPtr->GetIntFeature("HeightMax")->GetValue();


        //end of Sensor size
        long bytes = (long)(height->GetValue() * width->GetValue() * 4);
        //20221020赵伟甫
        //Buffer4ContinuesShot = malloc(bytes);


        CFloatFeaturePointer exposure = m_objFeatureControlPtr->GetFloatFeature("ExposureTime");
        exposure_us_ = exposure->GetValue();
        exposureMax_ = exposure->GetMax();
        exposureMin_ = exposure->GetMin();
        CPropertyAction* pAct = new CPropertyAction(this, &ClassGalaxy::OnExposure);
        ret = CreateProperty("Exposure(us)", CDeviceUtils::ConvertToString((long)exposure->GetValue()), MM::Float, false, pAct);
        SetPropertyLimits("Exposure(us)", exposureMin_, exposureMax_);
        assert(ret == DEVICE_OK);

        // List all camera features in the logs
        GxIAPICPP::gxstring_vector vectorFeatureNameList;
        m_objFeatureControlPtr->GetFeatureNameList(vectorFeatureNameList); 
        for (gxstring featureName : vectorFeatureNameList)
        {
           std::string feature = featureName.c_str();
           LogMessage("Camera Feature: " + feature, false);
        }

        pAct = new CPropertyAction(this, &ClassGalaxy::OnPixelType);
        ret = CreateProperty(MM::g_Keyword_PixelType, "NA", MM::String, false, pAct);
        assert(ret == DEVICE_OK);
        vector<string> pixelTypeValues;
        CEnumFeaturePointer PixelFormatList = m_objFeatureControlPtr->GetEnumFeature("PixelFormat");
        gxstring_vector LisePixelFormat = PixelFormatList->GetEnumEntryList();
        //为了赋值用
        for (size_t i = 0; i < LisePixelFormat.size(); i++)
        {
            string strValue(LisePixelFormat[i]);
            pixelTypeValues.push_back(strValue);
            
        }
        pixelType_ = PixelFormatList->GetValue();
        SetAllowedValues(MM::g_Keyword_PixelType, pixelTypeValues);
        CEnumFeaturePointer pixelFormat_ = m_objFeatureControlPtr->GetEnumFeature("PixelFormat");
        SetProperty(MM::g_Keyword_PixelType, pixelFormat_->GetValue().c_str());


        if (1)
        {
            // ResultingFrameRatePrevious = 8;
            // acqFramerate_ = 8, acqFramerateMax_ = 8, acqFramerateMin_ = 0.1;
            gain_ = 8, gainMax_ = 8, gainMin_ = 0;
            offset_ = 0, offsetMin_ = 0, offsetMax_ = 8;

            binningFactor_ = "1";
            reverseX_ = "0"; reverseY_ = "0";
            sensorReadoutMode_ = "Undefined";
            setAcqFrm_ = "";
            shutterMode_ = "None";
            temperature_ = "";
            temperatureState_ = "Undefined";
        }

        if (m_objDevicePtr->GetRemoteFeatureControl()->IsImplemented("BinningHorizontal")
            && m_objDevicePtr->GetRemoteFeatureControl()->IsImplemented("BinningVertical"))
        {
           CIntFeaturePointer BinningHorizontal = m_objFeatureControlPtr->GetIntFeature("BinningHorizontal");
           // Vertical ignored here, assume it acts the same as horizontal
           // CIntFeaturePointer BinningVertical = m_objFeatureControlPtr->GetIntFeature("BinningVertical");

           pAct = new CPropertyAction(this, &ClassGalaxy::OnBinning);
           binningFactor_.assign(CDeviceUtils::ConvertToString((long)BinningHorizontal->GetValue()));
           ret = CreateProperty(MM::g_Keyword_Binning, binningFactor_.c_str(), MM::String, false, pAct);
           //assumed that BinningHorizontal and BinningVertical allow same steps
           bool Isgrabbing = m_objStreamFeatureControlPtr->GetBoolFeature("StreamIsGrabbing")->GetValue();
           if (Isgrabbing)
           {
              m_objFeatureControlPtr->GetCommandFeature("AcquisitionStart")->Execute();
              m_objStreamPtr->StartGrab();
           }
           int64_t binVal = BinningHorizontal->GetValue();
           for (int64_t i = BinningHorizontal->GetMin(); i <= BinningHorizontal->GetMax(); i++)
           {
              try
              {
                 BinningHorizontal->SetValue(i);
                 AddAllowedValue(MM::g_Keyword_Binning, std::to_string(i).c_str());
              }
              catch (CGalaxyException&)
              {
                   // ignore, simply means we can not use this value to set binning
              }
           }
           BinningHorizontal->SetValue(binVal);
           if (Isgrabbing)
           {
              m_objFeatureControlPtr->GetCommandFeature("AcquisitionStart")->Execute();
              m_objStreamPtr->StartGrab();
           }
           CheckForBinningMode(pAct);
        }
        else 
        { // Hardware does not support binning, Micro-Manager still likes this as a property
           pAct = new CPropertyAction(this, &ClassGalaxy::OnBinning);
           ret = CreateProperty(MM::g_Keyword_Binning, "1", MM::Integer, true, pAct);
           AddAllowedValue(MM::g_Keyword_Binning, "1");
        }

        ///// Trigger Mode //////
        CEnumFeaturePointer TriggerMode = m_objFeatureControlPtr->GetEnumFeature("TriggerMode");
        if (!TriggerMode.IsNull())
        {
            pAct = new CPropertyAction(this, &ClassGalaxy::OnTriggerMode);
            ret = CreateProperty("TriggerMode", "Off", MM::String, false, pAct);
            vector<string> LSPVals;
            LSPVals.push_back("Off");
            LSPVals.push_back("On");
            SetAllowedValues("TriggerMode", LSPVals);
        }

        ///// Trigger Source //////
        CEnumFeaturePointer triggersource = m_objFeatureControlPtr->GetEnumFeature("TriggerSource");
        if (!triggersource.IsNull())
        {
            pAct = new CPropertyAction(this, &ClassGalaxy::OnTriggerSource);
            ret = CreateProperty("TriggerSource", "NA", MM::String, false, pAct);
            vector<string> LSPVals;
            gxstring_vector entries = triggersource->GetEnumEntryList();
            for (size_t i = 0; i < entries.size(); i++)
            {
                string strValue(entries[i]);
                LSPVals.push_back(strValue);
            }
            SetAllowedValues("TriggerSource", LSPVals);
        }

        CEnumFeaturePointer lineselector = m_objFeatureControlPtr->GetEnumFeature("LineSelector");
        if (!lineselector.IsNull())
        {
           gxstring lineVal = lineselector->GetValue();
           gxstring_vector entries = lineselector->GetEnumEntryList();
           for (size_t i = 0; i < entries.size(); i++)
           {
              string entry(entries[i]);
              GetCoreCallback()->LogMessage(this, entry.c_str(), false);
              // Let's hope there are never more than 10 lines
              std::string lastPart = entry.substr(entry.length() - 1);
              long lineNr = std::stoi(lastPart);
              lineselector->SetValue(entries[i]);
              CEnumFeaturePointer linemodes = m_objFeatureControlPtr->GetEnumFeature("LineMode");
              if (!linemodes.IsNull()) {
                 gxstring_vector modeentries = linemodes->GetEnumEntryList();
                 if (modeentries.size() > 0) {
                    CPropertyActionEx* pActEx = new CPropertyActionEx(this, &ClassGalaxy::OnLineMode, lineNr);
                    std::string propName = entry + "-Mode";
                    ret = CreateProperty(propName.c_str(),  modeentries[0].c_str(), MM::String, false, pActEx);
                    if (ret != DEVICE_OK)
                       return ret;
                    for (size_t j = 0; j < modeentries.size(); j++)
                    {
                       AddAllowedValue(propName.c_str(), modeentries[j].c_str());
                    }
                 }
              }
              CEnumFeaturePointer linesources = m_objFeatureControlPtr->GetEnumFeature("LineSource");
              if (!linesources.IsNull())
              {
                 gxstring_vector sourceentries = linesources->GetEnumEntryList();
                 if (sourceentries.size() > 0) {
                    CPropertyActionEx* pActEx = new CPropertyActionEx(this, &ClassGalaxy::OnLineSource, lineNr);
                    std::string propName = entry + "-Source";
                    CEnumFeaturePointer linemodes = m_objFeatureControlPtr->GetEnumFeature("LineMode");
                    if (linemodes.IsNull())
                    {
                       return ERR_CAMERA_SDK;
                    }
                    gxstring_vector modeentries = linemodes->GetEnumEntryList();
                    bool readOnly = false;
                    if (modeentries.size() == 1 && modeentries[0] == "Input")
                       readOnly = true;
                    ret = CreateProperty(propName.c_str(),  sourceentries[0].c_str(), MM::String, readOnly, pActEx);
                    if (ret != DEVICE_OK)
                       return ret;
                    for (size_t j = 0; j < sourceentries.size(); j++)
                    {
                       AddAllowedValue(propName.c_str(), sourceentries[j].c_str());
                    }
                 }

              }
           }

           // set line selector back 
           lineselector->SetValue(lineVal);
        }

        //20230217设置期望帧率使能
        if (m_objDevicePtr->GetRemoteFeatureControl()->IsImplemented("AcquisitionFrameRateMode"))
        {
            CEnumFeaturePointer AdjFrameRateMode = m_objFeatureControlPtr->GetEnumFeature("AcquisitionFrameRateMode");
            pAct = new CPropertyAction(this, &ClassGalaxy::OnAdjFrameRateMode);
            ret = CreateProperty("AcquisitionFrameRateMode", "NA", MM::String, false, pAct);
            vector<string> LSPVals;
            gxstring_vector entries = AdjFrameRateMode->GetEnumEntryList();
            for (size_t i = 0; i < entries.size(); i++)
            {
                string strValue(entries[i]);
                LSPVals.push_back(strValue);
            }
            SetAllowedValues("AcquisitionFrameRateMode", LSPVals);
        }

        // AcquisitionFrameRate
        if (m_objDevicePtr->GetRemoteFeatureControl()->IsImplemented("AcquisitionFrameRate"))
        {
            CFloatFeaturePointer AdjFrameRate = m_objFeatureControlPtr->GetFloatFeature("AcquisitionFrameRate");
            pAct = new CPropertyAction(this, &ClassGalaxy::OnAcquisitionFrameRate);
            ret = CreateProperty("AcquisitionFrameRate", CDeviceUtils::ConvertToString((float)0), MM::Float, false, pAct);
            //当前采集帧率需重新计算
            SetPropertyLimits("AcquisitionFrameRate", (double)AdjFrameRate->GetMin(), (double)AdjFrameRate->GetMax());
            assert(ret == DEVICE_OK);
        }

        // TriggerSelector.  Should this be exposed as a property?
        if (m_objDevicePtr->GetRemoteFeatureControl()->IsImplemented("TriggerSelector"))
        {
           m_objFeatureControlPtr->GetEnumFeature("TriggerSelector")->SetValue("FrameStart");
        }

        // TriggerActivation
        if (m_objDevicePtr->GetRemoteFeatureControl()->IsImplemented("TriggerActivation"))
        {
            CEnumFeaturePointer adjTriggerSelector = m_objFeatureControlPtr->GetEnumFeature("TriggerActivation");
            pAct = new CPropertyAction(this, &ClassGalaxy::OnTriggerActivation);
            ret = CreateProperty("TriggerActivation", "NA", MM::String, false, pAct);
            vector<string> LSPVals;
            gxstring_vector entries = adjTriggerSelector->GetEnumEntryList();
            for (size_t i = 0; i < entries.size(); i++)
            {
                string strValue(entries[i]);
                LSPVals.push_back(strValue);
            }
            SetAllowedValues("TriggerActivation", LSPVals);
            m_objFeatureControlPtr->GetEnumFeature("TriggerActivation")->SetValue("RisingEdge");
        }

        // TriggerDelay
        if (m_objDevicePtr->GetRemoteFeatureControl()->IsImplemented("TriggerDelay"))
        {
            CFloatFeaturePointer TriggerDelay = m_objFeatureControlPtr->GetFloatFeature("TriggerDelay");
            pAct = new CPropertyAction(this, &ClassGalaxy::OnTriggerDelay);
            ret = CreateProperty("TriggerDelay", CDeviceUtils::ConvertToString((double)0), MM::Integer, false, pAct);
            SetPropertyLimits("TriggerDelay", (double)TriggerDelay->GetMin(), (double)TriggerDelay->GetMax());
            assert(ret == DEVICE_OK);
        }

        // TriggerFilterRaisingEdge
        if (m_objDevicePtr->GetRemoteFeatureControl()->IsImplemented("TriggerFilterRaisingEdge"))
        {
            CFloatFeaturePointer TriggerFilterRaisingEdge = m_objFeatureControlPtr->GetFloatFeature("TriggerFilterRaisingEdge");
            pAct = new CPropertyAction(this, &ClassGalaxy::OnTriggerFilterRaisingEdge);
            ret = CreateProperty("TriggerFilterRaisingEdge", CDeviceUtils::ConvertToString((double)0), MM::Integer, false, pAct);
            SetPropertyLimits("TriggerFilterRaisingEdge", (double)TriggerFilterRaisingEdge->GetMin(), (double)TriggerFilterRaisingEdge->GetMax());
            assert(ret == DEVICE_OK);
        }

        //增益
        m_objFeatureControlPtr->GetEnumFeature("GainSelector")->SetValue("AnalogAll");
        m_objFeatureControlPtr->GetFloatFeature("Gain")->SetValue(0.0000);
        double d = m_objFeatureControlPtr->GetFloatFeature("Gain")->GetValue();
        if (m_objDevicePtr->GetRemoteFeatureControl()->IsImplemented("Gain"))
        {
            CFloatFeaturePointer Gain = m_objFeatureControlPtr->GetFloatFeature("Gain");

            pAct = new CPropertyAction(this, &ClassGalaxy::OnGain);
            ret = CreateProperty("Gain", CDeviceUtils::ConvertToString((double)0), MM::Integer, false, pAct);
            SetPropertyLimits("Gain", (double)Gain->GetMin(), (double)Gain->GetMax());
            assert(ret == DEVICE_OK);
        }

        // UserOutputSelector
        if (m_objDevicePtr->GetRemoteFeatureControl()->IsImplemented("UserOutputSelector"))
        {
           CEnumFeaturePointer useroutput = m_objFeatureControlPtr->GetEnumFeature("UserOutputSelector");
            m_objFeatureControlPtr->GetEnumFeature("UserOutputSelector")->SetValue("UserOutput0");
            pAct = new CPropertyAction(this, &ClassGalaxy::OnUserOutput);
            ret = CreateProperty("UserOutputSelector", "UserOutput0", MM::String, false, pAct);
            vector<string> LSPVals;
            gxstring_vector entries = useroutput->GetEnumEntryList();
            for (size_t i = 0; i < entries.size(); i++)
            {
                string strValue(entries[i]);
                LSPVals.push_back(strValue);
            }
            SetAllowedValues("UserOutputSelector", LSPVals);
        }

        pAct = new CPropertyAction(this, &ClassGalaxy::OnExposureTimeout);
        ret = CreateIntegerProperty("ExposureTimeoutSeconds", 5, false, pAct);

        //20230220设置图像转换RGBA8
        ret = UpdateStatus();

        if (ret != DEVICE_OK)
            return ret;

        //preparation for snaps
        ResizeSnapBuffer();

        /////////////////////////////////////////////////////////////////////////////////////////
        initialized_ = true;
        return DEVICE_OK;
    }

   catch (CGalaxyException gex) {
      return HandleError(gex);
   }

   return DEVICE_ERR;

}

void ClassGalaxy::CoverToRGB(GX_PIXEL_FORMAT_ENTRY emDstFormat,void* DstBuffer, CImageDataPointer pObjSrcImageData)
{
    try
    {
        TestFormatConvertPtr = IGXFactory::GetInstance().CreateImageFormatConvert();
        TestFormatConvertPtr->SetDstFormat(emDstFormat);
        TestFormatConvertPtr->SetInterpolationType(GX_RAW2RGB_NEIGHBOUR);
        TestFormatConvertPtr->SetAlphaValue(255);
        uint64_t Size = TestFormatConvertPtr->GetBufferSizeForConversion(pObjSrcImageData);
        TestFormatConvertPtr->Convert(pObjSrcImageData, DstBuffer, Size, false); //modify by LXM in 20240305
        //注意都能显示16位图像RGB
    }
   catch (CGalaxyException gex) {
      HandleError(gex);
   }
}

int ClassGalaxy::CheckForBinningMode(CPropertyAction* pAct)
{
    // Binning Mode
    //INodeMap& nodeMap(camera_->GetNodeMap());
    CEnumFeaturePointer BinningHorizontalMode = m_objFeatureControlPtr->GetEnumFeature("BinningHorizontalMode");
    vector<string> LSPVals;
    gxstring_vector LiseBinningHorizontalMode = BinningHorizontalMode->GetEnumEntryList();
    //为了赋值用
    for (size_t i = 0; i < LiseBinningHorizontalMode.size(); i++)
    {
        string strValue(LiseBinningHorizontalMode[i]);
        LSPVals.push_back(strValue);
    }
    SetAllowedValues("BinningHorizontalMode", LSPVals);

    CEnumFeaturePointer BinningVerticalMode = m_objFeatureControlPtr->GetEnumFeature("BinningVerticalMode");
    gxstring_vector LiseBinningVerticalMode = BinningVerticalMode->GetEnumEntryList();
    for (size_t i = 0; i < LiseBinningVerticalMode.size(); i++)
    {
       string strValue(LiseBinningVerticalMode[i]);
       LSPVals.push_back(strValue);
    }
    SetAllowedValues("BinningVerticalMode", LSPVals);

    return DEVICE_OK;
}

int ClassGalaxy::OnBinningMode(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    CEnumFeaturePointer BinningHorizontalMode = m_objFeatureControlPtr->GetEnumFeature("BinningHorizontalMode");
    CEnumFeaturePointer BinningVerticalMode = m_objFeatureControlPtr->GetEnumFeature("BinningVerticalMode");

    if (eAct == MM::AfterSet)
    {
       try
       {
           string binningMode;
           pProp->Get(binningMode);
           BinningHorizontalMode->SetValue(binningMode.c_str());
           BinningVerticalMode->SetValue(binningMode.c_str());
       }
       catch (CGalaxyException gex) {
          HandleError(gex);
       }
    }
    else if (eAct == MM::BeforeGet)
    {
        try {
           pProp->Set(((BinningVerticalMode->GetValue()).c_str()));
        }
        catch (CGalaxyException gex) {
           HandleError(gex);
        }
    }
    return DEVICE_OK;
}

int ClassGalaxy::Shutdown()
{
    try
    {
        if (m_objStreamFeatureControlPtr->GetBoolFeature("StreamIsGrabbing")->GetValue())
        {
            m_objFeatureControlPtr->GetCommandFeature("AcquisitionStop")->Execute();
            m_objStreamPtr->StopGrab();
            m_objStreamPtr->UnregisterCaptureCallback();

        }
    }
    catch (CGalaxyException gex) {
       HandleError(gex);
    }
    try
    {
        m_objStreamPtr->Close();

        m_objDevicePtr->Close();
    }
    catch (CGalaxyException gex) {
       HandleError(gex);
    }
    m_bIsOpen = false;
    LogMessage("ShutDown");
    return 0;
}

void ClassGalaxy::GetName(char* name) const
{
    CDeviceUtils::CopyLimitedString(name, g_CameraDeviceName);
}


int ClassGalaxy::SnapImage()
{
    try
    {
        LogMessage("ReadyToSnapImage");
        // modify by LXM in 20240305
        //AddToLog("---------------判断是否开采");
        m_objFeatureControlPtr->GetCommandFeature("AcquisitionStop")->Execute();
        m_objStreamPtr->StopGrab();
        if (ImageHandler_ != NULL)
        {
            m_objStreamPtr->UnregisterCaptureCallback();
            delete ImageHandler_;
            ImageHandler_ = NULL;
        }

        //开启流层采集
        m_objStreamPtr->StartGrab();
        //发送开采命令
        m_objFeatureControlPtr->GetCommandFeature("AcquisitionStart")->Execute();
        m_bIsOpen = true;//modify by LXM
        m_objStreamPtr->FlushQueue();

        SendSoftwareTrigger(); // will only be send if TriggerMode is ON and TriggerSource is software

        // Block for the full time the camera is exposing.  This is at least the exposure time

        //可以使用采单帧来获取
        // Needs to be made usef setable, especially for external trigger applications
        CImageDataPointer ptrGrabResult = m_objStreamPtr->GetImage(exposureTimeoutS_ * 1000);
        uint64_t length = ptrGrabResult->GetPayloadSize();
        
        if (ptrGrabResult->GetPayloadSize() != imgBufferSize_)
        {	// due to parameter change on  binning
            ResizeSnapBuffer();
        }
        CopyToImageBuffer(ptrGrabResult);

    }
    catch (CGalaxyException gex) {
        m_objFeatureControlPtr->GetCommandFeature("AcquisitionStop")->Execute();
        m_objStreamPtr->StopGrab();
        HandleError(gex);
    }
    catch (const std::exception& e)
    {
       GetCoreCallback()->LogMessage(this, e.what(), false);
       m_objFeatureControlPtr->GetCommandFeature("AcquisitionStop")->Execute();
       m_objStreamPtr->StopGrab();
       return DEVICE_ERR;
    }
    GetCoreCallback()->LogMessage(this, "SnapImage", true);
    return DEVICE_OK;
}

void ClassGalaxy::SendSoftwareTrigger()
{
   if (TriggerMode_ == "On")
   {
      CEnumFeaturePointer TriggerSource = m_objFeatureControlPtr->GetEnumFeature("TriggerSource");
      if (TriggerSource->GetValue() == "Software")
      {
         m_objFeatureControlPtr->GetCommandFeature("TriggerSoftware")->Execute();
      }
   }
}

void ClassGalaxy::CopyToImageBuffer(CImageDataPointer& objImageDataPointer)
{
    if (1)
    {
        GetImageSize();
        const char* subject("Bayer");

        std::size_t found = pixelType_.find(subject);
        //pixelType_.assign(pixelFormat_gx);
        //相机类型-明确相机的输出形式--待确认
        GX_VALID_BIT_LIST emValidBits = GX_BIT_0_7;
        //明确相机的采图格式
        emValidBits = GetBestValudBit(objImageDataPointer->GetPixelFormat());

        if (found != std::string::npos)
        {
            IsByerFormat = true;
        }
        if (pixelType_.compare("Mono8") == 0)
        {
            // Workaround : OnPixelType call back will not be fired always.
            //copy image buffer to a snap buffer allocated by device adapter
            const void* buffer = objImageDataPointer->GetBuffer();
            memcpy(imgBuffer_, buffer, GetImageBufferSize());
            SetProperty(MM::g_Keyword_PixelType, g_PixelType_8bit);
        }
        //20221025待定其他颜色格式
        else if (pixelType_.compare("Mono16") == 0 || pixelType_.compare("Mono12") == 0 || pixelType_.compare("Mono10") == 0)
        {
            //黑白8-16位
            //copy image buffer to a snap buffer allocated by device adapter
            void* buffer = objImageDataPointer->GetBuffer();
            memcpy(imgBuffer_, buffer, GetImageBufferSize());
            SetProperty(MM::g_Keyword_PixelType, g_PixelType_16bit);
        }
        else if (IsByerFormat && pixelType_.size() == 8)
        {
            try
            {
                RG8ToRGB24Packed(imgBuffer_, objImageDataPointer);
                
                SetProperty(MM::g_Keyword_PixelType, g_PixelType_8bitRGBA);
            }
            catch (CGalaxyException gex) {
                HandleError(gex);
            }
        }
        else if (IsByerFormat && pixelType_.size() == 9)
        {
            try
            {
                RG10ToRGB24Packed(imgBuffer_, objImageDataPointer);
                SetProperty(MM::g_Keyword_PixelType, g_PixelType_8bitRGBA);
            }
            catch (CGalaxyException gex) {
                HandleError(gex);
            }
        }
    }
}

/**
* Stops Grabbing, but oonly if the camera actually was grabbing.
* Returns a flag indicating whether the camera was initially grabbing.
*/
bool ClassGalaxy::StopGrabbing()
{
  bool isGrabbing = m_objStreamFeatureControlPtr->GetBoolFeature("StreamIsGrabbing")->GetValue();
  if (isGrabbing) 
  {
     m_objFeatureControlPtr->GetCommandFeature("AcquisitionStop")->Execute();
     m_objStreamPtr->StopGrab();
  }
  return isGrabbing;
}

void ClassGalaxy::StartGrabbing()
{
   m_objFeatureControlPtr->GetCommandFeature("AcquisitionStart")->Execute();
   m_objStreamPtr->StartGrab();
}

int ClassGalaxy::StartSequenceAcquisition(long numImages, double interval_ms, bool stopOnOverflow) {
   try
   {
      LogMessage("ReadyMultiSequenceAcquisition");
      if (ImageHandler_ != NULL) {
         m_objStreamPtr->UnregisterCaptureCallback();
         delete(ImageHandler_);
      }
      ImageHandler_ = new CircularBufferInserter(this, numImages, stopOnOverflow);
      m_objStreamPtr->RegisterCaptureCallback(ImageHandler_, this);


      int ret = GetCoreCallback()->PrepareForAcq(this);
      if (ret != DEVICE_OK) {
          return ret;
      }
      sequenceRunning_ = true;

      m_objFeatureControlPtr->GetCommandFeature("AcquisitionStart")->Execute();
      //开启流层采集
      m_objStreamPtr->StartGrab();

      SendSoftwareTrigger();

      LogMessage("StartSequenceAcquisition");
   }
   catch (CGalaxyException gex) {
       m_objFeatureControlPtr->GetCommandFeature("AcquisitionStop")->Execute();
       m_objStreamPtr->StopGrab();
       sequenceRunning_ = false;
       return HandleError(gex);
   }
    catch (const std::exception& e)
    {
       GetCoreCallback()->LogMessage(this, e.what(), false);
       m_objFeatureControlPtr->GetCommandFeature("AcquisitionStop")->Execute();
       m_objStreamPtr->StopGrab();
       sequenceRunning_ = false;
       return DEVICE_ERR;
    }

    return DEVICE_OK;
}


int ClassGalaxy::StartSequenceAcquisition(double /* interval_ms */) {
    try
    {
        LogMessage("ReadySequenceAcquisition");
        StopGrabbing();

        if (ImageHandler_ != NULL) {
           m_objStreamPtr->UnregisterCaptureCallback();
           delete(ImageHandler_);
           ImageHandler_ = NULL;
        }
        ImageHandler_ = new CircularBufferInserter(this);
        m_objStreamPtr->RegisterCaptureCallback(ImageHandler_, this);
                
        int ret = GetCoreCallback()->PrepareForAcq(this);
        if (ret != DEVICE_OK) {
            return ret;
        }
        sequenceRunning_ = true;
        //camera_->StartGrabbing(numImages, GrabStrategy_OneByOne, GrabLoop_ProvidedByInstantCamera);
        m_objFeatureControlPtr->GetCommandFeature("AcquisitionStart")->Execute();
        //开启流层采集
        m_objStreamPtr->StartGrab();
        LogMessage("StartSequenceAcquisition");

        SendSoftwareTrigger();
    }
    catch (CGalaxyException gex) {
       m_objFeatureControlPtr->GetCommandFeature("AcquisitionStop")->Execute();
       m_objStreamPtr->StopGrab();
       sequenceRunning_ = false;
       return HandleError(gex);
    }
    catch (const std::exception& e)
    {
       GetCoreCallback()->LogMessage(this, e.what(), false);
       m_objFeatureControlPtr->GetCommandFeature("AcquisitionStop")->Execute();
       m_objStreamPtr->StopGrab();
       sequenceRunning_ = false;
       return DEVICE_ERR;
    }

    return DEVICE_OK;
}


int ClassGalaxy::StopSequenceAcquisition()
{
    if (m_objStreamFeatureControlPtr->GetBoolFeature("StreamIsGrabbing")->GetValue())
    {
        m_objFeatureControlPtr->GetCommandFeature("AcquisitionStop")->Execute();
        m_objStreamPtr->StopGrab();
        GetCoreCallback()->AcqFinished(this, 0);
        m_objStreamPtr->UnregisterCaptureCallback();
    }
    sequenceRunning_ = false;
    LogMessage("StopSequenceAcquisition");
    return DEVICE_OK;
}


bool ClassGalaxy::IsCapturing()
{
   return sequenceRunning_;
}

void ClassGalaxy::ResizeSnapBuffer() {

    free(imgBuffer_);
    GetImageSize();
    imageBufferSize_ = Width_ * Height_ * GetImageBytesPerPixel();//原先是buffersize
    imgBuffer_ = malloc(imageBufferSize_);
}

bool ClassGalaxy::__IsPixelFormat8(GX_PIXEL_FORMAT_ENTRY emPixelFormatEntry)
{
    bool bIsPixelFormat8 = false;
    const unsigned  PIXEL_FORMATE_BIT = 0x00FF0000;  ///<用于与当前的数据格式进行与运算得到当前的数据位数
    unsigned uiPixelFormatEntry = (unsigned)emPixelFormatEntry;
    if ((uiPixelFormatEntry & PIXEL_FORMATE_BIT) == GX_PIXEL_8BIT)
    {
        bIsPixelFormat8 = true;
    }
    return bIsPixelFormat8;
}


unsigned char* ClassGalaxy::GetImageBufferFromCallBack(CImageDataPointer& objImageDataPointer)
{

    INT64 Width_ = m_objFeatureControlPtr->GetIntFeature("Width")->GetValue();

    INT64 Height_ = m_objFeatureControlPtr->GetIntFeature("Height")->GetValue();
     //相机类型-明确相机的输出形式--待确认
     GX_VALID_BIT_LIST emValidBits = GX_BIT_0_7;
     //明确相机的采图格式
    emValidBits = GetBestValudBit(objImageDataPointer->GetPixelFormat());

    if (colorCamera_)
    {
        imgBuffer_2 = (unsigned char*)objImageDataPointer->ConvertToRGB24(emValidBits, GX_RAW2RGB_NEIGHBOUR, true);
    }
    else
    {
        if (__IsPixelFormat8(objImageDataPointer->GetPixelFormat()))
        {
            imgBuffer_2 = (BYTE*)objImageDataPointer->GetBuffer();
        }
        else
        {
            imgBuffer_2 = (BYTE*)objImageDataPointer->ConvertToRaw8(emValidBits);
        }

        // 黑白相机需要翻转数据后显示
        for (int i = 0; i < Height_; i++)
        {
            //含义
            memcpy(m_pImageBuffer + i * Width_, imgBuffer_2 + (Height_ - i - 1) * Width_, (size_t)Width_);
            return (unsigned char*)imgBuffer_;
        }
    }
    //获取图像buffer
    return (unsigned char*)imgBuffer_;

}

const unsigned char* ClassGalaxy::GetImageBuffer()
{
   return (unsigned char*)imgBuffer_;
}

// Beware!  This uses cached values so can get stale.
void ClassGalaxy::GetImageSize()
{
    Width_= (unsigned int) m_objFeatureControlPtr->GetIntFeature("Width")->GetValue();
    Height_ = (unsigned int) m_objFeatureControlPtr->GetIntFeature("Height")->GetValue();
}

unsigned ClassGalaxy::GetNumberOfComponents() const
{
    return nComponents_;
}

unsigned ClassGalaxy::GetImageWidth() const
{ 
    //mutable unsigned Width_ = m_objFeatureControlPtr->GetIntFeature("Width")->GetValue();

    return Width_;
}

unsigned ClassGalaxy::GetImageHeight() const
{
    return Height_;
}

long ClassGalaxy::GetImageBufferSize()const
{
   return imageBufferSize_;

    //return GetImageWidth() * GetImageHeight() * GetImageBytesPerPixel();
}

long ClassGalaxy::GetImageSizeLarge()const
{
    return Width_ * Height_;

    //return GetImageWidth() * GetImageHeight() * GetImageBytesPerPixel();
}

int ClassGalaxy::OnBinning(MM::PropertyBase* pProp, MM::ActionType eAct)
{

    if (eAct == MM::AfterSet)
    {
       CIntFeaturePointer BinningHorizontal = m_objFeatureControlPtr->GetIntFeature("BinningHorizontal");
       CIntFeaturePointer BinningVertical = m_objFeatureControlPtr->GetIntFeature("BinningVertical");
       bool Isgrabbing = StopGrabbing();
       try
       {
           pProp->Get(binningFactor_);
           int64_t val = std::atoi(binningFactor_.c_str());
           BinningHorizontal->SetValue(val);
           BinningVertical->SetValue(val);
           if (Isgrabbing)
           {
              StartGrabbing();
           }
           pProp->Set(binningFactor_.c_str());
       }
       catch (CGalaxyException& e)
       {
           // Error handling.
           LogMessage(e.what());
       }
       // Update cached values:
       GetImageSize();
    }
    else if (eAct == MM::BeforeGet) 
    {
       try {
          if (m_objDevicePtr->GetRemoteFeatureControl()->IsImplemented("BinningHorizontal"))
          {
             CIntFeaturePointer BinningHorizontal = m_objFeatureControlPtr->GetIntFeature("BinningHorizontal");
             binningFactor_ = CDeviceUtils::ConvertToString((long)BinningHorizontal->GetValue());
             pProp->Set((long)BinningHorizontal->GetValue());
          }
          else
          {
             pProp->Set(1l);
          }
       }
       catch (CGalaxyException& e)
       {
          return HandleError(e);
       }
    }
    return DEVICE_OK;
}

int ClassGalaxy::OnDeviceLinkThroughputLimit(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    return 0;
}

int ClassGalaxy::OnExposure(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::AfterSet)
   {
      try
      {
         pProp->Get(exposure_us_);
         m_objFeatureControlPtr->GetFloatFeature("ExposureTime")->SetValue(exposure_us_);
      }
      catch (CGalaxyException& e)
      {
         return HandleError(e);
      }
   }
   else if (eAct == MM::BeforeGet) 
   {
      try 
      {
         exposure_us_ = m_objFeatureControlPtr->GetFloatFeature("ExposureTime")->GetValue();
         pProp->Set(exposure_us_);
      }
      catch (CGalaxyException& e)
      {
         return HandleError(e);
      }
   }
   return DEVICE_OK;
}

int ClassGalaxy::OnGain(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    try
    {
        //CFloatPtr gain(nodeMap_->GetNode("Gain"));
        //CIntegerPtr GainRaw(nodeMap_->GetNode("GainRaw"));
        CFloatFeaturePointer gain = m_objFeatureControlPtr->GetFloatFeature("Gain");		
        if (eAct == MM::AfterSet) {
            pProp->Get(gain_);
            if (gain_ > gain->GetMax()) {
                gain_ = gain->GetMax();
            }
            if (gain_ < gain->GetMin()) {
                gain_ = gain->GetMin();
            }
            if (1)
            {
                // the range gain depends on Pixel format sometimes.
                if (gain->GetMin() <= gain_ && gain->GetMax() >= gain_)
                {
                    gain->SetValue(gain_);
                }
                else
                {
                    LogMessage("gain value out of range");
                    gainMax_ = gain->GetMax();
                    gainMin_ = gain->GetMin();
                    gain_ = gain->GetValue();
                    SetPropertyLimits(MM::g_Keyword_Gain, gainMin_, gainMax_);
                    pProp->Set(gain_);
                }
            }
        }
        else if (eAct == MM::BeforeGet) {

            if (1)
            {
                gain_ = gain->GetValue();
                pProp->Set(gain_);
            }
        }
    }
    catch (const CGalaxyException& e)
    {
       return HandleError(e);
    }
    return DEVICE_OK;
}

int ClassGalaxy::OnHeight(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    CIntFeaturePointer Height = m_objFeatureControlPtr->GetIntFeature("Height");

    std::string strval;
    if (eAct == MM::AfterSet)
    {
        bool Isgrabbing = m_objStreamFeatureControlPtr->GetBoolFeature("StreamIsGrabbing")->GetValue();;
        if (Height.getUse())
        {
            try
            {
                if (Isgrabbing)
                {
                    m_objFeatureControlPtr->GetCommandFeature("AcquisitionStop")->Execute();
                    //关闭流层采集
                    m_objStreamPtr->StopGrab();
                    //camera_->StopGrabbing();
                }
                pProp->Get(strval);
                int64_t val = std::atoi(strval.c_str());
                int64_t inc = Height->GetInc();
                Height->SetValue(val - (val % inc));

                if (Isgrabbing)
                {
                    m_objFeatureControlPtr->GetCommandFeature("AcquisitionStart")->Execute();
                    m_objStreamPtr->StartGrab();
                }
                pProp->Set((long)Height->GetValue());

            }
            catch (CGalaxyException& e)
            {
               return HandleError(e);
            }
        }
    }
    else if (eAct == MM::BeforeGet) {

        try {
            if (Height.getUse())
            {
                pProp->Set((long)Height->GetValue());
            }
        }
        catch (CGalaxyException& e)
        {
            return HandleError(e);
        }
    }
    
    GetCoreCallback()->LogMessage(this, ("Height: " + to_string((long)Height->GetValue())).c_str(), true);
    GetImageSize();
    return DEVICE_OK;
}

int ClassGalaxy::OnInterPacketDelay(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    return 0;
}

int ClassGalaxy::OnPixelType(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    if (m_objStreamFeatureControlPtr->GetBoolFeature("StreamIsGrabbing")->GetValue())
    {
        m_objFeatureControlPtr->GetCommandFeature("AcquisitionStop")->Execute();
        //关闭流层采集
        m_objStreamPtr->StopGrab();
    }
    //CEnumerationPtr pixelFormat(nodeMap_->GetNode("PixelFormat"));

    CEnumFeaturePointer pixelFormat_ = m_objFeatureControlPtr->GetEnumFeature("PixelFormat");

    if (eAct == MM::AfterSet) {
        pProp->Get(pixelType_);
        try
        {
                //代码报错
                          
                //设置新值
                pixelFormat_->SetValue(pixelType_.c_str());
                const char* subject("Bayer");
                std::size_t found = pixelType_.find(subject);
                if (pixelType_.compare("Mono8") == 0)
                {
                   nComponents_ = 1;
                   bitDepth_ = 8;
                   bytesPerPixel_ = 1;
                   SetProperty(MM::g_Keyword_PixelType, g_PixelType_8bit);
                }
                else if (pixelType_.compare("Mono10") == 0)
                {
                   nComponents_ = 1;
                   bitDepth_ = 10;
                   bytesPerPixel_ = 2;
                   SetProperty(MM::g_Keyword_PixelType, g_PixelType_10bit);
                }
                else if (pixelType_.compare("Mono12") == 0)
                {
                   nComponents_ = 2;
                   bitDepth_ = 12;
                   bytesPerPixel_ = 1;
                   SetProperty(MM::g_Keyword_PixelType, g_PixelType_12bit);
                }
                else if (pixelType_.compare("Mono16") == 0)
                {
                   nComponents_ = 1;
                   bitDepth_ = 16;
                   bytesPerPixel_ = 2;
                   SetProperty(MM::g_Keyword_PixelType, g_PixelType_16bit);
                }
                else if (found == 0 && pixelType_.size() == 8)
                {
                   nComponents_ = 4;
                   bitDepth_ = 8;
                   bytesPerPixel_ = 1;
                   SetProperty(MM::g_Keyword_PixelType, g_PixelType_8bitRGBA);
                }
                else if (found == 0 && pixelType_.size() == 9)
                {
                   //nComponents_ = 4 * sizeof(unsigned short int);
                   nComponents_ = 4;
                   bitDepth_ = 8;
                   bytesPerPixel_ = 1;
                   SetProperty(MM::g_Keyword_PixelType, g_PixelType_8bitRGBA);
                }
                /*
                m_objFeatureControlPtr->GetEnumFeature("BlackLevelSelector")->SetValue("All");
                CFloatFeaturePointer offset = m_objFeatureControlPtr->GetFloatFeature("BlackLevel");
                offsetMax_ = offset->GetMax();
                offsetMin_ = offset->GetMin();
                SetPropertyLimits(MM::g_Keyword_Offset, offsetMin_, offsetMax_);
                */
        }
        catch (CGalaxyException& e)
        {
            LogMessage(e.what());
        }
    }
    else if (eAct == MM::BeforeGet) {
        pProp->Set(pixelType_.c_str());
    }
    if (1)
    {
       
        if (m_objStreamFeatureControlPtr->GetBoolFeature("StreamIsGrabbing")->GetValue())
        {
            //重开
            m_objFeatureControlPtr->GetCommandFeature("AcquisitionStart")->Execute();
            m_objStreamPtr->StartGrab();
        }
    }

    return DEVICE_OK;
}

int ClassGalaxy::OnTriggerSource(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    CEnumFeaturePointer TriggerSource = m_objFeatureControlPtr->GetEnumFeature("TriggerSource");

    string TriggerSource_;
    if (eAct == MM::AfterSet) {
        pProp->Get(TriggerSource_);
        TriggerSource->SetValue(TriggerSource_.c_str());
    }
    else if (eAct == MM::BeforeGet) {
        //CEnumerationPtr TriggerSource(nodeMap_->GetNode("TriggerSource"));
        pProp->Set(TriggerSource->GetValue().c_str());
    }
    return DEVICE_OK;
}

int ClassGalaxy::OnTriggerMode(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    try
    {
        CEnumFeaturePointer TriggerMode = m_objFeatureControlPtr->GetEnumFeature("TriggerMode");

        if (!TriggerMode.IsNull())
        {
            if (eAct == MM::AfterSet) {
               pProp->Get(TriggerMode_);
               pProp->Set(TriggerMode_.c_str());
               TriggerMode->SetValue(TriggerMode_.c_str());
            }
            else if (eAct == MM::BeforeGet) {
               pProp->Set(TriggerMode->GetValue().c_str());
            }
        }
     }
    catch (CGalaxyException& e)
    {
       return HandleError(e);
    }
    return DEVICE_OK;
}

int ClassGalaxy::OnTriggerActivation(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    try
    {
        CEnumFeaturePointer TriggerActivation = m_objFeatureControlPtr->GetEnumFeature("TriggerActivation");

        if (!TriggerActivation.IsNull())
        {
            if (eAct == MM::AfterSet) {
                pProp->Get(TriggerActivation_);
                pProp->Set(TriggerActivation_.c_str());
                TriggerActivation->SetValue(TriggerActivation_.c_str());
            }
            else if (eAct == MM::BeforeGet) {
                pProp->Set(TriggerActivation->GetValue().c_str());

            }
        }
    }
    catch (CGalaxyException& e)
    {
       return HandleError(e);
    }
    return DEVICE_OK;
}


int ClassGalaxy::OnLineMode(MM::PropertyBase* pProp, MM::ActionType eAct, long i)
{
   CEnumFeaturePointer lineselector = m_objFeatureControlPtr->GetEnumFeature("LineSelector");
   if (lineselector.IsNull())
   {
      return ERR_CAMERA_SDK;
   }
   gxstring lineVal = lineselector->GetValue();
   gxstring_vector entries = lineselector->GetEnumEntryList();
   if (entries.size() == 0) 
   {
      return ERR_CAMERA_SDK;
   }
   std::string entry(entries[0]);
   std::string line = entry.substr(0, entry.size() - 1) + (std::to_string(i)).c_str();
   lineselector->SetValue(line.c_str());
   CEnumFeaturePointer linemodes = m_objFeatureControlPtr->GetEnumFeature("LineMode");
   if (linemodes.IsNull()) 
   {
      return ERR_CAMERA_SDK;
   }
   gxstring_vector modeentries = linemodes->GetEnumEntryList();
   if (modeentries.size() == 0) 
   {
      return ERR_CAMERA_SDK;
   }

   if (eAct == MM::BeforeGet)
   {
      gxstring lineMode = linemodes->GetValue();
      pProp->Set(lineMode.c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      std::string linemode;
      pProp->Get(linemode);
      linemodes->SetValue(linemode.c_str());
   }

   // return lineselector to its orginal value
   lineselector->SetValue(lineVal);

   return DEVICE_OK;
}

int ClassGalaxy::OnLineSource(MM::PropertyBase* pProp, MM::ActionType eAct, long i)
{
   CEnumFeaturePointer lineselector = m_objFeatureControlPtr->GetEnumFeature("LineSelector");
   if (lineselector.IsNull())
   {
      return ERR_CAMERA_SDK;
   }
   gxstring lineVal = lineselector->GetValue();
   gxstring_vector entries = lineselector->GetEnumEntryList();
   if (entries.size() == 0) 
   {
      return ERR_CAMERA_SDK;
   }
   std::string entry(entries[0]);
   std::string line = entry.substr(0, entry.size() - 1) + (std::to_string(i)).c_str();
   lineselector->SetValue(line.c_str());
   CEnumFeaturePointer linesources = m_objFeatureControlPtr->GetEnumFeature("LineSource");
   if (linesources.IsNull()) 
   {
      return ERR_CAMERA_SDK;
   }
   gxstring_vector sourceentries = linesources->GetEnumEntryList();
   if (sourceentries.size() == 0) 
   {
      return ERR_CAMERA_SDK;
   }

   if (eAct == MM::BeforeGet)
   {
      gxstring lineSource = linesources->GetValue();
      pProp->Set(lineSource.c_str());
   }
   else if (eAct == MM::AfterSet)
   {
      std::string linesource;
      pProp->Get(linesource);
      linesources->SetValue(linesource.c_str());
   }

   // return lineselector to its orginal value
   lineselector->SetValue(lineVal);

   return DEVICE_OK;
}

int ClassGalaxy::OnUserOutput(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    CEnumFeaturePointer userOutputFP = m_objFeatureControlPtr->GetEnumFeature("UserOutputSelector");

    string userOutputSelector;
    if (eAct == MM::AfterSet) {
        pProp->Get(userOutputSelector);
        userOutputFP->SetValue(userOutputSelector.c_str());
    }
    else if (eAct == MM::BeforeGet) {
        pProp->Set(userOutputFP->GetValue().c_str());
    }
    return DEVICE_OK;
}


//AdjFrameRateMode
int ClassGalaxy::OnAdjFrameRateMode(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    try
    {
        CEnumFeaturePointer AcquisitionFrameRateMode = m_objFeatureControlPtr->GetEnumFeature("AcquisitionFrameRateMode");

        if (eAct == MM::AfterSet) {
            pProp->Get(AcquisitionFrameRateMode_);

            pProp->Set(AcquisitionFrameRateMode_.c_str());
            AcquisitionFrameRateMode->SetValue(AcquisitionFrameRateMode_.c_str());
        }
        else if (eAct == MM::BeforeGet) {
            pProp->Set(AcquisitionFrameRateMode->GetValue().c_str());
        }
    }
    catch (CGalaxyException& e)
    {
       return HandleError(e);
    }
    return DEVICE_OK;
}
//OnAcquisitionFrameRate
int ClassGalaxy::OnAcquisitionFrameRate(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    try
    {
        CFloatFeaturePointer AcquisitionFrameRate = m_objFeatureControlPtr->GetFloatFeature("AcquisitionFrameRate");

        if (eAct == MM::AfterSet) {
            pProp->Get(AcquisitionFrameRate_);
            pProp->Set(AcquisitionFrameRate_.c_str());
            AcquisitionFrameRate->SetValue(atoi(AcquisitionFrameRate_.c_str()));
        }
        else if (eAct == MM::BeforeGet) {
            pProp->Set(AcquisitionFrameRate->GetValue());
        }
    }
    catch (CGalaxyException& e)
    {
       return HandleError(e);
    }
    return DEVICE_OK;
}


int ClassGalaxy::OnWidth(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    CIntFeaturePointer Width = m_objFeatureControlPtr->GetIntFeature("Width");
    std::string strval;
    if (eAct == MM::AfterSet)
    {
        if (Width.getUse())
        {
            try
            {
                if (m_objStreamFeatureControlPtr->GetBoolFeature("StreamIsGrabbing")->GetValue())
                {
                    m_objFeatureControlPtr->GetCommandFeature("AcquisitionStop")->Execute();
                    //关闭流层采集
                    m_objStreamPtr->StopGrab();
                    //camera_->StopGrabbing();
                }
                pProp->Get(strval);
                int64_t val = std::atoi(strval.c_str());
                int64_t inc = Width->GetInc();
                int64_t a = val - (val % inc);
                Width->SetValue(val - (val % inc));
                pProp->Set((long)Width->GetValue());
                if (m_objStreamFeatureControlPtr->GetBoolFeature("StreamIsGrabbing")->GetValue())
                {
                    m_objFeatureControlPtr->GetCommandFeature("AcquisitionStart")->Execute();
                    m_objStreamPtr->StartGrab();
                }		
            }
            catch (CGalaxyException& e)
            {
               return HandleError(e);
            }
        }
    }
    else if (eAct == MM::BeforeGet) {
        try
        {
            if (Width.getUse())
            {
                pProp->Set((long)Width->GetValue());
            }
        }
        catch (CGalaxyException& e)
        {
           return HandleError(e);
        }
    }
    GetCoreCallback()->LogMessage(this, ("Width: " + to_string((long)Width->GetValue())).c_str(), true);
    GetImageSize();
    return DEVICE_OK;
}

int ClassGalaxy::OnTriggerDelay(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   CFloatFeaturePointer TriggerDelay = m_objFeatureControlPtr->GetFloatFeature("TriggerDelay");
   try 
   {
       if (eAct == MM::AfterSet) {
           pProp->Get(TriggerDelay_);
           pProp->Set(TriggerDelay_.c_str());
           TriggerDelay->SetValue(atoi(TriggerDelay_.c_str()));
       }
       else if (eAct == MM::BeforeGet) {
           pProp->Set(TriggerDelay->GetValue());
       }
   }
   catch (CGalaxyException gex) 
   {
      return HandleError(gex);
   }
   return DEVICE_OK;
}

//OnTriggerFilterRaisingEdge
int ClassGalaxy::OnTriggerFilterRaisingEdge(MM::PropertyBase* pProp, MM::ActionType eAct)
{
    CFloatFeaturePointer TriggerFilterRaisingEdge = m_objFeatureControlPtr->GetFloatFeature("TriggerFilterRaisingEdge");
    try
    {
        if (eAct == MM::AfterSet) {
            pProp->Get(TriggerFilterRaisingEdge_);
            pProp->Set(TriggerFilterRaisingEdge_.c_str());
            TriggerFilterRaisingEdge->SetValue(atoi(TriggerFilterRaisingEdge_.c_str()));
        }
        else if (eAct == MM::BeforeGet) {
            pProp->Set(TriggerFilterRaisingEdge->GetValue());
        }
    }
   catch (CGalaxyException gex) 
   {
      return HandleError(gex);
   }
   return DEVICE_OK;
}


int ClassGalaxy::OnExposureTimeout(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   if (eAct == MM::AfterSet)
   {
      pProp->Get(exposureTimeoutS_);
   }
   else if (eAct == MM::BeforeGet)
   { 
      pProp->Set(exposureTimeoutS_);
   }
   return DEVICE_OK;
}


unsigned ClassGalaxy::GetImageBytesPerPixel() const
{
    return nComponents_ * bytesPerPixel_;
}

unsigned ClassGalaxy::GetBitDepth() const
{

    const char* subject("Bayer");
    std::size_t found = pixelType_.find(subject);

    if (pixelType_ == "Mono8") {
        return 8;
    }
    else if (pixelType_ == "Mono10") {
        return 10;
    }
    else if (pixelType_ == "Mono12") {
        return 12;
    }
    else if (pixelType_ == "Mono16") {
        return 16;
    }
    else if (found != std::string::npos || pixelType_ == "BGR8" || pixelType_ == "RGB8") {
        return 8;
    }
    assert(0); //should not happen
    return 0;
}



double ClassGalaxy::GetExposure() const
{
   return exposure_us_ / 1000.0;
}

void ClassGalaxy::SetExposure(double exp)
{
   // Micro-Manager gives exposure in ms, Galaxy sets it in micro-seconds
   try 
   {
      m_objFeatureControlPtr->GetFloatFeature("ExposureTime")->SetValue(exp * 1000.0);
      // will this update immediately?
      exposure_us_ = m_objFeatureControlPtr->GetFloatFeature("ExposureTime")->GetValue();
   }
   catch (CGalaxyException gex) 
   {
      HandleError(gex);
   }
}

int ClassGalaxy::SetROI(unsigned x, unsigned y, unsigned xSize, unsigned ySize)
{
   bool isGrabbing = StopGrabbing();
   m_objFeatureControlPtr->GetEnumFeature("RegionSelector")->SetValue("Region0");

   x -= (x % (unsigned int)m_objFeatureControlPtr->GetIntFeature("OffsetX")->GetInc());
   y -= (y % (unsigned int)m_objFeatureControlPtr->GetIntFeature("OffsetY")->GetInc());
   xSize -= (xSize % (unsigned int)m_objFeatureControlPtr->GetIntFeature("Width")->GetInc());
   ySize -= (ySize % (unsigned int)m_objFeatureControlPtr->GetIntFeature("Height")->GetInc());

   if (xSize < (unsigned int)m_objFeatureControlPtr->GetIntFeature("Width")->GetMin()) {
      xSize = (unsigned int)m_objFeatureControlPtr->GetIntFeature("Width")->GetMin();
   }
   if (ySize < (unsigned int)m_objFeatureControlPtr->GetIntFeature("Height")->GetMin()) {
      ySize = (unsigned int)m_objFeatureControlPtr->GetIntFeature("Height")->GetMin();
   }	
   if (x < (unsigned int)m_objFeatureControlPtr->GetIntFeature("OffsetX")->GetMin()) {
      x = (unsigned int)m_objFeatureControlPtr->GetIntFeature("OffsetX")->GetMin();
   }
   if (y < (unsigned int)m_objFeatureControlPtr->GetIntFeature("OffsetY")->GetMin()) {
       y = (unsigned int)m_objFeatureControlPtr->GetIntFeature("OffsetY")->GetMin();
   }

   // TODO: check if values are too high?

   try 
   {
      m_objFeatureControlPtr->GetIntFeature("Width")->SetValue(xSize);
      m_objFeatureControlPtr->GetIntFeature("Height")->SetValue(ySize);
      m_objFeatureControlPtr->GetIntFeature("OffsetX")->SetValue(x);
      m_objFeatureControlPtr->GetIntFeature("OffsetY")->SetValue(y);
   }
   catch (CGalaxyException gex) 
   {
      return HandleError(gex);
   }
   if (isGrabbing)
      StartGrabbing();
    // Update cached values:
    GetImageSize();

   return DEVICE_OK;
}


int ClassGalaxy::GetROI(unsigned& x, unsigned& y, unsigned& xSize, unsigned& ySize)
{
   xSize = (unsigned int) m_objFeatureControlPtr->GetIntFeature("Width")->GetValue();
   ySize = (unsigned int) m_objFeatureControlPtr->GetIntFeature("Height")->GetValue();
   x = (unsigned int) m_objFeatureControlPtr->GetIntFeature("OffsetX")->GetValue();
   y = (unsigned int) m_objFeatureControlPtr->GetIntFeature("OffsetY")->GetValue();

   return DEVICE_OK;
}

int ClassGalaxy::ClearROI()
{
   bool isGrabbing = StopGrabbing();
   m_objFeatureControlPtr->GetEnumFeature("RegionSelector")->SetValue("Region0");

   int64_t width = m_objFeatureControlPtr->GetIntFeature("Width")->GetMax();
   int64_t height = m_objFeatureControlPtr->GetIntFeature("Height")->GetMax();

   m_objFeatureControlPtr->GetIntFeature("OffsetY")->SetValue(0);
   m_objFeatureControlPtr->GetIntFeature("OffsetX")->SetValue(0);
   m_objFeatureControlPtr->GetIntFeature("Height")->SetValue(height);
   m_objFeatureControlPtr->GetIntFeature("Width")->SetValue(width);

   if (isGrabbing)
      StartGrabbing();
   // Update cached values:
   GetImageSize();

   return DEVICE_OK;;
}

int ClassGalaxy::GetBinning() const
{
   char val [MM::MaxStrLength];
   GetProperty(MM::g_Keyword_Binning, val);
   return atoi(val);
}

int ClassGalaxy::SetBinning(int binSize)
{
   return SetProperty(MM::g_Keyword_Binning, std::to_string(binSize).c_str());
}

void ClassGalaxy::RGB24PackedToRGBA(void* destbuffer, void* srcbuffer, CImageDataPointer& objImageDataPointer)
{
    unsigned int srcOffset = 0;
    unsigned int dstOffset = 0;
    GX_VALID_BIT_LIST emValidBits = GX_BIT_0_7;
    uint64_t Payloadsize=objImageDataPointer->GetPayloadSize();
    //明确相机的采图格式
    emValidBits = GetBestValudBit(objImageDataPointer->GetPixelFormat());
    if (emValidBits!= GX_BIT_0_7)
    {
        Payloadsize = Payloadsize / 2;
    }
    for (size_t i = 0; i < Payloadsize; ++i)
    {
        try
        {   //                            4                              3
            memcpy((BYTE*)destbuffer + dstOffset, (BYTE*)srcbuffer + srcOffset, 3);
            srcOffset += 3;
            dstOffset += 4;
        }
        catch (const std::exception& e)
        {
           GetCoreCallback()->LogMessage(this, e.what(), false);
        }
    }
}

void ClassGalaxy::RG8ToRGB24Packed(void* destbuffer,CImageDataPointer& objImageDataPointer)
{
    try
    {
        GX_VALID_BIT_LIST emValidBits = GX_BIT_0_7;
        //明确相机的采图格式
        emValidBits = GetBestValudBit(objImageDataPointer->GetPixelFormat());

        //为了显示，需要都转成Raw8位
        if (emValidBits!= GX_BIT_0_7)
        {
            return;
        }
        //RGB24Packed-大小乘以3
        void* buffer = objImageDataPointer->ConvertToRGB24(emValidBits, GX_RAW2RGB_NEIGHBOUR, false);
        RGB24PackedToRGBA(destbuffer, buffer, objImageDataPointer);
        LogMessage("RG8ToRGB24Packed");
    }
    catch (const std::exception e)
    {
        GetCoreCallback()->LogMessage(this, e.what(), false);
    }
}

void ClassGalaxy::CoverRGB16ToRGBA16(unsigned short int* Desbuffer, unsigned short int* Srcbuffer)
{
    unsigned int srcOffset = 0;
    unsigned int dstOffset = 0;
    for (size_t i = 0; i < GetImageSizeLarge(); ++i)
    {
        try
        {   //                                         4                        3
            memcpy(Desbuffer + dstOffset, Srcbuffer + srcOffset, 3*sizeof(unsigned short int));
            Desbuffer[i*4+4] = 255;
            srcOffset += 3;
            dstOffset += 4;
        }
        catch (const std::exception& e)
        {
            LogMessage(e.what());
        }
    }
}
void ClassGalaxy::RG10ToRGB24Packed(void* pRGB24Bufdest, CImageDataPointer& objImageDataPointer)
{
    if (0)
    {
        //转成RGB8*2，在转成RGBA8*2*4
        size_t BufferSize = GetImageSizeLarge() * 3 * sizeof(unsigned short int);
        void* RGB16 = malloc(BufferSize);

        CoverToRGB(GX_PIXEL_FORMAT_RGB16, RGB16, objImageDataPointer);
        unsigned short int* Src = new unsigned short int[BufferSize]();
        memset(Src, 0, BufferSize);
        if (RGB16!=nullptr)
        {
            memcpy(Src, RGB16, BufferSize);
        }
        BufferSize= GetImageSizeLarge() * 4 * sizeof(unsigned short int);
        unsigned short int* Dst = new unsigned short int[BufferSize]();
        CoverRGB16ToRGBA16(Dst, Src);

        memcpy(pRGB24Bufdest, Dst, BufferSize);
        return;
    }


    GX_VALID_BIT_LIST emValidBits = GX_BIT_0_7;
    //明确相机的采图格式
    emValidBits = GetBestValudBit(objImageDataPointer->GetPixelFormat());

    BYTE* pRGB24Buf2 = (BYTE*)objImageDataPointer->ConvertToRGB24(emValidBits, GX_RAW2RGB_NEIGHBOUR, false);

    RGB24PackedToRGBA(pRGB24Bufdest, pRGB24Buf2, objImageDataPointer);

}

GX_VALID_BIT_LIST ClassGalaxy::GetBestValudBit(GX_PIXEL_FORMAT_ENTRY emPixelFormatEntry)
{
    GX_VALID_BIT_LIST emValidBits = GX_BIT_0_7;
    switch (emPixelFormatEntry)
    {
    case GX_PIXEL_FORMAT_MONO8:
    case GX_PIXEL_FORMAT_BAYER_GR8:
    case GX_PIXEL_FORMAT_BAYER_RG8:
    case GX_PIXEL_FORMAT_BAYER_GB8:
    case GX_PIXEL_FORMAT_BAYER_BG8:
    {
        emValidBits = GX_BIT_0_7;
        if (emPixelFormatEntry!= GX_PIXEL_FORMAT_MONO8)
        {
            IsByerFormat = true;
        }
        break;
    }
    case GX_PIXEL_FORMAT_MONO10:
    case GX_PIXEL_FORMAT_BAYER_GR10:
    case GX_PIXEL_FORMAT_BAYER_RG10:
    case GX_PIXEL_FORMAT_BAYER_GB10:
    case GX_PIXEL_FORMAT_BAYER_BG10:
    {
        emValidBits = GX_BIT_2_9;		
        if (emPixelFormatEntry != GX_PIXEL_FORMAT_MONO10)
        {
            IsByerFormat = true;
        }
        break;
    }
    case GX_PIXEL_FORMAT_MONO12:
    case GX_PIXEL_FORMAT_BAYER_GR12:
    case GX_PIXEL_FORMAT_BAYER_RG12:
    case GX_PIXEL_FORMAT_BAYER_GB12:
    case GX_PIXEL_FORMAT_BAYER_BG12:
    {
        emValidBits = GX_BIT_4_11;
        if (emPixelFormatEntry != GX_PIXEL_FORMAT_MONO12)
        {
            IsByerFormat = true;
        }
        break;
    }
    case GX_PIXEL_FORMAT_MONO14:
    {
        //暂时没有这样的数据格式待升级
        break;
    }
    case GX_PIXEL_FORMAT_MONO16:
    case GX_PIXEL_FORMAT_BAYER_GR16:
    case GX_PIXEL_FORMAT_BAYER_RG16:
    case GX_PIXEL_FORMAT_BAYER_GB16:
    case GX_PIXEL_FORMAT_BAYER_BG16:
    {
        //暂时没有这样的数据格式待升级
        if (emPixelFormatEntry != GX_PIXEL_FORMAT_MONO16)
        {
            IsByerFormat = true;
        }
        break;
    }
    default:
        break;
    }
    return emValidBits;
}

CircularBufferInserter::CircularBufferInserter(ClassGalaxy* dev) :
    dev_(dev),
    numImages_(-1),
    imgCounter_(0),
    stopOnOverflow_(false)
{}

CircularBufferInserter::CircularBufferInserter(ClassGalaxy* dev, long numImages, bool stopOnOverflow) :
    dev_(dev),
    numImages_(numImages),
    imgCounter_(0),
    stopOnOverflow_(stopOnOverflow)
{}

    //---------------------------------------------------------------------------------
    /**
    \brief   采集回调函数
    \param   objImageDataPointer      图像处理参数
    \param   pFrame                   用户参数
    \return  无
    */
    //----------------------------------------------------------------------------------
void CircularBufferInserter::DoOnImageCaptured(CImageDataPointer& objImageDataPointer, void* pUserParam)
{
    dev_->SendSoftwareTrigger();
    // char label[MM::MaxStrLength];
    //dev_->AddToLog("OnImageGrabbed");
    // Important:  meta data about the image are generated here:
    Metadata md;
    md.put(MM::g_Keyword_Metadata_CameraLabel, "");
    md.put(MM::g_Keyword_Metadata_ROI_X, CDeviceUtils::ConvertToString((long)objImageDataPointer->GetWidth()));
    md.put(MM::g_Keyword_Metadata_ROI_Y, CDeviceUtils::ConvertToString((long)objImageDataPointer->GetHeight()));
    md.put(MM::g_Keyword_Metadata_ImageNumber, CDeviceUtils::ConvertToString((long)objImageDataPointer->GetFrameID()));
    md.put(MM::g_Keyword_Metadata_Exposure, dev_->GetExposure());
    // Image grabbed successfully ?
    if (objImageDataPointer->GetStatus()== GX_FRAME_STATUS_SUCCESS)
    {
        //查询图像格式
        GX_PIXEL_FORMAT_ENTRY pixelFormat_gx = objImageDataPointer->GetPixelFormat();

        dev_->ResizeSnapBuffer();
        //黑白
        if (!dev_->colorCamera_)
        {
            //copy to intermediate buffer
            int ret = dev_->GetCoreCallback()->InsertImage(dev_, (const unsigned char*)objImageDataPointer->GetBuffer(),
                (unsigned)objImageDataPointer->GetWidth(), (unsigned)objImageDataPointer->GetHeight(),
                (unsigned)dev_->GetImageBytesPerPixel(), 1, md.Serialize().c_str(), FALSE);
            if (ret == DEVICE_BUFFER_OVERFLOW) {
                //if circular buffer overflows, just clear it and keep putting stuff in so live mode can continue
               if (stopOnOverflow_)
               {
                  dev_->StopSequenceAcquisition();
                  dev_->LogMessage("Error inserting image into sequence buffer", false);
               }
               else
               {
                  dev_->GetCoreCallback()->ClearImageBuffer(dev_);
               }
            }
        }
        else if (dev_->colorCamera_)
        {
            //彩色，注意这里全部转成8位RGB
            if (dev_->__IsPixelFormat8(pixelFormat_gx))
            {
                dev_->RG8ToRGB24Packed(dev_->imgBuffer_, objImageDataPointer);
            }
            else {
                dev_->RG10ToRGB24Packed(dev_->imgBuffer_, objImageDataPointer);
            }		
            //copy to intermediate buffer
            int ret = dev_->GetCoreCallback()->InsertImage(dev_, (const unsigned char*)dev_->imgBuffer_,
                (unsigned)dev_->GetImageWidth(), (unsigned)dev_->GetImageHeight(),
                (unsigned)dev_->GetImageBytesPerPixel(), 1, md.Serialize().c_str(), FALSE);
            if (ret == DEVICE_BUFFER_OVERFLOW) {
                //if circular buffer overflows, just clear it and keep putting stuff in so live mode can continue
                dev_->GetCoreCallback()->ClearImageBuffer(dev_);
            }
        }
        imgCounter_++;
        if (imgCounter_ == numImages_)
        {
           dev_->StopSequenceAcquisition();
           return;
        }
    }
    else
    {
        dev_->LogMessage("GetStatus failed in Sequence acquisition");
    }
}


int64_t ClassGalaxy::__GetStride(int64_t nWidth, bool bIsColor)
{
    return bIsColor ? nWidth * 3 : nWidth;
}


bool ClassGalaxy::__IsCompatible(BITMAPINFO* pBmpInfo, uint64_t nWidth, uint64_t nHeight)
{
    if (pBmpInfo == NULL
        || pBmpInfo->bmiHeader.biHeight != nHeight
        || pBmpInfo->bmiHeader.biWidth != nWidth
        )
    {
        return false;
    }
    return true;
}
void ClassGalaxy::__ColorPrepareForShowImg()
{
    //--------------------------------------------------------------------
    //---------------------------初始化bitmap头---------------------------
    m_pBmpInfo = (BITMAPINFO*)m_chBmpBuf;
    m_pBmpInfo->bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    m_pBmpInfo->bmiHeader.biWidth = (LONG)Width_;
    m_pBmpInfo->bmiHeader.biHeight = (LONG)Height_;

    m_pBmpInfo->bmiHeader.biPlanes = 1;
    m_pBmpInfo->bmiHeader.biBitCount = 24;
    m_pBmpInfo->bmiHeader.biCompression = BI_RGB;
    m_pBmpInfo->bmiHeader.biSizeImage = 0;
    m_pBmpInfo->bmiHeader.biXPelsPerMeter = 0;
    m_pBmpInfo->bmiHeader.biYPelsPerMeter = 0;
    m_pBmpInfo->bmiHeader.biClrUsed = 0;
    m_pBmpInfo->bmiHeader.biClrImportant = 0;
}
void ClassGalaxy::__UpdateBitmap(CImageDataPointer& objCImageDataPointer)
{
    if (!__IsCompatible(m_pBmpInfo, objCImageDataPointer->GetWidth(), objCImageDataPointer->GetHeight()))
    {

        if (colorCamera_)
        {
            __ColorPrepareForShowImg();
        }
        else
        {
            //__MonoPrepareForShowImg();
        }
    }
}
void ClassGalaxy::SaveBmp(CImageDataPointer& objCImageDataPointer, const std::string& strFilePath)
{
    GX_VALID_BIT_LIST emValidBits = GX_BIT_0_7;
    BYTE* pBuffer = NULL;

    if ((objCImageDataPointer.IsNull()) || (strFilePath == ""))
    {
        throw std::runtime_error("Argument is error");
    }

    //检查图像是否改变并更新Buffer
    __UpdateBitmap(objCImageDataPointer);

    emValidBits = GetBestValudBit(objCImageDataPointer->GetPixelFormat());

    if (colorCamera_)
    {
        //BYTE* pBuffer = (BYTE*)objCImageDataPointer->ImageProcess(objCfg);

        pBuffer = (BYTE*)objCImageDataPointer->ConvertToRGB24(emValidBits, GX_RAW2RGB_NEIGHBOUR, true);
    }
    else
    {
        if (__IsPixelFormat8(objCImageDataPointer->GetPixelFormat()))
        {
            pBuffer = (BYTE*)objCImageDataPointer->GetBuffer();
        }
        else
        {
            pBuffer = (BYTE*)objCImageDataPointer->ConvertToRaw8(emValidBits);
        }
        // 黑白相机需要翻转数据后显示
        for (unsigned int i = 0; i < Height_; i++)
        {
            memcpy(m_pImageBuffer + i * Width_, pBuffer + (Height_ - i - 1) * Width_, (size_t)Width_);
        }
        pBuffer = m_pImageBuffer;
    }

    DWORD		         dwImageSize = (DWORD)(__GetStride(Width_, colorCamera_) * Height_);
        BITMAPFILEHEADER     stBfh = { 0 };
        DWORD		         dwBytesRead = 0;

        stBfh.bfType = (WORD)'M' << 8 | 'B';			 //定义文件类型
        stBfh.bfOffBits = colorCamera_ ? sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER)
            : sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + (256 * 4);	//定义文件头大小true为彩色,false为黑白
        stBfh.bfSize = stBfh.bfOffBits + dwImageSize; //文件大小

        DWORD dwBitmapInfoHeader = colorCamera_ ? sizeof(BITMAPINFOHEADER)
            : sizeof(BITMAPINFOHEADER) + (256 * 4);	//定义BitmapInfoHeader大小true为彩色,false为黑白
        const char* strEn = strFilePath.c_str();

        //将const char*转化为LPCTSTR
        size_t length = sizeof(TCHAR) * (strlen(strEn) + 1);
        LPTSTR tcBuffer = new TCHAR[length];
        memset(tcBuffer, 0, length);
        MultiByteToWideChar(CP_ACP, 0, strEn, (int) strlen(strEn), tcBuffer, (int) length);
        LPCTSTR  pDest = (LPCTSTR)tcBuffer;

        //创建文件
        HANDLE hFile = ::CreateFile(pDest,
            GENERIC_WRITE,
            0,
            NULL,
            CREATE_ALWAYS,
            FILE_ATTRIBUTE_NORMAL,
            NULL);

        if (hFile == INVALID_HANDLE_VALUE)
        {
            throw std::runtime_error("Handle is invalid");
        }
        ::WriteFile(hFile, &stBfh, sizeof(BITMAPFILEHEADER), &dwBytesRead, NULL);
        ::WriteFile(hFile, m_pBmpInfo, dwBitmapInfoHeader, &dwBytesRead, NULL); //黑白和彩色自适应
        ::WriteFile(hFile, pBuffer, dwImageSize, &dwBytesRead, NULL);
        CloseHandle(hFile);
}
void ClassGalaxy::SaveBmp(CImageDataPointer& objCImageDataPointer,void* buffer,const std::string& strFilePath)
{
    GX_VALID_BIT_LIST emValidBits = GX_BIT_0_7;
    BYTE* pBuffer = NULL;

    if ((strFilePath == ""))
    {
        throw std::runtime_error("Argument is error");
    }

    //检查图像是否改变并更新Buffer
    __UpdateBitmap(objCImageDataPointer);

    emValidBits = GetBestValudBit(objCImageDataPointer->GetPixelFormat());

    if (colorCamera_)
    {
        //BYTE* pBuffer = (BYTE*)objCImageDataPointer->ImageProcess(objCfg);

        pBuffer = (BYTE*)objCImageDataPointer->ConvertToRGB24(emValidBits, GX_RAW2RGB_NEIGHBOUR, true);
    }
    else
    {
        if (__IsPixelFormat8(objCImageDataPointer->GetPixelFormat()))
        {
            pBuffer = (BYTE*)objCImageDataPointer->GetBuffer();
        }
        else
        {
            pBuffer = (BYTE*)objCImageDataPointer->ConvertToRaw8(emValidBits);
        }
        // 黑白相机需要翻转数据后显示
        for (unsigned int i = 0; i < Height_; i++)
        {
            memcpy(m_pImageBuffer + i * Width_, pBuffer + (Height_ - i - 1) * Width_, (size_t)Width_);
        }
        pBuffer = m_pImageBuffer;
    }

    DWORD		         dwImageSize = (DWORD)(__GetStride(Width_, colorCamera_) * Height_);
    BITMAPFILEHEADER     stBfh = { 0 };
    DWORD		         dwBytesRead = 0;

    stBfh.bfType = (WORD)'M' << 8 | 'B';			 //定义文件类型
    stBfh.bfOffBits = colorCamera_ ? sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER)
        : sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + (256 * 4);	//定义文件头大小true为彩色,false为黑白
    stBfh.bfSize = stBfh.bfOffBits + dwImageSize; //文件大小

    DWORD dwBitmapInfoHeader = colorCamera_ ? sizeof(BITMAPINFOHEADER)
        : sizeof(BITMAPINFOHEADER) + (256 * 4);	//定义BitmapInfoHeader大小true为彩色,false为黑白
    const char* strEn = strFilePath.c_str();

    //将const char*转化为LPCTSTR
    size_t length = sizeof(TCHAR) * (strlen(strEn) + 1);
    LPTSTR tcBuffer = new TCHAR[length];
    memset(tcBuffer, 0, length);
    MultiByteToWideChar(CP_ACP, 0, strEn, (int) strlen(strEn), tcBuffer, (int) length);
    LPCTSTR  pDest = (LPCTSTR)tcBuffer;

    //创建文件
    HANDLE hFile = ::CreateFile(pDest,
        GENERIC_WRITE,
        0,
        NULL,
        CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL,
        NULL);

    if (hFile == INVALID_HANDLE_VALUE)
    {
        throw std::runtime_error("Handle is invalid");
    }
    ::WriteFile(hFile, &stBfh, sizeof(BITMAPFILEHEADER), &dwBytesRead, NULL);
    ::WriteFile(hFile, m_pBmpInfo, dwBitmapInfoHeader, &dwBytesRead, NULL); //黑白和彩色自适应
    //::WriteFile(hFile, pBuffer, dwImageSize, &dwBytesRead, NULL);
    
    ::WriteFile(hFile, buffer, dwImageSize, &dwBytesRead, NULL);
    CloseHandle(hFile);
}

void ClassGalaxy::SaveRaw(CImageDataPointer& objCImageDataPointer, const std::string& strFilePath)
{
    if ((objCImageDataPointer.IsNull()) || (strFilePath == ""))
    {
        throw std::runtime_error("Argument is error");
    }

    //检查图像是否改变并更新Buffer
    __UpdateBitmap(objCImageDataPointer);

    DWORD   dwImageSize = (DWORD)objCImageDataPointer->GetPayloadSize();  // 写入文件的长度
    DWORD   dwBytesRead = 0;                // 文件读取的长度

    BYTE* pbuffer = (BYTE*)objCImageDataPointer->GetBuffer();

    const char* strEn = strFilePath.c_str();

    //将const char*转化为LPCTSTR
    size_t length = sizeof(TCHAR) * (strlen(strEn) + 1);
    LPTSTR tcBuffer = new TCHAR[length];
    memset(tcBuffer, 0, length);
    MultiByteToWideChar(CP_ACP, 0, strEn, (int) strlen(strEn), tcBuffer, (int) length);
    LPCTSTR  pDest = (LPCTSTR)tcBuffer;
    // 创建文件
    HANDLE hFile = ::CreateFile(pDest,
        GENERIC_WRITE,
        FILE_SHARE_READ,
        NULL,
        CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL,
        NULL);

    if (hFile == INVALID_HANDLE_VALUE)   // 创建失败则返回
    {
        throw std::runtime_error("Handle is invalid");
    }
    else                                 // 保存Raw图像          
    {
        ::WriteFile(hFile, pbuffer, dwImageSize, &dwBytesRead, NULL);
        CloseHandle(hFile);
    }
}

int ClassGalaxy::HandleError(CGalaxyException cag)
{
   LogMessage(cag.what());
   SetErrorText(cag.GetErrorCode(), cag.what());
   return cag.GetErrorCode();
}

