// SpinnakerC device adapter
// Translated from the SpinnakerCamera device adapter (by Cairn), which used
// the Spinnaker C++ API; SpinnakerC uses the C API.

#include "SpinnakerCCamera.h"
#include "CameraImageMetadata.h"
#include "ModuleInterface.h"

#include <vector>
#include <string>
#include <algorithm>
#include <cstring>
#include <sstream>

namespace {

constexpr auto NODE_PIXEL_FORMAT = "PixelFormat";
constexpr auto NODE_PIXEL_SIZE = "PixelSize";
constexpr auto NODE_WIDTH = "Width";
constexpr auto NODE_HEIGHT = "Height";
constexpr auto NODE_OFFSET_X = "OffsetX";
constexpr auto NODE_OFFSET_Y = "OffsetY";
constexpr auto NODE_EXPOSURE_TIME = "ExposureTime";
constexpr auto NODE_EXPOSURE_AUTO = "ExposureAuto";
constexpr auto NODE_EXPOSURE_MODE = "ExposureMode";
constexpr auto NODE_TRIGGER_MODE = "TriggerMode";
constexpr auto NODE_TRIGGER_SOURCE = "TriggerSource";
constexpr auto NODE_TRIGGER_SOFTWARE = "TriggerSoftware";
constexpr auto NODE_ACQUISITION_MODE = "AcquisitionMode";
constexpr auto NODE_ACQUISITION_FRAME_COUNT = "AcquisitionFrameCount";
constexpr auto NODE_BINNING_HORIZONTAL = "BinningHorizontal";
constexpr auto NODE_BINNING_VERTICAL = "BinningVertical";

struct CamNameAndSN
{
   std::string name;
   std::string serialNumber;
};

std::string readNodeStringValue(spinNodeHandle hNode)
{
   char buf[256];
   size_t bufLen = sizeof(buf);
   if (spinNodeToString(hNode, buf, &bufLen) == SPINNAKER_ERR_SUCCESS)
      return std::string(buf);
   return {};
}

std::vector<CamNameAndSN> GetSpinnakerCCameraNamesAndSNs()
{
   std::vector<CamNameAndSN> out;

   spinSystem hSystem = nullptr;
   if (spinSystemGetInstance(&hSystem) != SPINNAKER_ERR_SUCCESS)
      return out;

   spinCameraList hCamList = nullptr;
   if (spinCameraListCreateEmpty(&hCamList) != SPINNAKER_ERR_SUCCESS)
   {
      spinSystemReleaseInstance(hSystem);
      return out;
   }

   if (spinSystemGetCameras(hSystem, hCamList) != SPINNAKER_ERR_SUCCESS)
   {
      spinCameraListDestroy(hCamList);
      spinSystemReleaseInstance(hSystem);
      return out;
   }

   size_t numCams = 0;
   spinCameraListGetSize(hCamList, &numCams);

   for (size_t i = 0; i < numCams; i++)
   {
      spinCamera hCam = nullptr;
      if (spinCameraListGet(hCamList, i, &hCam) != SPINNAKER_ERR_SUCCESS)
         continue;

      spinNodeMapHandle hTLDeviceNodeMap = nullptr;
      if (spinCameraGetTLDeviceNodeMap(hCam, &hTLDeviceNodeMap) != SPINNAKER_ERR_SUCCESS)
      {
         spinCameraRelease(hCam);
         continue;
      }

      CamNameAndSN camInfo;

      spinNodeHandle hModelName = nullptr;
      if (spinNodeMapGetNode(hTLDeviceNodeMap, "DeviceModelName", &hModelName) == SPINNAKER_ERR_SUCCESS)
      {
         bool8_t readable = False;
         spinNodeIsReadable(hModelName, &readable);
         if (readable)
            camInfo.name = readNodeStringValue(hModelName);
      }

      spinNodeHandle hSerial = nullptr;
      if (spinNodeMapGetNode(hTLDeviceNodeMap, "DeviceSerialNumber", &hSerial) == SPINNAKER_ERR_SUCCESS)
      {
         bool8_t readable = False;
         spinNodeIsReadable(hSerial, &readable);
         if (readable)
            camInfo.serialNumber = readNodeStringValue(hSerial);
      }

      out.push_back(camInfo);
      spinCameraRelease(hCam);
   }

   spinCameraListClear(hCamList);
   spinCameraListDestroy(hCamList);
   spinSystemReleaseInstance(hSystem);
   return out;
}

} // anonymous namespace


// --- Module entry points ---

MODULE_API void InitializeModuleData()
{
   auto camInfos = GetSpinnakerCCameraNamesAndSNs();
   for (const auto& info : camInfos)
   {
      if (!info.name.empty())
         RegisterDevice(info.name.c_str(), MM::CameraDevice, "FLIR Spinnaker C Camera");
   }
}

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
   return new SpinnakerCCamera(deviceName);
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
   delete pDevice;
}


// --- Node access helpers ---

spinError SpinnakerCCamera::getNodeHandle(const char* name, spinNodeHandle* hNode) const
{
   return spinNodeMapGetNode(m_nodeMap, name, hNode);
}

bool SpinnakerCCamera::isNodeReadable(spinNodeHandle hNode) const
{
   bool8_t readable = False;
   spinNodeIsReadable(hNode, &readable);
   return readable != False;
}

bool SpinnakerCCamera::isNodeWritable(spinNodeHandle hNode) const
{
   bool8_t writable = False;
   spinNodeIsWritable(hNode, &writable);
   return writable != False;
}

spinError SpinnakerCCamera::getEnumSymbolic(spinNodeHandle hNode, std::string& symbolic) const
{
   spinNodeHandle hEntry = nullptr;
   spinError err = spinEnumerationGetCurrentEntry(hNode, &hEntry);
   if (err != SPINNAKER_ERR_SUCCESS)
      return err;

   char buf[256];
   size_t bufLen = sizeof(buf);
   err = spinEnumerationEntryGetSymbolic(hEntry, buf, &bufLen);
   if (err != SPINNAKER_ERR_SUCCESS)
      return err;

   symbolic = buf;
   return SPINNAKER_ERR_SUCCESS;
}

spinError SpinnakerCCamera::setEnumByName(spinNodeHandle hNode, const char* symbolic) const
{
   spinNodeHandle hEntry = nullptr;
   spinError err = spinEnumerationGetEntryByName(hNode, symbolic, &hEntry);
   if (err != SPINNAKER_ERR_SUCCESS)
      return err;

   int64_t value = 0;
   err = spinEnumerationEntryGetIntValue(hEntry, &value);
   if (err != SPINNAKER_ERR_SUCCESS)
      return err;

   return spinEnumerationSetIntValue(hNode, value);
}

spinError SpinnakerCCamera::getEnumSymbolics(spinNodeHandle hNode, std::vector<std::string>& symbolics) const
{
   symbolics.clear();
   size_t numEntries = 0;
   spinError err = spinEnumerationGetNumEntries(hNode, &numEntries);
   if (err != SPINNAKER_ERR_SUCCESS)
      return err;

   for (size_t i = 0; i < numEntries; i++)
   {
      spinNodeHandle hEntry = nullptr;
      err = spinEnumerationGetEntryByIndex(hNode, i, &hEntry);
      if (err != SPINNAKER_ERR_SUCCESS)
         continue;

      bool8_t avail = False;
      spinNodeIsAvailable(hEntry, &avail);
      if (!avail)
         continue;

      char buf[256];
      size_t bufLen = sizeof(buf);
      err = spinEnumerationEntryGetSymbolic(hEntry, buf, &bufLen);
      if (err == SPINNAKER_ERR_SUCCESS)
         symbolics.push_back(buf);
   }
   return SPINNAKER_ERR_SUCCESS;
}

spinError SpinnakerCCamera::getEnumIntValue(spinNodeHandle hNode, int64_t& value) const
{
   spinNodeHandle hEntry = nullptr;
   spinError err = spinEnumerationGetCurrentEntry(hNode, &hEntry);
   if (err != SPINNAKER_ERR_SUCCESS)
      return err;
   return spinEnumerationEntryGetIntValue(hEntry, &value);
}

spinError SpinnakerCCamera::getFloatValue(spinNodeHandle hNode, double& value) const
{
   return spinFloatGetValue(hNode, &value);
}

spinError SpinnakerCCamera::setFloatValue(spinNodeHandle hNode, double value) const
{
   return spinFloatSetValue(hNode, value);
}

spinError SpinnakerCCamera::getIntValue(spinNodeHandle hNode, int64_t& value) const
{
   return spinIntegerGetValue(hNode, &value);
}

spinError SpinnakerCCamera::setIntValue(spinNodeHandle hNode, int64_t value) const
{
   return spinIntegerSetValue(hNode, value);
}

spinError SpinnakerCCamera::getIntMin(spinNodeHandle hNode, int64_t& value) const
{
   return spinIntegerGetMin(hNode, &value);
}

spinError SpinnakerCCamera::getIntMax(spinNodeHandle hNode, int64_t& value) const
{
   return spinIntegerGetMax(hNode, &value);
}

spinError SpinnakerCCamera::getIntInc(spinNodeHandle hNode, int64_t& value) const
{
   return spinIntegerGetInc(hNode, &value);
}

spinError SpinnakerCCamera::getBoolValue(spinNodeHandle hNode, bool8_t& value) const
{
   return spinBooleanGetValue(hNode, &value);
}

spinError SpinnakerCCamera::setBoolValue(spinNodeHandle hNode, bool8_t value) const
{
   return spinBooleanSetValue(hNode, value);
}

spinError SpinnakerCCamera::executeCommand(spinNodeHandle hNode) const
{
   return spinCommandExecute(hNode);
}


// --- Error handling ---

int SpinnakerCCamera::checkError(spinError err, const char* context)
{
   if (err == SPINNAKER_ERR_SUCCESS)
      return DEVICE_OK;

   char errMsg[512];
   size_t errMsgLen = sizeof(errMsg);
   if (spinErrorGetLastMessage(errMsg, &errMsgLen) != SPINNAKER_ERR_SUCCESS)
      snprintf(errMsg, sizeof(errMsg), "Spinnaker error %d", static_cast<int>(err));

   std::string fullMsg = std::string(context) + ": " + errMsg;
   SetErrorText(SPKRC_ERROR, fullMsg.c_str());
   return SPKRC_ERROR;
}


// --- Property creation helpers ---

void SpinnakerCCamera::CreatePropertyFromEnum(const char* nodeName, const char* mmPropName,
   int (SpinnakerCCamera::*fpt)(MM::PropertyBase* pProp, MM::ActionType eAct))
{
   spinNodeHandle hNode = nullptr;
   if (getNodeHandle(nodeName, &hNode) != SPINNAKER_ERR_SUCCESS)
   {
      LogMessage(std::string(mmPropName) + " property not created: node not found");
      return;
   }

   bool readable = isNodeReadable(hNode);
   bool writable = isNodeWritable(hNode);

   if (!readable)
   {
      auto pAct = new CPropertyAction(this, fpt);
      CreateProperty(mmPropName, "", MM::String, true, pAct);
      AddAllowedValue(mmPropName, "");
      return;
   }

   auto pAct = new CPropertyAction(this, fpt);
   bool readOnly = !writable;

   std::string current;
   getEnumSymbolic(hNode, current);

   std::vector<std::string> symbolics;
   getEnumSymbolics(hNode, symbolics);

   CreateProperty(mmPropName, current.c_str(), MM::String, readOnly, pAct);
   for (const auto& s : symbolics)
      AddAllowedValue(mmPropName, s.c_str());
}

void SpinnakerCCamera::CreatePropertyFromFloat(const char* nodeName, const char* mmPropName,
   int (SpinnakerCCamera::*fpt)(MM::PropertyBase* pProp, MM::ActionType eAct))
{
   spinNodeHandle hNode = nullptr;
   if (getNodeHandle(nodeName, &hNode) != SPINNAKER_ERR_SUCCESS)
   {
      LogMessage(std::string(mmPropName) + " property not created: node not found");
      return;
   }

   bool readable = isNodeReadable(hNode);
   bool writable = isNodeWritable(hNode);

   if (!readable)
   {
      auto pAct = new CPropertyAction(this, fpt);
      CreateProperty(mmPropName, "0", MM::Float, true, pAct);
      return;
   }

   auto pAct = new CPropertyAction(this, fpt);
   bool readOnly = !writable;

   double val = 0.0;
   getFloatValue(hNode, val);

   char buf[64];
   snprintf(buf, sizeof(buf), "%f", val);
   CreateProperty(mmPropName, buf, MM::Float, readOnly, pAct);
}

void SpinnakerCCamera::CreatePropertyFromBool(const char* nodeName, const char* mmPropName,
   int (SpinnakerCCamera::*fpt)(MM::PropertyBase* pProp, MM::ActionType eAct))
{
   spinNodeHandle hNode = nullptr;
   if (getNodeHandle(nodeName, &hNode) != SPINNAKER_ERR_SUCCESS)
   {
      LogMessage(std::string(mmPropName) + " property not created: node not found");
      return;
   }

   bool readable = isNodeReadable(hNode);
   bool writable = isNodeWritable(hNode);

   if (!readable)
   {
      auto pAct = new CPropertyAction(this, fpt);
      CreateProperty(mmPropName, "0", MM::Integer, true, pAct);
      AddAllowedValue(mmPropName, "0");
      AddAllowedValue(mmPropName, "1");
      return;
   }

   auto pAct = new CPropertyAction(this, fpt);
   bool readOnly = !writable;

   bool8_t val = False;
   getBoolValue(hNode, val);

   CreateProperty(mmPropName, val ? "1" : "0", MM::Integer, readOnly, pAct);
   AddAllowedValue(mmPropName, "0");
   AddAllowedValue(mmPropName, "1");
}


// --- Property change handlers ---

int SpinnakerCCamera::OnEnumPropertyChanged(const char* nodeName,
   MM::PropertyBase* pProp, MM::ActionType eAct)
{
   spinNodeHandle hNode = nullptr;
   if (getNodeHandle(nodeName, &hNode) != SPINNAKER_ERR_SUCCESS)
      return DEVICE_OK;

   if (!isNodeReadable(hNode))
      return DEVICE_OK;

   if (eAct == MM::BeforeGet)
   {
      auto mmProp = dynamic_cast<MM::Property*>(pProp);
      if (mmProp != nullptr)
      {
         mmProp->SetReadOnly(!isNodeWritable(hNode));
         mmProp->ClearAllowedValues();
         std::vector<std::string> symbolics;
         getEnumSymbolics(hNode, symbolics);
         for (const auto& s : symbolics)
            mmProp->AddAllowedValue(s.c_str());
      }

      std::string current;
      if (getEnumSymbolic(hNode, current) == SPINNAKER_ERR_SUCCESS)
         pProp->Set(current.c_str());
      else
         pProp->Set("");
   }
   else if (eAct == MM::AfterSet)
   {
      std::string val;
      pProp->Get(val);

      spinError err = setEnumByName(hNode, val.c_str());
      if (err != SPINNAKER_ERR_SUCCESS)
         return checkError(err, ("Could not write " + pProp->GetName()).c_str());
   }
   return DEVICE_OK;
}

int SpinnakerCCamera::OnFloatPropertyChanged(const char* nodeName,
   MM::PropertyBase* pProp, MM::ActionType eAct)
{
   spinNodeHandle hNode = nullptr;
   if (getNodeHandle(nodeName, &hNode) != SPINNAKER_ERR_SUCCESS)
      return DEVICE_OK;

   if (!isNodeReadable(hNode))
      return DEVICE_OK;

   if (eAct == MM::BeforeGet)
   {
      auto mmProp = dynamic_cast<MM::Property*>(pProp);
      if (mmProp != nullptr)
         mmProp->SetReadOnly(!isNodeWritable(hNode));

      double val = 0.0;
      spinError err = getFloatValue(hNode, val);
      if (err != SPINNAKER_ERR_SUCCESS)
         return checkError(err, ("Could not read " + pProp->GetName()).c_str());
      pProp->Set(val);
   }
   else if (eAct == MM::AfterSet)
   {
      double val;
      pProp->Get(val);

      spinError err = setFloatValue(hNode, val);
      if (err != SPINNAKER_ERR_SUCCESS)
         return checkError(err, ("Could not write " + pProp->GetName()).c_str());
   }
   return DEVICE_OK;
}

int SpinnakerCCamera::OnBoolPropertyChanged(const char* nodeName,
   MM::PropertyBase* pProp, MM::ActionType eAct)
{
   spinNodeHandle hNode = nullptr;
   if (getNodeHandle(nodeName, &hNode) != SPINNAKER_ERR_SUCCESS)
      return DEVICE_OK;

   if (!isNodeReadable(hNode))
      return DEVICE_OK;

   if (eAct == MM::BeforeGet)
   {
      auto mmProp = dynamic_cast<MM::Property*>(pProp);
      if (mmProp != nullptr)
         mmProp->SetReadOnly(!isNodeWritable(hNode));

      bool8_t val = False;
      spinError err = getBoolValue(hNode, val);
      if (err != SPINNAKER_ERR_SUCCESS)
         return checkError(err, ("Could not read " + pProp->GetName()).c_str());
      pProp->Set(static_cast<long>(val != False));
   }
   else if (eAct == MM::AfterSet)
   {
      long val;
      pProp->Get(val);

      spinError err = setBoolValue(hNode, val != 0 ? True : False);
      if (err != SPINNAKER_ERR_SUCCESS)
         return checkError(err, ("Could not write " + pProp->GetName()).c_str());
   }
   return DEVICE_OK;
}


// --- Constructor / Destructor ---

SpinnakerCCamera::SpinnakerCCamera(const char* deviceName)
   :
   m_deviceName(deviceName ? deviceName : ""),
   m_system(nullptr),
   m_cam(nullptr),
   m_imagePtr(nullptr),
   m_nodeMap(nullptr),
   m_imageBuff(nullptr),
   m_aqThread(nullptr),
   m_stopOnOverflow(false)
{
   InitializeDefaultErrorMessages();

   auto camInfos = GetSpinnakerCCameraNamesAndSNs();

   std::string serialNumber;
   if (!camInfos.empty())
      serialNumber = camInfos[0].serialNumber;

   CreateProperty("Serial Number", serialNumber.c_str(), MM::String, false, nullptr, true);

   for (const auto& info : camInfos)
      if (info.name == m_deviceName)
         AddAllowedValue("Serial Number", info.serialNumber.c_str());

   m_aqThread = new SpinnakerCAcquisitionThread(this);
}

SpinnakerCCamera::~SpinnakerCCamera()
{
   StopSequenceAcquisition();
   delete m_aqThread;
   Shutdown();
}

void SpinnakerCCamera::GetName(char* name) const
{
   CDeviceUtils::CopyLimitedString(name, m_serialNumber.c_str());
}


// --- Initialize / Shutdown ---

int SpinnakerCCamera::Initialize()
{
   char snBuf[MM::MaxStrLength];
   GetProperty("Serial Number", snBuf);
   m_serialNumber = snBuf;

   if (m_serialNumber.empty())
   {
      SetErrorText(SPKRC_ERROR, "No Serial Number Provided! Cannot Identify Camera!");
      return SPKRC_ERROR;
   }

   spinError err = spinSystemGetInstance(&m_system);
   if (err != SPINNAKER_ERR_SUCCESS || m_system == nullptr)
   {
      SetErrorText(SPKRC_ERROR, "Spinnaker System Object Pointer is Null!");
      return SPKRC_ERROR;
   }

   spinCameraList hCamList = nullptr;
   spinCameraListCreateEmpty(&hCamList);
   spinSystemGetCameras(m_system, hCamList);

   size_t numCams = 0;
   spinCameraListGetSize(hCamList, &numCams);

   if (numCams == 0)
   {
      spinCameraListClear(hCamList);
      spinCameraListDestroy(hCamList);
      SetErrorText(SPKRC_ERROR, "No Cameras Attached!");
      return SPKRC_ERROR;
   }

   for (size_t i = 0; i < numCams; i++)
   {
      spinCamera hCam = nullptr;
      if (spinCameraListGet(hCamList, i, &hCam) != SPINNAKER_ERR_SUCCESS)
         continue;

      spinNodeMapHandle hTLNodeMap = nullptr;
      spinCameraGetTLDeviceNodeMap(hCam, &hTLNodeMap);

      spinNodeHandle hSNNode = nullptr;
      if (spinNodeMapGetNode(hTLNodeMap, "DeviceSerialNumber", &hSNNode) == SPINNAKER_ERR_SUCCESS)
      {
         bool8_t readable = False;
         spinNodeIsReadable(hSNNode, &readable);
         if (readable)
         {
            std::string sn = readNodeStringValue(hSNNode);
            if (sn == m_serialNumber)
            {
               m_cam = hCam;
               spinCameraInit(m_cam);
               break;
            }
         }
      }

      spinCameraRelease(hCam);
   }

   spinCameraListClear(hCamList);
   spinCameraListDestroy(hCamList);

   if (m_cam == nullptr)
   {
      Shutdown();
      std::string msg = "Could not find camera with serial number: " + m_serialNumber;
      SetErrorText(SPKRC_ERROR, msg.c_str());
      return SPKRC_ERROR;
   }

   spinCameraGetNodeMap(m_cam, &m_nodeMap);

   // Save original trigger mode, then set initial values
   spinNodeHandle hTriggerMode = nullptr;
   std::string originalTriggerMode;
   if (getNodeHandle(NODE_TRIGGER_MODE, &hTriggerMode) == SPINNAKER_ERR_SUCCESS &&
       isNodeReadable(hTriggerMode))
   {
      getEnumSymbolic(hTriggerMode, originalTriggerMode);
   }

   // Set initial values for acquisition
   spinNodeHandle hNode = nullptr;

   if (getNodeHandle(NODE_EXPOSURE_AUTO, &hNode) == SPINNAKER_ERR_SUCCESS && isNodeWritable(hNode))
      setEnumByName(hNode, "Off");

   if (getNodeHandle(NODE_EXPOSURE_MODE, &hNode) == SPINNAKER_ERR_SUCCESS && isNodeWritable(hNode))
      setEnumByName(hNode, "Timed");

   if (getNodeHandle(NODE_ACQUISITION_MODE, &hNode) == SPINNAKER_ERR_SUCCESS && isNodeWritable(hNode))
      setEnumByName(hNode, "SingleFrame");

   if (getNodeHandle(NODE_TRIGGER_MODE, &hNode) == SPINNAKER_ERR_SUCCESS && isNodeWritable(hNode))
      setEnumByName(hNode, "Off");

   // Frame rate auto (manual node lookup like original)
   spinNodeHandle hAFRA = nullptr;
   if (getNodeHandle("AcquisitionFrameRateAuto", &hAFRA) == SPINNAKER_ERR_SUCCESS && isNodeReadable(hAFRA))
   {
      auto pAct = new CPropertyAction(this, &SpinnakerCCamera::OnFrameRateAuto);
      std::vector<std::string> symbolics;
      getEnumSymbolics(hAFRA, symbolics);
      if (!symbolics.empty())
      {
         CreateProperty("Frame Rate Auto", symbolics[0].c_str(), MM::String, false, pAct);
         for (const auto& s : symbolics)
            AddAllowedValue("Frame Rate Auto", s.c_str());
      }
   }

   // Frame rate enable (manual node lookup like original)
   spinNodeHandle hAFRCE = nullptr;
   if (getNodeHandle("AcquisitionFrameRateEnable", &hAFRCE) == SPINNAKER_ERR_SUCCESS && isNodeReadable(hAFRCE))
   {
      LogMessage("Creating frame rate enabled...");
      auto pAct = new CPropertyAction(this, &SpinnakerCCamera::OnFrameRateEnabled);
      CreateProperty("Frame Rate Control Enabled", "0", MM::Integer, false, pAct);
      AddAllowedValue("Frame Rate Control Enabled", "1");
      AddAllowedValue("Frame Rate Control Enabled", "0");
   }
   else
   {
      LogMessage("Failed to create frame rate enabled...");
   }

   // ADC Bit Depth (manual node lookup like original)
   spinNodeHandle hADC = nullptr;
   if (getNodeHandle("AdcBitDepth", &hADC) == SPINNAKER_ERR_SUCCESS && isNodeReadable(hADC))
   {
      auto pAct = new CPropertyAction(this, &SpinnakerCCamera::OnADCBitDepth);
      std::vector<std::string> symbolics;
      getEnumSymbolics(hADC, symbolics);
      if (!symbolics.empty())
      {
         CreateProperty("ADC Bit Depth", symbolics[0].c_str(), MM::String, false, pAct);
         for (const auto& s : symbolics)
            AddAllowedValue("ADC Bit Depth", s.c_str());
      }
   }

   // Binning
   spinNodeHandle hVM = nullptr;
   spinNodeHandle hBH = nullptr;
   spinNodeHandle hBV = nullptr;
   getNodeHandle("VideoMode", &hVM);
   getNodeHandle(NODE_BINNING_HORIZONTAL, &hBH);
   getNodeHandle(NODE_BINNING_VERTICAL, &hBV);

   if (hVM != nullptr && isNodeWritable(hVM))
   {
      LogMessage("Using VideoMode for Binning");

      std::string currentVideoMode;
      getEnumSymbolic(hVM, currentVideoMode);

      std::vector<std::string> videoModes;
      getEnumSymbolics(hVM, videoModes);

      CreateIntegerProperty(MM::g_Keyword_Binning, 1, true);

      auto pAct = new CPropertyAction(this, &SpinnakerCCamera::OnVideoMode);
      CreateStringProperty("Video Mode", currentVideoMode.c_str(), false, pAct);
      for (const auto& vm : videoModes)
         AddAllowedValue("Video Mode", vm.c_str());

      for (const auto& vm : videoModes)
      {
         setEnumByName(hVM, vm.c_str());
         spinNodeHandle hBC = nullptr;
         if (getNodeHandle("BinningControl", &hBC) != SPINNAKER_ERR_SUCCESS)
            continue;
         if (isNodeWritable(hBC))
            continue;

         std::vector<std::string> binningModes;
         getEnumSymbolics(hBC, binningModes);

         std::string currentBinning;
         getEnumSymbolic(hBC, currentBinning);

         auto pActBM = new CPropertyAction(this, &SpinnakerCCamera::OnBinningModeEnum);
         CreateProperty("Binning Mode", currentBinning.c_str(), MM::String, false, pActBM);
         for (const auto& bm : binningModes)
            AddAllowedValue("Binning Mode", bm.c_str());

         setEnumByName(hVM, currentVideoMode.c_str());
         break;
      }
   }
   else if (hBH != nullptr && isNodeWritable(hBH) && hBV != nullptr && isNodeWritable(hBV))
   {
      LogMessage("Using BinningHorizontal and BinningVertical for Binning");

      setIntValue(hBH, 1);
      setIntValue(hBV, 1);

      auto pAct = new CPropertyAction(this, &SpinnakerCCamera::OnBinningInt);
      CreateProperty(MM::g_Keyword_Binning, "1", MM::String, false, pAct);

      int64_t maxH = 0, maxV = 0, minH = 0, minV = 0;
      getIntMax(hBH, maxH);
      getIntMax(hBV, maxV);
      getIntMin(hBH, minH);
      getIntMin(hBV, minV);

      int64_t maxBin = (std::min)(maxH, maxV);
      int64_t minBin = (std::max)(minH, minV);

      for (int64_t i = minBin; i <= maxBin; i++)
         AddAllowedValue(MM::g_Keyword_Binning, std::to_string(i).c_str());
   }
   else
   {
      LogMessage("Unknown Binning Control");
   }

   CreatePropertyFromEnum(NODE_PIXEL_FORMAT, "Pixel Format", &SpinnakerCCamera::OnPixelFormat);
   CreatePropertyFromEnum("TestPattern", "Test Pattern", &SpinnakerCCamera::OnTestPattern);
   CreatePropertyFromFloat("AcquisitionFrameRate", "Frame Rate", &SpinnakerCCamera::OnFrameRate);
   CreatePropertyFromFloat("Gain", "Gain", &SpinnakerCCamera::OnGain);
   CreatePropertyFromEnum("GainAuto", "Gain Auto", &SpinnakerCCamera::OnGainAuto);
   CreatePropertyFromEnum(NODE_EXPOSURE_AUTO, "Exposure Auto", &SpinnakerCCamera::OnExposureAuto);
   CreatePropertyFromBool("GammaEnable", "Gamma Enabled", &SpinnakerCCamera::OnGammaEnabled);
   CreatePropertyFromFloat("Gamma", "Gamma", &SpinnakerCCamera::OnGamma);
   CreatePropertyFromFloat("BlackLevel", "Black Level", &SpinnakerCCamera::OnBlackLevel);
   CreatePropertyFromEnum("BlackLevelAuto", "Black Level Auto", &SpinnakerCCamera::OnBlackLevelAuto);
   CreatePropertyFromBool("ReverseX", "Reverse X", &SpinnakerCCamera::OnReverseX);
   CreatePropertyFromBool("ReverseY", "Reverse Y", &SpinnakerCCamera::OnReverseY);

   // Enable TriggerMode to create trigger properties
   if (getNodeHandle(NODE_TRIGGER_MODE, &hNode) == SPINNAKER_ERR_SUCCESS && isNodeWritable(hNode))
      setEnumByName(hNode, "On");

   CreatePropertyFromEnum("TriggerSelector", "Trigger Selector", &SpinnakerCCamera::OnTriggerSelector);
   CreatePropertyFromEnum(NODE_TRIGGER_MODE, "Trigger Mode", &SpinnakerCCamera::OnTriggerMode);
   CreatePropertyFromEnum(NODE_TRIGGER_SOURCE, "Trigger Source", &SpinnakerCCamera::OnTriggerSource);
   CreatePropertyFromEnum("TriggerActivation", "Trigger Activation", &SpinnakerCCamera::OnTriggerActivation);
   CreatePropertyFromEnum("TriggerOverlap", "Trigger Overlap", &SpinnakerCCamera::OnTriggerOverlap);
   CreatePropertyFromFloat("TriggerDelay", "Trigger Delay", &SpinnakerCCamera::OnTriggerDelay);
   CreatePropertyFromEnum(NODE_EXPOSURE_MODE, "Exposure Mode", &SpinnakerCCamera::OnExposureMode);
   CreatePropertyFromEnum("UserOutputSelector", "User Output Selector", &SpinnakerCCamera::OnUserOutputSelector);
   CreatePropertyFromBool("UserOutputValue", "User Output Value", &SpinnakerCCamera::OnUserOutputValue);

   CreatePropertyFromEnum("LineSelector", "Line Selector", &SpinnakerCCamera::OnLineSelector);
   CreatePropertyFromEnum("LineMode", "Line Mode", &SpinnakerCCamera::OnLineMode);
   CreatePropertyFromBool("LineInverter", "Line Inverter", &SpinnakerCCamera::OnLineInverter);
   CreatePropertyFromEnum("LineSource", "Line Source", &SpinnakerCCamera::OnLineSource);

   // Restore original trigger mode
   if (!originalTriggerMode.empty())
   {
      if (getNodeHandle(NODE_TRIGGER_MODE, &hNode) == SPINNAKER_ERR_SUCCESS && isNodeWritable(hNode))
      {
         err = setEnumByName(hNode, originalTriggerMode.c_str());
         if (err != SPINNAKER_ERR_SUCCESS)
            return checkError(err, "Could not restore trigger mode");
      }
   }

   ClearROI();

   return DEVICE_OK;
}

int SpinnakerCCamera::Shutdown()
{
   if (m_cam != nullptr)
   {
      spinCameraDeInit(m_cam);
      spinCameraRelease(m_cam);
      m_cam = nullptr;
   }
   m_nodeMap = nullptr;

   if (m_system != nullptr)
   {
      spinSystemReleaseInstance(m_system);
      m_system = nullptr;
   }

   delete[] m_imageBuff;
   m_imageBuff = nullptr;

   return DEVICE_OK;
}


// --- Pixel format helpers ---

int64_t SpinnakerCCamera::getPixelFormatEnumValue() const
{
   spinNodeHandle hNode = nullptr;
   if (getNodeHandle(NODE_PIXEL_FORMAT, &hNode) != SPINNAKER_ERR_SUCCESS)
      return -1;
   int64_t val = -1;
   getEnumIntValue(hNode, val);
   return val;
}

int64_t SpinnakerCCamera::getPixelSizeEnumValue() const
{
   spinNodeHandle hNode = nullptr;
   if (getNodeHandle(NODE_PIXEL_SIZE, &hNode) != SPINNAKER_ERR_SUCCESS)
      return -1;
   int64_t val = -1;
   getEnumIntValue(hNode, val);
   return val;
}


// --- Image acquisition ---

int SpinnakerCCamera::SnapImage()
{
   MMThreadGuard g(m_pixelLock);

   spinError err = spinCameraBeginAcquisition(m_cam);
   if (err != SPINNAKER_ERR_SUCCESS)
      return checkError(err, "BeginAcquisition");

   // Software trigger if needed
   spinNodeHandle hTrigMode = nullptr;
   spinNodeHandle hTrigSource = nullptr;
   if (getNodeHandle(NODE_TRIGGER_MODE, &hTrigMode) == SPINNAKER_ERR_SUCCESS &&
       getNodeHandle(NODE_TRIGGER_SOURCE, &hTrigSource) == SPINNAKER_ERR_SUCCESS)
   {
      std::string mode, source;
      getEnumSymbolic(hTrigMode, mode);
      getEnumSymbolic(hTrigSource, source);
      if (mode == "On" && source == "Software")
      {
         spinNodeHandle hTrigSw = nullptr;
         if (getNodeHandle(NODE_TRIGGER_SOFTWARE, &hTrigSw) == SPINNAKER_ERR_SUCCESS)
            executeCommand(hTrigSw);
      }
   }

   err = spinCameraGetNextImageEx(m_cam, static_cast<uint64_t>(GetExposure()) + 1000, &m_imagePtr);
   if (err != SPINNAKER_ERR_SUCCESS)
   {
      spinCameraEndAcquisition(m_cam);
      return checkError(err, "GetNextImage");
   }

   return DEVICE_OK;
}

const unsigned char* SpinnakerCCamera::GetImageBuffer()
{
   MMThreadGuard g(m_pixelLock);

   if (m_imagePtr == nullptr)
      return nullptr;

   bool8_t incomplete = False;
   spinImageIsIncomplete(m_imagePtr, &incomplete);

   if (!incomplete)
   {
      int64_t pixFmt = getPixelFormatEnumValue();
      allocateImageBuffer(GetImageBufferSize(), pixFmt);

      if (m_imageBuff)
      {
         void* pData = nullptr;
         spinImageGetData(m_imagePtr, &pData);
         size_t bufSize = 0;
         spinImageGetBufferSize(m_imagePtr, &bufSize);

         if (pixFmt == PixelFormat_RGB8 || pixFmt == PixelFormat_RGB8Packed)
         {
            size_t theirSizeD3 = bufSize / 3;
            size_t ourSizeD4 = static_cast<size_t>(GetImageBufferSize()) / 4;
            size_t minSize = theirSizeD3 > ourSizeD4 ? ourSizeD4 : theirSizeD3;
            size_t size = minSize * 3;
            RGBtoBGRA(static_cast<uint8_t*>(pData), size);
         }
         else if (pixFmt == PixelFormat_Mono12p)
         {
            size_t w = 0, h = 0;
            spinImageGetWidth(m_imagePtr, &w);
            spinImageGetHeight(m_imagePtr, &h);
            Unpack12Bit(reinterpret_cast<uint16_t*>(m_imageBuff),
               static_cast<uint8_t*>(pData), w, h, false);
         }
         else if (pixFmt == PixelFormat_Mono12Packed)
         {
            size_t w = 0, h = 0;
            spinImageGetWidth(m_imagePtr, &w);
            spinImageGetHeight(m_imagePtr, &h);
            Unpack12Bit(reinterpret_cast<uint16_t*>(m_imageBuff),
               static_cast<uint8_t*>(pData), w, h, true);
         }
         else
         {
            size_t length = bufSize > static_cast<size_t>(GetImageBufferSize()) ?
               static_cast<size_t>(GetImageBufferSize()) : bufSize;
            std::memcpy(m_imageBuff, pData, length);
         }
      }
      else
      {
         LogMessage("Failed to allocate memory for image buffer!");
         spinImageRelease(m_imagePtr);
         m_imagePtr = nullptr;
         spinCameraEndAcquisition(m_cam);
         return nullptr;
      }
   }
   else
   {
      LogMessage("Image incomplete");
   }

   spinImageRelease(m_imagePtr);
   m_imagePtr = nullptr;

   spinCameraEndAcquisition(m_cam);

   return m_imageBuff;
}

unsigned SpinnakerCCamera::GetImageWidth() const
{
   spinNodeHandle hNode = nullptr;
   if (getNodeHandle(NODE_WIDTH, &hNode) != SPINNAKER_ERR_SUCCESS)
      return 0;
   int64_t val = 0;
   getIntValue(hNode, val);
   return static_cast<unsigned>(val);
}

unsigned SpinnakerCCamera::GetImageHeight() const
{
   spinNodeHandle hNode = nullptr;
   if (getNodeHandle(NODE_HEIGHT, &hNode) != SPINNAKER_ERR_SUCCESS)
      return 0;
   int64_t val = 0;
   getIntValue(hNode, val);
   return static_cast<unsigned>(val);
}

unsigned SpinnakerCCamera::GetImageBytesPerPixel() const
{
   int64_t pixFmt = getPixelFormatEnumValue();

   if (pixFmt == PixelFormat_RGB8 || pixFmt == PixelFormat_RGB8Packed || pixFmt == PixelFormat_BGRa8)
      return 4;

   int64_t pixSize = getPixelSizeEnumValue();
   switch (pixSize)
   {
   case PixelSize_Bpp1:
   case PixelSize_Bpp2:
   case PixelSize_Bpp4:
   case PixelSize_Bpp8:
      return 1;
   case PixelSize_Bpp10:
   case PixelSize_Bpp12:
   case PixelSize_Bpp14:
   case PixelSize_Bpp16:
      return 2;
   case PixelSize_Bpp20:
   case PixelSize_Bpp24:
      return 3;
   case PixelSize_Bpp30:
   case PixelSize_Bpp32:
      return 4;
   case PixelSize_Bpp48:
      return 6;
   case PixelSize_Bpp64:
      return 8;
   case PixelSize_Bpp96:
      return 12;
   }
   return 0;
}

unsigned SpinnakerCCamera::GetNumberOfComponents() const
{
   int64_t pixFmt = getPixelFormatEnumValue();
   if (pixFmt == PixelFormat_RGB8 || pixFmt == PixelFormat_RGB8Packed || pixFmt == PixelFormat_BGRa8)
      return 4;
   return 1;
}

unsigned SpinnakerCCamera::GetBitDepth() const
{
   int64_t pixFmt = getPixelFormatEnumValue();
   if (pixFmt == PixelFormat_RGB8 || pixFmt == PixelFormat_RGB8Packed || pixFmt == PixelFormat_BGRa8)
      return 8;

   int64_t pixSize = getPixelSizeEnumValue();
   switch (pixSize)
   {
   case PixelSize_Bpp1:  return 1;
   case PixelSize_Bpp2:  return 2;
   case PixelSize_Bpp4:  return 4;
   case PixelSize_Bpp8:  return 8;
   case PixelSize_Bpp10: return 10;
   case PixelSize_Bpp12: return 12;
   case PixelSize_Bpp14: return 14;
   case PixelSize_Bpp16: return 16;
   case PixelSize_Bpp20: return 20;
   case PixelSize_Bpp24: return 24;
   case PixelSize_Bpp30: return 30;
   case PixelSize_Bpp32: return 32;
   case PixelSize_Bpp48: return 48;
   case PixelSize_Bpp64: return 64;
   case PixelSize_Bpp96: return 96;
   }
   return 0;
}

long SpinnakerCCamera::GetImageBufferSize() const
{
   return static_cast<long>(GetImageWidth()) * GetImageHeight() * GetImageBytesPerPixel();
}


// --- Exposure ---

double SpinnakerCCamera::GetExposure() const
{
   spinNodeHandle hNode = nullptr;
   if (getNodeHandle(NODE_EXPOSURE_TIME, &hNode) != SPINNAKER_ERR_SUCCESS)
      return 0.0;
   double val = 0.0;
   getFloatValue(hNode, val);
   return val / 1000.0;
}

void SpinnakerCCamera::SetExposure(double exp)
{
   spinNodeHandle hNode = nullptr;
   if (getNodeHandle(NODE_EXPOSURE_TIME, &hNode) == SPINNAKER_ERR_SUCCESS && isNodeWritable(hNode))
   {
      spinError err = setFloatValue(hNode, exp * 1000.0);
      if (err == SPINNAKER_ERR_SUCCESS)
         GetCoreCallback()->OnExposureChanged(this, exp);
   }
}


// --- ROI ---

int SpinnakerCCamera::SetROI(unsigned x, unsigned y, unsigned xSize, unsigned ySize)
{
   spinNodeHandle hOffX = nullptr, hOffY = nullptr, hW = nullptr, hH = nullptr;
   if (getNodeHandle(NODE_OFFSET_X, &hOffX) != SPINNAKER_ERR_SUCCESS ||
       getNodeHandle(NODE_OFFSET_Y, &hOffY) != SPINNAKER_ERR_SUCCESS ||
       getNodeHandle(NODE_WIDTH, &hW) != SPINNAKER_ERR_SUCCESS ||
       getNodeHandle(NODE_HEIGHT, &hH) != SPINNAKER_ERR_SUCCESS)
   {
      SetErrorText(SPKRC_ERROR, "Could not access ROI nodes");
      return SPKRC_ERROR;
   }

   int64_t offXMin = 0, offYMin = 0;
   getIntMin(hOffX, offXMin);
   getIntMin(hOffY, offYMin);
   setIntValue(hOffX, offXMin);
   setIntValue(hOffY, offYMin);

   int64_t wInc = 1, hInc = 1, wMax = 0, hMax = 0;
   getIntInc(hW, wInc);
   getIntInc(hH, hInc);
   getIntMax(hW, wMax);
   getIntMax(hH, hMax);

   int64_t w = static_cast<int64_t>(xSize) - static_cast<int64_t>(xSize) % wInc;
   int64_t h = static_cast<int64_t>(ySize) - static_cast<int64_t>(ySize) % hInc;
   w = (std::min)(w, wMax);
   h = (std::min)(h, hMax);

   spinError err = setIntValue(hW, w);
   if (err != SPINNAKER_ERR_SUCCESS) { ClearROI(); return checkError(err, "Could not set ROI width"); }
   err = setIntValue(hH, h);
   if (err != SPINNAKER_ERR_SUCCESS) { ClearROI(); return checkError(err, "Could not set ROI height"); }

   int64_t offXInc = 1, offYInc = 1, offXMax = 0, offYMax = 0;
   getIntInc(hOffX, offXInc);
   getIntInc(hOffY, offYInc);
   getIntMax(hOffX, offXMax);
   getIntMax(hOffY, offYMax);

   int64_t ox = static_cast<int64_t>(x) - static_cast<int64_t>(x) % offXInc;
   int64_t oy = static_cast<int64_t>(y) - static_cast<int64_t>(y) % offYInc;
   ox = (std::min)(ox, offXMax);
   oy = (std::min)(oy, offYMax);

   err = setIntValue(hOffX, ox);
   if (err != SPINNAKER_ERR_SUCCESS) { ClearROI(); return checkError(err, "Could not set ROI offset X"); }
   err = setIntValue(hOffY, oy);
   if (err != SPINNAKER_ERR_SUCCESS) { ClearROI(); return checkError(err, "Could not set ROI offset Y"); }

   return DEVICE_OK;
}

int SpinnakerCCamera::GetROI(unsigned& x, unsigned& y, unsigned& xSize, unsigned& ySize)
{
   spinNodeHandle hOffX = nullptr, hOffY = nullptr, hW = nullptr, hH = nullptr;
   getNodeHandle(NODE_OFFSET_X, &hOffX);
   getNodeHandle(NODE_OFFSET_Y, &hOffY);
   getNodeHandle(NODE_WIDTH, &hW);
   getNodeHandle(NODE_HEIGHT, &hH);

   int64_t val = 0;
   getIntValue(hOffX, val); x = static_cast<unsigned>(val);
   getIntValue(hOffY, val); y = static_cast<unsigned>(val);
   getIntValue(hW, val); xSize = static_cast<unsigned>(val);
   getIntValue(hH, val); ySize = static_cast<unsigned>(val);

   return DEVICE_OK;
}

int SpinnakerCCamera::ClearROI()
{
   spinNodeHandle hOffX = nullptr, hOffY = nullptr, hW = nullptr, hH = nullptr;
   if (getNodeHandle(NODE_OFFSET_X, &hOffX) != SPINNAKER_ERR_SUCCESS ||
       getNodeHandle(NODE_OFFSET_Y, &hOffY) != SPINNAKER_ERR_SUCCESS ||
       getNodeHandle(NODE_WIDTH, &hW) != SPINNAKER_ERR_SUCCESS ||
       getNodeHandle(NODE_HEIGHT, &hH) != SPINNAKER_ERR_SUCCESS)
   {
      SetErrorText(SPKRC_ERROR, "Could not access ROI nodes");
      return SPKRC_ERROR;
   }

   setIntValue(hOffX, 0);
   setIntValue(hOffY, 0);

   int64_t wMax = 0, hMax = 0;
   getIntMax(hW, wMax);
   getIntMax(hH, hMax);

   spinError err = setIntValue(hW, wMax);
   if (err != SPINNAKER_ERR_SUCCESS)
      return checkError(err, "Could not clear ROI width");
   err = setIntValue(hH, hMax);
   if (err != SPINNAKER_ERR_SUCCESS)
      return checkError(err, "Could not clear ROI height");

   return DEVICE_OK;
}

int SpinnakerCCamera::GetBinning() const
{
   char buf[MM::MaxStrLength];
   int ret = GetProperty(MM::g_Keyword_Binning, buf);
   if (ret != DEVICE_OK)
      return 0;

   std::stringstream ss;
   int out = 0;
   ss << buf;
   ss >> out;
   return out;
}

int SpinnakerCCamera::SetBinning(int /*binSize*/)
{
   return SetProperty(MM::g_Keyword_Binning, "No Binning");
}


// --- Pixel manipulation ---

void SpinnakerCCamera::Unpack12Bit(uint16_t* unpacked, const uint8_t* packed,
   size_t width, size_t height, bool flip)
{
   unsigned int u_idx;
   int p_idx;
   for (u_idx = 0, p_idx = 0; u_idx < width * height; u_idx++)
   {
      if (u_idx % 2 == 0)
      {
         auto pt = reinterpret_cast<const Unpack12Struct*>(packed + p_idx);
         if (!flip)
            unpacked[u_idx] = ((static_cast<unsigned short>(pt->_1) & 0x0F) << 8) | static_cast<unsigned short>(pt->_2);
         else
            unpacked[u_idx] = (static_cast<unsigned short>(pt->_1) & 0x0F) | (static_cast<unsigned short>(pt->_2) << 4);
      }
      else
      {
         auto pt = reinterpret_cast<const Unpack12Struct*>(packed + p_idx);
         unpacked[u_idx] = (static_cast<unsigned short>(pt->_0) << 4) | (static_cast<unsigned short>(pt->_1) >> 4);
         p_idx += 3;
      }
   }
}

void SpinnakerCCamera::RGBtoBGRA(uint8_t* data, size_t imageBuffLength)
{
   long dest = 0;
   for (long i = 0; i < static_cast<long>(imageBuffLength); i++)
   {
      if (i % 3 == 0)
      {
         m_imageBuff[dest + 2] = data[i];
      }
      else if (i % 3 == 2)
      {
         m_imageBuff[dest - 2] = data[i];
         dest++;
         m_imageBuff[dest] = 0;
      }
      else
      {
         m_imageBuff[dest] = data[i];
      }
      dest++;
   }
}

int SpinnakerCCamera::allocateImageBuffer(std::size_t size, int64_t pixelFormatEnumValue)
{
   delete[] m_imageBuff;
   m_imageBuff = nullptr;

   if (pixelFormatEnumValue == PixelFormat_Mono12Packed || pixelFormatEnumValue == PixelFormat_Mono12p)
      m_imageBuff = reinterpret_cast<unsigned char*>(new uint16_t[(size + 1) / 2]);
   else
      m_imageBuff = new unsigned char[size];

   if (m_imageBuff == nullptr)
   {
      SetErrorText(SPKRC_ERROR, "Could not allocate sufficient memory for image");
      return SPKRC_ERROR;
   }

   return DEVICE_OK;
}


// --- Property callbacks (enum) ---

int SpinnakerCCamera::OnPixelFormat(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   return OnEnumPropertyChanged(NODE_PIXEL_FORMAT, pProp, eAct);
}

int SpinnakerCCamera::OnTestPattern(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   return OnEnumPropertyChanged("TestPattern", pProp, eAct);
}

int SpinnakerCCamera::OnExposureAuto(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   return OnEnumPropertyChanged(NODE_EXPOSURE_AUTO, pProp, eAct);
}

int SpinnakerCCamera::OnGainAuto(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   return OnEnumPropertyChanged("GainAuto", pProp, eAct);
}

int SpinnakerCCamera::OnBlackLevelAuto(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   return OnEnumPropertyChanged("BlackLevelAuto", pProp, eAct);
}

int SpinnakerCCamera::OnADCBitDepth(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   return OnEnumPropertyChanged("AdcBitDepth", pProp, eAct);
}

int SpinnakerCCamera::OnTriggerSelector(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   return OnEnumPropertyChanged("TriggerSelector", pProp, eAct);
}

int SpinnakerCCamera::OnTriggerMode(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   return OnEnumPropertyChanged(NODE_TRIGGER_MODE, pProp, eAct);
}

int SpinnakerCCamera::OnTriggerSource(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   return OnEnumPropertyChanged(NODE_TRIGGER_SOURCE, pProp, eAct);
}

int SpinnakerCCamera::OnTriggerActivation(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   return OnEnumPropertyChanged("TriggerActivation", pProp, eAct);
}

int SpinnakerCCamera::OnTriggerOverlap(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   return OnEnumPropertyChanged("TriggerOverlap", pProp, eAct);
}

int SpinnakerCCamera::OnExposureMode(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   return OnEnumPropertyChanged(NODE_EXPOSURE_MODE, pProp, eAct);
}

int SpinnakerCCamera::OnUserOutputSelector(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   return OnEnumPropertyChanged("UserOutputSelector", pProp, eAct);
}

int SpinnakerCCamera::OnLineSelector(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   return OnEnumPropertyChanged("LineSelector", pProp, eAct);
}

int SpinnakerCCamera::OnLineMode(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   return OnEnumPropertyChanged("LineMode", pProp, eAct);
}

int SpinnakerCCamera::OnLineSource(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   return OnEnumPropertyChanged("LineSource", pProp, eAct);
}


// --- Property callbacks (float) ---

int SpinnakerCCamera::OnFrameRate(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   return OnFloatPropertyChanged("AcquisitionFrameRate", pProp, eAct);
}

int SpinnakerCCamera::OnGain(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   return OnFloatPropertyChanged("Gain", pProp, eAct);
}

int SpinnakerCCamera::OnGamma(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   return OnFloatPropertyChanged("Gamma", pProp, eAct);
}

int SpinnakerCCamera::OnBlackLevel(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   return OnFloatPropertyChanged("BlackLevel", pProp, eAct);
}

int SpinnakerCCamera::OnTemperature(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   return OnFloatPropertyChanged("DeviceTemperature", pProp, eAct);
}

int SpinnakerCCamera::OnTriggerDelay(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   return OnFloatPropertyChanged("TriggerDelay", pProp, eAct);
}


// --- Property callbacks (bool) ---

int SpinnakerCCamera::OnGammaEnabled(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   return OnBoolPropertyChanged("GammaEnable", pProp, eAct);
}

int SpinnakerCCamera::OnReverseX(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   return OnBoolPropertyChanged("ReverseX", pProp, eAct);
}

int SpinnakerCCamera::OnReverseY(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   return OnBoolPropertyChanged("ReverseY", pProp, eAct);
}

int SpinnakerCCamera::OnUserOutputValue(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   return OnBoolPropertyChanged("UserOutputValue", pProp, eAct);
}

int SpinnakerCCamera::OnLineInverter(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   return OnBoolPropertyChanged("LineInverter", pProp, eAct);
}


// --- Property callbacks (special) ---

int SpinnakerCCamera::OnFrameRateEnabled(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   spinNodeHandle hAFRCE = nullptr;
   if (getNodeHandle("AcquisitionFrameRateEnable", &hAFRCE) != SPINNAKER_ERR_SUCCESS)
      return DEVICE_OK;

   if (eAct == MM::BeforeGet)
   {
      if (isNodeReadable(hAFRCE))
      {
         bool8_t val = False;
         getBoolValue(hAFRCE, val);
         pProp->Set(val ? "1" : "0");
      }
      else
      {
         pProp->Set("0");
      }
   }
   else if (eAct == MM::AfterSet)
   {
      if (isNodeWritable(hAFRCE))
      {
         long value;
         pProp->Get(value);
         spinError err = setBoolValue(hAFRCE, value != 0 ? True : False);
         if (err != SPINNAKER_ERR_SUCCESS)
            return checkError(err, "Could not set acquisition frame rate control enabled");
      }
      else
      {
         SetErrorText(SPKRC_ERROR, "Could not set frame rate control enabled");
         return SPKRC_ERROR;
      }
   }

   return DEVICE_OK;
}

int SpinnakerCCamera::OnFrameRateAuto(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   spinNodeHandle hAFRA = nullptr;
   if (getNodeHandle("AcquisitionFrameRateAuto", &hAFRA) != SPINNAKER_ERR_SUCCESS)
      return DEVICE_OK;

   if (eAct == MM::BeforeGet)
   {
      if (isNodeReadable(hAFRA))
      {
         std::string symbolic;
         getEnumSymbolic(hAFRA, symbolic);
         pProp->Set(symbolic.c_str());
      }
      else
      {
         SetErrorText(SPKRC_ERROR, "Could not read auto frame rate");
         return SPKRC_ERROR;
      }
   }
   else if (eAct == MM::AfterSet)
   {
      if (isNodeWritable(hAFRA))
      {
         std::string value;
         pProp->Get(value);
         spinError err = setEnumByName(hAFRA, value.c_str());
         if (err != SPINNAKER_ERR_SUCCESS)
            return checkError(err, "Could not set auto frame rate");
      }
      else
      {
         SetErrorText(SPKRC_ERROR, "Could not set auto frame rate");
         return SPKRC_ERROR;
      }
   }

   return DEVICE_OK;
}

int SpinnakerCCamera::OnVideoMode(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   spinNodeHandle hVM = nullptr;
   if (getNodeHandle("VideoMode", &hVM) != SPINNAKER_ERR_SUCCESS)
      return DEVICE_OK;

   if (eAct == MM::BeforeGet)
   {
      if (isNodeReadable(hVM))
      {
         std::string symbolic;
         getEnumSymbolic(hVM, symbolic);
         pProp->Set(symbolic.c_str());
      }
      else
      {
         SetErrorText(SPKRC_ERROR, "Could not read video mode!");
         return SPKRC_ERROR;
      }
   }
   else if (eAct == MM::AfterSet)
   {
      if (isNodeWritable(hVM))
      {
         std::string val;
         pProp->Get(val);
         spinError err = setEnumByName(hVM, val.c_str());
         if (err != SPINNAKER_ERR_SUCCESS)
            return checkError(err, "Could not set video mode");
      }
      else
      {
         SetErrorText(SPKRC_ERROR, "Could not write video mode");
         return SPKRC_ERROR;
      }
   }
   return DEVICE_OK;
}

int SpinnakerCCamera::OnBinningInt(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   spinNodeHandle hBH = nullptr, hBV = nullptr;
   getNodeHandle(NODE_BINNING_HORIZONTAL, &hBH);
   getNodeHandle(NODE_BINNING_VERTICAL, &hBV);

   if (eAct == MM::BeforeGet)
   {
      if (hBH != nullptr && isNodeReadable(hBH) && hBV != nullptr && isNodeReadable(hBV))
      {
         int64_t val = 0;
         getIntValue(hBH, val);
         pProp->Set(std::to_string(val).c_str());
      }
      else
      {
         SetErrorText(SPKRC_ERROR, "Could not read horizontal binning");
         return SPKRC_ERROR;
      }
   }
   else if (eAct == MM::AfterSet)
   {
      std::string val;
      pProp->Get(val);

      int64_t binVal = std::stoll(val);
      spinError err = setIntValue(hBH, binVal);
      if (err != SPINNAKER_ERR_SUCCESS)
         return checkError(err, "Could not set horizontal binning");
      err = setIntValue(hBV, binVal);
      if (err != SPINNAKER_ERR_SUCCESS)
         return checkError(err, "Could not set vertical binning");

      spinNodeHandle hW = nullptr, hH = nullptr;
      getNodeHandle(NODE_WIDTH, &hW);
      getNodeHandle(NODE_HEIGHT, &hH);

      int64_t wMax = 0, hMax = 0;
      // Use WidthMax/HeightMax like original
      spinNodeHandle hWMax = nullptr, hHMax = nullptr;
      if (getNodeHandle("WidthMax", &hWMax) == SPINNAKER_ERR_SUCCESS)
         getIntValue(hWMax, wMax);
      else
         getIntMax(hW, wMax);

      if (getNodeHandle("HeightMax", &hHMax) == SPINNAKER_ERR_SUCCESS)
         getIntValue(hHMax, hMax);
      else
         getIntMax(hH, hMax);

      setIntValue(hW, wMax);
      setIntValue(hH, hMax);
   }

   return DEVICE_OK;
}

int SpinnakerCCamera::OnBinningModeEnum(MM::PropertyBase* pProp, MM::ActionType eAct)
{
   spinNodeHandle hBC = nullptr;
   if (getNodeHandle("BinningControl", &hBC) != SPINNAKER_ERR_SUCCESS)
      return DEVICE_OK;

   if (eAct == MM::BeforeGet)
   {
      if (isNodeReadable(hBC))
      {
         std::string symbolic;
         getEnumSymbolic(hBC, symbolic);
         pProp->Set(symbolic.c_str());
      }
      else
      {
         pProp->Set("");
      }
   }
   else if (eAct == MM::AfterSet)
   {
      std::string val;
      pProp->Get(val);
      if (isNodeWritable(hBC))
      {
         setEnumByName(hBC, val.c_str());
      }
      else
      {
         SetErrorText(SPKRC_ERROR, "Could not write binning mode!");
         return SPKRC_ERROR;
      }
   }
   return DEVICE_OK;
}


// --- Sequence acquisition ---

int SpinnakerCCamera::StartSequenceAcquisition(double interval)
{
   if (!m_aqThread->IsStopped())
      return DEVICE_CAMERA_BUSY_ACQUIRING;

   int ret = GetCoreCallback()->PrepareForAcq(this);
   if (ret != DEVICE_OK)
      return ret;

   m_stopOnOverflow = false;
   m_aqThread->Start(-1, interval);
   return DEVICE_OK;
}

int SpinnakerCCamera::StartSequenceAcquisition(long numImages, double interval_ms, bool stopOnOverflow)
{
   if (!m_aqThread->IsStopped())
      return DEVICE_CAMERA_BUSY_ACQUIRING;

   int ret = GetCoreCallback()->PrepareForAcq(this);
   if (ret != DEVICE_OK)
      return ret;

   m_stopOnOverflow = stopOnOverflow;
   m_aqThread->Start(numImages, interval_ms);
   return DEVICE_OK;
}

int SpinnakerCCamera::StopSequenceAcquisition()
{
   if (!m_aqThread->IsStopped())
   {
      m_aqThread->Stop();
      m_aqThread->wait();
   }
   return DEVICE_OK;
}

bool SpinnakerCCamera::IsCapturing()
{
   return !m_aqThread->IsStopped();
}

int SpinnakerCCamera::MoveImageToCircularBuffer()
{
   if (!IsCapturing())
   {
      SetErrorText(SPKRC_ERROR, "Camera is not capturing! Cannot retrieve image!");
      return SPKRC_ERROR;
   }

   // Software trigger if needed
   spinNodeHandle hTrigMode = nullptr;
   spinNodeHandle hTrigSource = nullptr;
   if (getNodeHandle(NODE_TRIGGER_MODE, &hTrigMode) == SPINNAKER_ERR_SUCCESS &&
       getNodeHandle(NODE_TRIGGER_SOURCE, &hTrigSource) == SPINNAKER_ERR_SUCCESS)
   {
      std::string mode, source;
      getEnumSymbolic(hTrigMode, mode);
      getEnumSymbolic(hTrigSource, source);
      if (mode == "On" && source == "Software")
      {
         spinNodeHandle hTrigSw = nullptr;
         if (getNodeHandle(NODE_TRIGGER_SOFTWARE, &hTrigSw) == SPINNAKER_ERR_SUCCESS)
            executeCommand(hTrigSw);
      }
   }

   spinImage ip = nullptr;
   spinError err = spinCameraGetNextImageEx(m_cam, static_cast<uint64_t>(GetExposure()) + 1000, &ip);
   if (err != SPINNAKER_ERR_SUCCESS)
   {
      LogMessage("Failed to get next image in sequence");
      return DEVICE_OK;
   }

   bool8_t incomplete = False;
   spinImageIsIncomplete(ip, &incomplete);

   if (!incomplete)
   {
      MM::MMTime timeStamp = GetCurrentMMTime();
      char label[MM::MaxStrLength];
      GetLabel(label);

      MM::CameraImageMetadata md;
      md.AddTag(MM::g_Keyword_Metadata_CameraLabel, label);
      md.AddTag(MM::g_Keyword_Elapsed_Time_ms,
         CDeviceUtils::ConvertToString((timeStamp - m_aqThread->GetStartTime()).getMsec()));
      md.AddTag(MM::g_Keyword_Metadata_ROI_X,
         CDeviceUtils::ConvertToString(static_cast<long>(GetImageWidth())));
      md.AddTag(MM::g_Keyword_Metadata_ROI_Y,
         CDeviceUtils::ConvertToString(static_cast<long>(GetImageHeight())));

      char buf[MM::MaxStrLength];
      GetProperty(MM::g_Keyword_Binning, buf);
      md.AddTag(MM::g_Keyword_Binning, buf);

      MMThreadGuard g(m_pixelLock);

      void* pRawData = nullptr;
      spinImageGetData(ip, &pRawData);
      uint8_t* imageData = static_cast<uint8_t*>(pRawData);

      spinPixelFormatEnums pixFmt = PixelFormat_Mono8;
      spinImageGetPixelFormat(ip, &pixFmt);
      int64_t pixFmtVal = static_cast<int64_t>(pixFmt);

      if (pixFmtVal == PixelFormat_Mono12p ||
         pixFmtVal == PixelFormat_Mono12Packed ||
         pixFmtVal == PixelFormat_RGB8 ||
         pixFmtVal == PixelFormat_RGB8Packed)
      {
         if (m_imageBuff == nullptr)
         {
            int ret = allocateImageBuffer(GetImageBufferSize(), pixFmtVal);
            if (ret != DEVICE_OK)
            {
               spinImageRelease(ip);
               return ret;
            }
         }

         if (pixFmtVal == PixelFormat_RGB8 || pixFmtVal == PixelFormat_RGB8Packed)
         {
            size_t bufSize = 0;
            spinImageGetBufferSize(ip, &bufSize);
            size_t theirSizeD3 = bufSize / 3;
            size_t ourSizeD4 = static_cast<size_t>(GetImageBufferSize()) / 4;
            size_t minSize = theirSizeD3 > ourSizeD4 ? ourSizeD4 : theirSizeD3;
            size_t size = minSize * 3;
            RGBtoBGRA(imageData, size);
         }

         if (pixFmtVal == PixelFormat_Mono12p)
         {
            size_t w = 0, h = 0;
            spinImageGetWidth(ip, &w);
            spinImageGetHeight(ip, &h);
            Unpack12Bit(reinterpret_cast<uint16_t*>(m_imageBuff), imageData, w, h, false);
         }
         else if (pixFmtVal == PixelFormat_Mono12Packed)
         {
            size_t w = 0, h = 0;
            spinImageGetWidth(ip, &w);
            spinImageGetHeight(ip, &h);
            Unpack12Bit(reinterpret_cast<uint16_t*>(m_imageBuff), imageData, w, h, true);
         }

         imageData = m_imageBuff;
      }

      unsigned int w = GetImageWidth();
      unsigned int h = GetImageHeight();
      unsigned int b = GetImageBytesPerPixel();

      int ret = GetCoreCallback()->InsertImage(this, imageData, w, h, b, md.Serialize());
      spinImageRelease(ip);
      return ret;
   }
   else
   {
      LogMessage("Image incomplete in sequence acquisition");
   }

   spinImageRelease(ip);
   return DEVICE_OK;
}


// --- Acquisition Thread ---

SpinnakerCAcquisitionThread::SpinnakerCAcquisitionThread(SpinnakerCCamera* pCam)
   : m_numImages(-1),
   m_intervalMs(0),
   m_imageCounter(0),
   m_stop(true),
   m_suspend(false),
   m_spkrCam(pCam),
   m_startTime(0),
   m_actualDuration(0),
   m_lastFrameTime(0)
{
}

SpinnakerCAcquisitionThread::~SpinnakerCAcquisitionThread()
{
}

void SpinnakerCAcquisitionThread::Stop()
{
   MMThreadGuard g(m_stopLock);
   m_stop = true;
}

void SpinnakerCAcquisitionThread::Start(long numImages, double intervalMs)
{
   MMThreadGuard g1(m_stopLock);
   MMThreadGuard g2(m_suspendLock);
   m_numImages = numImages;
   m_intervalMs = intervalMs;
   m_imageCounter = 0;
   m_stop = false;
   m_suspend = false;
   activate();
   m_actualDuration = MM::MMTime{};
   m_startTime = m_spkrCam->GetCurrentMMTime();
   m_lastFrameTime = MM::MMTime{};

   int64_t pixFmt = m_spkrCam->getPixelFormatEnumValue();
   m_spkrCam->allocateImageBuffer(m_spkrCam->GetImageBufferSize(), pixFmt);

   spinNodeHandle hAcqMode = nullptr;
   m_spkrCam->getNodeHandle(NODE_ACQUISITION_MODE, &hAcqMode);

   if (numImages == -1)
   {
      m_spkrCam->setEnumByName(hAcqMode, "Continuous");
   }
   else
   {
      m_spkrCam->setEnumByName(hAcqMode, "MultiFrame");
      spinNodeHandle hAcqCount = nullptr;
      m_spkrCam->getNodeHandle(NODE_ACQUISITION_FRAME_COUNT, &hAcqCount);
      m_spkrCam->setIntValue(hAcqCount, numImages);
   }

   spinCameraBeginAcquisition(m_spkrCam->m_cam);
}

bool SpinnakerCAcquisitionThread::IsStopped()
{
   MMThreadGuard g(m_stopLock);
   return m_stop;
}

void SpinnakerCAcquisitionThread::Suspend()
{
   MMThreadGuard g(m_suspendLock);
   m_suspend = true;
}

bool SpinnakerCAcquisitionThread::IsSuspended()
{
   MMThreadGuard g(m_suspendLock);
   return m_suspend;
}

void SpinnakerCAcquisitionThread::Resume()
{
   MMThreadGuard g(m_suspendLock);
   m_suspend = false;
}

int SpinnakerCAcquisitionThread::svc(void) throw()
{
   int ret = DEVICE_ERR;

   try
   {
      do
      {
         ret = m_spkrCam->MoveImageToCircularBuffer();
      } while (DEVICE_OK == ret && !IsStopped() && (m_imageCounter++ < m_numImages || m_numImages == -1));

      if (IsStopped())
         m_spkrCam->LogMessage("SeqAcquisition interrupted by the user\n");
   }
   catch (...)
   {
      m_spkrCam->LogMessage("Unknown error in acquisition");
   }

   m_stop = true;
   m_actualDuration = m_spkrCam->GetCurrentMMTime() - m_startTime;

   spinCameraEndAcquisition(m_spkrCam->m_cam);

   spinNodeHandle hAcqMode = nullptr;
   m_spkrCam->getNodeHandle(NODE_ACQUISITION_MODE, &hAcqMode);
   if (hAcqMode != nullptr)
      m_spkrCam->setEnumByName(hAcqMode, "SingleFrame");

   auto* core = m_spkrCam->GetCoreCallback();
   if (core != nullptr)
      core->AcqFinished(m_spkrCam, 0);

   return DEVICE_OK;
}
