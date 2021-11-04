///////////////////////////////////////////////////////////////////////////////
// FILE:          MicroManager.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   pco camera module
//                
// AUTHOR:        Franz Reitner, Franz.Reitner@pco.de, 11/01/2010
// COPYRIGHT:     PCO AG, Kelheim, 2010-up to now ;-)
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

#include "..\..\MMDevice/ModuleInterface.h"
#define PCO_ERRT_H_CREATE_OBJECT
#include "PCO_err.h"
#include "PCO_errt.h"

#include "MicroManager.h"
#include "VersionNo.h"

#if defined _WIN64
#pragma comment(lib, ".\\lib\\pco_generic\\pco_kamlib64.lib")
#else
#pragma comment(lib, ".\\lib\\pco_generic\\pco_kamlib.lib")
#endif

#include <string>
#include <sstream>
#include <list>

using namespace std;

list<CPCOCam*> PCO_CamList;
const char* g_CameraDeviceName = "pco_camera";

const char* g_PixelType_8bit = "8bit";
const char* g_PixelType_16bit = "16bit";
const char* g_PixelType_RGB32bit = "RGB 32bit";

const char* g_TimeStamp_No = "No stamp";
const char* g_TimeStamp_B = "Binary";
const char* g_TimeStamp_BA = "Binary + ASCII";

int g_iCameraCount = 0;
int g_iSC2Count = 0;

/* DEVICE_INTERFACE_VERSION is defined in MMDevice.h under MMDevice folder
*/

///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////
MODULE_API void InitializeModuleData()
{
  //AddAvailableDeviceName(g_CameraDeviceName, "PCO generic camera adapter");
  RegisterDevice(g_CameraDeviceName, MM::CameraDevice, "pco.sdk camera adapter");
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
  PCO_CamList.remove((CPCOCam*) pDevice);
  delete pDevice;
}

MODULE_API MM::Device* CreateDevice(const char* pszDeviceName)
{
  if(pszDeviceName == 0)
    return 0;

  string strName(pszDeviceName);

  if(strName == g_CameraDeviceName)
  {
    CPCOCam *pcam = new CPCOCam();

    PCO_CamList.push_front(pcam);
    return pcam;
  }
  return 0;
}


///////////////////////////////////////////////////////////////////////////////
// CPCOCam constructor/destructor

CPCOCam::CPCOCam() :
CCameraBase<CPCOCam>(),
m_bSequenceRunning(false),
m_bInitialized(false),
m_bBusy(false),
m_dExposure(0.0),
sthd_(0),
m_bStopOnOverflow(false),
pixelDepth_(2),
pictime_(0.0)
{
  // initialize the array of camera settings
  m_iCameraNum = 0;
  m_iInterFace = 0;
  m_iCameraNumAtInterface = 0;
  m_nCameraType = 0;
  m_pszTimes[0] = 0;
  m_nSubMode = 0;
  m_nMode = 0;
  m_nTrig = 0;
  m_nRoiXMax = 0;
  m_nRoiXMin = 0;
  m_nRoiYMax = 0;
  m_nRoiYMin = 0;
  m_nHBin = 0;
  m_nVBin = 0;
  m_iGain = 0;
  m_iEMGain = 1;
  m_iGainCam = 0;
  m_iOffset = 0;
  m_uiFlags = 0;
  m_nCCDType = 0; // read-only
  m_bufnr = -1;
  m_pCamera = NULL;
  m_nTimesLen = MM_PCO_GENERIC_MAX_STRLEN;
  m_pCamera = new CCameraWrapper();

  m_bDemoMode = FALSE;
  m_bStartStopMode = TRUE;
  m_iFpsMode = 0;
  m_iNoiseFilterMode = 1;
  m_dFps = 10.0;
  m_iPixelRate = 0;
  m_iTimestamp = 0;
  m_iDoubleShutterMode = 0;
  m_iIRMode = 0;
  m_iNumImages = -1;
  dIntervall = 0.0;
  m_iAcquireMode = 0;

  m_bSettingsChanged = TRUE;
  m_bDoAutoBalance = TRUE;
  // DemoMode (pre-initialization property)
  CPropertyAction* pAct = new CPropertyAction(this, &CPCOCam::OnDemoMode);
  CreateProperty("DemoMode", "Off", MM::String, false, pAct, true);
  AddAllowedValue("DemoMode", "Off", 0);
  AddAllowedValue("DemoMode", "On", 1);


  InitializeDefaultErrorMessages();
  sthd_ = new SequenceThread(this);
  mxMutex = CreateMutex(NULL, FALSE, 0);
  m_bRecording = false;
  m_bCMOSLineTiming = false;
  m_wCMOSParameter = 0;
  m_wCMOSTimeBase = 0;
  m_dwCMOSLineTime = 0;
  m_dwCMOSExposureLines = 0;
  m_dwCMOSDelayLines = 0;
  m_bFlimMasterFrequencyMHz = true;
  m_bSoftwareTriggered = false;
  m_pic = NULL;
  m_dwCMOSLineTimeMin = 0;
  m_dwCMOSLineTimeMax = 0;
  m_dwCMOSFlags = 0;

  m_wFlimSourceSelect = 0;
  m_wFlimOutputWaveform = 0;
  m_wFlimPhaseNumber = 0;
  m_wFlimPhaseSymmetry = 0;
  m_wFlimPhaseOrder = 0;
  m_wFlimTapSelect = 0;
  m_wFlimAsymmetryCorrection = 0;
  m_wFlimCalculationMode = 0;
  m_wFlimReferencingMode = 0;
  m_wFlimThresholdLow = 0;
  m_wFlimThresholdHigh = 0;
  m_wFlimOutputMode = 0;
  m_dwFlimFrequency = 0;
  m_dwFlimPhaseMilliDeg = 0;

  m_iNextBuffer = -1;
  for(int i = 0; i < 4; i++)
  {
    m_iNextBufferToUse[i] = -1;
    m_iLastBufferUsed[i] = -1;
  }
  m_iNumImagesInserted = -1;
  m_bDotPhotonFilterAvailable = false;
  m_bDoDotPhoton = false;
  InitDotPhoton();
}

CPCOCam::~CPCOCam()
{
  if(m_bInitialized)
  {
    g_iCameraCount--;
    if(m_iInterFace == 3)
      g_iSC2Count--;
    Shutdown();

    if (m_bDotPhotonFilterAvailable)
    {
      UnInitDotPhoton();
    }
  }
  m_bInitialized = false;
  delete(m_pCamera);
  m_pCamera = NULL;

  if(g_iCameraCount <= 0)
  {
    InitLib(255, NULL, 0, NULL); // Removes allocated objects
    g_iCameraCount = 0;
    g_iSC2Count = 0;
  }
  
  delete(sthd_);
  CloseHandle(mxMutex);
}
void CPCOCam::GetName(char* name) const
{
  // Return the name used to referr to this device adapte
  CDeviceUtils::CopyLimitedString(name, g_CameraDeviceName);
}
// Camera type
int CPCOCam::OnCameraType(MM::PropertyBase* pProp, MM::ActionType eAct)
{
  if(eAct == MM::BeforeGet)
  {
    char szinterfaces[20][40] = {
        "Not specified",
        "FireWire",          // 1            // Firewire interface
        "CL Matrox",         // 2            // Cameralink Matrox Solios / Helios
        "CL Silicon Soft ME3",// 3           // Cameralink Silicon Software Me3
        "CL National Instr.",// 4            // Cameralink National Instruments
        "GigE",              // 5            // Gigabit Ethernet
        "USB",               // 6            // USB 2.0
        "CL Silicon Soft.Me4", // 7          // Cameralink Silicon Software Me4
        "USB3",              // 8            // USB 3.0
        "WLAN",              // 9            // WLan
        "CL Serial Int."    ,// 10           // Cameralink serial
        "CLHS",             // 11           // Cameralink HS Silicon Software Me5
        "Not specified 12",
        "Not specified 13",
        "Not specified 14",
        "Not specified 15",
        "Not specified 16",
        "Not specified 17",
        "Not specified 18",
        "Not specified 19"
    };
    char sztype[500];
    char szname[100];
    int ilen = 100, icamtype = 0, iccdtype = 0, icamid = 0;
    int iinterface = m_pCamera->m_strCamera.strAPIManager.wInterface;
    if (iinterface > 20)
      iinterface = 0;

    m_pCamera->GetCameraNameNType(szname, ilen, &icamtype, &iccdtype, &icamid);
    if(m_pCamera->m_iCamClass == 3)
      sprintf_s(sztype, 500, "%s - SN:%0X / Interface: %s", szname, icamid, szinterfaces[iinterface]);
    else
      sprintf_s(sztype, 500, "%s", szname);
    pProp->Set(sztype);
  }
  return DEVICE_OK;
}

int CPCOCam::OnLineTime(MM::PropertyBase* pProp, MM::ActionType eAct)
{
  if(eAct == MM::BeforeGet)
  {
    char szLineTime[50] = {"undefined"};
    double dlinetime;
    double dclock;
    WORD wsubtype = m_pCamera->m_strCamera.strGeneral.strCamType.wCamSubType;
    //int ipixelclockselect = m_pCamera->m_strCamera.strSensor.dwPixelRate;
    dclock = m_pCamera->m_strCamera.strSensor.dwPixelRate;//m_pCamera->m_strCamera.strSensor.strDescription.dwPixelRateDESC[ipixelclockselect];
    if(dclock > 0)
    {
      if((m_nCameraType == CAMERATYPE_PCO_EDGE) ||
        (m_nCameraType == CAMERATYPE_PCO_EDGE_GL) ||
        (wsubtype == CAMERASUBTYPE_PCO_EDGE_55) || 
        (wsubtype == CAMERASUBTYPE_PCO_EDGE_31))
      {
        dlinetime = 2624 / dclock;
      }
      else
      {
        if(m_nCameraType == CAMERATYPE_PCO_EDGE_HS)
        {
          if(dclock < 150000000)
            dlinetime = 2658 / dclock;
          else
            dlinetime = 2653 / dclock;
        }
        else
        {
          if(dclock < 150000000)
            dlinetime = 2647 / dclock;
          else
            dlinetime = 2658 / dclock;
        }
      }
      sprintf_s(szLineTime, "%4.2f", dlinetime * 1000000);
    }
    pProp->Set(szLineTime);
  }
  return DEVICE_OK;
}


// CCD type
int CPCOCam::OnCCDType(MM::PropertyBase* pProp, MM::ActionType eAct)
{
  if(eAct == MM::BeforeGet)
  {
    char sztype[500];
    if(m_pCamera->m_iCamClass == 3)
      sprintf_s(sztype, 500, "max X %d - max Y %d", roiXMaxFull_, roiYMaxFull_);
    else
      sprintf_s(sztype, 500, "pco CCD");
    pProp->Set(sztype);
  }
  return DEVICE_OK;
}

int CPCOCam::OnExposure(MM::PropertyBase* pProp, MM::ActionType eAct)
{
  int tb[3] = {1000000, 1000, 1};
  if(eAct == MM::BeforeGet)
  {
    if(m_pCamera->m_iCamClass == 3)
    {
      if(m_pCamera->m_strCamera.strTiming.wTimingControlMode == 0)// pco.camera
        m_dExposure = m_pCamera->m_strCamera.strTiming.dwExposureTable[0] / tb[m_pCamera->m_strCamera.strTiming.wTimeBaseExposure];
      else
        m_dExposure = m_pCamera->m_strCamera.strTiming.dwFrameRateExposure / tb[0];

      if(m_dExposure <= 0.0)
        m_dExposure = 1.0;
    }
    pProp->Set(m_dExposure);
  }
  else if(eAct == MM::AfterSet)
  {
    int nErr = 0;
    double dhelp;
    pProp->Get(dhelp);
    if(dhelp != m_dExposure)
    {
      m_dExposure = dhelp;

      if(m_pCamera->m_strCamera.strTiming.wTimingControlMode == 0)// pco.camera
      {
        if(m_dExposure < 10000.0)
        {
          if(m_dExposure < 0.1)
          {
            m_pCamera->m_strCamera.strTiming.wTimeBaseExposure = 0;
            m_pCamera->m_strCamera.strTiming.dwExposureTable[0] = (DWORD) (m_dExposure * 1000000.0);
          }
          else
          {
            m_pCamera->m_strCamera.strTiming.wTimeBaseExposure = 1;
            m_pCamera->m_strCamera.strTiming.dwExposureTable[0] = (DWORD) (m_dExposure * 1000.0);
          }
        }
        else
        {
          m_pCamera->m_strCamera.strTiming.wTimeBaseExposure = 2;
          m_pCamera->m_strCamera.strTiming.dwExposureTable[0] = (DWORD) m_dExposure;
        }
        m_pCamera->m_strCamera.strTiming.dwDelayTable[0] = 0;
      }
      else
      {
        m_pCamera->m_strCamera.strTiming.dwFrameRateExposure = (DWORD) (m_dExposure * tb[0]);
        m_pCamera->m_strCamera.strTiming.dwFrameRate = (DWORD) (m_dFps * 1000.0);
      }
      nErr = SetupCamera(false, false);
    }

    if(nErr != 0)
      return nErr;
  }
  return DEVICE_OK;
}


int CPCOCam::OnAcquireMode(MM::PropertyBase* pProp, MM::ActionType eAct)
{
  if(eAct == MM::BeforeGet)
  {
    if(m_pCamera->m_iCamClass == 3)
    {
      m_iAcquireMode = m_pCamera->m_strCamera.strRecording.wAcquMode;

      if(m_iAcquireMode == 0)
        pProp->Set("Internal");
      else
        pProp->Set("External");
    }
  }
  else if(eAct == MM::AfterSet)
  {
    int nErr = 0;
    long ihelp;
    string tmp;
    pProp->Get(tmp);
    ((MM::Property *) pProp)->GetData(tmp.c_str(), ihelp);
    //pProp->Get(ihelp);

    if(ihelp != m_iAcquireMode)
    {
      m_iAcquireMode = ihelp;
      if(m_pCamera->m_iCamClass == 3)
        m_pCamera->m_strCamera.strRecording.wAcquMode = (WORD) m_iAcquireMode;

      nErr = SetupCamera(true, false);
    }

    if(nErr != 0)
      return nErr;
  }
  return DEVICE_OK;
}


int CPCOCam::OnTriggerMode(MM::PropertyBase* pProp, MM::ActionType eAct)
{
  if(eAct == MM::BeforeGet)
  {
    if(m_pCamera->m_iCamClass == 3)
    {
      m_nTrig = m_pCamera->m_strCamera.strTiming.wTriggerMode;

      if(m_nTrig == 0)
        pProp->Set("Internal");
      else
      if(m_nTrig == 1)
        pProp->Set("Software");
      else
      if(m_nTrig == 2)
        pProp->Set("External");
      else
        pProp->Set("External Exp. Ctrl.");
    }
    if(m_pCamera->m_iCamClass == 2)
    {
      if(m_nTrig == 1)
        pProp->Set("Internal");
      else
        pProp->Set("External");
    }
  }
  else if(eAct == MM::AfterSet)
  {
    int nErr = 0;
    bool bwassoftwaretriggered = m_bSoftwareTriggered;
    long ihelp;
    string tmp;
    pProp->Get(tmp);
    ((MM::Property *) pProp)->GetData(tmp.c_str(), ihelp);
    //pProp->Get(ihelp);

    if(ihelp != m_nTrig)
    {
      m_nTrig = ihelp;
      if(m_pCamera->m_iCamClass == 3)
      {
        m_pCamera->m_strCamera.strTiming.wTriggerMode = (WORD) m_nTrig;
        if(m_nTrig == 1)
          m_bSoftwareTriggered = true;
        else
          m_bSoftwareTriggered = false;
      }

      nErr = SetupCamera(true, false);
      if(bwassoftwaretriggered)
      {
        m_pCamera->StopCam(&nErr);
        m_bRecording = false;
      }
    }

    if(nErr != 0)
      return nErr;
  }
  return DEVICE_OK;
}


int CPCOCam::OnTimestampMode(MM::PropertyBase* pProp, MM::ActionType eAct)
{
  if(eAct == MM::BeforeGet)
  {
    if(m_pCamera->m_iCamClass == 3)// pco.camera
      m_iTimestamp = m_pCamera->m_strCamera.strRecording.wTimeStampMode;
    if(m_iTimestamp == 0)
      pProp->Set(g_TimeStamp_No);
    else
    if(m_iTimestamp == 1)
      pProp->Set(g_TimeStamp_B);
    else
      pProp->Set(g_TimeStamp_BA);
  }
  else if(eAct == MM::AfterSet)
  {
    int nErr = 0;
    long ihelp;
    string tmp;
    pProp->Get(tmp);
    ((MM::Property *) pProp)->GetData(tmp.c_str(), ihelp);
    if(ihelp != m_iTimestamp)
    {
      m_iTimestamp = ihelp;
      if(m_pCamera->m_iCamClass == 3)// pco.camera
      {
        m_pCamera->m_strCamera.strRecording.wTimeStampMode = (WORD) m_iTimestamp;
      }
      nErr = SetupCamera(true, false);
    }

    if(nErr != 0)
      return nErr;
  }
  return DEVICE_OK;
}


int CPCOCam::OnFpsMode(MM::PropertyBase* pProp, MM::ActionType eAct)
{
  if(eAct == MM::BeforeGet)
  {
    if(m_iFpsMode == 0)
      pProp->Set("Off");
    else
      pProp->Set("On");
  }
  else if(eAct == MM::AfterSet)
  {
    int nErr = 0;
    string tmp;
    long fpsModeTmp;
    int ihelp;
    pProp->Get(tmp);
    ((MM::Property *) pProp)->GetData(tmp.c_str(), fpsModeTmp);
    ihelp = (fpsModeTmp == 1);

    if(ihelp != m_iFpsMode)
    {

      m_iFpsMode = ihelp;
      if(m_iFpsMode == 1)
      {
        sprintf_s(m_pszTimes, sizeof(m_pszTimes), "%ffps,%f,-1,-1", m_dFps, m_dExposure);
        if(m_pCamera->m_iCamClass == 3)// pco.camera
        {
          m_pCamera->m_strCamera.strTiming.wTimingControlMode = 1;
          m_wCMOSParameter = 0;
          m_pCamera->m_strCamera.strTiming.wCMOSParameter = m_wCMOSParameter;
        }
      }
      else
      {
        sprintf_s(m_pszTimes, sizeof(m_pszTimes), "0,%d,-1,-1", (int) m_dExposure);
        if(m_pCamera->m_iCamClass == 3)// pco.camera
        {
          m_pCamera->m_strCamera.strTiming.wTimingControlMode = 0;
        }
      }
      nErr = SetupCamera(true, false);
    }

    if(nErr != 0)
      return nErr;
  }
  return DEVICE_OK;
}

int CPCOCam::OnNoiseFilterMode(MM::PropertyBase* pProp, MM::ActionType eAct)
{
  if(eAct == MM::BeforeGet)
  {
    if(m_pCamera->m_iCamClass == 3)// pco.camera
      m_iNoiseFilterMode = m_pCamera->m_strCamera.strSensor.wNoiseFilterMode;
    if(m_iNoiseFilterMode == 0)
      pProp->Set("Off");
    else
      pProp->Set("On");
  }
  else if(eAct == MM::AfterSet)
  {
    int nErr = 0;
    string tmp;
    long noiseModeTmp;
    int ihelp;
    pProp->Get(tmp);
    ((MM::Property *) pProp)->GetData(tmp.c_str(), noiseModeTmp);
    ihelp = (noiseModeTmp == 1);

    if(ihelp != m_iNoiseFilterMode)
    {
      m_iNoiseFilterMode = ihelp;
      if(m_pCamera->m_iCamClass == 3)// pco.camera
        m_pCamera->m_strCamera.strSensor.wNoiseFilterMode = (WORD) m_iNoiseFilterMode;
      if(m_iNoiseFilterMode == 0)
        m_nMode &= 0xFFFFFF7F;
      else
        m_nMode |= 0x80;
      nErr = SetupCamera(true, false);
    }

    if(nErr != 0)
      return nErr;
  }
  return DEVICE_OK;
}

int CPCOCam::OnDoubleShutterMode(MM::PropertyBase* pProp, MM::ActionType eAct)
{
  if(eAct == MM::BeforeGet)
  {
    if(m_pCamera->m_iCamClass == 3)// pco.camera
    {
      m_iDoubleShutterMode = m_pCamera->m_strCamera.strSensor.wDoubleImage;
    }

    if(m_iDoubleShutterMode == 0)
      pProp->Set("Off");
    else
      pProp->Set("On");
  }
  else if(eAct == MM::AfterSet)
  {
    int nErr = 0;
    string tmp;
    long doubleshutterModeTmp;
    int ihelp;
    pProp->Get(tmp);
    ((MM::Property *) pProp)->GetData(tmp.c_str(), doubleshutterModeTmp);
    ihelp = (doubleshutterModeTmp == 1);

    if(ihelp != m_iDoubleShutterMode)
    {
      m_iDoubleShutterMode = ihelp;
      if(m_iDoubleShutterMode == 0)
        m_nMode &= 0xFFFF7FFF;
      else
        m_nMode |= 0x8000;
      if(m_pCamera->m_iCamClass == 3)// pco.camera
      {
        m_pCamera->m_strCamera.strSensor.wDoubleImage = (WORD) m_iDoubleShutterMode;
      }

      nErr = SetupCamera(true, true);
    }

    if(nErr != 0)
      return nErr;
  }
  return DEVICE_OK;
}

int CPCOCam::OnIRMode(MM::PropertyBase* pProp, MM::ActionType eAct)
{
  if(eAct == MM::BeforeGet)
  {
    if(m_pCamera->m_iCamClass == 3)// pco.camera
    {
      m_iIRMode = m_pCamera->m_strCamera.strSensor.wIR;
    }

    if(m_iIRMode == 0)
      pProp->Set("Off");
    else
      pProp->Set("On");
  }
  else if(eAct == MM::AfterSet)
  {
    int nErr = 0;
    string tmp;
    long irModeTmp;
    int ihelp;
    pProp->Get(tmp);
    ((MM::Property *) pProp)->GetData(tmp.c_str(), irModeTmp);
    ihelp = (irModeTmp == 1);

    if(ihelp != m_iIRMode)
    {
      m_iIRMode = ihelp;
      if(m_iIRMode == 0)
        m_nMode &= 0xFFFFFFBF;
      else
        m_nMode |= 0x40;
      if(m_pCamera->m_iCamClass == 3)// pco.camera
      {
        m_pCamera->m_strCamera.strSensor.wIR = (WORD) m_iIRMode;
      }

      nErr = SetupCamera(true, false);
    }

    if(nErr != 0)
      return nErr;
  }
  return DEVICE_OK;
}

int CPCOCam::OnFps(MM::PropertyBase* pProp, MM::ActionType eAct)
{
  if(eAct == MM::BeforeGet)
  {
    pProp->Set(m_dFps);
  }
  else if(eAct == MM::AfterSet)
  {
    int nErr = 0;
    double dhelp;
    pProp->Get(dhelp);
    if(dhelp != m_dFps)
    {
      m_dFps = dhelp;
      if(m_iFpsMode == 1)
      {
        sprintf_s(m_pszTimes, sizeof(m_pszTimes), "%ffps,%f,-1,-1", m_dFps, m_dExposure);
        if(m_pCamera->m_iCamClass == 3)// pco.camera
        {
          m_pCamera->m_strCamera.strTiming.dwFrameRateExposure = (DWORD) m_dExposure * 1000000;
          m_pCamera->m_strCamera.strTiming.dwFrameRate = (DWORD) (m_dFps * 1000.0);
        }
      }
      else
      {
        sprintf_s(m_pszTimes, sizeof(m_pszTimes), "0,%d,-1,-1", (int) m_dExposure);
        if(m_pCamera->m_iCamClass == 3)// pco.camera
        {
          m_pCamera->m_strCamera.strTiming.wTimeBaseExposure = 2;
          m_pCamera->m_strCamera.strTiming.dwExposureTable[0] = (DWORD) m_dExposure;
          m_pCamera->m_strCamera.strTiming.dwDelayTable[0] = 0;
        }
      }
      nErr = SetupCamera(true, false);
    }

    if(nErr != 0)
      return nErr;
  }
  return DEVICE_OK;
}

int CPCOCam::OnPixelRate(MM::PropertyBase* pProp, MM::ActionType eAct)
{
  if(eAct == MM::BeforeGet)
  {
    if(m_pCamera->m_iCamClass == 3)// pco.camera
    {
      m_iPixelRate = 1;
      if(m_pCamera->m_strCamera.strSensor.dwPixelRate == m_pCamera->m_strCamera.strSensor.strDescription.dwPixelRateDESC[0])
        m_iPixelRate = 0;
    }
    if(m_iPixelRate == 1)
      pProp->Set("fast scan");
    else
      pProp->Set("slow scan");
  }
  else if(eAct == MM::AfterSet)
  {
    int nErr = 0;
    string tmp;
    long fpsModeTmp;

    pProp->Get(tmp);
    ((MM::Property *) pProp)->GetData(tmp.c_str(), fpsModeTmp);

    if(fpsModeTmp != m_iPixelRate)
    {

      m_iPixelRate = fpsModeTmp;
      if(m_pCamera->m_iCamClass == 3)// pco.camera
      {
        m_pCamera->m_strCamera.strSensor.dwPixelRate = m_pCamera->m_strCamera.strSensor.strDescription.dwPixelRateDESC[m_iPixelRate];
        m_pCamera->m_strCamera.strTiming.dwDelayTable[0] = 0;

        if(m_iPixelRate == 1)
        {
          if(m_pCamera->m_strCamera.strTiming.wCMOSParameter == 1)
          {
            if((m_pCamera->m_strCamera.strTiming.dwCMOSFlags & 0x02) == false)
            {
              m_wCMOSParameter = 0;
              m_pCamera->m_strCamera.strTiming.wCMOSParameter = m_wCMOSParameter;
            }
          }
        }
      }

      nErr = SetupCamera(true, false);

      if(m_bCMOSLineTiming)
      {
        m_pCamera->GetCameraStruct((PCO_Camera*)&m_pCamera->m_strCamera.wSize);
        m_wCMOSParameter = m_pCamera->m_strCamera.strTiming.wCMOSParameter;
        m_wCMOSTimeBase = m_pCamera->m_strCamera.strTiming.wCMOSTimeBase;
        m_dwCMOSLineTime = m_pCamera->m_strCamera.strTiming.dwCMOSLineTime;
        m_dwCMOSLineTimeMin = m_pCamera->m_strCamera.strTiming.dwCMOSLineTimeMin;
        m_dwCMOSLineTimeMax = m_pCamera->m_strCamera.strTiming.dwCMOSLineTimeMax;
        m_dwCMOSFlags = m_pCamera->m_strCamera.strTiming.dwCMOSFlags;

        m_dwCMOSExposureLines = m_pCamera->m_strCamera.strTiming.dwCMOSExposureLines;
        m_dwCMOSDelayLines = m_pCamera->m_strCamera.strTiming.dwCMOSDelayLines;
        if(CheckLineTime(&m_dwCMOSLineTime) != DEVICE_OK)
          nErr = SetupCamera(true, false);
      }
    }

    if(nErr != 0)
      return nErr;
  }
  return DEVICE_OK;
}

int CPCOCam::OnDemoMode(MM::PropertyBase* pProp, MM::ActionType eAct)
{
  if(eAct == MM::BeforeGet)
  {
    if(m_bDemoMode)
      pProp->Set("On");
    else
      pProp->Set("Off");
  }
  else if(eAct == MM::AfterSet)
  {
    string tmp;
    long demoModeTmp;
    pProp->Get(tmp);
    ((MM::Property *) pProp)->GetData(tmp.c_str(), demoModeTmp);
    m_bDemoMode = (demoModeTmp == 1);
  }
  return DEVICE_OK;
}

// Binning
int CPCOCam::OnBinning(MM::PropertyBase* pProp, MM::ActionType eAct)
{
  int nErr = 0;

  if(m_bDemoMode)
    return DEVICE_OK;
  nErr = 0;
  if(eAct == MM::AfterSet)
  {
    long bin;
    pProp->Get(bin);

    if(bin != m_nHBin)
    {
      m_nHBin = bin;
      m_nVBin = bin;
      m_nRoiXMin = 1;
      m_nRoiYMin = 1;
      m_nRoiXMax = roiXMaxFull_;
      m_nRoiYMax = roiYMaxFull_;

      m_nRoiXMax = roiXMaxFull_ / m_nHBin;
      m_nRoiYMax = roiYMaxFull_ / m_nVBin;

      m_pCamera->m_strCamera.strSensor.wRoiX0 = (WORD) m_nRoiXMin;
      m_pCamera->m_strCamera.strSensor.wRoiY0 = (WORD) m_nRoiYMin;
      m_pCamera->m_strCamera.strSensor.wRoiX1 = (WORD) m_nRoiXMax;
      m_pCamera->m_strCamera.strSensor.wRoiY1 = (WORD) m_nRoiYMax;
      m_pCamera->m_strCamera.strSensor.wBinHorz = (WORD) m_nHBin;
      m_pCamera->m_strCamera.strSensor.wBinVert = (WORD) m_nVBin;

      nErr = SetupCamera(true, true);
      if(nErr != 0)
      {
        return nErr;
      }
      return ResizeImageBuffer();
    }
    return 0;
  }
  else if(eAct == MM::BeforeGet)
  {
    if(m_bSettingsChanged)
    {
      //if(m_pCamera->m_iCamClass == 3)// pco.camera
      {
        m_pCamera->GetCameraStruct((PCO_Camera*) &m_pCamera->m_strCamera.wSize);
        m_nHBin = m_pCamera->m_strCamera.strSensor.wBinHorz;
      }
      /*else
      {
        nErr = m_pCamera->getsettings(&m_nMode, &m_nTrig, &m_nRoiXMin, &m_nRoiXMax, &m_nRoiYMin, &m_nRoiYMax,
          &m_nHBin, &m_nVBin, m_pszTimes, &m_iGain, &m_iOffset, &m_uiFlags);
      }*/
      m_bSettingsChanged = FALSE;
    }
    if(nErr != 0)
      return nErr;
    pProp->Set((long) m_nHBin);
  }
  return DEVICE_OK;
}



int CPCOCam::SetupCamera(bool bStopRecording, bool bSizeChanged)
{
  unsigned int uiMode;
  int nErr = 0;
  int istopresult;
  int iOffsPxr;
  bool bwasrecording = m_bRecording;

  if(m_bDemoMode)
  {
    nErr = ResizeImageBuffer();
    if(nErr != 0)
      return nErr;
    return DEVICE_OK;
  }
  if(bStopRecording || m_bSoftwareTriggered)
  {
    if(WaitForSingleObject(mxMutex, 800) == WAIT_TIMEOUT)// Try to lock acquisition thread
    {
      return DEVICE_CAMERA_BUSY_ACQUIRING;
    }
    if(m_bRecording)
      nErr = m_pCamera->StopCam(&istopresult);
    m_bRecording = false;
    if(nErr != 0)
    {
      ReleaseMutex(mxMutex);
      return nErr;
    }
  }

  iOffsPxr = m_iOffset;

  m_nTimesLen = MM_PCO_GENERIC_MAX_STRLEN;
  if((m_nCameraType == 0x1300) || (m_nCameraType == 0x1310))
    iOffsPxr = m_iPixelRate;

  DWORD dwresult = 0;
  m_pCamera->SetCameraStruct((PCO_Camera*) &m_pCamera->m_strCamera.wSize, &dwresult);

  char szsetname[100] = "";
  m_pCamera->WriteSettingsToRegistryMM(0, &szsetname[0], m_pCamera->m_strCamera);

  m_bSettingsChanged = TRUE;
  if(bStopRecording || m_bSoftwareTriggered)
  {
    if(bSizeChanged)
    {
      nErr = ResizeImageBuffer();
      if(nErr != 0)
      {
        ReleaseMutex(mxMutex);
        return nErr;
      }
    }
    uiMode = 0x10000 + 0x0010;//Avoid adding buffers, Preview, Single
    nErr = m_pCamera->PreStartCam(uiMode, 0, 0, 0);            // schaltet automatisch auf internen Trigger
    if(nErr != 0)
    {
      ReleaseMutex(mxMutex);
      return nErr;
    }
    if(bwasrecording)
    {
      for(int j = 0; j < 4; j++)
        m_iLastBufferUsed[j] = -1;
      m_iNextBuffer = 0;
      nErr = m_pCamera->StartCam();
      m_bRecording = true;
    }
    ReleaseMutex(mxMutex);
  }
  return nErr;

}

// Pixel type
int CPCOCam::OnPixelType(MM::PropertyBase* pProp, MM::ActionType eAct)
{
  if(eAct == MM::AfterSet)
  {
    string pixType;
    if(m_bSequenceRunning == true)
    {
      return DEVICE_CAMERA_BUSY_ACQUIRING;
    }
    pProp->Get(pixType);
    if(pixType.compare(g_PixelType_16bit) == 0)
      pixelDepth_ = 2;
    else if(pixType.compare(g_PixelType_8bit) == 0)
      pixelDepth_ = 1;
    else if(pixType.compare(g_PixelType_RGB32bit) == 0)
      pixelDepth_ = 4;
    else
    {
      return DEVICE_INTERNAL_INCONSISTENCY;
    }
    return ResizeImageBuffer();
  }
  else if(eAct == MM::BeforeGet)
  {
    if(pixelDepth_ == 1)
      pProp->Set(g_PixelType_8bit);
    else if(pixelDepth_ == 2)
      pProp->Set(g_PixelType_16bit);
    else if(pixelDepth_ == 4)
      pProp->Set(g_PixelType_RGB32bit);
    else
    {
      return DEVICE_INTERNAL_INCONSISTENCY;
    }
  }
  return DEVICE_OK;
}

int CPCOCam::OnGain(MM::PropertyBase* pProp, MM::ActionType eAct)
{
  if(m_pCamera->m_iCamClass == 3)
  {
    if(eAct == MM::BeforeGet)
    {
      char szval[20] = { 0 };
      snprintf(szval, 20, "%d", (int)m_pCamera->m_strCamera.strSensor.wConvFact);
      m_iGainCam = m_pCamera->m_strCamera.strSensor.wConvFact;

      pProp->Set(szval);
    }
    else if(eAct == MM::AfterSet)
    {
      int igains[4] = {0, 1, 2, 3};
      int nErr = 0;
      long ihelp;
      string tmp;
      pProp->Get(tmp);
      ((MM::Property *) pProp)->GetData(tmp.c_str(), ihelp);

      if(ihelp != m_iGainCam)
      {
        m_iGainCam = ihelp;

        m_pCamera->m_strCamera.strSensor.wConvFact = (WORD)m_iGainCam;

        nErr = SetupCamera(true, false);
        if (nErr != 0)
        {
          return nErr;
        }
      }

      if(nErr != 0)
        return nErr;
    }
  }
  return DEVICE_OK;
}

HMODULE hcrypt;
HMODULE hmodule;
int icryptnum = -1;
int(*PCO_SetAppNameHandle)(const unsigned char szname[], HMODULE hlib);
int(*PCO_RemoveAppName)(int inum);
void EnableConvert(bool benable)
{
  if(benable)
  {
    unsigned char szname[32] = {"mmgr_dal_PCO_Camera"};
    GetModuleHandleEx(0, "mmgr_dal_PCO_Camera", &hmodule);
    hcrypt = LoadLibrary("PCO_CryptDll");
    if(hcrypt != NULL)
    {
      PCO_SetAppNameHandle = (int(*)(const unsigned char szname[], HMODULE hlib))
        GetProcAddress(hcrypt, "PCO_SetAppNameHandle");
      PCO_RemoveAppName = (int(*)(int inum))
        GetProcAddress(hcrypt, "PCO_RemoveAppName");
      if((PCO_SetAppNameHandle != NULL) && (PCO_RemoveAppName != NULL))
      {
        icryptnum = PCO_SetAppNameHandle(szname, hmodule);
      }
    }
  }
  else
  {
    if((PCO_SetAppNameHandle != NULL) && (PCO_RemoveAppName != NULL))
    {
      PCO_RemoveAppName(icryptnum);
      FreeLibrary(hcrypt);
      PCO_RemoveAppName = NULL;
      PCO_SetAppNameHandle = NULL;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// Function name   : CPCOCam::Initialize
// Description     : Initialize the camera
// Return type     : bool 

int CPCOCam::Initialize()
{
  // setup PCO camera
  // ----------------

  // Name
  int nRet = CreateProperty("Name", "pco camera", MM::String, true);
  if(nRet != DEVICE_OK)
    return nRet;

  // Description
  nRet = CreateProperty("Description", "pco.sdk driver module", MM::String, true);
  if(nRet != DEVICE_OK)
    return nRet;

  char szVersion[200];
  sprintf_s(szVersion, 200, "pco.sdk device adapter - V%s", STRFILEVER);
  LogMessage(szVersion, false);

  CPropertyAction* pAct;
  // Initialize the camera
  int nErr = 0;
  unsigned int uiresult = 0;

  InitLib(MMIJ, NULL, 0, NULL);

  if(!m_bDemoMode)
  {
    nErr = m_pCamera->PreInitSC2(g_iSC2Count, g_iCameraCount, &uiresult);
    if(nErr != PCO_NOERROR)
    {
      return DEVICE_NOT_CONNECTED;
    }
    else
    {
      m_iCameraNum = g_iCameraCount;
      m_iCameraNumAtInterface = g_iSC2Count;
      m_iInterFace = 3;
      g_iSC2Count++;
      g_iCameraCount++;
    }
  }
  else
  {
    m_pCamera->SetDemoMode(TRUE, 1280, 1024, FALSE, FALSE, 12, 0);
    nErr = m_pCamera->PreInitSen(0, 0, 0);
    sprintf_s(m_pCamera->szCamName, 20, "SensiCam demo");

    if(nErr != 0)
      return DEVICE_ERR;
  }

  if(!m_bDemoMode)
  {
    try
    {
      LogMessage("pco_generic. Try to Init!");
      nErr = m_pCamera->InitCamera(FALSE);
    }
    catch(...)
    {
      LogMessage("pco_generic. Failed to Init! Try catch error...Mr.Pco! What have you done??");
      delete(m_pCamera);
      SetErrorText(DEVICE_LOCALLY_DEFINED_ERROR, "PCO_Camera's \"Convert dll\" missing. Please copy pco_conv.dll to MM directory!");
      return DEVICE_LOCALLY_DEFINED_ERROR;
    }
  }
  if(nErr != 0)
  {
    WriteLog("pco_generic. Error %x in Init!", nErr);
    return nErr;
  }

  // camera type (read-only)
  pAct = new CPropertyAction(this, &CPCOCam::OnCameraType);
  nRet = CreateProperty("CameraType", "", MM::String, true, pAct);
  if(nRet != DEVICE_OK)
    return nRet;
  UpdateProperty("CameraType");

  m_nTimesLen = MM_PCO_GENERIC_MAX_STRLEN;
  m_dExposure = 10;
  m_nCameraType = m_pCamera->GetCamType();
  m_nCCDType = m_pCamera->GetCCDType();


  if (m_bDotPhotonFilterAvailable)
  {
    int err = InitDotPhotonCamera(m_pCamera->GetCameraHandle());
    if (err != PCO_NOERROR)
    {
      m_bDotPhotonFilterAvailable = false;
      m_bDoDotPhoton = false;
    }
  }

  char tab[400];
  m_pCamera->getsettings(&m_nMode, &m_nTrig, &m_nRoiXMin, &m_nRoiXMax, &m_nRoiYMin, &m_nRoiYMax,
    &m_nHBin, &m_nVBin, tab, &m_iGain, &m_iOffset, &m_uiFlags);
  if(m_bDemoMode)
    m_nCameraType = 7;

  if(m_pCamera->m_iCamClass == 3)
  {
    int tb[3] = {1000000, 1000, 1};
    char szsetname[100] = "";
    int i = 0;

    m_pCamera->m_strCamera.wSize = sizeof(PCO_Camera);
    m_pCamera->LoadSettingsFromRegistryMM(i, &szsetname[0], 100, m_pCamera->m_strCamera);
    m_pCamera->GetCameraStruct((PCO_Camera*) &m_pCamera->m_strCamera.wSize);
    //m_nHBin = m_nVBin = 1;
    //m_nMode = 0;//M_LONG;
    m_nSubMode = 0;//NORMALLONG;
    //m_nTrig = 0;
    sprintf_s(m_pszTimes, sizeof(m_pszTimes), "0,%.0f,-1,-1", m_dExposure);
    m_iDoubleShutterMode = m_pCamera->m_strCamera.strSensor.wDoubleImage;
    m_iIRMode = m_pCamera->m_strCamera.strSensor.wIR;
    m_iFpsMode = m_pCamera->m_strCamera.strTiming.wTimingControlMode;
    if(m_iFpsMode == 1)
      m_dExposure = m_pCamera->m_strCamera.strTiming.dwFrameRateExposure / tb[0];
    else
      m_dExposure = m_pCamera->m_strCamera.strTiming.dwExposureTable[0] / tb[m_pCamera->m_strCamera.strTiming.wTimeBaseExposure];
    m_dFps = m_pCamera->m_strCamera.strTiming.dwFrameRate;
    m_dFps /= 1000.0;
    m_nVBin = m_pCamera->m_strCamera.strSensor.wBinVert;
    m_pCamera->m_strCamera.strSensor.wBinHorz = m_pCamera->m_strCamera.strSensor.wBinVert;
    m_nRoiXMin = m_pCamera->m_strCamera.strSensor.wRoiX0;
    m_nRoiYMin = m_pCamera->m_strCamera.strSensor.wRoiY0;
    m_nRoiXMax = m_pCamera->m_strCamera.strSensor.wRoiX1;
    m_nRoiYMax = m_pCamera->m_strCamera.strSensor.wRoiY1;
    if(m_nTrig == 1)                   // Switch trigger mode to auto in case of internal SW trigger
    {
      m_bSoftwareTriggered = true;
      //m_nTrig = 0;
      //m_pCamera->m_strCamera.strTiming.wTriggerMode = 0;
    }
    else
      m_bSoftwareTriggered = false;
    m_iTimestamp = m_pCamera->m_strCamera.strRecording.wTimeStampMode;

    if((m_pCamera->m_strCamera.strAPIManager.wAPIManagementFlags & APIMANAGEMENTFLAG_LINE_TIMING) == APIMANAGEMENTFLAG_LINE_TIMING)
    {
      m_wCMOSParameter = m_pCamera->m_strCamera.strTiming.wCMOSParameter;
      m_wCMOSTimeBase = m_pCamera->m_strCamera.strTiming.wCMOSTimeBase;
      m_dwCMOSLineTime = m_pCamera->m_strCamera.strTiming.dwCMOSLineTime;
      m_dwCMOSLineTimeMin = m_pCamera->m_strCamera.strTiming.dwCMOSLineTimeMin;
      m_dwCMOSLineTimeMax = m_pCamera->m_strCamera.strTiming.dwCMOSLineTimeMax;
      m_dwCMOSFlags = m_pCamera->m_strCamera.strTiming.dwCMOSFlags;

      m_dwCMOSExposureLines = m_pCamera->m_strCamera.strTiming.dwCMOSExposureLines;
      m_dwCMOSDelayLines = m_pCamera->m_strCamera.strTiming.dwCMOSDelayLines;
      m_bCMOSLineTiming = true;
    }
  }
  else
  {
    return DEVICE_NOT_CONNECTED;
  }


  int imode, isubmode, igain;

  m_pCamera->ReloadSize();
  imode = m_nMode;
  isubmode = m_nSubMode;
  igain = m_iGain;
  // CCD type (read-only)
  pAct = new CPropertyAction(this, &CPCOCam::OnCCDType);
  nRet = CreateProperty("CCDType", "", MM::String, true, pAct);
  if(nRet != DEVICE_OK)
    return nRet;
  UpdateProperty("CCDType");

  // Binning
  pAct = new CPropertyAction(this, &CPCOCam::OnBinning);
  nRet = CreateProperty(MM::g_Keyword_Binning, "1", MM::Integer, false, pAct);
  if(nRet != DEVICE_OK)
    return nRet;

  vector<string> binValues;
  binValues.push_back("1");

  if(m_pCamera->m_iCamClass == 3)
  {
    if((m_pCamera->m_strCamera.strSensor.strDescription.wMaxBinHorzDESC >= 2) &&
      (m_pCamera->m_strCamera.strSensor.strDescription.wMaxBinVertDESC >= 2))
      binValues.push_back("2");

    if((m_pCamera->m_strCamera.strSensor.strDescription.wMaxBinHorzDESC >= 4) &&
      (m_pCamera->m_strCamera.strSensor.strDescription.wMaxBinVertDESC >= 4))
      binValues.push_back("4");

    if((m_pCamera->m_strCamera.strSensor.strDescription.wMaxBinHorzDESC >= 8) &&
      (m_pCamera->m_strCamera.strSensor.strDescription.wMaxBinVertDESC >= 8))
      binValues.push_back("8");
  }

  nRet = SetAllowedValues(MM::g_Keyword_Binning, binValues);
  if(nRet != DEVICE_OK)
    return nRet;

  if(m_bDemoMode)
  {
    roiXMaxFull_ = 1280;
    roiYMaxFull_ = 1024;
  }
  else
  {
    roiXMaxFull_ = m_pCamera->m_strCamera.strSensor.strDescription.wMaxHorzResStdDESC;
    roiYMaxFull_ = m_pCamera->m_strCamera.strSensor.strDescription.wMaxVertResStdDESC;
  }

  // Pixel type
  pAct = new CPropertyAction(this, &CPCOCam::OnPixelType);
  nRet = CreateProperty(MM::g_Keyword_PixelType, g_PixelType_16bit, MM::String, false, pAct);
  if(nRet != DEVICE_OK)
    return nRet;

  vector<string> pixTypes;
  pixTypes.push_back(g_PixelType_16bit);
  pixTypes.push_back(g_PixelType_8bit);

  if(m_pCamera->GetCCDCol(0))
    pixTypes.push_back(g_PixelType_RGB32bit);

  nRet = SetAllowedValues(MM::g_Keyword_PixelType, pixTypes);
  if(nRet != DEVICE_OK)
    return nRet;

  // Exposure
  pAct = new CPropertyAction(this, &CPCOCam::OnExposure);
  nRet = CreateProperty(MM::g_Keyword_Exposure, "10", MM::Float, false, pAct);
  if(nRet != DEVICE_OK)
    return nRet;

  if(m_pCamera->m_iCamClass == 3)
  {
    pAct = new CPropertyAction(this, &CPCOCam::OnTriggerMode);
    nRet = CreateProperty("Triggermode", "Internal", MM::String, false, pAct);
    if(nRet != DEVICE_OK)
      return nRet;
    nRet = AddAllowedValue("Triggermode", "Internal", 0);
    if(nRet != DEVICE_OK)
      return nRet;
    nRet = AddAllowedValue("Triggermode", "Software", 1);
    if(nRet != DEVICE_OK)
      return nRet;
    nRet = AddAllowedValue("Triggermode", "External", 2);
    if(nRet != DEVICE_OK)
      return nRet;

    nRet = AddAllowedValue("Triggermode", "External Exp. Ctrl.", 3);
    if(nRet != DEVICE_OK)
      return nRet;

    pAct = new CPropertyAction(this, &CPCOCam::OnTimestampMode);
    nRet = CreateProperty("Timestampmode", g_TimeStamp_No, MM::String, false, pAct);
    if(nRet != DEVICE_OK)
      return nRet;
    nRet = AddAllowedValue("Timestampmode", g_TimeStamp_No, 0);
    if(nRet != DEVICE_OK)
      return nRet;
    nRet = AddAllowedValue("Timestampmode", g_TimeStamp_B, 1);
    if(nRet != DEVICE_OK)
      return nRet;
    nRet = AddAllowedValue("Timestampmode", g_TimeStamp_BA, 2);
    if(nRet != DEVICE_OK)
      return nRet;

    if(m_pCamera->m_strCamera.strSensor.strDescription.wDoubleImageDESC >= 1)
    {
      pAct = new CPropertyAction(this, &CPCOCam::OnDoubleShutterMode);
      nRet = CreateProperty("Double Shutter Mode", "Off", MM::String, false, pAct);
      if(nRet != DEVICE_OK)
        return nRet;
      nRet = AddAllowedValue("Double Shutter Mode", "Off", 0);
      if(nRet != DEVICE_OK)
        return nRet;
      nRet = AddAllowedValue("Double Shutter Mode", "On", 1);
      if(nRet != DEVICE_OK)
        return nRet;
    }
    if(m_pCamera->m_strCamera.strSensor.strDescription.wIRDESC >= 1)
    {
      pAct = new CPropertyAction(this, &CPCOCam::OnIRMode);
      nRet = CreateProperty("IR Mode", "Off", MM::String, false, pAct);
      if(nRet != DEVICE_OK)
        return nRet;
      nRet = AddAllowedValue("IR Mode", "Off", 0);
      if(nRet != DEVICE_OK)
        return nRet;
      nRet = AddAllowedValue("IR Mode", "On", 1);
      if(nRet != DEVICE_OK)
        return nRet;
    }
    if((m_pCamera->m_strCamera.strSensor.strDescription.dwGeneralCapsDESC1 & GENERALCAPS1_NO_ACQUIREMODE) == 0)
    {// Bit 9: Acquire mode not available
      pAct = new CPropertyAction(this, &CPCOCam::OnAcquireMode);
      nRet = CreateProperty("Acquiremode", "Internal", MM::String, false, pAct);
      if(nRet != DEVICE_OK)
        return nRet;
      nRet = AddAllowedValue("Acquiremode", "Internal", 0);
      if(nRet != DEVICE_OK)
        return nRet;
      nRet = AddAllowedValue("Acquiremode", "External", 1);
      if(nRet != DEVICE_OK)
        return nRet;
    }

    if((m_nCameraType == CAMERATYPE_PCO_EDGE) ||// fps setting for pco.edge
      (m_nCameraType == CAMERATYPE_PCO_EDGE_42) ||
      (m_nCameraType == CAMERATYPE_PCO_EDGE_GL))
    {
      pAct = new CPropertyAction(this, &CPCOCam::OnFps);
      nRet = CreateProperty("Fps", "1", MM::Float, false, pAct);
      if(nRet != DEVICE_OK)
        return nRet;
      nRet = SetPropertyLimits("Fps", 1.0, 3000.0);
      if(nRet != DEVICE_OK)
        return nRet;

      pAct = new CPropertyAction(this, &CPCOCam::OnFpsMode);
      nRet = CreateProperty("Fps Mode", "Off", MM::String, false, pAct);
      if(nRet != DEVICE_OK)
        return nRet;
      nRet = AddAllowedValue("Fps Mode", "Off", 0);
      if(nRet != DEVICE_OK)
        return nRet;
      nRet = AddAllowedValue("Fps Mode", "On", 1);
      if(nRet != DEVICE_OK)
        return nRet;

    }

    if((m_nCameraType == CAMERATYPE_PCO_EDGE) ||// fps setting for pco.edge
      (m_nCameraType == CAMERATYPE_PCO_EDGE_42) ||
      (m_nCameraType == CAMERATYPE_PCO_EDGE_GL) ||
      (m_nCameraType == CAMERATYPE_PCO_EDGE_USB3) ||
      (m_nCameraType == CAMERATYPE_PCO_EDGE_HS))
    {
      pAct = new CPropertyAction(this, &CPCOCam::OnLineTime);
      nRet = CreateProperty("Line Time [us]", "", MM::String, true, pAct);
      if(nRet != DEVICE_OK)
        return nRet;
      UpdateProperty("Line Time (us)");
    }


    if((m_pCamera->m_strCamera.strSensor.strDescription.dwGeneralCapsDESC1 & GENERALCAPS1_NOISE_FILTER) == GENERALCAPS1_NOISE_FILTER)
    {
      pAct = new CPropertyAction(this, &CPCOCam::OnNoiseFilterMode);
      nRet = CreateProperty("Noisefilter", "Off", MM::String, false, pAct);
      if(nRet != DEVICE_OK)
        return nRet;
      nRet = AddAllowedValue("Noisefilter", "Off", 0);
      if(nRet != DEVICE_OK)
        return nRet;
      nRet = AddAllowedValue("Noisefilter", "On", 1);
      if(nRet != DEVICE_OK)
        return nRet;
    }

    if(m_pCamera->m_strCamera.strSensor.strDescription.dwPixelRateDESC[1] != 0)
    {
      pAct = new CPropertyAction(this, &CPCOCam::OnPixelRate);
      nRet = CreateProperty("PixelRate", "slow scan", MM::String, false, pAct);
      if(nRet != DEVICE_OK)
        return nRet;
      nRet = AddAllowedValue("PixelRate", "slow scan", 0);
      if(nRet != DEVICE_OK)
        return nRet;
      nRet = AddAllowedValue("PixelRate", "fast scan", 1);
      if(nRet != DEVICE_OK)
        return nRet;
    }
    if (m_pCamera->m_strCamera.strSensor.strDescription.wConvFactDESC[1] != 0)
    {
      char szval[20] = { 0 };
      pAct = new CPropertyAction(this, &CPCOCam::OnGain);
      snprintf(szval, 20, "%d", (int)m_pCamera->m_strCamera.strSensor.strDescription.wConvFactDESC[0]);
      nRet = CreateProperty("Conversion factor*100", szval, MM::String, false, pAct);
      if (nRet != DEVICE_OK)
        return nRet;
      for (int i = 0; i < 4; i++)
      {
        if (m_pCamera->m_strCamera.strSensor.strDescription.wConvFactDESC[i] == 0)
          break;
        snprintf(szval, 20, "%d", (int)m_pCamera->m_strCamera.strSensor.strDescription.wConvFactDESC[i]);
        nRet = AddAllowedValue("Conversion factor*100", szval, m_pCamera->m_strCamera.strSensor.strDescription.wConvFactDESC[i]);
        if (nRet != DEVICE_OK)
          return nRet;
      }
    }
    if (m_pCamera->m_strCamera.strSensor.strDescription.dwGeneralCapsDESC1 & GENERALCAPS1_HW_IO_SIGNAL_DESCRIPTOR)
      InitHWIO();
    if(m_bCMOSLineTiming)
      InitLineTiming();
    if(m_pCamera->m_strCamera.strGeneral.strCamType.wCamType == CAMERATYPE_PCO_FLIM)
      InitFlim();

    InitFanLED();
  }

  m_nMode = imode;
  m_nSubMode = isubmode;
  m_iGain = igain;
  nErr = SetupCamera(true, true);
  if(nErr != DEVICE_OK)
    return nErr;
  m_bInitialized = true;

  // set additional properties as read-only for now
  return DEVICE_OK;
}


///////////////////////////////////////////////////////////////////////////////
// Function name   : CPCOCam::Shutdown
// Description     : Deactivate the camera, reverse the initialization process
// Return type     : bool 

int CPCOCam::Shutdown()
{
  int istopresult = 0;

  if(m_pCamera != NULL)
  {
    if(!m_bDemoMode)
      m_pCamera->StopCam(&istopresult);
    m_bRecording = false;

    m_pCamera->CloseCam();
  }
  EnableConvert(FALSE);
  return DEVICE_OK;
}

/**
* Returns the number of physical channels in the image.
*/
unsigned int CPCOCam::GetNumberOfComponents() const
{
  if(img_.Depth() == 1)
    return 1;
  else if(img_.Depth() == 2)
  {
    return 1;
  }
  else if(img_.Depth() == 4)
  {
    return 4;
  }
  else
  {
    return 0; // should not happen
  }
}

unsigned  CPCOCam::GetImageBytesPerPixel() const
{
  /*  if(img_.Depth() == 4)// Faulty setting inside MM
  return 1;
  else*/
  return img_.Depth();
}


unsigned CPCOCam::GetBitDepth() const
{
  if(img_.Depth() == 1)
    return 8;
  else if(img_.Depth() == 2)
  {
    return m_pCamera->GetBitsPerPixel();
  }
  else if(img_.Depth() == 4)
  {
    return 8;
  }
  else
  {
    return 0; // should not happen
  }
}

int CPCOCam::GetBinning() const
{
  return m_nHBin;
}

int CPCOCam::SetBinning(int binSize)
{
  ostringstream os;
  os << binSize;
  return SetProperty(MM::g_Keyword_Binning, os.str().c_str());
}

/**
* Returns pixel data with interleaved RGB pixels in 32 bpp format
*/
const unsigned int* CPCOCam::GetImageBufferAsRGB32()
{
  return (unsigned int*) GetBuffer(0);
}

const unsigned char* CPCOCam::GetImageBuffer()
{
  return (unsigned char*) GetBuffer(m_iLastBufferUsed[0]);
}

///////////////////////////////////////////////////////////////////////////////
// Function name   : char* CPCOCam::GetImageBuffer
// Description     : Returns the raw image buffer
// Return type     : const unsigned 

const unsigned char* CPCOCam::GetBuffer(int ibufnum)
{
  int nErr = 0, iw, ih;

  if(ibufnum < 0)
    return NULL;

  m_pic = m_pCamera->GetBuffer(ibufnum);

  if(img_.Depth() == 2)
  {


    return (const unsigned char*) m_pic;
    //memcpy((void*) const_cast<unsigned char*>(img_.GetPixels()), (const void*)m_pic, img_.Width() * img_.Height() * 2);
  }
  else if(img_.Depth() == 1)
  {
    unsigned char *pchar;
    unsigned char *ppic8;
    if(m_bDoAutoBalance)
    {
      m_pCamera->SetLutMinMax(TRUE, TRUE);
      m_bDoAutoBalance = FALSE;
    }
    m_pCamera->Convert(ibufnum);
    iw = img_.Width();
    ih = img_.Height();
    ppic8 = m_pCamera->GetPic8();
    int iadd = iw % 4;
    if(iadd != 0)
      iadd = 4 - iadd;

    pchar = const_cast<unsigned char*>(img_.GetPixels());
    for(int y = 0; y < ih; y++)
    {
      for(int x = 0; x < iw; x++)
      {
        *pchar++ = *ppic8++;
      }
      ppic8 += iadd;
    }
  }
  else if(img_.Depth() == 4)
  {
    if(m_bDoAutoBalance)
    {
      m_pCamera->SetLutMinMax(TRUE, TRUE);
      m_pCamera->AutoBalance(0, 0, 0, 0, 0);
      m_bDoAutoBalance = FALSE;
    }
    m_pCamera->Convert(ibufnum);
    iw = img_.Width();
    ih = img_.Height();
    unsigned char *pchar;
    unsigned char *ppic8, *ph;
    int iadd;

    ppic8 = m_pCamera->GetPic8c();

    iadd = (iw * 3) % 4;
    if(iadd != 0)
      iadd = 4 - iadd;
    pchar = const_cast<unsigned char*>(img_.GetPixelsRW());
    for(int y = 0; y < ih; y++)
    {
      ph = &ppic8[y*(iw * 3 + iadd)];
      for(int x = 0; x < iw; x++)
      {
        *pchar++ = (unsigned char) *ph++;
        *pchar++ = (unsigned char) *ph++;
        *pchar++ = (unsigned char) *ph++;
        *pchar++ = 0;
      }
    }
  }

  if(nErr != 0)
    return 0;

  return img_.GetPixels();
}

int CPCOCam::SetROI(unsigned uX, unsigned uY, unsigned uXSize, unsigned uYSize)
{
  int nErr = 0;

  if(m_pCamera->m_iCamClass == 2)
    return DEVICE_OK;
  if(m_bDemoMode)
    return DEVICE_OK;

  if(m_bSettingsChanged)
  {
    /*nErr = m_pCamera->getsettings(&m_nMode, &m_nTrig, &m_nRoiXMin, &m_nRoiXMax, &m_nRoiYMin, &m_nRoiYMax,
      &m_nHBin, &m_nVBin, m_pszTimes, &m_iGain, &m_iOffset, &m_uiFlags);
    if(m_pCamera->m_iCamClass == 3)// pco.camera*/
    {
      m_pCamera->GetCameraStruct((PCO_Camera*) &m_pCamera->m_strCamera.wSize);
    }
    m_bSettingsChanged = FALSE;
  }
  if(nErr != 0)
    return nErr;

  if(m_pCamera->m_iCamClass == 1)
  {
    m_nRoiXMin = (int) (uX * m_nHBin);
    m_nRoiYMin = (int) (uY * m_nVBin);
    m_nRoiXMax = (int) ((uX + uXSize) * m_nHBin);
    m_nRoiYMax = (int) ((uY + uYSize) * m_nVBin);
  }
  else
  {
    m_nRoiXMin = (int) ceil(((double) uX + 1));
    m_nRoiYMin = (int) ceil(((double) uY + 1));
    m_nRoiXMax = (int) ceil((((double) uX + uXSize)));
    m_nRoiYMax = (int) ceil((((double) uY + uYSize)));
  }

  if(m_nRoiXMin > m_nRoiXMax)
    m_nRoiXMin = m_nRoiXMax - 1;
  if(m_nRoiYMin > m_nRoiYMax)
    m_nRoiYMin = m_nRoiYMax - 1;

  if(m_nRoiXMin < 1)
    m_nRoiXMin = 1;
  if(m_nRoiYMin < 1)
    m_nRoiYMin = 1;

  m_pCamera->m_strCamera.strSensor.wRoiX0 = (WORD) m_nRoiXMin;
  m_pCamera->m_strCamera.strSensor.wRoiY0 = (WORD) m_nRoiYMin;
  m_pCamera->m_strCamera.strSensor.wRoiX1 = (WORD) m_nRoiXMax;
  m_pCamera->m_strCamera.strSensor.wRoiY1 = (WORD) m_nRoiYMax;

  if(m_pCamera->m_iCamClass == 3)
  {
    if(m_nRoiXMax > roiXMaxFull_ / m_nHBin)
      m_nRoiXMax = roiXMaxFull_ / m_nHBin;
    if(m_nRoiYMax > roiYMaxFull_ / m_nVBin)
      m_nRoiYMax = roiYMaxFull_ / m_nVBin;
    SetNCheckROI(&m_nRoiXMin, &m_nRoiXMax, &m_nRoiYMin, &m_nRoiYMax);
  }
  nErr = SetupCamera(true, true);
  if(nErr != 0)
    return nErr;
  nErr = ResizeImageBuffer();
  if(nErr != 0)
    return nErr;

  return DEVICE_OK;
}

int CPCOCam::SetNCheckROI(int *Roix0, int *Roix1, int *Roiy0, int *Roiy1)
{
  m_pCamera->m_strCamera.strSensor.wRoiX0 = (WORD) *Roix0;
  m_pCamera->m_strCamera.strSensor.wRoiY0 = (WORD) *Roiy0;
  m_pCamera->m_strCamera.strSensor.wRoiX1 = (WORD) *Roix1;
  m_pCamera->m_strCamera.strSensor.wRoiY1 = (WORD) *Roiy1;

  WORD wmaxhorz;
  wmaxhorz = (m_pCamera->m_strCamera.strSensor.wSensorformat == 1) ? m_pCamera->m_strCamera.strSensor.strDescription.wMaxHorzResExtDESC
    : m_pCamera->m_strCamera.strSensor.strDescription.wMaxHorzResStdDESC;
  WORD wRoiSteppinghorz = m_pCamera->m_strCamera.strSensor.strDescription.wRoiHorStepsDESC;
  if((m_pCamera->m_strCamera.strAPIManager.wAPIManagementFlags & APIMANAGEMENTFLAG_SOFTROI) == APIMANAGEMENTFLAG_SOFTROI)
    wRoiSteppinghorz = m_pCamera->m_strCamera.strSensor.strDescription.wSoftRoiHorStepsDESC;

  if(wRoiSteppinghorz > 1)
  {
    m_pCamera->m_strCamera.strSensor.wRoiX0 /= wRoiSteppinghorz;
    m_pCamera->m_strCamera.strSensor.wRoiX0 *= wRoiSteppinghorz;
    m_pCamera->m_strCamera.strSensor.wRoiX0 += 1;
  }

  if(wRoiSteppinghorz == 0)
  {
    m_pCamera->m_strCamera.strSensor.wRoiX0 = 1;
    m_pCamera->m_strCamera.strSensor.wRoiX1 = wmaxhorz / m_pCamera->m_strCamera.strSensor.wBinHorz;
  }

  if(m_pCamera->m_strCamera.strSensor.wRoiX0 < 1)
    m_pCamera->m_strCamera.strSensor.wRoiX0 = 1;

  if(wRoiSteppinghorz > 1)
  {
    m_pCamera->m_strCamera.strSensor.wRoiX1 /= wRoiSteppinghorz;
    m_pCamera->m_strCamera.strSensor.wRoiX1 *= wRoiSteppinghorz;
  }

  if(m_pCamera->m_strCamera.strSensor.wRoiX1 > wmaxhorz)
    m_pCamera->m_strCamera.strSensor.wRoiX1 = wmaxhorz;
  if(m_pCamera->m_strCamera.strSensor.wRoiX0 > m_pCamera->m_strCamera.strSensor.wRoiX1)
    m_pCamera->m_strCamera.strSensor.wRoiX0 = (WORD) (m_pCamera->m_strCamera.strSensor.wRoiX1 - 1);

  WORD wmaxvert;
  wmaxvert = (m_pCamera->m_strCamera.strSensor.wSensorformat == 1) ? m_pCamera->m_strCamera.strSensor.strDescription.wMaxVertResExtDESC
    : m_pCamera->m_strCamera.strSensor.strDescription.wMaxVertResStdDESC;

  WORD wRoiSteppingvert = m_pCamera->m_strCamera.strSensor.strDescription.wRoiVertStepsDESC;
  if((m_pCamera->m_strCamera.strAPIManager.wAPIManagementFlags & APIMANAGEMENTFLAG_SOFTROI) == APIMANAGEMENTFLAG_SOFTROI)
    wRoiSteppingvert = m_pCamera->m_strCamera.strSensor.strDescription.wSoftRoiVertStepsDESC;

  if(wRoiSteppingvert > 1)
  {
    m_pCamera->m_strCamera.strSensor.wRoiY0 /= wRoiSteppingvert;
    m_pCamera->m_strCamera.strSensor.wRoiY0 *= wRoiSteppingvert;
    m_pCamera->m_strCamera.strSensor.wRoiY0 += 1;
  }
  if(wRoiSteppingvert == 0)
  {
    m_pCamera->m_strCamera.strSensor.wRoiY0 = 1;
    m_pCamera->m_strCamera.strSensor.wRoiY1 = wmaxvert / m_pCamera->m_strCamera.strSensor.wBinVert;
  }

  if(m_pCamera->m_strCamera.strSensor.wRoiY0 < 1)
    m_pCamera->m_strCamera.strSensor.wRoiY0 = 1;

  if(wRoiSteppingvert > 1)
  {
    m_pCamera->m_strCamera.strSensor.wRoiY1 /= wRoiSteppingvert;
    m_pCamera->m_strCamera.strSensor.wRoiY1 *= wRoiSteppingvert;
  }

  if(m_pCamera->m_strCamera.strSensor.wRoiY1 > wmaxvert)
    m_pCamera->m_strCamera.strSensor.wRoiY1 = wmaxvert;

  if(m_pCamera->m_strCamera.strSensor.wRoiY0 > m_pCamera->m_strCamera.strSensor.wRoiY1)
    m_pCamera->m_strCamera.strSensor.wRoiY0 = (WORD) (m_pCamera->m_strCamera.strSensor.wRoiY1 - 1);

  bool bSymmetricalROIHorz = FALSE;
  bool bSymmetricalROIVert = FALSE;

  if(m_pCamera->m_strCamera.strSensor.strDescription.wNumADCsDESC > 1)
    bSymmetricalROIHorz = (m_pCamera->m_strCamera.strSensor.wADCOperation == 1) ? FALSE : TRUE;

  WORD wh = m_pCamera->m_strCamera.strGeneral.strCamType.wCamType;
  if((wh == CAMERATYPE_PCO_DIMAX_STD) ||
    (wh == CAMERATYPE_PCO_DIMAX_TV) ||
    (wh == CAMERATYPE_PCO_DIMAX_AUTOMOTIVE))
  {
    bSymmetricalROIHorz = TRUE;
    bSymmetricalROIVert = TRUE;
  }
  if((wh == CAMERATYPE_PCO_EDGE) ||
    (wh == CAMERATYPE_PCO_EDGE_GL) ||
    //(wh == CAMERATYPE_PCO_EDGE_USB3) ||
    (wh == CAMERATYPE_PCO_EDGE_42))
    bSymmetricalROIVert = TRUE;

  if((m_pCamera->m_strCamera.strAPIManager.wAPIManagementFlags & APIMANAGEMENTFLAG_SOFTROI) == APIMANAGEMENTFLAG_SOFTROI)
  {
    bSymmetricalROIHorz = FALSE;
    bSymmetricalROIVert = FALSE;
  }

  if(m_pCamera->m_strCamera.strSensor.wSensorformat == 1)
  {
    if(bSymmetricalROIHorz)
    {
      if(m_pCamera->m_strCamera.strSensor.wRoiX0 < m_pCamera->m_strCamera.strSensor.strDescription.wMaxHorzResExtDESC / (2 * m_pCamera->m_strCamera.strSensor.wBinHorz))
        m_pCamera->m_strCamera.strSensor.wRoiX1 = m_pCamera->m_strCamera.strSensor.strDescription.wMaxHorzResExtDESC / m_pCamera->m_strCamera.strSensor.wBinHorz
        - m_pCamera->m_strCamera.strSensor.wRoiX0 + 1;
      else
        m_pCamera->m_strCamera.strSensor.wRoiX0 = m_pCamera->m_strCamera.strSensor.strDescription.wMaxHorzResExtDESC / m_pCamera->m_strCamera.strSensor.wBinHorz
        - m_pCamera->m_strCamera.strSensor.wRoiX1 + 1;
    }
    if(bSymmetricalROIVert)
    {
      if(m_pCamera->m_strCamera.strSensor.wRoiY0 < m_pCamera->m_strCamera.strSensor.strDescription.wMaxVertResExtDESC / (2 * m_pCamera->m_strCamera.strSensor.wBinVert))
        m_pCamera->m_strCamera.strSensor.wRoiY1 = m_pCamera->m_strCamera.strSensor.strDescription.wMaxVertResExtDESC / m_pCamera->m_strCamera.strSensor.wBinVert
        - m_pCamera->m_strCamera.strSensor.wRoiY0 + 1;
      else
        m_pCamera->m_strCamera.strSensor.wRoiY0 = m_pCamera->m_strCamera.strSensor.strDescription.wMaxVertResExtDESC / m_pCamera->m_strCamera.strSensor.wBinVert
        - m_pCamera->m_strCamera.strSensor.wRoiY1 + 1;
    }
  }
  else
  {
    if(bSymmetricalROIHorz)
    {
      if(m_pCamera->m_strCamera.strSensor.wRoiX0 < m_pCamera->m_strCamera.strSensor.strDescription.wMaxHorzResStdDESC / (2 * m_pCamera->m_strCamera.strSensor.wBinHorz))
        m_pCamera->m_strCamera.strSensor.wRoiX1 = m_pCamera->m_strCamera.strSensor.strDescription.wMaxHorzResStdDESC / m_pCamera->m_strCamera.strSensor.wBinHorz
        - m_pCamera->m_strCamera.strSensor.wRoiX0 + 1;
      else
        m_pCamera->m_strCamera.strSensor.wRoiX0 = m_pCamera->m_strCamera.strSensor.strDescription.wMaxHorzResStdDESC / m_pCamera->m_strCamera.strSensor.wBinHorz
        - m_pCamera->m_strCamera.strSensor.wRoiX1 + 1;
    }
    if(bSymmetricalROIVert)
    {
      if(m_pCamera->m_strCamera.strSensor.wRoiY0 < m_pCamera->m_strCamera.strSensor.strDescription.wMaxVertResStdDESC / (2 * m_pCamera->m_strCamera.strSensor.wBinVert))
        m_pCamera->m_strCamera.strSensor.wRoiY1 = m_pCamera->m_strCamera.strSensor.strDescription.wMaxVertResStdDESC / m_pCamera->m_strCamera.strSensor.wBinVert
        - m_pCamera->m_strCamera.strSensor.wRoiY0 + 1;
      else
        m_pCamera->m_strCamera.strSensor.wRoiY0 = m_pCamera->m_strCamera.strSensor.strDescription.wMaxVertResStdDESC / m_pCamera->m_strCamera.strSensor.wBinVert
        - m_pCamera->m_strCamera.strSensor.wRoiY1 + 1;
    }
  }
  if((m_pCamera->m_strCamera.strSensor.wRoiX1 - m_pCamera->m_strCamera.strSensor.wRoiX0 + 1) < m_pCamera->m_strCamera.strSensor.strDescription.wMinSizeHorzDESC)
  {
    if((m_pCamera->m_strCamera.strSensor.wRoiX1 + wRoiSteppinghorz) < wmaxhorz)
    {
      m_pCamera->m_strCamera.strSensor.wRoiX1 += wRoiSteppinghorz;
    }
    else
    {
      m_pCamera->m_strCamera.strSensor.wRoiX0 -= wRoiSteppinghorz;
    }
  }
  if((m_pCamera->m_strCamera.strSensor.wRoiY1 - m_pCamera->m_strCamera.strSensor.wRoiY0 + 1) < m_pCamera->m_strCamera.strSensor.strDescription.wMinSizeVertDESC)
  {
    if((m_pCamera->m_strCamera.strSensor.wRoiY1 + wRoiSteppingvert) < wmaxvert)
    {
      m_pCamera->m_strCamera.strSensor.wRoiY1 += wRoiSteppingvert;
    }
    else
    {
      m_pCamera->m_strCamera.strSensor.wRoiY0 -= wRoiSteppingvert;
    }
  }
  *Roix0 = m_pCamera->m_strCamera.strSensor.wRoiX0;
  *Roiy0 = m_pCamera->m_strCamera.strSensor.wRoiY0;
  *Roix1 = m_pCamera->m_strCamera.strSensor.wRoiX1;
  *Roiy1 = m_pCamera->m_strCamera.strSensor.wRoiY1;
  return PCO_NOERROR;
}

int CPCOCam::GetROI(unsigned& uX, unsigned& uY, unsigned& uXSize, unsigned& uYSize)
{
  int nErr = 0;
  if(m_bDemoMode)
  {
    uXSize = uX = 1280;
    uYSize = uY = 1024;

    return DEVICE_OK;
  }
  if(m_bSettingsChanged)
  {
    /*nErr = m_pCamera->getsettings(&m_nMode, &m_nTrig, &m_nRoiXMin, &m_nRoiXMax, &m_nRoiYMin, &m_nRoiYMax,
      &m_nHBin, &m_nVBin, m_pszTimes, &m_iGain, &m_iOffset, &m_uiFlags);
    if(m_pCamera->m_iCamClass == 3)// pco.camera*/
    {
      m_pCamera->GetCameraStruct((PCO_Camera*) &m_pCamera->m_strCamera.wSize);
    }
    m_bSettingsChanged = FALSE;
  }

  if(nErr != 0)
    return nErr;
  if(m_pCamera->m_iCamClass == 1)
  {
    uX = m_nRoiXMin / m_nHBin;
    uY = m_nRoiYMin / m_nVBin;

    uXSize = (m_nRoiXMax - m_nRoiXMin + 1) / m_nHBin;
    uYSize = (m_nRoiYMax - m_nRoiYMin + 1) / m_nVBin;
  }
  else
  {
    uX = m_nRoiXMin - 1;
    uY = m_nRoiYMin - 1;

    uXSize = m_nRoiXMax - m_nRoiXMin + 1;
    uYSize = m_nRoiYMax - m_nRoiYMin + 1;
  }
  return DEVICE_OK;
}

int CPCOCam::ClearROI()
{
  int nErr = 0;

  if(m_bDemoMode)
    return DEVICE_OK;

  m_nRoiXMin = 1;
  m_nRoiYMin = 1;
  m_nRoiXMax = roiXMaxFull_;
  m_nRoiYMax = roiYMaxFull_;

  m_pCamera->m_strCamera.strSensor.wRoiX0 = (WORD) m_nRoiXMin;
  m_pCamera->m_strCamera.strSensor.wRoiY0 = (WORD) m_nRoiYMin;
  m_pCamera->m_strCamera.strSensor.wRoiX1 = (WORD) m_nRoiXMax;
  m_pCamera->m_strCamera.strSensor.wRoiY1 = (WORD) m_nRoiYMax;

  
  if(m_pCamera->m_iCamClass == 2)// PixelFly
  {
    m_nRoiXMax = roiXMaxFull_ / (m_nHBin + 1);
    m_nRoiYMax = roiYMaxFull_ / (m_nVBin + 1);
  }
  if(m_pCamera->m_iCamClass == 3)// SC2
  {
    m_nRoiXMin = 1;
    m_nRoiYMin = 1;
    m_nRoiXMax = roiXMaxFull_ / m_nHBin;
    m_nRoiYMax = roiYMaxFull_ / m_nVBin;
    SetNCheckROI(&m_nRoiXMin, &m_nRoiXMax, &m_nRoiYMin, &m_nRoiYMax);
  }

  nErr = SetupCamera(true, true);

  if(nErr != 0)
    return nErr;

  // Liisa: read the current ROI to the variables to be used in SnapImage
  // Although the values set by SET_COC are correct here, it goes wrong somewhere later
  // and in SnapImage the old ROI is used
  if(m_bSettingsChanged)
  {
    /*nErr = m_pCamera->getsettings(&m_nMode, &m_nTrig, &m_nRoiXMin, &m_nRoiXMax, &m_nRoiYMin, &m_nRoiYMax,
      &m_nHBin, &m_nVBin, m_pszTimes, &m_iGain, &m_iOffset, &m_uiFlags);
    if(m_pCamera->m_iCamClass == 3)// pco.camera*/
    {
      m_pCamera->GetCameraStruct((PCO_Camera*) &m_pCamera->m_strCamera.wSize);
    }
    m_bSettingsChanged = FALSE;
  }
  // end Liisa

  if(nErr != 0)
    return nErr;

  nErr = ResizeImageBuffer();

  if(nErr != 0)
    return nErr;

  return DEVICE_OK;
}

void CPCOCam::SetExposure(double dExp)
{
  SetProperty(MM::g_Keyword_Exposure, CDeviceUtils::ConvertToString(dExp));
}

int CPCOCam::ResizeImageBuffer()
{
  // get image size
  int nWidth = 1, nHeight = 1;
  int as;

  int nErr = 0;
  if(m_bDemoMode)
  {
    nWidth = 1280;
    nHeight = 1024;
  }
  else
  {
    m_pCamera->ReloadSize();
    nErr = m_pCamera->getccdsize(&as, &nWidth, &nHeight);
    nWidth = m_pCamera->GetXRes();
    nHeight = m_pCamera->GetYRes();
    if(nErr != 0)
    {
      return nErr;
    }
    m_iWidth = nWidth;
    m_iHeight = nHeight;
  }

  if(!(pixelDepth_ == 1 || pixelDepth_ == 2 || pixelDepth_ == 4))
    return -1;
  img_.Resize(nWidth, nHeight, pixelDepth_);
  SetSizes(nWidth, nHeight, pixelDepth_);
  if(img_.Depth() == 1)
  {
    m_pCamera->SetConvertBWCol(TRUE, FALSE);
    m_pCamera->SetViewMode(TRUE, FALSE, FALSE, FALSE, 1.0, FALSE);//SetFlip(TRUE);  img_.Resize(nWidth, nHeight, pixelDepth_);
  }
  if(img_.Depth() == 4)
  {
    m_pCamera->SetConvertBWCol(FALSE, TRUE);
    m_pCamera->SetViewMode(TRUE, FALSE, FALSE, FALSE, 1.0, FALSE);//SetFlip(TRUE);  img_.Resize(nWidth, nHeight, pixelDepth_);
  }
  return DEVICE_OK;
}

int CPCOCam::PrepareSequenceAcqusition()
{
  return DEVICE_OK;
}

int CPCOCam::StartSequenceAcquisition(long numImages, double interval_ms, bool stopOnOverflow)
{
  int nErr = 0;

  if(Busy() || m_bSequenceRunning)
    return DEVICE_CAMERA_BUSY_ACQUIRING;

  if(m_bSoftwareTriggered)
  {
    m_pCamera->m_strCamera.strTiming.wTriggerMode = (WORD) 0;

    nErr = SetupCamera(true, false);
  }

  unsigned int uiMode = 0x80000 + 0x10000 + 0x0010;// Preview with ext.signals, update all, record (preview)
  nErr = m_pCamera->PreStartCam(uiMode, 0, 0, 0);            // schaltet automatisch auf internen Trigger
  for(int j = 0; j < 4; j++)
    m_iLastBufferUsed[j] = -1;
  m_iNextBuffer = 0;
  MM::Core *pcore = GetCoreCallback();
  int ret = DEVICE_OK;
  if(pcore != NULL)
    ret = pcore->PrepareForAcq(this);
  if(ret != DEVICE_OK)
    return ret;
  m_bSequenceRunning = true;
  m_iNumImages = numImages;
  m_iNumImagesInserted = 0;
  dIntervall = interval_ms;
  m_bDoAutoBalance = TRUE;

  sthd_->SetLength(numImages);
  SetSizes(GetImageWidth(), GetImageHeight(), GetImageBytesPerPixel());
  sthd_->Start(GetImageWidth(), GetImageHeight(), GetImageBytesPerPixel());

  m_bStopOnOverflow = stopOnOverflow;
  if(nErr != 0)
    return nErr;

  nErr = m_pCamera->StartCam();
  if(nErr == PCO_NOERROR)
    m_bRecording = true;

  return DEVICE_OK;
}

void CPCOCam::SetSizes(int iw, int ih, int ib)
{
  m_iWidth = iw;
  m_iHeight = ih;
  m_iBytesPerPixel = ib;
}

int CPCOCam::StopSequenceAcquisition()
{
  int nErr = 0;

  if(m_bSequenceRunning == false)
    return DEVICE_OK;

  sthd_->Stop();
  sthd_->wait();
  m_bSequenceRunning = false;
  m_pCamera->StopCam(&nErr);
  m_bRecording = false;

  if(m_bSoftwareTriggered)
  {
    m_pCamera->m_strCamera.strTiming.wTriggerMode = (WORD) 1;

    nErr = SetupCamera(true, false);
  }
  if(nErr != 0)
    return nErr;
  return DEVICE_OK;
}

int CPCOCam::StoppedByThread()
{
  int nErr = 0;

  m_pCamera->StopCam(&nErr);
  m_bSequenceRunning = false;
  m_bRecording = false;

  return DEVICE_OK;
}

bool CPCOCam::IsCapturing()
{
  return m_bSequenceRunning;
}

///////////////////////////////////////////////////////////////////////////////
// Function name   : CPCOCam::SnapImage
// Description     : Acquires a single frame and stores it in the internal
//                   buffer
// Return type     : bool 

int CPCOCam::SnapImage()
{
  int nErr = 0;

  if(m_bDemoMode)
    return DEVICE_OK;


  for(int i = 0; i < 4; i++)
  {
    m_iNextBufferToUse[i] = m_iLastBufferUsed[i];
  }
  if(m_bSequenceRunning == FALSE)
  {
    for(int i = 0; i < 4; i++)
    {
      m_iLastBufferUsed[i] = -1;
      m_iNextBufferToUse[i] = m_iLastBufferUsed[i];
    }

    SetSizes(GetImageWidth(), GetImageHeight(), GetImageBytesPerPixel());
    //m_iNextBufferToUse[0] = 0;
    m_iNextBuffer = 0;

    unsigned int uiMode = 0x80000 + 0x10000 + 0x0040 + 0x0010;//Preview with ext.signals, update all, single, record (preview)
    if(m_bSoftwareTriggered)
      uiMode = 0x10000 + 0x0010;
      
    nErr = m_pCamera->PreStartCam(uiMode, 0, 0, 0);            // schaltet automatisch auf internen Trigger

    nErr = m_pCamera->StartCam();
    if(nErr == PCO_NOERROR)
      m_bRecording = true;
    if(m_pCamera->m_iCamClass == 3)
      m_pCamera->ResetEvWait();
  }
  if(m_bSoftwareTriggered)
    m_pCamera->ForceTrigger();

  if (m_nTrig > 1)
  {
    bool bok = false;
    do
    {
      m_iLastBufferUsed[0] = m_iNextBuffer;

      nErr = m_pCamera->WaitForImage(&m_iNextBufferToUse[0], &m_iLastBufferUsed[0]);
      if (nErr == PCO_NOERROR)
        bok = true;
      if (::GetAsyncKeyState(VK_ESCAPE) & 0x8000) // Stops 'Auto Save' in progress
      {
        bok = true;
        nErr = PCO_NOERROR;
      }
    } while (!bok);
  }
  else
  {
    m_iLastBufferUsed[0] = m_iNextBuffer;

    nErr = m_pCamera->WaitForImage(&m_iNextBufferToUse[0], &m_iLastBufferUsed[0]);
  }
  if(m_bSequenceRunning == FALSE)
  {
    if(m_bSoftwareTriggered == FALSE)
    {
      int ilerr = 0;
      m_pCamera->StopCam(&ilerr);
      if(nErr == 0)
        nErr = ilerr;
      m_bRecording = false;
    }
  }
  if (m_bDoDotPhoton)
  {
    HANDLE hcamera = m_pCamera->GetCameraHandle();
    WORD* pImage = (WORD*)GetBuffer(m_iLastBufferUsed[0]);
    CalcImageDotphoton((WORD*)pImage, m_iWidth * m_iHeight, hcamera);
  }

  if(nErr != 0)
    return nErr;

  return DEVICE_OK;
}



int CPCOCam::InsertImage()
{
  const unsigned char* img;
  MM::Core *pcore = GetCoreCallback();
  int ret = DEVICE_OK;
  if(pcore != NULL)
  {
    int icurrent;

    for(int j = 0; j < 4; j++)
    {
      icurrent = m_iLastBufferUsed[j];
      if(icurrent < 0)
        break;
      m_iNextBuffer = icurrent + 1;
      if(m_iNextBuffer > 3)
        m_iNextBuffer = 0;
      img = GetBuffer(icurrent);
      if(img == 0)
        return ERR_TIMEOUT;

      m_iNumImagesInserted++;
      ret = pcore->InsertImage(this, img, m_iWidth, m_iHeight, m_iBytesPerPixel);
      if(!m_bStopOnOverflow && ret == DEVICE_BUFFER_OVERFLOW)
      {
        // do not stop on overflow - just reset the buffer
        pcore->ClearImageBuffer(this);
        return pcore->InsertImage(this, img, m_iWidth, m_iHeight, m_iBytesPerPixel);
      }
    }
  }
  else                                 // Else path for Unit-Tester
  {
    int icurrent;
    for(int j = 0; j < 4; j++)
    {
      icurrent = m_iLastBufferUsed[j];
      if(icurrent < 0)
        break;
      m_iNextBuffer = icurrent + 1;
      if(m_iNextBuffer > 3)
        m_iNextBuffer = 0;
      img = GetBuffer(icurrent);
      memcpy((void*) img_.GetPixels(), (void*) img, img_.Height() * img_.Width() * img_.Depth());
    }
  }
  return ret;
}


int CPCOCam::SequenceThread::svc()
{
  long count(0);
  int err = 0;
  static SYSTEMTIME  st;
  camera_->InsertImage();

  while(!stop_ && count < numImages_)
  {
    if(WaitForSingleObject(camera_->mxMutex, 2000) == WAIT_TIMEOUT)// Try to lock acquisition thread
    {
      return DEVICE_CAMERA_BUSY_ACQUIRING;
    }

    int ret = camera_->SnapImage();
    if(ret != DEVICE_OK)
    {
      err = 1;
      break;
    }

    if((m_svcWidth == camera_->m_iWidth) && (m_svcHeight == camera_->m_iHeight) && (m_svcBytePP == camera_->m_iBytesPerPixel))
    {
      ret = camera_->InsertImage();
    }
    if(ret != DEVICE_OK)
    {
      err = 1;
      break;
    }
    //CDeviceUtils::SleepMs(20);
    count = camera_->m_iNumImagesInserted;
    ReleaseMutex(camera_->mxMutex);
  }

  camera_->StoppedByThread();
  camera_->CleanupSequenceAcquisition();
  return err;
}

int CPCOCam::CleanupSequenceAcquisition()
{
  MM::Core* cb = GetCoreCallback();
  if(cb)
    return cb->AcqFinished(this, 0);

  return DEVICE_OK;
}

void CPCOCam::WriteLog(char* message, int nErr)
{
  char szmes[300];

  if(nErr != 0)
    sprintf_s(szmes, sizeof(szmes), "Error %x! %s", nErr, message);
  else
    sprintf_s(szmes, sizeof(szmes), "%s", message);
  LogMessage(szmes);
}
