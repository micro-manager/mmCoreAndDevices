#include "AlliedVisionHub.h"

#include "AlliedVisionCamera.h"

AlliedVisionHub::AlliedVisionHub(std::unique_ptr<VimbaXApi>& sdk) : m_sdk(sdk) {}

AlliedVisionHub::~AlliedVisionHub() {
  // Release static SDK variable from DLL, otherwise process will not be
  // killed, destructor not called
  m_sdk.reset();
}

int AlliedVisionHub::DetectInstalledDevices() {
  VmbUint32_t camNum;
  // Get the number of connected cameras first
  VmbError_t err = m_sdk->VmbCamerasList_t(nullptr, 0, &camNum, 0);
  if (VmbErrorSuccess == err) {
    VmbCameraInfo_t* camInfo = new VmbCameraInfo_t[camNum];

    // Get the cameras
    err = m_sdk->VmbCamerasList_t(camInfo, camNum, &camNum, sizeof *camInfo);

    if (err == VmbErrorSuccess) {
      for (VmbUint32_t i = 0; i < camNum; ++i) {
        if (camInfo[i].permittedAccess & VmbAccessModeFull) {
          MM::Device* pDev =
              new AlliedVisionCamera(camInfo[i].cameraIdString, m_sdk);
          AddInstalledDevice(pDev);
        }
      }
    }

    delete[] camInfo;
  }

  return err;
}

int AlliedVisionHub::Initialize() {
  LogMessage("Init HUB");  
  return DEVICE_OK;
}

int AlliedVisionHub::Shutdown() {
  LogMessage("Shutting down HUB");
  return DEVICE_OK;
}

void AlliedVisionHub::GetName(char* name) const {
  CDeviceUtils::CopyLimitedString(name, g_hubName);
}

bool AlliedVisionHub::Busy() { return false; }
