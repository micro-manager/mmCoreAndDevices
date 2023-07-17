#include "AlliedVisionHub.h"

#include "AlliedVisionCamera.h"

AlliedVisionHub::AlliedVisionHub() : m_sdk(std::make_shared<VimbaXApi>()) {
  InitializeDefaultErrorMessages(); 
  setApiErrorMessages();
}

int AlliedVisionHub::DetectInstalledDevices() {
  LogMessage("Detecting installed cameras...");
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
          MM::Device* pDev = new AlliedVisionCamera(camInfo[i].cameraIdString);
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
  if (m_sdk->isInitialized()) {
    return DEVICE_OK;
  } else {
    LogMessage("SDK not initialized!");
    return VmbErrorApiNotStarted;
  }
}

int AlliedVisionHub::Shutdown() {
  LogMessage("Shutting down HUB");
  return DEVICE_OK;
}

void AlliedVisionHub::GetName(char* name) const {
  CDeviceUtils::CopyLimitedString(name, g_hubName);
}

bool AlliedVisionHub::Busy() { return false; }

std::shared_ptr<VimbaXApi>& AlliedVisionHub::getSDK() { return m_sdk; }

void AlliedVisionHub::setApiErrorMessages() {
  SetErrorText(VmbErrorApiNotStarted, "Vimba X API not started");
  SetErrorText(VmbErrorNotFound, "Device cannot be found");
  SetErrorText(VmbErrorDeviceNotOpen, "Device cannot be opened");
  SetErrorText(VmbErrorBadParameter,
               "Invalid parameter passed to the function");
  SetErrorText(VmbErrorNotImplemented, "Feature not implemented");
  SetErrorText(VmbErrorNotSupported, "Feature not supported");
  SetErrorText(VmbErrorUnknown, "Unknown error");
  SetErrorText(VmbErrorInvalidValue,
               "The value is not valid: either out of bounds or not an "
               "increment of the minimum");
  SetErrorText(VmbErrorBadHandle, "Given device handle is not valid");
  SetErrorText(VmbErrorInvalidAccess,
               "Operation is invalid with the current access mode");
  SetErrorText(VmbErrorTimeout, "Timeout occured");
  SetErrorText(VmbErrorNotAvailable, "Something is not available");
  SetErrorText(VmbErrorNotInitialized, "Something is not initialized");
  SetErrorText(VmbErrorAlready, "The operation has been already done");
}
