/*=============================================================================
  Copyright (C) 2023 Allied Vision Technologies.  All Rights Reserved.

  Redistribution of this header file, in original or modified form, without
  prior written consent of Allied Vision Technologies is prohibited.

  THIS SOFTWARE IS PROVIDED BY THE AUTHOR "AS IS" AND ANY EXPRESS OR IMPLIED
  WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF TITLE,
  NON-INFRINGEMENT, MERCHANTABILITY AND FITNESS FOR A PARTICULAR  PURPOSE ARE
  DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
  INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
  AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
  TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

=============================================================================*/
#include "AlliedVisionHub.h"

#include "AlliedVisionCamera.h"

AlliedVisionHub::AlliedVisionHub() : m_sdk(std::make_shared<VimbaXApi>()) {}

int AlliedVisionHub::DetectInstalledDevices() {
  LogMessage("Detecting installed cameras...");
  VmbUint32_t camNum;
  // Get the number of connected cameras first
  VmbError_t err = m_sdk->VmbCamerasList_t(nullptr, 0, &camNum, 0);
  if (VmbErrorSuccess == err) {
    VmbCameraInfo_t *camInfo = new VmbCameraInfo_t[camNum];

    // Get the cameras
    err = m_sdk->VmbCamerasList_t(camInfo, camNum, &camNum, sizeof *camInfo);

    if (err == VmbErrorSuccess) {
      for (VmbUint32_t i = 0; i < camNum; ++i) {
        if (camInfo[i].permittedAccess & VmbAccessModeFull) {
          MM::Device *pDev = new AlliedVisionCamera(camInfo[i].cameraIdString);
          AddInstalledDevice(pDev);
        }
      }
    }

    delete[] camInfo;
  } else {
    LOG_ERROR(err, "Cannot get installed devices!");
  }

  return err;
}

int AlliedVisionHub::Initialize() {
  LogMessage("Init HUB");
  if (m_sdk->isInitialized()) {
    return DEVICE_OK;
  } else {
    LOG_ERROR(VmbErrorApiNotStarted, "SDK not initialized!");
    return VmbErrorApiNotStarted;
  }
}

int AlliedVisionHub::Shutdown() {
  LogMessage("Shutting down HUB");
  return DEVICE_OK;
}

void AlliedVisionHub::GetName(char *name) const {
  CDeviceUtils::CopyLimitedString(name, g_hubName);
}

bool AlliedVisionHub::Busy() { return false; }

std::shared_ptr<VimbaXApi> &AlliedVisionHub::getSDK() { return m_sdk; }
