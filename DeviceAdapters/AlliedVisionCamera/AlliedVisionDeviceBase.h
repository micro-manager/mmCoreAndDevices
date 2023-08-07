/*=============================================================================
  Copyright (C) 2012 - 2023 Allied Vision Technologies.  All Rights Reserved.

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
#ifndef ALLIEDVISIONDEVICEBASE_H
#define ALLIEDVISIONDEVICEBASE_H

#include "DeviceBase.h"
#include "SDK/Loader/LibLoader.h"

#define LOG_ERROR(err, message) logError(err, message, __FUNCTION__, __LINE__)

/**
 * @brief Base class for Allied Vision devices
 */
template <typename T, typename U>
class AlliedVisionDeviceBase : public CDeviceBase<T, U> {
  ///////////////////////////////////////////////////////////////////////////////
  // PUBLIC
  ///////////////////////////////////////////////////////////////////////////////
 public:
  /**
   * @brief Constructor
   */
  AlliedVisionDeviceBase() {
    CDeviceBase<T, U>::InitializeDefaultErrorMessages();
    setApiErrorMessages();
  };

  /**
   * @brief Destructor
   */
  virtual ~AlliedVisionDeviceBase() = default;

  void logError(int error, std::string message, std::string function = "",
                int line = 0) const {
    std::string prefix = "[" + function + "():" + std::to_string(line) + "] ";
    CDeviceBase<T, U>::LogMessage(prefix + message);
    CDeviceBase<T, U>::LogMessageCode(error);
  }

  ///////////////////////////////////////////////////////////////////////////////
  // PRIVATE
  ///////////////////////////////////////////////////////////////////////////////
 private:
  /**
   * @brief Setup error messages for Vimba API
   */
  void setApiErrorMessages() {
    CDeviceBase<T, U>::SetErrorText(VmbErrorApiNotStarted,
                                    "Vimba X API not started");
    CDeviceBase<T, U>::SetErrorText(VmbErrorNotFound, "Device cannot be found");
    CDeviceBase<T, U>::SetErrorText(VmbErrorDeviceNotOpen,
                                    "Device cannot be opened");
    CDeviceBase<T, U>::SetErrorText(VmbErrorBadParameter,
                                    "Invalid parameter passed to the function");
    CDeviceBase<T, U>::SetErrorText(VmbErrorNotImplemented,
                                    "Feature not implemented");
    CDeviceBase<T, U>::SetErrorText(VmbErrorNotSupported,
                                    "Feature not supported");
    CDeviceBase<T, U>::SetErrorText(VmbErrorUnknown, "Unknown error");
    CDeviceBase<T, U>::SetErrorText(
        VmbErrorInvalidValue,
        "The value is not valid: either out of bounds or not an "
        "increment of the minimum");
    CDeviceBase<T, U>::SetErrorText(VmbErrorBadHandle,
                                    "Given device handle is not valid");
    CDeviceBase<T, U>::SetErrorText(
        VmbErrorInvalidAccess,
        "Operation is invalid with the current access mode");
    CDeviceBase<T, U>::SetErrorText(VmbErrorTimeout, "Timeout occured");
    CDeviceBase<T, U>::SetErrorText(VmbErrorNotAvailable,
                                    "Something is not available");
    CDeviceBase<T, U>::SetErrorText(VmbErrorNotInitialized,
                                    "Something is not initialized");
    CDeviceBase<T, U>::SetErrorText(VmbErrorAlready,
                                    "The operation has been already done");
    CDeviceBase<T, U>::SetErrorText(VmbErrorFeaturesUnavailable,
                                    "Feature is currently unavailable");
  }
};

#endif
