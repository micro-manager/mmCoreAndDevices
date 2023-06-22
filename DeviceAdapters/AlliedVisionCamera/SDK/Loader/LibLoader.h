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
#ifndef LIBLOADER_H
#define LIBLOADER_H

#include <Windows.h>

#include <string>

#include "VmbC/VmbC.h"

/**
 * @brief Wrapper for single Windows process
 */
class ProcWrapper {
  // PUBLIC
 public:
  /**
   * @brief Constructor of wrapper
   * @param[in] funPtr  Pointer to the process
   */
  explicit ProcWrapper(FARPROC funPtr) : m_funPtr(funPtr) {}

  /**
   * @brief Operator () overload to call function
   */
  template <typename T, typename = std::enable_if_t<std::is_function_v<T>>>
  operator T*() const {
    return reinterpret_cast<T*>(m_funPtr);
  }
  // PRIVATE
 private:
  FARPROC m_funPtr;  //<! Function pointer
};

/**
 * @brief Class responsible for loading particular library
 */
class LibLoader {
  // PUBLIC
 public:
  /**
   * @brief Constructor of lib loader
   * @param[in] libName     Library name (includign extension)
   * @param[in] libPath     Library path
   */
  explicit LibLoader(const char* libName, const char* libPath);

  /**
   * @brief Destructor of lib loader
   */
  ~LibLoader();

  /**
   * @brief Resolve function from library
   * @param[in] functionName Function name to be resolved
   * @return ProcWrapper object to the resolved function. NULLPTR if not
   * resolved
   */
  ProcWrapper resolveFunction(const char* functionName) const;

  /**
   * @brief Getter to check if library is loaded and initialized
   * @return True if library is initialized
   */
  bool isInitialized() const;
  // PRIVATE
 private:
  // MEMBERS
  const char* m_libName;  //<! Library name
  const char* m_libPath;  //<! Library path
  HMODULE m_module;       //<! Windows HMODULE object for loaded library
  bool m_initialized;     //<! Is library initialized
};

/**
 * @brief Main Vimba API wrapper class that hold all of the resolved API
 * functions to be called
 */
class VimbaXApi {
  // PUBLIC
 public:
  /**
   * @brief Constructor of class
   */
  explicit VimbaXApi();

  /**
   * @brief Destructor of class
   */
  ~VimbaXApi() = default;

  // Deleted methods
  VimbaXApi(const VimbaXApi&) = delete;
  VimbaXApi(VimbaXApi&&) = delete;
  void operator=(VimbaXApi const&) = delete;

  // Vimba X API methods that will be resolved
  decltype(VmbStartup)* VmbStartup_t = nullptr;
  decltype(VmbVersionQuery)* VmbVersionQuery_t = nullptr;
  decltype(VmbShutdown)* VmbShutdown_t = nullptr;
  decltype(VmbCamerasList)* VmbCamerasList_t = nullptr;
  decltype(VmbCameraOpen)* VmbCameraOpen_t = nullptr;
  decltype(VmbCameraClose)* VmbCameraClose_t = nullptr;
  decltype(VmbPayloadSizeGet)* VmbPayloadSizeGet_t = nullptr;
  decltype(VmbFrameAnnounce)* VmbFrameAnnounce_t = nullptr;
  decltype(VmbCaptureStart)* VmbCaptureStart_t = nullptr;
  decltype(VmbCaptureEnd)* VmbCaptureEnd_t = nullptr;
  decltype(VmbCaptureFrameQueue)* VmbCaptureFrameQueue_t = nullptr;
  decltype(VmbCaptureFrameWait)* VmbCaptureFrameWait_t = nullptr;
  decltype(VmbCaptureQueueFlush)* VmbCaptureQueueFlush_t = nullptr;
  decltype(VmbFrameRevokeAll)* VmbFrameRevokeAll_t = nullptr;
  decltype(VmbFeatureCommandRun)* VmbFeatureCommandRun_t = nullptr;
  decltype(VmbFeatureEnumSet)* VmbFeatureEnumSet_t = nullptr;
  decltype(VmbFeaturesList)* VmbFeaturesList_t = nullptr;
  decltype(VmbFeatureBoolGet)* VmbFeatureBoolGet_t = nullptr;
  decltype(VmbFeatureEnumGet)* VmbFeatureEnumGet_t = nullptr;
  decltype(VmbFeatureFloatGet)* VmbFeatureFloatGet_t = nullptr;
  decltype(VmbFeatureIntGet)* VmbFeatureIntGet_t = nullptr;
  decltype(VmbFeatureStringGet)* VmbFeatureStringGet_t = nullptr;
  decltype(VmbChunkDataAccess)* VmbChunkDataAccess_t = nullptr;
  // PRIVATE
 private:
  LibLoader m_sdk;  //<! SDK library
};

#endif
