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
#ifndef LIBLOADER_H
#define LIBLOADER_H

#include <string>
#include <type_traits>

#include "VmbC/VmbC.h"
#include "VmbImageTransform/VmbTransform.h"

/**
 * @brief Wrapper for single Windows process
 */
class SymbolWrapper {
  ///////////////////////////////////////////////////////////////////////////////
  // PUBLIC
  ///////////////////////////////////////////////////////////////////////////////
 public:
  /**
   * @brief Constructor of wrapper
   * @param[in] funPtr  Pointer to the resolved symbol
   */
  explicit SymbolWrapper(void* funPtr) : m_funPtr(funPtr) {}

  /**
   * @brief Operator () overload to call function
   */
  template <typename T, typename = std::enable_if_t<std::is_function<T>::value>>
  operator T*() const {
    return reinterpret_cast<T*>(m_funPtr);
  }
  ///////////////////////////////////////////////////////////////////////////////
  // PRIVATE
  ///////////////////////////////////////////////////////////////////////////////
 private:
  void* m_funPtr;  //<! Function pointer
};

/**
 * @brief Class responsible for loading particular library
 */
class LibLoader {
  ///////////////////////////////////////////////////////////////////////////////
  // PUBLIC
  ///////////////////////////////////////////////////////////////////////////////
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
   * @param[in] functionName    Function name to be resolved
   * @param[out] allResolved     Bool flag to indicate if symbol was resolved
   * @return SymbolWrapper object to the resolved function. NULLPTR if not
   * resolved
   */
  SymbolWrapper resolveFunction(const char* functionName,
                                bool& allResolved) const;

  /**
   * @brief Getter to check if library is loaded
   * @return True if library is loaded
   */
  bool isLoaded() const;
  ///////////////////////////////////////////////////////////////////////////////
  // PRIVATE
  ///////////////////////////////////////////////////////////////////////////////
 private:
  const char* m_libName;  //<! Library name
  const char* m_libPath;  //<! Library path
  void* m_module;         //<! Handle for loaded library
  bool m_loaded;          //<! Is library initialized
};

/**
 * @brief Main Vimba API wrapper class that hold all of the resolved API
 * functions to be called
 */
class VimbaXApi {
  ///////////////////////////////////////////////////////////////////////////////
  // PUBLIC
  ///////////////////////////////////////////////////////////////////////////////
 public:
  /**
   * @brief Constructor of class
   */
  explicit VimbaXApi();

  /**
   * @brief Destructor of class
   */
  ~VimbaXApi();

  /**
   * @brief Is SDK initialized correctly
   * @return True if initialized correctly, otherwise false
   */
  bool isInitialized() const;

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
  decltype(VmbFeaturesList)* VmbFeaturesList_t = nullptr;
  decltype(VmbFeatureBoolGet)* VmbFeatureBoolGet_t = nullptr;
  decltype(VmbFeatureBoolSet)* VmbFeatureBoolSet_t = nullptr;
  decltype(VmbFeatureEnumGet)* VmbFeatureEnumGet_t = nullptr;
  decltype(VmbFeatureEnumSet)* VmbFeatureEnumSet_t = nullptr;
  decltype(VmbFeatureFloatGet)* VmbFeatureFloatGet_t = nullptr;
  decltype(VmbFeatureFloatSet)* VmbFeatureFloatSet_t = nullptr;
  decltype(VmbFeatureIntGet)* VmbFeatureIntGet_t = nullptr;
  decltype(VmbFeatureIntSet)* VmbFeatureIntSet_t = nullptr;
  decltype(VmbFeatureStringGet)* VmbFeatureStringGet_t = nullptr;
  decltype(VmbFeatureStringSet)* VmbFeatureStringSet_t = nullptr;
  decltype(VmbChunkDataAccess)* VmbChunkDataAccess_t = nullptr;
  decltype(VmbFeatureEnumRangeQuery)* VmbFeatureEnumRangeQuery_t = nullptr;
  decltype(VmbFeatureIntRangeQuery)* VmbFeatureIntRangeQuery_t = nullptr;
  decltype(VmbFeatureStringMaxlengthQuery)* VmbFeatureStringMaxlengthQuery_t =
      nullptr;
  decltype(VmbFeatureRawLengthQuery)* VmbFeatureRawLengthQuery_t = nullptr;
  decltype(VmbFeatureInfoQuery)* VmbFeatureInfoQuery_t = nullptr;
  decltype(VmbFeatureFloatRangeQuery)* VmbFeatureFloatRangeQuery_t = nullptr;
  decltype(VmbFeatureInvalidationRegister)* VmbFeatureInvalidationRegister_t =
      nullptr;
  decltype(VmbFeatureAccessQuery)* VmbFeatureAccessQuery_t = nullptr;
  decltype(VmbFeatureIntIncrementQuery)* VmbFeatureIntIncrementQuery_t =
      nullptr;
  decltype(VmbFeatureFloatIncrementQuery)* VmbFeatureFloatIncrementQuery_t =
      nullptr;
  decltype(VmbFeatureCommandIsDone)* VmbFeatureCommandIsDone_t = nullptr;
  decltype(VmbSetImageInfoFromPixelFormat)* VmbSetImageInfoFromPixelFormat_t =
      nullptr;
  decltype(VmbImageTransform)* VmbImageTransform_t = nullptr;
  ///////////////////////////////////////////////////////////////////////////////
  // PRIVATE
  ///////////////////////////////////////////////////////////////////////////////
 private:
  bool m_initialized;          //<! Flag if SDK is initialized
  LibLoader m_sdk;             //<! SDK library
  LibLoader m_imageTransform;  //<! Image Transform library
};

#endif
