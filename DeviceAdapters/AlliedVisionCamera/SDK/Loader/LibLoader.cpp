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
#include "LibLoader.h"

#include "Constants.h"

VimbaXApi::VimbaXApi()
    : m_sdk(VIMBA_X_LIB_NAME, VIMBA_X_LIB_DIR.c_str()), m_initialized(false) {
  if (m_sdk.isLoaded()) {
    // TODO implement error handling if function not resolved
    VmbStartup_t = m_sdk.resolveFunction("VmbStartup");
    VmbVersionQuery_t = m_sdk.resolveFunction("VmbVersionQuery");
    VmbShutdown_t = m_sdk.resolveFunction("VmbShutdown");
    VmbCamerasList_t = m_sdk.resolveFunction("VmbCamerasList");
    VmbCameraOpen_t = m_sdk.resolveFunction("VmbCameraOpen");
    VmbCameraClose_t = m_sdk.resolveFunction("VmbCameraClose");
    VmbPayloadSizeGet_t = m_sdk.resolveFunction("VmbPayloadSizeGet");
    VmbFrameAnnounce_t = m_sdk.resolveFunction("VmbFrameAnnounce");
    VmbCaptureStart_t = m_sdk.resolveFunction("VmbCaptureStart");
    VmbCaptureEnd_t = m_sdk.resolveFunction("VmbCaptureEnd");
    VmbCaptureFrameQueue_t = m_sdk.resolveFunction("VmbCaptureFrameQueue");
    VmbCaptureFrameWait_t = m_sdk.resolveFunction("VmbCaptureFrameWait");
    VmbCaptureQueueFlush_t = m_sdk.resolveFunction("VmbCaptureQueueFlush");
    VmbFrameRevokeAll_t = m_sdk.resolveFunction("VmbFrameRevokeAll");
    VmbFeatureCommandRun_t = m_sdk.resolveFunction("VmbFeatureCommandRun");
    VmbFeaturesList_t = m_sdk.resolveFunction("VmbFeaturesList");
    VmbFeatureBoolGet_t = m_sdk.resolveFunction("VmbFeatureBoolGet");
    VmbFeatureBoolSet_t = m_sdk.resolveFunction("VmbFeatureBoolSet");
    VmbFeatureEnumGet_t = m_sdk.resolveFunction("VmbFeatureEnumGet");
    VmbFeatureEnumSet_t = m_sdk.resolveFunction("VmbFeatureEnumSet");
    VmbFeatureFloatGet_t = m_sdk.resolveFunction("VmbFeatureFloatGet");
    VmbFeatureFloatSet_t = m_sdk.resolveFunction("VmbFeatureFloatSet");
    VmbFeatureIntGet_t = m_sdk.resolveFunction("VmbFeatureIntGet");
    VmbFeatureIntSet_t = m_sdk.resolveFunction("VmbFeatureIntSet");
    VmbFeatureStringGet_t = m_sdk.resolveFunction("VmbFeatureStringGet");
    VmbFeatureStringSet_t = m_sdk.resolveFunction("VmbFeatureStringSet");
    VmbChunkDataAccess_t = m_sdk.resolveFunction("VmbChunkDataAccess");
    VmbFeatureEnumRangeQuery_t =
        m_sdk.resolveFunction("VmbFeatureEnumRangeQuery");
    VmbFeatureIntRangeQuery_t =
        m_sdk.resolveFunction("VmbFeatureIntRangeQuery");
    VmbFeatureStringMaxlengthQuery_t =
        m_sdk.resolveFunction("VmbFeatureStringMaxlengthQuery");
    VmbFeatureInfoQuery_t = m_sdk.resolveFunction("VmbFeatureInfoQuery");
    VmbFeatureFloatRangeQuery_t =
        m_sdk.resolveFunction("VmbFeatureFloatRangeQuery");
    VmbFeatureInvalidationRegister_t =
        m_sdk.resolveFunction("VmbFeatureInvalidationRegister");
    VmbFeatureAccessQuery_t = m_sdk.resolveFunction("VmbFeatureAccessQuery");
    VmbFeatureIntIncrementQuery_t =
        m_sdk.resolveFunction("VmbFeatureIntIncrementQuery");
    VmbFeatureFloatIncrementQuery_t =
        m_sdk.resolveFunction("VmbFeatureFloatIncrementQuery");
    VmbFeatureCommandIsDone_t =
        m_sdk.resolveFunction("VmbFeatureCommandIsDone");
    //TODO check all resolved symbols
    auto err = VmbStartup_t(nullptr);
    if(err == VmbErrorSuccess) {
      m_initialized = true;
    }
  }
}

bool VimbaXApi::isInitialized() const { return m_initialized; }

VimbaXApi::~VimbaXApi() { 
    m_initialized = false;
    VmbShutdown_t(); }

LibLoader::LibLoader(const char* lib, const char* libPath)
    : m_libName(lib), m_libPath(libPath), m_module(nullptr), m_loaded(false) {
  SetDllDirectoryA(m_libPath);
  m_module = LoadLibraryA(m_libName);
  if (m_module != nullptr) {
    m_loaded = true;
  }
}

LibLoader::~LibLoader() {
  if (m_loaded) {
    FreeModule(m_module);
    m_module = nullptr;
    m_loaded = false;
    m_libName = nullptr;
  }
}

bool LibLoader::isLoaded() const { return m_loaded; }

ProcWrapper LibLoader::resolveFunction(const char* functionName) const {
  if (m_module && m_loaded) {
    return ProcWrapper(GetProcAddress((HMODULE)m_module, functionName));
  }

  return ProcWrapper(nullptr);
}