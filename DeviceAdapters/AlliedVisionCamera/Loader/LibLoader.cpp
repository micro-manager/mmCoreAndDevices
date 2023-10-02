/*=============================================================================
  Copyright (C) 2023 Allied Vision Technologies.  All Rights Reserved.

  This file is distributed under the BSD license.
  License text is included with the source distribution.

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

#ifdef __linux__
#include <dlfcn.h>
#elif _WIN32
#include <Windows.h>
#else
// nothing
#endif

#include "Constants.h"

VimbaXApi::VimbaXApi() :
    m_sdk(VIMBA_X_LIB_NAME, VIMBA_X_LIB_DIR.c_str()),
    m_initialized(false),
    m_imageTransform(VIMBA_X_IMAGE_TRANSFORM_NAME, VIMBA_X_LIB_DIR.c_str())
{
    if (m_sdk.isLoaded() && m_imageTransform.isLoaded())
    {
        bool allResolved = true;
        VmbStartup_t = m_sdk.resolveFunction("VmbStartup", allResolved);
        VmbVersionQuery_t = m_sdk.resolveFunction("VmbVersionQuery", allResolved);
        VmbShutdown_t = m_sdk.resolveFunction("VmbShutdown", allResolved);
        VmbCamerasList_t = m_sdk.resolveFunction("VmbCamerasList", allResolved);
        VmbCameraOpen_t = m_sdk.resolveFunction("VmbCameraOpen", allResolved);
        VmbCameraClose_t = m_sdk.resolveFunction("VmbCameraClose", allResolved);
        VmbCameraInfoQuery_t = m_sdk.resolveFunction("VmbCameraInfoQuery", allResolved);
        VmbPayloadSizeGet_t = m_sdk.resolveFunction("VmbPayloadSizeGet", allResolved);
        VmbFrameAnnounce_t = m_sdk.resolveFunction("VmbFrameAnnounce", allResolved);
        VmbCaptureStart_t = m_sdk.resolveFunction("VmbCaptureStart", allResolved);
        VmbCaptureEnd_t = m_sdk.resolveFunction("VmbCaptureEnd", allResolved);
        VmbCaptureFrameQueue_t = m_sdk.resolveFunction("VmbCaptureFrameQueue", allResolved);
        VmbCaptureFrameWait_t = m_sdk.resolveFunction("VmbCaptureFrameWait", allResolved);
        VmbCaptureQueueFlush_t = m_sdk.resolveFunction("VmbCaptureQueueFlush", allResolved);
        VmbFrameRevokeAll_t = m_sdk.resolveFunction("VmbFrameRevokeAll", allResolved);
        VmbFeatureCommandRun_t = m_sdk.resolveFunction("VmbFeatureCommandRun", allResolved);
        VmbFeaturesList_t = m_sdk.resolveFunction("VmbFeaturesList", allResolved);
        VmbFeatureBoolGet_t = m_sdk.resolveFunction("VmbFeatureBoolGet", allResolved);
        VmbFeatureBoolSet_t = m_sdk.resolveFunction("VmbFeatureBoolSet", allResolved);
        VmbFeatureEnumGet_t = m_sdk.resolveFunction("VmbFeatureEnumGet", allResolved);
        VmbFeatureEnumSet_t = m_sdk.resolveFunction("VmbFeatureEnumSet", allResolved);
        VmbFeatureFloatGet_t = m_sdk.resolveFunction("VmbFeatureFloatGet", allResolved);
        VmbFeatureFloatSet_t = m_sdk.resolveFunction("VmbFeatureFloatSet", allResolved);
        VmbFeatureIntGet_t = m_sdk.resolveFunction("VmbFeatureIntGet", allResolved);
        VmbFeatureIntSet_t = m_sdk.resolveFunction("VmbFeatureIntSet", allResolved);
        VmbFeatureStringGet_t = m_sdk.resolveFunction("VmbFeatureStringGet", allResolved);
        VmbFeatureStringSet_t = m_sdk.resolveFunction("VmbFeatureStringSet", allResolved);
        VmbChunkDataAccess_t = m_sdk.resolveFunction("VmbChunkDataAccess", allResolved);
        VmbFeatureEnumRangeQuery_t = m_sdk.resolveFunction("VmbFeatureEnumRangeQuery", allResolved);
        VmbFeatureIntRangeQuery_t = m_sdk.resolveFunction("VmbFeatureIntRangeQuery", allResolved);
        VmbFeatureStringMaxlengthQuery_t = m_sdk.resolveFunction("VmbFeatureStringMaxlengthQuery", allResolved);
        VmbFeatureInfoQuery_t = m_sdk.resolveFunction("VmbFeatureInfoQuery", allResolved);
        VmbFeatureFloatRangeQuery_t = m_sdk.resolveFunction("VmbFeatureFloatRangeQuery", allResolved);
        VmbFeatureInvalidationRegister_t = m_sdk.resolveFunction("VmbFeatureInvalidationRegister", allResolved);
        VmbFeatureAccessQuery_t = m_sdk.resolveFunction("VmbFeatureAccessQuery", allResolved);
        VmbFeatureIntIncrementQuery_t = m_sdk.resolveFunction("VmbFeatureIntIncrementQuery", allResolved);
        VmbFeatureFloatIncrementQuery_t = m_sdk.resolveFunction("VmbFeatureFloatIncrementQuery", allResolved);
        VmbFeatureCommandIsDone_t = m_sdk.resolveFunction("VmbFeatureCommandIsDone", allResolved);
        VmbSetImageInfoFromPixelFormat_t = m_imageTransform.resolveFunction("VmbSetImageInfoFromPixelFormat", allResolved);
        VmbImageTransform_t = m_imageTransform.resolveFunction("VmbImageTransform", allResolved);
        if (allResolved)
        {
            auto err = VmbStartup_t(nullptr);
            if (err == VmbErrorSuccess)
            {
                m_initialized = true;
            }
        }
    }
}

bool VimbaXApi::isInitialized() const
{
    return m_initialized;
}

VimbaXApi::~VimbaXApi()
{
    if (m_initialized)
    {
        m_initialized = false;
        VmbShutdown_t();
    }
}

bool LibLoader::isLoaded() const
{
    return m_loaded;
}
#ifdef __linux__
LibLoader::LibLoader(const char *lib, const char *libPath) :
    m_libName(lib),
    m_libPath(libPath),
    m_module(nullptr),
    m_loaded(false)
{
    std::string path = std::string(m_libName);
    dlerror();
    m_module = dlopen(path.c_str(), RTLD_NOW);
    if (m_module != nullptr)
    {
        m_loaded = true;
    }
}

LibLoader::~LibLoader()
{
    if (m_loaded)
    {
        dlclose(m_module);
        m_module = nullptr;
        m_loaded = false;
        m_libName = nullptr;
    }
}

SymbolWrapper LibLoader::resolveFunction(const char *functionName, bool &allResolved) const
{
    dlerror();
    void *funPtr = nullptr;
    if (allResolved)
    {
        funPtr = dlsym(m_module, functionName);
        allResolved = allResolved && (funPtr != nullptr);
    }
    return SymbolWrapper(funPtr);
}
#elif _WIN32
LibLoader::LibLoader(const char *lib, const char *libPath) :
    m_libName(lib),
    m_libPath(libPath),
    m_module(nullptr),
    m_loaded(false)
{
    SetDllDirectoryA(m_libPath);
    m_module = LoadLibraryA(m_libName);
    if (m_module != nullptr)
    {
        m_loaded = true;
    }
}

LibLoader::~LibLoader()
{
    if (m_loaded)
    {
        FreeModule(static_cast<HMODULE>(m_module));
        m_module = nullptr;
        m_loaded = false;
        m_libName = nullptr;
    }
}

SymbolWrapper LibLoader::resolveFunction(const char *functionName, bool &allResolved) const
{
    void *funPtr = nullptr;
    if (allResolved)
    {
        funPtr = GetProcAddress(static_cast<HMODULE>(m_module), functionName);
        allResolved = allResolved && (funPtr != nullptr);
    }
    return SymbolWrapper(funPtr);
}
#else
// nothing
#endif
