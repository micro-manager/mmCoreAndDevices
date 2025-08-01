//////////////////////////////////////////////////////////////////////////////
// FILE:          MMCore.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     MMCore
//-----------------------------------------------------------------------------
// DESCRIPTION:   The interface to the MM core services.
//
// COPYRIGHT:     University of California, San Francisco, 2006-2014
//                100X Imaging Inc, www.100ximaging.com, 2008
//
// LICENSE:       This library is free software; you can redistribute it and/or
//                modify it under the terms of the GNU Lesser General Public
//                License as published by the Free Software Foundation.
//
//                You should have received a copy of the GNU Lesser General Public
//                License along with the source distribution; if not, write to
//                the Free Software Foundation, Inc., 59 Temple Place, Suite 330,
//                Boston, MA  02111-1307  USA
//
//                This file is distributed in the hope that it will be useful,
//                but WITHOUT ANY WARRANTY; without even the implied warranty
//                of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//                IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//                CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//                INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.
//
// AUTHOR:        Nenad Amodaj, nenad@amodaj.com, 06/07/2005
//
// NOTES:
//                Public methods follow slightly different naming conventions than
//                the rest of the C++ code, i.e we have:
//                   getConfiguration();
//                instead of:
//                   GetConfiguration();
//                The alternative (lowercase function names) convention is used
//                because all public methods will most likely appear in other
//                programming environments (Java or Python).

#include "CircularBuffer.h"
#include "ConfigGroup.h"
#include "Configuration.h"
#include "CoreCallback.h"
#include "CoreFeatures.h"
#include "CoreProperty.h"
#include "CoreUtils.h"
#include "DeviceManager.h"
#include "Devices/DeviceInstances.h"
#include "LogManager.h"
#include "MMCore.h"
#include "MMEventCallback.h"
#include "PluginManager.h"

#include "DeviceThreads.h"
#include "DeviceUtils.h"
#include "ImageMetadata.h"
#include "ModuleInterface.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>
#include <deque>
#include <fstream>
#include <future>
#include <map>
#include <set>
#include <sstream>
#include <thread>
#include <vector>

/*
 * Important! Read this before changing this file:
 *
 * The following (major, minor, patch) triplet is the MMCore API version. Since
 * 3.0.0, this is maintained according to the rules outlined at
 * http://semver.org/ . Briefly,
 *
 * - Increment the major version when making backward-incompatible changes
 *   (changes that will require any existing code to be modified, or that may
 *   change behavior).
 * - Increment the minor version when adding methods or functionality without
 *   breaking backward-compatibility.
 * - Increment the patch version when fixing incorrect behavior in a
 *   backward-compatible manner.
 *
 * There is no need to increment the patch number when making changes that do
 * not change behavior (such as internal refactoring).
 *
 * There is no particular correspondence between the Core API version number
 * and the device/module interface version numbers or the MMStudio application
 * version number (each version is incremented independently of each other).
 *
 * This applies to all classes exposed through MMCoreJ and pymmcore (i.e. the
 * whole of the public API of the Core), not just CMMCore.
 *
 * Because currently there is no C++ DLL build of MMCore, what we care about is
 * the backward compatibility of the Java and Python bindings. So a change that
 * requires recompilation (without source changes) of (hypothetical) C++ code
 * calling MMCore does not, by itself, require incrementing the major version,
 * provided that the resulting MMCoreJ.jar can be dropped in without
 * recompilation of client Java code.
 *
 * (Keep the 3 numbers on one line to make it easier to look at diffs when
 * merging/rebasing.)
 */
const int MMCore_versionMajor = 11, MMCore_versionMinor = 9, MMCore_versionPatch = 0;


///////////////////////////////////////////////////////////////////////////////
// CMMCore class
// -------------

/**
 * Constructor.
 * Initializes buffers and error message text. It does not load any hardware
 * devices at this point.
 */
CMMCore::CMMCore() :
   logManager_(new mm::LogManager()),
   appLogger_(logManager_->NewLogger("App")),
   coreLogger_(logManager_->NewLogger("Core")),
   everSnapped_(false),
   pollingIntervalMs_(10),
   timeoutMs_(5000),
   autoShutter_(true),
   callback_(0),
   configGroups_(0),
   properties_(0),
   externalCallback_(0),
   pixelSizeGroup_(0),
   cbuf_(0),
   pluginManager_(new CPluginManager()),
   deviceManager_(new mm::DeviceManager()),
   pPostedErrorsLock_(NULL)
{
   configGroups_ = new ConfigGroupCollection();
   pixelSizeGroup_ = new PixelSizeConfigGroup();
   pPostedErrorsLock_ = new MMThreadLock();

   InitializeErrorMessages();

   callback_ = new CoreCallback(this);

   const unsigned seqBufMegabytes = (sizeof(void*) > 4) ? 250 : 25;
   cbuf_ = new CircularBuffer(seqBufMegabytes);

   nullAffine_ = new std::vector<double>(6);
   for (int i = 0; i < 6; i++) {
      nullAffine_->at(i) = 0.0;
   }

   CreateCoreProperties();
}

/**
 * Destructor.
 *
 * Cleans up and unloads all devices. However, it is strongly recommended
 * to explicitly call reset() before destroying the CMMCore object, because
 * errors cannot be handled in the destructor.
 *
 * It is also strongly recommended to unregister any event callback
 * (registerCallback(nullptr)) before destroying the CMMCore object.
 */
CMMCore::~CMMCore()
{
   // Applications should not expect the callback notifications to be available
   // when they are already allowing the Core object to be destroyed. Disable
   // for safety.
   registerCallback(nullptr);

   try
   {
      // TODO We should attempt to continue cleanup beyond the first device
      // that throws an error.
      reset();
   }
   catch (...)
   {
      LOG_ERROR(coreLogger_) << "Exception caught in CMMCore destructor.";
   }

   delete callback_;
   delete configGroups_;
   delete properties_;
   delete cbuf_;
   delete pixelSizeGroup_;
   delete pPostedErrorsLock_;

   LOG_INFO(coreLogger_) << "Core session ended";
}

/**
 * Enable or disable the given Core feature.
 *
 * Core features control whether experimental functionality (which is subject
 * to breaking changes) is exposed, or whether stricter API usage is enforced.
 *
 * Currently switchable features:
 * - "StrictInitializationChecks" (default: disabled) When enabled, an
 *   exception is thrown when an operation requiring an initialized device is
 *   attempted on a device that is not successfully initialized. When disabled,
 *   no exception is thrown and a warning is logged (and the operation may
 *   potentially cause incorrect behavior or a crash).
 * - "ParallelDeviceInitialization" (default: enabled) When enabled, serial ports
 *   are initialized in serial order, and all other devices are in parallel, using 
 *   multiple threads, one per device module.  Early testing shows this to be 
 *   reliable, but switch this off when issues are encountered during 
 *   device initialization.
 *
 * Permanently enabled features:
 * - None so far.
 *
 * Permanently disabled features:
 * - None so far.
 *
 * @param name the feature name.
 * @param enable whether to enable or disable the feature.
 *
 * @throws CMMError if the feature name is null or unknown, or attempting to
 * disable a permanently enabled feature, or attempting to enable a permanently
 * disabled feature.
 */
void CMMCore::enableFeature(const char* name, bool enable) MMCORE_LEGACY_THROW(CMMError)
{
    if (name == nullptr)
        throw CMMError("Null feature name", MMERR_NullPointerException);
    mm::features::enableFeature(name, enable);
}

/**
 * Return whether the given Core feature is currently enabled.
 *
 * See enableFeature() for the available features.
 *
 * @param name the feature name.
 * @returns whether the feature is enabled.
 *
 * @throws CMMError if the feature name is null or unknown.
 */
bool CMMCore::isFeatureEnabled(const char* name) MMCORE_LEGACY_THROW(CMMError)
{
    if (name == nullptr)
        throw CMMError("Null feature name", MMERR_NullPointerException);
    return mm::features::isFeatureEnabled(name);
}

/**
 * Set the primary Core log file.
 *
 * @param filename The log filename. If empty or null, the primary log file is
 * disabled.
 * @param truncate Whether to truncate the log file if it already exists.
 */
void CMMCore::setPrimaryLogFile(const char* filename, bool truncate) MMCORE_LEGACY_THROW(CMMError)
{
   std::string filenameStr;
   if (filename)
      filenameStr = filename;

   logManager_->SetPrimaryLogFilename(filenameStr, truncate);
}

/**
 * Return the name of the primary Core log file.
 */
std::string CMMCore::getPrimaryLogFile() const
{
   return logManager_->GetPrimaryLogFilename();
}

/**
 * Record text message in the log file.
 */
void CMMCore::logMessage(const char* msg)
{
   appLogger_(mm::logging::LogLevelInfo, msg);
}


/**
 * Record text message in the log file.
 */
void CMMCore::logMessage(const char* msg, bool debugOnly)
{
   appLogger_(debugOnly ? mm::logging::LogLevelDebug :
         mm::logging::LogLevelInfo, msg);
}


/**
 * Enable or disable logging of debug messages.
 * @param enable   if set to true, debug messages will be recorded in the log file
 */
void CMMCore::enableDebugLog(bool enable)
{
   logManager_->SetPrimaryLogLevel(enable ? mm::logging::LogLevelTrace :
         mm::logging::LogLevelInfo);
}

/**
 * Indicates if logging of debug messages is enabled
 */
bool CMMCore::debugLogEnabled()
{
   return (logManager_->GetPrimaryLogLevel() < mm::logging::LogLevelInfo);
}

/**
 * Enables or disables log message display on the standard console.
 * @param enable     if set to true, log file messages will be echoed on the stderr.
 */
void CMMCore::enableStderrLog(bool enable)
{
   logManager_->SetUseStdErr(enable);
}

/**
 * Indicates whether logging output goes to stdErr
 */
bool CMMCore::stderrLogEnabled()
{
   return logManager_->IsUsingStdErr();
}


/**
 * Start capturing logging output into an additional file.
 *
 * @param filename The filename to which the log will be captured
 * @param enableDebug Whether to include debug logging (regardless of whether
 * debug logging is enabled for the primary log).
 * @param truncate If false, append to the file.
 * @param synchronous If true, enable synchronous logging for this file
 * (logging calls will not return until the output is written to the file,
 * facilitating the debugging of crashes in some cases, but with a performance
 * cost).
 * @returns A handle required when calling stopSecondaryLogFile().
 */
int CMMCore::startSecondaryLogFile(const char* filename, bool enableDebug,
      bool truncate, bool synchronous) MMCORE_LEGACY_THROW(CMMError)
{
   if (!filename)
      throw CMMError("Filename is null");

   using namespace mm::logging;
   typedef mm::LogManager::LogFileHandle LogFileHandle;

   LogFileHandle handle = logManager_->AddSecondaryLogFile(
            (enableDebug ? LogLevelTrace : LogLevelInfo),
            filename, truncate,
            (synchronous ? SinkModeSynchronous : SinkModeAsynchronous));
   return static_cast<int>(handle);
}


/**
 * Stop capturing logging output into an additional file.
 *
 * @param handle The secondary log handle returned by startSecondaryLogFile().
 */
void CMMCore::stopSecondaryLogFile(int handle) MMCORE_LEGACY_THROW(CMMError)
{
   typedef mm::LogManager::LogFileHandle LogFileHandle;
   LogFileHandle h = static_cast<LogFileHandle>(handle);
   logManager_->RemoveSecondaryLogFile(h);
}

/**
 * Displays core version.
 */
std::string CMMCore::getVersionInfo() const
{
   std::ostringstream txt;
   std::string debug;
   txt << "MMCore version " << MMCore_versionMajor << "." << MMCore_versionMinor << "." << MMCore_versionPatch;
   return txt.str();
}

/** Returns the MMCore major version number. */
int CMMCore::getMMCoreVersionMajor() { return MMCore_versionMajor; }

/** Returns the MMCore minor version number. */
int CMMCore::getMMCoreVersionMinor() { return MMCore_versionMinor; }

/** Returns the MMCore patch version number. */
int CMMCore::getMMCoreVersionPatch() { return MMCore_versionPatch; }

/**
 * Get available devices from the specified device library.
 */
std::vector<std::string>
CMMCore::getAvailableDevices(const char* moduleName) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<LoadedDeviceAdapter> module =
      pluginManager_->GetDeviceAdapter(moduleName);
   return module->GetAvailableDeviceNames();
}

/**
 * Get descriptions for available devices from the specified library.
 */
std::vector<std::string>
CMMCore::getAvailableDeviceDescriptions(const char* moduleName) MMCORE_LEGACY_THROW(CMMError)
{
   // XXX It is a little silly that we return the list of descriptions, rather
   // than provide access to the description of each device.
   std::shared_ptr<LoadedDeviceAdapter> module =
      pluginManager_->GetDeviceAdapter(moduleName);
   std::vector<std::string> names = module->GetAvailableDeviceNames();
   std::vector<std::string> descriptions;
   descriptions.reserve(names.size());
   for (std::vector<std::string>::const_iterator
         it = names.begin(), end = names.end(); it != end; ++it)
   {
      descriptions.push_back(module->GetDeviceDescription(*it));
   }
   return descriptions;
}

/**
 * Get type information for available devices from the specified library.
 */
std::vector<long>
CMMCore::getAvailableDeviceTypes(const char* moduleName) MMCORE_LEGACY_THROW(CMMError)
{
   // XXX It is a little silly that we return the list of types, rather than
   // provide access to the type of each device.
   std::shared_ptr<LoadedDeviceAdapter> module =
      pluginManager_->GetDeviceAdapter(moduleName);
   std::vector<std::string> names = module->GetAvailableDeviceNames();
   std::vector<long> types;
   types.reserve(names.size());
   for (std::vector<std::string>::const_iterator
         it = names.begin(), end = names.end(); it != end; ++it)
   {
      MM::DeviceType devType = module->GetAdvertisedDeviceType(*it);
      types.push_back(static_cast<long>(devType));
   }
   return types;
}

/**
 * Returns the module and device interface versions.
 */
std::string CMMCore::getAPIVersionInfo() const
{
   std::ostringstream txt;
   txt << "Device API version " << DEVICE_INTERFACE_VERSION << ", " << "Module API version " << MODULE_INTERFACE_VERSION;
   return txt.str();
}

/** Returns the MMDevice module interface version number. */
int CMMCore::getMMDeviceModuleInterfaceVersion() { return MODULE_INTERFACE_VERSION; }

/** Returns the MMDevice device interface version number. */
int CMMCore::getMMDeviceDeviceInterfaceVersion() { return DEVICE_INTERFACE_VERSION; }

/**
 * Returns the entire system state, i.e. the collection of all property values from all devices.
 *
 * For legacy reasons, this function does not throw an exception if there is an
 * error. If there is an error, properties may be missing from the return
 * value.
 *
 * @return Configuration object containing a collection of device-property-value triplets
 */
Configuration CMMCore::getSystemState()
{
   Configuration config;
   std::vector<std::string> devices = deviceManager_->GetDeviceList();
   for (std::vector<std::string>::const_iterator i = devices.begin(), dend = devices.end(); i != dend; ++i)
   {
      std::shared_ptr<DeviceInstance> pDev = deviceManager_->GetDevice(*i);
      mm::DeviceModuleLockGuard guard(pDev);
      std::vector<std::string> propertyNames = pDev->GetPropertyNames();
      for (std::vector<std::string>::const_iterator it = propertyNames.begin(), end = propertyNames.end();
            it != end; ++it)
      {
         std::string val;
         try
         {
            val = pDev->GetProperty(*it);
         }
         catch (const CMMError&)
         {
            // XXX BUG This should not be ignored, but the interface does not
            // allow throwing from this function. Keeping old behavior for now.
         }

         bool readOnly = false;
         try
         {
            readOnly = pDev->GetPropertyReadOnly(it->c_str());
         }
         catch (const CMMError&)
         {
            // XXX BUG This should not be ignored, but the interface does not
            // allow throwing from this function. Keeping old behavior for now.
         }
         config.addSetting(PropertySetting(i->c_str(), it->c_str(), val.c_str(), readOnly));
      }
   }

   // add core properties
   std::vector<std::string> coreProps = properties_->GetNames();
   for (unsigned i=0; i < coreProps.size(); i++)
   {
      std::string name = coreProps[i];
      std::string val = properties_->Get(name.c_str());
      config.addSetting(PropertySetting(MM::g_Keyword_CoreDevice, name.c_str(), val.c_str(), properties_->IsReadOnly(name.c_str())));
   }

   return config;
}

/**
 * Returns the entire system state, i.e. the collection of all property values from all devices.
 * This method will return cached values instead of querying each device
 * @return  Configuration object containing a collection of device-property-value triplets
 */
Configuration CMMCore::getSystemStateCache() const
{
   MMThreadGuard scg(stateCacheLock_);
   return stateCache_;
}

/**
 * Returns a partial state of the system, only for devices included in the
 * specified configuration.
 */
Configuration CMMCore::getConfigState(const char* group, const char* config) MMCORE_LEGACY_THROW(CMMError)
{
   Configuration cfgData = getConfigData(group, config);

   Configuration state;
   for (size_t i=0; i < cfgData.size(); i++)
   {
      PropertySetting cs = cfgData.getSetting(i); // config setting
      std::string value = getProperty(cs.getDeviceLabel().c_str(), cs.getPropertyName().c_str());
      PropertySetting ss(cs.getDeviceLabel().c_str(), cs.getPropertyName().c_str(), value.c_str()); // state setting
      state.addSetting(ss);
   }
   return state;
}


/**
 * Returns the partial state of the system, only for the devices included in the
 * specified group. It will create a union of all devices referenced in a group.
 */
Configuration CMMCore::getConfigGroupState(const char* group) MMCORE_LEGACY_THROW(CMMError)
{
   return getConfigGroupState(group, false);
}

/**
 * Returns the partial state of the system cache, only for the devices included in the
 * specified group. It will create a union of all devices referenced in a group.
 */
Configuration CMMCore::getConfigGroupStateFromCache(const char* group) MMCORE_LEGACY_THROW(CMMError)
{
   return getConfigGroupState(group, true);
}

/**
 * Returns the partial state of the system, only for the devices included in the
 * specified group. It will create a union of all devices referenced in a group.
 */
Configuration CMMCore::getConfigGroupState(const char* group, bool fromCache) MMCORE_LEGACY_THROW(CMMError)
{
   CheckConfigGroupName(group);

   std::vector<std::string> allPresets =
      configGroups_->GetAvailableConfigs(group);

   Configuration state;

   // Loop over every property that appears in every preset, and collect the
   // value (from cache or from devices).
   for (std::vector<std::string>::const_iterator
         it = allPresets.begin(), end = allPresets.end(); it != end; ++it)
   {
      Configuration preset = getConfigData(group, it->c_str());

      for (size_t i = 0; i < preset.size(); i++)
      {
         PropertySetting cs = preset.getSetting(i);
         std::string deviceLabel = cs.getDeviceLabel();
         std::string propertyName = cs.getPropertyName();

         // Skip properties that we have already added.
         if (!state.isPropertyIncluded(deviceLabel.c_str(),
                  propertyName.c_str()))
         {
            std::string value;
            if (fromCache)
            {
               value = getPropertyFromCache(deviceLabel.c_str(),
                     propertyName.c_str());
            }
            else
            {
               value = getProperty(deviceLabel.c_str(),
                     propertyName.c_str());
            }

            PropertySetting ss(deviceLabel.c_str(), propertyName.c_str(),
                  value.c_str());
            state.addSetting(ss);
         }
      }
   }
   return state;
}

/**
 * Sets all properties contained in the Configuration object.
 * The procedure will attempt to set each property it encounters, but won't stop
 * if any of the properties fail or if the requested device is not present. It will
 * just quietly continue.
 *
 * @param conf    the configuration object representing the desired system state
 */
void CMMCore::setSystemState(const Configuration& conf)
{
   for (unsigned i=0; i<conf.size(); i++)
   {
      PropertySetting s = conf.getSetting(i);
      if (!s.getReadOnly())
      {
         try
         {
            setProperty(s.getDeviceLabel().c_str(), s.getPropertyName().c_str(), s.getPropertyValue().c_str());
         }
         catch (CMMError&)
         {
            // Do not give up yet.
         }
      }
   }
   // TODO Should throw if any of the property setting failed.

   updateSystemStateCache();
}

/**
 * Return the current device adapter search paths.
 */
std::vector<std::string> CMMCore::getDeviceAdapterSearchPaths()
{
   return pluginManager_->GetSearchPaths();
}

/**
 * Set the device adapter search paths.
 *
 * Upon subsequent attempts to load device adapters, these paths (and only
 * these paths) will be searched. Calling this function has no effect on device
 * adapters that have already been loaded.
 *
 * If you want to simply add to the list of paths, you must first retrieve the
 * current paths by calling getDeviceAdapterSearchPaths().
 *
 * @param paths   the device adapter search paths
 */
void CMMCore::setDeviceAdapterSearchPaths(const std::vector<std::string>& paths)
{
   pluginManager_->SetSearchPaths(paths.begin(), paths.end());
}

/**
 * Return the names of discoverable device adapters.
 *
 * Note that this list is constructed based on filename matching in the current
 * search paths. This method does not check whether the files are valid and
 * compatible device adapters.
 */
std::vector<std::string> CMMCore::getDeviceAdapterNames() MMCORE_LEGACY_THROW(CMMError)
{
   return pluginManager_->GetAvailableDeviceAdapters();
}

/**
 * Loads a device from the plugin library.
 * @param label    assigned name for the device during the core session
 * @param moduleName  the name of the device adapter module (short name, not full file name)
 * @param deviceName   the name of the device. The name must correspond to one of the names recognized
 *                 by the specific plugin library.
 */
void CMMCore::loadDevice(const char* label, const char* moduleName, const char* deviceName) MMCORE_LEGACY_THROW(CMMError)
{
   CheckDeviceLabel(label);
   if (!moduleName)
      throw CMMError("Null device adapter name");
   if (!deviceName)
      throw CMMError("Null device name");

   // Logger for logging from device adapter code
   mm::logging::Logger deviceLogger =
      logManager_->NewLogger("dev:" + std::string(label));
   // Logger for logging related to the device, by us the Core
   mm::logging::Logger coreLogger =
      logManager_->NewLogger("Core:dev:" + std::string(label));

   LOG_DEBUG(coreLogger_) << "Will load device " << deviceName <<
      " from " << moduleName;

   std::shared_ptr<LoadedDeviceAdapter> module =
      pluginManager_->GetDeviceAdapter(moduleName);
   std::shared_ptr<DeviceInstance> pDevice =
      deviceManager_->LoadDevice(module, deviceName, label, this,
            deviceLogger, coreLogger);
   pDevice->SetCallback(callback_);

   LOG_INFO(coreLogger_) << "Did load device " << deviceName <<
      " from " << moduleName << "; label = " << label;
}

void CMMCore::assignDefaultRole(std::shared_ptr<DeviceInstance> pDevice)
{
   // default special roles for particular devices
   // The roles which are assigned at the load time will make sense for a simple
   // configuration. More complicated configurations will typically override default settings.
   mm::DeviceModuleLockGuard guard(pDevice);
   const std::string label(pDevice->GetLabel());

   switch(pDevice->GetType())
   {
      case MM::CameraDevice:
         currentCameraDevice_ =
            std::static_pointer_cast<CameraInstance>(pDevice);
         LOG_INFO(coreLogger_) << "Default camera set to " << label;
         break;

      case MM::ShutterDevice:
         currentShutterDevice_ =
            std::static_pointer_cast<ShutterInstance>(pDevice);
         LOG_INFO(coreLogger_) << "Default shutter set to " << label;
         break;

      case MM::XYStageDevice:
         currentXYStageDevice_ =
            std::static_pointer_cast<XYStageInstance>(pDevice);
         LOG_INFO(coreLogger_) << "Default xy stage set to " << label;
         break;

      case MM::AutoFocusDevice:
         currentAutofocusDevice_ =
            std::static_pointer_cast<AutoFocusInstance>(pDevice);
         LOG_INFO(coreLogger_) << "Default autofocus set to " << label;
         break;

      case MM::SLMDevice:
         currentSLMDevice_ =
            std::static_pointer_cast<SLMInstance>(pDevice);
         LOG_INFO(coreLogger_) << "Default SLM set to " << label;
         break;

      case MM::GalvoDevice:
         currentGalvoDevice_ =
            std::static_pointer_cast<GalvoInstance>(pDevice);
         LOG_INFO(coreLogger_) << "Default galvo set to " << label;
         break;

      default:
         // no action on unrecognized device
         break;
   }
}

void CMMCore::removeDeviceRole(std::shared_ptr<DeviceInstance> pDev) {
   if (pDev == currentCameraDevice_.lock()) {
      setCameraDevice("");
   } else if (pDev == currentShutterDevice_.lock()) {
      setShutterDevice("");
   } else if (pDev == currentXYStageDevice_.lock()) {
      setXYStageDevice("");
   } else if (pDev == currentFocusDevice_.lock()) {
      setFocusDevice("");
   } else if (pDev == currentAutofocusDevice_.lock()) {
      setAutoFocusDevice("");
   } else if (pDev == currentImageProcessor_.lock()) {
      setImageProcessorDevice("");
   } else if (pDev == currentGalvoDevice_.lock()) {
      setGalvoDevice("");
   } else if (pDev == currentSLMDevice_.lock()) {
      setSLMDevice("");
   }
}

void CMMCore::removeAllDeviceRoles() {
   setCameraDevice("");
   setShutterDevice("");
   setXYStageDevice("");
   setFocusDevice("");
   setAutoFocusDevice("");
   setImageProcessorDevice("");
   setGalvoDevice("");
   setSLMDevice("");
}

/**
 * Unloads the device from the core and adjusts all configuration data.
 */
void CMMCore::unloadDevice(const char* label///< the name of the device to unload
                           ) MMCORE_LEGACY_THROW(CMMError)
{
   // "Core" cannot be unloaded.
   if (label != nullptr && std::string(label) == MM::g_Keyword_CoreDevice)
   {
      throw CMMError("Cannot unload " + ToQuotedString("Core"));
   }

   std::shared_ptr<DeviceInstance> pDevice = deviceManager_->GetDevice(label);

   try {
      removeDeviceRole(pDevice);

      mm::DeviceModuleLockGuard guard(pDevice);
      LOG_DEBUG(coreLogger_) << "Will unload device " << label;
      deviceManager_->UnloadDevice(pDevice);
      LOG_DEBUG(coreLogger_) << "Did unload device " << label;
      
      updateCoreProperties();
   }
   catch (CMMError& err) {
      logError("MMCore::unloadDevice", err.getMsg().c_str());
      throw;
   }
}


/**
 * Unloads all devices from the core and resets all configuration data.
 *
 * This function is not thread safe.
 */
void CMMCore::unloadAllDevices() MMCORE_LEGACY_THROW(CMMError)
{
   try {
      removeAllDeviceRoles();

      configGroups_->Clear();
      updateAllowedChannelGroups();

      // clear pixel size configurations
      if (!pixelSizeGroup_->IsEmpty())
      {
         std::vector<std::string> pixelSizes = pixelSizeGroup_->GetAvailable();
         for (std::vector<std::string>::iterator it = pixelSizes.begin();
               it != pixelSizes.end(); it++)
         {
            pixelSizeGroup_->Delete((*it).c_str());
         }
      }

      LOG_DEBUG(coreLogger_) << "Will unload all devices";
      deviceManager_->UnloadAllDevices();
      LOG_INFO(coreLogger_) << "Did unload all devices";

	   properties_->Refresh();

      // The system config has "changed" (to "(none)").
      // But don't notify if we will proceed to load a new config.
      if (externalCallback_ && !isLoadingSystemConfiguration_)
      {
         externalCallback_->onSystemConfigurationLoaded();
      }
   }
   catch (CMMError& err) {
      logError("MMCore::unloadAllDevices", err.getMsg().c_str());

      // The config has "changed" even in this case.
      if (externalCallback_ && !isLoadingSystemConfiguration_)
      {
         externalCallback_->onSystemConfigurationLoaded();
      }

      throw;
   }
}

/**
 * Unloads all devices from the core, clears all configuration data.
 */
void CMMCore::reset() MMCORE_LEGACY_THROW(CMMError)
{
   try
   {
   // before unloading everything try to apply shutdown configuration
   if (isConfigDefined(MM::g_CFGGroup_System, MM::g_CFGGroup_System_Shutdown))
      this->setConfig(MM::g_CFGGroup_System, MM::g_CFGGroup_System_Shutdown);
   }
   catch(...)
   {
	   logError("MMCore::reset", "problem setting System Shutdown configuration");
   }


   // of course one reason to reset is that some device is not configured correctly,
   // so we need to handle any exception thrown from here
   try
   {
      waitForSystem();
   }
   catch (CMMError& ) {}

   // unload devices
   unloadAllDevices();

   properties_->Refresh();

   LOG_INFO(coreLogger_) << "System reset";
}


/**
 * Calls Initialize() method for each loaded device.
 * Parallel implemnetation should be faster
 */
void CMMCore::initializeAllDevices() MMCORE_LEGACY_THROW(CMMError)
{
   if (this->isFeatureEnabled("ParallelDeviceInitialization"))
   {
      initializeAllDevicesParallel();
   }
   else
   {
      initializeAllDevicesSerial();
   }
}


/**
 * Calls Initialize() method for each loaded device.
 * This method also initialized allowed values for core properties, based
 * on the collection of loaded devices.
 */
void CMMCore::initializeAllDevicesSerial() MMCORE_LEGACY_THROW(CMMError)
{
   std::vector<std::string> devices = deviceManager_->GetDeviceList();
   LOG_INFO(coreLogger_) << "Will initialize " << devices.size() << " devices (serially)";

   for (size_t i = 0; i < devices.size(); i++)
   {
      std::shared_ptr<DeviceInstance> pDevice;
      try {
         pDevice = deviceManager_->GetDevice(devices[i]);
      }
      catch (CMMError& err) {
         logError(devices[i].c_str(), err.getMsg().c_str());
         throw;
      }
      mm::DeviceModuleLockGuard guard(pDevice);
      LOG_INFO(coreLogger_) << "Will initialize device " << devices[i];
      pDevice->Initialize();
      LOG_INFO(coreLogger_) << "Did initialize device " << devices[i];

      assignDefaultRole(pDevice);
   }

   LOG_INFO(coreLogger_) << "Finished initializing " << devices.size() << " devices";

   updateCoreProperties();
}


/**
 * Calls Initialize() method for each loaded device.
 * This implementation initializes devices on separate threads, one per device module (adapter).
 * This method also initializes allowed values for core properties, based
 * on the collection of loaded devices.
 */
void CMMCore::initializeAllDevicesParallel() MMCORE_LEGACY_THROW(CMMError)
{
   std::vector<std::string> devices = deviceManager_->GetDeviceList();
   LOG_INFO(coreLogger_) << "Will initialize " << devices.size() << " devices (in parallel)";
   
   std::map<std::shared_ptr<LoadedDeviceAdapter>, std::vector<std::pair<std::shared_ptr<DeviceInstance>, std::string>>> moduleMap;
   std::vector<std::shared_ptr<DeviceInstance>> ports;

   // first round, collect all DeviceAdapters
   for (size_t i = 0; i < devices.size(); i++)
   {
      std::shared_ptr<DeviceInstance> pDevice;
      try {
         pDevice = deviceManager_->GetDevice(devices[i]);
      }
      catch (CMMError& err) {
         logError(devices[i].c_str(), err.getMsg().c_str());
         throw;
      }
      if (pDevice->GetType() == MM::SerialDevice)
      {
         ports.push_back(pDevice);
      }
      else {
         std::shared_ptr<LoadedDeviceAdapter> pAdapter;
         pAdapter = pDevice->GetAdapterModule();

         if (moduleMap.find(pAdapter) == moduleMap.end())
         {
            std::vector<std::pair<std::shared_ptr<DeviceInstance>, std::string>> pDevices;
            pDevices.push_back(make_pair(pDevice, devices[i]));
            moduleMap.insert({ pAdapter, pDevices });
         }
         else
         {
            moduleMap.find(pAdapter)->second.push_back(make_pair(pDevice, devices[i]));
         }
      }
   }

   // Initialize ports first.  This should be fast, so no need to go parallel (also could not hurt really)
   for (std::shared_ptr<DeviceInstance> pPort : ports)
   {
      mm::DeviceModuleLockGuard guard(pPort);
      LOG_INFO(coreLogger_) << "Will initialize device " << pPort->GetLabel();
      pPort->Initialize();
      LOG_INFO(coreLogger_) << "Did initialize device " << pPort->GetLabel();
   }

   // second round, spin up threads to initialize non-port devices, one thread per module
   std::vector<std::future<int>> futures;
   for (auto& moduleDevices : moduleMap) {
      auto f = std::async(std::launch::async, &CMMCore::initializeVectorOfDevices, this, moduleDevices.second);
      futures.push_back(std::move(f));
   }

   // Make sure we wait for all futures even if one or more fails, so that we
   // handle all exceptions. Otherwise futures return by std::async may try to
   // throw from their destructor, which will call std::terminate().
   std::exception_ptr pex;
   for (auto& fut : futures) {
      try {
         fut.get();
      } catch (const std::exception&) {
         if (pex) {
            // Ignore second and subsequent exceptions
         } else {
            pex = std::current_exception();
         }
      }
   }
   if (pex) {
      std::rethrow_exception(pex);
   }

   // assign default roles syncronously
   for (auto& moduleDevices : moduleMap) {
      for (auto& deviceLabel : moduleDevices.second) {
         assignDefaultRole(deviceLabel.first);
      }
   }
   LOG_INFO(coreLogger_) << "Finished initializing " << devices.size() << " devices";

   updateCoreProperties();
   // not sure if this cleanup is needed, but should not hurt:
   moduleMap.clear();
   ports.clear();
}


/**
 * This helper function is executed by a single thread, allowing initializeAllDevices to operate multi-threaded.
 * All devices are supposed to originate from the same device adapter
 */
int CMMCore::initializeVectorOfDevices(std::vector<std::pair<std::shared_ptr<DeviceInstance>, std::string>> devicesLabels) {
   for (auto& deviceLabel : devicesLabels) {
      mm::DeviceModuleLockGuard guard(deviceLabel.first);
      LOG_INFO(coreLogger_) << "Will initialize device " << deviceLabel.second;
      deviceLabel.first->Initialize();
      LOG_INFO(coreLogger_) << "Did initialize device " << deviceLabel.second;
   }
   return DEVICE_OK;
}

/**
 * Updates CoreProperties (currently all Core properties are 
 * devices types) with the loaded hardware.
 * After this call, each of the Core-Device properties 
 * will be populated with the currently loaded devices 
 * of that type
 */
void CMMCore::updateCoreProperties() MMCORE_LEGACY_THROW(CMMError)
{
   updateCoreProperty(MM::g_Keyword_CoreCamera, MM::CameraDevice);
   updateCoreProperty(MM::g_Keyword_CoreShutter, MM::ShutterDevice);
   updateCoreProperty(MM::g_Keyword_CoreFocus,MM::StageDevice);
   updateCoreProperty(MM::g_Keyword_CoreXYStage,MM::XYStageDevice);
   updateCoreProperty(MM::g_Keyword_CoreAutoFocus,MM::AutoFocusDevice);
   updateCoreProperty(MM::g_Keyword_CoreImageProcessor,MM::ImageProcessorDevice);
   updateCoreProperty(MM::g_Keyword_CoreSLM,MM::SLMDevice);
   updateCoreProperty(MM::g_Keyword_CoreGalvo,MM::GalvoDevice);

   properties_->Refresh();
}

void CMMCore::updateCoreProperty(const char* propName, MM::DeviceType devType) MMCORE_LEGACY_THROW(CMMError)
{
   CheckPropertyName(propName);

   std::vector<std::string> devices = getLoadedDevicesOfType(devType);
   devices.push_back(""); // add empty value
   properties_->ClearAllowedValues(propName);
   for (size_t i=0; i<devices.size(); i++)
      properties_->AddAllowedValue(propName, devices[i].c_str());
}

/**
 * Initializes specific device.
 *
 * @param label   the device label
 */
void CMMCore::initializeDevice(const char* label ///< the device to initialize
                               ) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<DeviceInstance> pDevice = deviceManager_->GetDevice(label);

   mm::DeviceModuleLockGuard guard(pDevice);

   LOG_INFO(coreLogger_) << "Will initialize device " << label;
   pDevice->Initialize();
   LOG_INFO(coreLogger_) << "Did initialize device " << label;

   updateCoreProperties();
}


/**
 * Queries the initialization state of the given device.
 *
 * @param label the device label
 */
DeviceInitializationState
CMMCore::getDeviceInitializationState(const char* label) const MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<DeviceInstance> pDevice = deviceManager_->GetDevice(label);

   mm::DeviceModuleLockGuard guard(pDevice);
   if (pDevice->IsInitialized())
   {
      return DeviceInitializationState::InitializedSuccessfully;
   }
   if (pDevice->HasInitializationBeenAttempted())
   {
      return DeviceInitializationState::InitializationFailed;
   }
   return DeviceInitializationState::Uninitialized;
}



/**
 * Updates the state of the entire hardware.
 */
void CMMCore::updateSystemStateCache()
{
   LOG_DEBUG(coreLogger_) << "Will update system state cache";
   Configuration wk = getSystemState();
   {
      MMThreadGuard scg(stateCacheLock_);
      stateCache_ = wk;
   }
   LOG_INFO(coreLogger_) << "Did update system state cache";
}

/**
 * Returns device type.
 */
MM::DeviceType CMMCore::getDeviceType(const char* label) MMCORE_LEGACY_THROW(CMMError)
{
   if (IsCoreDeviceLabel(label))
      return MM::CoreDevice;

   std::shared_ptr<DeviceInstance> pDevice = deviceManager_->GetDevice(label);
   return pDevice->GetType();
}


/**
 * Returns device library (aka module, device adapter) name.
 */
std::string CMMCore::getDeviceLibrary(const char* label) MMCORE_LEGACY_THROW(CMMError)
{
   if (IsCoreDeviceLabel(label))
      return "";

   std::shared_ptr<DeviceInstance> pDevice = deviceManager_->GetDevice(label);

   mm::DeviceModuleLockGuard guard(pDevice);
   return pDevice->GetAdapterModule()->GetName();
}

/**
 * Forcefully unload a library. Experimental. Don't use.
 */
void CMMCore::unloadLibrary(const char* moduleName) MMCORE_LEGACY_THROW(CMMError)
{
  	if (moduleName == 0)
      throw CMMError(errorText_[MMERR_NullPointerException],  MMERR_NullPointerException);

   try {
      std::vector<std::string> devices = deviceManager_->GetDeviceList();
      std::vector<std::string>::reverse_iterator it;
      for (it=devices.rbegin(); it != devices.rend(); it++)
      {
         std::shared_ptr<DeviceInstance> pDev = deviceManager_->GetDevice(*it);
         mm::DeviceModuleLockGuard guard(pDev);

         if (pDev->GetAdapterModule()->GetName() == moduleName)
         {
            try {
               unloadDevice(pDev->GetLabel().c_str());
            } catch (CMMError& /*e*/) {} // ignore error; device may already have been unloaded
         }
      }
      pluginManager_->UnloadPluginLibrary(moduleName);
   }
   catch (CMMError& /* err */)
   {
      logError(moduleName, "Library updating failed.");
      throw;
   }
}

/**
 * Returns device name for a given device label.
 * "Name" is determined by the library and is immutable, while "label" is
 * user assigned and represents a high-level handle to a device.
 */
std::string CMMCore::getDeviceName(const char* label) MMCORE_LEGACY_THROW(CMMError)
{
   if (IsCoreDeviceLabel(label))
      return "Core";
   std::shared_ptr<DeviceInstance> pDevice = deviceManager_->GetDevice(label);

   mm::DeviceModuleLockGuard guard(pDevice);
   return pDevice->GetName();
}

/**
 * Returns parent device.
 */
std::string CMMCore::getParentLabel(const char* label) MMCORE_LEGACY_THROW(CMMError)
{
   if (IsCoreDeviceLabel(label))
      // XXX Should be a throw
      return "";
   std::shared_ptr<DeviceInstance> device = deviceManager_->GetDevice(label);
   mm::DeviceModuleLockGuard guard(device);
   return device->GetParentID();
}

/**
 * Sets parent device label
 */
void CMMCore::setParentLabel(const char* label, const char* parentLabel) MMCORE_LEGACY_THROW(CMMError)
{
   if (IsCoreDeviceLabel(label))
      // XXX Should be a throw
      return; // core can't have parent ID
   std::shared_ptr<DeviceInstance> pDev = deviceManager_->GetDevice(label);
   if (parentLabel && std::string(parentLabel).empty()) {
      // Empty label is acceptable, meaning no parent
   }
   else {
      // Note that the parent device is not checked for existence
      // XXX Should we require that the parent device exist?
      CheckDeviceLabel(parentLabel);
   }

   mm::DeviceModuleLockGuard guard(pDev);
   pDev->SetParentID(parentLabel);
}


/**
 * Returns description text for a given device label.
 * "Description" is determined by the library and is immutable.
 */
std::string CMMCore::getDeviceDescription(const char* label) MMCORE_LEGACY_THROW(CMMError)
{
   if (IsCoreDeviceLabel(label))
      return "Core device";
   std::shared_ptr<DeviceInstance> pDevice = deviceManager_->GetDevice(label);

   mm::DeviceModuleLockGuard guard(pDevice);
   return pDevice->GetDescription();
}


/**
 * Reports action delay in milliseconds for the specific device.
 * The delay is used in the synchronization process to ensure that
 * the action is performed, without polling.
 * Value of "0" means that action is either blocking or that polling
 * of device status is required.
 * Some devices ignore this setting.
 *
 * @return the delay time in milliseconds
 * @param label    the device label
 */
double CMMCore::getDeviceDelayMs(const char* label) MMCORE_LEGACY_THROW(CMMError)
{
   if (IsCoreDeviceLabel(label))
      return 0.0;
   std::shared_ptr<DeviceInstance> pDevice = deviceManager_->GetDevice(label);

   mm::DeviceModuleLockGuard guard(pDevice);
   return pDevice->GetDelayMs();
}


/**
 * Overrides the built-in value for the action delay.
 * Some devices ignore this setting.
 *
 * @param label      the device label
 * @param delayMs    the desired delay in milliseconds
 */
void CMMCore::setDeviceDelayMs(const char* label, double delayMs) MMCORE_LEGACY_THROW(CMMError)
{
   if (IsCoreDeviceLabel(label))
      return; // ignore
   std::shared_ptr<DeviceInstance> pDevice = deviceManager_->GetDevice(label);

   mm::DeviceModuleLockGuard guard(pDevice);
   pDevice->SetDelayMs(delayMs);
}

/**
 * Signals if the device will use the delay setting or not.
 *
 * @param label    the device label
 * @return true if the device uses a delay
 */
bool CMMCore::usesDeviceDelay(const char* label) MMCORE_LEGACY_THROW(CMMError)
{
   if (IsCoreDeviceLabel(label))
      return false;
   std::shared_ptr<DeviceInstance> pDevice = deviceManager_->GetDevice(label);

   mm::DeviceModuleLockGuard guard(pDevice);
   return pDevice->UsesDelay();
}

/**
 * Checks the busy status of the specific device.
 * @param label the device label
 * @return true if the device is busy
 */
bool CMMCore::deviceBusy(const char* label) MMCORE_LEGACY_THROW(CMMError)
{
   if (IsCoreDeviceLabel(label))
      return false;
   std::shared_ptr<DeviceInstance> pDevice = deviceManager_->GetDevice(label);

   mm::DeviceModuleLockGuard guard(pDevice);
   return pDevice->Busy();
}


/**
 * Waits (blocks the calling thread) for specified time in milliseconds.
 * @param intervalMs the time to sleep in milliseconds
 */
void CMMCore::sleep(double intervalMs) const
{
	CDeviceUtils::SleepMs( (long)(0.5 + intervalMs));
}


/**
 * Waits (blocks the calling thread) until the specified device becomes
 * non-busy.
 * @param label   the device label
 */
void CMMCore::waitForDevice(const char* label) MMCORE_LEGACY_THROW(CMMError)
{
   if (IsCoreDeviceLabel(label))
      return; // core property commands always block - no need to poll
   std::shared_ptr<DeviceInstance> pDevice = deviceManager_->GetDevice(label);

   waitForDevice(pDevice);
}


/**
 * Waits (blocks the calling thread) until the specified device becomes
 * @param pDev   the device instance
 */
void CMMCore::waitForDevice(std::shared_ptr<DeviceInstance> pDev) MMCORE_LEGACY_THROW(CMMError)
{
   LOG_DEBUG(coreLogger_) << "Waiting for device " << pDev->GetLabel() << "...";

   auto now = std::chrono::steady_clock::now();
   auto timeout = std::chrono::duration<long long, std::milli>(timeoutMs_);
   auto deadline = now + timeout;

   while (true)
   {
      {
         mm::DeviceModuleLockGuard guard(pDev);
         if (!pDev->Busy())
         {
            break;
         }
      }

      if (std::chrono::steady_clock::now() > deadline)
      {
         std::string label = pDev->GetLabel();
         std::ostringstream mez;
         mez << "wait timed out after " << timeoutMs_ << " ms. ";
         logError(label.c_str(), mez.str().c_str());
         throw CMMError("Wait for device " + ToQuotedString(label) + " timed out after " +
               ToString(timeoutMs_) + "ms",
               MMERR_DevicePollingTimeout);
      }

     sleep(pollingIntervalMs_);
   }
   LOG_DEBUG(coreLogger_) << "Finished waiting for device " << pDev->GetLabel();
}

/**
 * Checks the busy status of the entire system. The system will report busy if any
 * of the devices is busy.
 * @return status (true on busy)
 */
bool CMMCore::systemBusy() MMCORE_LEGACY_THROW(CMMError)
{
   return deviceTypeBusy(MM::AnyType);
}


/**
 * Blocks until all devices in the system become ready (not-busy).
 */
void CMMCore::waitForSystem() MMCORE_LEGACY_THROW(CMMError)
{
   waitForDeviceType(MM::AnyType);
}


/**
 * Checks the busy status for all devices of the specific type.
 * The system will report busy if any of the devices of the specified type are busy.
 *
 * @return true on busy
 * @param devType   a constant specifying the device type
 */
bool CMMCore::deviceTypeBusy(MM::DeviceType devType) MMCORE_LEGACY_THROW(CMMError)
{
   std::vector<std::string> devices = deviceManager_->GetDeviceList(devType);
   for (size_t i=0; i<devices.size(); i++)
   {
      try {
         std::shared_ptr<DeviceInstance> pDevice =
            deviceManager_->GetDevice(devices[i]);
         mm::DeviceModuleLockGuard guard(pDevice);
         if (pDevice->Busy())
            return true;
      }
      catch (...) {
         // trap all exceptions
         assert(!"Plugin manager can't access device it reported as available.");
      }
   }
   return false;
}


/**
 * Blocks until all devices of the specific type become ready (not-busy).
 * @param devType    a constant specifying the device type
 */
void CMMCore::waitForDeviceType(MM::DeviceType devType) MMCORE_LEGACY_THROW(CMMError)
{
   std::vector<std::string> devices = deviceManager_->GetDeviceList(devType);
   for (size_t i=0; i<devices.size(); i++)
      waitForDevice(devices[i].c_str());
}

/**
 * Blocks until all devices included in the configuration become ready.
 * @param group      the configuration group
 * @param configName the configuration preset
 */
void CMMCore::waitForConfig(const char* group, const char* configName) MMCORE_LEGACY_THROW(CMMError)
{
   CheckConfigGroupName(group);
   CheckConfigPresetName(configName);

   Configuration cfg = getConfigData(group, configName);
   try {
      for(size_t i=0; i<cfg.size(); i++)
         waitForDevice(cfg.getSetting(i).getDeviceLabel().c_str());
   } catch (CMMError& err) {
      // trap MM exceptions and keep quiet - this is not a good time to blow up
      logError("waitForConfig", err.getMsg().c_str());
   }
}

/**
 * Sets the position of the stage in microns.
 * @param label     the stage device label
 * @param position  the desired stage position, in microns
 */
void CMMCore::setPosition(const char* label, double position) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<StageInstance> pStage =
      deviceManager_->GetDeviceOfType<StageInstance>(label);

   LOG_DEBUG(coreLogger_) << "Will start absolute move of " << label <<
      " to position " << std::fixed << std::setprecision(5) << position <<
      " um";

   mm::DeviceModuleLockGuard guard(pStage);
   int ret = pStage->SetPositionUm(position);
   if (ret != DEVICE_OK)
   {
      logError(pStage->GetName().c_str(), getDeviceErrorText(ret, pStage).c_str());
      throw CMMError(getDeviceErrorText(ret, pStage).c_str(), MMERR_DEVICE_GENERIC);
   }
}
/**
 * Sets the position of the stage in microns. Uses the current Z positioner
 * (focus) device.
 * @param position  the desired stage position, in microns
 */
void CMMCore::setPosition(double position) MMCORE_LEGACY_THROW(CMMError)
{
    setPosition(getFocusDevice().c_str(), position);
}

/**
 * Sets the relative position of the stage in microns.
 * @param label    the single-axis drive device label
 * @param d        the amount to move the stage, in microns (positive or negative)
 */
void CMMCore::setRelativePosition(const char* label, double d) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<StageInstance> pStage =
      deviceManager_->GetDeviceOfType<StageInstance>(label);

   LOG_DEBUG(coreLogger_) << "Will start relative move of " << label <<
      " by offset " << std::fixed << std::setprecision(5) << d << " um";

   mm::DeviceModuleLockGuard guard(pStage);

   int ret = pStage->SetRelativePositionUm(d);
   if (ret != DEVICE_OK)
   {
      logError(pStage->GetName().c_str(), getDeviceErrorText(ret, pStage).c_str());
      throw CMMError(getDeviceErrorText(ret, pStage).c_str(), MMERR_DEVICE_GENERIC);
   }
}

/**
 * Sets the relative position of the stage in microns. Uses the current Z
 * positioner (focus) device.
 * @param d        the amount to move the stage, in microns (positive or negative)
 */
void CMMCore::setRelativePosition(double d) MMCORE_LEGACY_THROW(CMMError)
{
    setRelativePosition(getFocusDevice().c_str(), d);
}

/**
 * Returns the current position of the stage in microns.
 * @return the position in microns
 * @param label     the single-axis drive device label
 */
double CMMCore::getPosition(const char* label) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<StageInstance> pStage =
      deviceManager_->GetDeviceOfType<StageInstance>(label);

   mm::DeviceModuleLockGuard guard(pStage);
   double pos;
   int ret = pStage->GetPositionUm(pos);
   if (ret != DEVICE_OK)
   {
      logError(pStage->GetName().c_str(), getDeviceErrorText(ret, pStage).c_str());
      throw CMMError(getDeviceErrorText(ret, pStage).c_str(), MMERR_DEVICE_GENERIC);
   }
   return pos;
}

/**
 * Returns the current position of the stage in microns. Uses the current
 * Z positioner (focus) device.
 * @return the position in microns
 */
double CMMCore::getPosition() MMCORE_LEGACY_THROW(CMMError)
{
    return getPosition(getFocusDevice().c_str());
}

/**
 * Sets the position of the XY stage in microns.
 * @param label  the XY stage device label
 * @param x      the X axis position in microns
 * @param y      the Y axis position in microns
 */
void CMMCore::setXYPosition(const char* label, double x, double y) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<XYStageInstance> pXYStage =
      deviceManager_->GetDeviceOfType<XYStageInstance>(label);

   LOG_DEBUG(coreLogger_) << "Will start absolute move of " << label <<
      " to position (" << std::fixed << std::setprecision(3) << x << ", " <<
      y << ") um";

   mm::DeviceModuleLockGuard guard(pXYStage);
   int ret = pXYStage->SetPositionUm(x, y);
   if (ret != DEVICE_OK)
   {
      logError(pXYStage->GetName().c_str(), getDeviceErrorText(ret, pXYStage).c_str());
      throw CMMError(getDeviceErrorText(ret, pXYStage).c_str(), MMERR_DEVICE_GENERIC);
   }
}

/**
 * Sets the position of the XY stage in microns. Uses the current XY stage
 * device.
 * @param x      the X axis position in microns
 * @param y      the Y axis position in microns
 */
void CMMCore::setXYPosition(double x, double y) MMCORE_LEGACY_THROW(CMMError)
{
    setXYPosition(getXYStageDevice().c_str(), x, y);
}

/**
 * Sets the relative position of the XY stage in microns.
 * @param label  the xy stage device label
 * @param dx     the distance to move in X (positive or negative)
 * @param dy     the distance to move in Y (positive or negative)
 */
void CMMCore::setRelativeXYPosition(const char* label, double dx, double dy) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<XYStageInstance> pXYStage =
      deviceManager_->GetDeviceOfType<XYStageInstance>(label);

   LOG_DEBUG(coreLogger_) << "Will start relative move of " << label <<
      " by (" << std::fixed << std::setprecision(3) << dx << ", " << dy <<
      ") um";

   mm::DeviceModuleLockGuard guard(pXYStage);
   int ret = pXYStage->SetRelativePositionUm(dx, dy);
   if (ret != DEVICE_OK)
   {
      logError(pXYStage->GetName().c_str(), getDeviceErrorText(ret, pXYStage).c_str());
      throw CMMError(getDeviceErrorText(ret, pXYStage).c_str(), MMERR_DEVICE_GENERIC);
   }
}

/**
 * Sets the relative position of the XY stage in microns. Uses the current
 * XY stage device.
 * @param dx     the distance to move in X (positive or negative)
 * @param dy     the distance to move in Y (positive or negative)
 */
void CMMCore::setRelativeXYPosition(double dx, double dy) MMCORE_LEGACY_THROW(CMMError) {
    setRelativeXYPosition(getXYStageDevice().c_str(), dx, dy);
}

/**
 * Obtains the current position of the XY stage in microns.
 * @param label   the stage device label
 * @param x            a return parameter yielding the X position in microns
 * @param y            a return parameter yielding the Y position in microns
 */
void CMMCore::getXYPosition(const char* label, double& x, double& y) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<XYStageInstance> pXYStage =
      deviceManager_->GetDeviceOfType<XYStageInstance>(label);

   mm::DeviceModuleLockGuard guard(pXYStage);
   int ret = pXYStage->GetPositionUm(x, y);
   if (ret != DEVICE_OK)
   {
      logError(pXYStage->GetName().c_str(), getDeviceErrorText(ret, pXYStage).c_str());
      throw CMMError(getDeviceErrorText(ret, pXYStage).c_str(), MMERR_DEVICE_GENERIC);
   }
}

/**
 * Obtains the current position of the XY stage in microns. Uses the current
 * XY stage device.
 * @param x            a return parameter yielding the X position in microns
 * @param y            a return parameter yielding the Y position in microns
 */
void CMMCore::getXYPosition(double& x, double& y) MMCORE_LEGACY_THROW(CMMError)
{
    getXYPosition(getXYStageDevice().c_str(), x, y);
}

/**
 * Obtains the current position of the X axis of the XY stage in microns.
 * @return    the x position
 * @param  label   the stage device label
 */
double CMMCore::getXPosition(const char* label) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<XYStageInstance> pXYStage =
      deviceManager_->GetDeviceOfType<XYStageInstance>(label);

   mm::DeviceModuleLockGuard guard(pXYStage);
   double x, y;
   int ret = pXYStage->GetPositionUm(x, y);
   if (ret != DEVICE_OK)
   {
      logError(pXYStage->GetName().c_str(), getDeviceErrorText(ret, pXYStage).c_str());
      throw CMMError(getDeviceErrorText(ret, pXYStage).c_str(), MMERR_DEVICE_GENERIC);
   }

   return x;
}

/**
 * Obtains the current position of the X axis of the XY stage in microns. Uses
 * the current XY stage device.
 * @return    the x position
 */
double CMMCore::getXPosition() MMCORE_LEGACY_THROW(CMMError)
{
    return getXPosition(getXYStageDevice().c_str());
}

/**
 * Obtains the current position of the Y axis of the XY stage in microns.
 * @return   the y position
 * @param   label   the stage device label
 */
double CMMCore::getYPosition(const char* label) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<XYStageInstance> pXYStage =
      deviceManager_->GetDeviceOfType<XYStageInstance>(label);

   mm::DeviceModuleLockGuard guard(pXYStage);
   double x, y;
   int ret = pXYStage->GetPositionUm(x, y);
   if (ret != DEVICE_OK)
   {
      logError(pXYStage->GetName().c_str(), getDeviceErrorText(ret, pXYStage).c_str());
      throw CMMError(getDeviceErrorText(ret, pXYStage).c_str(), MMERR_DEVICE_GENERIC);
   }

   return y;
}

/**
 * Obtains the current position of the Y axis of the XY stage in microns. Uses
 * the current XY stage device.
 * @return    the y position
 */
double CMMCore::getYPosition() MMCORE_LEGACY_THROW(CMMError)
{
    return getYPosition(getXYStageDevice().c_str());
}

/**
 * Stop the XY or focus/Z stage motors
 *
 * Not all stages support this operation; check before use.
 *
 * @param label    the stage device label (either XY or focus/Z stage)
 */
void CMMCore::stop(const char* label) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<DeviceInstance> stage =
      deviceManager_->GetDevice(label);

   std::shared_ptr<StageInstance> zStage =
      std::dynamic_pointer_cast<StageInstance>(stage);
   if (zStage)
   {
      LOG_DEBUG(coreLogger_) << "Will stop stage " << label;

      mm::DeviceModuleLockGuard guard(zStage);
      int ret = zStage->Stop();
      if (ret != DEVICE_OK)
      {
         logError(label, getDeviceErrorText(ret, zStage).c_str());
         throw CMMError(getDeviceErrorText(ret, zStage));
      }

      LOG_DEBUG(coreLogger_) << "Did stop stage " << label;
      return;
   }

   std::shared_ptr<XYStageInstance> xyStage =
      std::dynamic_pointer_cast<XYStageInstance>(stage);
   if (xyStage)
   {
      LOG_DEBUG(coreLogger_) << "Will stop xy stage " << label;

      mm::DeviceModuleLockGuard guard(xyStage);
      int ret = xyStage->Stop();
      if (ret != DEVICE_OK)
      {
         logError(label, getDeviceErrorText(ret, xyStage).c_str());
         throw CMMError(getDeviceErrorText(ret, xyStage));
      }

      LOG_DEBUG(coreLogger_) << "Did stop xy stage " << label;
      return;
   }

   throw CMMError("Cannot stop " + ToQuotedString(label) +
         ": not a stage");
}

/**
 * Perform a hardware homing operation for an XY or focus/Z stage.
 *
 * Not all stages support this operation. The user should be warned before
 * calling this method, as it can cause large stage movements, potentially
 * resulting in collision (e.g. with an expensive objective lens).
 *
 * @param label    the stage device label (either XY or focus/Z stage)
 */
void CMMCore::home(const char* label) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<DeviceInstance> stage =
      deviceManager_->GetDevice(label);

   std::shared_ptr<StageInstance> zStage =
      std::dynamic_pointer_cast<StageInstance>(stage);
   if (zStage)
   {
      LOG_DEBUG(coreLogger_) << "Will home stage " << label;

      mm::DeviceModuleLockGuard guard(zStage);
      int ret = zStage->Home();
      if (ret != DEVICE_OK)
      {
         logError(label, getDeviceErrorText(ret, zStage).c_str());
         throw CMMError(getDeviceErrorText(ret, zStage));
      }

      LOG_DEBUG(coreLogger_) << "Did home stage " << label;
      return;
   }

   std::shared_ptr<XYStageInstance> xyStage =
      std::dynamic_pointer_cast<XYStageInstance>(stage);
   if (xyStage)
   {
      LOG_DEBUG(coreLogger_) << "Will home xy stage " << label;

      mm::DeviceModuleLockGuard guard(xyStage);
      int ret = xyStage->Home();
      if (ret != DEVICE_OK)
      {
         logError(label, getDeviceErrorText(ret, xyStage).c_str());
         throw CMMError(getDeviceErrorText(ret, xyStage));
      }

      LOG_DEBUG(coreLogger_) << "Did home xy stage " << label;
      return;
   }

   throw CMMError("Cannot home " + ToQuotedString(label) +
         ": not a stage");
}

/**
 * Zero the given XY stage's coordinates at the current position.
 *
 * The current position becomes the new origin. Not to be confused with
 * setAdapterOriginXY().
 *
 * @param label    the stage device label
 */
void CMMCore::setOriginXY(const char* label) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<XYStageInstance> pXYStage =
      deviceManager_->GetDeviceOfType<XYStageInstance>(label);

   mm::DeviceModuleLockGuard guard(pXYStage);
   int ret = pXYStage->SetOrigin();
   if (ret != DEVICE_OK)
   {
      logError(label, getDeviceErrorText(ret, pXYStage).c_str());
      throw CMMError(getDeviceErrorText(ret, pXYStage).c_str(), MMERR_DEVICE_GENERIC);
   }

   LOG_DEBUG(coreLogger_) << "Zeroed xy stage " << label << " at current position";
}

/**
 * Zero the current XY stage's coordinates at the current position.
 *
 * The current position becomes the new origin. Not to be confused with
 * setAdapterOriginXY().
 */
void CMMCore::setOriginXY() MMCORE_LEGACY_THROW(CMMError)
{
    setOriginXY(getXYStageDevice().c_str());
}

/**
 * Zero the given XY stage's X coordinate at the current position.
 *
 * The current position becomes the new X = 0.
 *
 * @param label    the xy stage device label
 */
void CMMCore::setOriginX(const char* label) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<XYStageInstance> pXYStage =
      deviceManager_->GetDeviceOfType<XYStageInstance>(label);

   mm::DeviceModuleLockGuard guard(pXYStage);
   int ret = pXYStage->SetXOrigin();
   if (ret != DEVICE_OK)
   {
      logError(label, getDeviceErrorText(ret, pXYStage).c_str());
      throw CMMError(getDeviceErrorText(ret, pXYStage).c_str(), MMERR_DEVICE_GENERIC);
   }

   LOG_DEBUG(coreLogger_) << "Zeroed x coordinate of xy stage " << label <<
      " at current position";
}

/**
 * Zero the given XY stage's X coordinate at the current position.
 *
 * The current position becomes the new X = 0.
 */
void CMMCore::setOriginX() MMCORE_LEGACY_THROW(CMMError)
{
   setOriginX(getXYStageDevice().c_str());
}

/**
 * Zero the given XY stage's Y coordinate at the current position.
 *
 * The current position becomes the new Y = 0.
 *
 * @param label    the xy stage device label
 */
void CMMCore::setOriginY(const char* label) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<XYStageInstance> pXYStage =
      deviceManager_->GetDeviceOfType<XYStageInstance>(label);

   mm::DeviceModuleLockGuard guard(pXYStage);
   int ret = pXYStage->SetYOrigin();
   if (ret != DEVICE_OK)
   {
      logError(label, getDeviceErrorText(ret, pXYStage).c_str());
      throw CMMError(getDeviceErrorText(ret, pXYStage).c_str(), MMERR_DEVICE_GENERIC);
   }

   LOG_DEBUG(coreLogger_) << "Zeroed y coordinate of xy stage " << label <<
      " at current position";
}

/**
 * Zero the given XY stage's Y coordinate at the current position.
 *
 * The current position becomes the new Y = 0.
 */
void CMMCore::setOriginY() MMCORE_LEGACY_THROW(CMMError)
{
   setOriginY(getXYStageDevice().c_str());
}

/**
 * Zero the given focus/Z stage's coordinates at the current position.
 *
 * The current position becomes the new origin (Z = 0). Not to be confused with
 * setAdapterOrigin().
 *
 * @param label    the stage device label
 */
void CMMCore::setOrigin(const char* label) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<StageInstance> pStage =
      deviceManager_->GetDeviceOfType<StageInstance>(label);

   mm::DeviceModuleLockGuard guard(pStage);
   int ret = pStage->SetOrigin();
   if (ret != DEVICE_OK)
   {
      logError(label, getDeviceErrorText(ret, pStage).c_str());
      throw CMMError(getDeviceErrorText(ret, pStage).c_str(), MMERR_DEVICE_GENERIC);
   }

   LOG_DEBUG(coreLogger_) << "Zeroed stage " << label << " at current position";
}

/**
 * Zero the current focus/Z stage's coordinates at the current position.
 *
 * The current position becomes the new origin (Z = 0). Not to be confused with
 * setAdapterOrigin().
 */
void CMMCore::setOrigin() MMCORE_LEGACY_THROW(CMMError)
{
    setOrigin(getFocusDevice().c_str());
}

/**
 * Enable software translation of coordinates for the given focus/Z stage.
 *
 * The current position of the stage becomes Z = newZUm. Only some stages
 * support this functionality; it is recommended that setOrigin() be used
 * instead where available.
 *
 * @param label    the stage device label
 * @param newZUm   the new coordinate to assign to the current Z position
 */
void CMMCore::setAdapterOrigin(const char* label, double newZUm) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<StageInstance> pStage =
      deviceManager_->GetDeviceOfType<StageInstance>(label);

   mm::DeviceModuleLockGuard guard(pStage);
   int ret = pStage->SetAdapterOriginUm(newZUm);
   if (ret != DEVICE_OK)
   {
      logError(label, getDeviceErrorText(ret, pStage).c_str());
      throw CMMError(getDeviceErrorText(ret, pStage).c_str(), MMERR_DEVICE_GENERIC);
   }

   LOG_DEBUG(coreLogger_) << "Adapter-zeroed stage " << label <<
      ", assigning coordinate " << std::fixed << std::setprecision(5) <<
      newZUm << " um to the current position";
}

/**
 * Enable software translation of coordinates for the current focus/Z stage.
 *
 * The current position of the stage becomes Z = newZUm. Only some stages
 * support this functionality; it is recommended that setOrigin() be used
 * instead where available.
 *
 * @param newZUm   the new coordinate to assign to the current Z position
 */
void CMMCore::setAdapterOrigin(double newZUm) MMCORE_LEGACY_THROW(CMMError)
{
    setAdapterOrigin(getFocusDevice().c_str(), newZUm);
}

/**
 * Enable software translation of coordinates for the given XY stage.
 *
 * The current position of the stage becomes (newXUm, newYUm). It is
 * recommended that setOriginXY() be used instead where available.
 *
 * @param label    the XY stage device label
 * @param newXUm   the new coordinate to assign to the current X position
 * @param newYUm   the new coordinate to assign to the current Y position
 */
void CMMCore::setAdapterOriginXY(const char* label,
      double newXUm, double newYUm) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<XYStageInstance> pXYStage =
      deviceManager_->GetDeviceOfType<XYStageInstance>(label);

   mm::DeviceModuleLockGuard guard(pXYStage);
   int ret = pXYStage->SetAdapterOriginUm(newXUm, newYUm);
   if (ret != DEVICE_OK)
   {
      logError(label, getDeviceErrorText(ret, pXYStage).c_str());
      throw CMMError(getDeviceErrorText(ret, pXYStage).c_str(), MMERR_DEVICE_GENERIC);
   }

   LOG_DEBUG(coreLogger_) << "Adapter-zeroed XY stage " << label <<
      ", assigning coordinates (" << std::fixed << std::setprecision(3) <<
      newXUm << ", " << newYUm << ") um to the current position";
}

/**
 * Enable software translation of coordinates for the current XY stage.
 *
 * The current position of the stage becomes (newXUm, newYUm). It is
 * recommended that setOriginXY() be used instead where available.
 *
 * @param newXUm   the new coordinate to assign to the current X position
 * @param newYUm   the new coordinate to assign to the current Y position
 */
void CMMCore::setAdapterOriginXY(double newXUm, double newYUm) MMCORE_LEGACY_THROW(CMMError)
{
    setAdapterOriginXY(getXYStageDevice().c_str(), newXUm, newYUm);
}


/**
 * \brief Get the focus direction of a stage.
 *
 * Returns +1 if increasing position brings objective closer to sample, -1 if
 * increasing position moves objective away from sample, or 0 if unknown. (Make
 * sure to check for zero!)
 *
 * The returned value is determined by the most recent call to
 * setFocusDirection() for the stage, or defaults to what the stage device
 * adapter declares (often 0, for unknown).
 *
 * An exception is thrown if the direction has not been set and the device
 * encounters an error when determining the default direction.
 */
int CMMCore::getFocusDirection(const char* stageLabel) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<StageInstance> stage =
      deviceManager_->GetDeviceOfType<StageInstance>(stageLabel);

   mm::DeviceModuleLockGuard guard(stage);
   switch (stage->GetFocusDirection()) {
      case MM::FocusDirectionTowardSample: return +1;
      case MM::FocusDirectionAwayFromSample: return -1;
      default: return 0;
   }
}


/**
 * \brief Set the focus direction of a stage.
 *
 * The sign should be +1 (or any positive value), zero, or -1 (or any negative
 * value), and is interpreted in the same way as the return value of
 * getFocusDirection().
 *
 * Once this method is called, getFocusDirection() for the stage will always
 * return the set value.
 *
 * For legacy reasons, an exception is not thrown if there is an error.
 * Instead, nothing is done if stageLabel is not a valid focus stage.
 */
void CMMCore::setFocusDirection(const char* stageLabel, int sign)
{
   MM::FocusDirection direction = MM::FocusDirectionUnknown;
   if (sign > 0)
      direction = MM::FocusDirectionTowardSample;
   if (sign < 0)
      direction = MM::FocusDirectionAwayFromSample;

   try
   {
      std::shared_ptr<StageInstance> stage =
         deviceManager_->GetDeviceOfType<StageInstance>(stageLabel);

      mm::DeviceModuleLockGuard guard(stage);
      stage->SetFocusDirection(direction);
   }
   catch (const CMMError&)
   {
   }
}


/**
 * Queries camera if exposure can be used in a sequence
 * @param cameraLabel    the camera device label
 * @return   true if exposure can be sequenced
 */
bool CMMCore::isExposureSequenceable(const char* cameraLabel) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<CameraInstance> pCamera =
      deviceManager_->GetDeviceOfType<CameraInstance>(cameraLabel);

   mm::DeviceModuleLockGuard guard(pCamera);

   bool isSequenceable;
   int ret = pCamera->IsExposureSequenceable(isSequenceable);
   if (ret != DEVICE_OK)
      throw CMMError(getDeviceErrorText(ret, pCamera));

   return isSequenceable;
}


/**
 * Starts an ongoing sequence of triggered exposures in a camera
 * This should only be called for cameras where exposure time is sequenceable
 * @param cameraLabel    the camera device label
 */
void CMMCore::startExposureSequence(const char* cameraLabel) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<CameraInstance> pCamera =
      deviceManager_->GetDeviceOfType<CameraInstance>(cameraLabel);

   mm::DeviceModuleLockGuard guard(pCamera);

   int ret = pCamera->StartExposureSequence();
   if (ret != DEVICE_OK)
      throw CMMError(getDeviceErrorText(ret, pCamera));
}

/**
 * Stops an ongoing sequence of triggered exposures in a camera
 * This should only be called for cameras where exposure time is sequenceable
 * @param cameraLabel   the camera device label
 */
void CMMCore::stopExposureSequence(const char* cameraLabel) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<CameraInstance> pCamera =
      deviceManager_->GetDeviceOfType<CameraInstance>(cameraLabel);

   mm::DeviceModuleLockGuard guard(pCamera);

   int ret = pCamera->StopExposureSequence();
   if (ret != DEVICE_OK)
      throw CMMError(getDeviceErrorText(ret, pCamera));
}

/**
 * Gets the maximum length of a camera's exposure sequence.
 * This should only be called for cameras where exposure time is sequenceable
 * @param cameraLabel    the camera device label
 */
long CMMCore::getExposureSequenceMaxLength(const char* cameraLabel) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<CameraInstance> pCamera =
      deviceManager_->GetDeviceOfType<CameraInstance>(cameraLabel);

   mm::DeviceModuleLockGuard guard(pCamera);
   long length;
   int ret = pCamera->GetExposureSequenceMaxLength(length);
   if (ret != DEVICE_OK)
      throw CMMError(getDeviceErrorText(ret, pCamera));

   return length;
}

/**
 * Transfer a sequence of exposure times to the camera.
 * This should only be called for cameras where exposure time is sequenceable
 * @param cameraLabel      the camera device label
 * @param exposureTime_ms  sequence of exposure times the camera will use during a sequence acquisition
 */
void CMMCore::loadExposureSequence(const char* cameraLabel, std::vector<double> exposureTime_ms) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<CameraInstance> pCamera =
      deviceManager_->GetDeviceOfType<CameraInstance>(cameraLabel);

   unsigned long maxLength = getExposureSequenceMaxLength(cameraLabel);
   if (exposureTime_ms.size() > maxLength) {
      throw CMMError("The length of the requested exposure sequence (" + ToString(exposureTime_ms.size()) +
            ") exceeds the maximum allowed (" + ToString(maxLength) +
            ") by the camera " + ToQuotedString(cameraLabel));
   }

   mm::DeviceModuleLockGuard guard(pCamera);

   int ret;
   ret = pCamera->ClearExposureSequence();
   if (ret != DEVICE_OK)
      throw CMMError(getDeviceErrorText(ret, pCamera));

   std::vector<double>::iterator it;
   for ( it=exposureTime_ms.begin() ; it < exposureTime_ms.end(); it++ )
   {
      ret = pCamera->AddToExposureSequence(*it);
      if (ret != DEVICE_OK)
         throw CMMError(getDeviceErrorText(ret, pCamera));
   }

   ret = pCamera->SendExposureSequence();
   if (ret != DEVICE_OK)
      throw CMMError(getDeviceErrorText(ret, pCamera));
}


/**
 * Queries stage if it can be used in a sequence
 * @param label   the stage device label
 * @return   true if the stage can be sequenced
 */
bool CMMCore::isStageSequenceable(const char* label) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<StageInstance> pStage =
      deviceManager_->GetDeviceOfType<StageInstance>(label);

   mm::DeviceModuleLockGuard guard(pStage);

   bool isSequenceable;
   int ret = pStage->IsStageSequenceable(isSequenceable);
   if (ret != DEVICE_OK)
      throw CMMError(getDeviceErrorText(ret, pStage));

   return isSequenceable;
}

/**
 * Queries if the stage can be used in a linear sequence
 * A linear sequence is defined by a stepsize and number of slices
 * @param label   the stage device label
 * @return   true if the stage supports linear sequences
 */
bool CMMCore::isStageLinearSequenceable(const char* label) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<StageInstance> pStage =
      deviceManager_->GetDeviceOfType<StageInstance>(label);

   mm::DeviceModuleLockGuard guard(pStage);

   bool isSequenceable;
   int ret = pStage->IsStageLinearSequenceable(isSequenceable);
   if (ret != DEVICE_OK)
      throw CMMError(getDeviceErrorText(ret, pStage));

   return isSequenceable;
}

/**
 * Starts an ongoing sequence of triggered events in a stage
 * This should only be called for stages
 * @param label    the stage device label
 */
void CMMCore::startStageSequence(const char* label) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<StageInstance> pStage =
      deviceManager_->GetDeviceOfType<StageInstance>(label);

   mm::DeviceModuleLockGuard guard(pStage);

   int ret = pStage->StartStageSequence();
   if (ret != DEVICE_OK)
      throw CMMError(getDeviceErrorText(ret, pStage));
}

/**
 * Stops an ongoing sequence of triggered events in a stage
 * This should only be called for stages that are sequenceable
 * @param label    the stage device label
 */
void CMMCore::stopStageSequence(const char* label) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<StageInstance> pStage =
      deviceManager_->GetDeviceOfType<StageInstance>(label);

   mm::DeviceModuleLockGuard guard(pStage);

   int ret = pStage->StopStageSequence();
   if (ret != DEVICE_OK)
      throw CMMError(getDeviceErrorText(ret, pStage));
}

/**
 * Gets the maximum length of a stage's position sequence.
 * This should only be called for stages that are sequenceable
 * @param label    the stage device label
 * @return         the maximum length (integer)
 */
long CMMCore::getStageSequenceMaxLength(const char* label) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<StageInstance> pStage =
      deviceManager_->GetDeviceOfType<StageInstance>(label);

   mm::DeviceModuleLockGuard guard(pStage);
   long length;
   int ret = pStage->GetStageSequenceMaxLength(length);
   if (ret != DEVICE_OK)
      throw CMMError(getDeviceErrorText(ret, pStage));

   return length;
}

/**
 * Transfer a sequence of events/states/whatever to the device
 * This should only be called for device-properties that are sequenceable
 * @param label              the device label
 * @param positionSequence   a sequence of positions that the stage will execute in response to external triggers
 */
void CMMCore::loadStageSequence(const char* label, std::vector<double> positionSequence) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<StageInstance> pStage =
      deviceManager_->GetDeviceOfType<StageInstance>(label);

   mm::DeviceModuleLockGuard guard(pStage);

   int ret;
   ret = pStage->ClearStageSequence();
   if (ret != DEVICE_OK)
      throw CMMError(getDeviceErrorText(ret, pStage));

   std::vector<double>::iterator it;
   for ( it=positionSequence.begin() ; it < positionSequence.end(); it++ )
   {
      ret = pStage->AddToStageSequence(*it);
      if (ret != DEVICE_OK)
         throw CMMError(getDeviceErrorText(ret, pStage));
   }

   ret = pStage->SendStageSequence();
   if (ret != DEVICE_OK)
      throw CMMError(getDeviceErrorText(ret, pStage));
}

/**
 * Loads a linear sequence (defined by stepsize and nr. of steps) into
 * the device.  Why was it not called loadStageLinearSequence???
 * @param label   Name of the stage device
 * @param dZ_um   Step size between slices in microns
 * @param nSlices    Number of slices fo ethis sequence
 *                   Presumably the sequence will repeat after this
 *                   number of TTLs was received
 */
void CMMCore::setStageLinearSequence(const char* label, double dZ_um, int nSlices) MMCORE_LEGACY_THROW(CMMError)
{
   if (nSlices < 0)
      throw CMMError("Linear sequence cannot have negative length");

   std::shared_ptr<StageInstance> pStage =
      deviceManager_->GetDeviceOfType<StageInstance>(label);

   mm::DeviceModuleLockGuard guard(pStage);

   int ret;
   ret = pStage->SetStageLinearSequence(dZ_um, nSlices);
   if (ret != DEVICE_OK)
      throw CMMError(getDeviceErrorText(ret, pStage));
}

/**
 * Queries XY stage if it can be used in a sequence
 * @param label    the XY stage device label
 */
bool CMMCore::isXYStageSequenceable(const char* label) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<XYStageInstance> pStage =
      deviceManager_->GetDeviceOfType<XYStageInstance>(label);

   mm::DeviceModuleLockGuard guard(pStage);

   bool isSequenceable;
   int ret = pStage->IsXYStageSequenceable(isSequenceable);
   if (ret != DEVICE_OK)
      throw CMMError(getDeviceErrorText(ret, pStage));

   return isSequenceable;

}


/**
 * Starts an ongoing sequence of triggered events in an XY stage
 * This should only be called for stages
 * @param label       the XY stage device label
 */
void CMMCore::startXYStageSequence(const char* label) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<XYStageInstance> pStage =
      deviceManager_->GetDeviceOfType<XYStageInstance>(label);

   mm::DeviceModuleLockGuard guard(pStage);

   int ret = pStage->StartXYStageSequence();
   if (ret != DEVICE_OK)
      throw CMMError(getDeviceErrorText(ret, pStage));
}

/**
 * Stops an ongoing sequence of triggered events in an XY stage
 * This should only be called for stages that are sequenceable
 * @param label     the XY stage device label
 */
void CMMCore::stopXYStageSequence(const char* label) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<XYStageInstance> pStage =
      deviceManager_->GetDeviceOfType<XYStageInstance>(label);

   mm::DeviceModuleLockGuard guard(pStage);

   int ret = pStage->StopXYStageSequence();
   if (ret != DEVICE_OK)
      throw CMMError(getDeviceErrorText(ret, pStage));
}

/**
 * Gets the maximum length of an XY stage's position sequence.
 * This should only be called for XY stages that are sequenceable
 * @param label   the XY stage device label
 * @return        the maximum allowed sequence length
 */
long CMMCore::getXYStageSequenceMaxLength(const char* label) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<XYStageInstance> pStage =
      deviceManager_->GetDeviceOfType<XYStageInstance>(label);

   mm::DeviceModuleLockGuard guard(pStage);
   long length;
   int ret = pStage->GetXYStageSequenceMaxLength(length);
   if (ret != DEVICE_OK)
      throw CMMError(getDeviceErrorText(ret, pStage));

   return length;
}

/**
 * Transfer a sequence of stage positions to the xy stage.
 * xSequence and ySequence must have the same length.
 * This should only be called for XY stages that are sequenceable
 * @param label        the XY stage device label
 * @param xSequence    the sequence of x positions that the stage will execute in response to external triggers
 * @param ySequence    the sequence of y positions that the stage will execute in response to external triggers
 */
void CMMCore::loadXYStageSequence(const char* label,
                                  std::vector<double> xSequence,
                                  std::vector<double> ySequence) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<XYStageInstance> pStage =
      deviceManager_->GetDeviceOfType<XYStageInstance>(label);

   mm::DeviceModuleLockGuard guard(pStage);

   int ret;
   ret = pStage->ClearXYStageSequence();
   if (ret != DEVICE_OK)
      throw CMMError(getDeviceErrorText(ret, pStage));

   std::vector<double>::iterator itx, ity;
   for ( itx=xSequence.begin(), ity=ySequence.begin() ;
         (itx < xSequence.end()) && (ity < ySequence.end()); itx++, ity++)
   {
      ret = pStage->AddToXYStageSequence(*itx, *ity);
      if (ret != DEVICE_OK)
         throw CMMError(getDeviceErrorText(ret, pStage));
   }

   ret = pStage->SendXYStageSequence();
   if (ret != DEVICE_OK)
      throw CMMError(getDeviceErrorText(ret, pStage));
}


/**
 * Acquires a single image with current settings.
 * Snap is not allowed while the acquisition thread is run
 */
void CMMCore::snapImage() MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<CameraInstance> camera = currentCameraDevice_.lock();
   if (camera)
   {
      if(camera->IsCapturing())
      {
         throw CMMError(getCoreErrorText(
            MMERR_NotAllowedDuringSequenceAcquisition).c_str()
            ,MMERR_NotAllowedDuringSequenceAcquisition);
      }

      mm::DeviceModuleLockGuard guard(camera);

      int ret = DEVICE_OK;
      try {
         // open the shutter
         std::shared_ptr<ShutterInstance> shutter =
            currentShutterDevice_.lock();
         if (autoShutter_ && shutter)
         {
            int sret = shutter->SetOpen(true);
            if (DEVICE_OK != sret)
            {
               logError("CMMCore::snapImage", getDeviceErrorText(sret, shutter).c_str());
               throw CMMError(getDeviceErrorText(sret, shutter).c_str(), MMERR_DEVICE_GENERIC);
            }
            waitForDevice(shutter);
         }

         LOG_DEBUG(coreLogger_) << "Will snap image from current camera";
         ret = camera->SnapImage();
         if (ret == DEVICE_OK)
         {
            LOG_DEBUG(coreLogger_) << "Did snap image from current camera";
         }
         else
         {
            LOG_ERROR(coreLogger_) << "Failed to snap image from current camera";
         }

			everSnapped_ = true;

         // close the shutter
         if (autoShutter_ && shutter)
         {
            int sret  = shutter->SetOpen(false);
            if (DEVICE_OK != sret)
            {
               logError("CMMCore::snapImage", getDeviceErrorText(sret, shutter).c_str());
               throw CMMError(getDeviceErrorText(sret, shutter).c_str(), MMERR_DEVICE_GENERIC);
            }
            waitForDevice(shutter);
         }
         if (externalCallback_)
         {
            externalCallback_->onImageSnapped(camera->GetLabel().c_str());
         }
		}catch( CMMError& e){
			throw e;
		}
		catch (...) {
         logError("CMMCore::snapImage", getCoreErrorText(MMERR_UnhandledException).c_str());
         throw CMMError(getCoreErrorText(MMERR_UnhandledException).c_str(), MMERR_UnhandledException);
      }

      if (ret != DEVICE_OK)
      {
         logError("CMMCore::snapImage", getDeviceErrorText(ret, camera).c_str());
         throw CMMError(getDeviceErrorText(ret, camera).c_str(), MMERR_DEVICE_GENERIC);
      }
   }
   else
   {
      throw CMMError(getCoreErrorText(MMERR_CameraNotAvailable).c_str(), MMERR_CameraNotAvailable);
   }
}

/**
 * If this option is enabled Shutter automatically opens and closes when the image
 * is acquired.
 * @param state      true for enabled
 */
void CMMCore::setAutoShutter(bool state)
{
   properties_->Set(MM::g_Keyword_CoreAutoShutter, state ? "1" : "0");
   autoShutter_ = state;
   {
      MMThreadGuard scg(stateCacheLock_);
      stateCache_.addSetting(PropertySetting(MM::g_Keyword_CoreDevice, MM::g_Keyword_CoreAutoShutter, state ? "1" : "0"));
   }
   LOG_DEBUG(coreLogger_) << "Autoshutter turned " << (state ? "on" : "off");
}

/**
 * Returns the current setting of the auto-shutter option.
 */
bool CMMCore::getAutoShutter()
{
   return autoShutter_;
}

/**
* Opens or closes the specified shutter.
* @param shutterLabel  the shutter device label
* @param state         the desired state of the shutter (true for open)
*/
void CMMCore::setShutterOpen(const char* shutterLabel, bool state) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<ShutterInstance> pShutter =
      deviceManager_->GetDeviceOfType<ShutterInstance>(shutterLabel);
   if (pShutter)
   {
      mm::DeviceModuleLockGuard guard(pShutter);
      int ret = pShutter->SetOpen(state);
      if (ret != DEVICE_OK)
      {
         logError("CMMCore::setShutterOpen()", getDeviceErrorText(ret, pShutter).c_str());
         throw CMMError(getDeviceErrorText(ret, pShutter).c_str(), MMERR_DEVICE_GENERIC);
      }

      if (pShutter->HasProperty(MM::g_Keyword_State))
      {
         {
            MMThreadGuard scg(stateCacheLock_);
            stateCache_.addSetting(PropertySetting(shutterLabel, MM::g_Keyword_State, CDeviceUtils::ConvertToString(state)));
         }
      }
   }
}

/**
 * Opens or closes the currently selected (default) shutter.
 * @param  state     the desired state of the shutter (true for open)
 */
void CMMCore::setShutterOpen(bool state) MMCORE_LEGACY_THROW(CMMError)
{
   std::string shutterLabel = getShutterDevice();
   if (shutterLabel.empty()) return;
   setShutterOpen(shutterLabel.c_str(), state);
}

/**
 * Returns the state of the specified shutter.
 * @param  shutterLabel   the name of the shutter
 */
bool CMMCore::getShutterOpen(const char* shutterLabel) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<ShutterInstance> pShutter =
      deviceManager_->GetDeviceOfType<ShutterInstance>(shutterLabel);
   bool state = true; // default open
   if (pShutter)
   {
      mm::DeviceModuleLockGuard guard(pShutter);
      int ret = pShutter->GetOpen(state);
      if (ret != DEVICE_OK)
      {
         logError("CMMCore::getShutterOpen()", getDeviceErrorText(ret, pShutter).c_str());
         throw CMMError(getDeviceErrorText(ret, pShutter).c_str(), MMERR_DEVICE_GENERIC);
      }
   }
   return state;
}

/**
 * Returns the state of the currently selected (default) shutter.
 */
bool CMMCore::getShutterOpen() MMCORE_LEGACY_THROW(CMMError)
{
   std::string shutterLabel = getShutterDevice();
   if (shutterLabel.empty()) return true;
   return getShutterOpen(shutterLabel.c_str());
}

/**
 * Exposes the internal image buffer.
 *
 * Use to get the image acquired by snapImage
 *
 * Multi-Channel cameras will return the content of the first
 * channel in this function
 *
 * Designed specifically for the SWIG wrapping for Java and scripting languages.
 *
 * Supported data types are byte (8 bits per pixel, 1 component), short 
 * (16 bits per pixel, 1 component), float (32 bits per pixel, 1 component, not
 * supported by the UI yet), RGB_32 (8 bits per component, 4 components), RGB_64
 * (16 bits per component, 4 components, not supported by UI yet).  
 * RGB buffers are expected to be in big endian ARGB format (ARGB8888), which means that
 * on little endian the format is BGRA888 
 * (see: https://en.wikipedia.org/wiki/RGBA_color_model).
 *
 * @return a pointer to the internal image buffer.
 * @throws CMMError   when the camera returns no data
 */
void* CMMCore::getImage() MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<CameraInstance> camera = currentCameraDevice_.lock();
   if (!camera)
      throw CMMError(getCoreErrorText(MMERR_CameraNotAvailable).c_str(), MMERR_CameraNotAvailable);
   else
   {
		if( ! everSnapped_)
		{
         logError("CMMCore::getImage()", getCoreErrorText(MMERR_InvalidImageSequence).c_str());
         throw CMMError(getCoreErrorText(MMERR_InvalidImageSequence).c_str(), MMERR_InvalidImageSequence);
      }

      // scope for the thread guard
      {
         MMThreadGuard g(*pPostedErrorsLock_);

         if(0 < postedErrors_.size())
         {
            std::pair< int, std::string>  toThrow(postedErrors_[0]);
            // todo, process the collection of posted errors.
            postedErrors_.clear();
            throw CMMError( toThrow.second.c_str(), toThrow.first);
         }
      }

      void* pBuf(0);
      try {
         mm::DeviceModuleLockGuard guard(camera);
         pBuf = const_cast<unsigned char*> (camera->GetImageBuffer());

         std::shared_ptr<ImageProcessorInstance> imageProcessor =
            currentImageProcessor_.lock();
         if (imageProcessor)
	      {
            imageProcessor->Process((unsigned char*)pBuf, camera->GetImageWidth(),  camera->GetImageHeight(), camera->GetImageBytesPerPixel() );
	      }
		} catch( CMMError& e){
			throw e;
		} catch (...) {
         logError("CMMCore::getImage()", getCoreErrorText(MMERR_UnhandledException).c_str());
         throw CMMError(getCoreErrorText(MMERR_UnhandledException).c_str(), MMERR_UnhandledException);
      }

      if (pBuf != 0)
         return pBuf;
      else
      {
         logError("CMMCore::getImage()", getCoreErrorText(MMERR_CameraBufferReadFailed).c_str());
         throw CMMError(getCoreErrorText(MMERR_CameraBufferReadFailed).c_str(), MMERR_CameraBufferReadFailed);
      }
   }
}

/**
 * Returns the internal image buffer for a given Camera Channel
 *
 * Use to get the image acquired by snapImage
 *
 * Single channel cameras will return the content of their image buffer
 * irrespective of the channelNr argument
 * Designed specifically for the SWIG wrapping for Java and scripting languages.
 *
 * @param channelNr   Channel number for which the image buffer is requested
 * @return a pointer to the internal image buffer.
 */
void* CMMCore::getImage(unsigned channelNr) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<CameraInstance> camera = currentCameraDevice_.lock();
   if (!camera)
      throw CMMError(getCoreErrorText(MMERR_CameraNotAvailable).c_str(), MMERR_CameraNotAvailable);
   else
   {
      void* pBuf(0);
      try {
         mm::DeviceModuleLockGuard guard(camera);
         pBuf = const_cast<unsigned char*> (camera->GetImageBuffer(channelNr));

         std::shared_ptr<ImageProcessorInstance> imageProcessor =
            currentImageProcessor_.lock();
         if (imageProcessor)
	      {
            imageProcessor->Process((unsigned char*)pBuf, camera->GetImageWidth(),  camera->GetImageHeight(), camera->GetImageBytesPerPixel() );
	      }
		} catch( CMMError& e){
			throw e;
		} catch (...) {
         logError("CMMCore::getImage()", getCoreErrorText(MMERR_UnhandledException).c_str());
         throw CMMError(getCoreErrorText(MMERR_UnhandledException).c_str(), MMERR_UnhandledException);
      }

      if (pBuf != 0)
         return pBuf;
      else
      {
         logError("CMMCore::getImage()", getCoreErrorText(MMERR_CameraBufferReadFailed).c_str());
         throw CMMError(getCoreErrorText(MMERR_CameraBufferReadFailed).c_str(), MMERR_CameraBufferReadFailed);
      }
   }
}

/**
* Returns the size of the internal image buffer.
*
* @return buffer size
*/
long CMMCore::getImageBufferSize()
{
   std::shared_ptr<CameraInstance> camera = currentCameraDevice_.lock();
   if (camera) {
      try
      {
         mm::DeviceModuleLockGuard guard(camera);
         return camera->GetImageBufferSize();
      }
      catch (const CMMError&) // Possibly uninitialized camera
      {
         // Fall through
      }
   }
   return 0;
}

/**
 * Starts streaming camera sequence acquisition.
 * This command does not block the calling thread for the duration of the acquisition.
 *
 * @param numImages        Number of images requested from the camera
 * @param intervalMs       The interval between images, currently only supported by Andor cameras
 * @param stopOnOverflow   whether or not the camera stops acquiring when the circular buffer is full
 */
void CMMCore::startSequenceAcquisition(long numImages, double intervalMs, bool stopOnOverflow) MMCORE_LEGACY_THROW(CMMError)
{
   // scope for the thread guard
   {
      MMThreadGuard g(*pPostedErrorsLock_);
      postedErrors_.clear();
   }

   std::shared_ptr<CameraInstance> camera = currentCameraDevice_.lock();
   if (camera)
   {
      if(camera->IsCapturing())
      {
         throw CMMError(getCoreErrorText(
            MMERR_NotAllowedDuringSequenceAcquisition).c_str()
            ,MMERR_NotAllowedDuringSequenceAcquisition);
      }

		try
		{
			if (!cbuf_->Initialize(camera->GetNumberOfChannels(), camera->GetImageWidth(), camera->GetImageHeight(), camera->GetImageBytesPerPixel()))
			{
				logError(getDeviceName(camera).c_str(), getCoreErrorText(MMERR_CircularBufferFailedToInitialize).c_str());
				throw CMMError(getCoreErrorText(MMERR_CircularBufferFailedToInitialize).c_str(), MMERR_CircularBufferFailedToInitialize);
			}
			cbuf_->Clear();
         mm::DeviceModuleLockGuard guard(camera);

         LOG_DEBUG(coreLogger_) << "Will start sequence acquisition from default camera";
			int nRet = camera->StartSequenceAcquisition(numImages, intervalMs, stopOnOverflow);
			if (nRet != DEVICE_OK)
				throw CMMError(getDeviceErrorText(nRet, camera).c_str(), MMERR_DEVICE_GENERIC);
		}
		catch (std::bad_alloc& ex)
		{
			std::ostringstream messs;
			messs << getCoreErrorText(MMERR_OutOfMemory).c_str() << " " << ex.what() << '\n';
			throw CMMError(messs.str().c_str() , MMERR_OutOfMemory);
		}
   }
   else
   {
      logError(getDeviceName(camera).c_str(), getCoreErrorText(MMERR_CameraNotAvailable).c_str());
      throw CMMError(getCoreErrorText(MMERR_CameraNotAvailable).c_str(), MMERR_CameraNotAvailable);
   }
   LOG_DEBUG(coreLogger_) << "Did start sequence acquisition from default camera";
   // onSequenceAcquisitionStarted will be called by CoreCallback::PrepareForAcq
}

/**
 * Starts streaming camera sequence acquisition for a specified camera.
 * This command does not block the calling thread for the duration of the acquisition.
 * The difference between this method and the one with the same name but operating on the "default"
 * camera is that it does not automatically initialize the circular buffer.
 */
void CMMCore::startSequenceAcquisition(const char* label, long numImages, double intervalMs, bool stopOnOverflow) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<CameraInstance> pCam =
      deviceManager_->GetDeviceOfType<CameraInstance>(label);

   mm::DeviceModuleLockGuard guard(pCam);
   if(pCam->IsCapturing())
      throw CMMError(getCoreErrorText(MMERR_NotAllowedDuringSequenceAcquisition).c_str(),
                     MMERR_NotAllowedDuringSequenceAcquisition);

   if (!cbuf_->Initialize(pCam->GetNumberOfChannels(), pCam->GetImageWidth(), pCam->GetImageHeight(), pCam->GetImageBytesPerPixel()))
   {
      logError(getDeviceName(pCam).c_str(), getCoreErrorText(MMERR_CircularBufferFailedToInitialize).c_str());
      throw CMMError(getCoreErrorText(MMERR_CircularBufferFailedToInitialize).c_str(), MMERR_CircularBufferFailedToInitialize);
   }
   cbuf_->Clear();
	
   LOG_DEBUG(coreLogger_) <<
      "Will start sequence acquisition from camera " << label;
   int nRet = pCam->StartSequenceAcquisition(numImages, intervalMs, stopOnOverflow);
   if (nRet != DEVICE_OK)
      throw CMMError(getDeviceErrorText(nRet, pCam).c_str(), MMERR_DEVICE_GENERIC);

   LOG_DEBUG(coreLogger_) <<
      "Did start sequence acquisition from camera " << label;
   // onSequenceAcquisitionStarted will be called by CoreCallback::PrepareForAcq
}

/**
 * Prepare the camera for the sequence acquisition to save the time in the
 * StartSequenceAcqusition() call which is supposed to come next.
 */
void CMMCore::prepareSequenceAcquisition(const char* label) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<CameraInstance> pCam =
      deviceManager_->GetDeviceOfType<CameraInstance>(label);

   mm::DeviceModuleLockGuard guard(pCam);
   if(pCam->IsCapturing())
      throw CMMError(getCoreErrorText(MMERR_NotAllowedDuringSequenceAcquisition).c_str(),
                     MMERR_NotAllowedDuringSequenceAcquisition);

   LOG_DEBUG(coreLogger_) << "Will prepare camera " << label <<
      " for sequence acquisition";
   int nRet = pCam->PrepareSequenceAcqusition();
   if (nRet != DEVICE_OK)
      throw CMMError(getDeviceErrorText(nRet, pCam).c_str(), MMERR_DEVICE_GENERIC);

   LOG_DEBUG(coreLogger_) << "Did prepare camera " << label <<
      " for sequence acquisition";
}


/**
 * Initialize circular buffer based on the current camera settings.
 */
void CMMCore::initializeCircularBuffer() MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<CameraInstance> camera = currentCameraDevice_.lock();
   if (camera)
   {
      mm::DeviceModuleLockGuard guard(camera);
      if (!cbuf_->Initialize(camera->GetNumberOfChannels(), camera->GetImageWidth(), camera->GetImageHeight(), camera->GetImageBytesPerPixel()))
      {
         logError(getDeviceName(camera).c_str(), getCoreErrorText(MMERR_CircularBufferFailedToInitialize).c_str());
         throw CMMError(getCoreErrorText(MMERR_CircularBufferFailedToInitialize).c_str(), MMERR_CircularBufferFailedToInitialize);
      }
      cbuf_->Clear();
   }
   else
   {
      throw CMMError(getCoreErrorText(MMERR_CameraNotAvailable).c_str(), MMERR_CameraNotAvailable);
   }
   LOG_DEBUG(coreLogger_) << "Circular buffer initialized based on current camera";
}

/**
 * Stops streaming camera sequence acquisition for a specified camera.
 * @param label   The camera name
 */
void CMMCore::stopSequenceAcquisition(const char* label) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<CameraInstance> pCam =
      deviceManager_->GetDeviceOfType<CameraInstance>(label);

   mm::DeviceModuleLockGuard guard(pCam);
   LOG_DEBUG(coreLogger_) << "Will stop sequence acquisition from camera " << label;
   int nRet = pCam->StopSequenceAcquisition();
   if (nRet != DEVICE_OK)
   {
      logError(label, getDeviceErrorText(nRet, pCam).c_str());
      throw CMMError(getDeviceErrorText(nRet, pCam).c_str(), MMERR_DEVICE_GENERIC);
   }

   LOG_DEBUG(coreLogger_) << "Did stop sequence acquisition from camera " << label;
   // onSequenceAcquisitionStopped will be called by CoreCallback::AcqFinished
}

/**
 * Starts the continuous camera sequence acquisition.
 * This command does not block the calling thread for the duration of the acquisition.
 */
void CMMCore::startContinuousSequenceAcquisition(double intervalMs) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<CameraInstance> camera = currentCameraDevice_.lock();
   if (camera)
   {
      mm::DeviceModuleLockGuard guard(camera);
      if(camera->IsCapturing())
      {
         throw CMMError(getCoreErrorText(
            MMERR_NotAllowedDuringSequenceAcquisition).c_str()
            ,MMERR_NotAllowedDuringSequenceAcquisition);
      }

      if (!cbuf_->Initialize(camera->GetNumberOfChannels(), camera->GetImageWidth(), camera->GetImageHeight(), camera->GetImageBytesPerPixel()))
      {
         logError(getDeviceName(camera).c_str(), getCoreErrorText(MMERR_CircularBufferFailedToInitialize).c_str());
         throw CMMError(getCoreErrorText(MMERR_CircularBufferFailedToInitialize).c_str(), MMERR_CircularBufferFailedToInitialize);
      }
      cbuf_->Clear();
      LOG_DEBUG(coreLogger_) << "Will start continuous sequence acquisition from current camera";
      int nRet = camera->StartSequenceAcquisition(intervalMs);
      if (nRet != DEVICE_OK)
         throw CMMError(getDeviceErrorText(nRet, camera).c_str(), MMERR_DEVICE_GENERIC);
   }
   else
   {
      logError("no camera available", getCoreErrorText(MMERR_CameraNotAvailable).c_str());
      throw CMMError(getCoreErrorText(MMERR_CameraNotAvailable).c_str(), MMERR_CameraNotAvailable);
   }
   LOG_DEBUG(coreLogger_) << "Did start continuous sequence acquisition from current camera";
   // onSequenceAcquisitionStarted will be called by CoreCallback::PrepareForAcq
}

/**
 * Stops streaming camera sequence acquisition.
 */
void CMMCore::stopSequenceAcquisition() MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<CameraInstance> camera = currentCameraDevice_.lock();
   if (camera)
   {
      mm::DeviceModuleLockGuard guard(camera);
      LOG_DEBUG(coreLogger_) << "Will stop sequence acquisition from current camera";
      int nRet = camera->StopSequenceAcquisition();
      if (nRet != DEVICE_OK)
      {
         logError(getDeviceName(camera).c_str(), getDeviceErrorText(nRet, camera).c_str());
         throw CMMError(getDeviceErrorText(nRet, camera).c_str(), MMERR_DEVICE_GENERIC);
      }
   }
   else
   {
      logError("no camera available", getCoreErrorText(MMERR_CameraNotAvailable).c_str());
      throw CMMError(getCoreErrorText(MMERR_CameraNotAvailable).c_str(), MMERR_CameraNotAvailable);
   }

   LOG_DEBUG(coreLogger_) << "Did stop sequence acquisition from current camera";
   // onSequenceAcquisitionStopped will be called by CoreCallback::AcqFinished
}

/**
 * Check if the current camera is acquiring the sequence
 * Returns false when the sequence is done
 */
bool CMMCore::isSequenceRunning() MMCORE_NOEXCEPT
{
   std::shared_ptr<CameraInstance> camera = currentCameraDevice_.lock();
   if (camera)
   {
      try
      {
         mm::DeviceModuleLockGuard guard(camera);
         return camera->IsCapturing();
      }
      catch (const CMMError&) // Possibly uninitialized camera
      {
         // Fall through
      }
   }
   return false;
};

/**
 * Check if the specified camera is acquiring the sequence
 * Returns false when the sequence is done
 */
bool CMMCore::isSequenceRunning(const char* label) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<CameraInstance> pCam =
      deviceManager_->GetDeviceOfType<CameraInstance>(label);

   mm::DeviceModuleLockGuard guard(pCam);
   return pCam->IsCapturing();
};

/**
 * Gets the last image from the circular buffer.
 * Returns 0 if the buffer is empty.
 */
void* CMMCore::getLastImage() MMCORE_LEGACY_THROW(CMMError)
{

   // scope for the thread guard
   {
      MMThreadGuard g(*pPostedErrorsLock_);

      if(0 < postedErrors_.size())
      {
         std::pair< int, std::string>  toThrow(postedErrors_[0]);
         // todo, process the collection of posted errors.
         postedErrors_.clear();
         throw CMMError( toThrow.second.c_str(), toThrow.first);

      }
   }

   unsigned char* pBuf = const_cast<unsigned char*>(cbuf_->GetTopImage());
   if (pBuf != 0)
      return pBuf;
   else
   {
      logError("CMMCore::getLastImage", getCoreErrorText(MMERR_CircularBufferEmpty).c_str());
      throw CMMError(getCoreErrorText(MMERR_CircularBufferEmpty).c_str(), MMERR_CircularBufferEmpty);
   }
}

void* CMMCore::getLastImageMD(unsigned channel, unsigned slice, Metadata& md) const MMCORE_LEGACY_THROW(CMMError)
{
   // Slices have never been implemented on the device interface side
   if (slice != 0)
      throw CMMError("Slice must be 0");

   const mm::ImgBuffer* pBuf = cbuf_->GetTopImageBuffer(channel);
   if (pBuf != 0)
   {
      md = pBuf->GetMetadata();
      return const_cast<unsigned char*>(pBuf->GetPixels());
   }
   else
      throw CMMError(getCoreErrorText(MMERR_CircularBufferEmpty).c_str(), MMERR_CircularBufferEmpty);
}

/**
 * Returns a pointer to the pixels of the image that was last inserted into the circular buffer
 * Also provides all metadata associated with that image
 *
 * Supported data types are byte (8 bits per pixel, 1 component), short 
 * (16 bits per pixel, 1 component), float (32 bits per pixel, 1 component, not
 * supported by the UI yet), RGB_32 (8 bits per component, 4 components), RGB_64
 * (16 bits per component, 4 components, not supported by UI yet).  
 * RGB buffers are expected to be in big endian ARGB format (ARGB8888), which means that
 * on little endian the format is BGRA888 
 * (see: https://en.wikipedia.org/wiki/RGBA_color_model).
 */
void* CMMCore::getLastImageMD(Metadata& md) const MMCORE_LEGACY_THROW(CMMError)
{
   return getLastImageMD(0, 0, md);
}

/**
 * Returns a pointer to the pixels of the image that was inserted n images ago
 * Also provides all metadata associated with that image
 *
 * Supported data types are byte (8 bits per pixel, 1 component), short 
 * (16 bits per pixel, 1 component), float (32 bits per pixel, 1 component, not
 * supported by the UI yet), RGB_32 (8 bits per component, 4 components), RGB_64
 * (16 bits per component, 4 components, not supported by UI yet).  
 * RGB buffers are expected to be in big endian ARGB format (ARGB8888), which means that
 * on little endian the format is BGRA888 
 * (see: https://en.wikipedia.org/wiki/RGBA_color_model).
 */
void* CMMCore::getNBeforeLastImageMD(unsigned long n, Metadata& md) const MMCORE_LEGACY_THROW(CMMError)
{
   const mm::ImgBuffer* pBuf = cbuf_->GetNthFromTopImageBuffer(n);
   if (pBuf != 0)
   {
      md = pBuf->GetMetadata();
      return const_cast<unsigned char*>(pBuf->GetPixels());
   }
   else
      throw CMMError(getCoreErrorText(MMERR_CircularBufferEmpty).c_str(), MMERR_CircularBufferEmpty);
}

/**
 * Gets and removes the next image from the circular buffer.
 * Returns 0 if the buffer is empty.
 *
 * Supported data types are byte (8 bits per pixel, 1 component), short 
 * (16 bits per pixel, 1 component), float (32 bits per pixel, 1 component, not
 * supported by the UI yet), RGB_32 (8 bits per component, 4 components), RGB_64
 * (16 bits per component, 4 components, not supported by UI yet).  
 * RGB buffers are expected to be in big endian ARGB format (ARGB8888), which means that
 * on little endian the format is BGRA888 
 * (see: https://en.wikipedia.org/wiki/RGBA_color_model).
 */
void* CMMCore::popNextImage() MMCORE_LEGACY_THROW(CMMError)
{
   unsigned char* pBuf = const_cast<unsigned char*>(cbuf_->GetNextImage());
   if (pBuf != 0)
      return pBuf;
   else
      throw CMMError(getCoreErrorText(MMERR_CircularBufferEmpty).c_str(), MMERR_CircularBufferEmpty);
}

/**
 * Gets and removes the next image (and metadata) from the circular buffer
 * channel indicates which cameraChannel image should be retrieved.
 * slice has not been implement and should always be 0
 */
void* CMMCore::popNextImageMD(unsigned channel, unsigned slice, Metadata& md) MMCORE_LEGACY_THROW(CMMError)
{
   // Slices have never been implemented on the device interface side
   if (slice != 0)
      throw CMMError("Slice must be 0");

   const mm::ImgBuffer* pBuf = cbuf_->GetNextImageBuffer(channel);
   if (pBuf != 0)
   {
      md = pBuf->GetMetadata();
      return const_cast<unsigned char*>(pBuf->GetPixels());
   }
   else
      throw CMMError(getCoreErrorText(MMERR_CircularBufferEmpty).c_str(), MMERR_CircularBufferEmpty);
}

/**
 * Gets and removes the next image (and metadata) from the circular buffer
 */
void* CMMCore::popNextImageMD(Metadata& md) MMCORE_LEGACY_THROW(CMMError)
{
   return popNextImageMD(0, 0, md);
}

/**
 * Removes all images from the circular buffer.
 *
 * It is rarely necessary to call this directly since starting a sequence
 * acquisition or changing the ROI will always clear the buffer.
 */
void CMMCore::clearCircularBuffer() MMCORE_LEGACY_THROW(CMMError)
{
   cbuf_->Clear();
}

/**
 * Reserve memory for the circular buffer.
 */
void CMMCore::setCircularBufferMemoryFootprint(unsigned sizeMB ///< n megabytes
                                               ) MMCORE_LEGACY_THROW(CMMError)
{
   delete cbuf_; // discard old buffer
   LOG_DEBUG(coreLogger_) << "Will set circular buffer size to " <<
      sizeMB << " MB";
	try
	{
		cbuf_ = new CircularBuffer(sizeMB);
	}
	catch (std::bad_alloc& ex)
	{
      // This is an out-of-memory error before even allocating the buffers.
		std::ostringstream messs;
		messs << getCoreErrorText(MMERR_OutOfMemory).c_str() << " " << ex.what() << '\n';
		throw CMMError(messs.str().c_str() , MMERR_OutOfMemory);
	}
	if (NULL == cbuf_) throw CMMError(getCoreErrorText(MMERR_OutOfMemory).c_str(), MMERR_OutOfMemory);


	try
	{

		// attempt to initialize based on the current camera settings
      std::shared_ptr<CameraInstance> camera = currentCameraDevice_.lock();
      if (camera)
		{
         mm::DeviceModuleLockGuard guard(camera);
         if (!cbuf_->Initialize(camera->GetNumberOfChannels(), camera->GetImageWidth(), camera->GetImageHeight(), camera->GetImageBytesPerPixel()))
				throw CMMError(getCoreErrorText(MMERR_CircularBufferFailedToInitialize).c_str(), MMERR_CircularBufferFailedToInitialize);
		}

      LOG_DEBUG(coreLogger_) << "Did set circular buffer size to " <<
         sizeMB << " MB";
	}
	catch (std::bad_alloc& ex)
	{
		std::ostringstream messs;
		messs << getCoreErrorText(MMERR_OutOfMemory).c_str() << " " << ex.what() << '\n';
		throw CMMError(messs.str().c_str() , MMERR_OutOfMemory);
	}
	if (NULL == cbuf_)
      throw CMMError(getCoreErrorText(MMERR_OutOfMemory).c_str(), MMERR_OutOfMemory);
}

/**
 * Returns the size of the Circular Buffer in MB
 */
unsigned CMMCore::getCircularBufferMemoryFootprint()
{
   if (cbuf_)
   {
      return cbuf_->GetMemorySizeMB();
   }
   return 0;
}

/**
 * Returns number ofimages available in the Circular Buffer
 */
long CMMCore::getRemainingImageCount()
{
   if (cbuf_)
   {
      return cbuf_->GetRemainingImageCount();
   }
   return 0;
}

/**
 * Returns the total number of images that can be stored in the buffer
 */
long CMMCore::getBufferTotalCapacity()
{
   if (cbuf_)
   {
      return cbuf_->GetSize();
   }
   return 0;
}


/**
 * Returns the number of images that can be added to the buffer
 * without overflowing
 */
long CMMCore::getBufferFreeCapacity()
{
   if (cbuf_)
   {
      return cbuf_->GetFreeSize();
   }
   return 0;
}

/**
 * Indicates whether the circular buffer is overflowed
 */
bool CMMCore::isBufferOverflowed() const
{
   return cbuf_->Overflow();
}

/**
 * Returns the label of the currently selected camera device.
 * @return camera name
 */
std::string CMMCore::getCameraDevice()
{
   std::shared_ptr<CameraInstance> camera = currentCameraDevice_.lock();
   if (camera)
   {
      return camera->GetLabel();
   }
   return std::string();
}

/**
 * Returns the label of the currently selected shutter device.
 * @return shutter name
 */
std::string CMMCore::getShutterDevice()
{
   std::shared_ptr<ShutterInstance> shutter = currentShutterDevice_.lock();
   if (shutter)
   {
      return shutter->GetLabel();
   }
   return std::string();
}

/**
 * Returns the label of the currently selected focus device.
 * @return focus stage name
 */
std::string CMMCore::getFocusDevice()
{
   std::shared_ptr<StageInstance> focus = currentFocusDevice_.lock();
   if (focus)
   {
      return focus->GetLabel();
   }
   return std::string();
}

/**
 * Returns the label of the currently selected XYStage device.
 */
std::string CMMCore::getXYStageDevice()
{
   std::shared_ptr<XYStageInstance> xyStage = currentXYStageDevice_.lock();
   if (xyStage)
   {
      return xyStage->GetLabel();
   }
   return std::string();
}

/**
 * Returns the label of the currently selected auto-focus device.
 */
std::string CMMCore::getAutoFocusDevice()
{
   std::shared_ptr<AutoFocusInstance> autofocus =
      currentAutofocusDevice_.lock();
   if (autofocus)
   {
      return autofocus->GetLabel();
   }
   return std::string();
}

/**
 * Sets the current auto-focus device.
 */
void CMMCore::setAutoFocusDevice(const char* autofocusLabel) MMCORE_LEGACY_THROW(CMMError)
{
   if (autofocusLabel && strlen(autofocusLabel)>0)
   {
      currentAutofocusDevice_ =
         deviceManager_->GetDeviceOfType<AutoFocusInstance>(autofocusLabel);
      LOG_INFO(coreLogger_) << "Default autofocus set to " << autofocusLabel;
   }
   else
   {
      currentAutofocusDevice_.reset();
      LOG_INFO(coreLogger_) << "Default autofocus unset";
   }
   std::string newAutofocusLabel = getAutoFocusDevice();
   properties_->Set(MM::g_Keyword_CoreAutoFocus, newAutofocusLabel.c_str());
   {
      MMThreadGuard scg(stateCacheLock_);
      stateCache_.addSetting(PropertySetting(MM::g_Keyword_CoreDevice, MM::g_Keyword_CoreAutoFocus, newAutofocusLabel.c_str()));
   }
}

/**
 * Returns the label of the currently selected image processor device.
 */
std::string CMMCore::getImageProcessorDevice()
{
   std::shared_ptr<ImageProcessorInstance> imageProcessor =
      currentImageProcessor_.lock();
   if (imageProcessor)
   {
      return imageProcessor->GetLabel();
   }
   return std::string();
}

/**
 * Returns the label of the currently selected SLM device.
 * @return slm name
 */
std::string CMMCore::getSLMDevice()
{
   std::shared_ptr<SLMInstance> slm = currentSLMDevice_.lock();
   if (slm)
   {
      return slm->GetLabel();
   }
   return std::string();
}

/**
 * Returns the label of the currently selected Galvo device.
 * @return galvo name
 */
std::string CMMCore::getGalvoDevice()
{
   std::shared_ptr<GalvoInstance> galvos = currentGalvoDevice_.lock();
   if (galvos)
   {
      return galvos->GetLabel();
   }
   return std::string();
}


/**
 * Sets the current image processor device.
 */
void CMMCore::setImageProcessorDevice(const char* procLabel) MMCORE_LEGACY_THROW(CMMError)
{
   if (procLabel && strlen(procLabel)>0)
   {
      currentImageProcessor_ =
         deviceManager_->GetDeviceOfType<ImageProcessorInstance>(procLabel);
      LOG_INFO(coreLogger_) << "Default image processor set to " << procLabel;
   }
   else
   {
      currentImageProcessor_.reset();
      LOG_INFO(coreLogger_) << "Default image processor unset";
   }
   std::string newProcLabel = getImageProcessorDevice();
   properties_->Set(MM::g_Keyword_CoreImageProcessor, newProcLabel.c_str());
   {
      MMThreadGuard scg(stateCacheLock_);
      stateCache_.addSetting(PropertySetting(MM::g_Keyword_CoreDevice, MM::g_Keyword_CoreImageProcessor, newProcLabel.c_str()));
   }
}

/**
 * Sets the current slm device.
 */
void CMMCore::setSLMDevice(const char* slmLabel) MMCORE_LEGACY_THROW(CMMError)
{
   if (slmLabel && strlen(slmLabel)>0)
   {
      currentSLMDevice_ =
         deviceManager_->GetDeviceOfType<SLMInstance>(slmLabel);
      LOG_INFO(coreLogger_) << "Default SLM set to " << slmLabel;
   }
   else
   {
      currentSLMDevice_.reset();
      LOG_INFO(coreLogger_) << "Default SLM unset";
   }
   std::string newSLMLabel = getSLMDevice();
   properties_->Set(MM::g_Keyword_CoreSLM, newSLMLabel.c_str());
   {
      MMThreadGuard scg(stateCacheLock_);
      stateCache_.addSetting(PropertySetting(MM::g_Keyword_CoreDevice, MM::g_Keyword_CoreSLM, newSLMLabel.c_str()));
   }
}


/**
 * Sets the current galvo device.
 */
void CMMCore::setGalvoDevice(const char* galvoLabel) MMCORE_LEGACY_THROW(CMMError)
{
   if (galvoLabel && strlen(galvoLabel)>0)
   {
      currentGalvoDevice_ =
         deviceManager_->GetDeviceOfType<GalvoInstance>(galvoLabel);
      LOG_INFO(coreLogger_) << "Default galvo set to " << galvoLabel;
   }
   else
   {
      currentGalvoDevice_.reset();
      LOG_INFO(coreLogger_) << "Default galvo unset";
   }
   std::string newGalvoLabel = getGalvoDevice();
   properties_->Set(MM::g_Keyword_CoreGalvo, newGalvoLabel.c_str());
   {
      MMThreadGuard scg(stateCacheLock_);
      stateCache_.addSetting(PropertySetting(MM::g_Keyword_CoreDevice, MM::g_Keyword_CoreGalvo, newGalvoLabel.c_str()));
   }
}

/**
 * Specifies the group determining the channel selection.
 */
void CMMCore::setChannelGroup(const char* chGroup) MMCORE_LEGACY_THROW(CMMError)
{
   // Don't do anything if the new channelgroup is the same as the old one
   if (channelGroup_.compare(chGroup) == 0)
   {
      return;
   }

   if (!chGroup)
   {
      chGroup = "";
   }
   
   // CoreProperty checks if this is a valid group, throws CMMError otherwise
   properties_->Set(MM::g_Keyword_CoreChannelGroup, chGroup);
   channelGroup_ = chGroup;
   LOG_INFO(coreLogger_) << "Channel group set to " << chGroup;

   {
      MMThreadGuard scg(stateCacheLock_);
      stateCache_.addSetting(PropertySetting(MM::g_Keyword_CoreDevice, MM::g_Keyword_CoreChannelGroup, channelGroup_.c_str()));
   }
   if (externalCallback_ != 0) 
   {
      externalCallback_->onChannelGroupChanged(channelGroup_.c_str());
   }
}

/**
 * Returns the group determining the channel selection.
 */
std::string CMMCore::getChannelGroup()
{

   return channelGroup_;
}

/**
 * Sets the current shutter device.
 * @param shutterLabel    the shutter device label
 */
void CMMCore::setShutterDevice(const char* shutterLabel) MMCORE_LEGACY_THROW(CMMError)
{
   if (!shutterLabel || strlen(shutterLabel) > 0) // Allow empty label
      CheckDeviceLabel(shutterLabel);

   // Nothing to do if this is the current shutter device:
   if (getShutterDevice().compare(shutterLabel) == 0)
      return;

   // To avoid confusion close the current shutter:
   bool shutterWasOpen = false;
   std::shared_ptr<ShutterInstance> oldShutter =
      currentShutterDevice_.lock();
   if (oldShutter)
   {
      shutterWasOpen = getShutterOpen(oldShutter->GetLabel().c_str());
      if (shutterWasOpen)
      {
         setShutterOpen(oldShutter->GetLabel().c_str(), false);
      }
   }

   if (strlen(shutterLabel) > 0)
   {
      currentShutterDevice_ =
         deviceManager_->GetDeviceOfType<ShutterInstance>(shutterLabel);

      if (shutterWasOpen)
         setShutterOpen(true);

      LOG_INFO(coreLogger_) << "Default shutter set to " << shutterLabel;
   }
   else
   {
      currentShutterDevice_.reset();
      LOG_INFO(coreLogger_) << "Default shutter unset";
   }
   std::string newShutterLabel = getShutterDevice();
   properties_->Set(MM::g_Keyword_CoreShutter, newShutterLabel.c_str());
   {
      MMThreadGuard scg(stateCacheLock_);
      stateCache_.addSetting(PropertySetting(MM::g_Keyword_CoreDevice, MM::g_Keyword_CoreShutter, newShutterLabel.c_str()));
   }
}

/**
 * Sets the current focus device.
 * @param focusLabel    the focus stage device label
 */
void CMMCore::setFocusDevice(const char* focusLabel) MMCORE_LEGACY_THROW(CMMError)
{
   if (focusLabel && strlen(focusLabel)>0)
   {
      currentFocusDevice_ =
         deviceManager_->GetDeviceOfType<StageInstance>(focusLabel);
      LOG_INFO(coreLogger_) << "Default stage set to " << focusLabel;
   }
   else
   {
      currentFocusDevice_.reset();
      LOG_INFO(coreLogger_) << "Default stage unset";
   }
   std::string newFocusLabel = getFocusDevice();
   properties_->Set(MM::g_Keyword_CoreFocus, newFocusLabel.c_str());
   {
      MMThreadGuard scg(stateCacheLock_);
      stateCache_.addSetting(PropertySetting(MM::g_Keyword_CoreDevice, MM::g_Keyword_CoreFocus, newFocusLabel.c_str()));
   }
}

/**
 * Sets the current XY device.
 */
void CMMCore::setXYStageDevice(const char* xyDeviceLabel) MMCORE_LEGACY_THROW(CMMError)
{
   if (xyDeviceLabel && strlen(xyDeviceLabel)>0)
   {
      currentXYStageDevice_ =
         deviceManager_->GetDeviceOfType<XYStageInstance>(xyDeviceLabel);
      LOG_INFO(coreLogger_) << "Default xy stage set to " << xyDeviceLabel;
   }
   else
   {
      currentXYStageDevice_.reset();
      LOG_INFO(coreLogger_) << "Default xy stage unset";
   }
   std::string newXYStageLabel = getXYStageDevice();
   properties_->Set(MM::g_Keyword_CoreXYStage, newXYStageLabel.c_str());
   {
      MMThreadGuard scg(stateCacheLock_);
      stateCache_.addSetting(PropertySetting(MM::g_Keyword_CoreDevice, MM::g_Keyword_CoreXYStage, newXYStageLabel.c_str()));
   }
}

/**
 * Sets the current camera device.
 * @param cameraLabel   the camera device label
 */
void CMMCore::setCameraDevice(const char* cameraLabel) MMCORE_LEGACY_THROW(CMMError)
{
   // If a sequence acquisition is running, the camera cannot be switched. (In
   // order to start sequences for multiple cameras, one must instead use the
   // version of startSequenceAcquisition() that takes the camera label.)

   // Note: there is a blatant race condition between this and the
   // starting/stopping of sequence acquisitions. This is hard to fix it at the
   // moment, as we would need a way to safely lock two cameras at the same
   // time.
   if (isSequenceRunning())
   {
      throw CMMError("Cannot switch camera device while sequence acquisition "
            "is running");
   }

   if (cameraLabel && strlen(cameraLabel) > 0)
   {
      currentCameraDevice_ =
         deviceManager_->GetDeviceOfType<CameraInstance>(cameraLabel);
      LOG_INFO(coreLogger_) << "Default camera set to " << cameraLabel;
   }
   else
   {
      currentCameraDevice_.reset();
      LOG_INFO(coreLogger_) << "Default camera unset";
   }
   std::string newCameraLabel = getCameraDevice();
   properties_->Set(MM::g_Keyword_CoreCamera, newCameraLabel.c_str());
   {
      MMThreadGuard scg(stateCacheLock_);
      stateCache_.addSetting(PropertySetting(MM::g_Keyword_CoreDevice, MM::g_Keyword_CoreCamera, newCameraLabel.c_str()));
   }
}

/**
 * Returns all property names supported by the device.
 *
 * @return property name array
 * @param label    the device label
 */
std::vector<std::string> CMMCore::getDevicePropertyNames(const char* label) MMCORE_LEGACY_THROW(CMMError)
{
   if (IsCoreDeviceLabel(label))
      return properties_->GetNames();
   std::shared_ptr<DeviceInstance> pDevice = deviceManager_->GetDevice(label);

   {
      mm::DeviceModuleLockGuard guard(pDevice);
      return pDevice->GetPropertyNames();
   }
}

/**
 * Returns an array of labels for currently loaded devices.
 * @return array of labels
 */
std::vector<std::string> CMMCore::getLoadedDevices() const
{
  std::vector<std::string> deviceList = deviceManager_->GetDeviceList();
  deviceList.push_back(MM::g_Keyword_CoreDevice);
  return deviceList;
}

/**
 * Returns an array of labels for currently loaded devices of specific type.
 * @param devType    the device type identifier
 * @return array of labels
 */
std::vector<std::string> CMMCore::getLoadedDevicesOfType(MM::DeviceType devType) const
{
   if (devType == MM::CoreDevice) {
      std::vector<std::string> coreDev;
      coreDev.push_back(MM::g_Keyword_CoreDevice);
      return coreDev;
   }

   return deviceManager_->GetDeviceList(devType);
}

/**
 * Returns all valid values for the specified property.
 * If the array is empty it means that there are no restrictions for values.
 * However, even if all values are allowed it is not guaranteed that all of them will be
 * actually accepted by the device at run time.
 *
 * @return the array of values
 * @param label     the device label
 * @param propName  the property name
 */
std::vector<std::string> CMMCore::getAllowedPropertyValues(const char* label, const char* propName) MMCORE_LEGACY_THROW(CMMError)
{
   if (IsCoreDeviceLabel(label))
      return properties_->GetAllowedValues(propName);

   std::shared_ptr<DeviceInstance> pDevice = deviceManager_->GetDevice(label);
   CheckPropertyName(propName);

   std::vector<std::string> valueList;

   {
      mm::DeviceModuleLockGuard guard(pDevice);
      unsigned nrValues = pDevice->GetNumberOfPropertyValues(propName);
      valueList.reserve(nrValues);
      for (unsigned i = 0; i < nrValues; ++i)
      {
         valueList.push_back(pDevice->GetPropertyValueAt(propName, i));
      }
   }

   return valueList;
}

/**
 * Returns the property value for the specified device.

 * @return the property value
 * @param label      the device label
 * @param propName   the property name
 */
std::string CMMCore::getProperty(const char* label, const char* propName) MMCORE_LEGACY_THROW(CMMError)
{
   if (IsCoreDeviceLabel(label))
      return properties_->Get(propName);
   std::shared_ptr<DeviceInstance> pDevice = deviceManager_->GetDevice(label);
   CheckPropertyName(propName);

   mm::DeviceModuleLockGuard guard(pDevice);
   std::string value = pDevice->GetProperty(propName);

   // use the opportunity to update the cache
   // Note, stateCache is mutable so that we can update it from this const function
   PropertySetting s(label, propName, value.c_str());
   {
      MMThreadGuard scg(stateCacheLock_);
      stateCache_.addSetting(s);
   }

   return value;
}

/**
 * Returns the cached property value for the specified device.

 * @return the property value
 * @param label       the device label
 * @param propName    the property name
 */
std::string CMMCore::getPropertyFromCache(const char* label, const char* propName) const MMCORE_LEGACY_THROW(CMMError)
{
   if (IsCoreDeviceLabel(label))
      return properties_->Get(propName);
   CheckDeviceLabel(label);
   CheckPropertyName(propName);

   {
      MMThreadGuard scg(stateCacheLock_);
      if (!stateCache_.isPropertyIncluded(label, propName))
         throw CMMError("Property " + ToQuotedString(propName) + " of device " +
               ToQuotedString(label) + " not found in cache",
               MMERR_PropertyNotInCache);
      PropertySetting s = stateCache_.getSetting(label, propName);
      return s.getPropertyValue();
   }
}

/**
 * Changes the value of the device property.
 *
 * @param label       the device label
 * @param propName    the property name
 * @param propValue   the new property value
 */
void CMMCore::setProperty(const char* label, const char* propName,
                          const char* propValue) MMCORE_LEGACY_THROW(CMMError)
{
   CheckDeviceLabel(label);
   CheckPropertyName(propName);
   CheckPropertyValue(propValue);

   if (IsCoreDeviceLabel(label))
   {
      LOG_DEBUG(coreLogger_) << "Will set Core property: " <<
         propName << " = " << propValue;

      properties_->Execute(propName, propValue);
      {
         MMThreadGuard scg(stateCacheLock_);
         stateCache_.addSetting(PropertySetting(MM::g_Keyword_CoreDevice, propName, propValue));
      }

      LOG_DEBUG(coreLogger_) << "Did set Core property: " <<
         propName << " = " << propValue;
   }
   else
   {
      std::shared_ptr<DeviceInstance> pDevice = deviceManager_->GetDevice(label);

      mm::DeviceModuleLockGuard guard(pDevice);

      pDevice->SetProperty(propName, propValue);

      {
         MMThreadGuard scg(stateCacheLock_);
         stateCache_.addSetting(PropertySetting(label, propName, propValue));
      }
   }
}

/**
 * Changes the value of the device property.
 *
 * @param label        the device label
 * @param propName     property name
 * @param propValue    the new property value
 */
void CMMCore::setProperty(const char* label, const char* propName,
                          const bool propValue) MMCORE_LEGACY_THROW(CMMError)
{
   setProperty(label, propName, (propValue ? "1" : "0"));
}

/**
 * Changes the value of the device property.
 *
 * @param label      the device label
 * @param propName   the property name
 * @param propValue  the new property value
 */
void CMMCore::setProperty(const char* label, const char* propName,
                          const long propValue) MMCORE_LEGACY_THROW(CMMError)
{
   setProperty(label, propName, ToString(propValue).c_str());
}

/**
 * Changes the value of the device property.
 *
 * @param label      the device label
 * @param propName   the property name
 * @param propValue  the new property value
 */
void CMMCore::setProperty(const char* label, const char* propName,
                          const float propValue) MMCORE_LEGACY_THROW(CMMError)
{
   setProperty(label, propName, ToString(propValue).c_str());
}

/**
 * Changes the value of the device property.
 *
 * @param label          the device label
 * @param propName       the property name
 * @param propValue      the new property value
 */
void CMMCore::setProperty(const char* label, const char* propName,
                          const double propValue) MMCORE_LEGACY_THROW(CMMError)
{
   setProperty(label, propName, ToString(propValue).c_str());
}


/**
 * Checks if device has a property with a specified name.
 * The exception will be thrown in case device label is not defined.
 */
bool CMMCore::hasProperty(const char* label, const char* propName) MMCORE_LEGACY_THROW(CMMError)
{
   if (IsCoreDeviceLabel(label))
      return properties_->Has(propName);
   std::shared_ptr<DeviceInstance> pDevice = deviceManager_->GetDevice(label);
   CheckPropertyName(propName);

   mm::DeviceModuleLockGuard guard(pDevice);
   return pDevice->HasProperty(propName);
}

/**
 * Tells us whether the property can be modified.
 *
 * @return true for a read-only property
 * @param label    the device label
 * @param propName the property name
 */
bool CMMCore::isPropertyReadOnly(const char* label, const char* propName) MMCORE_LEGACY_THROW(CMMError)
{
   if (IsCoreDeviceLabel(label))
      return properties_->IsReadOnly(propName);
   std::shared_ptr<DeviceInstance> pDevice = deviceManager_->GetDevice(label);
   CheckPropertyName(propName);

   mm::DeviceModuleLockGuard guard(pDevice);
   return pDevice->GetPropertyReadOnly(propName);
}

/**
 * Tells us whether the property must be defined prior to initialization.
 *
 * @return true for pre-init property
 * @param label      the device label
 * @param propName   the property name
 */
bool CMMCore::isPropertyPreInit(const char* label, const char* propName) MMCORE_LEGACY_THROW(CMMError)
{
   if (IsCoreDeviceLabel(label))
      return false;
   std::shared_ptr<DeviceInstance> pDevice = deviceManager_->GetDevice(label);
   CheckPropertyName(propName);

   mm::DeviceModuleLockGuard guard(pDevice);
   return pDevice->GetPropertyInitStatus(propName);
}

/**
 * Returns the property lower limit value, if the property has limits - 0 otherwise.
 */
double CMMCore::getPropertyLowerLimit(const char* label, const char* propName) MMCORE_LEGACY_THROW(CMMError)
{
   if (IsCoreDeviceLabel(label))
      return 0.0;
   std::shared_ptr<DeviceInstance> pDevice = deviceManager_->GetDevice(label);
   CheckPropertyName(propName);

   mm::DeviceModuleLockGuard guard(pDevice);
   return pDevice->GetPropertyLowerLimit(propName);
}

/**
 * Returns the property upper limit value, if the property has limits - 0 otherwise.
 */
double CMMCore::getPropertyUpperLimit(const char* label, const char* propName) MMCORE_LEGACY_THROW(CMMError)
{
   if (IsCoreDeviceLabel(label))
      return 0.0;
   std::shared_ptr<DeviceInstance> pDevice = deviceManager_->GetDevice(label);
   CheckPropertyName(propName);

   mm::DeviceModuleLockGuard guard(pDevice);
   return pDevice->GetPropertyUpperLimit(propName);
}

/**
 * Queries device if the specific property has limits.
 * @param label      the device name
 * @param propName   the property label
 */
bool CMMCore::hasPropertyLimits(const char* label, const char* propName) MMCORE_LEGACY_THROW(CMMError)
{
   if (IsCoreDeviceLabel(label))
      return false;
   std::shared_ptr<DeviceInstance> pDevice = deviceManager_->GetDevice(label);
   CheckPropertyName(propName);

   mm::DeviceModuleLockGuard guard(pDevice);
   return pDevice->HasPropertyLimits(propName);
}

/**
 * Queries device if the specified property can be used in a sequence
 * @param label      the device name
 * @param propName   the property label
 */
bool CMMCore::isPropertySequenceable(const char* label, const char* propName) MMCORE_LEGACY_THROW(CMMError)
{
   if (IsCoreDeviceLabel(label))
      return false;
   std::shared_ptr<DeviceInstance> pDevice = deviceManager_->GetDevice(label);
   CheckPropertyName(propName);

   mm::DeviceModuleLockGuard guard(pDevice);
   return pDevice->IsPropertySequenceable(propName);
}


/**
 * Queries device property for the maximum number of events that can be put in a sequence
 * @param label      the device name
 * @param propName   the property label
 */
long CMMCore::getPropertySequenceMaxLength(const char* label, const char* propName) MMCORE_LEGACY_THROW(CMMError)
{
   if (IsCoreDeviceLabel(label))
      return 0;
   std::shared_ptr<DeviceInstance> pDevice = deviceManager_->GetDevice(label);
   CheckPropertyName(propName);

   mm::DeviceModuleLockGuard guard(pDevice);
   return pDevice->GetPropertySequenceMaxLength(propName);
}


/**
 * Starts an ongoing sequence of triggered events in a property of a device
 * This should only be called for device-properties that are sequenceable
 * @param label      the device name
 * @param propName   the property label
 */
void CMMCore::startPropertySequence(const char* label, const char* propName) MMCORE_LEGACY_THROW(CMMError)
{
   if (IsCoreDeviceLabel(label))
      // XXX Should be a throw
      return;
   std::shared_ptr<DeviceInstance> pDevice = deviceManager_->GetDevice(label);
   CheckPropertyName(propName);

   mm::DeviceModuleLockGuard guard(pDevice);
   pDevice->StartPropertySequence(propName);
}

/**
 * Stops an ongoing sequence of triggered events in a property of a device
 * This should only be called for device-properties that are sequenceable
 * @param label     the device label
 * @param propName  the property name
 */
void CMMCore::stopPropertySequence(const char* label, const char* propName) MMCORE_LEGACY_THROW(CMMError)
{
   if (IsCoreDeviceLabel(label))
      // XXX Should be a throw
      return;
   std::shared_ptr<DeviceInstance> pDevice = deviceManager_->GetDevice(label);
   CheckPropertyName(propName);

   mm::DeviceModuleLockGuard guard(pDevice);
   pDevice->StopPropertySequence(propName);
}

/**
 * Transfer a sequence of events/states/whatever to the device
 * This should only be called for device-properties that are sequenceable
 * @param label           the device name
 * @param propName        the property label
 * @param eventSequence   the sequence of events/states that the device will execute in response to external triggers
 */
void CMMCore::loadPropertySequence(const char* label, const char* propName, std::vector<std::string> eventSequence) MMCORE_LEGACY_THROW(CMMError)
{
   if (IsCoreDeviceLabel(label))
      // XXX Should be a throw
      return;
   std::shared_ptr<DeviceInstance> pDevice = deviceManager_->GetDevice(label);
   CheckPropertyName(propName);

   mm::DeviceModuleLockGuard guard(pDevice);
   pDevice->ClearPropertySequence(propName);

   for (std::vector<std::string>::const_iterator it = eventSequence.begin(),
         end = eventSequence.end();
         it < end; ++it)
   {
      CheckPropertyValue(it->c_str());
      pDevice->AddToPropertySequence(propName, it->c_str());
   }

   pDevice->SendPropertySequence(propName);
}

/**
 * Returns the intrinsic property type.
 */
MM::PropertyType CMMCore::getPropertyType(const char* label, const char* propName) MMCORE_LEGACY_THROW(CMMError)
{
   if (IsCoreDeviceLabel(label))
   {
      return properties_->GetPropertyType(propName);
   }
   std::shared_ptr<DeviceInstance> pDevice = deviceManager_->GetDevice(label);
   CheckPropertyName(propName);

   mm::DeviceModuleLockGuard guard(pDevice);
   return pDevice->GetPropertyType(propName);
}


/**
 * Horizontal dimension of the image buffer in pixels.
 * @return   the width in pixels (an integer)
 */
unsigned CMMCore::getImageWidth()
{
   std::shared_ptr<CameraInstance> camera = currentCameraDevice_.lock();
   if (camera)
   {
      try
      {
         mm::DeviceModuleLockGuard guard(camera);
         return camera->GetImageWidth();
      }
      catch (const CMMError&) // Possibly uninitialized camera
      {
		 // Fall through
      }
   }
   return 0;
}

/**
 * Vertical dimension of the image buffer in pixels.
 * @return   the height in pixels (an integer)
 */
unsigned CMMCore::getImageHeight()
{
   std::shared_ptr<CameraInstance> camera = currentCameraDevice_.lock();
   if (camera)
   {
      try
      {
         mm::DeviceModuleLockGuard guard(camera);
         return camera->GetImageHeight();
      }
      catch (const CMMError&) // Possibly uninitialized camera
      {
         // Fall through
      }
   }
   return 0;
}

/**
 * How many bytes for each pixel. This value does not necessarily reflect the
 * capabilities of the particular camera A/D converter.
 * @return the number of bytes
 */
unsigned CMMCore::getBytesPerPixel()
{
   std::shared_ptr<CameraInstance> camera = currentCameraDevice_.lock();
   if (camera)
   {
      try
      {
         mm::DeviceModuleLockGuard guard(camera);
         return camera->GetImageBytesPerPixel();
      }
      catch (const CMMError&) // Possibly uninitialized camera
      {
         // Fall through
      }
   }
   return 0;
}

/**
 * How many bits of dynamic range are to be expected from the camera. This value should
 * be used only as a guideline - it does not guarantee that image buffer will contain
 * only values from the returned dynamic range.
 *
 * @return the number of bits
 */
unsigned CMMCore::getImageBitDepth()
{
   std::shared_ptr<CameraInstance> camera = currentCameraDevice_.lock();
   if (camera)
   {
      try
      {
         mm::DeviceModuleLockGuard guard(camera);
         return camera->GetBitDepth();
      }
      catch (const CMMError&) // Possibly uninitialized camera
      {
         // Fall through
      }
   }
   return 0;
}

/**
 * Returns the number of components the default camera is returning.
 * For example color camera will return 4 components (RGBA) on each snap.
 */
unsigned CMMCore::getNumberOfComponents()
{
   std::shared_ptr<CameraInstance> camera = currentCameraDevice_.lock();
   if (camera)
   {
      try
      {
         mm::DeviceModuleLockGuard guard(camera);
         return camera->GetNumberOfComponents();
      }
      catch (const CMMError&) // Possibly uninitialized camera
      {
         // Fall through
      }
   }
   return 0;
}

/**
 * Returns the number of simultaneous channels the default camera is returning.
 */
unsigned CMMCore::getNumberOfCameraChannels()
{
   std::shared_ptr<CameraInstance> camera = currentCameraDevice_.lock();
   if (camera)
   {
      try
      {
         mm::DeviceModuleLockGuard guard(camera);
         return camera->GetNumberOfChannels();
      }
      catch (const CMMError&) // Possibly uninitialized camera
      {
         // Fall through
      }
   }
   return 0;
}

/**
 * Returns the name of the requested channel as known by the default camera
 */
std::string CMMCore::getCameraChannelName(unsigned int channelNr)
{
   std::shared_ptr<CameraInstance> camera = currentCameraDevice_.lock();
   if (camera)
   {
      try
      {
         mm::DeviceModuleLockGuard guard(camera);
         return camera->GetChannelName(channelNr);
      }
      catch (const CMMError&) // Possibly uninitialized camera
      {
         // Fall through
      }
   }
   return std::string();
}

/**
 * Sets the exposure setting of the current camera in milliseconds.
 * @param dExp   the exposure in milliseconds
 */
void CMMCore::setExposure(double dExp) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<CameraInstance> camera = currentCameraDevice_.lock();
   if (!camera)
   {
      throw CMMError(getCoreErrorText(MMERR_CameraNotAvailable).c_str(), MMERR_CameraNotAvailable);
   }
   else
   {
      std::string cameraName;
      {
         mm::DeviceModuleLockGuard guard(camera);
         cameraName = camera->GetLabel();
      }
      setExposure(cameraName.c_str(), dExp);
   }
}

/**
 * Sets the exposure setting of the specified camera in milliseconds.
 * @param label  the camera device label
 * @param dExp   the exposure in milliseconds
 */
void CMMCore::setExposure(const char* label, double dExp) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<CameraInstance> pCamera =
      deviceManager_->GetDeviceOfType<CameraInstance>(label);

   {
      mm::DeviceModuleLockGuard guard(pCamera);
      LOG_DEBUG(coreLogger_) << "Will set camera " << label <<
         " exposure to " << std::fixed << std::setprecision(3) <<
         dExp << " ms";

      pCamera->SetExposure(dExp);
      if (pCamera->HasProperty(MM::g_Keyword_Exposure))
      {
         {
            MMThreadGuard scg(stateCacheLock_);
            stateCache_.addSetting(PropertySetting(label, MM::g_Keyword_Exposure, CDeviceUtils::ConvertToString(dExp)));
         }
      }
   }

   LOG_DEBUG(coreLogger_) << "Did set camera " << label <<
      " exposure to " << std::fixed << std::setprecision(3) <<
      dExp << " ms";
}

/**
 * Returns the current exposure setting of the camera in milliseconds.
 * @return the exposure time in milliseconds
 */
double CMMCore::getExposure() MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<CameraInstance> camera = currentCameraDevice_.lock();
   if (camera)
   {
      mm::DeviceModuleLockGuard guard(camera);
      return camera->GetExposure();
   }
   else
      //throw CMMError(getCoreErrorText(MMERR_CameraNotAvailable).c_str(), MMERR_CameraNotAvailable);
      return 0.0;
}

/**
* Returns the current exposure setting of the specified camera in milliseconds.
* @param label  the camera device label
* @return the exposure time in milliseconds
*/
double CMMCore::getExposure(const char* label) MMCORE_LEGACY_THROW(CMMError)
{
  std::shared_ptr<CameraInstance> pCamera =
        deviceManager_->GetDeviceOfType<CameraInstance>(label);
  if (pCamera)
  {
     mm::DeviceModuleLockGuard guard(pCamera);
     return pCamera->GetExposure();
  }
  else
     return 0.0;
}

/**
 * Set the hardware region of interest for the current camera.
 *
 * A successful call to this method will clear any images in the sequence
 * buffer, even if the ROI does not change.
 *
 * If multiple ROIs are set prior to this call, they will be replaced by the
 * new single ROI.
 *
 * The coordinates are in units of binned pixels. That is, conceptually,
 * binning is applied before the ROI.
 *
 * @param x      coordinate of the top left corner
 * @param y      coordinate of the top left corner
 * @param xSize  number of horizontal pixels
 * @param ySize  number of horizontal pixels
 */
void CMMCore::setROI(int x, int y, int xSize, int ySize) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<CameraInstance> camera = currentCameraDevice_.lock();
   if (camera)
   {
      mm::DeviceModuleLockGuard guard(camera);
      LOG_DEBUG(coreLogger_) << "Will set ROI of current camera to ("
         "left = " << x << ", top = " << y <<
         ", width = " << xSize << ", height = " << ySize << ")";
      int nRet = camera->SetROI(x, y, xSize, ySize);
      if (nRet != DEVICE_OK)
         throw CMMError(getDeviceErrorText(nRet, camera).c_str(), MMERR_DEVICE_GENERIC);

      // Any images left over in the sequence buffer may have sizes
      // inconsistent with the current image size. There is no way to "fix"
      // popNextImage() to handle this correctly, so we need to make sure we
      // discard such images.
      cbuf_->Clear();
   }
   else
      throw CMMError(getCoreErrorText(MMERR_CameraNotAvailable).c_str(), MMERR_CameraNotAvailable);

   LOG_DEBUG(coreLogger_) << "Did set ROI of current camera to ("
      "left = " << x << ", top = " << y <<
      ", width = " << xSize << ", height = " << ySize << ")";
}

/**
 * Return the current hardware region of interest for a camera. If multiple
 * ROIs are set, this method instead returns a rectangle that describes the
 * image that the camera will generate.
 *
 * The coordinates are in units of binned pixels. That is, conceptually,
 * binning is applied before the ROI.
 *
 * @param x      coordinate of the top left corner
 * @param y      coordinate of the top left corner
 * @param xSize  number of horizontal pixels
 * @param ySize  number of horizontal pixels
 */
void CMMCore::getROI(int& x, int& y, int& xSize, int& ySize) MMCORE_LEGACY_THROW(CMMError)
{
   unsigned uX(0), uY(0), uXSize(0), uYSize(0);
   std::shared_ptr<CameraInstance> camera = currentCameraDevice_.lock();
   if (camera)
   {
      mm::DeviceModuleLockGuard guard(camera);
      int nRet = camera->GetROI(uX, uY, uXSize, uYSize);
      if (nRet != DEVICE_OK)
         throw CMMError(getDeviceErrorText(nRet, camera).c_str(), MMERR_DEVICE_GENERIC);
   }

   x = (int) uX;
   y = (int) uY;
   xSize = (int) uXSize;
   ySize = (int) uYSize;
}

/**
* Set the hardware region of interest for a specified camera.
*
* A successful call to this method will clear any images in the sequence
* buffer, even if the ROI does not change.
*
* Warning: the clearing of the sequence buffer will interfere with any sequence
* acquisitions currently being performed on other cameras.
*
* If multiple ROIs are set prior to this call, they will be replaced by the
* new single ROI.
*
* The coordinates are in units of binned pixels. That is, conceptually,
* binning is applied before the ROI.
*
* @param label  camera label
* @param x      coordinate of the top left corner
* @param y      coordinate of the top left corner
* @param xSize  number of horizontal pixels
* @param ySize  number of horizontal pixels
*/
void CMMCore::setROI(const char* label, int x, int y, int xSize, int ySize) MMCORE_LEGACY_THROW(CMMError)
{
  std::shared_ptr<CameraInstance> camera = deviceManager_->GetDeviceOfType<CameraInstance>(label);
  if (camera)
  {
     mm::DeviceModuleLockGuard guard(camera);
     LOG_DEBUG(coreLogger_) << "Will set ROI of camera " << label <<
        " to (left = " << x << ", top = " << y <<
        ", width = " << xSize << ", height = " << ySize << ")";
     int nRet = camera->SetROI(x, y, xSize, ySize);
     if (nRet != DEVICE_OK)
        throw CMMError(getDeviceErrorText(nRet, camera).c_str(), MMERR_DEVICE_GENERIC);

     // Any images left over in the sequence buffer may have sizes
     // inconsistent with the current image size. There is no way to "fix"
     // popNextImage() to handle this correctly, so we need to make sure we
     // discard such images.
     cbuf_->Clear();
  }
  else
     throw CMMError(getCoreErrorText(MMERR_CameraNotAvailable).c_str(), MMERR_CameraNotAvailable);

  LOG_DEBUG(coreLogger_) << "Did set ROI of camera " << label <<
     " to (left = " << x << ", top = " << y <<
     ", width = " << xSize << ", height = " << ySize << ")";
}

/**
 * Return the current hardware region of interest for a camera. If multiple
 * ROIs are set, this method instead returns a rectangle that describes the
 * image that the camera will generate.
 *
 * @param label  camera label
 * @param x      coordinate of the top left corner
 * @param y      coordinate of the top left corner
 * @param xSize  number of horizontal pixels
 * @param ySize  number of vertical pixels
 */
void CMMCore::getROI(const char* label, int& x, int& y, int& xSize, int& ySize) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<CameraInstance> pCam =
      deviceManager_->GetDeviceOfType<CameraInstance>(label);

   unsigned uX(0), uY(0), uXSize(0), uYSize(0);
   mm::DeviceModuleLockGuard guard(pCam);
   int nRet = pCam->GetROI(uX, uY, uXSize, uYSize);
   if (nRet != DEVICE_OK)
      throw CMMError(getDeviceErrorText(nRet, pCam).c_str(), MMERR_DEVICE_GENERIC);

   x = (int) uX;
   y = (int) uY;
   xSize = (int) uXSize;
   ySize = (int) uYSize;
}

/**
 * Set the region of interest of the current camera to the full frame.
 *
 * A successful call to this method will clear any images in the sequence
 * buffer, even if the ROI does not change.
 */
void CMMCore::clearROI() MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<CameraInstance> camera = currentCameraDevice_.lock();
   if (camera)
   {
      // effectively clears the current ROI setting
      mm::DeviceModuleLockGuard guard(camera);
      int nRet = camera->ClearROI();
      if (nRet != DEVICE_OK)
         throw CMMError(getDeviceErrorText(nRet, camera).c_str(), MMERR_DEVICE_GENERIC);

      // Any images left over in the sequence buffer may have sizes
      // inconsistent with the current image size. There is no way to "fix"
      // popNextImage() to handle this correctly, so we need to make sure we
      // discard such images.
      cbuf_->Clear();
   }
}

/**
 * Queries the camera to determine if it supports multiple ROIs.
 */
bool CMMCore::isMultiROISupported() MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<CameraInstance> camera = currentCameraDevice_.lock();
   if (!camera)
   {
      throw CMMError(getCoreErrorText(MMERR_CameraNotAvailable).c_str(), MMERR_CameraNotAvailable);
   }
   mm::DeviceModuleLockGuard guard(camera);
   return camera->SupportsMultiROI();
}

/**
 * Queries the camera to determine if multiple ROIs are currently set.
 */
bool CMMCore::isMultiROIEnabled() MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<CameraInstance> camera = currentCameraDevice_.lock();
   if (!camera)
   {
      throw CMMError(getCoreErrorText(MMERR_CameraNotAvailable).c_str(), MMERR_CameraNotAvailable);
   }
   mm::DeviceModuleLockGuard guard(camera);
   return camera->IsMultiROISet();
}

/**
 * Set multiple ROIs for the current camera device. Will fail if the camera
 * does not support multiple ROIs, any widths or heights are non-positive,
 * or if the vectors do not all have the same length.
 *
 * @param xs X indices for the upper-left corners of each ROI.
 * @param ys Y indices for the upper-left corners of each ROI.
 * @param widths Width in pixels for each ROI.
 * @param heights Height in pixels for each ROI.
 */
void CMMCore::setMultiROI(std::vector<unsigned> xs, std::vector<unsigned> ys,
      std::vector<unsigned> widths,
      std::vector<unsigned> heights) MMCORE_LEGACY_THROW(CMMError)
{
   if (xs.size() != ys.size() ||
	   xs.size() != widths.size() ||
	   xs.size() != heights.size())
   {
	   throw CMMError("Inconsistent ROI parameter lengths");
   }
   std::shared_ptr<CameraInstance> camera = currentCameraDevice_.lock();
   if (!camera)
   {
      throw CMMError(getCoreErrorText(MMERR_CameraNotAvailable).c_str(), MMERR_CameraNotAvailable);
   }
   mm::DeviceModuleLockGuard guard(camera);
   const unsigned numROI = (unsigned) xs.size();
   int nRet = camera->SetMultiROI(xs.data(), ys.data(),
                                  widths.data(), heights.data(),
                                  numROI);
   if (nRet != DEVICE_OK)
   {
      throw CMMError(getDeviceErrorText(nRet, camera).c_str(), MMERR_DEVICE_GENERIC);
   }
}

/**
 * Get multiple ROIs from the current camera device. Will fail if the camera
 * does not support multiple ROIs. Will return empty vectors if multiple ROIs
 * are not currently being used.
 * @param xs (Return value) X indices for the upper-left corners of each ROI.
 * @param ys (Return value) Y indices for the upper-left corners of each ROI.
 * @param widths (Return value) Width in pixels for each ROI.
 * @param heights (Return value) Height in pixels for each ROI.
 */
void CMMCore::getMultiROI(std::vector<unsigned>& xs, std::vector<unsigned>& ys,
      std::vector<unsigned>& widths,
      std::vector<unsigned>& heights) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<CameraInstance> camera = currentCameraDevice_.lock();
   if (!camera)
   {
      throw CMMError(getCoreErrorText(MMERR_CameraNotAvailable).c_str(), MMERR_CameraNotAvailable);
   }
   mm::DeviceModuleLockGuard guard(camera);
   unsigned numROI;
   int nRet = camera->GetMultiROICount(numROI);
   if (nRet != DEVICE_OK)
   {
      throw CMMError(getDeviceErrorText(nRet, camera).c_str(), MMERR_DEVICE_GENERIC);
   }

   std::vector<unsigned> xsTmp(numROI);
   std::vector<unsigned> ysTmp(numROI);
   std::vector<unsigned> widthsTmp(numROI);
   std::vector<unsigned> heightsTmp(numROI);
   unsigned newNum = numROI;
   nRet = camera->GetMultiROI(xsTmp.data(), ysTmp.data(),
                              widthsTmp.data(), heightsTmp.data(),
                              &newNum);
   if (nRet != DEVICE_OK)
   {
      throw CMMError(getDeviceErrorText(nRet, camera).c_str(), MMERR_DEVICE_GENERIC);
   }
   if (newNum > numROI)
   {
      // Camera returned more ROIs than can fit into the arrays we provided.
      throw CMMError("Camera returned too many ROIs");
   }

   xs.swap(xsTmp);
   ys.swap(ysTmp);
   widths.swap(widthsTmp);
   heights.swap(heightsTmp);
}

/**
 * Sets the state (position) on the specific device. The command will fail if
 * the device does not support states.
 *
 * @param deviceLabel  the device label
 * @param state        the new state
 */
void CMMCore::setState(const char* deviceLabel, long state) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<StateInstance> pStateDev =
      deviceManager_->GetDeviceOfType<StateInstance>(deviceLabel);
   mm::DeviceModuleLockGuard guard(pStateDev);

   LOG_DEBUG(coreLogger_) << "Will set " << deviceLabel << " to state " << state;
   int nRet = pStateDev->SetPosition(state);
   if (nRet != DEVICE_OK)
      throw CMMError(getDeviceErrorText(nRet, pStateDev));

   if (pStateDev->HasProperty(MM::g_Keyword_State))
   {
      {
         MMThreadGuard scg(stateCacheLock_);
         stateCache_.addSetting(PropertySetting(deviceLabel, MM::g_Keyword_State, CDeviceUtils::ConvertToString(state)));
      }
   }
   if (pStateDev->HasProperty(MM::g_Keyword_Label))
   {
      std::string posLbl = pStateDev->GetPositionLabel(state);

      {
         MMThreadGuard scg(stateCacheLock_);
         stateCache_.addSetting(PropertySetting(deviceLabel, MM::g_Keyword_Label, posLbl.c_str()));
      }
   }

   LOG_DEBUG(coreLogger_) << "Did set " << deviceLabel << " to state " << state;
}

/**
 * Returns the current state (position) on the specific device. The command will fail if
 * the device does not support states.
 *
 * @return                the current state
 * @param deviceLabel     the device label
 */
long CMMCore::getState(const char* deviceLabel) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<StateInstance> pStateDev =
      deviceManager_->GetDeviceOfType<StateInstance>(deviceLabel);
   mm::DeviceModuleLockGuard guard(pStateDev);

   long state;
   int nRet = pStateDev->GetPosition(state);
   if (nRet != DEVICE_OK)
      throw CMMError(getDeviceErrorText(nRet, pStateDev));

   return state;
}

/**
 * Returns the total number of available positions (states).
 *
 * For legacy reasons, an exception is not thrown on error.
 * Instead, -1 is returned if deviceLabel is not a valid state device.
 */
long CMMCore::getNumberOfStates(const char* deviceLabel)
{
   try
   {
      std::shared_ptr<StateInstance> pStateDev =
         deviceManager_->GetDeviceOfType<StateInstance>(deviceLabel);
      mm::DeviceModuleLockGuard guard(pStateDev);
      return pStateDev->GetNumberOfPositions();
   }
   catch (const CMMError&)
   {
      return -1;
   }
}

/**
 * Sets device state using the previously assigned label (string).
 *
 * @param deviceLabel     the device label
 * @param stateLabel      the state label
 */
void CMMCore::setStateLabel(const char* deviceLabel, const char* stateLabel) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<StateInstance> pStateDev =
      deviceManager_->GetDeviceOfType<StateInstance>(deviceLabel);
   CheckStateLabel(stateLabel);

   mm::DeviceModuleLockGuard guard(pStateDev);
   LOG_DEBUG(coreLogger_) << "Will set " << deviceLabel << " to label " << stateLabel;
   int nRet = pStateDev->SetPosition(stateLabel);
   if (nRet != DEVICE_OK)
      throw CMMError(getDeviceErrorText(nRet, pStateDev));
   LOG_DEBUG(coreLogger_) << "Did set " << deviceLabel << " to label " << stateLabel;

   if (pStateDev->HasProperty(MM::g_Keyword_Label))
   {
      {
         MMThreadGuard scg(stateCacheLock_);
         stateCache_.addSetting(PropertySetting(deviceLabel, MM::g_Keyword_Label, stateLabel));
      }
   }
   if (pStateDev->HasProperty(MM::g_Keyword_State))
   {
      long state = getStateFromLabel(deviceLabel, stateLabel);
      {
         MMThreadGuard scg(stateCacheLock_);
         stateCache_.addSetting(PropertySetting(deviceLabel, MM::g_Keyword_State,
                  CDeviceUtils::ConvertToString(state)));
      }
   }
}

/**
 * Returns the current state as the label (string).
 *
 * @return   the current state's label
 * @param deviceLabel     the device label
 */
std::string CMMCore::getStateLabel(const char* deviceLabel) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<StateInstance> pStateDev =
      deviceManager_->GetDeviceOfType<StateInstance>(deviceLabel);

   mm::DeviceModuleLockGuard guard(pStateDev);
   return pStateDev->GetPositionLabel();
}

/**
 * Defines a label for the specific state/
 *
 * @param deviceLabel    the device label
 * @param state          the state to be labeled
 * @param label          the label for the specified state
 */
void CMMCore::defineStateLabel(const char* deviceLabel, long state, const char* label) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<StateInstance> pStateDev =
      deviceManager_->GetDeviceOfType<StateInstance>(deviceLabel);
   CheckStateLabel(label);

   mm::DeviceModuleLockGuard guard(pStateDev);
   // Remember old label so that we can update configurations that use it
   std::string oldLabel;
   try
   {
      oldLabel = pStateDev->GetPositionLabel(state);
   }
   catch (const CMMError&)
   {
      // Ok if not defined
   }

   // Set new label
   int nRet = pStateDev->SetPositionLabel(state, label);
   if (nRet != DEVICE_OK)
      throw CMMError(getDeviceErrorText(nRet, pStateDev));

   if (label != oldLabel)
   {
      // Fix existing configurations that use the old label
      std::vector<std::string> configGroups = getAvailableConfigGroups();
      std::vector<std::string>::const_iterator itcfg = configGroups.begin();
      while (itcfg != configGroups.end())
      {
         std::vector<std::string> configs = getAvailableConfigs((*itcfg).c_str());
         std::vector<std::string>::const_iterator itcf = configs.begin();
         while (itcf != configs.end())
         {
            Configuration conf = getConfigData((*itcfg).c_str(), (*itcf).c_str());
            if (!oldLabel.empty() && conf.isPropertyIncluded(deviceLabel, MM::g_Keyword_Label))
            {
               PropertySetting setting(deviceLabel, MM::g_Keyword_Label, oldLabel.c_str());
               if (conf.isSettingIncluded(setting))
               {
                  deleteConfig((*itcfg).c_str(), (*itcf).c_str(), deviceLabel, MM::g_Keyword_Label);
                  defineConfig((*itcfg).c_str(), (*itcf).c_str(), deviceLabel, MM::g_Keyword_Label, label);
               }
            }
            itcf++;
         }

         itcfg++;
      }
   }

   LOG_DEBUG(coreLogger_) << "Defined label " << label <<
      " for device " << deviceLabel << " state " << state;
}

/**
 * Return labels for all states
 *
 * @return  an array of state labels
 * @param deviceLabel       the device label
 */
std::vector<std::string> CMMCore::getStateLabels(const char* deviceLabel) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<StateInstance> pStateDev =
      deviceManager_->GetDeviceOfType<StateInstance>(deviceLabel);

   mm::DeviceModuleLockGuard guard(pStateDev);
   std::vector<std::string> stateLabels;
   for (unsigned i=0; i<pStateDev->GetNumberOfPositions(); i++)
   {
      stateLabels.push_back(pStateDev->GetPositionLabel(i));
   }
   return stateLabels;
}

/**
 * Obtain the state for a given label.
 *
 * @return the state (an integer)
 * @param deviceLabel     the device label
 * @param stateLabel      the label for which the state is being queried
 */
long CMMCore::getStateFromLabel(const char* deviceLabel, const char* stateLabel) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<StateInstance> pStateDev =
      deviceManager_->GetDeviceOfType<StateInstance>(deviceLabel);
   CheckStateLabel(stateLabel);

   mm::DeviceModuleLockGuard guard(pStateDev);
   long state;
   int nRet = pStateDev->GetLabelPosition(stateLabel, state);
   if (nRet != DEVICE_OK)
      throw CMMError(getDeviceErrorText(nRet, pStateDev));

   return state;
}

/**
 * Creates an empty configuration group.
 */
void CMMCore::defineConfigGroup(const char* groupName) MMCORE_LEGACY_THROW(CMMError)
{
   CheckConfigGroupName(groupName);

   if (!configGroups_->Define(groupName))
      throw CMMError(ToQuotedString(groupName) + ": " + getCoreErrorText(MMERR_DuplicateConfigGroup),
            MMERR_DuplicateConfigGroup);

   updateAllowedChannelGroups();

   LOG_DEBUG(coreLogger_) << "Created config group " << groupName;
}

/**
 * Deletes an entire configuration group.
 */
void CMMCore::deleteConfigGroup(const char* groupName) MMCORE_LEGACY_THROW(CMMError)
{
   CheckConfigGroupName(groupName);

   if (!configGroups_->Delete(groupName))
      throw CMMError(ToQuotedString(groupName) + ": " + getCoreErrorText(MMERR_NoConfigGroup),
            MMERR_NoConfigGroup);

   updateAllowedChannelGroups();

   LOG_DEBUG(coreLogger_) << "Deleted config group " << groupName;
}

/**
 * Renames a configuration group.
 */
void CMMCore::renameConfigGroup(const char* oldGroupName, const char* newGroupName) MMCORE_LEGACY_THROW(CMMError)
{
   CheckConfigGroupName(oldGroupName);
   CheckConfigGroupName(newGroupName);

   if (!configGroups_->RenameGroup(oldGroupName, newGroupName))
      throw CMMError(ToQuotedString(oldGroupName) + ": " + getCoreErrorText(MMERR_NoConfigGroup),
            MMERR_NoConfigGroup);

   LOG_DEBUG(coreLogger_) << "Renamed config group " << oldGroupName <<
      " to " << newGroupName;

   updateAllowedChannelGroups();

   if (0 == channelGroup_.compare(oldGroupName))
      setChannelGroup(newGroupName);
}

/**
 * Defines a configuration. If the configuration group/name was not previously defined
 * a new configuration will be automatically created; otherwise nothing happens.
 *
 * @param groupName    the configuration group name
 * @param configName   the configuration preset name
 */
void CMMCore::defineConfig(const char* groupName, const char* configName) MMCORE_LEGACY_THROW(CMMError)
{
   CheckConfigGroupName(groupName);
   CheckConfigPresetName(configName);

   bool groupExisted = configGroups_->isDefined(groupName);

   configGroups_->Define(groupName, configName);

   if (!groupExisted)
   {
      updateAllowedChannelGroups();
   }

   LOG_DEBUG(coreLogger_) << "Config group " << groupName <<
      ": added preset " << configName;
}

/**
 * Defines a single configuration entry (setting). If the configuration group/name
 * was not previously defined a new configuration will be automatically created.
 * If the name was previously defined the new setting will be added to its list of
 * property settings. The new setting will override previously defined ones if it
 * refers to the same property name.
 *
 * @param groupName    the group name
 * @param configName   the configuration name
 * @param deviceLabel  the device label
 * @param propName     the property name
 * @param value        the property value
 */
void CMMCore::defineConfig(const char* groupName, const char* configName, const char* deviceLabel, const char* propName, const char* value) MMCORE_LEGACY_THROW(CMMError)
{
   CheckConfigGroupName(groupName);
   CheckConfigPresetName(configName);
   CheckDeviceLabel(deviceLabel);
   CheckPropertyName(propName);
   CheckPropertyValue(value);

   bool groupExisted = configGroups_->isDefined(groupName);

   configGroups_->Define(groupName, configName, deviceLabel, propName, value);

   if (!groupExisted)
   {
      updateAllowedChannelGroups();
   }

   LOG_DEBUG(coreLogger_) << "Config group " << groupName <<
      ": preset " << configName << ": added setting " <<
      deviceLabel << "-" << propName << " = " << value;
}



/**
 * Defines a single pixel size entry (setting).
 * The system will treat pixel size configurations very similar to configuration presets,
 * i.e. it will try to detect if any of the pixel size presets matches the current state of
 * the system.
 * If the pixel size was previously defined the new setting will be added to its list of
 * property settings. The new setting will override previously defined ones if it
 * refers to the same property name.
 *
 * @param resolutionID identifier for one unique property setting
 * @param deviceLabel device label
 * @param propName property name
 * @param value property value
*/
void CMMCore::definePixelSizeConfig(const char* resolutionID, const char* deviceLabel, const char* propName, const char* value) MMCORE_LEGACY_THROW(CMMError)
{
   CheckConfigPresetName(resolutionID);
   CheckDeviceLabel(deviceLabel);
   CheckPropertyName(propName);
   CheckPropertyValue(value);

   pixelSizeGroup_->Define(resolutionID, deviceLabel, propName, value);

   LOG_DEBUG(coreLogger_) << "Pixel size config: "
      "preset " << resolutionID << ": added setting : " <<
      deviceLabel << "-" << propName << " = " << value;
}

/**
 * Defines an empty pixel size entry.
*/

void CMMCore::definePixelSizeConfig(const char* resolutionID) MMCORE_LEGACY_THROW(CMMError)
{
   CheckConfigPresetName(resolutionID);

   pixelSizeGroup_->Define(resolutionID);

   LOG_DEBUG(coreLogger_) << "Pixel size config: "
      "added preset " << resolutionID;
}

/**
 * Checks if the Pixel Size Resolution already exists
 *
 * @return true if the configuration is already defined
 */
bool CMMCore::isPixelSizeConfigDefined(const char* resolutionID) const MMCORE_LEGACY_THROW(CMMError)
{
   CheckConfigPresetName(resolutionID);

   return  pixelSizeGroup_->Find(resolutionID) != 0;
}

/**
 * Sets pixel size in microns for the specified resolution sensing configuration preset.
 */
void CMMCore::setPixelSizeUm(const char* resolutionID, double pixSize)  MMCORE_LEGACY_THROW(CMMError)
{
   CheckConfigPresetName(resolutionID);

   PixelSizeConfiguration* psc = pixelSizeGroup_->Find(resolutionID);
   if (psc == 0)
      throw CMMError(ToQuotedString(resolutionID) + ": " + getCoreErrorText(MMERR_NoConfigGroup),
            MMERR_NoConfigGroup);
   psc->setPixelSizeUm(pixSize);

   LOG_DEBUG(coreLogger_) << "Pixel size config: "
      "preset " << resolutionID << ": set resolution to " <<
      std::fixed << std::setprecision(5) << pixSize << " um/px";
}

/**
 * Sets the raw affine transform for the specific pixel size configuration
 * The affine transform consists of the first two rows of a 3x3 matrix,
 * the third row is alsways assumed to be 0.0 0.0 1.0.
 * The transform should be valid for binning 1 and no magnification device
 * (as given by the getMagnification() function).
 * Order: row[0]col[0] row[0]c[1] row[0]c[2] row[1]c[0] row[1]c[1] row[1]c[2]
 * The given vector has to have 6 doubles, or bad stuff will happen
 */
void CMMCore::setPixelSizeAffine(const char* resolutionID, std::vector<double> affine)  MMCORE_LEGACY_THROW(CMMError)
{
   CheckConfigPresetName(resolutionID);

   PixelSizeConfiguration* psc = pixelSizeGroup_->Find(resolutionID);
   if (psc == 0)
      throw CMMError(ToQuotedString(resolutionID) + ": " + getCoreErrorText(MMERR_NoConfigGroup),
            MMERR_NoConfigGroup);
   if (affine.size() != 6)
      throw CMMError(getCoreErrorText(MMERR_BadAffineTransform));

   psc->setPixelConfigAffineMatrix(affine);

   LOG_DEBUG(coreLogger_) << "Pixel size config: "
      "preset " << resolutionID << ": set affine matrix to " <<
      std::fixed << std::setprecision(5) << affine[0] << ", " <<
      std::fixed << std::setprecision(5) << affine[1] << ", " <<
      std::fixed << std::setprecision(5) << affine[2] << ", " <<
      std::fixed << std::setprecision(5) << affine[3] << ", " <<
      std::fixed << std::setprecision(5) << affine[4] << ", " <<
      std::fixed << std::setprecision(5) << affine[5];
}

/**
 * Sets the angle between the camera's x axis and the axis (direction) 
 * of the z drive.  This angle is dimensionless (i.e. the ratio of the 
 * translation in x caused by a translation in z, i.e. dx / dz).  
 * This angle can be different for different z drives (if there 
 * are multiple Z drives in the system, please add the Core-Focus device
 * to the pixel size configuration).  
 * See: https://github.com/micro-manager/micro-manager/issues/1984
 *
 * @param resolutionID   The pixel size configuration group name
 * @param dxdz       Angle of the Z-stage axis with the camera axis (dimensionless)
 */
void CMMCore::setPixelSizedxdz(const char* resolutionID, double dxdz)  MMCORE_LEGACY_THROW(CMMError)
{
   CheckConfigPresetName(resolutionID);

   PixelSizeConfiguration* psc = pixelSizeGroup_->Find(resolutionID);
   if (psc == 0)
      throw CMMError(ToQuotedString(resolutionID) + ": " + getCoreErrorText(MMERR_NoConfigGroup),
            MMERR_NoConfigGroup);
   psc->setdxdz(dxdz);

   LOG_DEBUG(coreLogger_) << "Pixel size config: "
      "preset " << resolutionID << ": set dxdz to " <<
      std::fixed << std::setprecision(5) << dxdz;
}

/**
 * Sets the angle between the camera's y axis and the axis (direction) 
 * of the z drive.  This angle is dimensionless (i.e. the ratio of the 
 * translation in y caused by a translation in z, i.e. dy / dz).  
 * This angle can be different for different z drives (if there 
 * are multiple Z drives in the system, please add the Core-Focus device
 * to the pixel size configuration).  
 * See: https://github.com/micro-manager/micro-manager/issues/1984
 *
 * @param resolutionID   The pixel size configuration group name
 * @param dydz       Angle of the Z-stage axis with the camera axis (dimensionless)
 */
void CMMCore::setPixelSizedydz(const char* resolutionID, double dydz)  MMCORE_LEGACY_THROW(CMMError)
{
   CheckConfigPresetName(resolutionID);

   PixelSizeConfiguration* psc = pixelSizeGroup_->Find(resolutionID);
   if (psc == 0)
      throw CMMError(ToQuotedString(resolutionID) + ": " + getCoreErrorText(MMERR_NoConfigGroup),
            MMERR_NoConfigGroup);
   psc->setdydz(dydz);

   LOG_DEBUG(coreLogger_) << "Pixel size config: "
      "preset " << resolutionID << ": set dydz to " <<
      std::fixed << std::setprecision(5) << dydz;
}

/**
 * Sets the opimal Z stepSize (in microns).
 * There is no magic here, this number is provided by the person configuring the
 * microscope, to be used by the person using the microscope.
 *
 * @param resolutionID   The pixel size configuration group name
 * @param optimalZ       Optimal z step in microns
 */
void CMMCore::setPixelSizeOptimalZUm(const char* resolutionID, double optimalZ)  MMCORE_LEGACY_THROW(CMMError)
{
   CheckConfigPresetName(resolutionID);

   PixelSizeConfiguration* psc = pixelSizeGroup_->Find(resolutionID);
   if (psc == 0)
      throw CMMError(ToQuotedString(resolutionID) + ": " + getCoreErrorText(MMERR_NoConfigGroup),
            MMERR_NoConfigGroup);
   psc->setOptimalZUm(optimalZ);

   LOG_DEBUG(coreLogger_) << "Pixel size config: "
      "preset " << resolutionID << ": set optimalZ to " <<
      std::fixed << std::setprecision(5) << optimalZ << " um.";
}

/**
 * Applies a Pixel Size Configuration. The command will fail if the
 * configuration was not previously defined.
 *
 * @param resolutionID   the pixel size configuration group name
 */
void CMMCore::setPixelSizeConfig(const char* resolutionID) MMCORE_LEGACY_THROW(CMMError)
{
   CheckConfigPresetName(resolutionID);

   PixelSizeConfiguration* psc = pixelSizeGroup_->Find(resolutionID);
   std::ostringstream os;
   os << resolutionID;
   if (!psc)
   {
      throw CMMError(ToQuotedString(resolutionID) + ": " + getCoreErrorText(MMERR_NoConfiguration),
            MMERR_NoConfiguration);
   }

   try {
      applyConfiguration(*psc);
   } catch (CMMError& err) {
      logError("setPixelSizeConfig", getCoreErrorText(err.getCode()).c_str());
      throw;
   }

   LOG_DEBUG(coreLogger_) << "Applied pixel size configuration preset " <<
      resolutionID;
}

/**
 * Applies a configuration to a group. The command will fail if the
 * configuration was not previously defined.
 *
 * @param groupName   the configuration group name
 * @param configName  the configuration preset name
 */
void CMMCore::setConfig(const char* groupName, const char* configName) MMCORE_LEGACY_THROW(CMMError)
{
   CheckConfigGroupName(groupName);
   CheckConfigPresetName(configName);

   Configuration* pCfg = configGroups_->Find(groupName, configName);
   std::ostringstream os;
   os << groupName << "/" << configName;
   if (!pCfg)
   {
      throw CMMError("Preset " + ToQuotedString(configName) +
            " of configuration group " + ToQuotedString(groupName) +
            " does not exist",
            MMERR_NoConfiguration);
   }

   LOG_DEBUG(coreLogger_) << "Config group " << groupName <<
      ": will apply preset " << configName;

   try {
      applyConfiguration(*pCfg);
   } catch (CMMError&) {
      throw;
   }

   LOG_DEBUG(coreLogger_) << "Config group " << groupName <<
      ": did apply preset " << configName;
}

/**
 * Renames a configuration within a specified group. The command will fail if the
 * configuration was not previously defined.
 *
 */
void CMMCore::renameConfig(const char* groupName, const char* oldConfigName, const char* newConfigName) MMCORE_LEGACY_THROW(CMMError)
{
   CheckConfigGroupName(groupName);
   CheckConfigPresetName(oldConfigName);
   CheckConfigPresetName(newConfigName);

   if (!configGroups_->RenameConfig(groupName, oldConfigName, newConfigName)) {
      logError("renameConfig", getCoreErrorText(MMERR_NoConfiguration).c_str());
      throw CMMError("Configuration group " + ToQuotedString(oldConfigName) +
            " does not exist",
            MMERR_NoConfiguration);
   }

   LOG_DEBUG(coreLogger_) << "Config group " << groupName <<
      ": renamed preset " << oldConfigName << " to " << newConfigName;
}

/**
 * Deletes a configuration from a group. The command will fail if the
 * configuration was not previously defined.
 *
 */
void CMMCore::deleteConfig(const char* groupName, const char* configName) MMCORE_LEGACY_THROW(CMMError)
{
   CheckConfigGroupName(groupName);
   CheckConfigPresetName(configName);

   std::ostringstream os;
   os << groupName << "/" << configName;
   if (!configGroups_->Delete(groupName, configName)) {
      logError("deleteConfig", getCoreErrorText(MMERR_NoConfiguration).c_str());
      throw CMMError("Configuration group " + ToQuotedString(groupName) +
            " does not exist",
            MMERR_NoConfiguration);
   }

   LOG_DEBUG(coreLogger_) << "Config group " << groupName <<
      ": deleted preset " << configName;
}

/**
 * Deletes a property from a configuration in the specified group. The command will fail if the
 * configuration was not previously defined.
 *
 */
void CMMCore::deleteConfig(const char* groupName, const char* configName, const char* deviceLabel, const char* propName) MMCORE_LEGACY_THROW(CMMError)
{
   CheckConfigGroupName(groupName);
   CheckConfigPresetName(configName);
   CheckDeviceLabel(deviceLabel);
   CheckPropertyName(propName);

   std::ostringstream os;
   os << groupName << "/" << configName << "/" << deviceLabel << "/" << propName;
   if (!configGroups_->Delete(groupName, configName, deviceLabel, propName)) {
      logError("deleteConfig", getCoreErrorText(MMERR_NoConfiguration).c_str());
      throw CMMError("Property " + ToQuotedString(propName) +
            " of device " + ToQuotedString(deviceLabel) +
            " is not in preset " + ToQuotedString(configName) +
            " of configuration group " + ToQuotedString(groupName),
            MMERR_NoConfiguration);
   }

   LOG_DEBUG(coreLogger_) << "Config group " << groupName <<
      ": preset " << configName << ": deleted property " <<
      deviceLabel << "-" << propName;
}




/**
 * Returns all defined configuration names in a given group
 *
 * For legacy reasons, an exception is not thrown if there is an error.
 * Instead, an empty vector is returned if group is not a valid config group.
 *
 * @return an array of configuration names
 */
std::vector<std::string> CMMCore::getAvailableConfigs(const char* group) const
{
   std::vector<std::string> ret;
   try
   {
      CheckConfigGroupName(group);

      ret = configGroups_->GetAvailableConfigs(group);
   }
   catch (const CMMError&)
   {
   }
   return ret;
}

/**
 * Returns the names of all defined configuration groups
 * @return  an array of names of configuration groups
 */
std::vector<std::string> CMMCore::getAvailableConfigGroups() const
{
   return configGroups_->GetAvailableGroups();
}

/**
 * Returns all defined resolution preset names
 * @return an array of resolution presets
 */
std::vector<std::string> CMMCore::getAvailablePixelSizeConfigs() const
{
   return pixelSizeGroup_->GetAvailable();
}

/**
 * Returns the current configuration for a given group.
 * An empty string is a valid return value, since the system state will not
 * always correspond to any of the defined configurations.
 * Also, in general it is possible that the system state fits multiple configurations.
 * This method will return only the first matching configuration, if any.
 *
 * @return The current configuration preset's name
 */
std::string CMMCore::getCurrentConfig(const char* groupName) MMCORE_LEGACY_THROW(CMMError)
{
   CheckConfigGroupName(groupName);

   std::vector<std::string> cfgs = configGroups_->GetAvailableConfigs(groupName);
   if (cfgs.empty())
      return "";

   Configuration curState = getConfigGroupState(groupName, false);

   for (size_t i=0; i<cfgs.size(); i++)
   {
      Configuration* pCfg = configGroups_->Find(groupName, cfgs[i].c_str());
      if (pCfg && curState.isConfigurationIncluded(*pCfg))
         return cfgs[i];
   }

   // no match
   return "";
}

/**
 * Returns the configuration for a given group based on the data in the cache.
 * An empty string is a valid return value, since the system state will not
 * always correspond to any of the defined configurations.
 * Also, in general it is possible that the system state fits multiple configurations.
 * This method will return only the first matching configuration, if any.
 *
 * @return The cache's current configuration preset name
 */
std::string CMMCore::getCurrentConfigFromCache(const char* groupName) MMCORE_LEGACY_THROW(CMMError)
{
   CheckConfigGroupName(groupName);

   std::vector<std::string> cfgs = configGroups_->GetAvailableConfigs(groupName);
   if (cfgs.empty())
      return "";

   Configuration curState = getConfigGroupState(groupName, true);

   for (size_t i=0; i<cfgs.size(); i++)
   {
      Configuration* pCfg = configGroups_->Find(groupName, cfgs[i].c_str());
      if (pCfg && curState.isConfigurationIncluded(*pCfg))
         return cfgs[i];
   }

   // no match
   return "";
}

/**
 * Returns the configuration object for a given group and name.
 *
 * @return The configuration object
 */
Configuration CMMCore::getConfigData(const char* groupName, const char* configName) MMCORE_LEGACY_THROW(CMMError)
{
   CheckConfigGroupName(groupName);
   CheckConfigPresetName(configName);

   Configuration* pCfg = configGroups_->Find(groupName, configName);
   if (!pCfg)
   {
      // not found
      std::ostringstream os;
      os << groupName << "/" << configName;
      logError(os.str().c_str(), getCoreErrorText(MMERR_NoConfiguration).c_str());
      throw CMMError("Configuration group " + ToQuotedString(groupName) +
            " or its preset " + ToQuotedString(configName) +
            " does not exist",
            MMERR_NoConfiguration);
   }
   return *pCfg;
}

/**
 * Returns the configuration object for a give pixel size preset.
 * @return The configuration object
 */
Configuration CMMCore::getPixelSizeConfigData(const char* configName) MMCORE_LEGACY_THROW(CMMError)
{
   CheckConfigPresetName(configName);

   Configuration* pCfg = pixelSizeGroup_->Find(configName);
   if (!pCfg)
   {
      // not found
      std::ostringstream os;
      os << "Pixel size" << "/" << configName;
      logError(os.str().c_str(), getCoreErrorText(MMERR_NoConfiguration).c_str());
      throw CMMError("Pixel size configuration preset " + ToQuotedString(configName) +
            " does not exist",
            MMERR_NoConfiguration);
   }
   return *pCfg;
}

/**
 * Renames a pixel size configuration. The command will fail if the
 * configuration was not previously defined.
 *
 */
void CMMCore::renamePixelSizeConfig(const char* oldConfigName, const char* newConfigName) MMCORE_LEGACY_THROW(CMMError)
{
   CheckConfigPresetName(oldConfigName);
   CheckConfigPresetName(newConfigName);

   if (!pixelSizeGroup_->Rename(oldConfigName, newConfigName)) {
      logError("renamePixelSizeConfig", getCoreErrorText(MMERR_NoConfiguration).c_str());
      throw CMMError("Pixel size configuration preset " + ToQuotedString(oldConfigName) +
            " does not exist",
            MMERR_NoConfiguration);
   }

   LOG_DEBUG(coreLogger_) << "Pixel size config: "
      "renamed preset " << oldConfigName << " to " << newConfigName;
}

/**
 * Deletes a pixel size configuration. The command will fail if the
 * configuration was not previously defined.
 *
 */
void CMMCore::deletePixelSizeConfig(const char* configName) MMCORE_LEGACY_THROW(CMMError)
{
   CheckConfigPresetName(configName);

   if (!pixelSizeGroup_->Delete(configName)) {
      logError("deletePixelSizeConfig", getCoreErrorText(MMERR_NoConfiguration).c_str());
      throw CMMError("Pixel size configuration preset " + ToQuotedString(configName) +
            " does not exist",
            MMERR_NoConfiguration);
   }

   LOG_DEBUG(coreLogger_) << "Pixel size config: "
      "deleted preset " << configName;
}

/**
 * Get the current pixel configuration name
 **/
std::string CMMCore::getCurrentPixelSizeConfig() MMCORE_LEGACY_THROW(CMMError)
{
	return getCurrentPixelSizeConfig(false);
}

/**
 * Get the current pixel configuration name
 **/
std::string CMMCore::getCurrentPixelSizeConfig(bool cached) MMCORE_LEGACY_THROW(CMMError)
{
   // get a list of configuration names
   std::vector<std::string> cfgs = pixelSizeGroup_->GetAvailable();
   if (cfgs.empty())
      return "";

   // create a union of configuration settings used in this group
   // and obtain the current state of the system
   Configuration curState;
   for (size_t i=0; i<cfgs.size(); i++) {
      PixelSizeConfiguration* cfgData = pixelSizeGroup_->Find(cfgs[i].c_str());
      assert(cfgData);
      for (size_t j=0; j < cfgData->size(); j++)
      {
         PropertySetting cs = cfgData->getSetting(j); // config setting
         if (!curState.isPropertyIncluded(cs.getDeviceLabel().c_str(), cs.getPropertyName().c_str()))
         {
            try
            {
				std::string value;
				if (!cached)
				{
                   value = getProperty(cs.getDeviceLabel().c_str(), cs.getPropertyName().c_str());
				}
				else
				{
               MMThreadGuard scg(stateCacheLock_);
               value = stateCache_.getSetting(cs.getDeviceLabel().c_str(), cs.getPropertyName().c_str()).getPropertyValue();
				}
               PropertySetting ss(cs.getDeviceLabel().c_str(), cs.getPropertyName().c_str(), value.c_str()); // state setting
               curState.addSetting(ss);
            }
            catch (CMMError& err)
            {
               // just log error
               logError("GetPixelSizeUm", err.getMsg().c_str());
            }
         }
      }
   }

   // check which one matches the current state
   for (size_t i=0; i<cfgs.size(); i++)
   {
      PixelSizeConfiguration* pCfg = pixelSizeGroup_->Find(cfgs[i].c_str());
      if (pCfg && curState.isConfigurationIncluded(*pCfg))
      {
		 return cfgs[i];
      }
   }

   return "";
}

/**
 * Returns the current pixel size in microns.
 * This method is based on sensing the current pixel size configuration and adjusting
 * for the binning.
 */
double CMMCore::getPixelSizeUm()
{
	 return getPixelSizeUm(false);
}

/**
 * Returns the current pixel size in microns.
 * This method is based on sensing the current pixel size configuration and adjusting
 * for the binning.
 *
 * For legacy reasons, an exception is not thrown if there is an error.
 * Instead, 0.0 is returned if any property values cannot be read, or if no
 * pixel size preset matches the property values.
 */
double CMMCore::getPixelSizeUm(bool cached)
{
   std::string resolutionID;
   try
   {
      resolutionID = getCurrentPixelSizeConfig(cached);
   }
   catch (const CMMError&)
   {
   }

   if (resolutionID.length() > 0)
   {
      // check which one matches the current state
      PixelSizeConfiguration* pCfg = pixelSizeGroup_->Find(resolutionID.c_str());
      if (!pCfg)
         return 0.0;

      double pixSize = pCfg->getPixelSizeUm();

      std::shared_ptr<CameraInstance> camera = currentCameraDevice_.lock();
      if (camera)
      {
         try
         {
            mm::DeviceModuleLockGuard guard(camera);
            pixSize *= camera->GetBinning();
         }
         catch (const CMMError&) // Possibly uninitialized camera
         {
            // Assume no binning
         }
      }

      pixSize /= getMagnificationFactor();

      return pixSize;
   }
   else
   {
      return 0.0;
   }
}

/**
 * Returns the pixel size in um for the requested pixel size group
 */
double CMMCore::getPixelSizeUmByID(const char* resolutionID) MMCORE_LEGACY_THROW(CMMError)
{
   CheckConfigPresetName(resolutionID);

   PixelSizeConfiguration* psc = pixelSizeGroup_->Find(resolutionID);
   if (psc == 0)
      throw CMMError(ToQuotedString(resolutionID) + ": " + getCoreErrorText(MMERR_NoConfigGroup),
            MMERR_NoConfigGroup);
   return psc->getPixelSizeUm();
}

/**
 * Returns the current Affine Transform to related camera pixels with stage movement..
 * This function returns the stored affine transform corrected for binning
 */
std::vector<double> CMMCore::getPixelSizeAffine() MMCORE_LEGACY_THROW(CMMError)
{
	 return getPixelSizeAffine(false);
}

/**
 * Returns the current Affine Transform to related camera pixels with stage movement..
 * This function returns the stored affine transform corrected for binning
 * and known magnification devices
 */
std::vector<double> CMMCore::getPixelSizeAffine(bool cached) MMCORE_LEGACY_THROW(CMMError)
{
   std::string resolutionID = getCurrentPixelSizeConfig(cached);
   if (resolutionID.length() > 0)
   {
      // check which one matches the current state
      PixelSizeConfiguration* pCfg = pixelSizeGroup_->Find(resolutionID.c_str());
      std::vector<double> af = pCfg->getPixelConfigAffineMatrix();

      std::shared_ptr<CameraInstance> camera = currentCameraDevice_.lock();
      int binning = 1;
      if (camera)
      {
         mm::DeviceModuleLockGuard guard(camera);
         binning = camera->GetBinning();
      }

      double factor = binning / getMagnificationFactor();

      if (factor != 1.0) {
         for (double& v : af) {
            v *= factor;
         }
      }
      return af;
   }
   else
   {
      // no config found, return a matrix with all 0.0s
      return *nullAffine_;
   }
}

/**
 * Returns the  Affine Transform to related camera pixels with stage movement
 * for the requested pixel size group
 * The raw affine transform without correction for binning and magnification
 * will be returned.
 */
std::vector<double> CMMCore::getPixelSizeAffineByID(const char* resolutionID) MMCORE_LEGACY_THROW(CMMError)
{
   CheckConfigPresetName(resolutionID);

   PixelSizeConfiguration* psc = pixelSizeGroup_->Find(resolutionID);
   if (psc == 0)
      throw CMMError(ToQuotedString(resolutionID) + ": " + getCoreErrorText(MMERR_NoConfigGroup),
            MMERR_NoConfigGroup);
   std::vector<double> affineTransform = psc->getPixelConfigAffineMatrix();

   return affineTransform;
}

/**
 * Returns the product of all Magnifiers in the system or 1.0 when none is found
 * This is used internally by GetPixelSizeUm
 *
 * @return products of all magnifier devices in the system or 1.0 when none is found
 */
double CMMCore::getMagnificationFactor() const
{
   double magnification = 1.0;
   std::vector<std::string> magnifiers = getLoadedDevicesOfType(MM::MagnifierDevice);
   for (size_t i=0; i<magnifiers.size(); i++)
   {
      std::shared_ptr<MagnifierInstance> magnifier =
         deviceManager_->GetDeviceOfType<MagnifierInstance>(magnifiers[i]);

      try
      {
         mm::DeviceModuleLockGuard guard(magnifier);
         magnification *= magnifier->GetMagnification();
      }
      catch (const CMMError&)
      {
         // Most likely the magnifier was not initialized.
         // Ignore it: only initialized magnifiers count.
      }
   }
   return magnification;
}

/**
 * Returns the angle between the camera's x axis and the axis (direction) 
 * of the z drive.  This angle is dimensionless (i.e. the ratio of the 
 * translation in x caused by a translation in z, i.e. dx / dz).  
 * This angle can be different for different z drives (if there 
 * are multiple Z drives in the system, please add the Core-Focus device
 * to the pixel size configuration).  
 * See: https://github.com/micro-manager/micro-manager/issues/1984
 *
 * @return        angle (dx/dz) of the Z-stage axis with the camera axis (dimensionless)
 */
double CMMCore::getPixelSizedxdz() MMCORE_LEGACY_THROW(CMMError)
{
	 return getPixelSizedxdz(false);
}

/**
 * Returns the angle between the camera's x axis and the axis (direction) 
 * of the z drive.  This angle is dimensionless (i.e. the ratio of the 
 * translation in x caused by a translation in z, i.e. dx / dz).  
 * This angle can be different for different z drives (if there 
 * are multiple Z drives in the system, please add the Core-Focus device
 * to the pixel size configuration).  
 * See: https://github.com/micro-manager/micro-manager/issues/1984
 *
 * @param cached  use the System state cache when true, otherwise checks
 *                the hardware.
 * @return        angle (dx/dz) of the Z-stage axis with the camera axis (dimensionless)
 */
double CMMCore::getPixelSizedxdz(bool cached) MMCORE_LEGACY_THROW(CMMError)
{
   std::string resolutionID;
   resolutionID = getCurrentPixelSizeConfig(cached);

   if (resolutionID.length() > 0)
   {
      // check which one matches the current state
      PixelSizeConfiguration* pCfg = pixelSizeGroup_->Find(resolutionID.c_str());
      if (!pCfg)
         return 0.0;

      return pCfg->getdxdz();
   }
   else
   {
      throw CMMError("No pixel size configuration found", MMERR_DEVICE_GENERIC);
   }
}

/**
 * Returns the angle between the camera's x axis and the axis (direction) 
 * of the z drive for the given pixel size configuration.  
 * This angle is dimensionless (i.e. the ratio of the 
 * translation in x caused by a translation in z, i.e. dx / dz).  
 * This angle can be different for different z drives (if there 
 * are multiple Z drives in the system, please add the Core-Focus device
 * to the pixel size configuration).  
 * See: https://github.com/micro-manager/micro-manager/issues/1984
 *
 * @param resolutionID   The pixel size configuration group name
 * @return        Angle (dx/dz) of the Z-stage axis with the camera axis (dimensionless)
 */
double CMMCore::getPixelSizedxdz(const char* resolutionID) MMCORE_LEGACY_THROW(CMMError)
{
   CheckConfigPresetName(resolutionID);

   PixelSizeConfiguration* psc = pixelSizeGroup_->Find(resolutionID);
   if (psc == 0)
      throw CMMError(ToQuotedString(resolutionID) + ": " + getCoreErrorText(MMERR_NoConfigGroup),
            MMERR_NoConfigGroup);
   return psc->getdxdz();
}

/**
 * Returns the angle between the camera's y axis and the axis (direction) 
 * of the z drive.  This angle is dimensionless (i.e. the ratio of the 
 * translation in y caused by a translation in z, i.e. dy / dz).  
 * This angle can be different for different z drives (if there 
 * are multiple Z drives in the system, please add the Core-Focus device
 * to the pixel size configuration).  
 * See: https://github.com/micro-manager/micro-manager/issues/1984
 *
 * @return   angle (dy/dz) of the Z-stage axis with the camera axis (dimensionless)
 */
double CMMCore::getPixelSizedydz() MMCORE_LEGACY_THROW(CMMError)
{
	 return getPixelSizedydz(false);
}

/**
 * Returns the angle between the camera's y axis and the axis (direction) 
 * of the z drive optionally using the System cache.  This angle is 
 * dimensionless (i.e. the ratio of the translation in y caused by 
 * a translation in z, i.e. dy / dz).  
 * This angle can be different for different z drives (if there 
 * are multiple Z drives in the system, please add the Core-Focus device
 * to the pixel size configuration).  
 * See: https://github.com/micro-manager/micro-manager/issues/1984
 *
 * @param cached   Uses System state cache to find active pixel size config when true
 * @return   angle (dy/dz) of the Z-stage axis with the camera axis (dimensionless)
 */
double CMMCore::getPixelSizedydz(bool cached) MMCORE_LEGACY_THROW(CMMError)
{
   std::string resolutionID;
   resolutionID = getCurrentPixelSizeConfig(cached);

   if (resolutionID.length() > 0)
   {
      // check which one matches the current state
      PixelSizeConfiguration* pCfg = pixelSizeGroup_->Find(resolutionID.c_str());
      if (!pCfg)
         return 0.0;

      return pCfg->getdydz();
   }
   else
   {
      throw CMMError("No pixel size configuration found", MMERR_DEVICE_GENERIC);
   }
}

/**
 * Returns the angle between the camera's y axis and the axis (direction) 
 * of the z drive for the given pixel size configuration.  
 * This angle is dimensionless (i.e. the ratio of the 
 * translation in y caused by a translation in z, i.e. dy / dz).  
 * This angle can be different for different z drives (if there 
 * are multiple Z drives in the system, please add the Core-Focus device
 * to the pixel size configuration).  
 * See: https://github.com/micro-manager/micro-manager/issues/1984
 *
 * @param resolutionID   Name of Pixel Size configuration for this dy /dz angle
 * @return   angle (dy/dz) of the Z-stage axis with the camera axis (dimensionless)
 */
double CMMCore::getPixelSizedydz(const char* resolutionID) MMCORE_LEGACY_THROW(CMMError)
{
   CheckConfigPresetName(resolutionID);

   PixelSizeConfiguration* psc = pixelSizeGroup_->Find(resolutionID);
   if (psc == 0)
      throw CMMError(ToQuotedString(resolutionID) + ": " + getCoreErrorText(MMERR_NoConfigGroup),
            MMERR_NoConfigGroup);
   return psc->getdydz();
}

/**
 * Returns the optimal z step size in um
 * There is no magic to this number, but lets the system configuration
 * communicate to the end user what the optimal Z step size is for this 
 * pixel size configuration
 */
double CMMCore::getPixelSizeOptimalZUm() MMCORE_LEGACY_THROW(CMMError)
{
	 return getPixelSizeOptimalZUm(false);
}

/**
 * Returns the optimal z step size in um, optionally using cached pixel configuration
 * There is no magic to this number, but lets the system configuration
 * communicate to the end user what the optimal Z step size is for this 
 * pixel size configuration
 *
 * @param cached   Uses System state cache to find active pixel size config when true
 */
double CMMCore::getPixelSizeOptimalZUm(bool cached) MMCORE_LEGACY_THROW(CMMError)
{
   std::string resolutionID;
   resolutionID = getCurrentPixelSizeConfig(cached);

   if (resolutionID.length() > 0)
   {
      // check which one matches the current state
      PixelSizeConfiguration* pCfg = pixelSizeGroup_->Find(resolutionID.c_str());
      if (!pCfg)
         return 0.0;

      return pCfg->getOptimalZUm();
   }
   else
   {
      throw CMMError("No pixel size configuration found", MMERR_DEVICE_GENERIC);
   }
}

/**
 * Returns the optimal z step size in um, optionally using cached pixel configuration
 * There is no magic to this number, but lets the system configuration
 * communicate to the end user what the optimal Z step size is for this 
 * pixel size configuration
 */
double CMMCore::getPixelSizeOptimalZUm(const char* resolutionID) MMCORE_LEGACY_THROW(CMMError)
{
   CheckConfigPresetName(resolutionID);

   PixelSizeConfiguration* psc = pixelSizeGroup_->Find(resolutionID);
   if (psc == 0)
      throw CMMError(ToQuotedString(resolutionID) + ": " + getCoreErrorText(MMERR_NoConfigGroup),
            MMERR_NoConfigGroup);
   return psc->getOptimalZUm();
}

/**
 * Checks if the configuration already exists within a group.
 *
 * @return true if the configuration is already defined
 */
bool CMMCore::isConfigDefined(const char* groupName, const char* configName)
{
   if (!groupName || !configName)
      return false;

   return  configGroups_->Find(groupName, configName) != 0;
}

/**
 * Checks if the group already exists.
 *
 * @return true if the group is already defined
 */
bool CMMCore::isGroupDefined(const char* groupName)
{
   if (!groupName)
      return false;

   return  configGroups_->isDefined(groupName);
}

/**
 * Sets all com port properties in a single call
 */
void CMMCore::setSerialProperties(const char* portName,
                                  const char* answerTimeout,
                                  const char* baudRate,
                                  const char* delayBetweenCharsMs,
                                  const char* handshaking,
                                  const char* parity,
                                  const char* stopBits) MMCORE_LEGACY_THROW(CMMError)
{
   setProperty(portName, MM::g_Keyword_AnswerTimeout, answerTimeout);
   setProperty(portName, MM::g_Keyword_BaudRate, baudRate);
   setProperty(portName, MM::g_Keyword_DelayBetweenCharsMs, delayBetweenCharsMs);
   setProperty(portName, MM::g_Keyword_Handshaking, handshaking);
   setProperty(portName, MM::g_Keyword_Parity, parity);
   setProperty(portName, MM::g_Keyword_StopBits, stopBits);
}

/**
 * Send string to the serial device and return an answer.
 * This command blocks until it receives an answer from the device terminated by the specified
 * sequence.
 */
void CMMCore::setSerialPortCommand(const char* portLabel, const char* command, const char* term) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<SerialInstance> pSerial =
      deviceManager_->GetDeviceOfType<SerialInstance>(portLabel);
   if (!command)
      command = ""; // XXX Or should we throw?
   if (!term)
      term = "";

   int ret = pSerial->SetCommand(command, term);
   if (ret != DEVICE_OK)
   {
      logError(portLabel, getDeviceErrorText(ret, pSerial).c_str());
      throw CMMError(getDeviceErrorText(ret, pSerial));
   }
}

/**
 * Continuously read from the serial port until the terminating sequence is encountered.
 */
std::string CMMCore::getSerialPortAnswer(const char* portLabel, const char* term) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<SerialInstance> pSerial =
      deviceManager_->GetDeviceOfType<SerialInstance>(portLabel);
   if (!term || term[0] == '\0')
      throw CMMError("Null or empty terminator; cannot delimit received message");

   const int bufLen = 1024;
   char answerBuf[bufLen];
   int ret = pSerial->GetAnswer(answerBuf, bufLen, term);
   if (ret != DEVICE_OK)
   {
      std::string errText = getDeviceErrorText(ret, pSerial).c_str();
      logError(portLabel, errText.c_str());
      throw CMMError(errText);
   }

   return std::string(answerBuf);
}

/**
 * Sends an array of characters to the serial port and returns immediately.
 */
void CMMCore::writeToSerialPort(const char* portLabel, const std::vector<char> &data) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<SerialInstance> pSerial =
      deviceManager_->GetDeviceOfType<SerialInstance>(portLabel);

   int ret = pSerial->Write((unsigned char*)(&(data[0])), (unsigned long)data.size());
   if (ret != DEVICE_OK)
   {
      logError(portLabel, getDeviceErrorText(ret, pSerial).c_str());
      throw CMMError(getDeviceErrorText(ret, pSerial));
   }
}

/**
 * Reads the contents of the Rx buffer.
 */
std::vector<char> CMMCore::readFromSerialPort(const char* portLabel) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<SerialInstance> pSerial =
      deviceManager_->GetDeviceOfType<SerialInstance>(portLabel);

   const int bufLen = 1024; // internal chunk size limit
   unsigned char answerBuf[bufLen];
   unsigned long read;
   int ret = pSerial->Read(answerBuf, bufLen, read);
   if (ret != DEVICE_OK)
   {
      logError(portLabel, getDeviceErrorText(ret, pSerial).c_str());
      throw CMMError(getDeviceErrorText(ret, pSerial));
   }

   std::vector<char> data;
   data.resize(read, 0);
   if (read > 0)
      std::memcpy(&(data[0]), answerBuf, read);

   return data;
}


/**
 * Write an 8-bit monochrome image to the SLM.
 */
void CMMCore::setSLMImage(const char* deviceLabel, unsigned char* pixels) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<SLMInstance> pSLM =
      deviceManager_->GetDeviceOfType<SLMInstance>(deviceLabel);
   if (!pixels)
      throw CMMError("Null image");
   mm::DeviceModuleLockGuard guard(pSLM);
   int ret = pSLM->SetImage(pixels);
   if (ret != DEVICE_OK)
   {
      logError(deviceLabel, getDeviceErrorText(ret, pSLM).c_str());
      throw CMMError(getDeviceErrorText(ret, pSLM));
   }
}

/**
 * Write a 32-bit color image to the SLM.
 */
void CMMCore::setSLMImage(const char* deviceLabel, imgRGB32 pixels) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<SLMInstance> pSLM =
      deviceManager_->GetDeviceOfType<SLMInstance>(deviceLabel);
   if (!pixels)
      throw CMMError("Null image");
   mm::DeviceModuleLockGuard guard(pSLM);
   int ret = pSLM->SetImage((unsigned int *) pixels);
   if (ret != DEVICE_OK)
   {
      logError(deviceLabel, getDeviceErrorText(ret, pSLM).c_str());
      throw CMMError(getDeviceErrorText(ret, pSLM));
   }
}

/**
 * Set all SLM pixels to a single 8-bit intensity.
 */
void CMMCore::setSLMPixelsTo(const char* deviceLabel, unsigned char intensity) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<SLMInstance> pSLM =
      deviceManager_->GetDeviceOfType<SLMInstance>(deviceLabel);

   mm::DeviceModuleLockGuard guard(pSLM);
   int ret = pSLM->SetPixelsTo(intensity);
   if (ret != DEVICE_OK)
   {
      logError(deviceLabel, getDeviceErrorText(ret, pSLM).c_str());
      throw CMMError(getDeviceErrorText(ret, pSLM));
   }
}

/**
 * Set all SLM pixels to an RGB color.
 */
void CMMCore::setSLMPixelsTo(const char* deviceLabel, unsigned char red, unsigned char green, unsigned char blue) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<SLMInstance> pSLM =
      deviceManager_->GetDeviceOfType<SLMInstance>(deviceLabel);

   mm::DeviceModuleLockGuard guard(pSLM);
   int ret = pSLM->SetPixelsTo(red, green, blue);
   if (ret != DEVICE_OK)
   {
      logError(deviceLabel, getDeviceErrorText(ret, pSLM).c_str());
      throw CMMError(getDeviceErrorText(ret, pSLM));
   }
}

/**
 * Display the waiting image on the SLM.
 */
void CMMCore::displaySLMImage(const char* deviceLabel) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<SLMInstance> pSLM =
      deviceManager_->GetDeviceOfType<SLMInstance>(deviceLabel);

   mm::DeviceModuleLockGuard guard(pSLM);
   int ret = pSLM->DisplayImage();
   if (ret != DEVICE_OK)
   {
      logError(deviceLabel, getDeviceErrorText(ret, pSLM).c_str());
      throw CMMError(getDeviceErrorText(ret, pSLM));
   }
}

/**
 * For SLM devices with build-in light source (such as projectors)
 * this will set the exposure time, but not (yet) start the illumination
 */
void CMMCore::setSLMExposure(const char* deviceLabel, double exposure_ms) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<SLMInstance> pSLM =
      deviceManager_->GetDeviceOfType<SLMInstance>(deviceLabel);

   mm::DeviceModuleLockGuard guard(pSLM);
   int ret = pSLM->SetExposure(exposure_ms);
   if (ret != DEVICE_OK)
   {
      logError(deviceLabel, getDeviceErrorText(ret, pSLM).c_str());
      throw CMMError(getDeviceErrorText(ret, pSLM));
   }
}

/**
 * Returns the exposure time that will be used by the SLM for illumination
 */
double CMMCore::getSLMExposure(const char* deviceLabel) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<SLMInstance> pSLM =
      deviceManager_->GetDeviceOfType<SLMInstance>(deviceLabel);

   mm::DeviceModuleLockGuard guard(pSLM);
   return pSLM->GetExposure();
}


/**
 * Returns the width (in "pixels") of the SLM
 *
 * @param deviceLabel name of the SLM
 */
unsigned CMMCore::getSLMWidth(const char* deviceLabel) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<SLMInstance> pSLM =
      deviceManager_->GetDeviceOfType<SLMInstance>(deviceLabel);

   mm::DeviceModuleLockGuard guard(pSLM);
   return pSLM->GetWidth();
}


/**
 * Returns the height (in "pixels") of the SLM
 *
 * @param deviceLabel name of the SLM
 */
unsigned CMMCore::getSLMHeight(const char* deviceLabel) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<SLMInstance> pSLM =
      deviceManager_->GetDeviceOfType<SLMInstance>(deviceLabel);

   mm::DeviceModuleLockGuard guard(pSLM);
   return pSLM->GetHeight();
}

/**
 * Returns the number of components (usually these depict colors) of the SLM
 * For instance, an RGB projector will return 3, but a grey scale SLM returns 1
 *
 * @param deviceLabel name of the SLM
 */
unsigned CMMCore::getSLMNumberOfComponents(const char* deviceLabel) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<SLMInstance> pSLM =
      deviceManager_->GetDeviceOfType<SLMInstance>(deviceLabel);

   mm::DeviceModuleLockGuard guard(pSLM);
   return pSLM->GetNumberOfComponents();
}

/**
 * Returns the number of bytes per SLM pixel
 *
 * @param deviceLabel name of the SLM
 */
unsigned CMMCore::getSLMBytesPerPixel(const char* deviceLabel) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<SLMInstance> pSLM =
      deviceManager_->GetDeviceOfType<SLMInstance>(deviceLabel);

   mm::DeviceModuleLockGuard guard(pSLM);
   return pSLM->GetBytesPerPixel();
}

/**
 * For SLMs that support sequences, returns the maximum 
 * length of the sequence that can be uploaded to the device
 *
 * @param deviceLabel name of the SLM
 */
long CMMCore::getSLMSequenceMaxLength(const char* deviceLabel) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<SLMInstance> pSLM =
      deviceManager_->GetDeviceOfType<SLMInstance>(deviceLabel);

   mm::DeviceModuleLockGuard guard(pSLM);
   long numEvents;
   int ret = pSLM->GetSLMSequenceMaxLength(numEvents);
   if (ret != DEVICE_OK)
      throw CMMError(getDeviceErrorText(ret, pSLM));
   return numEvents;
}

/**
 * Starts the sequence previously uploaded to the SLM
 *
 * @param deviceLabel name of the SLM
 */
void CMMCore::startSLMSequence(const char* deviceLabel) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<SLMInstance> pSLM =
      deviceManager_->GetDeviceOfType<SLMInstance>(deviceLabel);

   mm::DeviceModuleLockGuard guard(pSLM);
   int ret = pSLM->StartSLMSequence();
   if (ret != DEVICE_OK)
      throw CMMError(getDeviceErrorText(ret, pSLM));
}

/**
 * Stops the SLM sequence if previously started
 *
 * @param deviceLabel name of the SLM
 */
void CMMCore::stopSLMSequence(const char* deviceLabel) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<SLMInstance> pSLM =
      deviceManager_->GetDeviceOfType<SLMInstance>(deviceLabel);

   mm::DeviceModuleLockGuard guard(pSLM);
   int ret = pSLM->StopSLMSequence();
   if (ret != DEVICE_OK)
      throw CMMError(getDeviceErrorText(ret, pSLM));
}

/**
 * Load a sequence of images into the SLM
 *
 * @param deviceLabel name of the SLM
 * @param imageSequence pointers to the images to be used in the sequence
 */
void CMMCore::loadSLMSequence(const char* deviceLabel, std::vector<unsigned char *> imageSequence) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<SLMInstance> pSLM =
      deviceManager_->GetDeviceOfType<SLMInstance>(deviceLabel);


   mm::DeviceModuleLockGuard guard(pSLM);
   int ret = pSLM->ClearSLMSequence();
   if (ret != DEVICE_OK)
      throw CMMError(getDeviceErrorText(ret, pSLM));

   for (std::vector<unsigned char *>::const_iterator it = imageSequence.begin(),
         end = imageSequence.end();
         it < end; ++it)
   {
      ret = pSLM->AddToSLMSequence(*it);
      if (ret != DEVICE_OK)
         throw CMMError(getDeviceErrorText(ret, pSLM));
   }

   ret = pSLM->SendSLMSequence();
   if (ret != DEVICE_OK)
      throw CMMError(getDeviceErrorText(ret, pSLM));
}

/* GALVO CODE */

/**
 * Set the Galvo to an x,y position and fire the laser for a predetermined duration.
 */
void CMMCore::pointGalvoAndFire(const char* deviceLabel, double x, double y, double pulseTime_us) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<GalvoInstance> pGalvo =
      deviceManager_->GetDeviceOfType<GalvoInstance>(deviceLabel);

   mm::DeviceModuleLockGuard guard(pGalvo);

   int ret = pGalvo->PointAndFire(x,y,pulseTime_us);

   if (ret != DEVICE_OK)
   {
      logError(deviceLabel, getDeviceErrorText(ret, pGalvo).c_str());
      throw CMMError(getDeviceErrorText(ret, pGalvo));
   }
}

void CMMCore::setGalvoSpotInterval(const char* deviceLabel, double pulseTime_us) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<GalvoInstance> pGalvo =
      deviceManager_->GetDeviceOfType<GalvoInstance>(deviceLabel);

   mm::DeviceModuleLockGuard guard(pGalvo);

   int ret = pGalvo->SetSpotInterval(pulseTime_us);

   if (ret != DEVICE_OK)
   {
      logError(deviceLabel, getDeviceErrorText(ret, pGalvo).c_str());
      throw CMMError(getDeviceErrorText(ret, pGalvo));
   }
}


/**
 * Set the Galvo to an x,y position
 */
void CMMCore::setGalvoPosition(const char* deviceLabel, double x, double y) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<GalvoInstance> pGalvo =
      deviceManager_->GetDeviceOfType<GalvoInstance>(deviceLabel);

   mm::DeviceModuleLockGuard guard(pGalvo);

   int ret = pGalvo->SetPosition(x, y);

   if (ret != DEVICE_OK)
   {
      logError(deviceLabel, getDeviceErrorText(ret, pGalvo).c_str());
      throw CMMError(getDeviceErrorText(ret, pGalvo));
   }
}

/**
 * Get the Galvo x,y position
 */
void CMMCore::getGalvoPosition(const char* deviceLabel, double &x, double &y) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<GalvoInstance> pGalvo =
      deviceManager_->GetDeviceOfType<GalvoInstance>(deviceLabel);

   mm::DeviceModuleLockGuard guard(pGalvo);

   int ret = pGalvo->GetPosition(x, y);

   if (ret != DEVICE_OK)
   {
      logError(deviceLabel, getDeviceErrorText(ret, pGalvo).c_str());
      throw CMMError(getDeviceErrorText(ret, pGalvo));
   }
}

/**
 * Set the galvo's illumination state to on or off
 */
void CMMCore::setGalvoIlluminationState(const char* deviceLabel, bool on) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<GalvoInstance> pGalvo =
      deviceManager_->GetDeviceOfType<GalvoInstance>(deviceLabel);

   mm::DeviceModuleLockGuard guard(pGalvo);

   int ret = pGalvo->SetIlluminationState(on);

   if (ret != DEVICE_OK)
   {
      logError(deviceLabel, getDeviceErrorText(ret, pGalvo).c_str());
      throw CMMError(getDeviceErrorText(ret, pGalvo));
   }
}



/**
 * Get the Galvo x range
 */
double CMMCore::getGalvoXRange(const char* deviceLabel) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<GalvoInstance> pGalvo =
      deviceManager_->GetDeviceOfType<GalvoInstance>(deviceLabel);

   mm::DeviceModuleLockGuard guard(pGalvo);
   return pGalvo->GetXRange();
}

/**
 * Get the Galvo x minimum
 */
double CMMCore::getGalvoXMinimum(const char* deviceLabel) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<GalvoInstance> pGalvo =
      deviceManager_->GetDeviceOfType<GalvoInstance>(deviceLabel);

   mm::DeviceModuleLockGuard guard(pGalvo);
   return pGalvo->GetXMinimum();
}

/**
 * Get the Galvo y range
 */
double CMMCore::getGalvoYRange(const char* deviceLabel) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<GalvoInstance> pGalvo =
      deviceManager_->GetDeviceOfType<GalvoInstance>(deviceLabel);

   mm::DeviceModuleLockGuard guard(pGalvo);
   return pGalvo->GetYRange();
}

/**
 * Get the Galvo y minimum
 */
double CMMCore::getGalvoYMinimum(const char* deviceLabel) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<GalvoInstance> pGalvo =
      deviceManager_->GetDeviceOfType<GalvoInstance>(deviceLabel);

   mm::DeviceModuleLockGuard guard(pGalvo);
   return pGalvo->GetYMinimum();
}

/**
 * Add a vertex to a galvo polygon.
 */
void CMMCore::addGalvoPolygonVertex(const char* deviceLabel, int polygonIndex, double x, double y) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<GalvoInstance> pGalvo =
      deviceManager_->GetDeviceOfType<GalvoInstance>(deviceLabel);

   mm::DeviceModuleLockGuard guard(pGalvo);

   int ret =  pGalvo->AddPolygonVertex(polygonIndex, x, y);

   if (ret != DEVICE_OK)
   {
      logError(deviceLabel, getDeviceErrorText(ret, pGalvo).c_str());
      throw CMMError(getDeviceErrorText(ret, pGalvo));
   }
}

/**
 * Remove all added polygons
 */
void CMMCore::deleteGalvoPolygons(const char* deviceLabel) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<GalvoInstance> pGalvo =
      deviceManager_->GetDeviceOfType<GalvoInstance>(deviceLabel);

   mm::DeviceModuleLockGuard guard(pGalvo);

   int ret = pGalvo->DeletePolygons();

   if (ret != DEVICE_OK)
   {
      logError(deviceLabel, getDeviceErrorText(ret, pGalvo).c_str());
      throw CMMError(getDeviceErrorText(ret, pGalvo));
   }
}


/**
 * Load a set of galvo polygons to the device
 */
void CMMCore::loadGalvoPolygons(const char* deviceLabel) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<GalvoInstance> pGalvo =
      deviceManager_->GetDeviceOfType<GalvoInstance>(deviceLabel);

   mm::DeviceModuleLockGuard guard(pGalvo);

   int ret =  pGalvo->LoadPolygons();

   if (ret != DEVICE_OK)
   {
      logError(deviceLabel, getDeviceErrorText(ret, pGalvo).c_str());
      throw CMMError(getDeviceErrorText(ret, pGalvo));
   }
}

/**
 * Set the number of times to loop galvo polygons
 */
void CMMCore::setGalvoPolygonRepetitions(const char* deviceLabel, int repetitions) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<GalvoInstance> pGalvo =
      deviceManager_->GetDeviceOfType<GalvoInstance>(deviceLabel);

   mm::DeviceModuleLockGuard guard(pGalvo);

   int ret =  pGalvo->SetPolygonRepetitions(repetitions);

   if (ret != DEVICE_OK)
   {
      logError(deviceLabel, getDeviceErrorText(ret, pGalvo).c_str());
      throw CMMError(getDeviceErrorText(ret, pGalvo));
   }
}


/**
 * Run a loop of galvo polygons
 */
void CMMCore::runGalvoPolygons(const char* deviceLabel) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<GalvoInstance> pGalvo =
      deviceManager_->GetDeviceOfType<GalvoInstance>(deviceLabel);

   mm::DeviceModuleLockGuard guard(pGalvo);

   int ret =  pGalvo->RunPolygons();

   if (ret != DEVICE_OK)
   {
      logError(deviceLabel, getDeviceErrorText(ret, pGalvo).c_str());
      throw CMMError(getDeviceErrorText(ret, pGalvo));
   }
}

/**
 * Run a sequence of galvo positions
 */
void CMMCore::runGalvoSequence(const char* deviceLabel) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<GalvoInstance> pGalvo =
      deviceManager_->GetDeviceOfType<GalvoInstance>(deviceLabel);

   mm::DeviceModuleLockGuard guard(pGalvo);

   int ret =  pGalvo->RunSequence();

   if (ret != DEVICE_OK)
   {
      logError(deviceLabel, getDeviceErrorText(ret, pGalvo).c_str());
      throw CMMError(getDeviceErrorText(ret, pGalvo));
   }
}

/**
 * Get the name of the active galvo channel (for a multi-laser galvo device).
 */
std::string CMMCore::getGalvoChannel(const char* deviceLabel) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<GalvoInstance> pGalvo =
      deviceManager_->GetDeviceOfType<GalvoInstance>(deviceLabel);

   mm::DeviceModuleLockGuard guard(pGalvo);
   return pGalvo->GetChannel();
}

///////////////////////////////////////////////////////////////////////////////
//  Pressure Pump methods
///////////////////////////////////////////////////////////////////////////////


/**
* Stops the pressure pump
*/
void CMMCore::pressurePumpStop(const char* deviceLabel) MMCORE_LEGACY_THROW(CMMError)
{
    std::shared_ptr<PressurePumpInstance> pPump =
        deviceManager_->GetDeviceOfType<PressurePumpInstance>(deviceLabel);
    mm::DeviceModuleLockGuard guard(pPump);

    int ret = pPump->Stop();

    if (ret != DEVICE_OK)
    {
        logError(deviceLabel, getDeviceErrorText(ret, pPump).c_str());
        throw CMMError(getDeviceErrorText(ret, pPump));
    }
}

/**
* Calibrates the pump
*/
void CMMCore::pressurePumpCalibrate(const char* deviceLabel) MMCORE_LEGACY_THROW(CMMError)
{
    std::shared_ptr<PressurePumpInstance> pPump =
        deviceManager_->GetDeviceOfType<PressurePumpInstance>(deviceLabel);
    mm::DeviceModuleLockGuard guard(pPump);

    int ret = pPump->Calibrate();

    if (ret != DEVICE_OK)
    {
        logError(deviceLabel, getDeviceErrorText(ret, pPump).c_str());
        throw CMMError(getDeviceErrorText(ret, pPump));
    }
}

/**
* Returns boolean whether the pump is operational before calibration
*/
bool CMMCore::pressurePumpRequiresCalibration(const char* deviceLabel) MMCORE_LEGACY_THROW(CMMError)
{
    std::shared_ptr<PressurePumpInstance> pPump =
        deviceManager_->GetDeviceOfType<PressurePumpInstance>(deviceLabel);
    mm::DeviceModuleLockGuard guard(pPump);

    return pPump->RequiresCalibration();
}

/**
* Gets the pressure of the pump in kPa
*/
double CMMCore::getPumpPressureKPa(const char* deviceLabel) MMCORE_LEGACY_THROW(CMMError)
{
    std::shared_ptr<PressurePumpInstance> pPump =
        deviceManager_->GetDeviceOfType<PressurePumpInstance>(deviceLabel);
    mm::DeviceModuleLockGuard guard(pPump);

    double pressurekPa = 0;
    int ret = pPump->GetPressureKPa(pressurekPa);

    if (ret != DEVICE_OK)
    {
        logError(deviceLabel, getDeviceErrorText(ret, pPump).c_str());
        throw CMMError(getDeviceErrorText(ret, pPump));
    }
    return pressurekPa;
}

/**
* Sets the pressure of the pump in kPa
*/
void CMMCore::setPumpPressureKPa(const char* deviceLabel, double pressurekPa) MMCORE_LEGACY_THROW(CMMError)
{
    std::shared_ptr<PressurePumpInstance> pPump =
        deviceManager_->GetDeviceOfType<PressurePumpInstance>(deviceLabel);
    mm::DeviceModuleLockGuard guard(pPump);

    int ret = pPump->SetPressureKPa(pressurekPa);

    if (ret != DEVICE_OK)
    {
        logError(deviceLabel, getDeviceErrorText(ret, pPump).c_str());
        throw CMMError(getDeviceErrorText(ret, pPump));
    }
}

/**
* Stops the volumetric pump
*/
void CMMCore::volumetricPumpStop(const char* deviceLabel) MMCORE_LEGACY_THROW(CMMError)
{
    std::shared_ptr<VolumetricPumpInstance> pPump =
        deviceManager_->GetDeviceOfType<VolumetricPumpInstance>(deviceLabel);
    mm::DeviceModuleLockGuard guard(pPump);

    int ret = pPump->Stop();

    if (ret != DEVICE_OK)
    {
        logError(deviceLabel, getDeviceErrorText(ret, pPump).c_str());
        throw CMMError(getDeviceErrorText(ret, pPump));
    }
}

/**
* Homes the pump
*/
void CMMCore::volumetricPumpHome(const char* deviceLabel) MMCORE_LEGACY_THROW(CMMError)
{
    std::shared_ptr<VolumetricPumpInstance> pPump =
        deviceManager_->GetDeviceOfType<VolumetricPumpInstance>(deviceLabel);
    mm::DeviceModuleLockGuard guard(pPump);

    int ret = pPump->Home();

    if (ret != DEVICE_OK)
    {
        logError(deviceLabel, getDeviceErrorText(ret, pPump).c_str());
        throw CMMError(getDeviceErrorText(ret, pPump));
    }
}

bool CMMCore::volumetricPumpRequiresHoming(const char* deviceLabel) MMCORE_LEGACY_THROW(CMMError)
{
    std::shared_ptr<VolumetricPumpInstance> pPump =
        deviceManager_->GetDeviceOfType<VolumetricPumpInstance>(deviceLabel);
    mm::DeviceModuleLockGuard guard(pPump);

    return pPump->RequiresHoming();
}

/**
* Sets whether the pump direction needs to be inverted
*/
void CMMCore::invertPumpDirection(const char* deviceLabel, bool invert) MMCORE_LEGACY_THROW(CMMError)
{
    std::shared_ptr<VolumetricPumpInstance> pPump =
        deviceManager_->GetDeviceOfType<VolumetricPumpInstance>(deviceLabel);
    mm::DeviceModuleLockGuard guard(pPump);

    int ret = pPump->InvertDirection(invert);

    if (ret != DEVICE_OK)
    {
        logError(deviceLabel, getDeviceErrorText(ret, pPump).c_str());
        throw CMMError(getDeviceErrorText(ret, pPump));
    }
}

/**
* Gets whether the pump direction needs to be inverted
*/
bool CMMCore::isPumpDirectionInverted(const char* deviceLabel) MMCORE_LEGACY_THROW(CMMError)
{
    std::shared_ptr<VolumetricPumpInstance> pPump =
        deviceManager_->GetDeviceOfType<VolumetricPumpInstance>(deviceLabel);
    mm::DeviceModuleLockGuard guard(pPump);

    bool invert = false;
    int ret = pPump->IsDirectionInverted(invert);

    if (ret != DEVICE_OK)
    {
        logError(deviceLabel, getDeviceErrorText(ret, pPump).c_str());
        throw CMMError(getDeviceErrorText(ret, pPump));
    }
    return invert;
}

/**
* Sets the volume of fluid in the pump in uL. Note it does not withdraw upto
* this amount. It is merely to inform MM of the volume in a prefilled pump.
*/
void CMMCore::setPumpVolume(const char* deviceLabel, double volUl) MMCORE_LEGACY_THROW(CMMError)
{
    std::shared_ptr<VolumetricPumpInstance> pPump =
        deviceManager_->GetDeviceOfType<VolumetricPumpInstance>(deviceLabel);
    mm::DeviceModuleLockGuard guard(pPump);

    int ret = pPump->SetVolumeUl(volUl);

    if (ret != DEVICE_OK)
    {
        logError(deviceLabel, getDeviceErrorText(ret, pPump).c_str());
        throw CMMError(getDeviceErrorText(ret, pPump));
    }
}

/**
* Get the fluid volume in the pump in uL
*/
double CMMCore::getPumpVolume(const char* deviceLabel) MMCORE_LEGACY_THROW(CMMError)
{
    std::shared_ptr<VolumetricPumpInstance> pPump =
        deviceManager_->GetDeviceOfType<VolumetricPumpInstance>(deviceLabel);
    mm::DeviceModuleLockGuard guard(pPump);

    double volUl = 0;
    int ret = pPump->GetVolumeUl(volUl);

    if (ret != DEVICE_OK)
    {
        logError(deviceLabel, getDeviceErrorText(ret, pPump).c_str());
        throw CMMError(getDeviceErrorText(ret, pPump));
    }
    return volUl;
}

/**
* Sets the max volume of the pump in uL
*/
void CMMCore::setPumpMaxVolume(const char* deviceLabel, double volUl) MMCORE_LEGACY_THROW(CMMError)
{
    std::shared_ptr<VolumetricPumpInstance> pPump =
        deviceManager_->GetDeviceOfType<VolumetricPumpInstance>(deviceLabel);
    mm::DeviceModuleLockGuard guard(pPump);

    int ret = pPump->SetMaxVolumeUl(volUl);

    if (ret != DEVICE_OK)
    {
        logError(deviceLabel, getDeviceErrorText(ret, pPump).c_str());
        throw CMMError(getDeviceErrorText(ret, pPump));
    }
}

/**
* Gets the max volume of the pump in uL
*/
double CMMCore::getPumpMaxVolume(const char* deviceLabel) MMCORE_LEGACY_THROW(CMMError)
{
    std::shared_ptr<VolumetricPumpInstance> pPump =
        deviceManager_->GetDeviceOfType<VolumetricPumpInstance>(deviceLabel);
    mm::DeviceModuleLockGuard guard(pPump);

    double volUl = 0;
    int ret = pPump->GetMaxVolumeUl(volUl);

    if (ret != DEVICE_OK)
    {
        logError(deviceLabel, getDeviceErrorText(ret, pPump).c_str());
        throw CMMError(getDeviceErrorText(ret, pPump));
    }
    return volUl;
}

/**
* Sets the flowrate of the pump in uL per second
*/
void CMMCore::setPumpFlowrate(const char* deviceLabel, double UlperSec) MMCORE_LEGACY_THROW(CMMError)
{
    std::shared_ptr<VolumetricPumpInstance> pPump =
        deviceManager_->GetDeviceOfType<VolumetricPumpInstance>(deviceLabel);
    mm::DeviceModuleLockGuard guard(pPump);

    int ret = pPump->SetFlowrateUlPerSecond(UlperSec);

    if (ret != DEVICE_OK)
    {
        logError(deviceLabel, getDeviceErrorText(ret, pPump).c_str());
        throw CMMError(getDeviceErrorText(ret, pPump));
    }
}

/**
* Gets the flowrate of the pump in uL per second
*/
double CMMCore::getPumpFlowrate(const char* deviceLabel) MMCORE_LEGACY_THROW(CMMError)
{
    std::shared_ptr<VolumetricPumpInstance> pPump =
        deviceManager_->GetDeviceOfType<VolumetricPumpInstance>(deviceLabel);
    mm::DeviceModuleLockGuard guard(pPump);

    double UlperSec = 0;
    int ret = pPump->GetFlowrateUlPerSecond(UlperSec);

    if (ret != DEVICE_OK)
    {
        logError(deviceLabel, getDeviceErrorText(ret, pPump).c_str());
        throw CMMError(getDeviceErrorText(ret, pPump));
    }
    return UlperSec;
}

/**
* Start dispensing at the set flowrate until syringe is empty, or manually
* stopped (whichever occurs first).
*/
void CMMCore::pumpStart(const char* deviceLabel) MMCORE_LEGACY_THROW(CMMError)
{
    std::shared_ptr<VolumetricPumpInstance> pPump =
        deviceManager_->GetDeviceOfType<VolumetricPumpInstance>(deviceLabel);
    mm::DeviceModuleLockGuard guard(pPump);

    int ret = pPump->Start();

    if (ret != DEVICE_OK)
    {
        logError(deviceLabel, getDeviceErrorText(ret, pPump).c_str());
        throw CMMError(getDeviceErrorText(ret, pPump));
    }
}

/**
* Dispenses for the provided duration (in seconds) at the set flowrate
*/
void CMMCore::pumpDispenseDurationSeconds(const char* deviceLabel, double seconds) MMCORE_LEGACY_THROW(CMMError)
{
    std::shared_ptr<VolumetricPumpInstance> pPump =
        deviceManager_->GetDeviceOfType<VolumetricPumpInstance>(deviceLabel);
    mm::DeviceModuleLockGuard guard(pPump);

    int ret = pPump->DispenseDurationSeconds(seconds);

    if (ret != DEVICE_OK)
    {
        logError(deviceLabel, getDeviceErrorText(ret, pPump).c_str());
        throw CMMError(getDeviceErrorText(ret, pPump));
    }
}

/**
* Dispenses the provided volume (in uL) at the set flowrate
*/
void CMMCore::pumpDispenseVolumeUl(const char* deviceLabel, double microLiter) MMCORE_LEGACY_THROW(CMMError)
{
    std::shared_ptr<VolumetricPumpInstance> pPump =
        deviceManager_->GetDeviceOfType<VolumetricPumpInstance>(deviceLabel);
    mm::DeviceModuleLockGuard guard(pPump);

    int ret = pPump->DispenseVolumeUl(microLiter);

    if (ret != DEVICE_OK)
    {
        logError(deviceLabel, getDeviceErrorText(ret, pPump).c_str());
        throw CMMError(getDeviceErrorText(ret, pPump));
    }
}

/* SYSTEM STATE */


/**
 * Saves the current system state to a text file of the MM specific format.
 * The file records only read-write properties.
 * The file format is directly readable by the complementary loadSystemState() command.
 */
void CMMCore::saveSystemState(const char* fileName) MMCORE_LEGACY_THROW(CMMError)
{
   if (!fileName)
      throw CMMError("Null filename");

   std::ofstream os;
   os.open(fileName, std::ios_base::out | std::ios_base::trunc);
   if (!os.is_open())
   {
      logError(fileName, getCoreErrorText(MMERR_FileOpenFailed).c_str());
      throw CMMError(ToQuotedString(fileName) + ": " + getCoreErrorText(MMERR_FileOpenFailed),
            MMERR_FileOpenFailed);
   }

   // save system state
   Configuration config = getSystemState();
   for (size_t i=0; i<config.size(); i++)
   {
      PropertySetting s = config.getSetting(i);
      if (!isPropertyReadOnly(s.getDeviceLabel().c_str(), s.getPropertyName().c_str()))
      {
         os << MM::g_CFGCommand_Property << ',' << s.getDeviceLabel()
            << ',' << s.getPropertyName() << ',' << s.getPropertyValue() << '\n';
      }
   }
}

/**
 * Loads the system configuration from the text file conforming to the MM specific format.
 * The configuration contains a list of commands to build the desired system state from
 * read-write properties.
 *
 * Format specification: the same as in loadSystemConfiguration() command
 */
void CMMCore::loadSystemState(const char* fileName) MMCORE_LEGACY_THROW(CMMError)
{
   if (!fileName)
      throw CMMError("Null filename");

   std::ifstream is;
   is.open(fileName, std::ios_base::in);
   if (!is.is_open())
   {
      logError(fileName, getCoreErrorText(MMERR_FileOpenFailed).c_str());
      throw CMMError(ToQuotedString(fileName) + ": " + getCoreErrorText(MMERR_FileOpenFailed),
            MMERR_FileOpenFailed);
   }

   // Process commands
   const int maxLineLength = 4 * MM::MaxStrLength + 4; // accommodate up to 4 strings and delimiters
   char line[maxLineLength+1];
   std::vector<std::string> tokens;
   while(is.getline(line, maxLineLength, '\n'))
   {
      // strip a potential Windows/dos CR
      std::istringstream il(line);
      il.getline(line, maxLineLength, '\r');
      if (strlen(line) > 0)
      {
         if (line[0] == '#')
         {
            // comment, so skip processing
            continue;
         }

         // parse tokens
         tokens.clear();
         CDeviceUtils::Tokenize(line, tokens, MM::g_FieldDelimiters);

         // non-empty and non-comment lines mush have at least one token
         if (tokens.size() < 1)
            throw CMMError(getCoreErrorText(MMERR_InvalidCFGEntry) + " (" +
                  ToQuotedString(line) + ")",
                  MMERR_InvalidCFGEntry);

         if(tokens[0].compare(MM::g_CFGCommand_Property) == 0)
         {
            // set property command
            // --------------------
            if (tokens.size() != 4)
               // invalid format
               throw CMMError(getCoreErrorText(MMERR_InvalidCFGEntry) + " (" +
                     ToQuotedString(line) + ")",
                     MMERR_InvalidCFGEntry);
            try
            {
               // apply the command
               setProperty(tokens[1].c_str(), tokens[2].c_str(), tokens[3].c_str());
            }
            catch (CMMError&)
            {
               // Don't give up yet.
               // TODO Yes, do give up, unless cleanly recoverable.
            }
         }
      }
   }
}


/**
 * Saves the current system configuration to a text file of the MM specific format.
 * The configuration file records only the information essential to the hardware
 * setup: devices, labels, pre-initialization properties, and configurations.
 * The file format is the same as for the system state.
 */
void CMMCore::saveSystemConfiguration(const char* fileName) MMCORE_LEGACY_THROW(CMMError)
{
   if (!fileName)
      throw CMMError("Null filename");

   std::ofstream os;
   os.open(fileName, std::ios_base::out | std::ios_base::trunc);
   if (!os.is_open())
   {
      logError(fileName, getCoreErrorText(MMERR_FileOpenFailed).c_str());
      throw CMMError(ToQuotedString(fileName) + ": " + getCoreErrorText(MMERR_FileOpenFailed),
            MMERR_FileOpenFailed);
   }

   // insert the system reset command
   // this will unload all current devices
   os << "# Unload all devices\n";
   os << "Property,Core,Initialize,0\n";

   // save device list
   os << "# Load devices\n";
   std::vector<std::string> devices = deviceManager_->GetDeviceList();
   std::vector<std::string>::const_iterator it;
   for (it=devices.begin(); it != devices.end(); it++)
   {
      std::shared_ptr<DeviceInstance> pDev = deviceManager_->GetDevice(*it);
      mm::DeviceModuleLockGuard guard(pDev);
      os << MM::g_CFGCommand_Device << "," << *it << "," << pDev->GetAdapterModule()->GetName() << "," << pDev->GetName() << '\n';
   }

   // save the pre-initialization properties
   os << "# Pre-initialization properties\n";
   Configuration config = getSystemState();
   for (size_t i=0; i<config.size(); i++)
   {
      PropertySetting s = config.getSetting(i);
      if (s.getDeviceLabel() == MM::g_Keyword_CoreDevice)
         continue;

      // check if the property must be set before initialization
      std::shared_ptr<DeviceInstance> pDevice = deviceManager_->GetDevice(s.getDeviceLabel());
      if (pDevice)
      {
         mm::DeviceModuleLockGuard guard(pDevice);
         bool isPreInit = pDevice->GetPropertyInitStatus(s.getPropertyName().c_str());
         if (isPreInit)
         {
            os << MM::g_CFGCommand_Property << ',' << s.getDeviceLabel()
               << ',' << s.getPropertyName() << ',' << s.getPropertyValue() << '\n';
         }
      }
   }

   // save the parent (hub) references
   os << "# Hub references" << '\n';
   for (it=devices.begin(); it != devices.end(); it++)
   {
      std::shared_ptr<DeviceInstance> device = deviceManager_->GetDevice(*it);
      mm::DeviceModuleLockGuard guard(device);
      std::string parentID = device->GetParentID();
      if (!parentID.empty())
      {
         os << MM::g_CFGCommand_ParentID << ',' << device->GetLabel() << ',' << parentID << '\n';
      }
   }


   // insert the initialize command
   os << "Property,Core,Initialize,1\n";

   // save delays
   os << "# Delays\n";
   for (it=devices.begin(); it != devices.end(); it++)
   {
      std::shared_ptr<DeviceInstance> pDev = deviceManager_->GetDevice(*it);
      mm::DeviceModuleLockGuard guard(pDev);
      if (pDev->GetDelayMs() > 0.0)
         os << MM::g_CFGCommand_Delay << "," << *it << "," << pDev->GetDelayMs() << '\n';
   }

   // save focus directions
   os << "# Stage focus directions\n";
   std::vector<std::string> stageLabels =
      deviceManager_->GetDeviceList(MM::StageDevice);
   for (std::vector<std::string>::const_iterator stageIt = stageLabels.begin(),
         end = stageLabels.end(); stageIt != end; ++stageIt)
   {
      std::shared_ptr<StageInstance> stage =
         deviceManager_->GetDeviceOfType<StageInstance>(*stageIt);
      mm::DeviceModuleLockGuard guard(stage);
      int direction = getFocusDirection(stageIt->c_str());
      os << MM::g_CFGCommand_FocusDirection << ','
         << *stageIt << ',' << direction << '\n';
   }

   // save labels
   os << "# Labels\n";
   std::vector<std::string> deviceLabels = deviceManager_->GetDeviceList(MM::StateDevice);
   for (size_t i=0; i<deviceLabels.size(); i++)
   {
      std::shared_ptr<StateInstance> pSD =
         deviceManager_->GetDeviceOfType<StateInstance>(deviceLabels[i]);
      mm::DeviceModuleLockGuard guard(pSD);
      unsigned numPos = pSD->GetNumberOfPositions();
      for (unsigned long j=0; j<numPos; j++)
      {
         std::string stateLabel;
         try
         {
            stateLabel = pSD->GetPositionLabel(j);
         }
         catch (const CMMError&)
         {
            // Label not defined, just skip
            continue;
         }
         if (!stateLabel.empty())
         {
            os << MM::g_CFGCommand_Label << ',' << deviceLabels[i] << ',' << j << ',' << stateLabel << '\n';
         }
      }
   }
   os << '\n';

   // save configuration groups
   os << "# Group configurations\n";
   std::vector<std::string> groups = getAvailableConfigGroups();
   for (size_t i=0; i<groups.size(); i++)
   {
      // empty group record
      std::vector<std::string> configs = getAvailableConfigs(groups[i].c_str());
      if (configs.size() == 0)
            os << MM::g_CFGCommand_ConfigGroup << ',' << groups[i] << '\n';

      // normal group records
      for (size_t j=0; j<configs.size(); j++)
      {
         Configuration c = getConfigData(groups[i].c_str(), configs[j].c_str());
         for (size_t k=0; k<c.size(); k++)
         {
            PropertySetting s = c.getSetting(k);
            os << MM::g_CFGCommand_ConfigGroup << ',' << groups[i] << ','
               << configs[j] << ',' << s.getDeviceLabel() << ',' << s.getPropertyName() << ',' << s.getPropertyValue() << '\n';
         }
      }
   }
   os << '\n';

   // save Pixel Size configurations
   os << "# Pixel Size configurations\n";
   std::vector<std::string> pixelSizeGroups = getAvailablePixelSizeConfigs();
   for (size_t i = 0; i < pixelSizeGroups.size(); i++)
   {
      Configuration psc = getPixelSizeConfigData(pixelSizeGroups[i].c_str());
         for (size_t k=0; k< psc.size(); k++)
         {
            PropertySetting s = psc.getSetting(k);
            os << MM::g_CFGCommand_ConfigPixelSize << ',' << pixelSizeGroups[i] << ','
               << s.getDeviceLabel() << ',' << s.getPropertyName() << ',' << s.getPropertyValue() << '\n';
         }
         os << MM::g_CFGCommand_PixelSize_um << ',' << pixelSizeGroups[i].c_str() << ',' << getPixelSizeUmByID(pixelSizeGroups[i].c_str()) << '\n';
         std::vector<double> affines = getPixelSizeAffineByID(pixelSizeGroups[i].c_str());
         if (affines.size() == 6)
         {
            os << MM::g_CFGCommand_PixelSizeAffine << ',' << pixelSizeGroups[i].c_str() << ',';
            for (int l = 0; l < 5; l++)
            {
               os << affines[l] << ',';
            }
            os << affines[5] << '\n';
         }
         os << MM::g_CFGCommand_PixelSizedxdz << ',' << pixelSizeGroups[i].c_str() << ',' 
            << getPixelSizedxdz(pixelSizeGroups[i].c_str()) << '\n';
         os << MM::g_CFGCommand_PixelSizedydz << ',' << pixelSizeGroups[i].c_str() << ',' 
            << getPixelSizedydz(pixelSizeGroups[i].c_str()) << '\n';
         os << MM::g_CFGCommand_PixelSizeOptimalZUm << ',' << pixelSizeGroups[i].c_str() << ',' 
            << getPixelSizeOptimalZUm(pixelSizeGroups[i].c_str()) << '\n';
   }
   os << '\n';
    
   // save device roles
   os << "# Roles\n";
   std::shared_ptr<CameraInstance> camera = currentCameraDevice_.lock();
   if (camera)
   {
      os << MM::g_CFGCommand_Property << ',' << MM::g_Keyword_CoreDevice << ',' << MM::g_Keyword_CoreCamera << ',' << camera->GetLabel() << '\n';
   }
   std::shared_ptr<ShutterInstance> shutter = currentShutterDevice_.lock();
   if (shutter)
   {
      os << MM::g_CFGCommand_Property << ',' << MM::g_Keyword_CoreDevice << ',' << MM::g_Keyword_CoreShutter << ',' << shutter->GetLabel() << '\n';
   }
   std::shared_ptr<StageInstance> focus = currentFocusDevice_.lock();
   if (focus)
   {
      os << MM::g_CFGCommand_Property << ',' << MM::g_Keyword_CoreDevice << ',' << MM::g_Keyword_CoreFocus << ',' << focus->GetLabel() << '\n';
   }
}

/**
 * Loads the system configuration from the text file conforming to the MM specific format.
 * The configuration contains a list of commands to build the desired system state:
 * devices, labels, properties, and configurations.
 *
 * Format specification:
 * Each line consists of a number of string fields separated by "," (comma) characters.
 * Lines beginning with "#" are ignored (can be used for comments).
 * Each line in the file will be parsed by the system and as a result a corresponding command
 * will be immediately executed.
 * The first field in the line always specifies the command from the following set of values:
 *    Device - executes loadDevice()
 *    Label - executes defineStateLabel() command
 *    Property - executes setPropertyCommand()
 *    Configuration - ignored for backward compatibility
 *
 * The remaining fields in the line will be used for corresponding command parameters.
 * The number of parameters depends on the actual command used.
 *
 * This function is not thread-safe.
 */
void CMMCore::loadSystemConfiguration(const char* fileName) MMCORE_LEGACY_THROW(CMMError)
{
   try
   {
      isLoadingSystemConfiguration_ = true;
      loadSystemConfigurationImpl(fileName);
      isLoadingSystemConfiguration_ = false;
   }
   catch (const CMMError&)
   {
      isLoadingSystemConfiguration_ = false;

      // Unload all devices so as not to leave loaded but uninitialized devices
      // (which are prone to cause a crash when accessed) hanging around.
      LOG_INFO(coreLogger_) <<
         "Unloading all devices after failure to load system configuration";

      try
      {
         // Also emits onSystemConfigurationLoaded to indicate config changed:
         unloadAllDevices();
      }
      catch (const CMMError& err)
      {
         LOG_ERROR(coreLogger_) <<
            "Error occurred while unloading all devices: " <<
            err.getFullMsg();
      }

      LOG_INFO(coreLogger_) <<
         "Now rethrowing original error from system configuration loading";
      throw;
   }

   if (externalCallback_)
   {
      externalCallback_->onSystemConfigurationLoaded();
   }
}


void CMMCore::loadSystemConfigurationImpl(const char* fileName) MMCORE_LEGACY_THROW(CMMError)
{
   if (!fileName)
      throw CMMError("Null filename");

   LOG_INFO(coreLogger_) << "Loading system configuration from:" << ToQuotedString(fileName);

   std::ifstream is;
   is.open(fileName, std::ios_base::in);
   if (!is.is_open())
   {
      logError(fileName, getCoreErrorText(MMERR_FileOpenFailed).c_str());
      throw CMMError(ToQuotedString(fileName) + ": " + getCoreErrorText(MMERR_FileOpenFailed),
            MMERR_FileOpenFailed);
   }

   // Process commands
   const int maxLineLength = 4 * MM::MaxStrLength + 4; // accommodate up to 4 strings and delimiters
   char line[maxLineLength+1];
   std::vector<std::string> tokens;

   int lineCount = 0;

   while(is.getline(line, maxLineLength, '\n'))
   {
      // strip a potential Windows/dos CR
      std::istringstream il(line);
      il.getline(line, maxLineLength, '\r');

      lineCount++;
      if (strlen(line) > 0)
      {
         if (line[0] == '#')
         {
            // comment, so skip processing
            continue;
         }

         // parse tokens
         tokens.clear();
         CDeviceUtils::Tokenize(line, tokens, MM::g_FieldDelimiters);

         try
         {

            // non-empty and non-comment lines mush have at least one token
            if (tokens.size() < 1)
               throw CMMError(getCoreErrorText(MMERR_InvalidCFGEntry) + " (" +
                     ToQuotedString(line) + ")",
                     MMERR_InvalidCFGEntry);

            if(tokens[0].compare(MM::g_CFGCommand_Device) == 0)
            {
               // load device command
               // -------------------
               if (tokens.size() != 4)
                  throw CMMError(getCoreErrorText(MMERR_InvalidCFGEntry) + " (" +
                        ToQuotedString(line) + ")",
                        MMERR_InvalidCFGEntry);
               loadDevice(tokens[1].c_str(), tokens[2].c_str(), tokens[3].c_str());
            }
            else if(tokens[0].compare(MM::g_CFGCommand_Property) == 0)
            {
               // set property command
               // --------------------
               if (tokens.size() == 4)
                  setProperty(tokens[1].c_str(), tokens[2].c_str(), tokens[3].c_str());
               else if (tokens.size() == 3)
                  // ...assuming here that the last missing toke represents an empty string
                  setProperty(tokens[1].c_str(), tokens[2].c_str(), "");
               else
                  throw CMMError(getCoreErrorText(MMERR_InvalidCFGEntry) + " (" +
                        ToQuotedString(line) + ")",
                        MMERR_InvalidCFGEntry);
            }
            else if(tokens[0].compare(MM::g_CFGCommand_Delay) == 0)
            {
               // set delay command
               // -----------------
               if (tokens.size() != 3)
                  throw CMMError(getCoreErrorText(MMERR_InvalidCFGEntry) + " (" +
                        ToQuotedString(line) + ")",
                        MMERR_InvalidCFGEntry);
               setDeviceDelayMs(tokens[1].c_str(), atof(tokens[2].c_str()));
            }
            else if(tokens[0].compare(MM::g_CFGCommand_FocusDirection) == 0)
            {
               // set focus direction command
               // ---------------------------
               if (tokens.size() != 3)
                  throw CMMError(getCoreErrorText(MMERR_InvalidCFGEntry) + " (" +
                        ToQuotedString(line) + ")",
                        MMERR_InvalidCFGEntry);
               setFocusDirection(tokens[1].c_str(), atol(tokens[2].c_str()));
            }
            else if(tokens[0].compare(MM::g_CFGCommand_Label) == 0)
            {
               // define label command
               // --------------------
               if (tokens.size() != 4)
                  throw CMMError(getCoreErrorText(MMERR_InvalidCFGEntry) + " (" +
                        ToQuotedString(line) + ")",
                        MMERR_InvalidCFGEntry);
               defineStateLabel(tokens[1].c_str(), atol(tokens[2].c_str()), tokens[3].c_str());
            }
            else if(tokens[0].compare(MM::g_CFGCommand_Configuration) == 0)
            {
               // define configuration command
               // ----------------------------
               if (tokens.size() != 5)
                  throw CMMError(getCoreErrorText(MMERR_InvalidCFGEntry) + " (" +
                        ToQuotedString(line) + ")",
                        MMERR_InvalidCFGEntry);
               LOG_WARNING(coreLogger_) << "Obsolete command " << tokens[0] <<
                  " ignored in configuration file";
            }
            else if(tokens[0].compare(MM::g_CFGCommand_ConfigGroup) == 0)
            {
               // define grouped configuration command
               // ------------------------------------
               if (tokens.size() == 6)
                  defineConfig(tokens[1].c_str(), tokens[2].c_str(), tokens[3].c_str(), tokens[4].c_str(), tokens[5].c_str());
               else if (tokens.size() == 5)
               {
                  // we will assume here that the last (missing) token is representing an empty string
                  defineConfig(tokens[1].c_str(), tokens[2].c_str(), tokens[3].c_str(), tokens[4].c_str(), "");
               }
               else if (tokens.size() == 2)
                  defineConfigGroup(tokens[1].c_str());
               else
                  throw CMMError(getCoreErrorText(MMERR_InvalidCFGEntry) + " (" +
                        ToQuotedString(line) + ")",
                        MMERR_InvalidCFGEntry);
            }
            else if(tokens[0].compare(MM::g_CFGCommand_ConfigPixelSize) == 0)
            {
               // define pixel size configuration command
               // ---------------------------------------
               if (tokens.size() == 5)
                  definePixelSizeConfig(tokens[1].c_str(), tokens[2].c_str(), tokens[3].c_str(), tokens[4].c_str());
               else
                  throw CMMError(getCoreErrorText(MMERR_InvalidCFGEntry) + " (" +
                        ToQuotedString(line) + ")",
                        MMERR_InvalidCFGEntry);
            }
            else if(tokens[0].compare(MM::g_CFGCommand_PixelSize_um) == 0)
            {
               // set pixel size
               // --------------
               if (tokens.size() == 3)
                  setPixelSizeUm(tokens[1].c_str(), atof(tokens[2].c_str()));
               else
                  throw CMMError(getCoreErrorText(MMERR_InvalidCFGEntry) + " (" +
                        ToQuotedString(line) + ")",
                        MMERR_InvalidCFGEntry);
            }
            else if(tokens[0].compare(MM::g_CFGCommand_PixelSizeAffine) == 0)
            {
               // set affine transform
               // --------------
               //
               if (tokens.size() == 8)
               {
                  std::vector<double> *affineT = new std::vector<double>(6);
                  for (int i = 0; i < 6; i++)
                  {
                     affineT->at(i) = atof(tokens[i + 2].c_str());
                  }
                  setPixelSizeAffine(tokens[1].c_str(), *affineT);
                  delete affineT;
               }
               else
                  throw CMMError(getCoreErrorText(MMERR_InvalidCFGEntry) + " (" +
                        ToQuotedString(line) + ")",
                        MMERR_InvalidCFGEntry);
            }
            else if (tokens[0].compare(MM::g_CFGCommand_PixelSizedxdz) == 0)
            {
               if (tokens.size() == 3)
                  setPixelSizedxdz(tokens[1].c_str(), atof(tokens[2].c_str()));
               else
                  throw CMMError(getCoreErrorText(MMERR_InvalidCFGEntry) + " (" +
                        ToQuotedString(line) + ")",
                        MMERR_InvalidCFGEntry);
            }
            else if (tokens[0].compare(MM::g_CFGCommand_PixelSizedydz) == 0)
            {
               if (tokens.size() == 3)
                  setPixelSizedydz(tokens[1].c_str(), atof(tokens[2].c_str()));
               else
                  throw CMMError(getCoreErrorText(MMERR_InvalidCFGEntry) + " (" +
                        ToQuotedString(line) + ")",
                        MMERR_InvalidCFGEntry);
            }
            else if (tokens[0].compare(MM::g_CFGCommand_PixelSizeOptimalZUm) == 0)
            {
               if (tokens.size() == 3)
                  setPixelSizeOptimalZUm(tokens[1].c_str(), atof(tokens[2].c_str()));
               else
                  throw CMMError(getCoreErrorText(MMERR_InvalidCFGEntry) + " (" +
                        ToQuotedString(line) + ")",
                        MMERR_InvalidCFGEntry);
            }
            else if(tokens[0].compare(MM::g_CFGCommand_Equipment) == 0)
            {
              // Property blocks have been removed
              throw CMMError(getCoreErrorText(MMERR_InvalidCFGEntry) + " (" +
                    ToQuotedString(line) + ")",
                    MMERR_InvalidCFGEntry);
            }
            else if(tokens[0].compare(MM::g_CFGCommand_ImageSynchro) == 0)
            {
               // ImageSynchro has been removed
               throw CMMError(getCoreErrorText(MMERR_InvalidCFGEntry) + " (" +
                     ToQuotedString(line) + ")",
                     MMERR_InvalidCFGEntry);
            }
            else if(tokens[0].compare(MM::g_CFGCommand_ParentID) == 0)
            {
               // set parent ID
               // -------------
               if (tokens.size() != 3)
                  throw CMMError(getCoreErrorText(MMERR_InvalidCFGEntry) + " (" +
                        ToQuotedString(line) + ")",
                        MMERR_InvalidCFGEntry);

               setParentLabel(tokens[1].c_str(), tokens[2].c_str());
            }

         }
         catch (CMMError& err)
         {
            std::ostringstream errorText;
            errorText << "Line " << lineCount << ": " << line << '\n';
            errorText << err.getFullMsg() << "\n\n";
            throw CMMError(errorText.str().c_str(), MMERR_InvalidConfigurationFile);
         }
      }
   }

   // file parsing finished, try to set startup configuration
   if (isConfigDefined(MM::g_CFGGroup_System, MM::g_CFGGroup_System_Startup))
   {
      // We need to build the system state cache once here because setConfig()
      // can fail in certain cases otherwise.
      waitForSystem();
      updateSystemStateCache();

      this->setConfig(MM::g_CFGGroup_System, MM::g_CFGGroup_System_Startup);
   }

   waitForSystem();
   updateSystemStateCache();
}


/**
 * Register a callback (listener class).
 *
 * MMCore will send notifications on internal events using this interface.
 *
 * Pass nullptr to unregister.
 *
 * The caller is responsible for ensuring that the object pointed to by \p cb
 * remains valid until it is unregistered.
 *
 * This function is not thread safe.
 */
void CMMCore::registerCallback(MMEventCallback* cb)
{
   externalCallback_ = cb;
}


/**
 * Returns the latest focus score from the focusing device.
 * Use this value to estimate or record how reliable the focus is.
 * The range of values is device dependent.
 */
double CMMCore::getLastFocusScore()
{
   std::shared_ptr<AutoFocusInstance> autofocus =
      currentAutofocusDevice_.lock();
   if (autofocus)
   {
      try
      {
         mm::DeviceModuleLockGuard guard(autofocus);
         double score;
         int ret = autofocus->GetLastFocusScore(score);
         if (ret == DEVICE_OK)
            return score;
      }
      catch (const CMMError&) // Probably uninitialized device
      {
         // Fall through
      }
   }
   return 0.0;
}

/**
 * Returns the focus score from the default focusing device measured
 * at the current Z position.
 * Use this value to create profiles or just to verify that the image is in focus.
 * The absolute range of returned scores depends on the actual focusing device.
 */
double CMMCore::getCurrentFocusScore()
{
   std::shared_ptr<AutoFocusInstance> autofocus =
      currentAutofocusDevice_.lock();
   if (autofocus)
   {
      try
      {
         mm::DeviceModuleLockGuard guard(autofocus);
         double score;
         int ret = autofocus->GetCurrentFocusScore(score);
         if (ret == DEVICE_OK)
            return score;
      }
      catch (const CMMError&) // Probably uninitialized device
      {
         // Fall through
      }
   }
   return 0.0;
}


/**
 * Enables or disables the operation of the continuous focusing hardware device.
 */
void CMMCore::enableContinuousFocus(bool enable) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<AutoFocusInstance> autofocus =
      currentAutofocusDevice_.lock();
   if (autofocus)
   {
      mm::DeviceModuleLockGuard guard(autofocus);
	  int ret = autofocus->SetContinuousFocusing(enable);
      if (ret != DEVICE_OK)
      {
         logError(getDeviceName(autofocus).c_str(), getDeviceErrorText(ret, autofocus).c_str());
         throw CMMError(getDeviceErrorText(ret, autofocus).c_str(), MMERR_DEVICE_GENERIC);
      }

      LOG_DEBUG(coreLogger_) << "Continuous autofocus turned " <<
         (enable ? "on" : "off");
   }
   else
   {
      if (enable)
      {
         logError("Core",getCoreErrorText(MMERR_ContFocusNotAvailable).c_str());
         throw CMMError(getCoreErrorText(MMERR_ContFocusNotAvailable).c_str(), MMERR_ContFocusNotAvailable);
      }
   }
}

/**
 * Checks if the continuous focusing hardware device is ON or OFF.
 */
bool CMMCore::isContinuousFocusEnabled() MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<AutoFocusInstance> autofocus =
      currentAutofocusDevice_.lock();
   if (autofocus)
   {
      mm::DeviceModuleLockGuard guard(autofocus);
      bool state;
      int ret = autofocus->GetContinuousFocusing(state);
      if (ret != DEVICE_OK)
      {
         logError(getDeviceName(autofocus).c_str(), getDeviceErrorText(ret, autofocus).c_str());
         throw CMMError(getDeviceErrorText(ret, autofocus).c_str(), MMERR_DEVICE_GENERIC);
      }
      return state;
   }
   else
      return false; // no auto-focus device
}

/**
* Returns the lock-in status of the continuous focusing device.
*/
bool CMMCore::isContinuousFocusLocked() MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<AutoFocusInstance> autofocus =
      currentAutofocusDevice_.lock();
   if (autofocus)
	{
      mm::DeviceModuleLockGuard guard(autofocus);
      return autofocus->IsContinuousFocusLocked();
	}
	else
	{
		return false; // no auto-focus device
	}
}

/**
 * Check if a stage has continuous focusing capability (positions can be set while continuous focus runs).
 */
bool CMMCore::isContinuousFocusDrive(const char* stageLabel) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<StageInstance> pStage =
      deviceManager_->GetDeviceOfType<StageInstance>(stageLabel);

   mm::DeviceModuleLockGuard guard(pStage);
   return pStage->IsContinuousFocusDrive();
}


/**
 * Performs focus acquisition and lock for the one-shot focusing device.
 */
void CMMCore::fullFocus() MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<AutoFocusInstance> autofocus =
      currentAutofocusDevice_.lock();
   if (autofocus)
   {
      mm::DeviceModuleLockGuard guard(autofocus);
      int ret = autofocus->FullFocus();
      if (ret != DEVICE_OK)
      {
         logError(getDeviceName(autofocus).c_str(), getDeviceErrorText(ret, autofocus).c_str());
         throw CMMError(getDeviceErrorText(ret, autofocus).c_str(), MMERR_DEVICE_GENERIC);
      }
   }
   else
   {
      throw CMMError(getCoreErrorText(MMERR_AutoFocusNotAvailable).c_str(), MMERR_AutoFocusNotAvailable);
   }
}

/**
 * Performs incremental focus for the one-shot focusing device.
 */
void CMMCore::incrementalFocus() MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<AutoFocusInstance> autofocus =
      currentAutofocusDevice_.lock();
   if (autofocus)
   {
      mm::DeviceModuleLockGuard guard(autofocus);
      int ret = autofocus->IncrementalFocus();
      if (ret != DEVICE_OK)
      {
         logError(getDeviceName(autofocus).c_str(), getDeviceErrorText(ret, autofocus).c_str());
         throw CMMError(getDeviceErrorText(ret, autofocus).c_str(), MMERR_DEVICE_GENERIC);
      }
   }
   else
   {
      throw CMMError(getCoreErrorText(MMERR_AutoFocusNotAvailable).c_str(), MMERR_AutoFocusNotAvailable);
   }
}


/**
 * Applies offset the one-shot focusing device.
 */
void CMMCore::setAutoFocusOffset(double offset) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<AutoFocusInstance> autofocus =
      currentAutofocusDevice_.lock();
   if (autofocus)
   {
      mm::DeviceModuleLockGuard guard(autofocus);
      int ret = autofocus->SetOffset(offset);
      if (ret != DEVICE_OK)
      {
         logError(getDeviceName(autofocus).c_str(), getDeviceErrorText(ret, autofocus).c_str());
         throw CMMError(getDeviceErrorText(ret, autofocus).c_str(), MMERR_DEVICE_GENERIC);
      }
   }
   else
   {
      throw CMMError(getCoreErrorText(MMERR_AutoFocusNotAvailable).c_str(), MMERR_AutoFocusNotAvailable);
   }
}

/**
 * Measures offset for the one-shot focusing device.
 */
double CMMCore::getAutoFocusOffset() MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<AutoFocusInstance> autofocus =
      currentAutofocusDevice_.lock();
   if (autofocus)
   {
      mm::DeviceModuleLockGuard guard(autofocus);
      double offset;
      int ret = autofocus->GetOffset(offset);
      if (ret != DEVICE_OK)
      {
         logError(getDeviceName(autofocus).c_str(), getDeviceErrorText(ret, autofocus).c_str());
         throw CMMError(getDeviceErrorText(ret, autofocus).c_str(), MMERR_DEVICE_GENERIC);
      }
      return offset;
   }
   else
   {
      throw CMMError(getCoreErrorText(MMERR_AutoFocusNotAvailable).c_str(), MMERR_AutoFocusNotAvailable);
   }
}



///////////////////////////////////////////////////////////////////////////////
// Private methods
///////////////////////////////////////////////////////////////////////////////

void CMMCore::InitializeErrorMessages()
{
   errorText_[MMERR_OK] = "No errors.";
   errorText_[MMERR_GENERIC] = "Core error occurred.";
   errorText_[MMERR_DEVICE_GENERIC] = "Device error encountered.";
   errorText_[MMERR_NoDevice] = "Device not defined or initialized.";
   errorText_[MMERR_SetPropertyFailed] = "Property does not exist, or value not allowed.";
   errorText_[MMERR_LoadLibraryFailed] = "Unable to load library: file not accessible or corrupted.";
   errorText_[MMERR_LibraryFunctionNotFound] =
      "Unable to identify expected interface: the library is not compatible or corrupted.";
   errorText_[MMERR_CreateNotFound] =
      "Unable to identify CreateDevice function: the library is not compatible or corrupted.";
   errorText_[MMERR_DeleteNotFound] =
      "Unable to identify DeleteDevice function: the library is not compatible or corrupted.";
   errorText_[MMERR_CreateFailed] = "DeviceCreate function failed.";
   errorText_[MMERR_DeleteFailed] = "DeviceDelete function failed.";
   errorText_[MMERR_UnknownModule] = "Current device can't be unloaded: using unknown library.";
   errorText_[MMERR_UnexpectedDevice] = "Unexpected device encountered.";
   errorText_[MMERR_ModuleVersionMismatch] = "Module version mismatch.";
   errorText_[MMERR_DeviceVersionMismatch] = "Device interface version mismatch.";
   errorText_[MMERR_DeviceUnloadFailed] =
      "Requested device seems fine, but the current one failed to unload.";
   errorText_[MMERR_CameraNotAvailable] = "Camera not loaded or initialized.";
   errorText_[MMERR_InvalidStateDevice] = "Unsupported API. This device is not a state device";
   errorText_[MMERR_NoConfiguration] = "Configuration not defined";
   errorText_[MMERR_InvalidPropertyBlock] = "Property block not defined"; // No longer used
   errorText_[MMERR_UnhandledException] =
      "Internal inconsistency: unknown system exception encountered";
   errorText_[MMERR_DevicePollingTimeout] = "Device timed out";
   errorText_[MMERR_InvalidShutterDevice] = "Unsupported interface. This device is not a shutter.";
   errorText_[MMERR_DuplicateLabel] = "Specified device label already in use.";
   errorText_[MMERR_InvalidSerialDevice] =
      "Unsupported interface. The specified device is not a serial port.";
   errorText_[MMERR_InvalidSpecificDevice] =
      "Unsupported interface. Device is not of the correct type.";
   errorText_[MMERR_InvalidLabel] = "Can't find the device with the specified label.";
   errorText_[MMERR_FileOpenFailed] = "File open failed.";
   errorText_[MMERR_InvalidCFGEntry] =
      "Invalid configuration file line encountered. Wrong number of tokens for the current context.";
   errorText_[MMERR_InvalidContents] =
      "Reserved character(s) encountered in the value or name string.";
   errorText_[MMERR_InvalidCoreProperty] = "Unrecognized core property.";
   errorText_[MMERR_InvalidCoreValue] =
      "Core property is read-only or the requested value is not allowed.";
   errorText_[MMERR_NoConfigGroup] = "Configuration group not defined";
   errorText_[MMERR_DuplicateConfigGroup] = "Group name already in use.";
   errorText_[MMERR_CameraBufferReadFailed] = "Camera image buffer read failed.";
   errorText_[MMERR_CircularBufferFailedToInitialize] =
      "Failed to initialize circular buffer - memory requirements not adequate.";
   errorText_[MMERR_CircularBufferEmpty] = "Circular buffer is empty.";
   errorText_[MMERR_ContFocusNotAvailable] = "Auto-focus focus device not defined.";
   errorText_[MMERR_BadConfigName] = "Configuration name contains illegal characters (/\\*!')";
   errorText_[MMERR_NotAllowedDuringSequenceAcquisition] =
      "This operation can not be executed while sequence acquisition is running.";
   errorText_[MMERR_OutOfMemory] = "Out of memory.";
   errorText_[MMERR_InvalidImageSequence] = "Issue snapImage before getImage.";
   errorText_[MMERR_NullPointerException] = "Null Pointer Exception.";
   errorText_[MMERR_CreatePeripheralFailed] = "Hub failed to create specified peripheral device.";
   errorText_[MMERR_BadAffineTransform] = "Bad affine transform.  Affine transforms need to have 6 numbers; 2 rows of 3 column.";
}

void CMMCore::CreateCoreProperties()
{
   properties_ = new CorePropertyCollection(this);

   // Initialize
   CoreProperty propInit("0", false, MM::Integer);
   propInit.AddAllowedValue("0");
   propInit.AddAllowedValue("1");
   properties_->Add(MM::g_Keyword_CoreInitialize, propInit);

   // Auto shutter
   CoreProperty propAutoShutter("1", false, MM::Integer);
   propAutoShutter.AddAllowedValue("0");
   propAutoShutter.AddAllowedValue("1");
   properties_->Add(MM::g_Keyword_CoreAutoShutter, propAutoShutter);

   CoreProperty propCamera;
   properties_->Add(MM::g_Keyword_CoreCamera, propCamera);
   properties_->AddAllowedValue(MM::g_Keyword_CoreCamera, "");

   CoreProperty propShutter;
   properties_->Add(MM::g_Keyword_CoreShutter, propShutter);
   properties_->AddAllowedValue(MM::g_Keyword_CoreShutter, "");

   CoreProperty propFocus;
   properties_->Add(MM::g_Keyword_CoreFocus, propFocus);
   properties_->AddAllowedValue(MM::g_Keyword_CoreFocus, "");

   CoreProperty propXYStage;
   properties_->Add(MM::g_Keyword_CoreXYStage, propXYStage);
   properties_->AddAllowedValue(MM::g_Keyword_CoreXYStage, "");

   CoreProperty propAutoFocus;
   properties_->Add(MM::g_Keyword_CoreAutoFocus, propAutoFocus);
   properties_->AddAllowedValue(MM::g_Keyword_CoreAutoFocus, "");

   CoreProperty propImageProc;
   properties_->Add(MM::g_Keyword_CoreImageProcessor, propImageProc);
   properties_->AddAllowedValue(MM::g_Keyword_CoreImageProcessor, "");

   CoreProperty propSLM;
   properties_->Add(MM::g_Keyword_CoreSLM, propSLM);
   properties_->AddAllowedValue(MM::g_Keyword_CoreSLM, "");

   CoreProperty propGalvo;
   properties_->Add(MM::g_Keyword_CoreGalvo, propGalvo);
   properties_->AddAllowedValue(MM::g_Keyword_CoreGalvo, "");

   CoreProperty propChannelGroup;
   properties_->Add(MM::g_Keyword_CoreChannelGroup, propChannelGroup);
   properties_->AddAllowedValue(MM::g_Keyword_CoreChannelGroup, "");

   // Time after which we give up on checking the Busy flag status
   CoreProperty propBusyTimeoutMs("5000", false, MM::Integer);
   properties_->Add(MM::g_Keyword_CoreTimeoutMs, propBusyTimeoutMs);

   properties_->Refresh();
}

static bool ContainsForbiddenCharacters(const std::string& str)
{
   return (std::string::npos != str.find_first_of(MM::g_FieldDelimiters));
}

void CMMCore::CheckDeviceLabel(const char* label) MMCORE_LEGACY_THROW(CMMError)
{
   if (!label)
      throw CMMError("Null device label", MMERR_NullPointerException);
   if (strlen(label) == 0)
      throw CMMError("Empty device label");
   if (ContainsForbiddenCharacters(label))
      throw CMMError("Device label " + ToQuotedString(label) + " contains reserved characters",
            MMERR_InvalidContents);
}

void CMMCore::CheckPropertyName(const char* propName) MMCORE_LEGACY_THROW(CMMError)
{
   if (!propName)
      throw CMMError("Null property name", MMERR_NullPointerException);
   if (ContainsForbiddenCharacters(propName))
      throw CMMError("Property name " + ToQuotedString(propName) + " contains reserved characters",
            MMERR_InvalidContents);
}

void CMMCore::CheckPropertyValue(const char* value) MMCORE_LEGACY_THROW(CMMError)
{
   if (!value)
      throw CMMError("Null property value", MMERR_NullPointerException);
   if (ContainsForbiddenCharacters(value))
      throw CMMError("Property value " + ToQuotedString(value) + " contains reserved characters",
            MMERR_InvalidContents);
}

void CMMCore::CheckStateLabel(const char* stateLabel) MMCORE_LEGACY_THROW(CMMError)
{
   if (!stateLabel)
      throw CMMError("Null state label", MMERR_NullPointerException);
   if (ContainsForbiddenCharacters(stateLabel))
      throw CMMError("State label " + ToQuotedString(stateLabel) + " contains reserved characters",
            MMERR_InvalidContents);
}

void CMMCore::CheckConfigGroupName(const char* groupName) MMCORE_LEGACY_THROW(CMMError)
{
   if (!groupName)
      throw CMMError("Null configuration group name", MMERR_NullPointerException);
   if (ContainsForbiddenCharacters(groupName))
      throw CMMError("Configuration group name " + ToQuotedString(groupName) + " contains reserved characters",
            MMERR_InvalidContents);
}

void CMMCore::CheckConfigPresetName(const char* presetName) MMCORE_LEGACY_THROW(CMMError)
{
   if (!presetName)
      throw CMMError("Null configuration preset name", MMERR_NullPointerException);
   std::string nameString(presetName);
   // XXX Why do we have additional requirement for preset names?
   if (std::string::npos != nameString.find_first_of("/\\*!\'") ||
         ContainsForbiddenCharacters(nameString))
      throw CMMError("Configuration preset name " + ToQuotedString(nameString) +
            " contains reserved or invalid characters",
            MMERR_BadConfigName);
}

bool CMMCore::IsCoreDeviceLabel(const char* label) const MMCORE_LEGACY_THROW(CMMError)
{
   if (!label)
      throw CMMError("Null device label", MMERR_NullPointerException);
   return (strcmp(label, MM::g_Keyword_CoreDevice) == 0);
}

/**
 * Set all properties in a configuration
 * Upon error, don't stop, but try to set all failed properties again
 * until all success or no more change takes place
 * If errors remain, throw an error
 */
void CMMCore::applyConfiguration(const Configuration& config) MMCORE_LEGACY_THROW(CMMError)
{
   std::ostringstream sall;
   bool error = false;
   std::vector<PropertySetting> failedProps;
   for (size_t i=0; i<config.size(); i++)
   {
      PropertySetting setting = config.getSetting(i);

      // perform special processing for core commands
      if (setting.getDeviceLabel().compare(MM::g_Keyword_CoreDevice) == 0)
      {
         properties_->Execute(setting.getPropertyName().c_str(), setting.getPropertyValue().c_str());
         {
            MMThreadGuard scg(stateCacheLock_);
            stateCache_.addSetting(PropertySetting(MM::g_Keyword_CoreDevice, setting.getPropertyName().c_str(), setting.getPropertyValue().c_str()));
         }
      }
      else
      {
         // normal processing
         std::shared_ptr<DeviceInstance> pDevice =
            deviceManager_->GetDevice(setting.getDeviceLabel());
         mm::DeviceModuleLockGuard guard(pDevice);
         try
         {
            pDevice->SetProperty(setting.getPropertyName(),
                  setting.getPropertyValue());

            {
               MMThreadGuard scg(stateCacheLock_);
               stateCache_.addSetting(setting);
            }
         }
         catch (const CMMError&)
         {
            failedProps.push_back(setting);
            error = true;
         }
      }
   }
   if (error)
   {
      std::string errorString;
      while (failedProps.size() > (unsigned) applyProperties(failedProps, errorString) )
      {
         if (failedProps.size() == 0)
            return;
      }

      throw CMMError(errorString.c_str(), MMERR_DEVICE_GENERIC);
   }
}

/*
 * Helper function for applyConfiguration
 * It is possible that setting certain properties failed because they are dependent
 * on other properties to be set first. As a workaround, continue to apply these failed
 * properties until there are none left or none succeed
 * returns number of properties successfully set
 */
int CMMCore::applyProperties(std::vector<PropertySetting>& props, std::string& lastError)
{
  // int succeeded = 0;
   std::vector<PropertySetting> failedProps;
   for (size_t i=0; i<props.size(); i++)
   {
      // normal processing
      std::shared_ptr<DeviceInstance> pDevice =
         deviceManager_->GetDevice(props[i].getDeviceLabel());
      mm::DeviceModuleLockGuard guard(pDevice);
      try
      {
         pDevice->SetProperty(props[i].getPropertyName(),
               props[i].getPropertyValue());

         {
            MMThreadGuard scg(stateCacheLock_);
            stateCache_.addSetting(props[i]);
         }
      }
      catch (const CMMError& e)
      {
         failedProps.push_back(props[i]);
         std::string message = e.getFullMsg();
         logError(props[i].getDeviceLabel().c_str(), message.c_str());
         lastError = message;
      }
   }
   props = failedProps;
   return (int) failedProps.size();
}




std::string CMMCore::getDeviceErrorText(int deviceCode, std::shared_ptr<DeviceInstance> device)
{
   if (!device)
   {
      return "Cannot get error message for null device";
   }

   mm::DeviceModuleLockGuard guard(device);
   return "Error in device " + ToQuotedString(device->GetLabel()) + ": " +
      device->GetErrorText(deviceCode) + " (" + ToString(deviceCode) + ")";
}

/**
 * Returns a pre-defined error test with the given error code
 */
std::string CMMCore::getCoreErrorText(int code) const
{
   // core info
   std::string txt;
   std::map<int, std::string>::const_iterator it;
   it = errorText_.find(code);
   if (it != errorText_.end())
      txt = it->second;

   return txt;
}

void CMMCore::logError(const char* device, const char* msg)
{
   // TODO Fix various inconsistent usages of this function.
   LOG_ERROR(coreLogger_) << "Error occurred in device " << device << ": " << msg;
}

std::string CMMCore::getDeviceName(std::shared_ptr<DeviceInstance> pDev)
{
   mm::DeviceModuleLockGuard guard(pDev);
   return pDev->GetName();
}

void CMMCore::updateAllowedChannelGroups()
{
   std::vector<std::string> groups = getAvailableConfigGroups();
   properties_->ClearAllowedValues(MM::g_Keyword_CoreChannelGroup);
   properties_->AddAllowedValue(MM::g_Keyword_CoreChannelGroup, ""); // No channel group
   for (unsigned i=0; i<groups.size(); i++)
      properties_->AddAllowedValue(MM::g_Keyword_CoreChannelGroup, groups[i].c_str());

   // If we don't have the group assigned to ChannelGroup anymore, set ChannelGroup to blank.
   if (!isGroupDefined(getChannelGroup().c_str()))
      setChannelGroup("");
}

///////////////////////////////////////////////////////////////////////////////
//  Automatic device and serial port discovery methods
//

/**
 * Return whether or not the device supports automatic device detection
 * (i.e. whether or not detectDevice() may be safely called).
 *
 * For legacy reasons, an exception is not thrown if there is an error.
 * Instead, false is returned if label is not a valid device.
 */
bool CMMCore::supportsDeviceDetection(const char* label)
{
   try
   {
      std::shared_ptr<DeviceInstance> pDevice =
         deviceManager_->GetDevice(label);
      mm::DeviceModuleLockGuard guard(pDevice);
      return pDevice->SupportsDeviceDetection();
   }
   catch (const CMMError&)
   {
      return false;
   }
}

/**
 * Tries to communicate to a device through a given serial port
 * Used to automate discovery of correct serial port
 * Also configures the serial port correctly
 *
 * For legacy reasons, an exception is not thrown if there is an error.
 * Instead, MM::Unimplemented is returned if label is not a valid device.
 *
 * @param label  the label of the device for which the serial port should be found
 */
MM::DeviceDetectionStatus CMMCore::detectDevice(const char* label)
{
   try
   {
      CheckDeviceLabel(label);
   }
   catch (const CMMError&)
   {
      return MM::Unimplemented;
   }

   MM::DeviceDetectionStatus result = MM::Unimplemented;
   std::vector< std::string> propertiesToRestore;
   std::map< std::string, std::string> valuesToRestore;
   std::string port;

   try
   {
      std::shared_ptr<DeviceInstance> pDevice =
         deviceManager_->GetDevice(label);

      mm::DeviceModuleLockGuard guard(pDevice);
      try
      {
         port = pDevice->GetProperty(MM::g_Keyword_Port);
      }
      catch (const CMMError&)
      {
         // XXX BUG There was a comment here saying that we ignore errors
         // "if the port property does not exist", but the behavior is to
         // ignore any error in getting the _value_ of that property. I'm
         // keeping the behavior for now, since in practice the two are
         // usually equivalent, and fixing all the error handling in this
         // function would be more than a small project.
      }
      if (!port.empty())
      {
         // there is a valid serial port setting for this device, so
         // gather the properties that will be restored if we don't find the device

         propertiesToRestore.push_back(MM::g_Keyword_BaudRate);
         propertiesToRestore.push_back(MM::g_Keyword_DataBits);
         propertiesToRestore.push_back(MM::g_Keyword_StopBits);
         propertiesToRestore.push_back(MM::g_Keyword_Parity);
         propertiesToRestore.push_back(MM::g_Keyword_Handshaking);
         propertiesToRestore.push_back(MM::g_Keyword_AnswerTimeout);
         propertiesToRestore.push_back(MM::g_Keyword_DelayBetweenCharsMs);
         // record the current settings before running device detection.
         std::string previousValue;
         for( std::vector< std::string>::iterator sit = propertiesToRestore.begin(); sit!= propertiesToRestore.end(); ++sit)
         {
	    try
            {
               previousValue = getProperty(port.c_str(), (*sit).c_str());
               valuesToRestore[*sit] = std::string(previousValue);
	    }
            catch(...)
            {
               LOG_ERROR(coreLogger_) <<
                  "Device detection: error gathering property " << (*sit).c_str() <<
                  " of port " << port << " while testing for device " << label;
	    }
         }
      }

      // run device detection routine
      result = pDevice->DetectDevice();
   }
   catch(...)
   {
      LOG_ERROR(coreLogger_) << "Device detection: error testing ports " <<
         (port.empty() ? "none" : port) << " for device " << label;
   }

   // if the device is not there, restore the parameters to the original settings
   if ( MM::CanCommunicate != result)
   {
      for( std::vector< std::string>::iterator sit = propertiesToRestore.begin(); sit!= propertiesToRestore.end(); ++sit)
      {
         if (!port.empty())
         {
            try
            {
               setProperty(port.c_str(), (*sit).c_str(), (valuesToRestore[*sit]).c_str());
            }
            catch(...)
            {
               LOG_ERROR(coreLogger_) <<
                  "Device detection: error restoring port " << port <<
                  " state after testing for device " << label;
            }
         }
      }
   }

   return result;
}

/**
 * Performs auto-detection and loading of child devices that are attached to a Hub device.
 * For example, if a motorized microscope is represented by a Hub device, it is capable of
 * discovering what specific child devices are currently attached. In that case this call might
 * report that Z-stage, filter changer and objective turret are currently installed and return three
 * device names in the string list.
 *
 * Currently, this method can only be called once, right after loading the hub
 * device. Doing otherwise results in undefined behavior. This function was
 * intended for use during initial configuration, not routine loading of
 * devices. These restrictions may be relaxed in the future if possible.
 *
 * @param hubDeviceLabel    the label for the device of type Hub
 */
std::vector<std::string> CMMCore::getInstalledDevices(const char* hubDeviceLabel) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<HubInstance> pHub =
      deviceManager_->GetDeviceOfType<HubInstance>(hubDeviceLabel);

   mm::DeviceModuleLockGuard guard(pHub);
   return pHub->GetInstalledPeripheralNames();
}

std::vector<std::string> CMMCore::getLoadedPeripheralDevices(const char* hubLabel) MMCORE_LEGACY_THROW(CMMError)
{
   CheckDeviceLabel(hubLabel);
   return deviceManager_->GetLoadedPeripherals(hubLabel);
}

std::string CMMCore::getInstalledDeviceDescription(const char* hubLabel, const char* deviceLabel) MMCORE_LEGACY_THROW(CMMError)
{
   std::shared_ptr<HubInstance> pHub =
      deviceManager_->GetDeviceOfType<HubInstance>(hubLabel);
   CheckDeviceLabel(deviceLabel);

   std::string description;
   {
      mm::DeviceModuleLockGuard guard(pHub);
      description = pHub->GetInstalledPeripheralDescription(deviceLabel);
   }
   return description.empty() ? "N/A" : description;
}

/**
 * \brief Testing only: load a mock device adapter.
 * 
 * This function is designed for unit testing of MMCore itself, and its
 * interface is subject to change. It is also not designed for language
 * bindings (Java, Python) in mind (at least for now).
 * 
 * Do not use this in production code.
 * 
 * The caller is responsible for keeping \p implementation valid until this
 * Core is destroyed (or until unloadLibrary(name) is called, but that is
 * not recommended.)
 */
void CMMCore::loadMockDeviceAdapter(const char* name,
      MockDeviceAdapter* implementation) MMCORE_LEGACY_THROW(CMMError)
{
   if (!name)
      throw CMMError("Null device adapter name");
   pluginManager_->LoadMockAdapter(name, implementation);
}
