///////////////////////////////////////////////////////////////////////////////
// FILE:          MMCore.h
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
// NOTES:         Public methods follow a slightly different naming convention than
//                the rest of the C++ code, i.e we have:
//                   getConfiguration();
//                instead of:
//                   GetConfiguration();
//                The alternative (lowercase function names) convention is used
//                because public method names appear as wrapped methods in other
//                languages, in particular Java.

#pragma once

/*
 * Important! Read this before changing this file.
 *
 * Please see the version number and explanatory comment in the implementation
 * file (MMCore.cpp).
 */

#include "../MMDevice/DeviceThreads.h"
#include "../MMDevice/MMDevice.h"
#include "../MMDevice/MMDeviceConstants.h"
#include "Configuration.h"
#include "Error.h"
#include "ErrorCodes.h"
#include "Logging/Logger.h"
#include "MockDeviceAdapter.h"

#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <string>
#include <vector>


#if !defined(SWIGJAVA) && !defined(SWIGPYTHON)
#   ifdef _MSC_VER
#      define MMCORE_DEPRECATED(prototype) __declspec(deprecated) prototype
#   elif defined(__GNUC__)
#      define MMCORE_DEPRECATED(prototype) prototype __attribute__((deprecated))
#   else
#      define MMCORE_DEPRECATED(prototype) prototype
#   endif
#else
#   define MMCORE_DEPRECATED(prototype) prototype
#endif


class CPluginManager;
class CircularBuffer;
class ConfigGroupCollection;
class CoreCallback;
class CorePropertyCollection;
class MMEventCallback;
class Metadata;
class PixelSizeConfigGroup;

class AutoFocusInstance;
class CameraInstance;
class DeviceInstance;
class GalvoInstance;
class ImageProcessorInstance;
class SLMInstance;
class ShutterInstance;
class StageInstance;
class XYStageInstance;
class PressurePumpInstance;
class VolumetricPumpInstance;

class CMMCore;

namespace mm {
   class DeviceManager;
   class LogManager;
} // namespace mm

typedef unsigned int* imgRGB32;

enum DeviceInitializationState {
   Uninitialized,
   InitializedSuccessfully,
   InitializationFailed,
};


/// The Micro-Manager Core.
/**
 * Provides a device-independent interface for hardware control. Additionally,
 * provides some facilities (such as configuration groups) for application
 * programming.
 *
 * The signatures of most of the public member functions are designed to be
 * wrapped by SWIG with minimal manual configuration.
 */
class CMMCore
{
   friend class CoreCallback;
   friend class CorePropertyCollection;

public:
   CMMCore();
   ~CMMCore();

   /// A static method that does nothing.
   /**
    * This method can be called as a sanity check when dynamically loading the
    * Core library (e.g. through a foreign function interface for a high-level
    * language).
    */
   static void noop() {}

   /** \name Core feature control. */
   ///@{
   static void enableFeature(const char* name, bool enable) MMCORE_LEGACY_THROW(CMMError);
   static bool isFeatureEnabled(const char* name) MMCORE_LEGACY_THROW(CMMError);
   ///@}

   /** \name Initialization and setup. */
   ///@{
   void loadDevice(const char* label, const char* moduleName,
         const char* deviceName) MMCORE_LEGACY_THROW(CMMError);
   void unloadDevice(const char* label) MMCORE_LEGACY_THROW(CMMError);
   void unloadAllDevices() MMCORE_LEGACY_THROW(CMMError);
   void initializeAllDevices() MMCORE_LEGACY_THROW(CMMError);
   void initializeDevice(const char* label) MMCORE_LEGACY_THROW(CMMError);
   DeviceInitializationState getDeviceInitializationState(const char* label) const MMCORE_LEGACY_THROW(CMMError);
   void reset() MMCORE_LEGACY_THROW(CMMError);

   void unloadLibrary(const char* moduleName) MMCORE_LEGACY_THROW(CMMError);

   void updateCoreProperties() MMCORE_LEGACY_THROW(CMMError);

   std::string getCoreErrorText(int code) const;

   // Note that version functions need to be implemented in the .cpp file so
   // that they reflect the actual code being run (e.g., in a DLL), not the
   // header or language bindings layer (where applicable).

   // These two are not 'static' for backward compatibility (would break binary
   // compatibility of Java bindings).
   std::string getVersionInfo() const;
   std::string getAPIVersionInfo() const;

   static int getMMCoreVersionMajor();
   static int getMMCoreVersionMinor();
   static int getMMCoreVersionPatch();
   static int getMMDeviceModuleInterfaceVersion();
   static int getMMDeviceDeviceInterfaceVersion();

   Configuration getSystemState();
   void setSystemState(const Configuration& conf);
   Configuration getConfigState(const char* group, const char* config) MMCORE_LEGACY_THROW(CMMError);
   Configuration getConfigGroupState(const char* group) MMCORE_LEGACY_THROW(CMMError);
   void saveSystemState(const char* fileName) MMCORE_LEGACY_THROW(CMMError);
   void loadSystemState(const char* fileName) MMCORE_LEGACY_THROW(CMMError);
   void saveSystemConfiguration(const char* fileName) MMCORE_LEGACY_THROW(CMMError);
   void loadSystemConfiguration(const char* fileName) MMCORE_LEGACY_THROW(CMMError);
   void registerCallback(MMEventCallback* cb);
   ///@}

   /** \name Logging and log management. */
   ///@{
   void setPrimaryLogFile(const char* filename, bool truncate = false) MMCORE_LEGACY_THROW(CMMError);
   std::string getPrimaryLogFile() const;

   void logMessage(const char* msg);
   void logMessage(const char* msg, bool debugOnly);
   void enableDebugLog(bool enable);
   bool debugLogEnabled();
   void enableStderrLog(bool enable);
   bool stderrLogEnabled();

   int startSecondaryLogFile(const char* filename, bool enableDebug,
         bool truncate = true, bool synchronous = false) MMCORE_LEGACY_THROW(CMMError);
   void stopSecondaryLogFile(int handle) MMCORE_LEGACY_THROW(CMMError);

   ///@}

   /** \name Device listing. */
   ///@{
   std::vector<std::string> getDeviceAdapterSearchPaths();
   void setDeviceAdapterSearchPaths(const std::vector<std::string>& paths);

   std::vector<std::string> getDeviceAdapterNames() MMCORE_LEGACY_THROW(CMMError);

   std::vector<std::string> getAvailableDevices(const char* library) MMCORE_LEGACY_THROW(CMMError);
   std::vector<std::string> getAvailableDeviceDescriptions(const char* library) MMCORE_LEGACY_THROW(CMMError);
   std::vector<long> getAvailableDeviceTypes(const char* library) MMCORE_LEGACY_THROW(CMMError);
   ///@}

   /** \name Generic device control.
    *
    * Functionality supported by all devices.
    */
   ///@{
   std::vector<std::string> getLoadedDevices() const;
   std::vector<std::string> getLoadedDevicesOfType(MM::DeviceType devType) const;
   MM::DeviceType getDeviceType(const char* label) MMCORE_LEGACY_THROW(CMMError);
   std::string getDeviceLibrary(const char* label) MMCORE_LEGACY_THROW(CMMError);
   std::string getDeviceName(const char* label) MMCORE_LEGACY_THROW(CMMError);
   std::string getDeviceDescription(const char* label) MMCORE_LEGACY_THROW(CMMError);

   std::vector<std::string> getDevicePropertyNames(const char* label) MMCORE_LEGACY_THROW(CMMError);
   bool hasProperty(const char* label, const char* propName) MMCORE_LEGACY_THROW(CMMError);
   std::string getProperty(const char* label, const char* propName) MMCORE_LEGACY_THROW(CMMError);
   void setProperty(const char* label, const char* propName, const char* propValue) MMCORE_LEGACY_THROW(CMMError);
   void setProperty(const char* label, const char* propName, const bool propValue) MMCORE_LEGACY_THROW(CMMError);
   void setProperty(const char* label, const char* propName, const long propValue) MMCORE_LEGACY_THROW(CMMError);
   void setProperty(const char* label, const char* propName, const float propValue) MMCORE_LEGACY_THROW(CMMError);
   void setProperty(const char* label, const char* propName, const double propValue) MMCORE_LEGACY_THROW(CMMError);

   std::vector<std::string> getAllowedPropertyValues(const char* label, const char* propName) MMCORE_LEGACY_THROW(CMMError);
   bool isPropertyReadOnly(const char* label, const char* propName) MMCORE_LEGACY_THROW(CMMError);
   bool isPropertyPreInit(const char* label, const char* propName) MMCORE_LEGACY_THROW(CMMError);
   bool isPropertySequenceable(const char* label, const char* propName) MMCORE_LEGACY_THROW(CMMError);
   bool hasPropertyLimits(const char* label, const char* propName) MMCORE_LEGACY_THROW(CMMError);
   double getPropertyLowerLimit(const char* label, const char* propName) MMCORE_LEGACY_THROW(CMMError);
   double getPropertyUpperLimit(const char* label, const char* propName) MMCORE_LEGACY_THROW(CMMError);
   MM::PropertyType getPropertyType(const char* label, const char* propName) MMCORE_LEGACY_THROW(CMMError);

   void startPropertySequence(const char* label, const char* propName) MMCORE_LEGACY_THROW(CMMError);
   void stopPropertySequence(const char* label, const char* propName) MMCORE_LEGACY_THROW(CMMError);
   long getPropertySequenceMaxLength(const char* label, const char* propName) MMCORE_LEGACY_THROW(CMMError);
   void loadPropertySequence(const char* label, const char* propName, std::vector<std::string> eventSequence) MMCORE_LEGACY_THROW(CMMError);

   bool deviceBusy(const char* label) MMCORE_LEGACY_THROW(CMMError);
   void waitForDevice(const char* label) MMCORE_LEGACY_THROW(CMMError);
   void waitForConfig(const char* group, const char* configName) MMCORE_LEGACY_THROW(CMMError);
   bool systemBusy() MMCORE_LEGACY_THROW(CMMError);
   void waitForSystem() MMCORE_LEGACY_THROW(CMMError);
   bool deviceTypeBusy(MM::DeviceType devType) MMCORE_LEGACY_THROW(CMMError);
   void waitForDeviceType(MM::DeviceType devType) MMCORE_LEGACY_THROW(CMMError);

   double getDeviceDelayMs(const char* label) MMCORE_LEGACY_THROW(CMMError);
   void setDeviceDelayMs(const char* label, double delayMs) MMCORE_LEGACY_THROW(CMMError);
   bool usesDeviceDelay(const char* label) MMCORE_LEGACY_THROW(CMMError);

   void setTimeoutMs(long timeoutMs) {if (timeoutMs > 0) timeoutMs_ = timeoutMs;}
   long getTimeoutMs() { return timeoutMs_;}

   void sleep(double intervalMs) const;
   ///@}

   /** \name Management of 'current' device for specific roles. */
   ///@{
   std::string getCameraDevice();
   std::string getShutterDevice();
   std::string getFocusDevice();
   std::string getXYStageDevice();
   std::string getAutoFocusDevice();
   std::string getImageProcessorDevice();
   std::string getSLMDevice();
   std::string getGalvoDevice();
   std::string getChannelGroup();
   void setCameraDevice(const char* cameraLabel) MMCORE_LEGACY_THROW(CMMError);
   void setShutterDevice(const char* shutterLabel) MMCORE_LEGACY_THROW(CMMError);
   void setFocusDevice(const char* focusLabel) MMCORE_LEGACY_THROW(CMMError);
   void setXYStageDevice(const char* xyStageLabel) MMCORE_LEGACY_THROW(CMMError);
   void setAutoFocusDevice(const char* focusLabel) MMCORE_LEGACY_THROW(CMMError);
   void setImageProcessorDevice(const char* procLabel) MMCORE_LEGACY_THROW(CMMError);
   void setSLMDevice(const char* slmLabel) MMCORE_LEGACY_THROW(CMMError);
   void setGalvoDevice(const char* galvoLabel) MMCORE_LEGACY_THROW(CMMError);
   void setChannelGroup(const char* channelGroup) MMCORE_LEGACY_THROW(CMMError);
   ///@}

   /** \name System state cache.
    *
    * The system state cache retains the last-set or last-read value of each
    * device property.
    */
   ///@{
   Configuration getSystemStateCache() const;
   void updateSystemStateCache();
   std::string getPropertyFromCache(const char* deviceLabel,
         const char* propName) const MMCORE_LEGACY_THROW(CMMError);
   std::string getCurrentConfigFromCache(const char* groupName) MMCORE_LEGACY_THROW(CMMError);
   Configuration getConfigGroupStateFromCache(const char* group) MMCORE_LEGACY_THROW(CMMError);
   ///@}

   /** \name Configuration groups. */
   ///@{
   void defineConfig(const char* groupName, const char* configName) MMCORE_LEGACY_THROW(CMMError);
   void defineConfig(const char* groupName, const char* configName,
         const char* deviceLabel, const char* propName,
         const char* value) MMCORE_LEGACY_THROW(CMMError);
   void defineConfigGroup(const char* groupName) MMCORE_LEGACY_THROW(CMMError);
   void deleteConfigGroup(const char* groupName) MMCORE_LEGACY_THROW(CMMError);
   void renameConfigGroup(const char* oldGroupName,
         const char* newGroupName) MMCORE_LEGACY_THROW(CMMError);
   bool isGroupDefined(const char* groupName);
   bool isConfigDefined(const char* groupName, const char* configName);
   void setConfig(const char* groupName, const char* configName) MMCORE_LEGACY_THROW(CMMError);
   void deleteConfig(const char* groupName, const char* configName) MMCORE_LEGACY_THROW(CMMError);
   void deleteConfig(const char* groupName, const char* configName,
         const char* deviceLabel, const char* propName) MMCORE_LEGACY_THROW(CMMError);
   void renameConfig(const char* groupName, const char* oldConfigName,
         const char* newConfigName) MMCORE_LEGACY_THROW(CMMError);
   std::vector<std::string> getAvailableConfigGroups() const;
   std::vector<std::string> getAvailableConfigs(const char* configGroup) const;
   std::string getCurrentConfig(const char* groupName) MMCORE_LEGACY_THROW(CMMError);
   Configuration getConfigData(const char* configGroup,
         const char* configName) MMCORE_LEGACY_THROW(CMMError);
   ///@}

   /** \name The pixel size configuration group. */
   ///@{
   std::string getCurrentPixelSizeConfig() MMCORE_LEGACY_THROW(CMMError);
   std::string getCurrentPixelSizeConfig(bool cached) MMCORE_LEGACY_THROW(CMMError);
   double getPixelSizeUm();
   double getPixelSizeUm(bool cached);
   double getPixelSizeUmByID(const char* resolutionID) MMCORE_LEGACY_THROW(CMMError);
   std::vector<double> getPixelSizeAffine() MMCORE_LEGACY_THROW(CMMError);
   std::vector<double> getPixelSizeAffine(bool cached) MMCORE_LEGACY_THROW(CMMError);
   std::vector<double> getPixelSizeAffineByID(const char* resolutionID) MMCORE_LEGACY_THROW(CMMError);
   double getPixelSizedxdz() MMCORE_LEGACY_THROW(CMMError);
   double getPixelSizedxdz(bool cached) MMCORE_LEGACY_THROW(CMMError);
   double getPixelSizedxdz(const char* resolutionID) MMCORE_LEGACY_THROW(CMMError);
   double getPixelSizedydz() MMCORE_LEGACY_THROW(CMMError);
   double getPixelSizedydz(bool cached) MMCORE_LEGACY_THROW(CMMError);
   double getPixelSizedydz(const char* resolutionID) MMCORE_LEGACY_THROW(CMMError);
   double getPixelSizeOptimalZUm() MMCORE_LEGACY_THROW(CMMError);
   double getPixelSizeOptimalZUm(bool cached) MMCORE_LEGACY_THROW(CMMError);
   double getPixelSizeOptimalZUm(const char* resolutionID) MMCORE_LEGACY_THROW(CMMError);
   double getMagnificationFactor() const;
   void setPixelSizeUm(const char* resolutionID, double pixSize)  MMCORE_LEGACY_THROW(CMMError);
   void setPixelSizeAffine(const char* resolutionID, std::vector<double> affine)  MMCORE_LEGACY_THROW(CMMError);
   void setPixelSizedxdz(const char* resolutionID, double dXdZ)  MMCORE_LEGACY_THROW(CMMError);
   void setPixelSizedydz(const char* resolutionID, double dYdZ)  MMCORE_LEGACY_THROW(CMMError);
   void setPixelSizeOptimalZUm(const char* resolutionID, double optimalZ)  MMCORE_LEGACY_THROW(CMMError);
   void definePixelSizeConfig(const char* resolutionID,
         const char* deviceLabel, const char* propName,
         const char* value) MMCORE_LEGACY_THROW(CMMError);
   void definePixelSizeConfig(const char* resolutionID) MMCORE_LEGACY_THROW(CMMError);
   std::vector<std::string> getAvailablePixelSizeConfigs() const;
   bool isPixelSizeConfigDefined(const char* resolutionID) const MMCORE_LEGACY_THROW(CMMError);
   void setPixelSizeConfig(const char* resolutionID) MMCORE_LEGACY_THROW(CMMError);
   void renamePixelSizeConfig(const char* oldConfigName,
         const char* newConfigName) MMCORE_LEGACY_THROW(CMMError);
   void deletePixelSizeConfig(const char* configName) MMCORE_LEGACY_THROW(CMMError);
   Configuration getPixelSizeConfigData(const char* configName) MMCORE_LEGACY_THROW(CMMError);
   ///@}

   /** \name Image acquisition. */
   ///@{
   void setROI(int x, int y, int xSize, int ySize) MMCORE_LEGACY_THROW(CMMError);
   void setROI(const char* label, int x, int y, int xSize, int ySize) MMCORE_LEGACY_THROW(CMMError);
   void getROI(int& x, int& y, int& xSize, int& ySize) MMCORE_LEGACY_THROW(CMMError);
   void getROI(const char* label, int& x, int& y, int& xSize, int& ySize) MMCORE_LEGACY_THROW(CMMError);
   void clearROI() MMCORE_LEGACY_THROW(CMMError);

   bool isMultiROISupported() MMCORE_LEGACY_THROW(CMMError);
   bool isMultiROIEnabled() MMCORE_LEGACY_THROW(CMMError);
   void setMultiROI(std::vector<unsigned> xs, std::vector<unsigned> ys,
           std::vector<unsigned> widths,
           std::vector<unsigned> heights) MMCORE_LEGACY_THROW(CMMError);
   void getMultiROI(std::vector<unsigned>& xs, std::vector<unsigned>& ys,
           std::vector<unsigned>& widths,
           std::vector<unsigned>& heights) MMCORE_LEGACY_THROW(CMMError);

   void setExposure(double exp) MMCORE_LEGACY_THROW(CMMError);
   void setExposure(const char* cameraLabel, double dExp) MMCORE_LEGACY_THROW(CMMError);
   double getExposure() MMCORE_LEGACY_THROW(CMMError);
   double getExposure(const char* label) MMCORE_LEGACY_THROW(CMMError);

   void snapImage() MMCORE_LEGACY_THROW(CMMError);
   void* getImage() MMCORE_LEGACY_THROW(CMMError);
   void* getImage(unsigned numChannel) MMCORE_LEGACY_THROW(CMMError);

   unsigned getImageWidth();
   unsigned getImageHeight();
   unsigned getBytesPerPixel();
   unsigned getImageBitDepth();
   unsigned getNumberOfComponents();
   unsigned getNumberOfCameraChannels();
   std::string getCameraChannelName(unsigned int channelNr);
   long getImageBufferSize();

   void setAutoShutter(bool state);
   bool getAutoShutter();
   void setShutterOpen(bool state) MMCORE_LEGACY_THROW(CMMError);
   bool getShutterOpen() MMCORE_LEGACY_THROW(CMMError);
   void setShutterOpen(const char* shutterLabel, bool state) MMCORE_LEGACY_THROW(CMMError);
   bool getShutterOpen(const char* shutterLabel) MMCORE_LEGACY_THROW(CMMError);

   void startSequenceAcquisition(long numImages, double intervalMs,
         bool stopOnOverflow) MMCORE_LEGACY_THROW(CMMError);
   void startSequenceAcquisition(const char* cameraLabel, long numImages,
         double intervalMs, bool stopOnOverflow) MMCORE_LEGACY_THROW(CMMError);
   void prepareSequenceAcquisition(const char* cameraLabel) MMCORE_LEGACY_THROW(CMMError);
   void startContinuousSequenceAcquisition(double intervalMs) MMCORE_LEGACY_THROW(CMMError);
   void stopSequenceAcquisition() MMCORE_LEGACY_THROW(CMMError);
   void stopSequenceAcquisition(const char* cameraLabel) MMCORE_LEGACY_THROW(CMMError);
   bool isSequenceRunning() MMCORE_NOEXCEPT;
   bool isSequenceRunning(const char* cameraLabel) MMCORE_LEGACY_THROW(CMMError);

   void* getLastImage() MMCORE_LEGACY_THROW(CMMError);
   void* popNextImage() MMCORE_LEGACY_THROW(CMMError);
   void* getLastImageMD(unsigned channel, unsigned slice, Metadata& md)
      const MMCORE_LEGACY_THROW(CMMError);
   void* popNextImageMD(unsigned channel, unsigned slice, Metadata& md)
      MMCORE_LEGACY_THROW(CMMError);
   void* getLastImageMD(Metadata& md) const MMCORE_LEGACY_THROW(CMMError);
   void* getNBeforeLastImageMD(unsigned long n, Metadata& md)
      const MMCORE_LEGACY_THROW(CMMError);
   void* popNextImageMD(Metadata& md) MMCORE_LEGACY_THROW(CMMError);

   long getRemainingImageCount();
   long getBufferTotalCapacity();
   long getBufferFreeCapacity();
   bool isBufferOverflowed() const;
   void setCircularBufferMemoryFootprint(unsigned sizeMB) MMCORE_LEGACY_THROW(CMMError);
   unsigned getCircularBufferMemoryFootprint();
   void initializeCircularBuffer() MMCORE_LEGACY_THROW(CMMError);
   void clearCircularBuffer() MMCORE_LEGACY_THROW(CMMError);

   bool isExposureSequenceable(const char* cameraLabel) MMCORE_LEGACY_THROW(CMMError);
   void startExposureSequence(const char* cameraLabel) MMCORE_LEGACY_THROW(CMMError);
   void stopExposureSequence(const char* cameraLabel) MMCORE_LEGACY_THROW(CMMError);
   long getExposureSequenceMaxLength(const char* cameraLabel) MMCORE_LEGACY_THROW(CMMError);
   void loadExposureSequence(const char* cameraLabel,
         std::vector<double> exposureSequence_ms) MMCORE_LEGACY_THROW(CMMError);
   ///@}

   /** \name Autofocus control. */
   ///@{
   double getLastFocusScore();
   double getCurrentFocusScore();
   void enableContinuousFocus(bool enable) MMCORE_LEGACY_THROW(CMMError);
   bool isContinuousFocusEnabled() MMCORE_LEGACY_THROW(CMMError);
   bool isContinuousFocusLocked() MMCORE_LEGACY_THROW(CMMError);
   bool isContinuousFocusDrive(const char* stageLabel) MMCORE_LEGACY_THROW(CMMError);
   void fullFocus() MMCORE_LEGACY_THROW(CMMError);
   void incrementalFocus() MMCORE_LEGACY_THROW(CMMError);
   void setAutoFocusOffset(double offset) MMCORE_LEGACY_THROW(CMMError);
   double getAutoFocusOffset() MMCORE_LEGACY_THROW(CMMError);
   ///@}

   /** \name State device control. */
   ///@{
   void setState(const char* stateDeviceLabel, long state) MMCORE_LEGACY_THROW(CMMError);
   long getState(const char* stateDeviceLabel) MMCORE_LEGACY_THROW(CMMError);
   long getNumberOfStates(const char* stateDeviceLabel);
   void setStateLabel(const char* stateDeviceLabel,
         const char* stateLabel) MMCORE_LEGACY_THROW(CMMError);
   std::string getStateLabel(const char* stateDeviceLabel) MMCORE_LEGACY_THROW(CMMError);
   void defineStateLabel(const char* stateDeviceLabel,
         long state, const char* stateLabel) MMCORE_LEGACY_THROW(CMMError);
   std::vector<std::string> getStateLabels(const char* stateDeviceLabel)
      MMCORE_LEGACY_THROW(CMMError);
   long getStateFromLabel(const char* stateDeviceLabel,
         const char* stateLabel) MMCORE_LEGACY_THROW(CMMError);
   ///@}

   /** \name Focus (Z) stage control. */
   ///@{
   void setPosition(const char* stageLabel, double position) MMCORE_LEGACY_THROW(CMMError);
   void setPosition(double position) MMCORE_LEGACY_THROW(CMMError);
   double getPosition(const char* stageLabel) MMCORE_LEGACY_THROW(CMMError);
   double getPosition() MMCORE_LEGACY_THROW(CMMError);
   void setRelativePosition(const char* stageLabel, double d) MMCORE_LEGACY_THROW(CMMError);
   void setRelativePosition(double d) MMCORE_LEGACY_THROW(CMMError);
   void setOrigin(const char* stageLabel) MMCORE_LEGACY_THROW(CMMError);
   void setOrigin() MMCORE_LEGACY_THROW(CMMError);
   void setAdapterOrigin(const char* stageLabel, double newZUm) MMCORE_LEGACY_THROW(CMMError);
   void setAdapterOrigin(double newZUm) MMCORE_LEGACY_THROW(CMMError);

   void setFocusDirection(const char* stageLabel, int sign);
   int getFocusDirection(const char* stageLabel) MMCORE_LEGACY_THROW(CMMError);

   bool isStageSequenceable(const char* stageLabel) MMCORE_LEGACY_THROW(CMMError);
   bool isStageLinearSequenceable(const char* stageLabel) MMCORE_LEGACY_THROW(CMMError);
   void startStageSequence(const char* stageLabel) MMCORE_LEGACY_THROW(CMMError);
   void stopStageSequence(const char* stageLabel) MMCORE_LEGACY_THROW(CMMError);
   long getStageSequenceMaxLength(const char* stageLabel) MMCORE_LEGACY_THROW(CMMError);
   void loadStageSequence(const char* stageLabel,
         std::vector<double> positionSequence) MMCORE_LEGACY_THROW(CMMError);
   void setStageLinearSequence(const char* stageLabel, double dZ_um, int nSlices) MMCORE_LEGACY_THROW(CMMError);
   ///@}

   /** \name XY stage control. */
   ///@{
   void setXYPosition(const char* xyStageLabel,
         double x, double y) MMCORE_LEGACY_THROW(CMMError);
   void setXYPosition(double x, double y) MMCORE_LEGACY_THROW(CMMError);
   void setRelativeXYPosition(const char* xyStageLabel,
         double dx, double dy) MMCORE_LEGACY_THROW(CMMError);
   void setRelativeXYPosition(double dx, double dy) MMCORE_LEGACY_THROW(CMMError);
   void getXYPosition(const char* xyStageLabel,
         double &x_stage, double &y_stage) MMCORE_LEGACY_THROW(CMMError);
   void getXYPosition(double &x_stage, double &y_stage) MMCORE_LEGACY_THROW(CMMError);
   double getXPosition(const char* xyStageLabel) MMCORE_LEGACY_THROW(CMMError);
   double getYPosition(const char* xyStageLabel) MMCORE_LEGACY_THROW(CMMError);
   double getXPosition() MMCORE_LEGACY_THROW(CMMError);
   double getYPosition() MMCORE_LEGACY_THROW(CMMError);
   void stop(const char* xyOrZStageLabel) MMCORE_LEGACY_THROW(CMMError);
   void home(const char* xyOrZStageLabel) MMCORE_LEGACY_THROW(CMMError);
   void setOriginXY(const char* xyStageLabel) MMCORE_LEGACY_THROW(CMMError);
   void setOriginXY() MMCORE_LEGACY_THROW(CMMError);
   void setOriginX(const char* xyStageLabel) MMCORE_LEGACY_THROW(CMMError);
   void setOriginX() MMCORE_LEGACY_THROW(CMMError);
   void setOriginY(const char* xyStageLabel) MMCORE_LEGACY_THROW(CMMError);
   void setOriginY() MMCORE_LEGACY_THROW(CMMError);
   void setAdapterOriginXY(const char* xyStageLabel,
         double newXUm, double newYUm) MMCORE_LEGACY_THROW(CMMError);
   void setAdapterOriginXY(double newXUm, double newYUm) MMCORE_LEGACY_THROW(CMMError);

   bool isXYStageSequenceable(const char* xyStageLabel) MMCORE_LEGACY_THROW(CMMError);
   void startXYStageSequence(const char* xyStageLabel) MMCORE_LEGACY_THROW(CMMError);
   void stopXYStageSequence(const char* xyStageLabel) MMCORE_LEGACY_THROW(CMMError);
   long getXYStageSequenceMaxLength(const char* xyStageLabel) MMCORE_LEGACY_THROW(CMMError);
   void loadXYStageSequence(const char* xyStageLabel,
         std::vector<double> xSequence,
         std::vector<double> ySequence) MMCORE_LEGACY_THROW(CMMError);
   ///@}

   /** \name Serial port control. */
   ///@{
   void setSerialProperties(const char* portName,
      const char* answerTimeout,
      const char* baudRate,
      const char* delayBetweenCharsMs,
      const char* handshaking,
      const char* parity,
      const char* stopBits) MMCORE_LEGACY_THROW(CMMError);

   void setSerialPortCommand(const char* portLabel, const char* command,
         const char* term) MMCORE_LEGACY_THROW(CMMError);
   std::string getSerialPortAnswer(const char* portLabel,
         const char* term) MMCORE_LEGACY_THROW(CMMError);
   void writeToSerialPort(const char* portLabel,
         const std::vector<char> &data) MMCORE_LEGACY_THROW(CMMError);
   std::vector<char> readFromSerialPort(const char* portLabel)
      MMCORE_LEGACY_THROW(CMMError);
   ///@}

   /** \name SLM control.
    *
    * Control of spatial light modulators such as liquid crystal on silicon
    * (LCoS), digital micromirror devices (DMD), or multimedia projectors.
    */
   ///@{
   void setSLMImage(const char* slmLabel,
         unsigned char * pixels) MMCORE_LEGACY_THROW(CMMError);
   void setSLMImage(const char* slmLabel, imgRGB32 pixels) MMCORE_LEGACY_THROW(CMMError);
   void setSLMPixelsTo(const char* slmLabel,
         unsigned char intensity) MMCORE_LEGACY_THROW(CMMError);
   void setSLMPixelsTo(const char* slmLabel,
         unsigned char red, unsigned char green,
         unsigned char blue) MMCORE_LEGACY_THROW(CMMError);
   void displaySLMImage(const char* slmLabel) MMCORE_LEGACY_THROW(CMMError);
   void setSLMExposure(const char* slmLabel, double exposure_ms)
      MMCORE_LEGACY_THROW(CMMError);
   double getSLMExposure(const char* slmLabel) MMCORE_LEGACY_THROW(CMMError);
   unsigned getSLMWidth(const char* slmLabel) MMCORE_LEGACY_THROW(CMMError);
   unsigned getSLMHeight(const char* slmLabel) MMCORE_LEGACY_THROW(CMMError);
   unsigned getSLMNumberOfComponents(const char* slmLabel) MMCORE_LEGACY_THROW(CMMError);
   unsigned getSLMBytesPerPixel(const char* slmLabel) MMCORE_LEGACY_THROW(CMMError);

   long getSLMSequenceMaxLength(const char* slmLabel) MMCORE_LEGACY_THROW(CMMError);
   void startSLMSequence(const char* slmLabel) MMCORE_LEGACY_THROW(CMMError);
   void stopSLMSequence(const char* slmLabel) MMCORE_LEGACY_THROW(CMMError);
   void loadSLMSequence(const char* slmLabel,
         std::vector<unsigned char*> imageSequence) MMCORE_LEGACY_THROW(CMMError);
   ///@}

   /** \name Galvo control.
    *
    * Control of beam-steering devices.
    */
   ///@{
   void pointGalvoAndFire(const char* galvoLabel, double x, double y,
         double pulseTime_us) MMCORE_LEGACY_THROW(CMMError);
   void setGalvoSpotInterval(const char* galvoLabel,
         double pulseTime_us) MMCORE_LEGACY_THROW(CMMError);
   void setGalvoPosition(const char* galvoLabel, double x, double y)
      MMCORE_LEGACY_THROW(CMMError);
   void getGalvoPosition(const char* galvoLabel,
         double &x_stage, double &y_stage) MMCORE_LEGACY_THROW(CMMError); // using x_stage to get swig to work
   void setGalvoIlluminationState(const char* galvoLabel, bool on)
      MMCORE_LEGACY_THROW(CMMError);
   double getGalvoXRange(const char* galvoLabel) MMCORE_LEGACY_THROW(CMMError);
   double getGalvoXMinimum(const char* galvoLabel) MMCORE_LEGACY_THROW(CMMError);
   double getGalvoYRange(const char* galvoLabel) MMCORE_LEGACY_THROW(CMMError);
   double getGalvoYMinimum(const char* galvoLabel) MMCORE_LEGACY_THROW(CMMError);
   void addGalvoPolygonVertex(const char* galvoLabel, int polygonIndex,
         double x, double y) MMCORE_LEGACY_THROW(CMMError);
   void deleteGalvoPolygons(const char* galvoLabel) MMCORE_LEGACY_THROW(CMMError);
   void loadGalvoPolygons(const char* galvoLabel) MMCORE_LEGACY_THROW(CMMError);
   void setGalvoPolygonRepetitions(const char* galvoLabel, int repetitions)
      MMCORE_LEGACY_THROW(CMMError);
   void runGalvoPolygons(const char* galvoLabel) MMCORE_LEGACY_THROW(CMMError);
   void runGalvoSequence(const char* galvoLabel) MMCORE_LEGACY_THROW(CMMError);
   std::string getGalvoChannel(const char* galvoLabel) MMCORE_LEGACY_THROW(CMMError);
   ///@}

   /** \name PressurePump control
   *
   * Control of pressure pumps
   */
   ///@{
   void pressurePumpStop(const char* pumpLabel) MMCORE_LEGACY_THROW(CMMError);
   void pressurePumpCalibrate(const char* pumpLabel) MMCORE_LEGACY_THROW(CMMError);
   bool pressurePumpRequiresCalibration(const char* pumpLabel) MMCORE_LEGACY_THROW(CMMError);
   void setPumpPressureKPa(const char* pumplabel, double pressure) MMCORE_LEGACY_THROW(CMMError);
   double getPumpPressureKPa(const char* pumplabel) MMCORE_LEGACY_THROW(CMMError);
   ///@}

   /** \name VolumetricPump control
   *
   * Control of volumetric pumps
   */
   ///@{
   void volumetricPumpStop(const char* pumpLabel) MMCORE_LEGACY_THROW(CMMError);
   void volumetricPumpHome(const char* pumpLabel) MMCORE_LEGACY_THROW(CMMError);
   bool volumetricPumpRequiresHoming(const char* pumpLabel) MMCORE_LEGACY_THROW(CMMError);
   void invertPumpDirection(const char* pumpLabel, bool invert) MMCORE_LEGACY_THROW(CMMError);
   bool isPumpDirectionInverted(const char* pumpLabel) MMCORE_LEGACY_THROW(CMMError);
   void setPumpVolume(const char* pumpLabel, double volume) MMCORE_LEGACY_THROW(CMMError);
   double getPumpVolume(const char* pumpLabel) MMCORE_LEGACY_THROW(CMMError);
   void setPumpMaxVolume(const char* pumpLabel, double volume) MMCORE_LEGACY_THROW(CMMError);
   double getPumpMaxVolume(const char* pumpLabel) MMCORE_LEGACY_THROW(CMMError);
   void setPumpFlowrate(const char* pumpLabel, double volume) MMCORE_LEGACY_THROW(CMMError);
   double getPumpFlowrate(const char* pumpLabel) MMCORE_LEGACY_THROW(CMMError);
   void pumpStart(const char* pumpLabel) MMCORE_LEGACY_THROW(CMMError);
   void pumpDispenseDurationSeconds(const char* pumpLabel, double seconds) MMCORE_LEGACY_THROW(CMMError);
   void pumpDispenseVolumeUl(const char* pumpLabel, double microLiter) MMCORE_LEGACY_THROW(CMMError);
   ///@}

   /** \name Device discovery. */
   ///@{
   bool supportsDeviceDetection(const char* deviceLabel);
   MM::DeviceDetectionStatus detectDevice(const char* deviceLabel);
   ///@}

   /** \name Hub and peripheral devices. */
   ///@{
   std::string getParentLabel(const char* peripheralLabel) MMCORE_LEGACY_THROW(CMMError);
   void setParentLabel(const char* deviceLabel,
         const char* parentHubLabel) MMCORE_LEGACY_THROW(CMMError);

   std::vector<std::string> getInstalledDevices(const char* hubLabel) MMCORE_LEGACY_THROW(CMMError);
   std::string getInstalledDeviceDescription(const char* hubLabel,
         const char* peripheralLabel) MMCORE_LEGACY_THROW(CMMError);
   std::vector<std::string> getLoadedPeripheralDevices(const char* hubLabel) MMCORE_LEGACY_THROW(CMMError);
   ///@}

#if !defined(SWIGJAVA) && !defined(SWIGPYTHON)
   /** \name Testing */
   ///@{
   void loadMockDeviceAdapter(const char* name,
         MockDeviceAdapter* implementation) MMCORE_LEGACY_THROW(CMMError);
   ///@}
#endif

private:
   // make object non-copyable
   CMMCore(const CMMCore&);
   CMMCore& operator=(const CMMCore&);

private:
   // LogManager should be the first data member, so that it is available for
   // as long as possible during construction and (especially) destruction.
   std::shared_ptr<mm::LogManager> logManager_;
   mm::logging::Logger appLogger_;
   mm::logging::Logger coreLogger_;

   bool everSnapped_;

   std::weak_ptr<CameraInstance> currentCameraDevice_;
   std::weak_ptr<ShutterInstance> currentShutterDevice_;
   std::weak_ptr<StageInstance> currentFocusDevice_;
   std::weak_ptr<XYStageInstance> currentXYStageDevice_;
   std::weak_ptr<AutoFocusInstance> currentAutofocusDevice_;
   std::weak_ptr<SLMInstance> currentSLMDevice_;
   std::weak_ptr<GalvoInstance> currentGalvoDevice_;
   std::weak_ptr<ImageProcessorInstance> currentImageProcessor_;

   std::string channelGroup_;
   long pollingIntervalMs_;
   long timeoutMs_;
   bool autoShutter_;
   std::vector<double> *nullAffine_;
   MM::Core* callback_;                 // core services for devices
   ConfigGroupCollection* configGroups_;
   CorePropertyCollection* properties_;
   MMEventCallback* externalCallback_;  // notification hook to the higher layer (e.g. GUI)
   PixelSizeConfigGroup* pixelSizeGroup_;
   CircularBuffer* cbuf_;

   std::shared_ptr<CPluginManager> pluginManager_;
   std::shared_ptr<mm::DeviceManager> deviceManager_;
   std::map<int, std::string> errorText_;

   // Must be unlocked when calling MMEventCallback or calling device methods
   // or acquiring a module lock
   mutable MMThreadLock stateCacheLock_;
   mutable Configuration stateCache_; // Synchronized by stateCacheLock_

   MMThreadLock* pPostedErrorsLock_;
   mutable std::deque<std::pair< int, std::string> > postedErrors_;

   // True while interpreting the config file (but not while rolling back on
   // failure):
   bool isLoadingSystemConfiguration_ = false;

private:
   void InitializeErrorMessages();
   void CreateCoreProperties();

   // Parameter/value validation
   static void CheckDeviceLabel(const char* label) MMCORE_LEGACY_THROW(CMMError);
   static void CheckPropertyName(const char* propName) MMCORE_LEGACY_THROW(CMMError);
   static void CheckPropertyValue(const char* propValue) MMCORE_LEGACY_THROW(CMMError);
   static void CheckStateLabel(const char* stateLabel) MMCORE_LEGACY_THROW(CMMError);
   static void CheckConfigGroupName(const char* groupName) MMCORE_LEGACY_THROW(CMMError);
   static void CheckConfigPresetName(const char* presetName) MMCORE_LEGACY_THROW(CMMError);
   bool IsCoreDeviceLabel(const char* label) const MMCORE_LEGACY_THROW(CMMError);

   void applyConfiguration(const Configuration& config) MMCORE_LEGACY_THROW(CMMError);
   int applyProperties(std::vector<PropertySetting>& props, std::string& lastError);
   void waitForDevice(std::shared_ptr<DeviceInstance> pDev) MMCORE_LEGACY_THROW(CMMError);
   Configuration getConfigGroupState(const char* group, bool fromCache) MMCORE_LEGACY_THROW(CMMError);
   std::string getDeviceErrorText(int deviceCode, std::shared_ptr<DeviceInstance> pDevice);
   std::string getDeviceName(std::shared_ptr<DeviceInstance> pDev);
   void logError(const char* device, const char* msg);
   void updateAllowedChannelGroups();
   void assignDefaultRole(std::shared_ptr<DeviceInstance> pDev);
   void removeDeviceRole(std::shared_ptr<DeviceInstance> pDev);
   void removeAllDeviceRoles();
   void updateCoreProperty(const char* propName, MM::DeviceType devType) MMCORE_LEGACY_THROW(CMMError);
   void loadSystemConfigurationImpl(const char* fileName) MMCORE_LEGACY_THROW(CMMError);
   void initializeAllDevicesSerial() MMCORE_LEGACY_THROW(CMMError);
   void initializeAllDevicesParallel() MMCORE_LEGACY_THROW(CMMError);
   int initializeVectorOfDevices(std::vector<std::pair<std::shared_ptr<DeviceInstance>, std::string> > pDevices);
};
