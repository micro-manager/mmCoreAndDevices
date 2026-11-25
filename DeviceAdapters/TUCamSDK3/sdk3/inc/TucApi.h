/**
 * @copyright Copyright (C) Tucsen Photonics Co., Ltd. 2012-2025. All rights reserved.
 * @brief Tucsen Photonics SDK API Header File - Comprehensive Camera Control Interface
 *
 * This file contains the complete API function declarations for Tucsen Photonics SDK v3.0,
 * providing a comprehensive interface for camera device management, image acquisition,
 * and advanced processing capabilities. The API is organized into logical functional
 * categories for optimal developer experience and maintainability.
 *
 * The SDK supports multiple camera interfaces including USB, GigE, Camera Link, and CoaXPress,
 * with full GenICam compliance and advanced features such as real-time image processing,
 * video recording, and cross-platform compatibility.
 *
 * For more information, please visit: https://www.tucsen.com/
 *
 * API Organization (14 Functional Categories):
 * 1. SDK Lifecycle Management - Initialization, version info, and termination
 * 2. Device Discovery & Management - Enumeration, information, and connection control
 * 3. Feature Parameter Management - Feature discovery and metadata access
 * 4. Feature Value Access (Numeric) - Integer and floating-point parameter control
 * 5. Feature Value Access (Boolean/String) - Boolean and string parameter control
 * 6. Feature Value Access (Enumeration) - Enumeration parameter control and information
 * 7. Image Acquisition Control - Start/stop acquisition and statistics
 * 8. Buffer Management - Buffer pool and individual buffer operations
 * 9. Buffer Advanced Operations - Data processing, conversion, and file I/O
 * 10. Event Handling - Callback registration and event management
 * 11. Video Recording - Real-time video capture and encoding
 * 12. Logging & Profile Management - Debug logging and device configuration profiles
 * 13. Image Drawing - Real-time image display and rendering
 * 14. CPU Performance Optimization - High-speed data acquisition CPU optimization
 *
 * Key Features:
 * - Thread-safe design with multi-threaded concurrent access support
 * - Cross-platform compatibility (Windows, Linux)
 * - Multi-language binding support (C, C++, C#, Python)
 * - Real-time image acquisition with configurable buffer management
 * - Advanced ISP (Image Signal Processing) pipeline integration
 * - Comprehensive error handling and status reporting
 * - Flexible callback system for asynchronous operations
 * - Professional video recording with multiple codec support
 * - Device profile management for configuration persistence
 * - Built-in logging system for debugging and monitoring
 *
 * @file TucApi.h
 * @author zhang ren
 * @date 2025-07-01
 * @version v3.0.0
 * @note This API is thread-safe and supports multi-threaded concurrent access
 * @warning Must call TucInitialize() before using any other API functions
 * @see TucDefine.h
 * @since v1.0.0
 */

#ifndef TUC_API_H
#define TUC_API_H

#ifdef __cplusplus
extern "C"
{
#endif

/* Include necessary headers */
#include "TucDefine.h"
#include "TucSDKExports.h"

/* Parameter direction qualifiers */
#ifndef IN
#define IN
#endif

#ifndef OUT
#define OUT
#endif

#ifndef INOUT
#define INOUT
#endif

    // =============================================================================
    // 1. SDK Lifecycle Management - Initialization, version info, and termination
    // =============================================================================

    /**
     * @brief Initialize Tucsen SDK environment and core components
     *
     * This function initializes the Tucsen SDK environment, including core components such as
     * device management, logging system, GenTL wrapper management, and internal data structures.
     * It must be called before using any other Tucsen SDK functions to ensure proper SDK operation.
     *
     * The initialization process includes:
     * - Setting up the logging system based on the specified log level
     * - Setting up internal data structures and memory management
     * - Initializing the logging system for debugging and monitoring
     * - Loading and validating GenTL wrapper libraries
     * - Preparing device enumeration and management capabilities
     * - Setting up default configuration paths and parameters
     *
     * @param[in] logLevel Log level, see TUC_LOG_LEVEL enumeration
     * @return TUC_STATUS_OK on success, error code on failure
     *
     * @note This function is thread-safe and can be called multiple times safely
     * @warning Must be called before any other Tucsen SDK  functions
     * @see TucTerminate(), TucDeviceEnumerate()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucInitialize(IN TUC_LOG_LEVEL logLevel);

    /**
     * @brief Terminate Tucsen SDK environment and cleanup resources
     *
     * This function terminates the Tucsen SDK environment and performs comprehensive cleanup
     * of all allocated resources. It should be called when the Tucsen SDK is no longer needed
     * to ensure proper resource deallocation and prevent memory leaks.
     *
     * The termination process includes:
     * - Closing all open devices and releasing device handles
     * - Clearing device management data structures
     * - Resetting internal state variables (initialization flag, device count, profile path)
     * - Cleaning up GenTL wrapper libraries and resources
     * - Flushing and closing logging system
     * - Releasing memory and internal data structures
     *
     * @return TUC_STATUS_OK on success, error code on failure
     *
     * @note This function is thread-safe and can be called multiple times safely
     * @warning Must be called after all device operations are completed
     * @see TucInitialize(), TucDeviceClose(), TucDeviceEnumerate()
     * @since v1.0.0
     */
    /**
     * @brief Get SDK build information including version, build date, compiler info and platform details
     *
     * This function retrieves comprehensive build information for the Tucsen SDK  including:
     * - File version information (fixed version numbers)
     * - Product version information (with auto-incremented build number based on compile time)
     * - Build date and time (compilation timestamp)
     * - Compiler information and platform details
     * - RC file information (company name, copyright, etc.)
     *
     * The product build number is automatically generated from the compilation timestamp
     * in YYYYMMDDHHMM format to ensure unique versioning for each build.
     *
     * @param[out] pBuildInfo Pointer to TUC_BUILD_INFO structure to receive build information
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if pBuildInfo is NULL,
     *         TUC_STATUS_INTERNAL_ERROR on other failures
     * @note This function is thread-safe and can be called multiple times
     * @see TUC_BUILD_INFO
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucGetSDKBuildInfo(OUT TUC_BUILD_INFO *pBuildInfo);

    /**
     * @brief Terminate Tucsen SDK environment and cleanup resources
     *
     * This function terminates the Tucsen SDK environment and performs comprehensive cleanup
     * of all allocated resources. It should be called when the Tucsen SDK is no longer needed
     * to ensure proper resource deallocation and prevent memory leaks.
     *
     * The termination process includes:
     * - Closing all open devices and releasing device handles
     * - Clearing device management data structures
     * - Resetting internal state variables (initialization flag, device count, profile path)
     * - Cleaning up GenTL wrapper libraries and resources
     * - Flushing and closing logging system
     * - Releasing memory and internal data structures
     *
     * @return TUC_STATUS_OK on success, error code on failure
     *
     * @note This function is thread-safe and can be called multiple times safely
     * @warning Must be called after all device operations are completed
     * @see TucInitialize(), TucDeviceClose(), TucDeviceEnumerate()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucTerminate(void);

    // =============================================================================
    // 2. Device Discovery & Management - Enumeration, information, and connection control
    // =============================================================================

    /**
     * @brief Enumerate Tucsen devices in the system
     *
     * This function scans for all available Tucsen devices in the system, including USB, GigE,
     * Camera Link, and other interface types. The enumeration results update the internal
     * device list for subsequent device operations.
     *
     * The enumeration process includes:
     * - Scanning USB bus for Tucsen devices
     * - Scanning network for GigE devices
     * - Scanning Camera Link interface devices
     * - Scanning CoaXPress interface devices
     * - Updating internal device management data structures
     * - Returning the total number of available devices in the system
     *
     * @param[out] pCount Pointer to device count, returns the total number of available Tucsen devices
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if pCount is NULL,
     *         TUC_STATUS_INTERNAL_ERROR on other failures
     * @note This function is thread-safe and can be called multiple times
     * @warning Must call TucInitialize() first to initialize the SDK
     * @see TucInitialize(), TucDeviceOpen(), TucGetDeviceInfo()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucDeviceEnumerate(OUT uint32_t *pCount);

    /**
     * @brief Get basic device information
     *
     * This function retrieves basic information for a Tucsen device at the specified index.
     * The device must be previously enumerated using TucDeviceEnumerate() before its information
     * can be retrieved. This function provides general device details including device type,
     * interface information, and basic hardware specifications.
     *
     * @param[in] index Device index, starting from 0, must be less than the total device count
     * @param[out] pInfo Pointer to TUC_DEVINFO structure to receive device information
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if pInfo is NULL,
     *         TUC_STATUS_INVALID_DEVICE_INDEX if index is out of range
     *
     * @note This function is thread-safe and requires devices to be enumerated first
     * @warning Must call TucDeviceEnumerate() first to get valid device indices
     * @see TucDeviceEnumerate(), TucDeviceOpen(), TucGetCLDeviceInfo(), TucGetUSBDeviceInfo()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucGetDeviceInfo(IN uint32_t index, OUT TUC_DEVINFO *pInfo);

    /**
     * @brief Get Camera Link device detailed information
     *
     * This function retrieves detailed information for a Camera Link Tucsen device at the specified index.
     * The device must be previously enumerated using TucDeviceEnumerate() before its information
     * can be retrieved. This function provides Camera Link specific device details including
     * interface information, configuration settings, and hardware specifications.
     *
     * @param[in] index Device index, starting from 0, must be less than the total device count
     * @param[out] pInfo Pointer to TUC_DEVINFO_CL structure to receive Camera Link device information
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if pInfo is NULL,
     *         TUC_STATUS_INVALID_DEVICE_INDEX if index is out of range
     *
     * @note This function is thread-safe and requires devices to be enumerated first
     * @see TucDeviceEnumerate(), TucGetDeviceInfo(), TucDeviceOpen()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucGetCLDeviceInfo(IN uint32_t index, OUT TUC_DEVINFO_CL *pInfo);

    /**
     * @brief Get USB device detailed information
     *
     * This function retrieves detailed information for a USB Tucsen device at the specified index.
     * The device must be previously enumerated using TucDeviceEnumerate() before its information
     * can be retrieved. This function provides USB specific device details including
     * vendor ID, product ID, interface information, and USB configuration details.
     *
     * @param[in] index Device index, starting from 0, must be less than the total device count
     * @param[out] pInfo Pointer to TUC_DEVINFO_USB structure to receive USB device information
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if pInfo is NULL,
     *         TUC_STATUS_INVALID_DEVICE_INDEX if index is out of range
     *
     * @note This function is thread-safe and requires devices to be enumerated first
     * @see TucDeviceEnumerate(), TucGetDeviceInfo(), TucDeviceOpen()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucGetUSBDeviceInfo(IN uint32_t index, OUT TUC_DEVINFO_USB *pInfo);

    /**
     * @brief Get CoaXPress device detailed information
     *
     * This function retrieves detailed information for a CoaXPress (CXP) Tucsen device at the specified index.
     * The device must be previously enumerated using TucDeviceEnumerate() before its information
     * can be retrieved. This function provides CoaXPress specific device details including
     * link configuration, bandwidth information, and CXP protocol specifications.
     *
     * @param[in] index Device index, starting from 0, must be less than the total device count
     * @param[out] pInfo Pointer to TUC_DEVINFO_CXP structure to receive CoaXPress device information
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if pInfo is NULL,
     *         TUC_STATUS_INVALID_DEVICE_INDEX if index is out of range
     *
     * @note This function is thread-safe and requires devices to be enumerated first
     * @see TucDeviceEnumerate(), TucGetDeviceInfo(), TucDeviceOpen()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucGetCXPDeviceInfo(IN uint32_t index, OUT TUC_DEVINFO_CXP *pInfo);

    /**
     * @brief Get Gigabit Ethernet device detailed information
     *
     * This function retrieves detailed information for a Gigabit Ethernet (GigE) Tucsen device at the specified index.
     * The device must be previously enumerated using TucDeviceEnumerate() before its information
     * can be retrieved. This function provides GigE specific device details including
     * IP address, MAC address, network configuration, and GigE Vision protocol information.
     *
     * @param[in] index Device index, starting from 0, must be less than the total device count
     * @param[out] pInfo Pointer to TUC_DEVINFO_GIGE structure to receive GigE device information
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if pInfo is NULL,
     *         TUC_STATUS_INVALID_DEVICE_INDEX if index is out of range
     *
     * @note This function is thread-safe and requires devices to be enumerated first
     * @see TucDeviceEnumerate(), TucGetDeviceInfo(), TucDeviceOpen()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucGetGigEDeviceInfo(IN uint32_t index, OUT TUC_DEVINFO_GIGE *pInfo);

    /**
     * @brief Open Tucsen device at specified index with access policy
     *
     * This function opens a Tucsen device at the specified index and establishes a connection
     * with the device using the specified access policy. The device must be previously
     * enumerated using TucDeviceEnumerate() before it can be opened.
     *
     * The device opening process includes:
     * - Validating the device index against available devices
     * - Establishing communication with the device hardware
     * - Initializing device-specific data structures
     * - Setting up access control based on the specified policy
     * - Creating and returning a device handle for subsequent operations
     *
     * Note: Current version only supports EXCLUSIVE access policy. Other policies
     * will be automatically converted to EXCLUSIVE access.
     *
     * @param[in] index Device index, starting from 0, must be less than the total device count
     * @param[in] policy Device access policy (TUC_OPEN_POLICY_EXCLUSIVE is currently the only supported policy)
     * @param[out] phDevice Pointer to device handle, returns a valid handle on success
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if parameters are invalid,
     *         TUC_STATUS_INVALID_DEVICE_INDEX if index is out of range,
     *         TUC_STATUS_OPEN_DEVICE_FAILED on device communication failure
     * @note This function is thread-safe and requires a valid device index from TucDeviceEnumerate()
     * @warning Must call TucDeviceEnumerate() first to get valid device indices
     * @see TucDeviceEnumerate(), TucDeviceClose(), TucGetDeviceInfo()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucDeviceOpen(IN uint32_t index, IN TUC_OPEN_POLICY policy, OUT TUC_HDEVICE *phDevice);

    /**
     * @brief Open Tucsen device by serial number or user ID with access policy
     *
     * This function opens a Tucsen device by searching for a device with the specified serial number
     * or user ID and establishes a connection with the device using the specified access policy.
     * The device must be previously enumerated using TucDeviceEnumerate() before it can be opened.
     *
     * The device opening process includes:
     * - Searching for device with matching serial number or user ID
     * - Validating the device identifier against available devices
     * - Establishing communication with the device hardware
     * - Initializing device-specific data structures
     * - Setting up access control based on the specified policy
     * - Creating and returning a device handle for subsequent operations
     *
     * Note: Current version only supports EXCLUSIVE access policy. Other policies
     * will be automatically converted to EXCLUSIVE access.
     *
     * @param[in] deviceId Device identifier (serial number or user ID)
     * @param[in] idType Type of device identifier (TUC_DEVICE_ID_SERIAL or TUC_DEVICE_ID_USER)
     * @param[in] policy Device access policy (TUC_OPEN_POLICY_EXCLUSIVE is currently the only supported policy)
     * @param[out] phDevice Pointer to device handle, returns a valid handle on success
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if parameters are invalid,
     *         TUC_STATUS_DEVICE_NOT_FOUND if device with specified ID is not found,
     *         TUC_STATUS_OPEN_DEVICE_FAILED on device communication failure
     * @note This function is thread-safe and requires devices to be enumerated first
     * @warning Must call TucDeviceEnumerate() first to get valid device list
     * @see TucDeviceEnumerate(), TucDeviceClose(), TucGetDeviceInfo()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucDeviceOpenEx(IN const char *deviceId, IN TUC_DEVICE_ID_TYPE idType, IN TUC_OPEN_POLICY policy, OUT TUC_HDEVICE *phDevice);

    /**
     * @brief Close Tucsen device and release resources
     *
     * This function closes a Tucsen device and releases all associated resources including
     * device handles, buffers, and internal data structures. The device must be previously
     * opened using TucDeviceOpen() before it can be closed.
     *
     * The device closing process includes:
     * - Validating the device handle and checking if it's still open
     * - Stopping any ongoing acquisition or data transfer operations
     * - Releasing device-specific resources and memory allocations
     * - Silently releasing pixel memory allocated by TucImageLoadFromFile() if TucImageUnload()
     *   was not called (any outstanding pImageInfo->data from Load becomes invalid)
     * - Closing communication channels with the device hardware
     * - Cleaning up internal device management data structures
     * - Invalidating the device handle for subsequent operations
     *
     * @param[in] hDevice Device handle obtained from TucDeviceOpen(), must be valid
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if hDevice is NULL,
     *         TUC_STATUS_INVALID_DEVICE_HANDLE if handle is invalid or already closed,
     *         TUC_STATUS_CLOSE_DEVICE_FAILED on device communication failure
     * @note This function is thread-safe and can be called multiple times safely
     * @warning Must be called after all device operations are completed to prevent resource leaks
     * @see TucDeviceOpen(), TucDeviceEnumerate(), TucGetDeviceInfo(), TucImageUnload()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucDeviceClose(IN TUC_HDEVICE hDevice);

    /**
     * @brief Get basic device information for specified device index
     *
     * This function retrieves basic device information for a Tucsen device at the specified index.
     * The device must be previously enumerated using TucDeviceEnumerate() before its information
     * can be retrieved. This function provides essential device details such as device name,
     * model, vendor, serial number, and other basic identification information.
     *
     * The information retrieval process includes:
     * - Validating the device index against available devices
     * - Accessing device-specific information from the internal device list
     * - Retrieving basic device identification and model details
     * - Filling the device information structure with available data
     * - Returning comprehensive device information for user applications
     *
     * @param[in] index Device index, starting from 0, must be less than the total device count
     * @param[out] pInfo Pointer to TUC_DEVINFO structure to receive device information
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if pInfo is NULL,
     *         TUC_STATUS_INVALID_DEVICE_INDEX if index is out of range,
     *         TUC_STATUS_GET_DEVICE_INFO_FAILED on device information retrieval failure
     * @note This function is thread-safe and requires a valid device index from TucDeviceEnumerate()
     * @warning Must call TucDeviceEnumerate() first to get valid device indices
     * @see TucDeviceEnumerate(), TucDeviceOpen(), TucGetCLDeviceInfo(), TucGetUSBDeviceInfo()
     * @since v1.0.0
     */

    // =============================================================================
    // 3. Feature Parameter Management - Information retrieval and access control
    // =============================================================================

    /**
     * @brief Get device feature parameter count
     *
     * This function retrieves the total number of available feature parameters for a Tucsen device.
     * The device must be previously opened using TucDeviceOpen() before its feature information
     * can be retrieved. This function provides the count of all configurable and readable features
     * that are available on the device.
     *
     * The feature count retrieval process includes:
     * - Validating the device handle against opened devices
     * - Accessing device-specific feature registry from the internal device state
     * - Counting all available feature parameters including numeric, boolean, string, and enum types
     * - Returning the total feature count for user applications
     *
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @param[out] pCount Pointer to uint32_t variable to receive the total feature count
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if pCount is NULL,
     *         TUC_STATUS_INVALID_DEVICE_HANDLE if hDevice is invalid,
     *         TUC_STATUS_GET_FEATURE_COUNT_FAILED on feature count retrieval failure
     * @note This function is thread-safe and requires a valid device handle from TucDeviceOpen()
     * @warning Must call TucDeviceOpen() first to get a valid device handle
     * @see TucDeviceOpen(), TucGetFeatureInfo(), TucIsFeatureReadable(), TucIsFeatureWritable()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucGetFeatureCount(IN TUC_HDEVICE hDevice, OUT uint32_t *pCount);

    /**
     * @brief Get feature parameter information by name
     *
     * This function retrieves comprehensive information about a specific feature parameter on a Tucsen device.
     * The device must be previously opened using TucDeviceOpen() and the feature name must be valid
     * before its information can be retrieved. This function provides detailed feature metadata including
     * type, access mode, visibility, description, and other configuration details.
     *
     * The feature information retrieval process includes:
     * - Validating the device handle and feature name parameters
     * - Accessing device-specific feature registry to locate the specified feature
     * - Retrieving comprehensive feature metadata and configuration information
     * - Filling the feature information structure with available data
     * - Returning detailed feature information for user applications
     *
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @param[in] featureName Name of the feature parameter to query (case-sensitive)
     * @param[out] pInfo Pointer to TUC_FEATURE_INFO structure to receive feature information
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if any parameter is NULL,
     *         TUC_STATUS_INVALID_DEVICE_HANDLE if hDevice is invalid,
     *         TUC_STATUS_FEATURE_NOT_FOUND if featureName does not exist,
     *         TUC_STATUS_GET_FEATURE_INFO_FAILED on feature information retrieval failure
     * @note This function is thread-safe and requires a valid device handle from TucDeviceOpen()
     * @warning Must call TucDeviceOpen() first to get a valid device handle
     * @see TucDeviceOpen(), TucGetFeatureCount(), TucIsFeatureReadable(), TucIsFeatureWritable()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucGetFeatureInfoByName(IN TUC_HDEVICE hDevice, IN const char *featureName, OUT TUC_FEATURE_INFO *pInfo);


    /**
     * @brief Get feature parameter information by index
     *
     * This function retrieves the next feature parameter information for a Tucsen device.
     * The device must be previously opened using TucDeviceOpen() and the feature name must be valid
     * before its information can be retrieved. This function provides detailed feature metadata including
     * type, access mode, visibility, description, and other configuration details.
     *
     * The feature information retrieval process includes:
     * - Validating the device handle and feature name parameters
     * - Accessing device-specific feature registry to locate the specified feature
     * - Retrieving comprehensive feature metadata and configuration information
     * - Filling the feature information structure with available data
     * - Returning detailed feature information for user applications
     *
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @param[in] index Index of the feature parameter to query
     * @param[out] pInfo Pointer to TUC_FEATURE_INFO structure to receive feature information
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if any parameter is NULL,
     *         TUC_STATUS_INVALID_DEVICE_HANDLE if hDevice is invalid,
     *         TUC_STATUS_FEATURE_NOT_FOUND if index is out of range,
     *         TUC_STATUS_GET_FEATURE_INFO_FAILED on feature information retrieval failure
     * @note This function is thread-safe and requires a valid device handle from TucDeviceOpen()
     * @warning Must call TucDeviceOpen() first to get a valid device handle
     * @see TucDeviceOpen(), TucGetFeatureCount(), TucIsFeatureReadable(), TucIsFeatureWritable()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucGetFeatureInfoByIndex(IN TUC_HDEVICE hDevice, IN uint32_t index, OUT TUC_FEATURE_INFO *pInfo);

    /**
     * @brief Check if feature parameter is readable
     *
     * This function checks whether a specific feature parameter on a Tucsen device can be read.
     * The device must be previously opened using TucDeviceOpen() and the feature name must be valid
     * before its readability can be checked. This function determines if the feature's current value
     * can be retrieved using the corresponding getter function.
     *
     * The readability check process includes:
     * - Validating the device handle and feature name parameters
     * - Accessing device-specific feature registry to locate the specified feature
     * - Checking the feature's access mode and current state for readability
     * - Determining if the feature is in a readable state (not busy, available, etc.)
     * - Returning the readability status for user applications
     *
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @param[in] featureName Name of the feature parameter to check (case-sensitive)
     * @param[out] pReadable Pointer to bool variable to receive readability status
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if any parameter is NULL,
     *         TUC_STATUS_INVALID_DEVICE_HANDLE if hDevice is invalid,
     *         TUC_STATUS_FEATURE_NOT_FOUND if featureName does not exist,
     *         TUC_STATUS_CHECK_FEATURE_READABLE_FAILED on readability check failure
     * @note This function is thread-safe and requires a valid device handle from TucDeviceOpen()
     * @warning Must call TucDeviceOpen() first to get a valid device handle
     * @see TucDeviceOpen(), TucGetFeatureInfo(), TucIsFeatureWritable(), TucGetFeatureInt64()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucIsFeatureReadable(IN TUC_HDEVICE hDevice, IN const char *featureName, OUT bool *pReadable);

    /**
     * @brief Check if feature parameter is writable
     *
     * This function checks whether a specific feature parameter on a Tucsen device can be written to.
     * The device must be previously opened using TucDeviceOpen() and the feature name must be valid
     * before its writability can be checked. This function determines if the feature's value
     * can be modified using the corresponding setter function.
     *
     * The writability check process includes:
     * - Validating the device handle and feature name parameters
     * - Accessing device-specific feature registry to locate the specified feature
     * - Checking the feature's access mode and current state for writability
     * - Determining if the feature is in a writable state (not busy, not read-only, etc.)
     * - Returning the writability status for user applications
     *
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @param[in] featureName Name of the feature parameter to check (case-sensitive)
     * @param[out] pWritable Pointer to bool variable to receive writability status
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if any parameter is NULL,
     *         TUC_STATUS_INVALID_DEVICE_HANDLE if hDevice is invalid,
     *         TUC_STATUS_FEATURE_NOT_FOUND if featureName does not exist,
     *         TUC_STATUS_CHECK_FEATURE_WRITABLE_FAILED on writability check failure
     * @note This function is thread-safe and requires a valid device handle from TucDeviceOpen()
     * @warning Must call TucDeviceOpen() first to get a valid device handle
     * @see TucDeviceOpen(), TucGetFeatureInfo(), TucIsFeatureReadable(), TucSetFeatureInt64()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucIsFeatureWritable(IN TUC_HDEVICE hDevice, IN const char *featureName, OUT bool *pWritable);

    // =============================================================================
    // 4. Feature Parameter Set/Get Interfaces - Numeric Types
    // =============================================================================

    /**
     * @brief Set feature integer value (64-bit)
     *
     * This function sets the value of a 64-bit integer feature parameter on a Tucsen device.
     * The device must be previously opened using TucDeviceOpen() and the feature must be writable
     * before its value can be modified. This function supports features such as exposure time,
     * gain, width, height, and other integer-based camera parameters.
     *
     * The feature value setting process includes:
     * - Validating the device handle and feature name parameters
     * - Checking if the feature exists and is writable on the device
     * - Validating the input value against feature constraints (min/max, step, etc.)
     * - Applying the new value to the device feature
     * - Confirming the value was successfully set
     *
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @param[in] featureName Name of the feature parameter to set (case-sensitive)
     * @param[in] value 64-bit integer value to set for the feature
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if any parameter is invalid,
     *         TUC_STATUS_INVALID_DEVICE_HANDLE if hDevice is invalid,
     *         TUC_STATUS_FEATURE_NOT_FOUND if featureName does not exist,
     *         TUC_STATUS_FEATURE_NOT_WRITABLE if feature is read-only,
     *         TUC_STATUS_FEATURE_VALUE_OUT_OF_RANGE if value exceeds limits,
     *         TUC_STATUS_SET_FEATURE_VALUE_FAILED on value setting failure
     * @note This function is thread-safe and requires a valid device handle from TucDeviceOpen()
     * @warning Must call TucDeviceOpen() first and ensure feature is writable
     * @see TucDeviceOpen(), TucGetFeatureInt64(), TucIsFeatureWritable(), TucGetFeatureInfo()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucSetFeatureInt64(IN TUC_HDEVICE hDevice, IN const char *featureName, IN int64_t value);

    /**
     * @brief Get feature integer value (64-bit)
     *
     * This function retrieves the current value of a 64-bit integer feature parameter from a Tucsen device.
     * The device must be previously opened using TucDeviceOpen() and the feature must be readable
     * before its value can be retrieved. This function supports features such as exposure time,
     * gain, width, height, and other integer-based camera parameters.
     *
     * The feature value retrieval process includes:
     * - Validating the device handle and feature name parameters
     * - Checking if the feature exists and is readable on the device
     * - Accessing the current value of the feature from the device
     * - Retrieving the feature value and any associated metadata
     * - Returning the current value for user applications
     *
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @param[in] featureName Name of the feature parameter to get (case-sensitive)
     * @param[out] pValue Pointer to int64_t variable to receive the current feature value
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if any parameter is NULL,
     *         TUC_STATUS_INVALID_DEVICE_HANDLE if hDevice is invalid,
     *         TUC_STATUS_FEATURE_NOT_FOUND if featureName does not exist,
     *         TUC_STATUS_FEATURE_NOT_READABLE if feature is not readable,
     *         TUC_STATUS_GET_FEATURE_VALUE_FAILED on value retrieval failure
     * @note This function is thread-safe and requires a valid device handle from TucDeviceOpen()
     * @warning Must call TucDeviceOpen() first and ensure feature is readable
     * @see TucDeviceOpen(), TucSetFeatureInt64(), TucIsFeatureReadable(), TucGetFeatureInfo()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucGetFeatureInt64(IN TUC_HDEVICE hDevice, IN const char *featureName, OUT int64_t *pValue);

    /**
     * @brief Get feature integer value and range information (64-bit)
     *
     * This function retrieves the complete value and range information for a 64-bit integer feature parameter
     * from a Tucsen device, including current value, minimum, maximum, increment step, and default value.
     * The device must be previously opened using TucDeviceOpen() and the feature must be readable
     * before its information can be retrieved.
     *
     * This function is particularly useful for:
     * - Validating user input before setting feature values
     * - Creating UI controls (sliders, spin boxes) with appropriate ranges
     * - Understanding feature constraints and capabilities
     * - Implementing value validation and error handling
     * - Initializing UI controls with current and default values
     *
     * The information retrieval process includes:
     * - Validating the device handle and feature name parameters
     * - Checking if the feature exists and is readable on the device
     * - Verifying that the feature is of integer type
     * - Retrieving current value, minimum, maximum, increment, and default value from the device
     * - Returning complete information for application use
     *
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @param[in] featureName Name of the integer feature parameter to query (case-sensitive)
     * @param[out] pInfo Pointer to TUC_FEATURE_INT64_INFO structure to receive information
     * @return TUC_STATUS_OK on success, TUC_STATUS_NOT_INITIALIZED if SDK is not initialized,
     *         TUC_STATUS_INVALID_PARAM if hDevice, featureName or pInfo is NULL,
     *         TUC_STATUS_DEVICE_NOT_OPENED if device is not opened,
     *         TUC_STATUS_FEATURE_NOT_SUPPORTED if feature is not available or not integer type,
     *         TUC_STATUS_FEATURE_TYPE_MISMATCH if feature is not integer type,
     *         TUC_STATUS_GET_FEATURE_VALUE_FAILED on feature value retrieval failure,
     *         TUC_STATUS_INTERNAL_ERROR on internal failure
     * @note This function is thread-safe and requires a valid device handle from TucDeviceOpen()
     * @note The currentValue field may change between calls; use TucGetFeatureInt64() for real-time value queries
     * @warning Must call TucDeviceOpen() first and ensure feature is readable
     * @see TucDeviceOpen(), TucGetFeatureInt64(), TucSetFeatureInt64(), TucGetFeatureInfo()
     * @since v1.0.0
     */
     TUC_API TUC_STATUS TucGetFeatureInt64Info(IN TUC_HDEVICE hDevice, IN const char *featureName, OUT TUC_FEATURE_INT64_INFO *pInfo);

    /**
     * @brief Set feature float value
     *
     * This function sets the value of a floating-point feature parameter on a Tucsen device.
     * The device must be previously opened using TucDeviceOpen() and the feature must be writable
     * before its value can be modified. This function supports features such as gain, temperature,
     * and other floating-point based camera parameters with decimal precision.
     *
     * The feature value setting process includes:
     * - Validating the device handle and feature name parameters
     * - Checking if the feature exists and is writable on the device
     * - Validating the input value against feature constraints (min/max, step, precision, etc.)
     * - Converting the double value to the device's internal representation if needed
     * - Applying the new value to the device feature
     * - Confirming the value was successfully set
     *
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @param[in] featureName Name of the feature parameter to set (case-sensitive)
     * @param[in] value Double precision floating-point value to set for the feature
     * @return TUC_STATUS_OK on success, TUC_STATUS_NOT_INITIALIZED if SDK is not initialized,
     *         TUC_STATUS_INVALID_PARAM if hDevice or featureName is NULL, TUC_STATUS_DEVICE_NOT_OPENED if device is not opened,
     *         TUC_STATUS_FEATURE_NOT_SUPPORTED if features are not available or setting failed, TUC_STATUS_INTERNAL_ERROR on internal failure
     * @note This function is thread-safe and requires a valid device handle from TucDeviceOpen()
     * @warning Must call TucDeviceOpen() first and ensure feature is writable
     * @see TucDeviceOpen(), TucGetFeatureFloat(), TucIsFeatureWritable(), TucGetFeatureInfo()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucSetFeatureFloat(IN TUC_HDEVICE hDevice, IN const char *featureName, IN double value);

    /**
     * @brief Get feature float value
     *
     * This function retrieves the current value of a floating-point feature parameter from a Tucsen device.
     * The device must be previously opened using TucDeviceOpen() and the feature must be readable
     * before its value can be retrieved. This function supports features such as gain, temperature,
     * and other floating-point based camera parameters with decimal precision.
     *
     * The feature value retrieval process includes:
     * - Validating the device handle and feature name parameters
     * - Checking if the feature exists and is readable on the device
     * - Accessing the current value of the feature from the device
     * - Converting the device's internal representation to double precision if needed
     * - Retrieving the feature value and any associated metadata
     * - Returning the current value for user applications
     *
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @param[in] featureName Name of the feature parameter to get (case-sensitive)
     * @param[out] pValue Pointer to double variable to receive the current feature value
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if any parameter is NULL,
     *         TUC_STATUS_INVALID_DEVICE_HANDLE if hDevice is invalid,
     *         TUC_STATUS_FEATURE_NOT_FOUND if featureName does not exist,
     *         TUC_STATUS_FEATURE_NOT_READABLE if feature is not readable,
     *         TUC_STATUS_GET_FEATURE_VALUE_FAILED on value retrieval failure
     * @note This function is thread-safe and requires a valid device handle from TucDeviceOpen()
     * @warning Must call TucDeviceOpen() first and ensure feature is readable
     * @see TucDeviceOpen(), TucSetFeatureFloat(), TucIsFeatureReadable(), TucGetFeatureInfo()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucGetFeatureFloat(IN TUC_HDEVICE hDevice, IN const char *featureName, OUT double *pValue);

    /**
     * @brief Get feature float value and range information
     *
     * This function retrieves the complete value and range information for a floating-point feature parameter
     * from a Tucsen device, including current value, minimum, maximum, increment step, and default value.
     * The device must be previously opened using TucDeviceOpen() and the feature must be readable
     * before its information can be retrieved.
     *
     * This function is particularly useful for:
     * - Validating user input before setting feature values
     * - Creating UI controls (sliders, spin boxes) with appropriate ranges and precision
     * - Understanding feature constraints and capabilities
     * - Implementing value validation and error handling
     * - Handling floating-point precision and rounding
     * - Initializing UI controls with current and default values
     *
     * The information retrieval process includes:
     * - Validating the device handle and feature name parameters
     * - Checking if the feature exists and is readable on the device
     * - Verifying that the feature is of floating-point type
     * - Retrieving current value, minimum, maximum, increment, and default value from the device
     * - Returning complete information for application use
     *
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @param[in] featureName Name of the floating-point feature parameter to query (case-sensitive)
     * @param[out] pInfo Pointer to TUC_FEATURE_FLOAT_INFO structure to receive information
     * @return TUC_STATUS_OK on success, TUC_STATUS_NOT_INITIALIZED if SDK is not initialized,
     *         TUC_STATUS_INVALID_PARAM if hDevice, featureName or pInfo is NULL,
     *         TUC_STATUS_DEVICE_NOT_OPENED if device is not opened,
     *         TUC_STATUS_FEATURE_NOT_SUPPORTED if feature is not available or not float type,
     *         TUC_STATUS_FEATURE_TYPE_MISMATCH if feature is not floating-point type,
     *         TUC_STATUS_GET_FEATURE_VALUE_FAILED on feature value retrieval failure,
     *         TUC_STATUS_INTERNAL_ERROR on internal failure
     * @note This function is thread-safe and requires a valid device handle from TucDeviceOpen()
     * @note The currentValue field may change between calls; use TucGetFeatureFloat() for real-time value queries
     * @warning Must call TucDeviceOpen() first and ensure feature is readable
     * @see TucDeviceOpen(), TucGetFeatureFloat(), TucSetFeatureFloat(), TucGetFeatureInfo()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucGetFeatureFloatInfo(IN TUC_HDEVICE hDevice, IN const char *featureName, OUT TUC_FEATURE_FLOAT_INFO *pInfo);

    // =============================================================================
    // 5. Feature Parameter Set/Get Interfaces - Boolean and String Types
    // =============================================================================

    /**
     * @brief Set feature boolean value
     *
     * This function sets the value of a boolean feature parameter on a Tucsen device.
     * The device must be previously opened using TucDeviceOpen() and the feature must be writable
     * before its value can be modified. This function supports features such as trigger enable,
     * auto exposure, auto gain, and other on/off camera parameters.
     *
     * The feature value setting process includes:
     * - Validating the device handle and feature name parameters
     * - Checking if the feature exists and is writable on the device
     * - Validating the input boolean value against feature constraints
     * - Converting the boolean value to the device's internal representation if needed
     * - Applying the new value to the device feature
     * - Confirming the value was successfully set
     *
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @param[in] featureName Name of the feature parameter to set (case-sensitive)
     * @param[in] value Boolean value to set for the feature (true/false)
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if any parameter is invalid,
     *         TUC_STATUS_INVALID_DEVICE_HANDLE if hDevice is invalid,
     *         TUC_STATUS_FEATURE_NOT_FOUND if featureName does not exist,
     *         TUC_STATUS_FEATURE_NOT_WRITABLE if feature is read-only,
     *         TUC_STATUS_FEATURE_VALUE_OUT_OF_RANGE if value is not supported,
     *         TUC_STATUS_SET_FEATURE_VALUE_FAILED on value setting failure
     * @note This function is thread-safe and requires a valid device handle from TucDeviceOpen()
     * @warning Must call TucDeviceOpen() first and ensure feature is writable
     * @see TucDeviceOpen(), TucGetFeatureBool(), TucIsFeatureWritable(), TucGetFeatureInfo()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucSetFeatureBool(IN TUC_HDEVICE hDevice, IN const char *featureName, IN bool value);

    /**
     * @brief Get feature boolean value
     *
     * This function retrieves the current value of a boolean feature parameter from a Tucsen device.
     * The device must be previously opened using TucDeviceOpen() and the feature must be readable
     * before its value can be retrieved. This function supports features such as trigger enable,
     * auto exposure, auto gain, and other on/off camera parameters.
     *
     * The feature value retrieval process includes:
     * - Validating the device handle and feature name parameters
     * - Checking if the feature exists and is readable on the device
     * - Accessing the current value of the feature from the device
     * - Converting the device's internal representation to boolean if needed
     * - Retrieving the feature value and any associated metadata
     * - Returning the current value for user applications
     *
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @param[in] featureName Name of the feature parameter to get (case-sensitive)
     * @param[out] pValue Pointer to bool variable to receive the current feature value
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if any parameter is NULL,
     *         TUC_STATUS_INVALID_DEVICE_HANDLE if hDevice is invalid,
     *         TUC_STATUS_FEATURE_NOT_FOUND if featureName does not exist,
     *         TUC_STATUS_FEATURE_NOT_READABLE if feature is not readable,
     *         TUC_STATUS_GET_FEATURE_VALUE_FAILED on value retrieval failure
     * @note This function is thread-safe and requires a valid device handle from TucDeviceOpen()
     * @warning Must call TucDeviceOpen() first and ensure feature is readable
     * @see TucDeviceOpen(), TucSetFeatureBool(), TucIsFeatureReadable(), TucGetFeatureInfo()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucGetFeatureBool(IN TUC_HDEVICE hDevice, IN const char *featureName, OUT bool *pValue);

    /**
     * @brief Set feature string value
     *
     * This function sets the value of a string feature parameter on a Tucsen device.
     * The device must be previously opened using TucDeviceOpen() and the feature must be writable
     * before its value can be modified. This function supports features such as device name,
     * user defined name, and other text-based camera parameters.
     *
     * The feature value setting process includes:
     * - Validating the device handle and feature name parameters
     * - Checking if the feature exists and is writable on the device
     * - Validating the input string value against feature constraints (length, format, etc.)
     * - Converting the string value to the device's internal representation if needed
     * - Applying the new value to the device feature
     * - Confirming the value was successfully set
     *
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @param[in] featureName Name of the feature parameter to set (case-sensitive)
     * @param[in] value Null-terminated string value to set for the feature
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if any parameter is NULL,
     *         TUC_STATUS_INVALID_DEVICE_HANDLE if hDevice is invalid,
     *         TUC_STATUS_FEATURE_NOT_FOUND if featureName does not exist,
     *         TUC_STATUS_FEATURE_NOT_WRITABLE if feature is read-only,
     *         TUC_STATUS_FEATURE_VALUE_OUT_OF_RANGE if string length exceeds limits,
     *         TUC_STATUS_SET_FEATURE_VALUE_FAILED on value setting failure
     * @note This function is thread-safe and requires a valid device handle from TucDeviceOpen()
     * @warning Must call TucDeviceOpen() first and ensure feature is writable
     * @see TucDeviceOpen(), TucGetFeatureString(), TucIsFeatureWritable(), TucGetFeatureInfo()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucSetFeatureString(IN TUC_HDEVICE hDevice, IN const char *featureName, IN const char *value);

    /**
     * @brief Get feature string value
     *
     * This function retrieves the current value of a string feature parameter from a Tucsen device.
     * The device must be previously opened using TucDeviceOpen() and the feature must be readable
     * before its value can be retrieved. This function supports features such as device name,
     * user defined name, and other text-based camera parameters.
     *
     * The feature value retrieval process includes:
     * - Validating the device handle and feature name parameters
     * - Checking if the feature exists and is readable on the device
     * - Accessing the current value of the feature from the device
     * - Converting the device's internal representation to string if needed
     * - Retrieving the feature value and determining its length
     * - Copying the string value to the provided buffer
     * - Updating the buffer size information
     *
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @param[in] featureName Name of the feature parameter to get (case-sensitive)
     * @param[out] pValue Pointer to character buffer to receive the current feature value
     * @param[in,out] pSize Pointer to buffer size (input: maximum size, output: actual size)
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if any parameter is NULL,
     *         TUC_STATUS_INVALID_DEVICE_HANDLE if hDevice is invalid,
     *         TUC_STATUS_FEATURE_NOT_FOUND if featureName does not exist,
     *         TUC_STATUS_FEATURE_NOT_READABLE if feature is not readable,
     *         TUC_STATUS_BUFFER_TOO_SMALL if buffer size is insufficient,
     *         TUC_STATUS_GET_FEATURE_VALUE_FAILED on value retrieval failure
     * @note This function is thread-safe and requires a valid device handle from TucDeviceOpen()
     * @warning Must call TucDeviceOpen() first and ensure feature is readable
     * @see TucDeviceOpen(), TucSetFeatureString(), TucIsFeatureReadable(), TucGetFeatureInfo()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucGetFeatureString(IN TUC_HDEVICE hDevice, IN const char *featureName, OUT char *pValue, INOUT uint32_t *pSize);

    // =============================================================================
    // 6. Feature Parameter Set/Get Interfaces - Enumeration Types
    // =============================================================================

    /**
     * @brief Set feature enum value by value
     *
     * This function sets the value of an enumeration feature parameter on a Tucsen device.
     * The device must be previously opened using TucDeviceOpen() and the feature must be writable
     * before its value can be modified. This function supports features such as pixel format,
     * trigger mode, acquisition mode, and other enumeration-based camera parameters.
     *
     * The feature value setting process includes:
     * - Validating the device handle and feature name parameters
     * - Checking if the feature exists and is writable on the device
     * - Validating the input enum value against available enumeration options
     * - Converting the enum value to the device's internal representation if needed
     * - Applying the new value to the device feature
     * - Confirming the value was successfully set
     *
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @param[in] featureName Name of the feature parameter to set (case-sensitive)
     * @param[in] value Integer enum value to set for the feature
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if any parameter is invalid,
     *         TUC_STATUS_INVALID_DEVICE_HANDLE if hDevice is invalid,
     *         TUC_STATUS_FEATURE_NOT_FOUND if featureName does not exist,
     *         TUC_STATUS_FEATURE_NOT_WRITABLE if feature is read-only,
     *         TUC_STATUS_FEATURE_VALUE_OUT_OF_RANGE if enum value is not supported,
     *         TUC_STATUS_SET_FEATURE_VALUE_FAILED on value setting failure
     * @note This function is thread-safe and requires a valid device handle from TucDeviceOpen()
     * @warning Must call TucDeviceOpen() first and ensure feature is writable
     * @see TucDeviceOpen(), TucGetFeatureEnum(), TucIsFeatureWritable(), TucGetFeatureEnumInfo()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucSetFeatureEnum(IN TUC_HDEVICE hDevice, IN const char *featureName, IN int64_t value);

    /**
     * @brief Get feature enum value by value
     *
     * This function retrieves the current value of an enumeration feature parameter from a Tucsen device.
     * The device must be previously opened using TucDeviceOpen() and the feature must be readable
     * before its value can be retrieved. This function supports features such as pixel format,
     * trigger mode, acquisition mode, and other enumeration-based camera parameters.
     *
     * The feature value retrieval process includes:
     * - Validating the device handle and feature name parameters
     * - Checking if the feature exists and is readable on the device
     * - Accessing the current value of the feature from the device
     * - Converting the device's internal representation to integer enum value if needed
     * - Retrieving the feature value and any associated metadata
     * - Returning the current enum value for user applications
     *
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @param[in] featureName Name of the feature parameter to get (case-sensitive)
     * @param[out] pValue Pointer to int64_t variable to receive the current enum value
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if any parameter is NULL,
     *         TUC_STATUS_INVALID_DEVICE_HANDLE if hDevice is invalid,
     *         TUC_STATUS_FEATURE_NOT_FOUND if featureName does not exist,
     *         TUC_STATUS_FEATURE_NOT_READABLE if feature is not readable,
     *         TUC_STATUS_GET_FEATURE_VALUE_FAILED on value retrieval failure
     * @note This function is thread-safe and requires a valid device handle from TucDeviceOpen()
     * @warning Must call TucDeviceOpen() first and ensure feature is readable
     * @see TucDeviceOpen(), TucSetFeatureEnum(), TucIsFeatureReadable(), TucGetFeatureEnumInfo()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucGetFeatureEnum(IN TUC_HDEVICE hDevice, IN const char *featureName, OUT int64_t *pValue);

    /**
     * @brief Get feature enum count
     *
     * This function retrieves the total number of available enumeration values for a specific feature parameter.
     * The device must be previously opened using TucDeviceOpen() and the feature must be readable
     * before its enumeration count can be retrieved. This function provides the count of all valid
     * enumeration options for features such as pixel format, trigger mode, and acquisition mode.
     *
     * The enumeration count retrieval process includes:
     * - Validating the device handle and feature name parameters
     * - Checking if the feature exists and is readable on the device
     * - Accessing the feature's enumeration registry to count available options
     * - Counting all valid enumeration values for the specified feature
     * - Returning the total enumeration count for user applications
     *
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @param[in] featureName Name of the feature parameter to query (case-sensitive)
     * @param[out] pCount Pointer to uint32_t variable to receive the enumeration count
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if any parameter is NULL,
     *         TUC_STATUS_INVALID_DEVICE_HANDLE if hDevice is invalid,
     *         TUC_STATUS_FEATURE_NOT_FOUND if featureName does not exist,
     *         TUC_STATUS_FEATURE_NOT_READABLE if feature is not readable,
     *         TUC_STATUS_GET_FEATURE_ENUM_COUNT_FAILED on enumeration count retrieval failure
     * @note This function is thread-safe and requires a valid device handle from TucDeviceOpen()
     * @warning Must call TucDeviceOpen() first and ensure feature is readable
     * @see TucDeviceOpen(), TucGetFeatureEnum(), TucGetFeatureEnumInfo(), TucGetFeatureEnumEntryByIndex()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucGetFeatureEnumCount(IN TUC_HDEVICE hDevice, IN const char *featureName, OUT uint32_t *pCount);

    /**
     * @brief Get feature enum value by index
     *
     * This function retrieves the symbolic name of an enumeration value by its index for a specific feature parameter.
     * The device must be previously opened using TucDeviceOpen() and the feature must be readable
     * before its enumeration information can be retrieved. This function provides the human-readable
     * name of enumeration values for features such as pixel format, trigger mode, and acquisition mode.
     *
     * The enumeration symbolic retrieval process includes:
     * - Validating the device handle and feature name parameters
     * - Checking if the feature exists and is readable on the device
     * - Validating the index parameter against available enumeration count
     * - Accessing the feature's enumeration registry to retrieve the symbolic name
     * - Converting the symbolic name to string representation if needed
     * - Copying the symbolic name to the provided buffer
     * - Updating the buffer size information
     *
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @param[in] featureName Name of the feature parameter to query (case-sensitive)
     * @param[in] index Index of the enumeration value (0-based, must be less than enum count)
     * @param[out] pValue Pointer to character buffer to receive the symbolic name
     * @param[in,out] pSize Pointer to buffer size (input: maximum size, output: actual size)
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if any parameter is NULL,
     *         TUC_STATUS_INVALID_DEVICE_HANDLE if hDevice is invalid,
     *         TUC_STATUS_FEATURE_NOT_FOUND if featureName does not exist,
     *         TUC_STATUS_FEATURE_NOT_READABLE if feature is not readable,
     *         TUC_STATUS_INVALID_ENUM_INDEX if index is out of range,
     *         TUC_STATUS_BUFFER_TOO_SMALL if buffer size is insufficient,
     *         TUC_STATUS_GET_FEATURE_ENUM_SYMBOLIC_FAILED on symbolic retrieval failure
     * @note This function is thread-safe and requires a valid device handle from TucDeviceOpen()
     * @warning Must call TucDeviceOpen() first and ensure feature is readable
     * @see TucDeviceOpen(), TucGetFeatureEnumCount(), TucGetFeatureEnumInfo(), TucGetFeatureEnumEntryByIndex()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucGetFeatureEnumSymbolic(IN TUC_HDEVICE hDevice, IN const char *featureName, IN uint32_t index, OUT char *pValue, INOUT uint32_t *pSize);


    /**
     * @brief Get feature enum information
     *
     * This function retrieves comprehensive information about an enumeration feature parameter on a Tucsen device.
     * The device must be previously opened using TucDeviceOpen() and the feature must be readable
     * before its enumeration information can be retrieved. This function provides detailed metadata about
     * enumeration features including available values, symbolic names, and configuration details.
     *
     * IMPORTANT: Memory Management for Cross-Language Support
     * - Caller MUST pre-allocate the pEntries buffer before calling this function
     * - Use TucGetFeatureEnumEntryCount() to determine the required buffer size
     * - This design ensures compatibility with C, C++, C#, Python, and other languages
     * - No dynamic memory allocation occurs inside the SDK, preventing memory leaks
     *
     * The enumeration information retrieval process includes:
     * - Validating the device handle and feature name parameters
     * - Checking if the feature exists and is readable on the device
     * - Validating that caller has pre-allocated sufficient memory for pEntries
     * - Accessing the feature's enumeration registry to retrieve comprehensive information
     * - Filling the pre-allocated buffer with enumeration metadata
     * - Returning detailed enumeration information for user applications
     *
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @param[in] featureName Name of the feature parameter to query (case-sensitive)
     * @param[in,out] pEnumInfo Pointer to TUC_FEATURE_ENUM_INFO structure
     *                    - pEnumInfo->pEntries MUST be pre-allocated by caller
     *                    - pEnumInfo->entryCount MUST be set to buffer size before call
     *                    - On return, entryCount will contain actual number of entries filled
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if any parameter is NULL,
     *         TUC_STATUS_INVALID_DEVICE_HANDLE if hDevice is invalid,
     *         TUC_STATUS_FEATURE_NOT_FOUND if featureName does not exist,
     *         TUC_STATUS_FEATURE_NOT_READABLE if feature is not readable,
     *         TUC_STATUS_BUFFER_TOO_SMALL if pre-allocated buffer is insufficient,
     *         TUC_STATUS_GET_FEATURE_ENUM_INFO_FAILED on enumeration information retrieval failure
     * @note This function is thread-safe and requires a valid device handle from TucDeviceOpen()
     * @warning Caller MUST pre-allocate pEntries buffer and set entryCount before calling
     * @warning Caller is responsible for freeing the pEntries memory after use
     * @see TucDeviceOpen(), TucGetFeatureEnumEntryCount(), TucGetFeatureEnum(), TucGetFeatureEnumEntryByIndex()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucGetFeatureEnumInfo(IN TUC_HDEVICE hDevice, IN const char *featureName, INOUT TUC_FEATURE_ENUM_INFO *pEnumInfo);

    /**
     * @brief Get feature enum entry count
     *
     * This function retrieves the total number of enumeration entries for a specific feature parameter.
     * The device must be previously opened using TucDeviceOpen() and the feature must be readable
     * before its enumeration entry count can be retrieved. This function provides the count of all
     * detailed enumeration entries including symbolic names, values, and descriptions for features
     * such as pixel format, trigger mode, and acquisition mode.
     *
     * The enumeration entry count retrieval process includes:
     * - Validating the device handle and feature name parameters
     * - Checking if the feature exists and is readable on the device
     * - Accessing the feature's enumeration registry to count detailed entries
     * - Counting all enumeration entries with complete metadata information
     * - Returning the total enumeration entry count for user applications
     *
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @param[in] featureName Name of the feature parameter to query (case-sensitive)
     * @param[out] pEntryCount Pointer to uint32_t variable to receive the enumeration entry count
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if any parameter is NULL,
     *         TUC_STATUS_INVALID_DEVICE_HANDLE if hDevice is invalid,
     *         TUC_STATUS_FEATURE_NOT_FOUND if featureName does not exist,
     *         TUC_STATUS_FEATURE_NOT_READABLE if feature is not readable,
     *         TUC_STATUS_GET_FEATURE_ENUM_ENTRY_COUNT_FAILED on enumeration entry count retrieval failure
     * @note This function is thread-safe and requires a valid device handle from TucDeviceOpen()
     * @warning Must call TucDeviceOpen() first and ensure feature is readable
     * @see TucDeviceOpen(), TucGetFeatureEnumCount(), TucGetFeatureEnumInfo(), TucGetFeatureEnumEntryByIndex()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucGetFeatureEnumEntryCount(IN TUC_HDEVICE hDevice, IN const char *featureName, OUT uint32_t *pEntryCount);

    /**
     * @brief Get feature enum entry by index
     *
     * This function retrieves detailed information about a specific enumeration entry by its index.
     * The device must be previously opened using TucDeviceOpen() and the feature must be readable
     * before its enumeration entry information can be retrieved. This function provides comprehensive
     * details about enumeration entries including symbolic names, values, descriptions, and metadata
     * for features such as pixel format, trigger mode, and acquisition mode.
     *
     * The enumeration entry retrieval process includes:
     * - Validating the device handle and feature name parameters
     * - Checking if the feature exists and is readable on the device
     * - Validating the index parameter against available enumeration entry count
     * - Accessing the feature's enumeration registry to retrieve the specific entry
     * - Gathering complete entry metadata including symbolic name, value, and description
     * - Filling the enumeration entry structure with available data
     * - Returning detailed enumeration entry information for user applications
     *
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @param[in] featureName Name of the feature parameter to query (case-sensitive)
     * @param[in] index Index of the enumeration entry (0-based, must be less than entry count)
     * @param[out] pEntry Pointer to TUC_FEATURE_ENUM_ENTRY structure to receive entry information
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if any parameter is NULL,
     *         TUC_STATUS_INVALID_DEVICE_HANDLE if hDevice is invalid,
     *         TUC_STATUS_FEATURE_NOT_FOUND if featureName does not exist,
     *         TUC_STATUS_FEATURE_NOT_READABLE if feature is not readable,
     *         TUC_STATUS_INVALID_ENUM_ENTRY_INDEX if index is out of range,
     *         TUC_STATUS_GET_FEATURE_ENUM_ENTRY_FAILED on enumeration entry retrieval failure
     * @note This function is thread-safe and requires a valid device handle from TucDeviceOpen()
     * @warning Must call TucDeviceOpen() first and ensure feature is readable
     * @see TucDeviceOpen(), TucGetFeatureEnumEntryCount(), TucGetFeatureEnumInfo(), TucGetFeatureEnum()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucGetFeatureEnumEntryByIndex(IN TUC_HDEVICE hDevice, IN const char *featureName, IN uint32_t index, OUT TUC_FEATURE_ENUM_ENTRY *pEntry);

    /**
     * @brief Get feature enum entry by name
     *
     * This function retrieves detailed information about a specific enumeration entry by its symbolic name.
     * The device must be previously opened using TucDeviceOpen() and the feature must be readable
     * before its enumeration entry information can be retrieved. This function provides comprehensive
     * details about enumeration entries including symbolic names, values, descriptions, and metadata
     * for features such as pixel format, trigger mode, and acquisition mode.
     *
     * The enumeration entry retrieval process includes:
     * - Validating the device handle and feature name parameters
     * - Checking if the feature exists and is readable on the device
     * - Searching the feature's enumeration registry for the specified entry name
     * - Locating the matching enumeration entry by symbolic name
     * - Gathering complete entry metadata including symbolic name, value, and description
     * - Filling the enumeration entry structure with available data
     * - Returning detailed enumeration entry information for user applications
     *
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @param[in] featureName Name of the feature parameter to query (case-sensitive)
     * @param[in] entryName Symbolic name of the enumeration entry to find (case-sensitive)
     * @param[out] pEntry Pointer to TUC_FEATURE_ENUM_ENTRY structure to receive entry information
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if any parameter is NULL,
     *         TUC_STATUS_INVALID_DEVICE_HANDLE if hDevice is invalid,
     *         TUC_STATUS_FEATURE_NOT_FOUND if featureName does not exist,
     *         TUC_STATUS_FEATURE_NOT_READABLE if feature is not readable,
     *         TUC_STATUS_ENUM_ENTRY_NOT_FOUND if entryName does not exist,
     *         TUC_STATUS_GET_FEATURE_ENUM_ENTRY_FAILED on enumeration entry retrieval failure
     * @note This function is thread-safe and requires a valid device handle from TucDeviceOpen()
     * @warning Must call TucDeviceOpen() first and ensure feature is readable
     * @see TucDeviceOpen(), TucGetFeatureEnumEntryCount(), TucGetFeatureEnumInfo(), TucGetFeatureEnumEntryByIndex()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucGetFeatureEnumEntryByName(IN TUC_HDEVICE hDevice, IN const char *featureName, IN const char *entryName, OUT TUC_FEATURE_ENUM_ENTRY *pEntry);

    /**
     * @brief Execute feature command
     *
     * This function executes a command feature on a Tucsen device.
     * The device must be previously opened using TucDeviceOpen() and the feature must be executable
     * before the command can be executed. This function supports command features such as device reset,
     * firmware update, calibration, and other action-based camera operations.
     *
     * The feature command execution process includes:
     * - Validating the device handle and feature name parameters
     * - Checking if the feature exists and is executable on the device
     * - Verifying the device is in a state suitable for command execution
     * - Executing the specified command on the device
     * - Monitoring command execution progress and completion
     * - Confirming the command was successfully executed
     *
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @param[in] featureName Name of the command feature to execute (case-sensitive)
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if featureName is NULL,
     *         TUC_STATUS_INVALID_DEVICE_HANDLE if hDevice is invalid,
     *         TUC_STATUS_FEATURE_NOT_FOUND if featureName does not exist,
     *         TUC_STATUS_FEATURE_NOT_EXECUTABLE if feature is not a command,
     *         TUC_STATUS_DEVICE_BUSY if device is in busy state,
     *         TUC_STATUS_EXECUTE_FEATURE_COMMAND_FAILED on command execution failure
     * @note This function is thread-safe and requires a valid device handle from TucDeviceOpen()
     * @warning Must call TucDeviceOpen() first and ensure device is in suitable state
     * @see TucDeviceOpen(), TucGetFeatureInfo(), TucIsFeatureReadable(), TucIsFeatureWritable()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucExecuteFeatureCommand(IN TUC_HDEVICE hDevice, IN const char *featureName);

    /**
     * @brief Query whether a command feature has finished (GenICam \c ICommand::IsDone with Verify=true)
     *
     * This function does not call Execute; use \c TucExecuteFeatureCommand() first and poll completion
     * from your application as needed.
     *
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @param[in] featureName Name of the command feature (case-sensitive)
     * @param[out] pDone Set to true if the command reports done, false if still in progress
     * @return TUC_STATUS_OK on success,
     *         TUC_STATUS_INVALID_PARAM if any pointer argument is NULL,
     *         TUC_STATUS_INVALID_DEVICE_HANDLE if hDevice is invalid,
     *         TUC_STATUS_DEVICE_NOT_OPENED if the device is not opened,
     *         TUC_STATUS_FEATURE_NOT_SUPPORTED if the feature stack is unavailable,
     *         TUC_STATUS_FEATURE_NOT_FOUND if featureName does not exist,
     *         TUC_STATUS_FEATURE_TYPE_MISMATCH if the feature is not a command,
     *         TUC_STATUS_INTERNAL_ERROR on unexpected failure
     * @note Thread-safe; completion polling and timeouts are the caller's responsibility
     * @see TucExecuteFeatureCommand()
     * @since v3.1.0
     */
    TUC_API TUC_STATUS TucFeatureCommandIsDone(IN TUC_HDEVICE hDevice, IN const char *featureName, OUT bool *pDone);

    // =============================================================================
    // 7. Image Acquisition Management - Start, stop, and abort image acquisition
    // =============================================================================

    /**
     * @brief Start image acquisition
     *
     * This function initiates image acquisition on a Tucsen device with specified acquisition parameters.
     * The device must be previously opened using TucDeviceOpen() and have sufficient buffers announced 
     * before starting acquisition. This function supports both finite frame count acquisition and continuous 
     * streaming modes with configurable start flags for different acquisition behaviors.
     *
     * The acquisition start process includes:
     * - Validating the device handle and acquisition parameters
     * - Checking if sufficient buffers have been announced for acquisition
     * - Verifying the device is in a state suitable for starting acquisition
     * - Configuring acquisition parameters and start flags on the device
     * - Initiating the image acquisition pipeline and data streaming
     * - Monitoring acquisition startup progress and confirming successful start
     *
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @param[in] flags Acquisition start flags controlling acquisition behavior (TUC_ACQ_START_FLAGS_DEFAULT for normal operation)
     * @param[in] frameCount Number of frames to acquire (0 for continuous acquisition, >0 for finite frame count)
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if parameters are invalid,
     *         TUC_STATUS_INVALID_DEVICE_HANDLE if hDevice is invalid,
     *         TUC_STATUS_DEVICE_NOT_OPENED if device is not opened,
     *         TUC_STATUS_BUFFER_ANNOUNCED_LESS if insufficient buffers announced,
     *         TUC_STATUS_START_ACQUISITION_FAILED if acquisition start fails,
     *         TUC_STATUS_DEVICE_BUSY if device is already acquiring
     * @note This function is thread-safe and requires a valid device handle from TucDeviceOpen()
     * @warning Must call TucDeviceOpen() first and announce sufficient buffers before starting acquisition
     * @see TucDeviceOpen(), TucBuffersAllocAndAnnounce(), TucBufferAnnounce(), TucAcquisitionStop(), TucAcquisitionWait()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucAcquisitionStart(IN TUC_HDEVICE hDevice, IN TUC_ACQ_START_FLAGS flags, IN uint64_t frameCount);

    /**
     * @brief Stop image acquisition
     *
     * This function stops the active image acquisition on a Tucsen device with configurable stop behavior.
     * The device must have active acquisition started by TucAcquisitionStart() before stopping can be performed.
     * This function provides controlled termination of the acquisition pipeline with options for graceful or
     * immediate stop modes depending on application requirements.
     *
     * The acquisition stop process includes:
     * - Validating the device handle and stop flag parameters
     * - Checking if acquisition is currently active on the device
     * - Applying the specified stop behavior (graceful or immediate)
     * - Terminating the image acquisition pipeline safely
     * - Flushing remaining buffers and cleaning up acquisition resources
     * - Confirming acquisition has been completely stopped
     *
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @param[in] flags Stop flags controlling stop behavior (TUC_ACQ_STOP_FLAGS_DEFAULT for graceful stop, TUC_ACQ_STOP_FLAGS_KILL for immediate stop)
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if flags are invalid,
     *         TUC_STATUS_INVALID_DEVICE_HANDLE if hDevice is invalid,
     *         TUC_STATUS_DEVICE_NOT_OPENED if device is not opened,
     *         TUC_STATUS_ACQUISITION_NOT_ACTIVE if acquisition is not running,
     *         TUC_STATUS_STOP_ACQUISITION_FAILED if acquisition stop fails
     * @note This function is thread-safe and requires a valid device handle from TucDeviceOpen()
     * @warning Must call TucAcquisitionStart() first before stopping acquisition
     * @see TucDeviceOpen(), TucAcquisitionStart(), TucAcquisitionAbort(), TucBufferFlushQueue()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucAcquisitionStop(IN TUC_HDEVICE hDevice, IN TUC_ACQ_STOP_FLAGS flags);

    /**
     * @brief Abort image acquisition
     *
     * This function immediately aborts the active image acquisition on a Tucsen device without graceful cleanup.
     * The device must have active acquisition started by TucAcquisitionStart() before aborting can be performed.
     * Unlike TucAcquisitionStop(), this function performs an emergency stop that immediately terminates all
     * acquisition operations, potentially leaving buffers in incomplete states for urgent situations.
     *
     * The acquisition abort process includes:
     * - Validating the device handle parameter
     * - Checking if acquisition is currently active on the device
     * - Immediately terminating all ongoing acquisition operations
     * - Forcibly stopping the image acquisition pipeline
     * - Abandoning incomplete buffers and pending operations
     * - Confirming acquisition has been forcibly terminated
     *
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_DEVICE_HANDLE if hDevice is invalid,
     *         TUC_STATUS_DEVICE_NOT_OPENED if device is not opened,
     *         TUC_STATUS_ACQUISITION_NOT_ACTIVE if acquisition is not running,
     *         TUC_STATUS_ABORT_ACQUISITION_FAILED if acquisition abort fails
     * @note This function is thread-safe and requires a valid device handle from TucDeviceOpen()
     * @warning This is an emergency stop function that may leave buffers in incomplete states. Use TucAcquisitionStop() for normal termination.
     * @see TucDeviceOpen(), TucAcquisitionStart(), TucAcquisitionStop(), TucBufferFlushQueue()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucAcquisitionAbort(IN TUC_HDEVICE hDevice);

    /**
     * @brief Wait for image acquisition completion
     *
     * This function waits for an acquisition to complete and retrieves complete image information.
     * It blocks until a frame is acquired or the specified timeout expires. The returned
     * image information includes complete metadata and processed image data from ISP pipeline.
     *
     * @param hDevice [IN] Device handle
     * @param pImageInfo [OUT] Pointer to image information structure
     * @param timeoutMs [IN] Timeout in milliseconds
     * @return TUC_STATUS_OK on success, error code on failure or timeout
     * @note The returned image information contains processed image data with complete ISP processing
     * @since v3.0.0 - Changed from TUC_BUFFER_INFO to TUC_IMAGE_INFO for enhanced functionality
     */
    TUC_API TUC_STATUS TucAcquisitionWait(IN TUC_HDEVICE hDevice, OUT TUC_IMAGE_INFO *pImageInfo, IN uint32_t timeoutMs);

    /**
     * @brief Get acquisition statistics
     *
     * This function retrieves comprehensive acquisition performance statistics from a Tucsen device.
     * The device can be queried for statistics at any time during or after acquisition operations
     * to monitor buffer utilization, delivery performance, and detect potential acquisition issues
     * such as buffer underruns or incomplete frame deliveries.
     *
     * The statistics retrieval process includes:
     * - Validating the device handle and statistics structure parameters
     * - Checking if the device supports acquisition statistics reporting
     * - Querying the current acquisition pipeline status and counters
     * - Retrieving buffer management statistics and performance metrics
     * - Calculating delivery rates and error statistics
     * - Populating the statistics structure with current values
     *
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @param[out] pStatistics Pointer to TUC_ACQUISITION_STATISTICS structure to receive statistics data
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if pStatistics is NULL,
     *         TUC_STATUS_INVALID_DEVICE_HANDLE if hDevice is invalid,
     *         TUC_STATUS_DEVICE_NOT_OPENED if device is not opened,
     *         TUC_STATUS_FEATURE_NOT_SUPPORTED if device does not support statistics,
     *         TUC_STATUS_GET_STATISTICS_FAILED if statistics retrieval fails
     * @note This function is thread-safe and requires a valid device handle from TucDeviceOpen()
     * @warning The statistics structure must be valid and have sufficient size to receive all statistics data
     * @see TucDeviceOpen(), TucAcquisitionStart(), TucAcquisitionStop(), TUC_ACQUISITION_STATISTICS
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucAcquisitionGetStatistics(IN TUC_HDEVICE hDevice, OUT TUC_ACQUISITION_STATISTICS *pStatistics);

    // =============================================================================
    // 8. Buffer Management - Queue, allocate, and revoke buffer pool
    // =============================================================================

    /**
     * @brief Add buffer to queue
     *
     * This function adds a buffer to the device's acquisition queue for image data reception.
     * The buffer must be previously announced using either TucBuffersAllocAndAnnounce() or
     * TucBufferAnnounce(). Buffers in the queue are used by the device to store incoming
     * image data during acquisition.
     *
     * The buffer queuing process includes:
     * - Validating the device handle and buffer handle parameters
     * - Verifying the buffer is properly announced and not already in queue
     * - Adding the buffer to the device's acquisition queue
     * - Making the buffer available for image data reception
     *
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @param[in] hBuffer Buffer handle from TucBuffersAllocAndAnnounce() or TucBufferAnnounce()
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if parameters are invalid,
     *         TUC_STATUS_INVALID_DEVICE_HANDLE if hDevice is invalid,
     *         TUC_STATUS_INVALID_OPERATION if buffer handle is invalid, buffer already in queue,
     *         or device is not in manual buffer queue policy mode
     * @note This function is thread-safe and requires a valid device handle from TucDeviceOpen().
     *       Only available when buffer queue policy is set to TUC_BUFFER_QUEUE_POLICY_MANUAL.
     * @warning Must call TucDeviceOpen() and TucBufferSetQueuePolicy(MANUAL) first. 
     *          The buffer must be properly announced before queuing.
     * @see TucDeviceOpen(), TucBufferSetQueuePolicy(), TucBuffersAllocAndAnnounce(), TucBufferAnnounce()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucBufferQueue(IN TUC_HDEVICE hDevice, IN TUC_HBUFFER hBuffer);

    /**
     * @brief Allocate and announce buffer pool
     *
     * This function allocates and announces a pool of image buffers for the device.
     * The buffer pool is managed internally by the SDK and provides efficient memory
     * management for high-performance image acquisition. This function should be called
     * before starting image acquisition to ensure sufficient buffer resources.
     *
     * The buffer pool allocation process includes:
     * - Validating the device handle and buffer count parameters
     * - Allocating memory for the specified number of buffers
     * - Announcing buffers to the device's acquisition pipeline
     * - Returning a pool handle for subsequent buffer management operations
     *
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @param[in] bufferCount Number of buffers to allocate (recommended: 4-16 for optimal performance)
     * @param[out] pPoolHandle Pointer to receive the buffer pool handle
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if parameters are invalid,
     *         TUC_STATUS_INVALID_DEVICE_HANDLE if hDevice is invalid,
     *         TUC_STATUS_OUT_OF_MEMORY if insufficient memory,
     *         TUC_STATUS_INVALID_OPERATION if buffer pool already exists
     * @note This function is thread-safe and requires a valid device handle from TucDeviceOpen()
     * @warning Must call TucDeviceOpen() first. Cannot be used simultaneously with TucBufferAnnounce.
     *          Use either buffer pool management (this function) or user buffer announcement (TucBufferAnnounce).
     * @see TucDeviceOpen(), TucBuffersRevokeAndDestroy(), TucAcquisitionStart()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucBuffersAllocAndAnnounce(IN TUC_HDEVICE hDevice, IN uint32_t bufferCount, OUT TUC_HBUFFERS *pPoolHandle);

    /**
     * @brief Revoke and destroy buffer pool
     *
     * This function revokes and destroys a previously allocated buffer pool and all its buffers.
     * The function releases all memory associated with the buffer pool and invalidates the pool handle.
     * This function should be called when the buffer pool is no longer needed or before switching
     * to user buffer announcement mode.
     *
     * The buffer pool revocation process includes:
     * - Validating the device handle and pool handle parameters
     * - Stopping any ongoing acquisition that uses the buffer pool
     * - Revoking all buffers from the device's acquisition pipeline
     * - Releasing all allocated memory for the buffers
     * - Invalidating the pool handle
     *
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @param[in] poolHandle Buffer pool handle from TucBuffersAllocAndAnnounce()
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if parameters are invalid,
     *         TUC_STATUS_INVALID_DEVICE_HANDLE if hDevice is invalid,
     *         TUC_STATUS_INVALID_OPERATION if pool handle is invalid or acquisition is active
     * @note This function is thread-safe and requires a valid device handle from TucDeviceOpen()
     * @warning Must call TucDeviceOpen() first. The pool handle becomes invalid after this call.
     *          Call TucBuffersAllocAndAnnounce() again if buffer pool is needed later.
     * @see TucDeviceOpen(), TucBuffersAllocAndAnnounce(), TucAcquisitionStop()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucBuffersRevokeAndDestroy(IN TUC_HDEVICE hDevice, IN TUC_HBUFFERS poolHandle);

    /**
     * @brief Announce user buffer
     *
     * This function announces a user-provided memory buffer to the device for image acquisition.
     * The user buffer is registered with the SDK and can be used for receiving image data
     * during acquisition. This approach provides flexibility for applications that need
     * to manage their own memory allocation or integrate with existing memory pools.
     *
     * The user buffer announcement process includes:
     * - Validating the device handle and buffer parameters
     * - Verifying the user buffer is accessible and has sufficient size
     * - Registering the buffer with the device's acquisition pipeline
     * - Returning a buffer handle for subsequent buffer operations
     *
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @param[in] pUserBuffer Pointer to user-allocated memory buffer
     * @param[in] bufferSize Size of the user buffer in bytes
     * @param[out] phBuffer Pointer to receive the buffer handle
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if parameters are invalid,
     *         TUC_STATUS_INVALID_DEVICE_HANDLE if hDevice is invalid,
     *         TUC_STATUS_BUFFER_TOO_SMALL if buffer size is insufficient,
     *         TUC_STATUS_INVALID_OPERATION if buffer pool already exists
     * @note This function is thread-safe and requires a valid device handle from TucDeviceOpen()
     * @warning Must call TucDeviceOpen() first. Cannot be used simultaneously with TucBuffersAllocAndAnnounce.
     *          Use either user buffer announcement (this function) or buffer pool management (TucBuffersAllocAndAnnounce).
     *          The user is responsible for managing the lifetime of the provided buffer.
     * @see TucDeviceOpen(), TucBufferRevoke(), TucBufferQueue(), TucAcquisitionStart()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucBufferAnnounce(IN TUC_HDEVICE hDevice, IN void *pUserBuffer, IN uint32_t bufferSize, OUT TUC_HBUFFER *phBuffer); 

    /**
     * @brief Revoke buffer announcement
     *
     * This function revokes a previously announced user buffer from the device.
     * The buffer is removed from the device's acquisition pipeline and the buffer handle
     * becomes invalid. This function should be called when the user buffer is no longer
     * needed or before switching to buffer pool management mode.
     *
     * The buffer revocation process includes:
     * - Validating the device handle and buffer handle parameters
     * - Removing the buffer from the device's acquisition pipeline
     * - Invalidating the buffer handle
     * - Cleaning up internal buffer registration
     *
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @param[in] hBuffer Buffer handle from TucBufferAnnounce()
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if parameters are invalid,
     *         TUC_STATUS_INVALID_DEVICE_HANDLE if hDevice is invalid,
     *         TUC_STATUS_INVALID_OPERATION if buffer handle is invalid or acquisition is active
     * @note This function is thread-safe and requires a valid device handle from TucDeviceOpen()
     * @warning Must call TucDeviceOpen() first. The buffer handle becomes invalid after this call.
     *          Call TucBufferAnnounce() again if the same buffer is needed later.
     * @see TucDeviceOpen(), TucBufferAnnounce(), TucAcquisitionStop()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucBufferRevoke(IN TUC_HDEVICE hDevice, IN TUC_HBUFFER hBuffer);

    /**
     * @brief Get buffer information
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @param[in] hBuffer Buffer handle obtained from TucBuffersAllocAndAnnounce() or TucBufferGetByID()
     * @param[out] pBufferInfo Pointer to TUC_BUFFER_INFO structure to receive buffer information
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if parameters are invalid,
     *         TUC_STATUS_INVALID_DEVICE_HANDLE if hDevice is invalid,
     *         TUC_STATUS_INVALID_BUFFER_HANDLE if hBuffer is invalid
     * @note This function retrieves comprehensive buffer information including dimensions, format, state, and metadata
     * @warning Must call TucDeviceOpen() and TucBuffersAllocAndAnnounce() first
     * @see TucDeviceOpen(), TucBuffersAllocAndAnnounce(), TucBufferGetByID(), TUC_BUFFER_INFO
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucBufferGetInfo(IN TUC_HDEVICE hDevice, IN TUC_HBUFFER hBuffer, OUT TUC_BUFFER_INFO *pBufferInfo);

    /**
     * @brief Get buffer by ID
     * @param hDevice [IN] Device handle
     * @param index [IN] Buffer index
     * @param phBuffer [OUT] Pointer to buffer handle
     * @return TUC_STATUS_OK on success, error code on failure
     */
    TUC_API TUC_STATUS TucBufferGetByID(IN TUC_HDEVICE hDevice, IN uint32_t index, OUT TUC_HBUFFER *phBuffer);

    /**
     * @brief Flush buffer queue
     *
     * This function flushes the buffer queue according to the specified operation type.
     * The flush operation allows applications to control the flow of buffers between
     * the input pool and output queue, or to discard buffers as needed.
     *
     * The flush operation types include:
     * - TUC_ACQ_QUEUE_TYPE_INPUT_TO_OUTPUT: Flushes input pool to output queue
     * - TUC_ACQ_QUEUE_TYPE_OUTPUT_DISCARD: Discards all buffers in output queue
     * - TUC_ACQ_QUEUE_TYPE_ALL_TO_INPUT: Puts all buffers back to input pool
     * - TUC_ACQ_QUEUE_TYPE_UNQUEUED_TO_INPUT: Puts unqueued buffers in input pool
     * - TUC_ACQ_QUEUE_TYPE_ALL_DISCARD: Discards all buffers in input pool and output queue
     *
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @param[in] operation Queue operation type defining the flush behavior
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if parameters are invalid,
     *         TUC_STATUS_INVALID_DEVICE_HANDLE if hDevice is invalid,
     *         TUC_STATUS_INVALID_OPERATION if flush operation fails
     * @note This function is thread-safe and requires a valid device handle from TucDeviceOpen()
     * @warning Must call TucDeviceOpen() first. The operation affects the current buffer state
     *          and may impact ongoing acquisition if buffers are being actively used.
     * @see TucDeviceOpen(), TucBuffersAllocAndAnnounce(), TucBufferQueue(), TucAcquisitionStart()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucBufferFlushQueue(IN TUC_HDEVICE hDevice, IN TUC_ACQ_QUEUE_TYPE operation);


    // =============================================================================
    // 9.Buffer Advanced Operations - Data processing, conversion, and file I/O
    // =============================================================================

    /**
     * @brief Read pixel component values at (posX, posY) from the current ISP pipeline sink image.
     *
     * Coordinates are 0-based in the processed image (after ROI / transform / binning as applied
     * by the pipeline). Supports interleaved Mono8/Mono16 and multi-channel 8/16-bit formats up to
     * TUC_PIXEL_VALUES_MAX_CH channels. Values are returned in memory channel order for the reported PFNC.
     *
     * @param[in] hDevice Device handle from TucDeviceOpen()
     * @param[in] posX Horizontal pixel index (0 .. width-1)
     * @param[in] posY Vertical pixel index (0 .. height-1)
     * @param[out] pValues Filled with pfnc, channelCount, and v[] (unused entries zeroed)
     * @return TUC_STATUS_OK on success,
     *         TUC_STATUS_INVALID_PARAM if pValues is NULL,
     *         TUC_STATUS_NOT_READY if pipeline or image data is unavailable,
     *         TUC_STATUS_INDEX_OUT_OF_RANGE if coordinates are out of bounds,
     *         TUC_STATUS_FEATURE_NOT_SUPPORTED if pixel layout is not supported (e.g. packed >16bpp),
     *         TUC_STATUS_INVALID_HANDLE if hDevice is invalid
     * @note Requires at least one processed frame in the pipeline (e.g. after acquisition delivers data).
     * @see TucBufferGetInfo(), TUC_PIXEL_VALUES, TUC_PIXEL_VALUES_MAX_CH
     * @since v3.0.0
     */
    TUC_API TUC_STATUS TucGetPixelValuesAt(
        IN TUC_HDEVICE hDevice,
        IN int32_t posX,
        IN int32_t posY,
        OUT TUC_PIXEL_VALUES *pValues);

    /**
     * @brief Copy buffer data to memory
     *
     * This function copies a rectangular region of image data from a buffer to a destination memory location.
     * The function supports region-based copying with specified offset and dimensions, automatically
     * detects pixel format to calculate correct byte counts, and performs optimized memory operations.
     *
     * The copy operation includes:
     * - Region-based copying with X/Y offset support
     * - Automatic pixel format detection and byte calculation
     * - Optimized memory copying with pointer arithmetic
     * - Boundary validation and error checking
     * - Support for various pixel formats (Mono, RGB, RGBA, Bayer, YUV, YCbCr)
     *
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @param[in] hSrc Source buffer handle obtained from TucBuffersAllocAndAnnounce() or TucBufferGetByID()
     * @param[in] srcOffsetX Source X offset for the copy region (0-based)
     * @param[in] srcOffsetY Source Y offset for the copy region (0-based)
     * @param[in] width Width of the region to copy (must be > 0)
     * @param[in] height Height of the region to copy (must be > 0)
     * @param[out] pDstBuffer Pointer to destination memory buffer for the copied data
     * @param[in] dstBufferSize Size of the destination buffer in bytes
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if parameters are invalid,
     *         TUC_STATUS_INVALID_DEVICE_HANDLE if hDevice is invalid,
     *         TUC_STATUS_INVALID_BUFFER_HANDLE if hSrc is invalid,
     *         TUC_STATUS_BUFFER_TOO_SMALL if destination buffer is insufficient,
     *         TUC_STATUS_FEATURE_NOT_SUPPORTED if pixel format is not supported,
     *         TUC_STATUS_INVALID_OPERATION if buffer has no image data
     * @note This function is thread-safe and requires a valid device handle from TucDeviceOpen().
     *       The function automatically calculates required memory size based on pixel format.
     *       For optimal performance, ensure destination buffer size matches the calculated requirement.
     * @warning Must call TucDeviceOpen() and TucBuffersAllocAndAnnounce() first. The source buffer
     *          must contain valid image data (imagePresent flag must be true). The copy region
     *          must not exceed the buffer's image dimensions.
     * @see TucDeviceOpen(), TucBuffersAllocAndAnnounce(), TucBufferGetByID(), TucBufferGetInfo()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucBufferCopyToMemory(
        IN TUC_HDEVICE hDevice,
        IN TUC_HBUFFER hSrc,
        IN uint32_t srcOffsetX,
        IN uint32_t srcOffsetY,
        IN uint32_t width,
        IN uint32_t height,
        OUT void *pDstBuffer,
        IN uint32_t dstBufferSize
    );

    /**
     * @brief Copy buffer data to TUC_BUFFER_INFO structure
     *
     * This function copies image data from a buffer handle to user-provided memory
     * specified in pBufferInfo->pBuffer, and fills complete metadata into the structure.
     * The memory layout should be: TUC_BUFFER_INFO structure followed by raw image data,
     * where pBuffer points to the offset position after the structure.
     *
     * Before calling:
     * - User must allocate memory: sizeof(TUC_BUFFER_INFO) + payloadSize
     * - User must set pBufferInfo->pBuffer to point to (base + sizeof(TUC_BUFFER_INFO))
     * - User must set pBufferInfo->bufferSize to payloadSize
     *
     * After successful call:
     * - Image data is copied to pBufferInfo->pBuffer
     * - All metadata fields are filled (imageWidth, imageHeight, pixelFormat, etc.)
     *
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @param[in] hBuffer Source buffer handle
     * @param[in,out] pBufferInfo Buffer information structure
     *                [IN] pBuffer: pointer to pre-allocated memory (after structure)
     *                [IN] bufferSize: size of allocated data memory (payloadSize)
     *                [OUT] All other fields filled with source buffer metadata
     * @return TUC_STATUS_OK on success,
     *         TUC_STATUS_INVALID_PARAM if pBufferInfo or pBufferInfo->pBuffer is NULL,
     *         TUC_STATUS_BUFFER_TOO_SMALL if bufferSize is insufficient,
     *         error code on other failures
     * @note User is responsible for allocating and freeing memory.
     *       Use TucBufferGetInfo() first to determine required payloadSize.
     * @see TucBufferGetInfo(), TucBufferInfoConvertToImage()
     * @since v1.1.0
     */
    TUC_API TUC_STATUS TucBufferCopyToBufferInfo(
        IN TUC_HDEVICE hDevice,
        IN TUC_HBUFFER hBuffer,
        INOUT TUC_BUFFER_INFO *pBufferInfo
    );

    /**
     * @brief Create buffer conversion pipeline for the device
     *
     * This function creates the buffer conversion (ISP) pipeline for the given device.
     * The pipeline is used by TucBufferConvertToImage() to convert raw buffer data
     * to processed image. Must be called before using buffer conversion features.
     *
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if hDevice is null,
     *         TUC_STATUS_NOT_INITIALIZED if SDK not initialized,
     *         TUC_STATUS_INVALID_DEVICE_HANDLE if device handle is invalid
     * @note Thread-safe. Call TucBufferConverterDestroy() when conversion is no longer needed.
     * @see TucBufferConverterDestroy(), TucBufferConvertToImage()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucBufferConverterCreate(IN TUC_HDEVICE hDevice);

    /**
     * @brief Destroy buffer conversion pipeline for the device
     *
     * This function destroys the buffer conversion pipeline previously created by
     * TucBufferConverterCreate(). Frees resources associated with the pipeline.
     *
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if hDevice is null,
     *         TUC_STATUS_NOT_INITIALIZED if SDK not initialized,
     *         TUC_STATUS_INVALID_DEVICE_HANDLE if device handle is invalid
     * @note Thread-safe. Safe to call if pipeline was not created.
     * @see TucBufferConverterCreate(), TucBufferConvertToImage()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucBufferConverterDestroy(IN TUC_HDEVICE hDevice);

    /**
     * @brief Convert buffer data to processed image
     * @param hDevice [IN] Device handle
     * @param hBuffer [IN] Buffer handle containing raw data
     * @param conversionPolicy [IN] Conversion policy
     * @param pImageInfo [OUT] Image information
     * @return TUC_STATUS_OK on success, error code on failure
     * @details This function converts raw buffer data to processed image using the device's
     *          ISP pipeline. The converted image can be retrieved using TucBufferGetInfo()
     *          and TucBufferCopyToMemory() after successful conversion.
     * @note The buffer must contain valid image data (imagePresent flag must be true).
     *       This function modifies the buffer contents by applying ISP processing.
     * @see TucDeviceOpen(), TucBuffersAllocAndAnnounce(), TucBufferGetByID()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucBufferConvertToImage(IN TUC_HDEVICE hDevice, IN TUC_HBUFFER hBuffer, IN TUC_CONVERSION_POLICY conversionPolicy, OUT TUC_IMAGE_INFO *pImageInfo);

    /**
     * @brief Convert TUC_BUFFER_INFO to TUC_IMAGE_INFO
     *
     * This function converts raw image data described by TUC_BUFFER_INFO to a processed
     * image using the device's ISP pipeline. The output image data memory is managed
     * internally by SDK.
     *
     * Input requirements:
     * - pBufferInfo->pBuffer must point to valid raw image data
     * - pBufferInfo->imageWidth, imageHeight, pixelFormat must be valid
     *
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @param[in] pBufferInfo Source buffer information with valid raw data
     * @param[in] conversionPolicy Conversion policy (BASIC or FULL ISP processing)
     * @param[out] pImageInfo Output image information (data pointer managed by SDK)
     * @return TUC_STATUS_OK on success, error code on failure
     * @note The pImageInfo->data is managed by SDK and valid until:
     *       - Next call to this function with same device
     *       - Device is closed via TucDeviceClose()
     *       Do NOT free pImageInfo->data manually.
     * @see TucBufferCopyToBufferInfo(), TucBufferConvertToImage()
     * @since v1.1.0
     */
    TUC_API TUC_STATUS TucBufferInfoConvertToImage(
        IN TUC_HDEVICE hDevice,
        IN const TUC_BUFFER_INFO *pBufferInfo,
        IN TUC_CONVERSION_POLICY conversionPolicy,
        OUT TUC_IMAGE_INFO *pImageInfo
    );

    /**
     * @brief Save buffer data to file
     * 
     * This function saves the contents of a buffer to a file on disk. The buffer must contain
     * valid image data that was previously acquired from the device. The save format and options
     * are controlled through the TUC_SAVE_OPTIONS structure.
     *
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @param[in] hBuffer Buffer handle containing the image data to save
     * @param[in] filename Output filename with full path (e.g., "C:/images/capture.bmp")
     * @param[in] pOptions Pointer to save options structure defining format and quality
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if parameters are invalid,
     *         TUC_STATUS_INVALID_DEVICE_HANDLE if hDevice is invalid,
     *         TUC_STATUS_INVALID_BUFFER_HANDLE if hBuffer is invalid,
     *         TUC_STATUS_FILE_IO_ERROR if file cannot be created or written
     * @note This function is thread-safe and requires valid device and buffer handles.
     *       The buffer must contain valid image data before calling this function.
     * @warning Ensure sufficient disk space and write permissions for the target directory.
     * @see TUC_SAVE_OPTIONS, TucBufferQueue(), TucBufferDequeue()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucBufferSaveToFile(
        IN TUC_HDEVICE hDevice,
        IN TUC_HBUFFER hBuffer,
        IN const char *filename,
        IN TUC_SAVE_OPTIONS *pOptions
    );

    /**
     * @brief Save image data to file with automatic format detection
     * 
     * This function saves image data to a file on disk using the ImageSave implementation.
     * The function automatically determines the appropriate file extension based on the save
     * format specified in the options. This provides a convenient way to save images without
     * worrying about file extensions.
     *
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @param[in] pImageInfo Pointer to image information structure containing the image data
     * @param[in] filename Output filename without extension (e.g., "C:/images/capture")
     * @param[in] pOptions Pointer to save options structure defining format, quality, and metadata
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if parameters are invalid,
     *         TUC_STATUS_INVALID_DEVICE_HANDLE if hDevice is invalid,
     *         TUC_STATUS_INVALID_IMAGE_DATA if pImageInfo contains invalid data,
     *         TUC_STATUS_FILE_IO_ERROR if file cannot be created or written
     * @note This function is thread-safe and requires a valid device handle.
     *       The filename should not include an extension as it will be automatically added.
     *       Supported formats include BMP, JPEG, PNG, TIFF based on TUC_SAVE_OPTIONS.
     * @warning Ensure sufficient disk space and write permissions for the target directory.
     *          The image data must be valid and accessible through the TUC_IMAGE_INFO structure.
     *          Note: Do NOT free the memory in imageInfo.data, imageInfo.chunk, or imageInfo.histogram
     *          as these are managed internally by the SDK.
     * @see TUC_IMAGE_INFO, TUC_SAVE_OPTIONS, TucImageLoadFromFile(), TucImageUnload()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucImageSaveToFile(
        IN TUC_HDEVICE hDevice,
        IN TUC_IMAGE_INFO *pImageInfo,
        IN const char *filename,
        IN TUC_SAVE_OPTIONS *pOptions
    );

    /**
     * @brief Create a high-speed storage writer (Modules/Storage: RAW/TIFF, aligned I/O)
     *
     * At most one active writer per open device. Destroy the current writer with TucStorageWriterDestroy()
     * before creating another.
     *
     * @param[in] hDevice Valid device handle from TucDeviceOpen()
     * @param[in] outputPath Output directory only (no filename). Files are created under this path; names come from TucStorageWrite().
     * @param[in] pTemplateImageInfo Image template for buffer sizing; width, height, and bytesCount must be valid
     * @param[in] pOptions Storage options; pOptions->version must be TUC_STORAGE_OPTIONS_VERSION (see TUC_STORAGE_OPTIONS).
     *            For TIFF/TIFF_STACK, pOptions->tiffDpiX / tiffDpiY set resolution in dpi (0 uses TUC_TIFF_DPI_DEFAULT, typically 96).
     * @param[out] phStorage Receives the storage writer handle on success
     * @return TUC_STATUS_OK on success, or an error code if parameters are invalid or initialization fails
     * @note Thread-safe with respect to the device lock. Requires an open device.
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucStorageWriterCreate(
        IN TUC_HDEVICE hDevice,
        IN const char *outputPath,
        IN const TUC_IMAGE_INFO *pTemplateImageInfo,
        IN const TUC_STORAGE_OPTIONS *pOptions,
        OUT TUC_HSTORAGE *phStorage);

    /**
     * @brief Destroy a storage writer created with TucStorageWriterCreate()
     * @param[in] hDevice Valid device handle
     * @param[in] hStorage Storage writer handle returned by TucStorageWriterCreate()
     * @return TUC_STATUS_OK on success
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucStorageWriterDestroy(IN TUC_HDEVICE hDevice, IN TUC_HSTORAGE hStorage);

    /**
     * @brief Write using the high-speed storage writer (single file, stack frame, or stack finish)
     *
     * The operation is selected by pWriteOpts->mode (TUC_STORAGE_WRITE_MODE). Stack frame limits use
     * maxStackFrames and maxStackFileMB from the options passed to TucStorageWriterCreate().
     *
     * @param[in] hDevice Valid device handle
     * @param[in] hStorage Storage writer handle
     * @param[in] filename Base file name only (no path), under outputPath from TucStorageWriterCreate(). Same rule as TucImageSaveToFile /
     *                     ImageSave::save: extension may be omitted - the SDK appends ".raw" or ".tif" from pOptions->format. If the name
     *                     already ends with ".raw" (RAW modes) or ".tif"/".tiff" (TIFF modes), case-insensitive, it is left unchanged.
     *                     Required for TUC_STORAGE_WRITE_SINGLE and the first TUC_STORAGE_WRITE_STACK_FRAME; may be NULL for subsequent
     *                     stack frames; ignored for TUC_STORAGE_WRITE_STACK_FINISH.
     * @param[in] pWriteOpts Write options; version must be TUC_STORAGE_WRITE_OPTIONS_VERSION
     * @param[in,out] pImageInfo Frame payload; may be NULL only for TUC_STORAGE_WRITE_STACK_FINISH
     * @return TUC_STATUS_OK on success, or an error code on failure
     * @note Thread-safe with respect to the device lock.
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucStorageWrite(
        IN TUC_HDEVICE hDevice,
        IN TUC_HSTORAGE hStorage,
        IN const char *filename,
        IN const TUC_STORAGE_WRITE_OPTIONS *pWriteOpts,
        IN OUT TUC_IMAGE_INFO *pImageInfo);

    /**
     * @brief Load image data from file with automatic format detection
     *
     * This function loads image data from a file on disk and populates the TUC_IMAGE_INFO structure.
     * The function automatically detects the file format (BMP, JPEG, PNG, TIFF, RAW) and loads
     * the image data accordingly. Pixel memory for pImageInfo->data is allocated by the SDK.
     *
     * RAW (.raw) files must carry a TUCSEN3 frame header (ImageInfo). The header fields
     * (geometry, exposure, gain, timestamps, etc.) are restored into pImageInfo.
     *
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @param[in] filename Input filename with full path (e.g., "C:/images/capture.bmp")
     * @param[out] pImageInfo Pointer to image information structure to receive the loaded data
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if parameters are invalid,
     *         TUC_STATUS_INVALID_DEVICE_HANDLE if hDevice is invalid,
     *         TUC_STATUS_DEVICE_NOT_OPENED if the device is not open,
     *         TUC_STATUS_LOAD_FAILED if the file cannot be decoded
     * @note This function is thread-safe and requires a valid open device handle.
     *       Only the TUC_IMAGE_INFO structure itself needs to be provided by the caller;
     *       pImageInfo->data is allocated by the SDK.
     *       A subsequent Load on the same device automatically frees the previous buffer.
     * @warning Do NOT free pImageInfo->data with free/delete. Call TucImageUnload() when finished,
     *          or close the device with TucDeviceClose() which silently releases any loaded buffer
     *          and invalidates outstanding data pointers.
     * @see TUC_IMAGE_INFO, TucImageSaveToFile(), TucImageUnload(), TucDeviceClose()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucImageLoadFromFile(IN TUC_HDEVICE hDevice, IN const char *filename, OUT TUC_IMAGE_INFO *pImageInfo);

    /**
     * @brief Release pixel memory allocated by TucImageLoadFromFile()
     *
     * Frees the SDK-owned buffer referenced by pImageInfo->data. The pointer must match the
     * buffer currently owned by the device from the most recent successful TucImageLoadFromFile().
     * Pointers from TucBufferInfoConvertToImage() or other sources must not be passed here.
     *
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @param[in,out] pImageInfo Image info returned by TucImageLoadFromFile(); on success
     *                pImageInfo->data is set to NULL and bytesCount to 0
     * @return TUC_STATUS_OK on success,
     *         TUC_STATUS_INVALID_PARAM if pImageInfo is NULL or data does not match the
     *         device-owned Load buffer,
     *         TUC_STATUS_INVALID_DEVICE_HANDLE if hDevice is invalid,
     *         TUC_STATUS_DEVICE_NOT_OPENED if the device is not open
     * @note Thread-safe. If Unload is skipped, TucDeviceClose() silently frees the buffer.
     * @see TucImageLoadFromFile(), TucDeviceClose()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucImageUnload(IN TUC_HDEVICE hDevice, IN OUT TUC_IMAGE_INFO *pImageInfo);

    /**
     * @brief Set buffer queue policy
     * 
     * This function sets the buffer queue management policy for the specified device.
     * The policy determines how buffers are managed during acquisition:
     * - AUTO (0): Automatic queue management - buffers are automatically queued and released
     * - MANUAL (1): Manual queue management - user must explicitly call TucBufferQueue
     *
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @param[in] policy Buffer queue policy (TUC_BUFFER_QUEUE_POLICY_AUTO or TUC_BUFFER_QUEUE_POLICY_MANUAL)
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if parameters are invalid,
     *         TUC_STATUS_INVALID_DEVICE_HANDLE if hDevice is invalid,
     *         TUC_STATUS_DEVICE_NOT_OPENED if device is not opened
     * @note This function is thread-safe and requires a valid device handle from TucDeviceOpen().
     *       Policy validation is performed to ensure only valid policy values are accepted.
     * @warning Must call TucDeviceOpen() first. Policy change takes effect immediately.
     *          Changing policy during active acquisition may affect performance.
     * @see TucDeviceOpen(), TucBufferGetQueuePolicy(), TucBufferQueue()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucBufferSetQueuePolicy(IN TUC_HDEVICE hDevice, IN TUC_BUFFER_QUEUE_POLICY policy);

    /**
     * @brief Get buffer queue policye
     * 
     * This function retrieves the current buffer queue management policy for the specified device.
     *
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @param[out] pPolicy Pointer to receive the current buffer queue policy
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if parameters are invalid,
     *         TUC_STATUS_INVALID_DEVICE_HANDLE if hDevice is invalid,
     *         TUC_STATUS_DEVICE_NOT_OPENED if device is not opened
     * @note This function is thread-safe and requires a valid device handle from TucDeviceOpen()
     * @warning Must call TucDeviceOpen() first
     * @see TucDeviceOpen(), TucBufferSetQueuePolicy(), TucBufferQueue()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucBufferGetQueuePolicy(IN TUC_HDEVICE hDevice, OUT TUC_BUFFER_QUEUE_POLICY *pPolicy);

    // =============================================================================
    // 10. Event Handling - Register and unregister event callback functions
    // =============================================================================

    /**
     * @brief Register buffer callback function
     * 
     * This function registers a callback function to be called when a new buffer becomes available 
     * during image acquisition. The callback provides access to acquired image data through buffer 
     * handles and allows for real-time processing and analysis of captured frames.
     *
     * The buffer callback process includes:
     * - Registering the user-provided callback function with the device
     * - Validating callback function pointer and device handle
     * - Configuring the underlying stream to invoke callbacks on buffer arrival
     * - Providing buffer handle and user context to the callback function
     * - Managing callback lifecycle and proper cleanup during device operations
     *
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @param[in] callback Buffer callback function pointer (cannot be NULL)
     * @param[in] pUserContext User context pointer passed to callback function (can be NULL)
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if parameters are invalid,
     *         TUC_STATUS_INVALID_DEVICE_HANDLE if hDevice is invalid,
     *         TUC_STATUS_DEVICE_NOT_OPENED if device is not opened,
     *         TUC_STATUS_REGISTER_CALLBACK_FAILED if callback registration fails,
     *         TUC_STATUS_CALLBACK_ALREADY_REGISTERED if callback is already registered
     * @note This function is thread-safe and requires a valid device handle from TucDeviceOpen()
     * @warning Must call TucDeviceOpen() first before registering buffer callbacks
     * @see TucDeviceOpen(), TucUnregisterBufferCallback(), TucAcquisitionStart(), TucBuffersAllocAndAnnounce()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucRegisterBufferCallback(IN TUC_HDEVICE hDevice, IN TUC_BUFFER_CALLBACK callback, IN void *pUserContext);

    /**
     * @brief Register event callback function for device events
     *
     * This function registers a callback function to receive notifications when specific events
     * occur on the specified device. The callback function will be invoked asynchronously when
     * the requested event type is detected, providing detailed event information including
     * timestamp, event ID, and additional data.
     *
     * The event callback registration process includes:
     * - Validating the device handle and callback function parameters
     * - Registering the callback for the specified event type
     * - Setting up event monitoring and notification mechanisms
     * - Configuring callback context and user data handling
     * - Enabling asynchronous event notification for the device
     *
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @param[in] eventType Type of event to monitor (see TUC_EVENT_TYPE enumeration)
     * @param[in] callback Event callback function pointer to be invoked on event occurrence
     * @param[in] pUserContext User context pointer passed to callback function (can be NULL)
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if parameters are invalid,
     *         TUC_STATUS_INVALID_DEVICE_HANDLE if hDevice is invalid,
     *         TUC_STATUS_DEVICE_NOT_OPENED if device is not opened,
     *         TUC_STATUS_REGISTER_CALLBACK_FAILED if callback registration fails
     *
     * @note This function is thread-safe and requires a valid device handle from TucDeviceOpen()
     * @note Provides enhanced event information including timestamp and event ID
     * @see TucDeviceOpen(), TucUnregisterEventCallback(), TUC_EVENT_TYPE, TUC_EVENT_CALLBACK
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucRegisterEventCallback(IN TUC_HDEVICE hDevice, IN TUC_EVENT_TYPE eventType, IN TUC_EVENT_CALLBACK callback, IN void *pUserContext);

    /**
     * @brief Unregister buffer callback function
     *
     * This function unregisters a previously registered buffer callback function from the specified device.
     * After calling this function, the device will no longer invoke callback functions when buffers become
     * available during image acquisition. This is essential for proper cleanup and resource management.
     *
     * The buffer callback unregistration process includes:
     * - Validating the device handle and callback registration status
     * - Removing the callback function from the device's callback registry
     * - Cleaning up any callback-related resources and memory
     * - Ensuring proper synchronization during ongoing acquisition operations
     * - Restoring the device to a state without active buffer callbacks
     *
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if parameters are invalid,
     *         TUC_STATUS_INVALID_DEVICE_HANDLE if hDevice is invalid,
     *         TUC_STATUS_DEVICE_NOT_OPENED if device is not opened,
     *         TUC_STATUS_UNREGISTER_CALLBACK_FAILED if callback unregistration fails,
     *         TUC_STATUS_CALLBACK_NOT_REGISTERED if no callback is currently registered
     * @note This function is thread-safe and can be called during active acquisition
     * @warning It is recommended to call this function before closing the device or stopping acquisition
     * @see TucDeviceOpen(), TucRegisterBufferCallback(), TucDeviceClose(), TucAcquisitionStop()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucUnregisterBufferCallback(IN TUC_HDEVICE hDevice);

    /**
     * @brief Unregister event callback function
     * @param hDevice [IN] Device handle
     * @param eventType [IN] Event type
     * @return TUC_STATUS_OK on success, error code on failure
     */
    TUC_API TUC_STATUS TucUnregisterEventCallback(IN TUC_HDEVICE hDevice, IN TUC_EVENT_TYPE eventType);

    /**
     * @brief Get event statistics for a device
     * @param hDevice [IN] Device handle
     * @param eventType [IN] Event type (or -1 for all events)
     * @param pStatistics [OUT] Pointer to event statistics structure
     * @return TUC_STATUS_OK on success, error code on failure
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucGetEventStatistics(
        IN TUC_HDEVICE hDevice,
        IN TUC_EVENT_TYPE eventType,
        OUT TUC_EVENT_STATISTICS *pStatistics
    );


    // =============================================================================
    // 11. Video Recording - Create, destroy, and write video files
    // =============================================================================

    /**
     * @brief Create video recorder
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @param[in] filePath Recording file path for the video file
     * @param[in] pConfig Recording configuration parameters including codec, quality, etc.
     * @param[in] pDimension Recording dimension parameters (optional, if NULL uses pipeline image info)
     * @param[out] phRecorder Pointer to recorder handle for subsequent operations
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if parameters are invalid,
     *         TUC_STATUS_INVALID_DEVICE_HANDLE if hDevice is invalid,
     *         TUC_STATUS_DEVICE_NOT_OPENED if device is not opened,
     *         TUC_STATUS_RECORDER_CREATE_FAILED if recorder creation fails
     * @note This function is thread-safe and requires a valid device handle from TucDeviceOpen()
     * @warning The device must be opened before calling this function
     * @see TucDeviceOpen(), TucRecorderDestroy(), TucRecorderWriteImage()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucRecorderCreate(IN TUC_HDEVICE hDevice, IN const char *filePath, IN TUC_RECORDER_CONFIG *pConfig, IN TUC_RECORDER_DIMENSION *pDimension, OUT TUC_HRECORDER *phRecorder);   

    /**
     * @brief Destroy video recorder
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @param[in] hRecorder Recorder handle obtained from TucRecorderCreate()
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if parameters are invalid,
     *         TUC_STATUS_INVALID_DEVICE_HANDLE if hDevice is invalid,
     *         TUC_STATUS_INVALID_RECORDER_HANDLE if hRecorder is invalid,
     *         TUC_STATUS_DEVICE_NOT_OPENED if device is not opened,
     *         TUC_STATUS_RECORDER_DESTROY_FAILED if recorder destruction fails
     * @note This function is thread-safe and should be called to clean up recorder resources
     * @warning The recorder handle becomes invalid after this call
     * @see TucRecorderCreate(), TucRecorderWriteImage()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucRecorderDestroy(IN TUC_HDEVICE hDevice, IN TUC_HRECORDER hRecorder);

    /**
     * @brief Write image to recorder
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @param[in] hRecorder Recorder handle obtained from TucRecorderCreate()
     * @param[in] pImageInfo Image information including buffer pointer, size, and format
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if parameters are invalid,
     *         TUC_STATUS_INVALID_DEVICE_HANDLE if hDevice is invalid,
     *         TUC_STATUS_INVALID_RECORDER_HANDLE if hRecorder is invalid,
     *         TUC_STATUS_DEVICE_NOT_OPENED if device is not opened,
     *         TUC_STATUS_RECORDER_WRITE_FAILED if image writing fails
     * @note This function is thread-safe and can be called during active recording
     * @warning The image data must be valid and accessible during the call
     * @see TucRecorderCreate(), TucRecorderDestroy()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucRecorderWriteImage(IN TUC_HDEVICE hDevice, IN TUC_HRECORDER hRecorder, IN TUC_IMAGE_INFO *pImageInfo);

    // =============================================================================
    // 12. Logging and Profile File Settings - Create, destroy, and write log files
    // =============================================================================

    /**
     * @brief Create logger instance
     * @param[in] logLevel Log level
     * @param[in] filename Log file name (without path)
     * @param[in] pConfig Logger configuration including file policy and rotation settings
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if parameters are invalid,
     *         TUC_STATUS_LOGGER_CREATE_FAILED if logger creation fails
     * @note This function is thread-safe and creates a global logger instance managed by SDK core
     * @warning The filename should not include path separators, logs will be saved to the default logs directory
     * @note For TUC_LOG_FILE_ROTATE_SIZE mode, maxFileSizeMB and maxFileCount in config are used
     * @see TucLoggerDestroy(), TucLoggerWrite()
     * @since v3.0.0
     */
    TUC_API TUC_STATUS TucLoggerCreate(IN TUC_LOG_LEVEL logLevel, IN const char *filename, IN const TUC_LOGGER_CONFIG *pConfig);

    /**
     * @brief Write log message to logger
     * @param[in] logLevel Log level for the message
     * @param[in] message Log message content
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if parameters are invalid,
     *         TUC_STATUS_LOGGER_WRITE_FAILED if writing fails
     * @note This function is thread-safe and requires a created logger
     * @warning The logger must be created before calling this function
     * @see TucLoggerCreate(), TucLoggerDestroy()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucLoggerWrite(IN TUC_LOG_LEVEL logLevel, IN const char *message);

    /**
     * @brief Destroy logger instance
     * @return TUC_STATUS_OK on success, TUC_STATUS_LOGGER_DESTROY_FAILED if logger destruction fails
     * @note This function is thread-safe and releases all resources associated with the logger
     * @see TucLoggerCreate(), TucLoggerWrite()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucLoggerDestroy();

    /**
     * @brief Save device profile
     *
     * This function saves the current device configuration parameters to a profile file.
     * The profile is saved using the default policy (TUC_PROFILE_POLICY_ONLY_MODEL) which
     * creates a model-specific configuration file. The saved profile can be loaded later
     * using TucProfileLoad() to restore the device settings.
     *
     * The profile saving process includes:
     * - Validating the device handle and group name parameters
     * - Checking if the device is opened and accessible
     * - Collecting current feature parameter values from the device
     * - Creating an XML-based profile file with the specified group name
     * - Saving the profile to the configured profile directory
     *
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @param[in] groupName Name of the parameter group to save (e.g., "DefaultParameter", "CustomSettings")
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if any parameter is NULL,
     *         TUC_STATUS_INVALID_DEVICE_HANDLE if hDevice is invalid,
     *         TUC_STATUS_DEVICE_NOT_OPENED if device is not opened,
     *         TUC_STATUS_SAVE_PROFILE_FAILED on profile saving failure
     * @note This function is thread-safe and requires a valid device handle from TucDeviceOpen()
     * @warning The device must be opened before calling this function
     * @see TucDeviceOpen(), TucProfileLoad(), TucSetProfileSettings()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucProfileSave(IN TUC_HDEVICE hDevice, IN const char *groupName);

    /**
     * @brief Load device profile
     *
     * This function loads device configuration parameters from a previously saved profile file.
     * The profile is loaded using the default policy (TUC_PROFILE_POLICY_ONLY_MODEL) which
     * loads the model-specific configuration file. This function can restore device settings
     * that were previously saved using TucProfileSave().
     *
     * The profile loading process includes:
     * - Validating the device handle and group name parameters
     * - Checking if the device is opened and accessible
     * - Locating the profile file based on the group name and device model
     * - Parsing the XML-based profile file
     * - Applying the loaded parameter values to the device features
     *
     * @param[in] hDevice Device handle obtained from TucDeviceOpen()
     * @param[in] groupName Name of the parameter group to load (e.g., "DefaultParameter", "CustomSettings")
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if any parameter is NULL,
     *         TUC_STATUS_INVALID_DEVICE_HANDLE if hDevice is invalid,
     *         TUC_STATUS_DEVICE_NOT_OPENED if device is not opened,
     *         TUC_STATUS_PROFILE_NOT_FOUND if profile file does not exist,
     *         TUC_STATUS_LOAD_PROFILE_FAILED on profile loading failure
     * @note This function is thread-safe and requires a valid device handle from TucDeviceOpen()
     * @warning The device must be opened before calling this function
     * @see TucDeviceOpen(), TucProfileSave(), TucSetProfileSettings()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucProfileLoad(IN TUC_HDEVICE hDevice, IN const char *groupName);

    /**
     * @brief Set device profile settings
     *
     * This function is used to set the device profile path and policy, which must be called after interface
     * initialization and before opening the device. The profile contains default parameter settings
     * for the device, such as exposure time, gain, white balance, etc. After setting the profile
     * settings, the corresponding profile will be automatically loaded when opening the device later.
     *
     * @param[in] hDevice Device handle, currently unused in this version, can be set to nullptr
     * @param[in] filePath Profile file path, supports relative and absolute paths
     * @param[in] policy Profile policy, controls the loading and saving behavior of profiles
     *
     * @return TUC_STATUS_OK on success, error code on failure
     *
     * @note This function must be called after TucInitialize() and before TucDeviceOpen()
     * @warning If the profile file path is invalid or the file does not exist, device opening may fail
     * @see TucInitialize(), TucDeviceOpen(), TucProfileSave(), TucProfileLoad()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucSetProfileSettings(IN TUC_HDEVICE hDevice, IN const char *filePath, IN TUC_PROFILE_POLICY policy);

    /**
     * @brief Get current profile settings
     *
     * This function retrieves the current profile settings including both the profile path
     * and policy settings. It can be used to query the current profile settings that were
     * previously set using TucSetProfileSettings().
     *
     * @param[in] hDevice Device handle, currently unused in this version, can be set to nullptr
     * @param[out] policy Pointer to receive the current profile policy
     * @param[out] filePath Buffer to receive the profile path
     * @param[inout] size Size of the buffer (input) and actual size needed (output)
     * @return TUC_STATUS_OK on success, error code on failure
     *
     * @note This function is thread-safe
     * @see TucSetProfileSettings(), TucProfileSave(), TucProfileLoad()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucGetProfileSettings(IN TUC_HDEVICE hDevice, OUT TUC_PROFILE_POLICY *policy, OUT char *filePath, INOUT uint32_t *size);

    /**
     * @brief Export device parameters to a user-specified file (independent of automatic profile directory).
     * @param[in] filePath Full path of the output file (UTF-8). Parent directories are created as needed.
     * @param[in] format File format; only @ref TUC_PROFILE_FILE_FORMAT_XML is supported in this version.
     * @param[in] policy Written to file header; must match the policy used later for import when serial matching applies.
     * @note For GenICam (generic) devices the SDK may stop acquisition internally before reading parameters.
     */
    TUC_API TUC_STATUS TucProfileExportToFile(IN TUC_HDEVICE hDevice, IN const char *filePath,
        IN TUC_PROFILE_FILE_FORMAT format, IN TUC_PROFILE_POLICY policy);

    /**
     * @brief Import device parameters from a user-specified file. Stops on first set failure and rolls back to pre-import values.
     * @param[in] filePath Full path of the profile file (UTF-8).
     * @param[in] format File format; only @ref TUC_PROFILE_FILE_FORMAT_XML is supported in this version.
     * @param[in] policy Must be consistent with the file header when the file records a policy value.
     * @note For GenICam devices the SDK stops acquisition internally before applying parameters. Non-GenICam devices only apply nodes marked profile-saved.
     */
    TUC_API TUC_STATUS TucProfileImportFromFile(IN TUC_HDEVICE hDevice, IN const char *filePath,
        IN TUC_PROFILE_FILE_FORMAT format, IN TUC_PROFILE_POLICY policy);

    // =============================================================================
    // 13. Image Drawing - Create, destroy, and draw image frames
    // =============================================================================

    /**
     * @brief Create image drawing context
     *
     * This function creates a new drawing context for image rendering operations.
     * The drawing context manages the relationship between a device and a target window,
     * providing a foundation for subsequent drawing operations.
     *
     * @param[in] hDevice Device handle associated with the drawing context
     * @param[in] pWindowHandle Handle to the target window for drawing operations
     * @param[out] phContext Pointer to receive the created drawing context handle
     * @return TUC_STATUS_OK on success, TUC_STATUS_DRAW_CONTEXT_ALREADY_CREATED if context already exists, other error codes on failure
     *
     * @note This function is thread-safe
     * @see TucDrawContextDestroy(), TucDrawImage()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucDrawContextCreate(IN TUC_HDEVICE hDevice, IN void *pWindowHandle, OUT TUC_HCONTEXT_DRAW *phContext);

    /**
     * @brief Destroy image drawing context
     *
     * This function destroys a previously created drawing context and releases all
     * associated resources. After calling this function, the drawing context handle
     * becomes invalid and should not be used for further operations.
     *
     * @param[in] hDevice Device handle associated with the drawing context
     * @param[in] hContext Drawing context handle to be destroyed
     * @return TUC_STATUS_OK on success, TUC_STATUS_DRAW_CONTEXT_NOT_FOUND if context not found, other error codes on failure
     *
     * @note This function is thread-safe
     * @see TucDrawContextCreate()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucDrawContextDestroy(IN TUC_HDEVICE hDevice, IN TUC_HCONTEXT_DRAW hContext);

    /**
     * @brief Draw image frame with specified dimensions
     *
     * This function renders an image frame to the target window using the specified
     * drawing context. It supports flexible source and destination rectangle configuration,
     * allowing for image scaling, cropping, and positioning within the target window.
     *
     * @param[in] hDevice Device handle associated with the drawing context
     * @param[in] hContext Drawing context handle for the rendering operation
     * @param[in] pImageInfo Pointer to image information structure containing image data and metadata
     * @param[in] pDimension Pointer to drawing dimension structure specifying source and destination rectangles
     * @return TUC_STATUS_OK on success, TUC_STATUS_DRAW_CONTEXT_NOT_FOUND if context not found, 
     *         TUC_STATUS_DRAW_CONTEXT_WRITE_FAILED if drawing operation failed, other error codes on failure
     *
     * @note This function is thread-safe
     * @see TucDrawContextCreate(), TucGetDrawStatistics()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucDrawImage(IN TUC_HDEVICE hDevice, IN TUC_HCONTEXT_DRAW hContext, IN TUC_IMAGE_INFO *pImageInfo, IN const TUC_DRAW_DIMENSION *pDimension);

    /**
     * @brief Get drawing performance statistics
     *
     * This function retrieves performance statistics for drawing operations associated
     * with the specified drawing context. The statistics include frame count, frames per
     * second (FPS), and timing information that can be used for performance monitoring.
     *
     * @param[in] hDevice Device handle associated with the drawing context
     * @param[in] hContext Drawing context handle for which to retrieve statistics
     * @param[out] pStatistics Pointer to structure that will receive the drawing statistics
     * @return TUC_STATUS_OK on success, TUC_STATUS_DRAW_CONTEXT_NOT_FOUND if context not found, other error codes on failure
     *
     * @note This function is thread-safe
     * @see TucDrawImage(), TucDrawContextCreate()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucGetDrawStatistics(IN TUC_HDEVICE hDevice, IN TUC_HCONTEXT_DRAW hContext, OUT TUC_DRAW_STATISTICS *pStatistics);

    // =============================================================================
    // 14. CPU Performance Optimization - High-speed data acquisition CPU optimization
    // =============================================================================

    /**
     * @brief Set CPU optimization policy (process scope)
     *
     * Configures the SDK process-wide CPU affinity plan (Auto / Manual / Disabled). Active acquirers
     * (streaming devices) share one plan; CTI and TUCam worker threads consume the published plan.
     *
     * @param[in] hDevice Device handle from TucDeviceOpen() (validated; must be opened). Policy is
     *                    applied to the **process**, not to this handle alone. Per-device policy via
     *                    hDevice is **reserved** for a future API revision.
     * @param[in] policy CPU optimization policy to apply
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if policy is invalid,
     *         TUC_STATUS_INVALID_DEVICE_HANDLE if hDevice is invalid,
     *         TUC_STATUS_DEVICE_NOT_OPENED if device is not opened
     *
     * @note Process-wide: changing policy affects all devices in the same process.
     * @note Real-time priority may require administrator privileges.
     * @note This function is thread-safe.
     * @see TucSetThreadCoreAndPriority(), TucGetCPUCapabilities()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucSetOptimizeCPUPolicy(IN TUC_HDEVICE hDevice, IN TUC_CPU_OPTIMIZATION_POLICY policy);

    /**
     * @brief Set thread CPU affinity and priority (Manual mode, process plan slot)
     *
     * Updates the manual override for the given thread role in the **process-wide** affinity plan
     * for the device identified by hDevice (device key / serial). Requires policy Manual.
     * GRAB / PIPELINE / STORAGE apply to the calling or SDK thread when invoked; CTI roles are
     * applied inside the transport layer on next plan read.
     *
     * @param[in] hDevice Device handle from TucDeviceOpen() - selects the **plan slot** for this
     *                    device in the shared process plan. Per-device-only binding API is **reserved**.
     * @param[in] threadType Thread role to configure
     * @param[in] coreId CPU core ID (-1 = no affinity mask)
     * @param[in] priority Thread priority level
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_OPERATION if policy is not Manual,
     *         TUC_STATUS_INVALID_PARAM if parameters are invalid,
     *         TUC_STATUS_INVALID_DEVICE_HANDLE if hDevice is invalid
     *
     * @note Effective only when TucSetOptimizeCPUPolicy(..., TUC_CPU_POLICY_MANUAL) is active.
     * @note Core ID -1 clears affinity binding for that role.
     * @note This function is thread-safe.
     * @see TucSetOptimizeCPUPolicy(), TucGetCPUCapabilities()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucSetThreadCoreAndPriority(IN TUC_HDEVICE hDevice, IN TUC_THREAD_TYPE threadType, IN int coreId, IN TUC_THREAD_PRIORITY priority);

    /**
     * @brief Get system CPU capabilities and recommendations
     * 
     * This function queries system CPU hardware capabilities, supported features, and provides
     * intelligent recommendations for optimal configuration. It does not require a device handle
     * and can be called at any time to assess system performance potential.
     * 
     * The capability detection process includes:
     * - Detecting CPU core count, physical cores, and NUMA topology
     * - Checking for hyper-threading and NUMA availability
     * - Verifying administrative privileges for advanced features
     * - Measuring cache sizes and system performance characteristics
     * - Calculating recommended core allocation using reverse allocation strategy
     * - Determining optimal optimization policy based on hardware capabilities
     * 
     * @param[out] pCapabilities Pointer to structure to receive CPU capability information
     * @return TUC_STATUS_OK on success, TUC_STATUS_INVALID_PARAM if pCapabilities is null,
     *         TUC_STATUS_INTERNAL_ERROR on system detection failure
     * 
     * @note This function does not require a device handle
     * @note Recommendations use reverse allocation strategy avoiding CPU0
     * @note This function is thread-safe
     * @see TucSetOptimizeCPUPolicy(), TucSetThreadCoreAndPriority()
     * @since v1.0.0
     */
    TUC_API TUC_STATUS TucGetCPUCapabilities(OUT TUC_CPU_CAPABILITIES *pCapabilities);

#ifdef __cplusplus
}
#endif

#endif /* TUC_API_H */
