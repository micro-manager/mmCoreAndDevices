/**
 * @copyright Copyright (C) Tucsen Photonics Co., Ltd. 2012-2025. All rights reserved.
 * @brief TUC SDK Definition Header File - Type Definitions and Constants
 *
 * This file contains all type definitions, enums, structures, and constants for TUC SDK,
 * organized by 14 functional categories to match the API structure. It serves as the 
 * foundation for all TUC SDK API interfaces and provides comprehensive type definitions
 * for device management, image acquisition, and processing capabilities.
 *
 * For more information, please visit: https://www.tucsen.com/
 *
 * Organization (14 Functional Categories):
 *  1. SDK Lifecycle Management       - Initialization, version info, and termination
 *  2. Device Discovery & Management  - Enumeration, information, and connection control
 *  3. Feature Parameter Management   - Feature discovery and metadata access
 *  4. Feature Value Access (Numeric) - Integer and floating-point parameter control
 *  5. Feature Value Access (Boolean/String) - Boolean and string parameter control
 *  6. Feature Value Access (Enumeration)   - Enumeration parameter control and information
 *  7. Image Acquisition Control      - Start/stop acquisition and statistics
 *  8. Buffer Management              - Buffer pool and individual buffer operations
 *  9. Buffer Advanced Operations     - Data processing, conversion, and file I/O
 * 10. Event Handling                 - Callback registration and event management
 * 11. Video Recording                - Real-time video capture and encoding
 * 12. Logging & Profile Management   - Debug logging and device configuration profiles
 * 13. Image Drawing                  - Real-time image display and rendering
 * 14. CPU Performance Optimization   - High-speed data acquisition CPU optimization
 *
 * @file TucDefine.h
 * @author zhang ren
 * @date 2025-07-01
 * @version v3.0.0
 * @note This header file is included by all Tucsen Photonics SDK API files
 * @warning All structures are designed for C compatibility
 * @see TucApi.h, TucCore.h
 * @since v1.0.0
 */
#include "PFTU.h"

#ifndef TUC_DEFINE_H
#define TUC_DEFINE_H

#ifdef __cplusplus
extern "C"
{
#endif

/* Include necessary headers */
#if !defined(__cplusplus)
#include <stdbool.h>
#endif
#include <stddef.h>
#include <stdint.h>

/* Platform-specific byte alignment macros */
#ifdef _WIN32
    /* Microsoft Visual C++ */
    #define TUC_PACK_PUSH(n) __pragma(pack(push, n))
    #define TUC_PACK_POP() __pragma(pack(pop))
    #define TUC_ALIGNED(n) __declspec(align(n))
    #define TUC_PACKED __pragma(pack(1))
#elif defined(__GNUC__) || defined(__clang__)
    /* GCC and Clang */
    #define TUC_PACK_PUSH(n)   TUC_PACK_PUSH_IMPL(n)
    #define TUC_PACK_PUSH_IMPL(n)  TUC_PACK_PUSH_##n()
    #define TUC_PACK_PUSH_8()  _Pragma("pack(push, 8)")
    #define TUC_PACK_PUSH_4()  _Pragma("pack(push, 4)")
    #define TUC_PACK_POP() _Pragma("pack(pop)")
    #define TUC_ALIGNED(n) __attribute__((aligned(n)))
    #define TUC_PACKED __attribute__((packed))
#else
    /* Other compilers - fallback to no packing */
    #define TUC_PACK_PUSH(n)
    #define TUC_PACK_POP()
    #define TUC_ALIGNED(n)
    #define TUC_PACKED
#endif

/* Cross-platform structure packing - optimal alignment for performance and compatibility
 * 
 * Alignment Strategy:
 * - Platform-specific alignment: 8-byte for 64-bit, 4-byte for 32-bit
 * - Benefits: 
 *   * 64-bit platforms: Proper pointer alignment, prevents data corruption
 *   * 32-bit platforms: Memory efficiency with 4-byte alignment
 *   * Performance: Optimal memory access patterns for each platform
 *   * Compatibility: Platform-appropriate alignment for all data types
 *   * memcpy safety: Predictable structure layouts for direct copying
 * 
 * Field Ordering Principles:
 * - Group fields by size (largest first) to minimize padding
 * - Place frequently accessed fields first for better cache performance
 * - Use explicit padding/reserved fields for ABI stability
 * - Document field sizes and alignment requirements
 * 
 * Platform Detection:
 * - 64-bit: Use 8-byte alignment for proper pointer handling
 * - 32-bit: Use 4-byte alignment for memory efficiency
 */
#if defined(_WIN64) || defined(__x86_64__) || defined(__aarch64__) || (defined(__SIZEOF_POINTER__) && __SIZEOF_POINTER__ == 8)
    /* 64-bit platforms - use 8-byte alignment for pointer compatibility */
    TUC_PACK_PUSH(8)
#else
    /* 32-bit platforms - use 4-byte alignment for memory efficiency */
    TUC_PACK_PUSH(4)
#endif

/* Compile-time structure size validation macros */
#ifdef __cplusplus
    #define TUC_STATIC_ASSERT(condition, message) static_assert(condition, #message)
#else
    #define TUC_STATIC_ASSERT(condition, message) typedef char tuc_static_assert_##message[(condition) ? 1 : -1]
#endif

/* Callback function calling convention */
#ifdef _WIN32
#define TUC_CALLBACK __stdcall
#else
#define TUC_CALLBACK
#endif

#define TUC_STATUS_IS_SUCCESS(status) ((status) == TUC_STATUS_OK)

    // =============================================================================
    // Global Status Code Definitions - Used across all categories
    // =============================================================================

    /**
     * @brief Status code enumeration
     * @remarks Defines all possible return values for TUC SDK functions
     */
    typedef enum _TUC_STATUS
    {
        // =============================================================================
        // Success Status (0)
        // =============================================================================
        TUC_STATUS_OK = 0,                             // Success
         
        // =============================================================================
        // 1. SDK Lifecycle Management Errors (1-199)
        // =============================================================================
        TUC_STATUS_NOT_INITIALIZED = 1,            // SDK not initialized
        TUC_STATUS_ALREADY_INITIALIZED = 2,        // SDK already initialized
        TUC_STATUS_INITIALIZATION_FAILED = 3,      // SDK initialization failed
        TUC_STATUS_TERMINATION_FAILED = 4,         // SDK termination failed
        TUC_STATUS_UNKNOWN_ERROR = 5,              // Unknown error
        TUC_STATUS_INTERNAL_ERROR = 6,             // Internal error
        TUC_STATUS_NOT_IMPLEMENTED = 7,            // Function not implemented
        TUC_STATUS_NOT_READY = 8,                  // Not ready
        TUC_STATUS_TIMEOUT = 9,                    // Timeout
        TUC_STATUS_ACCESS_DENIED = 10,             // Access denied
        TUC_STATUS_INVALID_OPERATION = 11,         // Invalid operation
        TUC_STATUS_OUT_OF_MEMORY = 12,             // Out of memory
        TUC_STATUS_INSUFFICIENT_MEMORY = 13,       // Insufficient memory
        TUC_STATUS_INVALID_PARAM = 14,             // Invalid parameter
        TUC_STATUS_INVALID_HANDLE = 15,            // Invalid handle
        TUC_STATUS_INDEX_OUT_OF_RANGE = 16,        // Index out of range
        TUC_STATUS_BUFFER_TOO_SMALL = 17,          // Buffer too small
        TUC_STATUS_IO_ERROR = 18,                  // IO error
        TUC_STATUS_FILE_NOT_FOUND = 19,            // File not found
        TUC_STATUS_INVALID_FILE = 20,              // Invalid file format
        TUC_STATUS_ABORTED = 21,                     // Operation aborted by user / host request
         
        // =============================================================================
        // 2. Device Discovery & Management Errors (200-299)
        // =============================================================================
        TUC_STATUS_DEVICE_NOT_FOUND = 200,         // Device not found
        TUC_STATUS_DEVICE_BUSY = 201,              // Device busy
        TUC_STATUS_DEVICE_DISCONNECTED = 202,      // Device disconnected
        TUC_STATUS_DEVICE_NOT_SUPPORTED = 203,     // Device not supported
        TUC_STATUS_DEVICE_INITIALIZATION_FAILED = 204, // Device initialization failed
        TUC_STATUS_INVALID_DEVICE_HANDLE = 205,    // Invalid device handle
        TUC_STATUS_DEVICE_NOT_OPENED = 206,        // Device not opened
        TUC_STATUS_DEVICE_ALREADY_OPENED = 207,    // Device already opened
        TUC_STATUS_INVALID_DEVICE_INDEX = 208,     // Invalid device index
        TUC_STATUS_INVALID_DEVICE_TYPE = 209,      // Invalid device type
        TUC_STATUS_ENUMERATION_FAILED = 210,       // Device enumeration failed
        TUC_STATUS_GET_DEVICE_INFO_FAILED = 211,   // Get device info failed
        TUC_STATUS_OPEN_DEVICE_FAILED = 212,       // Open device failed
        TUC_STATUS_CLOSE_DEVICE_FAILED = 213,      // Close device failed
         
        // =============================================================================
        // 3. Feature Parameter Management Errors (300-399)
        // =============================================================================
        TUC_STATUS_FEATURE_NOT_FOUND = 300,        // Feature not found
        TUC_STATUS_FEATURE_TYPE_MISMATCH = 301,    // Feature type mismatch
        TUC_STATUS_FEATURE_NOT_SUPPORTED = 302,    // Feature not supported
        TUC_STATUS_FEATURE_COMMAND_FAILED = 303,   // Feature command execution failed
        TUC_STATUS_GET_FEATURE_COUNT_FAILED = 304, // Get feature count failed
        TUC_STATUS_GET_FEATURE_INFO_FAILED = 305,  // Get feature info failed
        TUC_STATUS_CHECK_FEATURE_READABLE_FAILED = 306, // Check feature readable failed
        TUC_STATUS_CHECK_FEATURE_WRITABLE_FAILED = 307, // Check feature writable failed
         
        // =============================================================================
        // 4. Feature Value Access (Numeric) Errors (400-499)
        // =============================================================================
        TUC_STATUS_FEATURE_VALUE_OUT_OF_RANGE = 400, // Feature value out of range
        TUC_STATUS_SET_FEATURE_VALUE_FAILED = 401,   // Set feature value failed
        TUC_STATUS_GET_FEATURE_VALUE_FAILED = 402,   // Get feature value failed
         
        // =============================================================================
        // 5. Feature Value Access (Boolean/String) Errors (500-599)
        // =============================================================================
        // (No specific errors defined for this category)
         
        // =============================================================================
        // 6. Feature Value Access (Enumeration) Errors (600-699)
        // =============================================================================
        TUC_STATUS_FEATURE_ENUM_ENTRY_NOT_FOUND = 600, // Feature enum entry not found
        TUC_STATUS_INVALID_ENUM_VALUE = 601,           // Invalid enum value
         
        // =============================================================================
        // 7. Image Acquisition Control Errors (700-799)
        // =============================================================================
        TUC_STATUS_ACQUISITION_NOT_STARTED = 700,      // Acquisition not started
        TUC_STATUS_ACQUISITION_ALREADY_STARTED = 701,  // Acquisition already started
        TUC_STATUS_ACQUISITION_START_FAILED = 702,     // Acquisition start failed
        TUC_STATUS_ACQUISITION_STOP_FAILED = 703,      // Acquisition stop failed
        TUC_STATUS_ACQUISITION_ABORTED = 704,          // Acquisition aborted
        TUC_STATUS_ACQUISITION_RUNNING = 705,          // Acquisition running
        TUC_STATUS_ACQUISITION_STOPPED = 706,          // Acquisition stopped
        TUC_STATUS_START_ACQUISITION_FAILED = 707,    // Start acquisition failed

        // =============================================================================
        // 8. Buffer Management Errors (800-899)
        // =============================================================================
        TUC_STATUS_INVALID_BUFFER_HANDLE = 800,    // Invalid buffer handle
        TUC_STATUS_BUFFER_NOT_ANNOUNCED = 801,     // Buffer not announced
        TUC_STATUS_BUFFER_ALREADY_ANNOUNCED = 802, // Buffer already announced
        TUC_STATUS_BUFFER_NOT_QUEUED = 803,        // Buffer not queued
        TUC_STATUS_BUFFER_ALREADY_QUEUED = 804,    // Buffer already queued
        TUC_STATUS_BUFFER_QUEUE_EMPTY = 805,       // Buffer queue empty
        TUC_STATUS_BUFFER_QUEUE_FULL = 806,        // Buffer queue full
        TUC_STATUS_BUFFER_SIZE_MISMATCH = 807,     // Buffer size mismatch
        TUC_STATUS_BUFFER_UNDERRUN = 808,          // Buffer underrun
        TUC_STATUS_BUFFER_INCOMPLETE = 809,        // Buffer incomplete
        TUC_STATUS_BUFFER_FULL = 810,              // Buffer full
        TUC_STATUS_BUFFER_EMPTY = 811,             // Buffer empty
        TUC_STATUS_BUFFER_IN_USE = 812,            // Buffer in use
        TUC_STATUS_BUFFER_ANNOUNCED_LESS = 813,    // Buffer announced less than min
        TUC_STATUS_NO_CHUNK_DATA = 814,            // No chunk data
        TUC_STATUS_GET_BUFFER_INFO_FAILED = 815,   // Get buffer info failed

        // =============================================================================
        // 9. Buffer Advanced Operations Errors (900-999)
        // =============================================================================
        TUC_STATUS_CONVERSION_FAILED = 900, // Conversion failed
        TUC_STATUS_SAVE_FAILED = 901,       // Save operation failed
        TUC_STATUS_LOAD_FAILED = 902,       // Load operation failed

        // =============================================================================
        // 10. Event Handling Errors (1000-1099)
        // =============================================================================
        TUC_STATUS_EVENT_NOT_SUPPORTED = 1000,            // Event not supported
        TUC_STATUS_EVENT_ALREADY_REGISTERED = 1001,       // Event already registered
        TUC_STATUS_EVENT_NOT_REGISTERED = 1002,           // Event not registered
        TUC_STATUS_CALLBACK_REGISTRATION_FAILED = 1003,   // Callback registration failed
        TUC_STATUS_CALLBACK_UNREGISTRATION_FAILED = 1004, // Callback unregistration failed
        TUC_STATUS_EVENT_TIMEOUT = 1005,                  // Event timeout
        TUC_STATUS_REGISTER_CALLBACK_FAILED = 1006,       // Register callback failed
        TUC_STATUS_UNREGISTER_CALLBACK_FAILED = 1007,     // Unregister callback failed
        TUC_STATUS_NO_MORE_CALLBACKS = 1008,              // No more callback slots available
        TUC_STATUS_THREAD_CREATE_FAILED = 1009,           // Thread creation failed

        // =============================================================================
        // 11. Video Recording Errors (1100-1199)
        // =============================================================================
        TUC_STATUS_RECORDER_NOT_CREATED = 1100,     // Recorder not created
        TUC_STATUS_RECORDER_ALREADY_CREATED = 1101, // Recorder already created
        TUC_STATUS_RECORDER_NOT_STARTED = 1102,     // Recorder not started
        TUC_STATUS_RECORDER_ALREADY_STARTED = 1103, // Recorder already started
        TUC_STATUS_RECORDER_START_FAILED = 1104,    // Recorder start failed
        TUC_STATUS_RECORDER_STOP_FAILED = 1105,     // Recorder stop failed
        TUC_STATUS_RECORDER_CONFIG_INVALID = 1106,  // Recorder config invalid
        TUC_STATUS_RECORDER_RESOURCE_ERROR = 1107,  // Recorder resource error
        TUC_STATUS_RECORDER_FORMAT_ERROR = 1108,    // Recorder format error
        TUC_STATUS_RECORDER_OUT_OF_SPACE = 1109,    // Recorder out of space

        // =============================================================================
        // 12. Logging & Profile Management Errors (1200-1299)
        // =============================================================================
        TUC_STATUS_LOG_CONFIG_INVALID = 1200,       // Log config invalid
        TUC_STATUS_LOG_FILE_CREATION_FAILED = 1201, // Log file creation failed
        TUC_STATUS_LOGGER_CREATE_FAILED = 1202,     // Logger creation failed
        TUC_STATUS_LOGGER_ALREADY_CREATED = 1203,   // Logger already created
        TUC_STATUS_LOGGER_DESTROY_FAILED = 1204,    // Logger destruction failed
        TUC_STATUS_LOGGER_WRITE_FAILED = 1205,      // Logger write failed
        TUC_STATUS_LOGGER_NOT_CREATED = 1206,       // Logger not created
        TUC_STATUS_PROFILE_SAVE_FAILED = 1207,      // Profile save failed
        TUC_STATUS_PROFILE_LOAD_FAILED = 1208,      // Profile load failed
        TUC_STATUS_PROFILE_GET_FAILED = 1209,       // Profile get failed
        TUC_STATUS_PROFILE_SET_FAILED = 1210,       // Profile set failed
        TUC_STATUS_SAVE_PROFILES_FAILED = 1211,     // Save profiles failed
        TUC_STATUS_LOAD_PROFILES_FAILED = 1212,     // Load profiles failed
        TUC_STATUS_CREATE_DIRECTORY_FAILED = 1213,  // Create directory failed
        TUC_STATUS_PATH_NOT_EXISTS = 1214,          // Path not exists
        TUC_STATUS_PROFILE_FILE_HEADER_MISMATCH = 1215, // Profile file header/device mismatch
        TUC_STATUS_PROFILE_FILE_PARSE_FAILED = 1216,    // Profile file parse error
        TUC_STATUS_PROFILE_EXPORT_FAILED = 1217,        // Explicit-path profile export failed
        TUC_STATUS_PROFILE_IMPORT_SET_FAILED = 1218,    // Profile import failed during apply
        TUC_STATUS_PROFILE_IMPORT_ROLLBACK_FAILED = 1219, // Profile import rollback failed

        // =============================================================================
        // 13. Image Drawing Errors (1300-1399)
        // =============================================================================
        TUC_STATUS_DRAW_CONTEXT_NOT_FOUND = 1300,       // Draw context not found
        TUC_STATUS_DRAW_CONTEXT_CREATION_FAILED = 1301, // Draw context creation failed
        TUC_STATUS_DRAW_OPERATION_FAILED = 1302,        // Draw operation failed
        TUC_STATUS_DRAW_INVALID_DIMENSION = 1303,       // Draw invalid dimension
        TUC_STATUS_DRAW_CONTEXT_ALREADY_CREATED = 1304, // Draw context already created
        TUC_STATUS_DRAW_CONTEXT_WRITE_FAILED = 1305,    // Draw context write failed

        // =============================================================================
        // 14. CPU Performance Optimization Errors (1400-1499)
        // =============================================================================
        TUC_STATUS_CPU_OPTIMIZATION_FAILED = 1400,       // CPU optimization failed
        TUC_STATUS_CPU_AFFINITY_SET_FAILED = 1401,       // CPU affinity set failed
        TUC_STATUS_THREAD_PRIORITY_SET_FAILED = 1402,    // Thread priority set failed
        TUC_STATUS_CPU_CAPABILITIES_QUERY_FAILED = 1403, // CPU capabilities query failed

        // =============================================================================
        // Reserved for Future Use (1500-9999)
        // =============================================================================
        TUC_STATUS_RESERVED_START = 1500, // Reserved range start
        TUC_STATUS_RESERVED_END = 9999    // Reserved range end
    } TUC_STATUS;

    // =============================================================================
// 1. SDK Lifecycle Management
// =============================================================================

/* Length macro definitions */
#define TUC_FILENAME_MAX 256
#define TUC_INFO_LENGTH_MAX 128

    /**
     * @brief SDK build information structure
     * @remarks Contains SDK build and version information
     */
    typedef struct _TUC_BUILD_INFO
    {
        // 4-byte aligned fields - File Version
        uint32_t fileMajorVersion;    // File major version number
        uint32_t fileMinorVersion;    // File minor version number
        uint32_t fileRevisionVersion; // File revision version number
        uint32_t fileCustomNumber;    // File custom number

        // 4-byte aligned fields - Product Version (Timestamp)
        uint32_t productMajorVersion;    // Product major version number
        uint32_t productMinorVersion;    // Product minor version number
        uint32_t productRevisionVersion; // Product revision version number
        uint64_t productBuildNumber;     // Product build number (Timestamp)

        // Character array fields - Build Information
        char buildDate[TUC_INFO_LENGTH_MAX];    // Build date string
        char buildTime[TUC_INFO_LENGTH_MAX];    // Build time string
        char compilerInfo[TUC_INFO_LENGTH_MAX]; // Compiler information
        char platformInfo[TUC_INFO_LENGTH_MAX]; // Platform information

        // Character array fields - RC File Information
        char companyName[TUC_INFO_LENGTH_MAX];      // Company name
        char fileDescription[TUC_INFO_LENGTH_MAX];  // File description
        char internalName[TUC_INFO_LENGTH_MAX];     // Internal name
        char legalCopyright[TUC_INFO_LENGTH_MAX];   // Legal copyright
        char originalFilename[TUC_INFO_LENGTH_MAX]; // Original filename
        char productName[TUC_INFO_LENGTH_MAX];      // Product name

        char reserved[128]; // Reserved
    } TUC_BUILD_INFO;

    // =============================================================================
    // 2. Device Discovery & Management
    // =============================================================================

    /**
     * @brief Basic handle types for SDK objects
     * @remarks All handles are opaque pointers to internal structures
     */
    typedef void *TUC_HDEVICE;       // Device handle
    typedef void *TUC_HBUFFER;       // Buffer handle
    typedef void *TUC_HBUFFERS;      // Buffer pool handle
    typedef void *TUC_HRECORDER;     // Recorder handle
    typedef void *TUC_HSTORAGE;      // High-speed storage writer handle (Storage module)
    typedef void *TUC_HCONTEXT_DRAW; // Drawing context handle

    /**
     * @brief Basic device information structure
     * @remarks Contains basic information about a device
     * @warning This structure is frequently used in memcpy operations - maintain exact byte layout
     * @note All fields are char arrays, naturally 1-byte aligned, no padding needed
     */
    typedef struct _TUC_DEVINFO
    {
        char fullName[TUC_FILENAME_MAX];                // Full name (256 bytes)
        char modelName[TUC_INFO_LENGTH_MAX];            // Model name (128 bytes)
        char vendorName[TUC_INFO_LENGTH_MAX];           // Vendor name (128 bytes)
        char deviceClass[TUC_INFO_LENGTH_MAX];          // Device class (128 bytes)
        char serialNumber[TUC_INFO_LENGTH_MAX];         // Serial number (128 bytes)
        char friendlyName[TUC_INFO_LENGTH_MAX];         // Friendly name (128 bytes)
        char deviceVersion[TUC_INFO_LENGTH_MAX];        // Device version (128 bytes)
        char userDefinedName[TUC_INFO_LENGTH_MAX];      // User defined name (128 bytes)
        char transportLayerType[TUC_INFO_LENGTH_MAX];   // Transport layer type (128 bytes)
        char reserved[128];                             // Reserved (128 bytes)
    } TUC_DEVINFO;

    /**
     * @brief CL device detailed information structure
     * @remarks Contains CL-specific device information
     */
    typedef struct _TUC_DEVINFO_CL
    {
        uint32_t hostInterface; // Host interface type(1:USB30, 2:PCIex4)
        uint32_t boardIndex;    // Board index
        uint8_t reserved[256];  // Reserved
    } TUC_DEVINFO_CL;

    /**
     * @brief USB device detailed information structure
     * @remarks Contains USB-specific device information
     */
    typedef struct _TUC_DEVINFO_USB
    {
        uint32_t vendorId;                      // Device vendor ID
        uint32_t productId;                     // Product ID
        uint32_t usbVersion;                    // USB connection version
        char serialNumber[TUC_INFO_LENGTH_MAX]; // Device serial number
        char deviceGuid[TUC_INFO_LENGTH_MAX];   // Device GUID
        char reserved[128];                     // Reserved
    } TUC_DEVINFO_USB;

    /**
     * @brief CXP device detailed information structure
     * @remarks Contains CXP-specific device information
     */
    typedef struct _TUC_DEVINFO_CXP
    {
        uint32_t boardIndex;   // Board index
        uint32_t masterPort;   // Master port
        uint32_t slaveCount;   // Slave count
        uint32_t slavePort[7]; // Slave port array
        uint32_t cameraId;     // Camera ID
        uint8_t reserved[256]; // Reserved
    } TUC_DEVINFO_CXP;

    /**
     * @brief GigE device detailed information structure
     * @remarks Contains GigE-specific device information
     */
    typedef struct _TUC_DEVINFO_GIGE
    {
        char macAddress[TUC_INFO_LENGTH_MAX];     // Device MAC address
        char ipAddress[TUC_INFO_LENGTH_MAX];      // Device IP address
        char subnetMask[TUC_INFO_LENGTH_MAX];     // Subnet mask
        char gateway[TUC_INFO_LENGTH_MAX];        // Default gateway
        char nicMacAddress[TUC_INFO_LENGTH_MAX];  // NIC MAC address
        char nicIpAddress[TUC_INFO_LENGTH_MAX];   // NIC IP address
        char nicSubnetMask[TUC_INFO_LENGTH_MAX];  // NIC subnet mask
        char nicGateway[TUC_INFO_LENGTH_MAX];     // NIC default gateway
        char nicAdapterName[TUC_FILENAME_MAX];    // NIC adapter name
        char nicDescription[TUC_INFO_LENGTH_MAX]; // NIC description
        char reserved[128];                       // Reserved
    } TUC_DEVINFO_GIGE;

    /**
     * @brief Device identifier type enumeration
     * @remarks Defines the type of device identifier used for device opening
     */
    typedef enum _TUC_DEVICE_ID_TYPE
    {
        TUC_DEVICE_ID_SERIAL = 0, // Serial number identifier
        TUC_DEVICE_ID_USER = 1    // User-defined identifier
    } TUC_DEVICE_ID_TYPE;

    /**
     * @brief Device open policy enumeration
     * @remarks Defines the access policy when opening a device
     */
    typedef enum _TUC_OPEN_POLICY
    {
        TUC_OPEN_POLICY_READONLY = 0,   // Read-only access (not supported in current version)
        TUC_OPEN_POLICY_CONTROL = 1,    // Control access (read/write) (not supported in current version)
        TUC_OPEN_POLICY_EXCLUSIVE = 2   // Exclusive access (default, only supported policy)
    } TUC_OPEN_POLICY;

    // =============================================================================
    // 3. Feature Parameter Management - Enumeration Definitions First
    // =============================================================================

    /**
     * @brief Feature parameter type enumeration
     * @remarks Defines the data type of feature parameters
     */
    /**
     * @brief Feature parameter type enumeration
     * @remarks Defines the data type of feature parameters
     * @note The enum values are mapped to NodeInterfaceType for direct conversion
     */
    typedef enum _TUC_FEATURE_TYPE
    {
        TUC_FEATURE_TYPE_UNKNOWN = 0,    // Unknown type (maps to NodeInterfaceType::IVALUE/IBASE/ICATEGORY/IENUMENTRY/IPORT)
        TUC_FEATURE_TYPE_INTEGER = 2,    // Integer (maps to NodeInterfaceType::IINTEGER)
        TUC_FEATURE_TYPE_BOOL = 3,       // Boolean (maps to NodeInterfaceType::IBOOLEAN)
        TUC_FEATURE_TYPE_COMMAND = 4,    // Command (maps to NodeInterfaceType::ICOMMAND)
        TUC_FEATURE_TYPE_FLOAT = 5,      // Float (maps to NodeInterfaceType::IFLOAT)
        TUC_FEATURE_TYPE_STRING = 6,     // String (maps to NodeInterfaceType::ISTRING)
        TUC_FEATURE_TYPE_REGISTER = 7,   // Register (maps to NodeInterfaceType::IREGISTER)
        TUC_FEATURE_TYPE_ENUM = 9        // Enumeration (maps to NodeInterfaceType::IENUMERATION)
    } TUC_FEATURE_TYPE;

    /**
     * @brief Feature access mode enumeration
     * @remarks Defines the access permissions for feature parameters
     */
    typedef enum _TUC_FEATURE_ACCESS
    {
        TUC_ACCESS_NOT_IMPLEMENTED = 0, // Not implemented (NI - Not implemented)
        TUC_ACCESS_NOT_AVAILABLE = 1,   // Not available (NA - Not available)
        TUC_ACCESS_WRITE_ONLY = 2,      // Write only (WO - Write Only)
        TUC_ACCESS_READ_ONLY = 3,       // Read only (RO - Read Only)
        TUC_ACCESS_READ_WRITE = 4       // Read and write (RW - Read and Write)
    } TUC_FEATURE_ACCESS;

    /**
     * @brief Feature visibility level enumeration
     * @remarks Defines the visibility level of feature parameters
     */
    typedef enum _TUC_FEATURE_VISIBILITY
    {
        TUC_VISIBILITY_BEGINNER = 0,  // Beginner visible
        TUC_VISIBILITY_EXPERT = 1,    // Expert visible
        TUC_VISIBILITY_GURU = 2,      // Guru visible
        TUC_VISIBILITY_INVISIBLE = 3, // Invisible
        TUC_VISIBILITY_UNDEFINED = 99 // Undefined
    } TUC_FEATURE_VISIBILITY;

    /**
     * @brief Feature representation type enumeration
     * @remarks Defines how feature parameters should be displayed in UI
     */
    typedef enum _TUC_FEATURE_REPRESENTATION
    {
        TUC_REPRESENTATION_LINEAR = 0,      // Linear slider
        TUC_REPRESENTATION_LOGARITHMIC = 1, // Logarithmic slider
        TUC_REPRESENTATION_BOOLEAN = 2,     // Checkbox
        TUC_REPRESENTATION_PURENUMBER = 3,  // Decimal number edit box
        TUC_REPRESENTATION_HEXNUMBER = 4,   // Hexadecimal number edit box
        TUC_REPRESENTATION_IPV4ADDRESS = 5, // IP address format
        TUC_REPRESENTATION_MACADDRESS = 6,  // MAC address format
        TUC_REPRESENTATION_UNDEFINED = 99   // Undefined
    } TUC_FEATURE_REPRESENTATION;

    /**
     * @brief Feature caching mode enumeration
     * @remarks Defines the caching behavior for feature parameters
     */
    typedef enum _TUC_FEATURE_CACHING_MODE
    {
        TUC_CACHING_MODE_NO_CACHE = 0,      // No cache
        TUC_CACHING_MODE_WRITE_THROUGH = 1, // Write to cache and register
        TUC_CACHING_MODE_WRITE_AROUND = 2,  // Write to register, read to cache
        TUC_CACHING_MODE_UNDEFINED = 99     // Undefined
    } TUC_FEATURE_CACHING_MODE;

    /**
     * @brief Feature parameter information structure
     * @remarks Contains detailed information about a feature parameter
     */
    typedef struct _TUC_FEATURE_INFO
    {
        // 8-byte aligned fields (Most frequently accessed)
        char name[TUC_INFO_LENGTH_MAX];        // Feature name (Primary identifier)
        char displayName[TUC_INFO_LENGTH_MAX]; // Display name (UI display)

        // 4-byte aligned fields (Core properties - High access frequency)
        TUC_FEATURE_TYPE featureType;              // Feature type
        TUC_FEATURE_ACCESS accessMode;             // Access mode
        TUC_FEATURE_VISIBILITY visibility;         // Visibility level
        TUC_FEATURE_REPRESENTATION representation; // Representation
        TUC_FEATURE_CACHING_MODE cachingMode;      // Caching mode
        uint32_t pollingTime;                      // Polling time (ms)
        int32_t treeLevel;                         // Node depth level in feature tree (0 = root)

        // 1-byte aligned fields (Boolean flags - Medium access frequency)
        bool isFeature;    // Is this a feature node
        bool isStreamable; // Is this node streamable
        bool isDeprecated; // Is this node deprecated
        bool isCachable;   // Is this node cachable

        // Character array fields (Secondary information - Lower access frequency)
        char description[TUC_FILENAME_MAX]; // Feature description
        char unit[TUC_INFO_LENGTH_MAX];     // Unit
        char toolTip[TUC_INFO_LENGTH_MAX];  // Tool tip text
        char docURL[TUC_INFO_LENGTH_MAX];   // Documentation URL
        char eventID[TUC_INFO_LENGTH_MAX];  // Event ID

        // Reserved space for future extensions
        char reserved[124]; // Reserved
    } TUC_FEATURE_INFO;

    // =============================================================================
    // 4. Feature Value Access (Numeric)
    // =============================================================================

    /**
     * @brief Feature integer value and range information structure
     * @remarks Contains complete value and range information for integer features,
     *          including current value, default value, minimum, maximum, and increment
     * @note This structure provides all necessary information for UI control creation,
     *       value validation, and parameter management in a single call
     */
    typedef struct _TUC_FEATURE_INT64_INFO
    {
        // 8-byte aligned fields (Core value and range information)
        int64_t currentValue;  // Current value of the feature
                               // (Reference GenICam IInteger::GetValue)
                               // Note: May change between calls, use for UI initialization
        
        int64_t minimum;      // Minimum allowed value
                               // (Reference GenICam IInteger::GetMin)
        
        int64_t maximum;      // Maximum allowed value
                               // (Reference GenICam IInteger::GetMax)
        
        int64_t increment;    // Increment step value
                               // (Reference GenICam IInteger::GetInc)
        
        int64_t defaultValue; // Default value (typically constant)
        
        // Reserved space for future extensions
        int64_t reserved[3];   // Reserved
    } TUC_FEATURE_INT64_INFO;

    /**
     * @brief Feature float value and range information structure
     * @remarks Contains complete value and range information for floating-point features,
     *          including current value, default value, minimum, maximum, and increment
     * @note This structure provides all necessary information for UI control creation,
     *       value validation, and parameter management in a single call
     */
    typedef struct _TUC_FEATURE_FLOAT_INFO
    {
        // 8-byte aligned fields (Core value and range information)
        double currentValue;    // Current value of the feature
                                // (Reference GenICam IFloat::GetValue)
                                // Note: May change between calls, use for UI initialization
        
        double minimum;         // Minimum allowed value
                                // (Reference GenICam IFloat::GetMin)
        
        double maximum;         // Maximum allowed value
                                // (Reference GenICam IFloat::GetMax)
            
        double increment;       // Increment step value， if hasIncrement is false, increment is 0
                                // (Reference GenICam IFloat::GetInc)
        
        double defaultValue;    // Default value (typically constant)
        
        // Reserved space for future extensions
        double reserved[3];   // Reserved
    } TUC_FEATURE_FLOAT_INFO;

    // =============================================================================
    // 5. Feature Value Access (Boolean/String)
    // =============================================================================

    // (This section would contain boolean/string feature access definitions)

    // =============================================================================
    // 6. Feature Value Access (Enumeration)
    // =============================================================================

    /**
     * @brief Feature enumeration entry structure
     * @remarks Contains information about a single enumeration value
     * @note Reference GenICam IEnumEntry interface - only essential fields
     */
    typedef struct _TUC_FEATURE_ENUM_ENTRY
    {
        int64_t value;          // Entry value 
                                // (Reference GenICam IEnumEntry::GetValue)

        double numericValue;    // Numeric representation
                                // (Reference GenICam IEnumEntry::GetNumericValue)  

        char symbolic[TUC_INFO_LENGTH_MAX]; // Entry name/Symbolic
                                            // (Reference GenICam IEnumEntry::GetSymbolic)

        char reserved[64];      // Reserved for future use
    } TUC_FEATURE_ENUM_ENTRY;

    /**
     * @brief Feature enumeration information structure
     * @remarks Contains complete information about an enumeration feature
     * @note Reference GenICam IEnumeration interface - only essential fields
     */
    typedef struct _TUC_FEATURE_ENUM_INFO
    {
        // 8-byte aligned fields 
        int64_t currentIntValue; // Current integer value
                                 // (Reference GenICam IEnumeration::GetIntValue)

        TUC_FEATURE_ENUM_ENTRY *pEntries; // Entry array pointer
                                          // (Reference GenICam IEnumeration::GetEntries)

        // 4-byte aligned fields 
        uint32_t entryCount; // Entry count

        // Character array fields
        char currentSymbolic[TUC_INFO_LENGTH_MAX]; // Current value/Symbolic
                                                   // (Reference GenICam IEnumeration::operator*)

        char reserved[64]; // Reserved for future use
    } TUC_FEATURE_ENUM_INFO;

    // =============================================================================
    // 7. Image Acquisition Control
    // =============================================================================

    /**
     * @brief Acquisition start flags enumeration
     * @remarks Defines flags for starting image acquisition
     */
    typedef enum _TUC_ACQ_START_FLAGS
    {
        TUC_ACQ_START_FLAGS_DEFAULT = 0 // Default behavior
    } TUC_ACQ_START_FLAGS;

    /**
     * @brief Acquisition stop flags enumeration
     * @remarks Defines flags for stopping image acquisition
     */
    typedef enum _TUC_ACQ_STOP_FLAGS
    {
        TUC_ACQ_STOP_FLAGS_DEFAULT = 0, // Default behavior: Stop after completing current task
        TUC_ACQ_STOP_FLAGS_KILL = 1     // Immediate stop: Stop immediately and keep buffer
    } TUC_ACQ_STOP_FLAGS;

    /**
     * @brief Acquisition statistics structure
     * @remarks Contains statistics about image acquisition performance
     */
    typedef struct _TUC_ACQUISITION_STATISTICS
    {
        uint64_t numAnnounced;     // Number of announced buffers
        uint64_t numStarted;       // Number of started buffers
        uint64_t numQueued;        // Number of queued buffers
        uint64_t numDelivered;     // Number of delivered buffers
        uint64_t numAwaitDelivery; // Number of await delivery buffers
        uint64_t numUnderrun;      // Number of underrun buffers
        uint64_t numIncomplete;    // Number of incomplete buffers
        uint8_t reserved[128];     // Reserved
    } TUC_ACQUISITION_STATISTICS;

    // =============================================================================
    // 8. Buffer Management
    // =============================================================================

    /**
     * @brief Acquisition queue type enumeration
     * @remarks Defines different queue types for buffer management
     * @note Aligned with GenTL::ACQ_QUEUE_TYPE standard
     */
    typedef enum _TUC_ACQ_QUEUE_TYPE
    {
        TUC_ACQ_QUEUE_TYPE_INPUT_TO_OUTPUT = 0,     // Flushes input pool to output queue
        TUC_ACQ_QUEUE_TYPE_OUTPUT_DISCARD = 1,      // Discards all buffers in output queue
        TUC_ACQ_QUEUE_TYPE_ALL_TO_INPUT = 2,        // Puts all buffers in input pool
        TUC_ACQ_QUEUE_TYPE_UNQUEUED_TO_INPUT = 3,   // Puts unqueued buffers in input pool
        TUC_ACQ_QUEUE_TYPE_ALL_DISCARD = 4,         // Discards all buffers in input pool and output queue

        TUC_ACQ_QUEUE_TYPE_CUSTOM_ID = 1000         // Starting value for custom IDs
    } TUC_ACQ_QUEUE_TYPE;

    /**
     * @brief Buffer state enumeration
     * @remarks Defines the current state of a buffer
     */
    typedef enum _TUC_BUFFER_STATE
    {
        TUC_BUFFER_STATE_INPUT = 0,      // In input pool
        TUC_BUFFER_STATE_ACQUIRING = 1,  // Acquiring
        TUC_BUFFER_STATE_OUTPUT = 2,     // In output queue
        TUC_BUFFER_STATE_INCOMPLETE = 3, // Incomplete
        TUC_BUFFER_STATE_ERROR = 4       // Error state
    } TUC_BUFFER_STATE;

    /**
     * @brief Buffer queue policy enumeration
     * @remarks Defines the buffer queue management policy
     */
    typedef enum _TUC_BUFFER_QUEUE_POLICY
    {
        TUC_BUFFER_QUEUE_POLICY_AUTO = 0,       // Automatic queue management (default)
        TUC_BUFFER_QUEUE_POLICY_MANUAL = 1      // Manual queue management
    } TUC_BUFFER_QUEUE_POLICY;

    /**
     * @brief Buffer information structure
     * @remarks Contains comprehensive information about a buffer, aligned with GenICam standard
     * @note Fields are organized by GenICam BUFFER_INFO_CMD_LIST order and access frequency
     */
    typedef struct _TUC_BUFFER_INFO
    {
        // Core Buffer Information (8-byte aligned, high access frequency)
        void *pBuffer;        // Buffer base address (GenICam BUFFER_INFO_BASE)
        uint64_t timestamp;   // Acquisition timestamp (GenICam BUFFER_INFO_TIMESTAMP)
        uint64_t timestampNS; // Timestamp in nanoseconds (GenICam BUFFER_INFO_TIMESTAMP_NS)
        uint64_t frameId;     // Frame identifier (GenICam BUFFER_INFO_FRAMEID)
        uint64_t sizeFilled;  // Bytes written to buffer (GenICam BUFFER_INFO_SIZE_FILLED)
        uint64_t dataSize;    // Actual data size (GenICam BUFFER_INFO_DATA_SIZE)

        // Buffer Dimensions and Layout (4-byte aligned, medium access frequency)
        uint32_t bufferSize;           // Total buffer size in bytes (GenICam BUFFER_INFO_SIZE)
        uint32_t imageWidth;           // Image width in pixels (GenICam BUFFER_INFO_WIDTH)
        uint32_t imageHeight;          // Image height in pixels (GenICam BUFFER_INFO_HEIGHT)
        uint32_t xOffset;              // X offset in pixels (GenICam BUFFER_INFO_XOFFSET)
        uint32_t yOffset;              // Y offset in pixels (GenICam BUFFER_INFO_YOFFSET)
        uint32_t xPadding;             // X padding in pixels (GenICam BUFFER_INFO_XPADDING)
        uint32_t yPadding;             // Y padding in pixels (GenICam BUFFER_INFO_YPADDING)
        uint32_t imageOffset;          // Image data offset (GenICam BUFFER_INFO_IMAGEOFFSET)
        uint32_t deliveredImageHeight; // Actual transmitted height 
                                       // (GenICam BUFFER_INFO_DELIVERED_IMAGEHEIGHT)

        // Format and Payload Information (4-byte aligned, medium access frequency)
        uint64_t pixelFormat;          // Pixel format (GenICam BUFFER_INFO_PIXELFORMAT)
        uint32_t payloadType;          // Payload type (GenICam BUFFER_INFO_PAYLOADTYPE)
        uint64_t pixelFormatNamespace; // Pixel format namespace 
                                       // (GenICam BUFFER_INFO_PIXELFORMAT_NAMESPACE)
        uint64_t chunkLayoutId; // Chunk layout identifier (GenICam BUFFER_INFO_CHUNKLAYOUTID)
        uint32_t deliveredChunkPayloadSize; // Chunk payload size 
                                            // (GenICam BUFFER_INFO_DELIVERED_CHUNKPAYLOADSIZE)

        // Buffer State Flags (1-byte aligned, high access frequency)
        bool isNewData;    // Contains new data (GenICam BUFFER_INFO_NEW_DATA)
        bool isQueued;     // In input pool or output queue (GenICam BUFFER_INFO_IS_QUEUED)
        bool isAcquiring;  // Currently being filled (GenICam BUFFER_INFO_IS_ACQUIRING)
        bool isIncomplete; // Filled with errors (GenICam BUFFER_INFO_IS_INCOMPLETE)
        bool imagePresent; // Contains image data (GenICam BUFFER_INFO_IMAGEPRESENT)
        bool isDataLargerThanBuffer; // Data exceeds buffer size (GenICam
                                     // BUFFER_INFO_DATA_LARGER_THAN_BUFFER)
        bool containsChunkData;      // Contains chunk data (GenICam BUFFER_INFO_CONTAINS_CHUNKDATA)
        bool isComposite;            // Multi-part buffer (GenICam BUFFER_INFO_IS_COMPOSITE)
        bool isPixelLittleEndian;    // Pixel endianness (GenICam BUFFER_INFO_PIXEL_ENDIANNESS)

        // Multi-part Buffer Information (4-byte aligned, low access frequency)
        uint32_t numberOfParts; // Number of buffer parts
        uint32_t partIndex;     // Current part index (for multi-part buffers)

        // String Information (character arrays, low access frequency)
        char tlType[TUC_INFO_LENGTH_MAX]; // Transport layer type (GenICam BUFFER_INFO_TLTYPE)
        char filename[TUC_FILENAME_MAX];  // Associated filename (GenICam BUFFER_INFO_FILENAME)

        char reserved[64]; // Reserved for future use
    } TUC_BUFFER_INFO;

    /**
     * @brief Max channel count for TucGetPixelValuesAt (Mono=1, RGB/RGBA up to 4).
     */
#ifndef TUC_PIXEL_VALUES_MAX_CH
#define TUC_PIXEL_VALUES_MAX_CH 4
#endif

    /**
     * @brief Pixel sample at (x,y) on current ISP sink image (interleaved 8/16-bit per channel).
     * @remarks Valid entries are v[0]..v[channelCount-1]. Channel order matches memory layout
     *          for the reported PFNC (e.g. RGB/BGR as stored after decode).
     */
    typedef struct _TUC_PIXEL_VALUES
    {
        uint64_t pfnc;                              /**< GenICam PFNC (aligned with sink ImageInfo format/matrixFormat). */
        uint32_t channelCount;                      /**< Valid channels: 1..TUC_PIXEL_VALUES_MAX_CH. */
        uint32_t values[TUC_PIXEL_VALUES_MAX_CH];   /**< Component values; only first channelCount used. */
        uint8_t reserved[32];                       /**< Reserved for future use. */
    } TUC_PIXEL_VALUES;

    // =============================================================================
    // 9. Buffer Advanced Operations
    // =============================================================================

    /**
     * @brief Conversion policy enumeration
     * @remarks Defines the conversion policy for buffer data
     */
    typedef enum _TUC_CONVERSION_POLICY
    {
        TUC_CONVERSION_POLICY_BASIC = 0, // Basic conversion policy
        TUC_CONVERSION_POLICY_FULL = 1   // Full conversion policy
    } TUC_CONVERSION_POLICY;

    /**
     * @brief Save format enumeration
     * @remarks Defines supported image save formats for TUC_SAVE_OPTIONS
     */
    typedef enum _TUC_SAVE_FORMAT
    {
        TUC_SAVE_FORMAT_UNKNOWN = 0x00, // None format
        TUC_SAVE_FORMAT_RAW = 0x01,     // RAW format
        TUC_SAVE_FORMAT_TIFF = 0x02,    // TIFF format
        TUC_SAVE_FORMAT_PNG = 0x04,     // PNG format
        TUC_SAVE_FORMAT_JPEG = 0x08,    // JPEG format
        TUC_SAVE_FORMAT_BMP = 0x10      // BMP format
    } TUC_SAVE_FORMAT;

    /**
     * @brief Save options structure
     * @remarks Contains options for saving buffer data to file
     */
    typedef struct _TUC_SAVE_OPTIONS
    {
        // 4-byte aligned fields
        uint32_t format;      // Save format (use TUC_SAVE_FORMAT enum values)
        uint32_t quality;     // Reserved Compression quality (1-100)
        uint32_t compression; // Reserved Compression type
       
        // Character array fields
        uint8_t reserved[128]; // Reserved
    } TUC_SAVE_OPTIONS;

    /** @brief TUC_STORAGE_OPTIONS.version must match this value for the layout below */
    #define TUC_STORAGE_OPTIONS_VERSION 1u

    /**
     * @brief High-speed storage format (numeric values match tu::modules::storage::StorageFormat)
     */
    typedef enum _TUC_STORAGE_FMT
    {
        TUC_STORAGE_FMT_RAW = 0,         // single-file RAW
        TUC_STORAGE_FMT_TIFF = 1,        // single-file TIFF
        TUC_STORAGE_FMT_RAW_STACK = 2,   // RAW stack
        TUC_STORAGE_FMT_TIFF_STACK = 3   // TIFF stack
    } TUC_STORAGE_FMT;

    /**
     * @brief RAW stack on-disk layout (numeric values match tu::modules::storage::RawStackFormat, 0-based)
     * @remarks See Storage module docs for HEADER_DATA_* packing details.
     */
    typedef enum _TUC_RAW_STACK_LAYOUT
    {
        TUC_RAW_STACK_HDHDP = 0, // HEADER_DATA_HEADER_DATA_PADDING
        TUC_RAW_STACK_HDDP  = 1  // HEADER_DATA_DATA_PADDING
    } TUC_RAW_STACK_LAYOUT;

    /** Default TIFF resolution (dpi) when @a tiffDpiX / @a tiffDpiY are 0 */
    #define TUC_TIFF_DPI_DEFAULT 96u

    /**
     * @brief Options for TucStorageWriterCreate (Modules/Storage)
     * @remarks Total struct size unchanged from v1 (276 bytes): @a tiffDpiX/@a tiffDpiY reuse first 8 bytes
     *          of the former @a reserved prefix; older callers zeroing the full reserved region get dpi 0,0 → default 96.
     */
    typedef struct _TUC_STORAGE_OPTIONS
    {
        uint32_t version;         // TUC_STORAGE_OPTIONS_VERSION
        uint32_t format;          // TUC_STORAGE_FMT_*
        uint32_t maxStackFrames;  // Stack: max frames (0 = module default)
        uint32_t maxStackFileMB;  // Stack: file size cap in MB (0 = module default)
        uint32_t rawStackLayout;  // TUC_RAW_STACK_* when format is RAW (stack)
        uint32_t tiffDpiX;        // TIFF: X resolution (dpi); 0 = TUC_TIFF_DPI_DEFAULT; ignored for non-TIFF formats
        uint32_t tiffDpiY;        // TIFF: Y resolution (dpi); 0 = use tiffDpiX if non-zero, else TUC_TIFF_DPI_DEFAULT
        uint8_t reserved[128];    // Set to zero; future optional TLV/binary layout may use this region 
    } TUC_STORAGE_OPTIONS;

    /** @brief TUC_STORAGE_WRITE_OPTIONS structure version */
    #define TUC_STORAGE_WRITE_OPTIONS_VERSION 1u

    /**
     * @brief Write operation mode for TucStorageWrite (pWriteOpts->mode)
     * @remarks Stack capacity uses maxStackFrames / maxStackFileMB from TucStorageWriterCreate options.
     */
    typedef enum _TUC_STORAGE_WRITE_MODE
    {
        TUC_STORAGE_WRITE_SINGLE = 0,       // One file, one frame; filename required; exclusive with active stack
        TUC_STORAGE_WRITE_STACK_FRAME = 1,  // Append stack frame; first call needs filename (stack path)
        TUC_STORAGE_WRITE_STACK_FINISH = 2  // Close stack; pImageInfo may be NULL; filename ignored
    } TUC_STORAGE_WRITE_MODE;

    /**
     * @brief Per-call options for TucStorageWrite
     * @remarks Unused bytes must be zero. version must be TUC_STORAGE_WRITE_OPTIONS_VERSION.
     */
    typedef struct _TUC_STORAGE_WRITE_OPTIONS
    {
        uint32_t version; // TUC_STORAGE_WRITE_OPTIONS_VERSION
        uint32_t mode;    // TUC_STORAGE_WRITE_MODE
        uint8_t reserved[56];
    } TUC_STORAGE_WRITE_OPTIONS;

    /**
     * @brief Image information structure
     * @remarks Contains complete image metadata and pointers to image data
     * @note Structure size is exactly 1024 bytes for ABI compatibility
     * @warning Pointers are valid only within the context of the image object
     * @warning This structure is frequently used in memcpy operations - maintain exact byte layout
     * @warning Pointer sizes vary between 32/64-bit platforms - reserved space accounts for this
     * @note Optimized field ordering for 4-byte alignment and cache efficiency
     */
    typedef struct _TUC_IMAGE_INFO
    {
        // Fixed-size header (12 bytes) - naturally aligned
        uint8_t signature[8]; // Copyright signature string (8 bytes)
        uint32_t version;     // Version number (4 bytes) - 4-byte aligned

        // Pixel format info (4 bytes) - naturally aligned
        uint8_t channels;        // Number of image channels (1 byte)
        uint8_t bitDepth;        // Image bit depth (reference MATRIX_8U like CV_8U) (1 byte)
        uint8_t bitDepthValid;   // Valid bit depth of image data (1 byte)
        uint8_t bytesPerElement; // Bytes per element in image data (1 byte)

        // Image geometry and format (68 bytes = 17 * 4) - all 4-byte aligned
        uint32_t headerSize;  // Header information length
        uint32_t width;       // Image width
        uint32_t height;      // Image height
        uint32_t offsetX;     // Horizontal offset
        uint32_t offsetY;     // Vertical offset
        uint32_t paddingX;    // Horizontal padding data
        uint32_t paddingY;    // Vertical padding data
        uint32_t format;      // Image data format (reference TUIMG_FORMAT)
        uint32_t widthStep;   // Image width step
        uint32_t bytesCount;  // Total bytes occupied by image
        uint32_t bytesChunk;  // Total bytes occupied by chunk data
        uint32_t payloadSize; // Payload size
        uint32_t payloadType; // Payload type
        uint32_t matrixType;  // Matrix transformation type (reference TransformType)
        uint32_t matrixFormat;  // Image data format after matrix transformation 
                                // (mainly for Bayer format)
        uint32_t histogramSize; // Histogram data length
        uint32_t numChunks;     // Number of chunk data blocks

        // Pointer fields (platform-dependent size: 12 or 24 bytes) - 8-byte aligned on 64-bit
        uint8_t *data;      // Pointer to image data memory start position
        uint8_t *chunk;     // Pointer to image chunk data memory start position
        uint8_t *histogram; // Pointer to image histogram statistics data memory start position

        // Timing and metadata (48 bytes = 6 * 8) - all 8-byte aligned
        uint64_t frameId;        // Frame ID of image data
        uint64_t gain;           // Current gain of image data
        uint64_t exposure;       // Current exposure of image data
        uint64_t timestamp;      // Current timestamp of image data
        uint64_t timestampLast;  // Previous frame timestamp of image data
        uint64_t timestampDelta; // Inter-frame timestamp difference of image data

        /* Reserved space calculation:
         * Total target: 1024 bytes
         * Fixed part: 8 + 4 + 4 + 68 + 48 = 132 bytes
         * Pointer part: 12 bytes (32-bit) or 24 bytes (64-bit)
         * Alignment padding: 0 bytes (32-bit) or 8 bytes (64-bit)
         * Platform-specific reserved space:
         * - 32-bit: 1024 - 144 = 880 bytes
         * - 64-bit: 1024 - 160 = 864 bytes
         * Note: Platform-specific alignment ensures proper pointer handling
         */
#if defined(_WIN64) || defined(__x86_64__) || defined(__aarch64__) || (defined(__SIZEOF_POINTER__) && __SIZEOF_POINTER__ == 8)
        uint8_t reserved[864]; // 64-bit platforms: 1024 - 160 = 864 bytes
#else
        uint8_t reserved[880]; // 32-bit platforms: 1024 - 144 = 880 bytes
#endif
    } TUC_IMAGE_INFO;

    /* Compile-time verification that TUC_IMAGE_INFO is exactly 1024 bytes */
    TUC_STATIC_ASSERT(sizeof(TUC_IMAGE_INFO) == 1024, TUC_IMAGE_INFO_must_be_1024_bytes);

    // =============================================================================
    // 10. Event Handling
    // =============================================================================

    /**
     * @brief Event type enumeration
     * @remarks Defines system events that can be monitored
     */
    typedef enum _TUC_EVENT_TYPE
    {
        // Device status events
        TUC_EVENT_TYPE_DEVICE_ERROR = 0,    // Device error
        TUC_EVENT_TYPE_DEVICE_DISCONNECTED = 1, // Device disconnected
        TUC_EVENT_TYPE_TEMPERATURE_WARNING = 2, // Temperature warning

        // Core acquisition events (Corresponds to GenICam standard)
        TUC_EVENT_TYPE_ACQUISITION_START = 3, // Acquisition start
        TUC_EVENT_TYPE_ACQUISITION_STOP = 4,  // Acquisition end
        TUC_EVENT_TYPE_ACQUISITION_ERROR = 5, // Acquisition error (Corresponds to EVENT_ERROR)

        // Buffer management events
        TUC_EVENT_TYPE_BUFFER_OVERFLOW = 6,   // Buffer overflow
        TUC_EVENT_TYPE_BUFFER_UNDERRUN = 7,   // Buffer underrun
        TUC_EVENT_TYPE_BUFFER_INCOMPLETE = 8, // Buffer incomplete
        TUC_EVENT_TYPE_FEATURE_CHANGE = 9     // Feature change (Corresponds to GenICam)
    } TUC_EVENT_TYPE;

    /**
     * @brief Event data structure
     * @remarks Contains comprehensive event information including timestamp and additional data
     */
    typedef struct _TUC_EVENT_DATA
    {
        TUC_EVENT_TYPE eventType;        // Event type
        uint64_t timestamp;              // Event timestamp (microseconds since device start)
        uint32_t eventId;                // Event ID (corresponds to GenICam EventID)
        uint32_t sequenceNumber;         // Event sequence number for ordering
        uint32_t dataSize;               // Size of additional event data
        void *pEventData;                // Pointer to additional event data (can be NULL)
        TUC_HDEVICE deviceHandle;        // Device handle that generated the event
    } TUC_EVENT_DATA;

    /**
     * @brief Event callback function type
     * @param pEventData [IN] Pointer to event data structure
     * @param pUserContext [IN] User context pointer
     */
    typedef void(TUC_CALLBACK *TUC_EVENT_CALLBACK)(
        const TUC_EVENT_DATA *pEventData, void *pUserContext
    );

    /**
     * @brief Event statistics structure
     */
    typedef struct _TUC_EVENT_STATISTICS
    {
        uint64_t totalEvents;       // Total events received
        uint64_t filteredEvents;    // Events passed through internal filter
        uint64_t droppedEvents;     // Events dropped due to sampling or rate limiting
        uint8_t reserved[128];      // Reserved
    } TUC_EVENT_STATISTICS;

    /**
     * @brief Buffer callback function type
     * @param handle [IN] Device handle
     * @param bufferHandle [IN] Buffer handle containing buffer data
     * @param pUserContext [IN] User context pointer
     */
    typedef void(TUC_CALLBACK *TUC_BUFFER_CALLBACK)(
        TUC_HDEVICE handle, TUC_HBUFFER bufferHandle, void *pUserContext
    );

    // =============================================================================
    // 11. Video Recording
    // =============================================================================

    /**
     * @brief Recorder dimension configuration structure
     * @remarks Contains dimension configuration parameters for video recording
     */
    typedef struct _TUC_RECORDER_DIMENSION
    {
        // 4-byte aligned fields
        uint32_t width;       // Video width
        uint32_t height;      // Video height
        uint32_t pixelFormat; // Pixel format
    } TUC_RECORDER_DIMENSION;

    /**
     * @brief Recorder configuration structure
     * @remarks Contains configuration parameters for video recording
     */
    typedef struct _TUC_RECORDER_CONFIG
    {
        double frameRate;   // Frame rate (fps)
        uint32_t quality;   // Quality setting (1-100)

        // Character array fields
        char videoCodec[TUC_INFO_LENGTH_MAX];      // Video codec
        char containerFormat[TUC_INFO_LENGTH_MAX]; // Container format
        char reserved[128];                        // Reserved
    } TUC_RECORDER_CONFIG;

    // =============================================================================
    // 12. Logging & Profile Management
    // =============================================================================

    /**
     * @brief Log level enumeration
     * @remarks Defines logging levels for debug output
     */
    typedef enum _TUC_LOG_LEVEL
    {
        TUC_LOG_LEVEL_TRACE = 0,        // Trace level
        TUC_LOG_LEVEL_DEBUG = 1,        // Debug level
        TUC_LOG_LEVEL_INFO = 2,         // Info level
        TUC_LOG_LEVEL_WARNING = 3,      // Warning level
        TUC_LOG_LEVEL_ERROR = 4,        // Error level
        TUC_LOG_LEVEL_CRITICAL = 5,     // Critical level
        TUC_LOG_LEVEL_OFF = 6           // Log off
    } TUC_LOG_LEVEL;

    /**
     * @brief Log file policy enumeration
     * @remarks Defines how log files should be managed
     */
    typedef enum _TUC_LOG_FILE_POLICY
    {
        TUC_LOG_FILE_OVERWRITE = 0,     // Overwrite old file on each startup
        TUC_LOG_FILE_APPEND = 1,        // Append to end of old file
        TUC_LOG_FILE_ROTATE_SIZE = 2,   // Auto-rotate by file size (e.g. 10MB/volume)
    } TUC_LOG_FILE_POLICY;

    /**
     * @brief Logger configuration structure
     * @remarks Contains logger configuration parameters including file policy and rotation settings
     */
    typedef struct _TUC_LOGGER_CONFIG
    {
        TUC_LOG_FILE_POLICY filePolicy; // Log file management policy

        uint32_t maxFileSizeMB; // Maximum file size in MB 
                                // (for TUC_LOG_FILE_ROTATE_SIZE mode)

        uint32_t maxFileCount;  // Maximum number of log files to keep 
                                // (for TUC_LOG_FILE_ROTATE_SIZE mode)

        uint8_t reserved[128];  // Reserved for future use
    } TUC_LOGGER_CONFIG;

    /**
     * @brief Profile operation policy
     * @remarks Used for profile save/load/config operations
     */
    typedef enum _TUC_PROFILE_POLICY
    {
        TUC_PROFILE_POLICY_ONLY_MODEL = 0, // Only model policy
        TUC_PROFILE_POLICY_WITH_SERIAL = 1 // With serial policy
    } TUC_PROFILE_POLICY;

    /**
     * @brief User profile file format (explicit path import/export)
     */
    typedef enum _TUC_PROFILE_FILE_FORMAT
    {
        TUC_PROFILE_FILE_FORMAT_XML = 0,   // XML schema v1 (default)
        TUC_PROFILE_FILE_FORMAT_JSON = 1   // Reserved; not implemented yet
    } TUC_PROFILE_FILE_FORMAT;

    // =============================================================================
    // 13. Image Drawing
    // =============================================================================

    /**
     * @brief Drawing dimension structure
     * @remarks Contains dimension information for image drawing and display
     */
    typedef struct _TUC_DRAW_DIMENSION
    {
        int32_t srcX;      // Source X
        int32_t srcY;      // Source Y
        int32_t srcWidth;  // Source width
        int32_t srcHeight; // Source height

        int32_t dstX;      // Destination X
        int32_t dstY;      // Destination Y
        int32_t dstWidth;  // Destination width
        int32_t dstHeight; // Destination height
    } TUC_DRAW_DIMENSION;

    typedef struct _TUC_DRAW_STATISTICS
    {
        double fps;            // Frames per second
        uint64_t frameCount;   // Frame count
        uint64_t lastDrawTime; // Last draw time
    } TUC_DRAW_STATISTICS;

    // =============================================================================
    // 14. CPU Performance Optimization
    // =============================================================================

    /**
     * @brief CPU optimization policy enumeration (process scope)
     * @remarks Applies to the entire SDK process. Multi-device plans use per-device slots in the
     *          affinity plan; per-device policy via hDevice is reserved for a future release.
     */
    typedef enum _TUC_CPU_OPTIMIZATION_POLICY
    {
        TUC_CPU_POLICY_DISABLED = 0,    // Disable optimization
        TUC_CPU_POLICY_AUTO = 1,        // Automatic optimization (recommended)
        TUC_CPU_POLICY_MANUAL = 2       // Manual mode
    } TUC_CPU_OPTIMIZATION_POLICY;

    /**
     * @brief Thread type enumeration (process plan + manual override)
     * @remarks Bindings are published in a process-wide CPU affinity plan (keyed by device serial).
     *          hDevice in TucSetThreadCoreAndPriority selects the plan slot; device-only policy API is reserved.
     */
    typedef enum _TUC_THREAD_TYPE
    {
        TUC_THREAD_TYPE_GRAB = 0,       // Data acquisition thread
        TUC_THREAD_TYPE_CALLBACK = 1,   // External callback thread (no separate affinity slot)
        TUC_THREAD_TYPE_PIPELINE = 2,   // Image processing / UI wait thread
        TUC_THREAD_TYPE_CTI_XFER = 3,   // CTI bulk transfer thread (plan override; CTI applies)
        TUC_THREAD_TYPE_CTI_AUX = 4,    // CTI U3V event receiver thread (plan override; CTI applies)
        TUC_THREAD_TYPE_STORAGE = 5,    // Storage write caller thread (bind externally, like PIPELINE)
    } TUC_THREAD_TYPE;

    /**
     * @brief Thread priority enumeration
     * @remarks Defines thread scheduling priority levels
     */
    typedef enum _TUC_THREAD_PRIORITY
    {
        TUC_PRIORITY_NORMAL = 0,       // Normal priority
        TUC_PRIORITY_HIGH = 1,         // High priority
        TUC_PRIORITY_REALTIME = 2,     // Real-time priority
        TUC_PRIORITY_TIME_CRITICAL = 3 // Time-critical priority
    } TUC_THREAD_PRIORITY;

    /**
     * @brief CPU capabilities information structure
     * @remarks Contains system CPU hardware capabilities and recommended configuration
     */
    typedef struct _TUC_CPU_CAPABILITIES
    {
        // Hardware information
        uint32_t totalCoreCount;    // Total number of cores (logical cores)
        uint32_t physicalCoreCount; // Number of physical cores
        uint32_t numaNodeCount;     // Number of NUMA nodes

        // Capability flags
        bool isNUMAAvailable;          // NUMA availability
        bool isHyperThreading;         // Hyper-threading support
        bool hasAdminPrivileges;       // Administrator privileges
        bool supportsRealtimePriority; // Real-time priority support

        // Cache information
        uint32_t l1CacheSizeKB; // L1 cache size (KB)
        uint32_t l2CacheSizeKB; // L2 cache size (KB)
        uint32_t l3CacheSizeKB; // L3 cache size (KB)

        // System information
        char cpuModel[64];  // CPU model
        char osVersion[32]; // Operating system version

        char reserved[64]; // Reserved field
    } TUC_CPU_CAPABILITIES;

    /* Restore default packing */
    TUC_PACK_POP()

#ifdef __cplusplus
}
#endif

#endif /* TUC_DEFINE_H */
