/**
 * @copyright Copyright (C) Tucsen Photonics Co., Ltd. 2012-2025. All rights reserved.
 * @brief Tucsen SDK Unified Export Macro Definitions
 *
 * This file provides unified export macro definitions for the Tucsen Photonics SDK,
 * solving dual-interface export issues and enabling simultaneous export of both
 * legacy and new API functions. The macros ensure proper symbol visibility and
 * linkage across different platforms and build configurations.
 *
 * The file defines export macros that:
 * - Support both legacy and new API interfaces simultaneously
 * - Handle platform-specific export attributes (Windows __declspec(dllexport), etc.)
 * - Ensure correct symbol visibility for dynamic library linkage
 * - Provide unified interface for API function declarations
 * - Enable backward compatibility while supporting new API designs
 *
 * Key Features:
 * - Platform-agnostic export macro definitions
 * - Support for multiple API versions in the same library
 * - Proper handling of C/C++ linkage conventions
 * - Compatibility with different compiler toolchains
 * - Unified symbol export mechanism for SDK functions
 *
 * For more information, please visit: https://www.tucsen.com/
 *
 * @file TucSDKExports.h
 * @author zhangren
 * @date 2025-09-18
 * @version v1.0.0
 * @note This header provides the foundation for SDK function export declarations
 * @warning Export macros must be used consistently across all SDK API headers
 * @see TucApi.h
 * @since v1.0.0
 */
#ifndef TUC_SDK_EXPORTS_H
#define TUC_SDK_EXPORTS_H

/* =============================================================================
 * Platform Detection
 * =============================================================================
 */
#ifdef _WIN32
    #define TUC_PLATFORM_WINDOWS
#elif defined(__linux__)
    #define TUC_PLATFORM_LINUX
#elif defined(__APPLE__)
    #define TUC_PLATFORM_MACOS
#else
    #define TUC_PLATFORM_UNKNOWN
#endif

/* =============================================================================
 * Simplified Unified Export Macro Design
 * =============================================================================
 */

/* Detect build mode - backward compatible automatic detection */
#if defined(TUC_EXPORTS) || defined(TUCAM_EXPORTS) || defined(TUCSDK_EXPORTS)
    #define SDK_BUILDING 1
#else
    #define SDK_BUILDING 0
#endif

/* Platform-specific export macro definitions */
#ifdef TUC_PLATFORM_WINDOWS
    #if SDK_BUILDING
        #define TUC_API __declspec(dllexport)
        #define TUCAM_API extern "C" __declspec(dllexport)
    #else
        #define TUC_API __declspec(dllimport)
        #define TUCAM_API extern "C" __declspec(dllimport)
    #endif
#else
    /* Linux/macOS platform */
    #if SDK_BUILDING
        #define TUC_API __attribute__((visibility("default")))
        #define TUCAM_API extern "C" __attribute__((visibility("default")))
    #else
        #define TUC_API
        #define TUCAM_API extern "C"
    #endif
#endif

/* =============================================================================
 * Backward Compatibility Support (Optional)
 * =============================================================================
 */
#ifdef TUC_ENABLE_LEGACY_EXPORT_MACROS
    /* Provide backward compatibility only when explicitly enabled */
    #ifdef TUCSDK_EXPORTS
        #define TUC_EXPORTS_COMPAT
        #define TUCAM_EXPORTS_COMPAT
    #endif
#endif

/* =============================================================================
 * Calling Convention
 * =============================================================================
 */
#ifdef TUC_PLATFORM_WINDOWS
    #ifndef TUC_CALL
        #define TUC_CALL __stdcall
    #endif
    #ifndef TUCAM_CALL
        #define TUCAM_CALL __stdcall
    #endif
#else
    #define TUC_CALL
    #define TUCAM_CALL
#endif

/* =============================================================================
 * Version Information Macros
 * =============================================================================
 */
#define TUCSDK_VERSION_MAJOR    3
#define TUCSDK_VERSION_MINOR    0
#define TUCSDK_VERSION_PATCH    0
#define TUCSDK_VERSION_BUILD    1

#define TUCSDK_MAKE_VERSION(major, minor, patch, build) \
    ((major << 24) | (minor << 16) | (patch << 8) | build)

#define TUCSDK_VERSION \
    TUCSDK_MAKE_VERSION(TUCSDK_VERSION_MAJOR, TUCSDK_VERSION_MINOR, \
                        TUCSDK_VERSION_PATCH, TUCSDK_VERSION_BUILD)

/* =============================================================================
 * Compile-time Assertion Macros
 * =============================================================================
 */
#define TUCSDK_STATIC_ASSERT(condition, message) \
    static_assert(condition, message)

/* Note: These static assertions are disabled as they interfere with normal compilation */
/* TUCSDK_STATIC_ASSERT(sizeof(TUCSDK_API) > 0, "TUCSDK_API must be defined"); */
/* TUCSDK_STATIC_ASSERT(sizeof(TUCSDK_C_API) > 0, "TUCSDK_C_API must be defined"); */

#endif /* TUC_SDK_EXPORTS_H */
