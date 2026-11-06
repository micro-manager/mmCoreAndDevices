# Copyright (c) 2026, Dr. Volodymyr Cherkas
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.16)
project(EmergentCamera LANGUAGES CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "This build configuration targets Linux only. macOS is not supported. Use the Visual Studio project for Windows.")
endif()

set(MM_SOURCE_DIR "" CACHE PATH "Micro-Manager mmCoreAndDevices source directory")
set(EVT_SDK_ROOT "/opt/EVT/eSDK" CACHE PATH "Installed Emergent Linux eSDK directory")
set(EVT_SDK_RUNTIME_ROOT "${EVT_SDK_ROOT}" CACHE PATH "SDK directory on the target Linux machine")
set(MM_DEVICE_DIR "${MM_SOURCE_DIR}/MMDevice")
if(NOT EXISTS "${MM_DEVICE_DIR}/ModuleInterface.h")
    message(FATAL_ERROR "Set MM_SOURCE_DIR to the mmCoreAndDevices directory containing MMDevice.")
endif()
if(NOT EXISTS "${EVT_SDK_ROOT}/include/EmergentCameraAPIs.h")
    message(FATAL_ERROR "Install the Emergent Linux eSDK or set EVT_SDK_ROOT.")
endif()
foreach(library EmergentCamera EmergentGigEVision EmergentGenICam EmergentUtil)
    if(NOT EXISTS "${EVT_SDK_ROOT}/lib/lib${library}.so")
        message(FATAL_ERROR "Missing Linux SDK library: lib${library}.so")
    endif()
endforeach()

find_package(Threads REQUIRED)
add_library(EmergentCameraAdapter SHARED
    src/Module.cpp src/EmergentCamera.cpp src/EmergentSettings.cpp
    "${MM_DEVICE_DIR}/Debayer.cpp"
    "${MM_DEVICE_DIR}/DeviceUtils.cpp"
    "${MM_DEVICE_DIR}/ImgBuffer.cpp"
    "${MM_DEVICE_DIR}/MMDevice.cpp"
    "${MM_DEVICE_DIR}/ModuleInterface.cpp"
    "${MM_DEVICE_DIR}/Property.cpp")
target_compile_features(EmergentCameraAdapter PRIVATE cxx_std_17)
target_compile_definitions(EmergentCameraAdapter PRIVATE MODULE_EXPORTS _REENTRANT)
target_include_directories(EmergentCameraAdapter PRIVATE "${MM_DEVICE_DIR}" "${EVT_SDK_ROOT}/include")
# Link by name: the vendor DSOs omit SONAME, so full paths would be embedded.
target_link_directories(EmergentCameraAdapter PRIVATE "${EVT_SDK_ROOT}/lib")
target_link_libraries(EmergentCameraAdapter PRIVATE
    EmergentCamera EmergentGigEVision EmergentGenICam EmergentUtil
    Threads::Threads "${CMAKE_DL_LIBS}")
target_link_options(EmergentCameraAdapter PRIVATE "LINKER:--no-undefined")
set_target_properties(EmergentCameraAdapter PROPERTIES
    OUTPUT_NAME mmgr_dal_EmergentCamera
    SOVERSION 0
    CXX_EXTENSIONS OFF
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "$ORIGIN;${EVT_SDK_RUNTIME_ROOT}/lib")
