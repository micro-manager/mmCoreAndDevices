///////////////////////////////////////////////////////////////////////////////
// FILE:          G2SFileUtil.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Go2Scope devices. Includes the experimental StorageDevice
//
// AUTHOR:        Milos Jovanovic <milos@tehnocad.rs>
//
// COPYRIGHT:     Nenad Amodaj, Chan Zuckerberg Initiative, 2024
//
// LICENSE:       This file is distributed under the BSD license.
//                License text is included with the source distribution.
//
//                This file is distributed in the hope that it will be useful,
//                but WITHOUT ANY WARRANTY; without even the implied warranty
//                of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//                IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//                CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//                INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.
//
// NOTE:          Storage Device development is supported in part by
//                Chan Zuckerberg Initiative (CZI)
// 
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <cstdint>

//===============================================================================================================================
// Driver version
//===============================================================================================================================
#define G2STIFF_VERSION										"1.0.1" 

//===============================================================================================================================
// Literals
//===============================================================================================================================
#define DEFAULT_BIGTIFF										true
#define DEFAULT_DIRECT_IO									false
#define TIFF_MAX_BUFFER_SIZE								2147483648U
#define G2STIFF_HEADER_SIZE								512
#define G2STIFF_TAG_COUNT									12
#define G2STIFF_TAG_COUNT_NOMETA							11

//===============================================================================================================================
// Utility functions
//===============================================================================================================================
void																writeInt(unsigned char* buff, std::uint8_t len, std::uint64_t val) noexcept;
std::uint64_t													readInt(const unsigned char* buff, std::uint8_t len) noexcept;
