///////////////////////////////////////////////////////////////////////////////
// FILE:          AcqZarrAdapter.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Go2Scope devices. Includes the experimental StorageDevice
//
// AUTHORS:       Milos Jovanovic <milos@tehnocad.rs>
//						Nenad Amodaj <nenad@go2scope.com>
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
#include "MMDevice.h"
#include "DeviceBase.h"

//////////////////////////////////////////////////////////////////////////////
// Error codes
//
//////////////////////////////////////////////////////////////////////////////
#define ERR_INTERNAL                 144002
#define ERR_FAILED_CREATING_FILE     144003

#define ERR_ZARR                     140100
#define ERR_ZARR_SETTINGS            140101
#define ERR_ZARR_NUMDIMS             140102
#define ERR_ZARR_STREAM_CREATE       140103
#define ERR_ZARR_STREAM_CLOSE        140104
#define ERR_ZARR_STREAM_LOAD         140105
#define ERR_ZARR_STREAM_APPEND       140106
#define ERR_ZARR_STREAM_ACCESS       140107

static const char* g_AcqZarrStorage = "AcquireZarrStorage";
