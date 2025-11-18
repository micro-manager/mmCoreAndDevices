///////////////////////////////////////////////////////////////////////////////
// FILE:          EvidentSDK.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Evident IX5 SDK wrapper for DLL integration
//
// COPYRIGHT:     University of California, San Francisco, 2025
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
// AUTHOR:        Nico Stuurman, 2025

#pragma once

#include <windows.h>
#include <string>

// Define DLL_IMPORT as empty - we load the DLL dynamically
#ifndef DLL_IMPORT
#define DLL_IMPORT
#endif

// Forward declare types used in mdk_if.h that we don't need
typedef enum { e_DISerial = 2 } eDeviceInterface;
typedef enum { e_DTUnknown = 0 } eDeviceType;

// Define constants from mdk_def.h
#define MAX_STRING 512
#ifndef IN
#define IN
#endif
#ifndef OUT
#define OUT
#endif

// Define callback type from gt_type.h
typedef int (__stdcall *GT_CALLBACK_ENTRY)(
    ULONG MsgId,
    ULONG wParam,
    ULONG lParam,
    PVOID pv,
    PVOID pContext,
    PVOID pCaller
);
typedef GT_CALLBACK_ENTRY GT_MDK_CALLBACK;

// Include the actual SDK header to ensure structure matches exactly
#include "mdk_if.h"

namespace EvidentSDK {

// Re-export SDK types for convenience (macros like MAX_COMMAND_SIZE are already global)
using ::MDK_MSL_CMD;
using ::eMDK_CmdStatus;

// Use function pointer types from mdk_if.h (they're defined at global scope)
using ::fn_MSL_PM_Initialize;
using ::fn_MSL_PM_EnumInterface;
using ::fn_MSL_PM_GetInterfaceInfo;
using ::fn_MSL_PM_GetPortName;
using ::fn_MSL_PM_OpenInterface;
using ::fn_MSL_PM_CloseInterface;
using ::fn_MSL_PM_SendCommand;
using ::fn_MSL_PM_RegisterCallback;

// SDK Error Codes
const int SDK_ERR_OFFSET = 10300;
const int SDK_ERR_DLL_NOT_FOUND = SDK_ERR_OFFSET + 1;
const int SDK_ERR_DLL_INIT_FAILED = SDK_ERR_OFFSET + 2;
const int SDK_ERR_FUNCTION_NOT_FOUND = SDK_ERR_OFFSET + 3;
const int SDK_ERR_NO_INTERFACE = SDK_ERR_OFFSET + 4;
const int SDK_ERR_OPEN_FAILED = SDK_ERR_OFFSET + 5;
const int SDK_ERR_SEND_FAILED = SDK_ERR_OFFSET + 6;
const int SDK_ERR_CALLBACK_FAILED = SDK_ERR_OFFSET + 7;

// Helper functions
inline void InitCommand(MDK_MSL_CMD& cmd)
{
    memset(&cmd, 0x00, sizeof(MDK_MSL_CMD));
    cmd.m_Signature = GT_MDK_CMD_SIGNATURE;  // Required signature
    cmd.m_Timeout = 50000;  // 50 seconds default
    cmd.m_Sync = FALSE;     // Asynchronous
}

inline void SetCommandString(MDK_MSL_CMD& cmd, const std::string& command)
{
    // SDK expects command string to include \r\n terminator
    std::string cmdWithTerminator = command + "\r\n";

    size_t len = cmdWithTerminator.length();
    if (len >= MAX_COMMAND_SIZE)
        len = MAX_COMMAND_SIZE - 1;

    memcpy(cmd.m_Cmd, cmdWithTerminator.c_str(), len);
    cmd.m_CmdSize = (DWORD)len;
}

inline std::string GetResponseString(const MDK_MSL_CMD& cmd)
{
    if (cmd.m_RspSize > 0 && cmd.m_RspSize < MAX_RESPONSE_SIZE)
        return std::string((const char*)cmd.m_Rsp, cmd.m_RspSize);
    return "";
}

} // namespace EvidentSDK
