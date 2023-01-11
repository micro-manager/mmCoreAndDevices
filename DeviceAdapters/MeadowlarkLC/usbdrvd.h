#ifndef USBDRVD_H_
#define USBDRVD_H_

//##############################################################################
//#                           Meadowlark Confidential
//#                        Property of Meadowlark Optics
//#
//#                           Copyright (2011-2013)
//#                             Meadowlark Optics
//#                            All Rights Reserved
//#
//#    File:           usbdrvd.h
//#
//#    Description:    This file provides some of the functionality of 
//#                    the original 32-bit usbdrvd.dll with support for
//#                    64-bit systems. This reimplementation provides a close 
//#                    approximation of the original DLL, and should act as a
//#                    drop in replacement. Some parameters and return values 
//#                    are no longer used in this version, but are kept in
//#                    order to match the original interface. Therefore 
//#                    code written for the original DLL should work with
//#                    the new DLL, although code which is written for 
//#                    the new DLL may not work with the origiinal DLL if 
//#                    the same coding requirements are not maintained. 
//#                    To remain consistent, coding standards for the 
//#                    original DLL should be maintained to allow compatability 
//#                    with both libraries.
//#	
//#    Notes:		   In version 1.1 onward, before any other functions are used,
//#					   USBDRVD_GetDevCount must be called.  USBDRVD_OpenDevice and 
//#                    USB_CloseDevice are deprecated, and are only present for
//#                    compatibility with older software and devices.
//#
//#    Author:         Jason Remington
//#
//#    Revision History:
//#
//#    JMR - 05-03-11 VERSION 1.0 RELEASE
//#    JWP - 04-22-13 VERSION 1.1 RELEASE
//#
//##############################################################################

#include <Windows.h>


//##############################################################################
//FLEXIBLE DLL IMPORT / EXPORT DEFINITION BLOCK###################/// @cond HIDE
//This ifdef block allows for the use of a single header file for both building
//the USB DLL, as well as for using the USB DLL.This ifdef block uses a
//preprocessor definition to dynamically use the correct export or import flag
//under windows, or to not use those flags at all if being used in a different OS.
//
//Define USBDRVD_EXPORTS when building the DLL (i.e. -DUSBDRVD_EXPORTS in gcc commandline)
//DO NOT Define USBDRVD_EXPORTS when simply using the DLL.
//
//The compiler definition for _WIN32 and _WIN64 should be defined by all
//compilers if the code is compiled on windows automatically and do not need
//to be defined explicitly by the user.
#if defined(_WIN32) || defined(_WIN64)//IF OS==WINDOWS

    //##########################################################################
    //DEFINE EXPORTS / IMPORTS OF DLL FUNCTIONS#################################
    #ifdef USBDRVD_EXPORTS //TRUE IF USBDRVD_EXPORTS IS DEFINED, AND DLL IS BEING BUILT
        #define USBDRVD_API __declspec(dllexport)    //Export DLL functions
    #else //TRUE IF USBDRVD_EXPORTS IS UNDEFINED, AND DLL IS SIMPLY BEING USED
        #define USBDRVD_API __declspec(dllimport)    //Import DLL functions
    #endif
    //##########################################################################





    //##########################################################################
    //#DEFINE CALLING CONVENTION TO MAKE COMPATIBLE WITH C++ NAME MANGLING######
    #define USBDRVD_CALL __cdecl
    //##########################################################################

#else //ELSE, OS!=WINDOWS, DEFINE EXPORTS AS EMPTY STRINGS

    //##########################################################################
    //DEFINE FLAGS WITH NO VALUE FOR NON-WINDOWS OS.############################
    #define USBDRVD_API
    #define USBDRVD_CALL
    //##########################################################################
#endif
//##################################################################/// @endcond





//##############################################################################
//C STRUCT AND FUNCTION DEFINITIONS#############################################
#ifdef __cplusplus //THIS IFDEF ALLOWS FOR COMPATABILITY IN BOTH C AND C++ CODE
extern "C"
{
#endif

//C function definitions go here. Use USBDRVD_API and USBDRVD_CALL flags appropriately.

    USBDRVD_API UINT      USBDRVD_CALL USBDRVD_GetDevCount(DWORD USB_PID);

    USBDRVD_API HANDLE    USBDRVD_CALL USBDRVD_OpenDevice(UINT deviceNumber,
                                                        DWORD attributes,
                                                        DWORD USB_PID);

    USBDRVD_API void      USBDRVD_CALL USBDRVD_CloseDevice(HANDLE device);

    USBDRVD_API UINT      USBDRVD_CALL USBDRVD_GetPipeCount(HANDLE device);

    USBDRVD_API ULONG     USBDRVD_CALL USBDRVD_BulkRead(HANDLE device, ULONG pipe,
                                                        BYTE *buffer, ULONG count);

    USBDRVD_API ULONG     USBDRVD_CALL USBDRVD_BulkWrite(HANDLE device, ULONG pipe,
                                                        BYTE *buffer, ULONG count);

    USBDRVD_API HANDLE    USBDRVD_CALL USBDRVD_PipeOpen(UINT deviceNumber, UINT pipe,
                                                        DWORD attributes,
                                                        const GUID *USB_GUID);

    USBDRVD_API void      USBDRVD_CALL USBDRVD_PipeClose(HANDLE pipe);

    USBDRVD_API ULONG     USBDRVD_CALL USBDRVD_InterruptRead(HANDLE device, ULONG pipe,
                                                        BYTE *buffer, ULONG count);

    USBDRVD_API ULONG     USBDRVD_CALL USBDRVD_InterruptWrite(HANDLE device, ULONG pipe,
                                                        BYTE *buffer, ULONG count);

    USBDRVD_API ULONG     USBDRVD_CALL USBDRVD_IsoRead(HANDLE device, ULONG pipe,
                                                        BYTE *buffer, ULONG count);

    USBDRVD_API ULONG     USBDRVD_CALL USBDRVD_IsoWrite(HANDLE device, ULONG pipe,
                                                        BYTE *buffer, ULONG count);

#ifdef __cplusplus
} /* __cplusplus */
#endif
//##############################################################################





//##############################################################################
//C++ CLASS AND FUNCTION DEFINITIONS############################################

//C++ class and function definitions go here. Use USBDRVD_API and USBDRVD_CALL flags
//appropriately.

//##############################################################################
#endif /* USBDRVD_H_ */
