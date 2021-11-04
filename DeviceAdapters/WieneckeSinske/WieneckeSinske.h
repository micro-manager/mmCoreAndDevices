///////////////////////////////////////////////////////////////////////////////
// FILE:          WieneckeSinske.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Wienecke & Sinske Stage Controller Driver
//                XY Stage
//             
//
// AUTHOR:        S3L GmbH, info@s3l.de, www.s3l.de,  11/21/2017
// COPYRIGHT:     S3L GmbH, Rosdorf, 2017
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
#ifndef _WIENECKESINSKE_H_
#define _WIENECKESINSKE_H_

#include "MMDevice.h"
#include "DeviceBase.h"
#include "DeviceThreads.h"

#include <string>
#include <map>

extern const char* g_XYStageDeviceDeviceName;
extern const char* g_ZPiezoCANDeviceName;
extern const char* g_ZPiezoWSDeviceName;

#endif // _WIENECKESINSKE_H_
