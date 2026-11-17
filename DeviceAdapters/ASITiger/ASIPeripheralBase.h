///////////////////////////////////////////////////////////////////////////////
// FILE:          ASIPeripheralBase.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   Transitional alias for ASIDevice, deleted once every device uses ASIDevice
//
// COPYRIGHT:     Applied Scientific Instrumentation, Eugene OR
//
// LICENSE:       This file is distributed under the BSD license.
//
//                This file is distributed in the hope that it will be useful,
//                but WITHOUT ANY WARRANTY; without even the implied warranty
//                of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
//                IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//                CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//                INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.
//
// AUTHOR:        Jon Daniels (jon@asiimaging.com) 09/2013
//

#pragma once

#include "ASITiger.h"
#include "ASIDevice.h"
#include "ASIHub.h"

// Transitional: a device that still names ASIPeripheralBase<TDeviceBase, UConcreteDevice> gets exactly
// ASIDevice<TDeviceBase<UConcreteDevice>>. This header is deleted once every device names ASIDevice.
template <template <typename> class TDeviceBase, class UConcreteDevice>
using ASIPeripheralBase = ASIDevice<TDeviceBase<UConcreteDevice>>;
