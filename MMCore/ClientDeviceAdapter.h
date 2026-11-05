// PROJECT:       Micro-Manager
// SUBSYSTEM:     MMCore
//
// COPYRIGHT:     2025 Board of Regents of the University of Wisconsin System
//
// LICENSE:       This file is distributed under the "Lesser GPL" (LGPL) license.
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
// AUTHOR:        Mark Tsuchida

#pragma once

#include "MMDevice.h"

#include <functional>

/**
 * \brief Experimental: interface for a device adapter implemented by the
 * client of MMCore.
 *
 * This is an alternative to loading a device adapter from a library: derive
 * from this class and pass an instance to CMMCore::loadClientDeviceAdapter().
 *
 * This interface is experimental and subject to change. It is a C++ API and is
 * not intended to be exposed as-is in language bindings; binding layers may
 * use it to implement their own device adapter mechanisms.
 *
 * The devices created must be compiled against the same MMDevice (device
 * interface version) as MMCore. This is not checked.
 *
 * None of the member functions may throw exceptions.
 */
struct ClientDeviceAdapter {
   using RegisterDeviceFunc = std::function<void(const char*, MM::DeviceType, const char*)>;

   virtual ~ClientDeviceAdapter() = default;

   /**
    * \brief Called once, when the adapter is loaded.
    *
    * Call \p registerDevice (name, type, description) for each available
    * device; this is the equivalent of RegisterDevice() in ModuleInterface.h.
    */
   virtual void InitializeModuleData(RegisterDeviceFunc registerDevice) = 0;

   /**
    * \brief Create the named device; return null on failure.
    */
   virtual MM::Device* CreateDevice(const char* name) = 0;

   /**
    * \brief Delete a device previously returned by CreateDevice().
    */
   virtual void DeleteDevice(MM::Device* device) = 0;
};
