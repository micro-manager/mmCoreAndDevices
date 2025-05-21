// PROJECT:       Micro-Manager
// SUBSYSTEM:     MMCore
//
// COPYRIGHT:     University of California, San Francisco, 2013-2014
//                2025, Board of Regents of the University of Wisconsin System
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

#include "LoadedDeviceAdapterImpl.h"

#include "../../MMDevice/ModuleInterface.h"
#include "LoadedModule.h"

#include <memory>
#include <string>


class LoadedDeviceAdapterImplRegular : public LoadedDeviceAdapterImpl
{
public:
   explicit LoadedDeviceAdapterImplRegular(const std::string& filename);

   void Unload() override;

   void InitializeModuleData() override;
   long GetModuleVersion() const override;
   long GetDeviceInterfaceVersion() const override;
   unsigned GetNumberOfDevices() const override;
   bool GetDeviceName(unsigned index, char* buf, unsigned bufLen) const override;
   bool GetDeviceDescription(const char* deviceName,
      char* buf, unsigned bufLen) const override;
   bool GetDeviceType(const char* deviceName, int* type) const override;
   MM::Device* CreateDevice(const char* deviceName) override;
   void DeleteDevice(MM::Device* device) override;

private:
   std::unique_ptr<LoadedModule> module_;

   fnInitializeModuleData InitializeModuleData_;
   fnCreateDevice CreateDevice_;
   fnDeleteDevice DeleteDevice_;
   fnGetModuleVersion GetModuleVersion_;
   fnGetDeviceInterfaceVersion GetDeviceInterfaceVersion_;
   fnGetNumberOfDevices GetNumberOfDevices_;
   fnGetDeviceName GetDeviceName_;
   fnGetDeviceType GetDeviceType_;
   fnGetDeviceDescription GetDeviceDescription_;
};
