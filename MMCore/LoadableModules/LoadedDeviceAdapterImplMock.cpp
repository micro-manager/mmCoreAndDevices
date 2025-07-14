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

#include "LoadedDeviceAdapterImplMock.h"

#include "../../MMDevice/MMDevice.h"
#include "../../MMDevice/ModuleInterface.h"


void LoadedDeviceAdapterImplMock::InitializeModuleData()
{
   impl_->InitializeModuleData(
      [&](const char* name, MM::DeviceType type, const char* desc) {
         registeredDevices_.RegisterDevice(name, type, desc);
      });
}


long LoadedDeviceAdapterImplMock::GetModuleVersion() const
{
   return MODULE_INTERFACE_VERSION;
}


long LoadedDeviceAdapterImplMock::GetDeviceInterfaceVersion() const
{
   return DEVICE_INTERFACE_VERSION;
}


unsigned LoadedDeviceAdapterImplMock::GetNumberOfDevices() const
{
   return registeredDevices_.GetNumberOfDevices();
}


bool LoadedDeviceAdapterImplMock::GetDeviceName(unsigned index, char* buf, unsigned bufLen) const
{
   return registeredDevices_.GetDeviceName(index, buf, bufLen);
}


bool LoadedDeviceAdapterImplMock::GetDeviceDescription(const char* deviceName,
   char* buf, unsigned bufLen) const
{
   return registeredDevices_.GetDeviceDescription(deviceName, buf, bufLen);
}


bool LoadedDeviceAdapterImplMock::GetDeviceType(const char* deviceName, int* type) const
{
   return registeredDevices_.GetDeviceType(deviceName, type);
}


MM::Device* LoadedDeviceAdapterImplMock::CreateDevice(const char* deviceName)
{
   return impl_->CreateDevice(deviceName);
}


void LoadedDeviceAdapterImplMock::DeleteDevice(MM::Device* device)
{
   return impl_->DeleteDevice(device);
}
