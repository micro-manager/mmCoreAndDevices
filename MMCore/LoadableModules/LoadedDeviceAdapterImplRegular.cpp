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

#include "LoadedDeviceAdapterImplRegular.h"


LoadedDeviceAdapterImplRegular::LoadedDeviceAdapterImplRegular(const std::string& filename)
   : module_(std::make_unique<LoadedModule>(filename)),
     InitializeModuleData_(reinterpret_cast<fnInitializeModuleData>(module_->GetFunction("InitializeModuleData"))),
     CreateDevice_(reinterpret_cast<fnCreateDevice>(module_->GetFunction("CreateDevice"))),
     DeleteDevice_(reinterpret_cast<fnDeleteDevice>(module_->GetFunction("DeleteDevice"))),
     GetModuleVersion_(reinterpret_cast<fnGetModuleVersion>(module_->GetFunction("GetModuleVersion"))),
     GetDeviceInterfaceVersion_(reinterpret_cast<fnGetDeviceInterfaceVersion>(module_->GetFunction("GetDeviceInterfaceVersion"))),
     GetNumberOfDevices_(reinterpret_cast<fnGetNumberOfDevices>(module_->GetFunction("GetNumberOfDevices"))),
     GetDeviceName_(reinterpret_cast<fnGetDeviceName>(module_->GetFunction("GetDeviceName"))),
     GetDeviceType_(reinterpret_cast<fnGetDeviceType>(module_->GetFunction("GetDeviceType"))),
     GetDeviceDescription_(reinterpret_cast<fnGetDeviceDescription>(module_->GetFunction("GetDeviceDescription")))
{
}


void LoadedDeviceAdapterImplRegular::Unload()
{
   module_->Unload();
}


void LoadedDeviceAdapterImplRegular::InitializeModuleData()
{
   InitializeModuleData_();
}


long LoadedDeviceAdapterImplRegular::GetModuleVersion() const
{
   return GetModuleVersion_();
}


long LoadedDeviceAdapterImplRegular::GetDeviceInterfaceVersion() const
{
   return GetDeviceInterfaceVersion_();
}


unsigned LoadedDeviceAdapterImplRegular::GetNumberOfDevices() const
{
   return GetNumberOfDevices_();
}


bool LoadedDeviceAdapterImplRegular::GetDeviceName(unsigned index, char* buf, unsigned bufLen) const
{
   return GetDeviceName_(index, buf, bufLen);
}


bool LoadedDeviceAdapterImplRegular::GetDeviceDescription(const char* deviceName,
   char* buf, unsigned bufLen) const
{
   return GetDeviceDescription_(deviceName, buf, bufLen);
}


bool LoadedDeviceAdapterImplRegular::GetDeviceType(const char* deviceName, int* type) const
{
   return GetDeviceType_(deviceName, type);
}


MM::Device* LoadedDeviceAdapterImplRegular::CreateDevice(const char* deviceName)
{
   return CreateDevice_(deviceName);
}


void LoadedDeviceAdapterImplRegular::DeleteDevice(MM::Device* device)
{
   DeleteDevice_(device);
}
