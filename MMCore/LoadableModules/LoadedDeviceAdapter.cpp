// PROJECT:       Micro-Manager
// SUBSYSTEM:     MMCore
//
// DESCRIPTION:   Device adapter module
//
// COPYRIGHT:     University of California, San Francisco, 2013-2014
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
// AUTHOR:        Mark Tsuchida,
//                based on parts of CPluginManager by Nenad Amodaj

#include "LoadedDeviceAdapter.h"

#include "../Devices/DeviceInstances.h"
#include "../CoreUtils.h"
#include "../Error.h"

#include <functional>
#include <memory>


LoadedDeviceAdapter::LoadedDeviceAdapter(const std::string& name,
   std::unique_ptr<LoadedDeviceAdapterImpl>&& impl) :
   name_(name),
   impl_(std::move(impl))
{
   CheckInterfaceVersion();
   impl_->InitializeModuleData();
}


MMThreadLock*
LoadedDeviceAdapter::GetLock()
{
   return &lock_;
}


std::vector<std::string>
LoadedDeviceAdapter::GetAvailableDeviceNames() const
{
   unsigned deviceCount = impl_->GetNumberOfDevices();
   std::vector<std::string> deviceNames;
   deviceNames.reserve(deviceCount);
   for (unsigned i = 0; i < deviceCount; ++i)
   {
      ModuleStringBuffer nameBuf(this, "GetDeviceName");
      bool ok = impl_->GetDeviceName(i, nameBuf.GetBuffer(), (unsigned int) nameBuf.GetMaxStrLen());
      if (!ok)
      {
         throw CMMError("Cannot get device name at index " + ToString(i) +
               " from device adapter module " + ToQuotedString(name_));
      }
      deviceNames.push_back(nameBuf.Get());
   }
   return deviceNames;
}


std::string
LoadedDeviceAdapter::GetDeviceDescription(const std::string& deviceName) const
{
   ModuleStringBuffer descBuf(this, "GetDeviceDescription");
   bool ok = impl_->GetDeviceDescription(deviceName.c_str(), descBuf.GetBuffer(),
        (unsigned int) descBuf.GetMaxStrLen());
   if (!ok)
   {
      throw CMMError("Cannot get description for device " +
            ToQuotedString(deviceName) + " of device adapter module " +
            ToQuotedString(name_));
   }
   return descBuf.Get();
}


MM::DeviceType
LoadedDeviceAdapter::GetAdvertisedDeviceType(const std::string& deviceName) const
{
   int typeInt = MM::UnknownType;
   bool ok = impl_->GetDeviceType(deviceName.c_str(), &typeInt);
   if (!ok || typeInt == MM::UnknownType)
   {
      throw CMMError("Cannot get type of device " +
            ToQuotedString(deviceName) + " of device adapter module " +
            ToQuotedString(name_));
   }
   return static_cast<MM::DeviceType>(typeInt);
}


std::shared_ptr<DeviceInstance>
LoadedDeviceAdapter::LoadDevice(CMMCore* core, const std::string& name,
      const std::string& label,
      mm::logging::Logger deviceLogger,
      mm::logging::Logger coreLogger)
{
   MM::Device* pDevice = impl_->CreateDevice(name.c_str());
   if (!pDevice)
      throw CMMError("Device adapter " + ToQuotedString(GetName()) +
            " failed to instantiate device " + ToQuotedString(name));

   MM::DeviceType expectedType;
   try
   {
      expectedType = GetAdvertisedDeviceType(name);
   }
   catch (const CMMError&)
   {
      // The type of a device that was not explicitly registered (e.g. a
      // peripheral device or a device provided only for backward
      // compatibility) will not be available.
      expectedType = MM::UnknownType;
   }
   MM::DeviceType actualType = pDevice->GetType();
   if (expectedType == MM::UnknownType)
      expectedType = actualType;

   std::shared_ptr<LoadedDeviceAdapter> shared_this(shared_from_this());
   DeleteDeviceFunction deleter = [this](MM::Device* dev) { impl_->DeleteDevice(dev); };

   switch (expectedType)
   {
      case MM::CameraDevice:
         return std::make_shared<CameraInstance>(core, shared_this, name, pDevice, deleter, label, deviceLogger, coreLogger);
      case MM::ShutterDevice:
         return std::make_shared<ShutterInstance>(core, shared_this, name, pDevice, deleter, label, deviceLogger, coreLogger);
      case MM::StageDevice:
         return std::make_shared<StageInstance>(core, shared_this, name, pDevice, deleter, label, deviceLogger, coreLogger);
      case MM::XYStageDevice:
         return std::make_shared<XYStageInstance>(core, shared_this, name, pDevice, deleter, label, deviceLogger, coreLogger);
      case MM::StateDevice:
         return std::make_shared<StateInstance>(core, shared_this, name, pDevice, deleter, label, deviceLogger, coreLogger);
      case MM::SerialDevice:
         return std::make_shared<SerialInstance>(core, shared_this, name, pDevice, deleter, label, deviceLogger, coreLogger);
      case MM::GenericDevice:
         return std::make_shared<GenericInstance>(core, shared_this, name, pDevice, deleter, label, deviceLogger, coreLogger);
      case MM::AutoFocusDevice:
         return std::make_shared<AutoFocusInstance>(core, shared_this, name, pDevice, deleter, label, deviceLogger, coreLogger);
      case MM::ImageProcessorDevice:
         return std::make_shared<ImageProcessorInstance>(core, shared_this, name, pDevice, deleter, label, deviceLogger, coreLogger);
      case MM::SignalIODevice:
         return std::make_shared<SignalIOInstance>(core, shared_this, name, pDevice, deleter, label, deviceLogger, coreLogger);
      case MM::MagnifierDevice:
         return std::make_shared<MagnifierInstance>(core, shared_this, name, pDevice, deleter, label, deviceLogger, coreLogger);
      case MM::SLMDevice:
         return std::make_shared<SLMInstance>(core, shared_this, name, pDevice, deleter, label, deviceLogger, coreLogger);
      case MM::GalvoDevice:
         return std::make_shared<GalvoInstance>(core, shared_this, name, pDevice, deleter, label, deviceLogger, coreLogger);
      case MM::HubDevice:
         return std::make_shared<HubInstance>(core, shared_this, name, pDevice, deleter, label, deviceLogger, coreLogger);
      case MM::PressurePumpDevice:
          return std::make_shared<PressurePumpInstance>(core, shared_this, name, pDevice, deleter, label, deviceLogger, coreLogger);
      case MM::VolumetricPumpDevice:
          return std::make_shared<VolumetricPumpInstance>(core, shared_this, name, pDevice, deleter, label, deviceLogger, coreLogger);
      default:
         deleter(pDevice);
         throw CMMError("Device " + ToQuotedString(name) +
               " of device adapter " + ToQuotedString(GetName()) +
               " has invalid or unknown type (" + ToQuotedString(actualType) + ")");
   }
}


void
LoadedDeviceAdapter::ModuleStringBuffer::ThrowBufferOverflowError() const
{
   std::string name(module_ ? module_->GetName() : "<unknown>");
   throw CMMError("Buffer overflow in device adapter module " +
         ToQuotedString(name) + " while calling " + funcName_ + "(); "
         "this is most likely a bug in the device adapter");
}


void
LoadedDeviceAdapter::CheckInterfaceVersion() const
{
   long moduleInterfaceVersion, deviceInterfaceVersion;
   try
   {
      moduleInterfaceVersion = impl_->GetModuleVersion();
      deviceInterfaceVersion = impl_->GetDeviceInterfaceVersion();
   }
   catch (const CMMError& e)
   {
      throw CMMError("Cannot verify interface compatibility of device adapter", e);
   }

   if (moduleInterfaceVersion != MODULE_INTERFACE_VERSION)
      throw CMMError("Incompatible module interface version (MMCore requires " +
            ToString(MODULE_INTERFACE_VERSION) +
            "; device adapter has " + ToString(moduleInterfaceVersion) + ")");

   if (deviceInterfaceVersion != DEVICE_INTERFACE_VERSION)
      throw CMMError("Incompatible device interface version (MMCore requires " +
            ToString(DEVICE_INTERFACE_VERSION) +
            "; device adapter has " + ToString(deviceInterfaceVersion) + ")");
}
