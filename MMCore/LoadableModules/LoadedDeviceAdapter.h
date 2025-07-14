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
// AUTHOR:        Mark Tsuchida

#pragma once

#include "../../MMDevice/DeviceThreads.h"
#include "../../MMDevice/MMDevice.h"
#include "../../MMDevice/ModuleInterface.h"
#include "../Logging/Logger.h"
#include "LoadedDeviceAdapterImpl.h"

#include <cstring>
#include <memory>

class CMMCore;


class DeviceInstance;


class LoadedDeviceAdapter /* final */ :
	public std::enable_shared_from_this<LoadedDeviceAdapter>
{
public:
   LoadedDeviceAdapter(const LoadedDeviceAdapter&) = delete;
   LoadedDeviceAdapter& operator=(const LoadedDeviceAdapter&) = delete;

   LoadedDeviceAdapter(const std::string& name,
      std::unique_ptr<LoadedDeviceAdapterImpl>&& impl);

   // TODO Unload() should mark the instance invalid (or require instance
   // deletion to unload)
   void Unload() { impl_->Unload(); } // For developer use only

   std::string GetName() const { return name_; }

   // The "module lock", used to synchronize _most_ access to the device
   // adapter.
   MMThreadLock* GetLock();

   std::vector<std::string> GetAvailableDeviceNames() const;
   std::string GetDeviceDescription(const std::string& deviceName) const;
   MM::DeviceType GetAdvertisedDeviceType(const std::string& deviceName) const;

   std::shared_ptr<DeviceInstance> LoadDevice(CMMCore* core,
         const std::string& name, const std::string& label,
         mm::logging::Logger deviceLogger,
         mm::logging::Logger coreLogger);

private:
   /**
    * \brief Utility class for getting fixed-length strings from the module
    */
   class ModuleStringBuffer
   {
      char buf_[MM::MaxStrLength + 1];
      const LoadedDeviceAdapter* module_;
      const std::string& funcName_;

   public:
      ModuleStringBuffer(const ModuleStringBuffer&) = delete;
      ModuleStringBuffer& operator=(const ModuleStringBuffer&) = delete;

      ModuleStringBuffer(const LoadedDeviceAdapter* module,
            const std::string& functionName) :
         module_(module), funcName_(functionName)
      { std::memset(buf_, 0, sizeof(buf_)); }

      char* GetBuffer() { return buf_; }
      size_t GetMaxStrLen() { return sizeof(buf_) - 1; }
      std::string Get() const { Check(); return buf_; }
      bool IsEmpty() const { Check(); return (buf_[0] == '\0'); }

   private:
      void Check() const
      { if (buf_[sizeof(buf_) - 1] != '\0') ThrowBufferOverflowError(); }
      void ThrowBufferOverflowError() const;
   };

   void CheckInterfaceVersion() const;

   const std::string name_;
   MMThreadLock lock_;
   std::unique_ptr<LoadedDeviceAdapterImpl> impl_;
};
