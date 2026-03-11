// A synchronized wrapper for Configuration
//
// LICENSE:       This file is distributed under the "Lesser GPL" (LGPL)
//                license. License text is included with the source
//                distribution.

#pragma once

#include "Configuration.h"

#include <mutex>
#include <optional>

class SynchronizedConfiguration {
public:
   void addSetting(const PropertySetting& setting) {
      std::lock_guard<std::mutex> lock(mutex_);
      config_.addSetting(setting);
   }

   std::optional<PropertySetting> getSetting(const char* device,
         const char* prop) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!config_.isPropertyIncluded(device, prop))
         return std::nullopt;
      return config_.getSetting(device, prop);
   }

   Configuration get() const {
      std::lock_guard<std::mutex> lock(mutex_);
      return config_;
   }

   void set(Configuration config) {
      std::lock_guard<std::mutex> lock(mutex_);
      config_ = std::move(config);
   }

private:
   mutable std::mutex mutex_;
   Configuration config_;
};
