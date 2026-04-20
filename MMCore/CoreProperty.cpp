///////////////////////////////////////////////////////////////////////////////
// FILE:          CoreProperty.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     MMCore
//-----------------------------------------------------------------------------
// DESCRIPTION:   Implements the "core property" mechanism. The MMCore exposes
//                some of its own settings as a virtual device.
//
// AUTHOR:        Nenad Amodaj, nenad@amodaj.com, 10/23/2005
//
// COPYRIGHT:     University of California, San Francisco, 2006
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

#include "CoreProperty.h"
#include "CoreUtils.h"
#include "MMCore.h"
#include "Error.h"
#include "Notification.h"

#include <algorithm>

namespace mmcore {
namespace internal {

const CorePropertyDef& CorePropertyCollection::FindOrThrow(const char* propName) const
{
   auto it = properties_.find(propName);
   if (it == properties_.end())
      throw CMMError("Invalid Core property (" + ToString(propName) + ")",
            MMERR_InvalidCoreProperty);
   return it->second;
}

std::string CorePropertyCollection::Get(const char* propName) const
{
   return FindOrThrow(propName).getter();
}

void CorePropertyCollection::Set(const char* propName, const std::string& value)
{
   const auto& def = FindOrThrow(propName);

   if (def.readOnly)
      throw CMMError("Cannot set Core property " + ToString(propName) +
            " to value \"" + value + "\" (read-only)",
            MMERR_InvalidCoreValue);

   if (def.allowedValues) {
      auto allowed = def.allowedValues();
      if (!allowed.empty()) {
         if (std::find(allowed.begin(), allowed.end(), value) == allowed.end())
            throw CMMError("Cannot set Core property " + ToString(propName) +
                  " to invalid value \"" + value + "\"",
                  MMERR_InvalidCoreValue);
      }
   }

   def.setter(value);

   core_->postNotification(
      notification::PropertyChanged{"Core", propName, def.getter()});
}

bool CorePropertyCollection::Has(const char* propName) const
{
   return properties_.find(propName) != properties_.end();
}

std::vector<std::string> CorePropertyCollection::GetNames() const
{
   std::vector<std::string> names;
   names.reserve(properties_.size());
   for (const auto& kv : properties_)
      names.push_back(kv.first);
   return names;
}

bool CorePropertyCollection::IsReadOnly(const char* propName) const
{
   return FindOrThrow(propName).readOnly;
}

MM::PropertyType CorePropertyCollection::GetPropertyType(const char* propName) const
{
   return FindOrThrow(propName).type;
}

std::vector<std::string> CorePropertyCollection::GetAllowedValues(const char* propName) const
{
   const auto& def = FindOrThrow(propName);
   if (def.allowedValues)
      return def.allowedValues();
   return {};
}

void CorePropertyCollection::Add(const char* name, CorePropertyDef def)
{
   properties_[name] = std::move(def);
}

bool CorePropertyCollection::IsPropertyPreInit(const char*) const { return false; }
bool CorePropertyCollection::HasPropertyLimits(const char*) const { return false; }
double CorePropertyCollection::GetPropertyLowerLimit(const char*) const { return 0.0; }
double CorePropertyCollection::GetPropertyUpperLimit(const char*) const { return 0.0; }
bool CorePropertyCollection::IsPropertySequenceable(const char*) const { return false; }
long CorePropertyCollection::GetPropertySequenceMaxLength(const char*) const { return 0; }

} // namespace internal
} // namespace mmcore
