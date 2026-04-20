///////////////////////////////////////////////////////////////////////////////
// FILE:          CoreProperty.h
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

#pragma once

#include "MMEventCallback.h"

#include "MMDeviceConstants.h"

#include <functional>
#include <map>
#include <string>
#include <vector>

class CMMCore;
class MMEventCallback;

namespace mmcore {
namespace internal {

struct CorePropertyDef {
   MM::PropertyType type;
   bool readOnly;
   std::function<std::string()> getter;
   std::function<void(const std::string&)> setter;
   std::function<std::vector<std::string>()> allowedValues;
};

class CorePropertyCollection
{
public:
   CorePropertyCollection(CMMCore* core) : core_(core) {}
   ~CorePropertyCollection() {}

   std::string Get(const char* propName) const;
   bool Has(const char* name) const;
   std::vector<std::string> GetAllowedValues(const char* propName) const;
   bool IsReadOnly(const char* propName) const;
   MM::PropertyType GetPropertyType(const char* propName) const;
   std::vector<std::string> GetNames() const;
   void Set(const char* propName, const std::string& value);

   bool IsPropertyPreInit(const char* propName) const;
   bool HasPropertyLimits(const char* propName) const;
   double GetPropertyLowerLimit(const char* propName) const;
   double GetPropertyUpperLimit(const char* propName) const;
   bool IsPropertySequenceable(const char* propName) const;
   long GetPropertySequenceMaxLength(const char* propName) const;

   void Add(const char* name, CorePropertyDef def);
   void Clear() { properties_.clear(); }

private:
   const CorePropertyDef& FindOrThrow(const char* propName) const;

   CMMCore* core_;
   std::map<std::string, CorePropertyDef> properties_;
};

} // namespace internal
} // namespace mmcore
