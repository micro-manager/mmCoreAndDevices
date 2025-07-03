// PROJECT:       Micro-Manager
// SUBSYSTEM:     MMCore
//
// COPYRIGHT:     2023, Board of Regents of the University of Wisconsin System
//                All Rights reserved
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

#include <string>

namespace mm {
namespace features {

struct Flags {
   bool strictInitializationChecks = false;
   bool ParallelDeviceInitialization = true;
   // How to add a new Core feature: see the comment in the .cpp file.
};

namespace internal {

extern Flags g_flags;

}

inline const Flags& flags() { return internal::g_flags; }

void enableFeature(const std::string& name, bool enable);
bool isFeatureEnabled(const std::string& name);

} // namespace features
} // namespace mm
