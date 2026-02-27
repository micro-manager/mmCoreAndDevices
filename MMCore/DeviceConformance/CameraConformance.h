// COPYRIGHT:     2026, Board of Regents of the University of Wisconsin System
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

#pragma once

#include "SeqAcqTestMonitor.h"

#include <atomic>
#include <memory>
#include <string>

namespace mmcore {
namespace internal {

class CameraInstance;

std::string RunCameraConformanceTests(
      std::shared_ptr<CameraInstance> camera,
      std::atomic<SeqAcqTestMonitor*>& testMonitor,
      const char* testName);

} // namespace internal
} // namespace mmcore
