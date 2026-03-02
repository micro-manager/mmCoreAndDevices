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
#include <vector>

namespace mmcore {
namespace internal {

class CameraInstance;

struct AssertionResult {
   bool passed;
   std::string message;
   std::vector<std::string> details;
};

struct TestResult {
   std::string name;
   bool passed;
   std::vector<AssertionResult> assertions;
};

std::string RunCameraConformanceTests(
      std::shared_ptr<CameraInstance> camera,
      std::atomic<SeqAcqTestMonitor*>& testMonitor,
      const char* testName,
      const std::string& deviceLabel,
      const std::string& deviceName,
      const std::string& adapterName);

} // namespace internal
} // namespace mmcore
