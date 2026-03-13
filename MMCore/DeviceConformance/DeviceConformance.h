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

#include "ConformanceTestConfig.h"
#include "SeqAcqTestMonitor.h"

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

namespace mmcore {
namespace internal {

class DeviceInstance;

enum class AssertionStatus { Pass, Fail, Warning };

struct AssertionResult {
   AssertionStatus status;
   std::string message;
   std::vector<std::string> details;
};

enum class TestStatus { Pass, Fail, Warning, Skipped };

struct TestResult {
   std::string name;
   std::vector<AssertionResult> assertions;
};

struct TestEntry {
   std::string slug;
   std::function<TestResult()> func;
   std::vector<std::string> dependsOn;
};

nlohmann::json CollectDeviceProperties(
      std::shared_ptr<DeviceInstance> device);

std::string RunDeviceConformanceTests(
      std::shared_ptr<DeviceInstance> device,
      std::atomic<SeqAcqTestMonitor*>& seqAcqTestMonitor,
      const ConformanceTestConfig& config,
      const char* testName);

} // namespace internal
} // namespace mmcore
