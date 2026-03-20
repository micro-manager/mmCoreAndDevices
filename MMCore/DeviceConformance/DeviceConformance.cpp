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

#include "DeviceConformance.h"

#include "CameraConformance.h"
#include "CoreUtils.h"
#include "DeviceManager.h"
#include "Devices/CameraInstance.h"
#include "Devices/DeviceInstance.h"
#include "Error.h"

#include "MMDeviceConstants.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <unordered_map>

namespace mmcore {
namespace internal {

namespace {

std::string FormatISO8601(std::chrono::system_clock::time_point tp) {
   auto time = std::chrono::system_clock::to_time_t(tp);
   std::tm tm{};
#ifdef _WIN32
   gmtime_s(&tm, &time);
#else
   gmtime_r(&time, &tm);
#endif
   std::ostringstream ss;
   ss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
   return ss.str();
}

const char* AssertionStatusToString(AssertionStatus s) {
   switch (s) {
      case AssertionStatus::Pass: return "pass";
      case AssertionStatus::Fail: return "fail";
      case AssertionStatus::Warning: return "warning";
   }
   return "unknown";
}

const char* TestStatusToString(TestStatus s) {
   switch (s) {
      case TestStatus::Pass: return "pass";
      case TestStatus::Fail: return "fail";
      case TestStatus::Warning: return "warning";
      case TestStatus::Skipped: return "skipped";
   }
   return "unknown";
}

TestStatus DeriveTestStatus(const std::vector<AssertionResult>& assertions) {
   bool anyWarning = false;
   for (const auto& a : assertions) {
      if (a.status == AssertionStatus::Fail)
         return TestStatus::Fail;
      if (a.status == AssertionStatus::Warning)
         anyWarning = true;
   }
   return anyWarning ? TestStatus::Warning : TestStatus::Pass;
}

nlohmann::json AssertionToJson(const AssertionResult& a) {
   nlohmann::json j;
   j["status"] = AssertionStatusToString(a.status);
   j["message"] = a.message;
   if (!a.details.empty())
      j["details"] = a.details;
   return j;
}

nlohmann::json TestToJson(const TestResult& t, TestStatus status) {
   nlohmann::json j;
   j["name"] = t.name;
   j["status"] = TestStatusToString(status);
   nlohmann::json assertions = nlohmann::json::array();
   for (const auto& a : t.assertions)
      assertions.push_back(AssertionToJson(a));
   j["assertions"] = assertions;
   return j;
}

std::string RunConformanceTests(
      const std::vector<TestEntry>& tests,
      const char* testName,
      const std::string& deviceLabel,
      const std::string& deviceName,
      const std::string& adapterName,
      const std::string& deviceType,
      const nlohmann::json& deviceState) {
   using namespace std::chrono;

   std::string selectedTest;
   if (testName && testName[0] != '\0')
      selectedTest = testName;

   if (!selectedTest.empty()) {
      bool found = false;
      for (const auto& t : tests) {
         if (selectedTest == t.slug) {
            found = true;
            break;
         }
      }
      if (!found) {
         throw CMMError("Unknown conformance test: " + selectedTest);
      }
   }

   const auto startTime = system_clock::now();
   const auto startSteady = steady_clock::now();

   std::vector<TestResult> results;
   std::vector<TestStatus> statuses;
   std::unordered_map<std::string, TestStatus> completedStatuses;

   for (const auto& t : tests) {
      if (!selectedTest.empty() && selectedTest != t.slug)
         continue;

      bool skip = false;
      if (selectedTest.empty()) {
         for (const auto& dep : t.dependsOn) {
            auto it = completedStatuses.find(dep);
            if (it == completedStatuses.end() ||
                  it->second != TestStatus::Pass) {
               skip = true;
               break;
            }
         }
      }

      TestStatus status;
      if (skip) {
         TestResult r;
         r.name = t.slug;
         results.push_back(std::move(r));
         status = TestStatus::Skipped;
      } else {
         results.push_back(t.func());
         status = DeriveTestStatus(results.back().assertions);
      }
      statuses.push_back(status);
      completedStatuses[t.slug] = status;
   }

   const auto endSteady = steady_clock::now();
   double durationMs =
      duration_cast<duration<double, std::milli>>(endSteady - startSteady)
         .count();

   int passedCount = 0, failedCount = 0, warningCount = 0, skippedCount = 0;
   for (auto s : statuses) {
      switch (s) {
         case TestStatus::Pass: ++passedCount; break;
         case TestStatus::Fail: ++failedCount; break;
         case TestStatus::Warning: ++warningCount; break;
         case TestStatus::Skipped: ++skippedCount; break;
      }
   }

   nlohmann::json testsJson = nlohmann::json::array();
   for (size_t i = 0; i < results.size(); ++i)
      testsJson.push_back(TestToJson(results[i], statuses[i]));

   nlohmann::json j;
   j["version"] = 3;
   j["timestamp"] = FormatISO8601(startTime);
   j["device"] = {
      {"label", deviceLabel},
      {"name", deviceName},
      {"library", adapterName},
   };
   j["deviceType"] = deviceType;
   j["deviceState"] = deviceState;
   j["tests"] = testsJson;
   j["summary"] = {
      {"total", static_cast<int>(results.size())},
      {"passed", passedCount},
      {"failed", failedCount},
      {"warnings", warningCount},
      {"skipped", skippedCount},
      {"durationMs", durationMs},
   };

   return j.dump(2);
}

} // anonymous namespace

nlohmann::json CollectDeviceProperties(
      std::shared_ptr<DeviceInstance> device) {
   nlohmann::json props = nlohmann::json::object();
   DeviceModuleLockGuard guard(device);
   for (const auto& name : device->GetPropertyNames()) {
      try {
         props[name] = device->GetProperty(name);
      } catch (...) {
         props[name] = nullptr;
      }
   }
   return props;
}

std::string RunDeviceConformanceTests(
      std::shared_ptr<DeviceInstance> device,
      std::atomic<SeqAcqTestMonitor*>& seqAcqTestMonitor,
      const ConformanceTestConfig& config,
      const char* testName) {
   const auto deviceLabel = device->GetLabel();
   const auto deviceName = device->GetName();
   const auto adapterName = device->GetAdapterModule()->GetName();
   const auto deviceType = device->GetType();
   const auto deviceTypeStr = ToString(deviceType);

   nlohmann::json deviceState;
   deviceState["properties"] = CollectDeviceProperties(device);

   std::vector<TestEntry> tests;
   if (deviceType == MM::CameraDevice) {
      auto pCam = std::static_pointer_cast<CameraInstance>(device);
      {
         DeviceModuleLockGuard guard(pCam);
         if (pCam->IsCapturing())
            throw CMMError(
               "Not allowed during sequence acquisition",
               MMERR_NotAllowedDuringSequenceAcquisition);
      }
      deviceState["settings"] = CollectCameraState(pCam);
      tests = GetCameraConformanceTests(pCam, seqAcqTestMonitor, config);
   }

   return RunConformanceTests(tests, testName, deviceLabel, deviceName,
      adapterName, deviceTypeStr, deviceState);
}

} // namespace internal
} // namespace mmcore
