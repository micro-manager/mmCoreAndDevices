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

nlohmann::json AssertionToJson(const AssertionResult& a) {
   nlohmann::json j;
   j["passed"] = a.passed;
   j["message"] = a.message;
   if (!a.details.empty())
      j["details"] = a.details;
   return j;
}

nlohmann::json TestToJson(const TestResult& t) {
   nlohmann::json j;
   j["name"] = t.name;
   j["passed"] = t.passed;
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
      const std::string& deviceType) {
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
   for (const auto& t : tests) {
      if (!selectedTest.empty() && selectedTest != t.slug)
         continue;
      results.push_back(t.func());
   }

   const auto endSteady = steady_clock::now();
   double durationMs =
      duration_cast<duration<double, std::milli>>(endSteady - startSteady)
         .count();

   int passedCount = 0;
   for (const auto& r : results)
      if (r.passed)
         ++passedCount;

   nlohmann::json testsJson = nlohmann::json::array();
   for (const auto& r : results)
      testsJson.push_back(TestToJson(r));

   nlohmann::json j;
   j["version"] = 1;
   j["timestamp"] = FormatISO8601(startTime);
   j["device"] = {
      {"label", deviceLabel},
      {"name", deviceName},
      {"library", adapterName},
   };
   j["deviceType"] = deviceType;
   j["tests"] = testsJson;
   j["summary"] = {
      {"total", static_cast<int>(results.size())},
      {"passed", passedCount},
      {"durationMs", durationMs},
   };

   return j.dump(2);
}

} // anonymous namespace

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
      tests = GetCameraConformanceTests(pCam, seqAcqTestMonitor, config);
   }

   return RunConformanceTests(tests, testName, deviceLabel, deviceName,
      adapterName, deviceTypeStr);
}

} // namespace internal
} // namespace mmcore
