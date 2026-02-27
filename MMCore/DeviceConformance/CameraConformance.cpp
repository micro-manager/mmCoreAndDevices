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

#include "CameraConformance.h"

#include "DeviceManager.h"
#include "Devices/CameraInstance.h"
#include "Error.h"

#include "MMDeviceConstants.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <ctime>
#include <functional>
#include <iomanip>
#include <sstream>
#include <thread>

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

} // anonymous namespace

std::string RunCameraConformanceTests(
      std::shared_ptr<CameraInstance> pCam,
      std::atomic<SeqAcqTestMonitor*>& seqAcqTestMonitor,
      const char* testName,
      const std::string& deviceLabel,
      const std::string& deviceName,
      const std::string& adapterName) {
   using namespace std::chrono;

   const auto startTime = system_clock::now();
   const auto startSteady = steady_clock::now();

   const MM::Device* rawCam = pCam->GetRawPtr();
   const auto testTimeout = seconds(10);
   const auto postErrorDelay = seconds(2);

   // RAII guard: stop camera first (joining its thread), then clear atomic.
   struct MonitorGuard {
      std::atomic<SeqAcqTestMonitor*>& atom;
      std::shared_ptr<CameraInstance>& cam;
      ~MonitorGuard() {
         try {
            DeviceModuleLockGuard g(cam);
            if (cam->IsCapturing())
               cam->StopSequenceAcquisition();
         } catch (...) {}
         atom.store(nullptr, std::memory_order_release);
      }
   };

   auto startFinite = [&](long numImages, double intervalMs) {
      DeviceModuleLockGuard guard(pCam);
      int nRet = pCam->StartSequenceAcquisition(numImages, intervalMs, false);
      if (nRet != DEVICE_OK)
         throw CMMError("Camera failed to start finite sequence acquisition");
   };

   auto startContinuous = [&](double intervalMs) {
      DeviceModuleLockGuard guard(pCam);
      int nRet = pCam->StartSequenceAcquisition(intervalMs);
      if (nRet != DEVICE_OK)
         throw CMMError("Camera failed to start continuous sequence acquisition");
   };

   auto stopCamera = [&]() {
      DeviceModuleLockGuard guard(pCam);
      if (pCam->IsCapturing())
         pCam->StopSequenceAcquisition();
   };

   std::vector<TestResult> results;

   auto testPrepareBeforeInsert = [&]() {
      TestResult result;
      result.name = "seq-prepare-before-insert";

      SeqAcqTestMonitor monitor(rawCam);
      seqAcqTestMonitor.store(&monitor, std::memory_order_release);
      MonitorGuard mg{seqAcqTestMonitor, pCam};

      startFinite(5, 0.0);
      monitor.WaitForInsertImageCount(5, testTimeout);

      if (!monitor.PrepareForAcqCalled()) {
         result.assertions.push_back(
            {false, "PrepareForAcq was not called", {}});
      } else if (!monitor.PrepareBeforeFirstInsert()) {
         result.assertions.push_back(
            {false, "PrepareForAcq was called after InsertImage", {}});
      } else {
         result.assertions.push_back(
            {true, "PrepareForAcq called before first InsertImage", {}});
      }

      result.passed = result.assertions.back().passed;
      results.push_back(std::move(result));
   };

   auto testFinishedAfterCount = [&]() {
      TestResult result;
      result.name = "seq-finished-after-count";

      SeqAcqTestMonitor monitor(rawCam);
      seqAcqTestMonitor.store(&monitor, std::memory_order_release);
      MonitorGuard mg{seqAcqTestMonitor, pCam};

      startFinite(5, 0.0);
      monitor.WaitForInsertImageCount(5, testTimeout);

      if (monitor.WaitForAcqFinished(testTimeout)) {
         result.assertions.push_back(
            {true, "AcqFinished called after finite sequence completed", {}});
      } else {
         AssertionResult a;
         a.passed = false;
         a.message = "AcqFinished not called after finite sequence (5 frames)";
         stopCamera();
         if (monitor.WaitForAcqFinished(testTimeout)) {
            a.details.push_back(
               "AcqFinished was called after stopSequenceAcquisition");
         } else {
            a.details.push_back(
               "AcqFinished was not called even after stopSequenceAcquisition");
         }
         result.assertions.push_back(std::move(a));
      }

      result.passed = result.assertions.back().passed;
      results.push_back(std::move(result));
   };

   auto testFinishedOnError = [&](const char* slug, int errorCode,
         const char* errorName, bool continuous) {
      TestResult result;
      result.name = slug;

      SeqAcqTestMonitor monitor(rawCam);
      monitor.SetErrorInjection(errorCode, 3);
      seqAcqTestMonitor.store(&monitor, std::memory_order_release);
      MonitorGuard mg{seqAcqTestMonitor, pCam};

      if (continuous)
         startContinuous(0.0);
      else
         startFinite(1000000, 0.0);

      monitor.WaitForInsertImageCount(3, testTimeout);

      bool pass = true;

      if (monitor.WaitForAcqFinished(testTimeout)) {
         result.assertions.push_back(
            {true, std::string("AcqFinished called after ") + errorName, {}});
      } else {
         AssertionResult a;
         a.passed = false;
         a.message = std::string("AcqFinished not called after ") + errorName;
         pass = false;
         stopCamera();
         if (monitor.WaitForAcqFinished(testTimeout)) {
            a.details.push_back(
               "AcqFinished was called after stopSequenceAcquisition");
         } else {
            a.details.push_back(
               "AcqFinished was not called even after stopSequenceAcquisition");
         }
         result.assertions.push_back(std::move(a));
      }

      std::this_thread::sleep_for(postErrorDelay);
      int afterError = monitor.InsertImageCountAfterError();
      if (afterError > 1) {
         result.assertions.push_back(
            {false, std::to_string(afterError - 1) +
               " InsertImage call(s) after error return", {}});
         pass = false;
      } else {
         result.assertions.push_back(
            {true, "No further InsertImage calls after error", {}});
      }

      result.passed = pass;
      results.push_back(std::move(result));
   };

   auto testFinishedOnErrorFinite = [&]() {
      testFinishedOnError("seq-finished-on-error-finite",
         DEVICE_ERR, "DEVICE_ERR", false);
   };

   auto testFinishedOnErrorContinuous = [&]() {
      testFinishedOnError("seq-finished-on-error-continuous",
         DEVICE_ERR, "DEVICE_ERR", true);
   };

   auto testFinishedOnOverflowFinite = [&]() {
      testFinishedOnError("seq-finished-on-overflow-finite",
         DEVICE_BUFFER_OVERFLOW, "DEVICE_BUFFER_OVERFLOW", false);
   };

   auto testFinishedOnOverflowContinuous = [&]() {
      testFinishedOnError("seq-finished-on-overflow-continuous",
         DEVICE_BUFFER_OVERFLOW, "DEVICE_BUFFER_OVERFLOW", true);
   };

   struct TestEntry {
      const char* slug;
      std::function<void()> func;
   };
   TestEntry tests[] = {
      {"seq-prepare-before-insert", testPrepareBeforeInsert},
      {"seq-finished-after-count", testFinishedAfterCount},
      {"seq-finished-on-error-finite", testFinishedOnErrorFinite},
      {"seq-finished-on-error-continuous", testFinishedOnErrorContinuous},
      {"seq-finished-on-overflow-finite", testFinishedOnOverflowFinite},
      {"seq-finished-on-overflow-continuous", testFinishedOnOverflowContinuous},
   };

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
         throw CMMError("Unknown camera test: " + selectedTest);
      }
   }

   for (const auto& t : tests) {
      if (!selectedTest.empty() && selectedTest != t.slug)
         continue;
      t.func();
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
   j["deviceType"] = "Camera";
   j["tests"] = testsJson;
   j["summary"] = {
      {"total", static_cast<int>(results.size())},
      {"passed", passedCount},
      {"durationMs", durationMs},
   };

   return j.dump(2);
}

} // namespace internal
} // namespace mmcore
