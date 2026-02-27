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

#include <chrono>
#include <functional>
#include <sstream>
#include <thread>

namespace mmcore {
namespace internal {

std::string RunCameraConformanceTests(
      std::shared_ptr<CameraInstance> pCam,
      std::atomic<SeqAcqTestMonitor*>& seqAcqTestMonitor,
      const char* testName) {
   using namespace std::chrono;

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

   std::ostringstream report;
   int totalTests = 0;
   int passedTests = 0;

   // --- Test 1: seq-prepare-before-insert ---
   auto testPrepareBeforeInsert = [&]() {
      report << "--- seq-prepare-before-insert ---\n";
      ++totalTests;

      SeqAcqTestMonitor monitor(rawCam);
      seqAcqTestMonitor.store(&monitor, std::memory_order_release);
      MonitorGuard mg{seqAcqTestMonitor, pCam};

      startFinite(5, 0.0);
      monitor.WaitForInsertImageCount(5, testTimeout);

      bool pass = true;
      if (!monitor.PrepareForAcqCalled()) {
         report << "FAIL: PrepareForAcq was not called\n";
         pass = false;
      } else if (!monitor.PrepareBeforeFirstInsert()) {
         report << "FAIL: PrepareForAcq was called after InsertImage\n";
         pass = false;
      } else {
         report << "PASS: PrepareForAcq called before first InsertImage\n";
      }
      if (pass) ++passedTests;
   };

   // --- Test 2: seq-finished-after-count ---
   auto testFinishedAfterCount = [&]() {
      report << "--- seq-finished-after-count ---\n";
      ++totalTests;

      SeqAcqTestMonitor monitor(rawCam);
      seqAcqTestMonitor.store(&monitor, std::memory_order_release);
      MonitorGuard mg{seqAcqTestMonitor, pCam};

      startFinite(5, 0.0);
      monitor.WaitForInsertImageCount(5, testTimeout);

      bool pass = false;
      if (monitor.WaitForAcqFinished(testTimeout)) {
         report << "PASS: AcqFinished called after finite sequence completed\n";
         pass = true;
      } else {
         report << "FAIL: AcqFinished not called after finite sequence (5 frames)\n";
         stopCamera();
         if (monitor.WaitForAcqFinished(testTimeout)) {
            report << "  (AcqFinished was called after stopSequenceAcquisition)\n";
         } else {
            report << "  (AcqFinished was not called even after stopSequenceAcquisition)\n";
         }
      }
      if (pass) ++passedTests;
   };

   // --- Tests 3-6 share a pattern: error/overflow injection ---
   auto testFinishedOnError = [&](const char* slug, int errorCode,
         const char* errorName, bool continuous) {
      report << "--- " << slug << " ---\n";
      ++totalTests;

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
         report << "PASS: AcqFinished called after " << errorName << "\n";
      } else {
         report << "FAIL: AcqFinished not called after " << errorName << "\n";
         pass = false;
         stopCamera();
         if (monitor.WaitForAcqFinished(testTimeout)) {
            report << "  (AcqFinished was called after stopSequenceAcquisition)\n";
         } else {
            report << "  (AcqFinished was not called even after stopSequenceAcquisition)\n";
         }
      }

      std::this_thread::sleep_for(postErrorDelay);
      int afterError = monitor.InsertImageCountAfterError();
      if (afterError > 1) {
         report << "FAIL: " << (afterError - 1)
                << " InsertImage call(s) after error return\n";
         pass = false;
      } else {
         report << "PASS: No further InsertImage calls after error\n";
      }

      if (pass) ++passedTests;
   };

   // --- Test 3: seq-finished-on-error-finite ---
   auto testFinishedOnErrorFinite = [&]() {
      testFinishedOnError("seq-finished-on-error-finite",
         DEVICE_ERR, "DEVICE_ERR", false);
   };

   // --- Test 4: seq-finished-on-error-continuous ---
   auto testFinishedOnErrorContinuous = [&]() {
      testFinishedOnError("seq-finished-on-error-continuous",
         DEVICE_ERR, "DEVICE_ERR", true);
   };

   // --- Test 5: seq-finished-on-overflow-finite ---
   auto testFinishedOnOverflowFinite = [&]() {
      testFinishedOnError("seq-finished-on-overflow-finite",
         DEVICE_BUFFER_OVERFLOW, "DEVICE_BUFFER_OVERFLOW", false);
   };

   // --- Test 6: seq-finished-on-overflow-continuous ---
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
      report << "\n";
   }

   report << "=== Summary: " << passedTests << " / " << totalTests
          << " tests passed ===\n";
   return report.str();
}

} // namespace internal
} // namespace mmcore
