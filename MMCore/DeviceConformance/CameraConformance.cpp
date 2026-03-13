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

#include <thread>

namespace mmcore {
namespace internal {

namespace {

struct CameraTestContext {
   std::shared_ptr<CameraInstance> pCam;
   std::atomic<SeqAcqTestMonitor*>& seqAcqTestMonitor;
   const MM::Device* rawCam;
   std::chrono::milliseconds testTimeout;
   std::chrono::milliseconds postErrorDelay;

   struct MonitorGuard {
      std::atomic<SeqAcqTestMonitor*>& atom;
      std::shared_ptr<CameraInstance> cam;
      ~MonitorGuard() {
         try {
            DeviceModuleLockGuard g(cam);
            cam->StopSequenceAcquisition();
         } catch (...) {}
         atom.store(nullptr, std::memory_order_release);
      }
   };

   void StartFinite(long numImages, double intervalMs) {
      DeviceModuleLockGuard guard(pCam);
      int nRet = pCam->StartSequenceAcquisition(numImages, intervalMs, false);
      if (nRet != DEVICE_OK)
         throw CMMError("Camera failed to start finite sequence acquisition");
   }

   void StartContinuous(double intervalMs) {
      DeviceModuleLockGuard guard(pCam);
      int nRet = pCam->StartSequenceAcquisition(intervalMs);
      if (nRet != DEVICE_OK)
         throw CMMError(
            "Camera failed to start continuous sequence acquisition");
   }

   void StopCamera() {
      DeviceModuleLockGuard guard(pCam);
      pCam->StopSequenceAcquisition();
   }

   TestResult TestSeqBasic() {
      TestResult result;
      result.name = "seq-basic";

      SeqAcqTestMonitor monitor(rawCam);
      seqAcqTestMonitor.store(&monitor, std::memory_order_release);
      MonitorGuard mg{seqAcqTestMonitor, pCam};

      try {
         StartFinite(1, 0.0);
      } catch (const CMMError&) {
         result.assertions.push_back(
            {AssertionStatus::Warning,
               "Camera failed to start sequence acquisition", {}});
         return result;
      }

      if (monitor.WaitForInsertImageCount(1, testTimeout)) {
         result.assertions.push_back(
            {AssertionStatus::Pass,
               "Camera produced 1 image via sequence acquisition", {}});
      } else {
         result.assertions.push_back(
            {AssertionStatus::Fail,
               "No image arrived from sequence acquisition", {}});
      }

      return result;
   }

   TestResult TestPrepareBeforeInsert() {
      TestResult result;
      result.name = "seq-prepare-before-insert";

      SeqAcqTestMonitor monitor(rawCam);
      seqAcqTestMonitor.store(&monitor, std::memory_order_release);
      MonitorGuard mg{seqAcqTestMonitor, pCam};

      try {
         StartFinite(5, 0.0);
      } catch (const CMMError&) {
         result.assertions.push_back(
            {AssertionStatus::Warning,
               "Camera failed to start sequence acquisition", {}});
         return result;
      }
      monitor.WaitForInsertImageCount(5, testTimeout);

      if (!monitor.PrepareForAcqCalled()) {
         result.assertions.push_back(
            {AssertionStatus::Fail, "PrepareForAcq was not called", {}});
      } else if (!monitor.PrepareBeforeFirstInsert()) {
         result.assertions.push_back(
            {AssertionStatus::Fail,
               "PrepareForAcq was called after InsertImage", {}});
      } else {
         result.assertions.push_back(
            {AssertionStatus::Pass,
               "PrepareForAcq called before first InsertImage", {}});
      }

      return result;
   }

   TestResult TestFinishedAfterCount() {
      TestResult result;
      result.name = "seq-finished-after-count";

      SeqAcqTestMonitor monitor(rawCam);
      seqAcqTestMonitor.store(&monitor, std::memory_order_release);
      MonitorGuard mg{seqAcqTestMonitor, pCam};

      try {
         StartFinite(5, 0.0);
      } catch (const CMMError&) {
         result.assertions.push_back(
            {AssertionStatus::Warning,
               "Camera failed to start sequence acquisition", {}});
         return result;
      }
      monitor.WaitForInsertImageCount(5, testTimeout);

      if (monitor.WaitForAcqFinished(testTimeout)) {
         result.assertions.push_back(
            {AssertionStatus::Pass,
               "AcqFinished called after finite sequence completed", {}});
      } else {
         AssertionResult a;
         a.status = AssertionStatus::Fail;
         a.message =
            "AcqFinished not called after finite sequence (5 frames)";
         StopCamera();
         if (monitor.WaitForAcqFinished(testTimeout)) {
            a.details.push_back(
               "AcqFinished was called after stopSequenceAcquisition");
         } else {
            a.details.push_back(
               "AcqFinished was not called even after "
               "stopSequenceAcquisition");
         }
         result.assertions.push_back(std::move(a));
      }

      return result;
   }

   TestResult TestFinishedOnError(const char* slug, int errorCode,
         const char* errorName, bool continuous) {
      TestResult result;
      result.name = slug;

      SeqAcqTestMonitor monitor(rawCam);
      monitor.SetErrorInjection(errorCode, 3);
      seqAcqTestMonitor.store(&monitor, std::memory_order_release);
      MonitorGuard mg{seqAcqTestMonitor, pCam};

      try {
         if (continuous)
            StartContinuous(0.0);
         else
            StartFinite(1000000, 0.0);
      } catch (const CMMError&) {
         result.assertions.push_back(
            {AssertionStatus::Warning,
               "Camera failed to start sequence acquisition", {}});
         return result;
      }

      monitor.WaitForInsertImageCount(3, testTimeout);

      if (monitor.WaitForAcqFinished(testTimeout)) {
         result.assertions.push_back(
            {AssertionStatus::Pass,
               std::string("AcqFinished called after ") + errorName, {}});
      } else {
         AssertionResult a;
         a.status = AssertionStatus::Fail;
         a.message =
            std::string("AcqFinished not called after ") + errorName;
         StopCamera();
         if (monitor.WaitForAcqFinished(testTimeout)) {
            a.details.push_back(
               "AcqFinished was called after stopSequenceAcquisition");
         } else {
            a.details.push_back(
               "AcqFinished was not called even after "
               "stopSequenceAcquisition");
         }
         result.assertions.push_back(std::move(a));
      }

      std::this_thread::sleep_for(postErrorDelay);
      int afterError = monitor.InsertImageCountAfterError();
      if (afterError > 1) {
         result.assertions.push_back(
            {AssertionStatus::Fail, std::to_string(afterError - 1) +
               " InsertImage call(s) after error return", {}});
      } else {
         result.assertions.push_back(
            {AssertionStatus::Pass,
               "No further InsertImage calls after error", {}});
      }

      return result;
   }
};

} // anonymous namespace

std::vector<TestEntry> GetCameraConformanceTests(
      std::shared_ptr<CameraInstance> pCam,
      std::atomic<SeqAcqTestMonitor*>& seqAcqTestMonitor,
      const ConformanceTestConfig& config) {
   auto ctx = std::make_shared<CameraTestContext>(CameraTestContext{
      pCam,
      seqAcqTestMonitor,
      pCam->GetRawPtr(),
      config.positiveTimeout,
      config.negativeTimeout,
   });

   std::vector<TestEntry> tests;

   tests.push_back({"seq-basic", [ctx]() {
      return ctx->TestSeqBasic();
   }, {}});
   tests.push_back({"seq-prepare-before-insert", [ctx]() {
      return ctx->TestPrepareBeforeInsert();
   }, {"seq-basic"}});
   tests.push_back({"seq-finished-after-count", [ctx]() {
      return ctx->TestFinishedAfterCount();
   }, {"seq-basic"}});
   tests.push_back({"seq-finished-on-error-finite", [ctx]() {
      return ctx->TestFinishedOnError("seq-finished-on-error-finite",
         DEVICE_ERR, "DEVICE_ERR", false);
   }, {"seq-basic"}});
   tests.push_back({"seq-finished-on-error-continuous", [ctx]() {
      return ctx->TestFinishedOnError("seq-finished-on-error-continuous",
         DEVICE_ERR, "DEVICE_ERR", true);
   }, {"seq-basic"}});
   tests.push_back({"seq-finished-on-overflow-finite", [ctx]() {
      return ctx->TestFinishedOnError("seq-finished-on-overflow-finite",
         DEVICE_BUFFER_OVERFLOW, "DEVICE_BUFFER_OVERFLOW", false);
   }, {"seq-basic"}});
   tests.push_back({"seq-finished-on-overflow-continuous", [ctx]() {
      return ctx->TestFinishedOnError("seq-finished-on-overflow-continuous",
         DEVICE_BUFFER_OVERFLOW, "DEVICE_BUFFER_OVERFLOW", true);
   }, {"seq-basic"}});

   return tests;
}

} // namespace internal
} // namespace mmcore
