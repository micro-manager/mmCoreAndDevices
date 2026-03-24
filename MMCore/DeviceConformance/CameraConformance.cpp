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

#include <thread>
#include <vector>

namespace mmcore {
namespace internal {

nlohmann::json CollectCameraState(
      std::shared_ptr<CameraInstance> camera) {
   nlohmann::json s = nlohmann::json::object();
   DeviceModuleLockGuard guard(camera);

   try { s["exposure"] = camera->GetExposure(); }
   catch (...) { s["exposure"] = nullptr; }

   try { s["binning"] = camera->GetBinning(); }
   catch (...) { s["binning"] = nullptr; }

   try {
      unsigned x, y, w, h;
      camera->GetROI(x, y, w, h);
      s["roi"] = {{"x", x}, {"y", y}, {"width", w}, {"height", h}};
   } catch (...) { s["roi"] = nullptr; }

   try { s["imageWidth"] = camera->GetImageWidth(); }
   catch (...) { s["imageWidth"] = nullptr; }

   try { s["imageHeight"] = camera->GetImageHeight(); }
   catch (...) { s["imageHeight"] = nullptr; }

   try { s["imageBytesPerPixel"] = camera->GetImageBytesPerPixel(); }
   catch (...) { s["imageBytesPerPixel"] = nullptr; }

   try { s["bitDepth"] = camera->GetBitDepth(); }
   catch (...) { s["bitDepth"] = nullptr; }

   try { s["numberOfComponents"] = camera->GetNumberOfComponents(); }
   catch (...) { s["numberOfComponents"] = nullptr; }

   try {
      unsigned nCh = camera->GetNumberOfChannels();
      nlohmann::json channels = nlohmann::json::array();
      for (unsigned i = 0; i < nCh; ++i)
         channels.push_back(camera->GetChannelName(i));
      s["channels"] = channels;
   } catch (...) { s["channels"] = nullptr; }

   try { s["imageBufferSize"] = camera->GetImageBufferSize(); }
   catch (...) { s["imageBufferSize"] = nullptr; }

   try { s["multiROISupported"] = camera->SupportsMultiROI(); }
   catch (...) { s["multiROISupported"] = nullptr; }

   try { s["multiROIEnabled"] = camera->IsMultiROISet(); }
   catch (...) { s["multiROIEnabled"] = nullptr; }

   try {
      if (camera->IsMultiROISet()) {
         unsigned count = 0;
         camera->GetMultiROI(nullptr, nullptr, nullptr, nullptr, &count);
         std::vector<unsigned> xs(count), ys(count), ws(count), hs(count);
         camera->GetMultiROI(xs.data(), ys.data(), ws.data(), hs.data(),
            &count);
         nlohmann::json rois = nlohmann::json::array();
         for (unsigned i = 0; i < count; ++i)
            rois.push_back(
               {{"x", xs[i]}, {"y", ys[i]},
                  {"width", ws[i]}, {"height", hs[i]}});
         s["multiROIs"] = rois;
      }
   } catch (...) { s["multiROIs"] = nullptr; }

   bool isSeq = false;
   try {
      camera->IsExposureSequenceable(isSeq);
      s["exposureSequenceable"] = isSeq;
   } catch (...) { s["exposureSequenceable"] = nullptr; }

   if (isSeq) {
      try {
         long maxLen = 0;
         camera->GetExposureSequenceMaxLength(maxLen);
         s["exposureSequenceMaxLength"] = maxLen;
      } catch (...) { s["exposureSequenceMaxLength"] = nullptr; }
   }

   return s;
}

namespace {

bool HasEvent(const std::vector<SeqAcqLogEntry>& log, SeqAcqEvent event) {
   for (const auto& e : log)
      if (e.event == event)
         return true;
   return false;
}

bool PrepareBeforeFirstInsert(const std::vector<SeqAcqLogEntry>& log) {
   for (const auto& e : log) {
      if (e.event == SeqAcqEvent::PrepareForAcq)
         return true;
      if (e.event == SeqAcqEvent::InsertImage)
         return false;
   }
   return false;
}

int CountInsertsAfterError(const std::vector<SeqAcqLogEntry>& log) {
   bool seenError = false;
   int count = 0;
   for (const auto& e : log) {
      if (e.event != SeqAcqEvent::InsertImage)
         continue;
      if (seenError)
         ++count;
      else if (e.returnCode != DEVICE_OK)
         seenError = true;
   }
   return count;
}

bool WasCapturingDuringAcqFinished(const std::vector<SeqAcqLogEntry>& log) {
   for (const auto& e : log)
      if (e.event == SeqAcqEvent::AcqFinished)
         return e.isCapturing;
   return false;
}

int CountInsertsAfterFinished(const std::vector<SeqAcqLogEntry>& log) {
   bool seenFinished = false;
   int count = 0;
   for (const auto& e : log) {
      if (e.event == SeqAcqEvent::AcqFinished)
         seenFinished = true;
      else if (seenFinished && e.event == SeqAcqEvent::InsertImage)
         ++count;
   }
   return count;
}

struct CameraTestContext {
   std::shared_ptr<CameraInstance> pCam;
   std::atomic<SeqAcqTestMonitor*>& seqAcqTestMonitor;
   MM::Camera* rawCamera;
   std::chrono::milliseconds testTimeout;
   std::chrono::milliseconds negativeTimeout;

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

      SeqAcqTestMonitor monitor(rawCamera);
      seqAcqTestMonitor.store(&monitor, std::memory_order_release);
      MonitorGuard mg{seqAcqTestMonitor, pCam};

      {
         DeviceModuleLockGuard guard(pCam);
         if (pCam->IsCapturing()) {
            result.assertions.push_back(
               {AssertionStatus::Warning,
                  "IsCapturing() was true before starting sequence acquisition",
                  {}});
            return result;
         }
         result.assertions.push_back(
            {AssertionStatus::Pass,
               "IsCapturing() was false before starting sequence acquisition",
               {}});
      }

      try {
         StartFinite(1, 0.0);
      } catch (const CMMError&) {
         result.assertions.push_back(
            {AssertionStatus::Warning,
               "Camera failed to start sequence acquisition", {}});
         return result;
      }

      if (monitor.WaitForEvent(SeqAcqEvent::InsertImage, 1, testTimeout)) {
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

      SeqAcqTestMonitor monitor(rawCamera);
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
      monitor.WaitForEvent(SeqAcqEvent::InsertImage, 5, testTimeout);

      auto log = monitor.GetLog();
      if (!HasEvent(log, SeqAcqEvent::PrepareForAcq)) {
         result.assertions.push_back(
            {AssertionStatus::Fail, "PrepareForAcq was not called", {}});
      } else if (!PrepareBeforeFirstInsert(log)) {
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

      SeqAcqTestMonitor monitor(rawCamera);
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
      monitor.WaitForEvent(SeqAcqEvent::InsertImage, 5, testTimeout);

      if (monitor.WaitForEvent(SeqAcqEvent::AcqFinished, 1, testTimeout)) {
         result.assertions.push_back(
            {AssertionStatus::Pass,
               "AcqFinished called after finite sequence completed", {}});
         auto log = monitor.GetLog();
         if (WasCapturingDuringAcqFinished(log)) {
            result.assertions.push_back(
               {AssertionStatus::Fail,
                  "IsCapturing() was true during AcqFinished", {}});
         } else {
            result.assertions.push_back(
               {AssertionStatus::Pass,
                  "IsCapturing() was false during AcqFinished", {}});
         }
      } else {
         AssertionResult a;
         a.status = AssertionStatus::Fail;
         a.message =
            "AcqFinished not called after finite sequence (5 frames)";
         StopCamera();
         if (monitor.WaitForEvent(
               SeqAcqEvent::AcqFinished, 1, testTimeout)) {
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

   TestResult TestFinishedOnError(int errorCode, const char* errorName,
         bool continuous) {
      TestResult result;

      SeqAcqTestMonitor monitor(rawCamera);
      monitor.SetInsertImageError(errorCode, 3);
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

      monitor.WaitForEvent(SeqAcqEvent::InsertImage, 4, testTimeout);

      if (monitor.WaitForEvent(
            SeqAcqEvent::AcqFinished, 1, testTimeout)) {
         result.assertions.push_back(
            {AssertionStatus::Pass,
               std::string("AcqFinished called after ") + errorName, {}});
         auto log = monitor.GetLog();
         if (WasCapturingDuringAcqFinished(log)) {
            result.assertions.push_back(
               {AssertionStatus::Fail,
                  "IsCapturing() was true during AcqFinished", {}});
         } else {
            result.assertions.push_back(
               {AssertionStatus::Pass,
                  "IsCapturing() was false during AcqFinished", {}});
         }
      } else {
         AssertionResult a;
         a.status = AssertionStatus::Fail;
         a.message =
            std::string("AcqFinished not called after ") + errorName;
         StopCamera();
         if (monitor.WaitForEvent(
               SeqAcqEvent::AcqFinished, 1, testTimeout)) {
            a.details.push_back(
               "AcqFinished was called after stopSequenceAcquisition");
         } else {
            a.details.push_back(
               "AcqFinished was not called even after "
               "stopSequenceAcquisition");
         }
         result.assertions.push_back(std::move(a));
      }

      std::this_thread::sleep_for(negativeTimeout);
      auto log = monitor.GetLog();
      int afterError = CountInsertsAfterError(log);
      if (afterError > 0) {
         result.assertions.push_back(
            {AssertionStatus::Fail, std::to_string(afterError) +
               " InsertImage call(s) after error return", {}});
      } else {
         result.assertions.push_back(
            {AssertionStatus::Pass,
               "No further InsertImage calls after error", {}});
      }

      return result;
   }

   TestResult TestPrepareErrorPropagated() {
      TestResult result;

      SeqAcqTestMonitor monitor(rawCamera);
      monitor.SetPrepareForAcqError(DEVICE_ERR);
      seqAcqTestMonitor.store(&monitor, std::memory_order_release);
      MonitorGuard mg{seqAcqTestMonitor, pCam};

      {
         DeviceModuleLockGuard guard(pCam);
         if (pCam->IsCapturing()) {
            result.assertions.push_back(
               {AssertionStatus::Warning,
                  "IsCapturing() was true before starting sequence acquisition",
                  {}});
            return result;
         }
         result.assertions.push_back(
            {AssertionStatus::Pass,
               "IsCapturing() was false before starting sequence acquisition",
               {}});
      }

      bool startFailed = false;
      try {
         StartFinite(1, 0.0);
      } catch (const CMMError&) {
         startFailed = true;
      }

      if (!startFailed) {
         result.assertions.push_back(
            {AssertionStatus::Fail,
               "Camera did not propagate PrepareForAcq error", {}});
         return result;
      }

      auto log = monitor.GetLog();
      if (!HasEvent(log, SeqAcqEvent::PrepareForAcq)) {
         result.assertions.push_back(
            {AssertionStatus::Warning,
               "Camera failed before calling PrepareForAcq", {}});
         return result;
      }

      std::this_thread::sleep_for(negativeTimeout);
      log = monitor.GetLog();

      if (HasEvent(log, SeqAcqEvent::InsertImage)) {
         result.assertions.push_back(
            {AssertionStatus::Fail,
               "InsertImage called after PrepareForAcq error", {}});
      } else {
         result.assertions.push_back(
            {AssertionStatus::Pass,
               "No InsertImage calls after PrepareForAcq error", {}});
      }

      if (HasEvent(log, SeqAcqEvent::AcqFinished)) {
         result.assertions.push_back(
            {AssertionStatus::Fail,
               "AcqFinished called after PrepareForAcq error", {}});
      } else {
         result.assertions.push_back(
            {AssertionStatus::Pass,
               "No AcqFinished call after PrepareForAcq error", {}});
      }

      {
         DeviceModuleLockGuard guard(pCam);
         if (pCam->IsCapturing()) {
            result.assertions.push_back(
               {AssertionStatus::Fail,
                  "IsCapturing() returned true after failed start", {}});
         } else {
            result.assertions.push_back(
               {AssertionStatus::Pass,
                  "IsCapturing() returned false after failed start", {}});
         }
      }

      return result;
   }

   TestResult TestExplicitStop(bool continuous) {
      TestResult result;

      SeqAcqTestMonitor monitor(rawCamera);
      seqAcqTestMonitor.store(&monitor, std::memory_order_release);
      MonitorGuard mg{seqAcqTestMonitor, pCam};

      {
         DeviceModuleLockGuard guard(pCam);
         if (pCam->IsCapturing()) {
            result.assertions.push_back(
               {AssertionStatus::Warning,
                  "IsCapturing() was true before starting sequence "
                  "acquisition",
                  {}});
            return result;
         }
      }

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

      if (!monitor.WaitForEvent(
            SeqAcqEvent::InsertImage, 3, testTimeout)) {
         result.assertions.push_back(
            {AssertionStatus::Warning,
               "Camera did not produce 3 images before stop", {}});
         return result;
      }

      {
         auto log = monitor.GetLog();
         bool allCapturing = true;
         for (const auto& e : log) {
            if (e.event == SeqAcqEvent::InsertImage && !e.isCapturing) {
               allCapturing = false;
               break;
            }
         }
         if (allCapturing) {
            result.assertions.push_back(
               {AssertionStatus::Pass,
                  "IsCapturing() was true during all InsertImage calls",
                  {}});
         } else {
            result.assertions.push_back(
               {AssertionStatus::Fail,
                  "IsCapturing() was false during an InsertImage call",
                  {}});
         }
      }

      StopCamera();

      {
         DeviceModuleLockGuard guard(pCam);
         if (pCam->IsCapturing()) {
            result.assertions.push_back(
               {AssertionStatus::Fail,
                  "IsCapturing() was true after "
                  "StopSequenceAcquisition",
                  {}});
         } else {
            result.assertions.push_back(
               {AssertionStatus::Pass,
                  "IsCapturing() was false after "
                  "StopSequenceAcquisition",
                  {}});
         }
      }

      if (monitor.WaitForEvent(
            SeqAcqEvent::AcqFinished, 1, testTimeout)) {
         result.assertions.push_back(
            {AssertionStatus::Pass,
               "AcqFinished called after StopSequenceAcquisition",
               {}});
         auto log = monitor.GetLog();
         if (WasCapturingDuringAcqFinished(log)) {
            result.assertions.push_back(
               {AssertionStatus::Fail,
                  "IsCapturing() was true during AcqFinished", {}});
         } else {
            result.assertions.push_back(
               {AssertionStatus::Pass,
                  "IsCapturing() was false during AcqFinished", {}});
         }
      } else {
         result.assertions.push_back(
            {AssertionStatus::Fail,
               "AcqFinished not called after "
               "StopSequenceAcquisition",
               {}});
      }

      std::this_thread::sleep_for(negativeTimeout);
      auto log = monitor.GetLog();
      int afterFinished = CountInsertsAfterFinished(log);
      if (afterFinished > 0) {
         result.assertions.push_back(
            {AssertionStatus::Fail,
               std::to_string(afterFinished) +
                  " InsertImage call(s) after AcqFinished",
               {}});
      } else {
         result.assertions.push_back(
            {AssertionStatus::Pass,
               "No further InsertImage calls after AcqFinished", {}});
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
      static_cast<MM::Camera*>(pCam->GetRawPtr()),
      config.positiveTimeout,
      config.negativeTimeout});


   std::vector<TestEntry> tests;

   tests.push_back({"seq-prepare-error-propagated", [ctx]() {
      return ctx->TestPrepareErrorPropagated();
   }, {}});
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
      return ctx->TestFinishedOnError(DEVICE_ERR, "DEVICE_ERR", false);
   }, {"seq-basic"}});
   tests.push_back({"seq-finished-on-error-continuous", [ctx]() {
      return ctx->TestFinishedOnError(DEVICE_ERR, "DEVICE_ERR", true);
   }, {"seq-basic"}});
   tests.push_back({"seq-finished-on-overflow-finite", [ctx]() {
      return ctx->TestFinishedOnError(
         DEVICE_BUFFER_OVERFLOW, "DEVICE_BUFFER_OVERFLOW", false);
   }, {"seq-basic"}});
   tests.push_back({"seq-finished-on-overflow-continuous", [ctx]() {
      return ctx->TestFinishedOnError(
         DEVICE_BUFFER_OVERFLOW, "DEVICE_BUFFER_OVERFLOW", true);
   }, {"seq-basic"}});
   tests.push_back({"seq-explicit-stop-finite", [ctx]() {
      return ctx->TestExplicitStop(false);
   }, {"seq-basic"}});
   tests.push_back({"seq-explicit-stop-continuous", [ctx]() {
      return ctx->TestExplicitStop(true);
   }, {"seq-basic"}});

   return tests;
}

} // namespace internal
} // namespace mmcore
