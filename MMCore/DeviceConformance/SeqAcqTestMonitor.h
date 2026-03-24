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

#include "MMDevice.h"
#include "MMDeviceConstants.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <vector>

namespace mmcore {
namespace internal {

enum class SeqAcqEvent { PrepareForAcq, InsertImage, AcqFinished };

struct SeqAcqLogEntry {
   SeqAcqEvent event;
   int returnCode;
   std::chrono::steady_clock::time_point timestamp;
   bool isCapturing;
};

class SeqAcqTestMonitor {
public:
   explicit SeqAcqTestMonitor(MM::Camera* camera)
      : camera_(camera) {}

   SeqAcqTestMonitor(const SeqAcqTestMonitor&) = delete;
   SeqAcqTestMonitor& operator=(const SeqAcqTestMonitor&) = delete;

   bool IsMonitoring(const MM::Device* caller) const {
      return caller == camera_;
   }

   void SetPrepareForAcqError(int errorCode);
   void SetInsertImageError(int errorCode, int afterSuccessfulCount);

   int OnPrepareForAcq();
   int OnInsertImage();
   void OnAcqFinished();

   bool WaitForEvent(SeqAcqEvent event, int count,
      std::chrono::milliseconds timeout);
   std::vector<SeqAcqLogEntry> GetLog() const;

private:
   MM::Camera* const camera_;

   mutable std::mutex mutex_;
   std::condition_variable cv_;

   std::vector<SeqAcqLogEntry> log_;
   int successfulInsertCount_ = 0;

   int injectErrorCode_ = DEVICE_OK;
   int injectAfterCount_ = 0;
   bool errorInjected_ = false;

   int prepareForAcqError_ = DEVICE_OK;
};

} // namespace internal
} // namespace mmcore
