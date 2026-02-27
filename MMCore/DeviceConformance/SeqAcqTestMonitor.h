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

#include <chrono>
#include <condition_variable>
#include <mutex>

namespace mmcore {
namespace internal {

class SeqAcqTestMonitor {
public:
   explicit SeqAcqTestMonitor(const MM::Device* target) : target_(target) {}

   SeqAcqTestMonitor(const SeqAcqTestMonitor&) = delete;
   SeqAcqTestMonitor& operator=(const SeqAcqTestMonitor&) = delete;

   bool IsMonitoring(const MM::Device* caller) const {
      return caller == target_;
   }

   void SetErrorInjection(int errorCode, int afterSuccessfulCount);

   void OnPrepareForAcq();
   int OnInsertImage();
   void OnAcqFinished();

   bool WaitForInsertImageCount(int n, std::chrono::milliseconds timeout);
   bool WaitForAcqFinished(std::chrono::milliseconds timeout);

   bool PrepareForAcqCalled() const;
   int InsertImageCount() const;
   bool AcqFinishedCalled() const;
   bool PrepareBeforeFirstInsert() const;
   int InsertImageCountAfterError() const;

private:
   const MM::Device* const target_;

   mutable std::mutex mutex_;
   std::condition_variable cv_;

   bool prepareForAcqCalled_ = false;
   int insertImageCount_ = 0;
   bool acqFinishedCalled_ = false;

   bool prepareBeforeFirstInsert_ = false;

   int injectErrorCode_ = DEVICE_OK;
   int injectAfterCount_ = 0;
   bool errorInjected_ = false;
   int insertImageCountAfterError_ = 0;
};

} // namespace internal
} // namespace mmcore
