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

#include "SeqAcqTestMonitor.h"

namespace mmcore {
namespace internal {

void SeqAcqTestMonitor::SetErrorInjection(int errorCode,
      int afterSuccessfulCount) {
   std::lock_guard<std::mutex> lock(mutex_);
   injectErrorCode_ = errorCode;
   injectAfterCount_ = afterSuccessfulCount;
}

void SeqAcqTestMonitor::OnPrepareForAcq() {
   std::lock_guard<std::mutex> lock(mutex_);
   prepareForAcqCalled_ = true;
   if (insertImageCount_ == 0)
      prepareBeforeFirstInsert_ = true;
}

int SeqAcqTestMonitor::OnInsertImage() {
   std::lock_guard<std::mutex> lock(mutex_);
   if (errorInjected_) {
      ++insertImageCountAfterError_;
      cv_.notify_all();
      return injectErrorCode_;
   }
   if (injectErrorCode_ != DEVICE_OK &&
         insertImageCount_ >= injectAfterCount_) {
      errorInjected_ = true;
      ++insertImageCountAfterError_;
      cv_.notify_all();
      return injectErrorCode_;
   }
   ++insertImageCount_;
   cv_.notify_all();
   return DEVICE_OK;
}

void SeqAcqTestMonitor::OnAcqFinished() {
   std::lock_guard<std::mutex> lock(mutex_);
   acqFinishedCalled_ = true;
   cv_.notify_all();
}

bool SeqAcqTestMonitor::WaitForInsertImageCount(int n,
      std::chrono::milliseconds timeout) {
   std::unique_lock<std::mutex> lock(mutex_);
   return cv_.wait_for(lock, timeout,
      [&] { return insertImageCount_ >= n || errorInjected_; });
}

bool SeqAcqTestMonitor::WaitForAcqFinished(
      std::chrono::milliseconds timeout) {
   std::unique_lock<std::mutex> lock(mutex_);
   return cv_.wait_for(lock, timeout,
      [&] { return acqFinishedCalled_; });
}

bool SeqAcqTestMonitor::PrepareForAcqCalled() const {
   std::lock_guard<std::mutex> lock(mutex_);
   return prepareForAcqCalled_;
}

int SeqAcqTestMonitor::InsertImageCount() const {
   std::lock_guard<std::mutex> lock(mutex_);
   return insertImageCount_;
}

bool SeqAcqTestMonitor::AcqFinishedCalled() const {
   std::lock_guard<std::mutex> lock(mutex_);
   return acqFinishedCalled_;
}

bool SeqAcqTestMonitor::PrepareBeforeFirstInsert() const {
   std::lock_guard<std::mutex> lock(mutex_);
   return prepareBeforeFirstInsert_;
}

int SeqAcqTestMonitor::InsertImageCountAfterError() const {
   std::lock_guard<std::mutex> lock(mutex_);
   return insertImageCountAfterError_;
}

} // namespace internal
} // namespace mmcore
