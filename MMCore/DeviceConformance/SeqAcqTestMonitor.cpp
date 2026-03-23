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

void SeqAcqTestMonitor::SetPrepareForAcqError(int errorCode) {
   std::lock_guard<std::mutex> lock(mutex_);
   prepareForAcqError_ = errorCode;
}

void SeqAcqTestMonitor::SetInsertImageError(int errorCode,
      int afterSuccessfulCount) {
   std::lock_guard<std::mutex> lock(mutex_);
   injectErrorCode_ = errorCode;
   injectAfterCount_ = afterSuccessfulCount;
}

int SeqAcqTestMonitor::OnPrepareForAcq() {
   std::lock_guard<std::mutex> lock(mutex_);
   int retCode = prepareForAcqError_;
   log_.push_back({SeqAcqEvent::PrepareForAcq, retCode,
      std::chrono::steady_clock::now()});
   cv_.notify_all();
   return retCode;
}

int SeqAcqTestMonitor::OnInsertImage() {
   std::lock_guard<std::mutex> lock(mutex_);
   int retCode = DEVICE_OK;
   if (injectErrorCode_ != DEVICE_OK &&
         successfulInsertCount_ >= injectAfterCount_) {
      errorInjected_ = true;
   }
   if (errorInjected_) {
      retCode = injectErrorCode_;
   } else {
      ++successfulInsertCount_;
   }
   log_.push_back({SeqAcqEvent::InsertImage, retCode,
      std::chrono::steady_clock::now()});
   cv_.notify_all();
   return retCode;
}

void SeqAcqTestMonitor::OnAcqFinished() {
   std::lock_guard<std::mutex> lock(mutex_);
   log_.push_back({SeqAcqEvent::AcqFinished, DEVICE_OK,
      std::chrono::steady_clock::now()});
   cv_.notify_all();
}

bool SeqAcqTestMonitor::WaitForEvent(SeqAcqEvent event, int count,
      std::chrono::milliseconds timeout) {
   std::unique_lock<std::mutex> lock(mutex_);
   return cv_.wait_for(lock, timeout, [&] {
      int n = 0;
      for (const auto& entry : log_)
         if (entry.event == event)
            ++n;
      return n >= count;
   });
}

std::vector<SeqAcqLogEntry> SeqAcqTestMonitor::GetLog() const {
   std::lock_guard<std::mutex> lock(mutex_);
   return log_;
}

} // namespace internal
} // namespace mmcore
