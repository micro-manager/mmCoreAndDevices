// Thread-safe notification queue for async callback delivery.
//
// LICENSE:       This file is distributed under the "Lesser GPL" (LGPL)
//                license. License text is included with the source
//                distribution.

#pragma once

#include "Notification.h"

#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>

namespace mmcore {
namespace internal {

class NotificationQueue {
public:
   void Push(Notification notification) {
      {
         std::lock_guard<std::mutex> lock(mutex_);
         queue_.push_back(std::move(notification));
      }
      cv_.notify_one();
   }

   std::optional<Notification> WaitAndPop() {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock,
         [this] { return interrupted_ || !queue_.empty(); });
      if (interrupted_) {
         interrupted_ = false;
         return std::nullopt;
      }
      Notification n = std::move(queue_.front());
      queue_.pop_front();
      return n;
   }

   // Wake WaitAndPop (returns nullopt) without discarding pending items.
   // The flag is consumed by the WaitAndPop call that returns nullopt.
   void RequestInterrupt() {
      {
         std::lock_guard<std::mutex> lock(mutex_);
         interrupted_ = true;
      }
      cv_.notify_one();
   }

private:
   std::mutex mutex_;
   std::condition_variable cv_;
   std::deque<Notification> queue_;
   bool interrupted_ = false;
};

} // namespace internal
} // namespace mmcore
