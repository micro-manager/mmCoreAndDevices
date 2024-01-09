// Mock device adapter for testing of device change notifications
//
// Copyright (C) 2024 Board of Regents of the University of Wisconsin System
//
// This file is distributed under the BSD license. License text is included
// with the source distribution.
//
// This file is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE.
//
// IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES.
//
// Author: Mark A. Tsuchida

#pragma once

#include <cassert>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

// DelayedNotifier calls scheduled callables after a delay relative to when
// they are scheduled. The delay is fixed at the time of scheduling, and
// actions are always called on a background thread, except when the delay is
// zero, in which case the action is called on the scheduling thread.
// Member functions are thread-safe.

class DelayedNotifier {
   using Clock = std::chrono::steady_clock;
   using TimePoint = Clock::time_point;

   struct Item {
      TimePoint when;
      std::function<void()> what;

      Item(TimePoint timeout, std::function<void()> action) :
         when(timeout), what(action) {}
   };

   // Priority queue with soonest item at top
   struct LaterItem {
      bool operator()(const Item& lhs, const Item& rhs) const {
         return lhs.when > rhs.when;
      }
   };
   using PrioQ = std::priority_queue<Item, std::vector<Item>, LaterItem>;

   mutable std::mutex mut_;
   std::condition_variable cv_;
   std::chrono::microseconds delay_{};
   PrioQ scheduledItems_;
   bool stopRequested_ = false;

   std::thread notifyThread_;
   std::once_flag threadStarted_;

   TimePoint NextTimeout() const {
      if (scheduledItems_.empty()) {
         return TimePoint::max();
      }
      return scheduledItems_.top().when;
   }

   void IssueNotifications() {
      std::unique_lock<std::mutex> lock(mut_);
      for (;;) {
         const auto timeout = NextTimeout();
         cv_.wait_until(lock, timeout,
               [&] { return NextTimeout() < timeout || stopRequested_; });
         if (stopRequested_) {
            return;
         }
         const auto now = Clock::now();
         while (NextTimeout() <= now) {
            std::function<void()> func = scheduledItems_.top().what;
            scheduledItems_.pop();
            lock.unlock();
            func();
            lock.lock();
         }
      }
   }

public:
   ~DelayedNotifier() {
      if (notifyThread_.joinable()) {
         {
            std::lock_guard<std::mutex> lock(mut_);
            stopRequested_ = true;
         }
         cv_.notify_one();
         notifyThread_.join();
      }
   }

   std::chrono::microseconds Delay() const {
      std::lock_guard<std::mutex> lock(mut_);
      return delay_;
   }

   // Set the delay; applies to actions scheduled in the future only.
   void Delay(std::chrono::microseconds delay) {
      assert(delay.count() >= 0);
      std::lock_guard<std::mutex> lock(mut_);
      delay_ = delay;
   }

   void Schedule(std::function<void()> action) {
      assert(action);
      bool callSync{};
      bool needToInterruptWait{};
      {
         std::lock_guard<std::mutex> lock(mut_);
         if (delay_.count() == 0) {
            callSync = true;
         } else {
            const auto prevTimeout = NextTimeout();
            const auto timeout = Clock::now() + delay_;
            scheduledItems_.emplace(timeout, action);
            needToInterruptWait = timeout < prevTimeout;
         }
      }
      if (callSync) {
         action();
      } else {
         std::call_once(threadStarted_, [this] { // Lazy-start thread
            notifyThread_ = std::thread([this] { IssueNotifications(); });
         });
      }
      if (needToInterruptWait) {
         cv_.notify_one();
      }
   }

   void CancelAll() {
      std::lock_guard<std::mutex> lock(mut_);
      while (!scheduledItems_.empty()) {
         scheduledItems_.pop();
      }
   }
};
