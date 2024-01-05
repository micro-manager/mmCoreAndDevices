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
#include <cmath>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>

// Models of physical device state. SyncProcessModel and AsyncProcessModel have
// the same compile-time interface. Member functions are not thread-safe: they
// must be called with external synchronization.
//
// SyncProcessModel issues change notifications synchronously (i.e., from the
// thread setting the value) and never becomes busy.
//
// AsyncProcessModel separates setpoint from process variable; the latter is
// slewed so that it reaches the setpoint over time (so the model is "busy"
// while still slewing). Change notifications are periodically issued from a
// background thread during the slewing, with the last notification always
// occuring when the PV reaches the SP.

class SyncProcessModel {
   // Setpoint and process variable are always equal.
   double value_ = 0.0;

   std::function<void(double)> update_;
   double lastUpdatedPV_ = 0.0;

   void Update() {
      if (value_ == lastUpdatedPV_) {
         return;
      }
      update_(value_);
      lastUpdatedPV_ = value_;
   }

public:
   explicit SyncProcessModel(std::function<void(double)> updateFunc) :
         update_(std::move(updateFunc)) {
      assert(update_);
   }

   void ReciprocalSlewRateSeconds(double rRate_s) {
      (void)rRate_s;
      assert(false);
   }

   double ReciprocalSlewRateSeconds() const {
      assert(false);
      return 1.0;
   }

   void UpdateIntervalSeconds(double interval_s) {
      (void)interval_s;
      assert(false);
   }

   double UpdateIntervalSeconds() const {
      assert(false);
      return 0.1;
   }

   bool IsSlewing() const { return false; }

   double ProcessVariable() const { return value_; }

   double Setpoint() const { return value_; }

   void Halt() { /* no-op */ }

   void Setpoint(double setpoint) {
      assert(std::isfinite(setpoint));
      value_ = setpoint;
      Update();
   }
};

class AsyncProcessModel {
   std::chrono::duration<double> rRate_{1.0}; // Reciprocal rate, per PV unit
   std::chrono::duration<double> updateInterval_{0.1};

   // PV and SP protected by mut_ unless slewThread_ known not to be running
   mutable std::mutex mut_;
   double procVar_ = 0.0;
   double setpoint_ = 0.0;

   std::thread slewThread_;
   std::mutex stopMut_;
   std::condition_variable stopCv_;
   bool stopRequested_ = false;

   std::mutex updateMut_;
   std::function<void(double)> update_;
   double lastUpdatedPV_ = 0.0;

   // Only call from slew thread; cancel triggered by Halt()
   template <typename TimePoint>
   bool CancelableWaitUntil(TimePoint timeout) {
      std::unique_lock<std::mutex> lock(stopMut_);
      stopCv_.wait_until(lock, timeout, [&] { return stopRequested_; });
      return stopRequested_;
   }

   // Only call from slew thread
   void Update() {
      std::lock_guard<std::mutex> lock(updateMut_);
      if (procVar_ == lastUpdatedPV_) {
         return;
      }
      update_(procVar_);
      lastUpdatedPV_ = procVar_;
   }

public:
   explicit AsyncProcessModel(std::function<void(double)> updateFunc) :
         update_(std::move(updateFunc)) {
      assert(update_);
   }

   ~AsyncProcessModel() {
      // Avoid sending updates upon halting
      {
         std::lock_guard<std::mutex> lock(updateMut_);
         update_ = [](auto) {};
      }
      Halt();
   }

   // Set slew rate (as reciprocal slew rate); applies to next setpoint
   void ReciprocalSlewRateSeconds(double rRate_s) {
      assert(rRate_s > 0.0);
      rRate_ = std::chrono::duration<double>{rRate_s};
   }

   double ReciprocalSlewRateSeconds() const {
      return rRate_.count();
   }

   // Set update interval; applies to next setpoint
   void UpdateIntervalSeconds(double interval_s) {
      assert(interval_s > 0.0);
      updateInterval_ = std::chrono::duration<double>{interval_s};
   }

   double UpdateIntervalSeconds() const {
      return updateInterval_.count();
   }

   bool IsSlewing() const { // Aka "busy", "in motion"
      std::lock_guard<std::mutex> lock(mut_);
      return procVar_ != setpoint_;
   }

   double ProcessVariable() const {
      std::lock_guard<std::mutex> lock(mut_);
      return procVar_;
   }

   double Setpoint() const {
      std::lock_guard<std::mutex> lock(mut_);
      return setpoint_;
   }

   void Halt() {
      if (!slewThread_.joinable()) {
         return;
      }

      {
         std::lock_guard<std::mutex> lock(stopMut_);
         stopRequested_ = true;
      }
      stopCv_.notify_one();
      slewThread_.join();

      setpoint_ = procVar_;
   }

   void Setpoint(double setpoint) {
      assert(std::isfinite(setpoint));
      Halt(); // Cancel previous slew, if any, updating last PV
      setpoint_ = setpoint;
      if (procVar_ == setpoint_) {
         return;
      }

      const auto orig = procVar_;
      stopRequested_ = false;
      slewThread_ = std::thread(
            [this, orig, setpoint, absRRate = rRate_,
                  updateInterval = updateInterval_] {
         const auto displacement = setpoint - orig;
         const auto rRate = displacement < 0.0 ? -absRRate : absRRate;
         const auto duration = displacement * rRate;
         const auto startTime = std::chrono::steady_clock::now();
         const auto finishTime = startTime + duration;
         auto updateTime = startTime + updateInterval;
         for (;;) {
            if (finishTime <= updateTime) {
               if (!CancelableWaitUntil(finishTime)) {
                  std::lock_guard<std::mutex> lock(mut_);
                  procVar_ = setpoint;
               }
               Update();
               return;
            }

            if (CancelableWaitUntil(updateTime)) {
               Update();
               return;
            }
            const auto elapsed = updateTime - startTime;
            const double pv = orig + elapsed / rRate;
            {
               std::lock_guard<std::mutex> lock(mut_);
               procVar_ = pv;
            }
            Update();
            updateTime += updateInterval;
         }
      });
   }
};
