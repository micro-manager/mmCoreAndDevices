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

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <numeric>
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

template <std::size_t Dim>
class SyncProcessModel {
   using ValueType = std::array<double, Dim>;

   // Setpoint and process variable are always equal.
   ValueType value_{};

   std::function<void(ValueType)> update_;
   ValueType lastUpdatedPV_{};

   void Update() {
      if (value_ == lastUpdatedPV_) {
         return;
      }
      update_(value_);
      lastUpdatedPV_ = value_;
   }

public:
   static constexpr bool isAsync = false;

   explicit SyncProcessModel(std::function<void(ValueType)> updateFunc) :
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

   ValueType ProcessVariable() const { return value_; }

   ValueType Setpoint() const { return value_; }

   void Halt() { /* no-op */ }

   void Setpoint(ValueType setpoint) {
      assert(std::all_of(setpoint.cbegin(), setpoint.cend(),
             [](double v) { return std::isfinite(v); }));
      value_ = setpoint;
      Update();
   }
};

template <std::size_t Dim>
class SlewModel {
   static_assert(Dim > 0, "Cannot handle empty value type");
   using ValueType = std::array<double, Dim>;

   ValueType origin_;
   ValueType target_;
   ValueType displacement_;
   std::array<std::chrono::duration<double>, Dim> rRate_;
   std::chrono::duration<double> duration_;
   std::chrono::steady_clock::time_point startTime_;
   decltype(startTime_ + duration_) finishTime_;

public:
   explicit SlewModel(ValueType origin, ValueType target,
                      std::chrono::duration<double> absRRate) :
      origin_(origin), target_(target),
      displacement_([this] {
         ValueType d;
         std::transform(target_.cbegin(), target_.cend(), origin_.cbegin(),
            d.begin(), [](auto a, auto b) { return a - b; });
         return d;
      }()),
      rRate_([this, absRRate] {
         decltype(rRate_) rr;
         std::transform(displacement_.cbegin(), displacement_.cend(),
            rr.begin(),
            [absRRate](auto d) { return d < 0.0 ? -absRRate : absRRate; });
         return rr;
      }()),
      duration_([this] {
         std::array<std::chrono::duration<double>, Dim> dur;
         std::transform(displacement_.cbegin(), displacement_.cend(),
                        rRate_.cbegin(), dur.begin(),
                        [](auto disp, auto rr) { return disp * rr; });
         return *std::max_element(dur.cbegin(), dur.cend());
      }()),
      startTime_(std::chrono::steady_clock::now()),
      finishTime_(startTime_ + duration_)
   {}

   ValueType Origin() const {
      return origin_;
   }

   ValueType Target() const {
      return target_;
   }

   auto StartTime() const {
      return startTime_;
   }

   auto FinishTime() const {
      return finishTime_;
   }

   template <typename TimePoint>
   ValueType ValueAtTime(TimePoint tp) const {
      const auto elapsed = tp - startTime_;
      auto elemAtTime = [this, elapsed](std::size_t i) {
         const double v = origin_[i] + elapsed / rRate_[i];
         if ((rRate_[i].count() > 0.0 && v > target_[i]) ||
             (rRate_[i].count() < 0.0 && v < target_[i])) {
            return target_[i];
         }
         return v;
      };
      std::array<std::size_t, Dim> indices;
      std::iota(indices.begin(), indices.end(), 0);
      ValueType value;
      std::transform(indices.cbegin(), indices.cend(), value.begin(),
                     elemAtTime);
      return value;
   }
};

template <std::size_t Dim>
class AsyncProcessModel {
   using ValueType = std::array<double, Dim>;

   std::chrono::duration<double> rRate_{1.0}; // Reciprocal rate, per PV unit
   std::chrono::duration<double> updateInterval_{0.1};

   // PV and SP protected by mut_ unless slewThread_ known not to be running
   mutable std::mutex mut_;
   ValueType procVar_{};
   ValueType setpoint_{};

   std::thread slewThread_;
   std::mutex stopMut_;
   std::condition_variable stopCv_;
   bool stopRequested_ = false;

   std::mutex updateMut_;
   std::function<void(ValueType)> update_;
   ValueType lastUpdatedPV_{};

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
   static constexpr bool isAsync = true;

   explicit AsyncProcessModel(std::function<void(ValueType)> updateFunc) :
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

   ValueType ProcessVariable() const {
      std::lock_guard<std::mutex> lock(mut_);
      return procVar_;
   }

   ValueType Setpoint() const {
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

   void Setpoint(ValueType setpoint) {
      assert(std::all_of(setpoint.cbegin(), setpoint.cend(),
             [](double v) { return std::isfinite(v); }));
      Halt(); // Cancel previous slew, if any, updating last PV
      setpoint_ = setpoint;
      if (procVar_ == setpoint_) {
         return;
      }

      const auto slew = SlewModel<Dim>(procVar_, setpoint, rRate_);
      stopRequested_ = false;
      slewThread_ = std::thread([this, slew, dt = updateInterval_] {
         auto updateTime = slew.StartTime() + dt;
         for (;;) {
            if (slew.FinishTime() <= updateTime) {
               if (!CancelableWaitUntil(slew.FinishTime())) {
                  std::lock_guard<std::mutex> lock(mut_);
                  procVar_ = slew.Target();
               }
               Update();
               return;
            }

            if (CancelableWaitUntil(updateTime)) {
               Update();
               return;
            }
            const auto pv = slew.ValueAtTime(updateTime);
            {
               std::lock_guard<std::mutex> lock(mut_);
               procVar_ = pv;
            }
            Update();
            updateTime += dt;
         }
      });
   }
};
