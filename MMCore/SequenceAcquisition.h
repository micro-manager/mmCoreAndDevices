// PROJECT:       Micro-Manager
// SUBSYSTEM:     MMCore
//
// COPYRIGHT:     University of California, San Francisco, 2026,
//                All Rights reserved
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

#include <atomic>
#include <memory>
#include <string>

namespace mmcore {
namespace internal {

class CameraInstance;

class SequenceAcquisition {
public:
   static std::shared_ptr<SequenceAcquisition> Create(
      std::shared_ptr<CameraInstance> camera);
   ~SequenceAcquisition();

   SequenceAcquisition(const SequenceAcquisition&) = delete;
   SequenceAcquisition& operator=(const SequenceAcquisition&) = delete;

   const std::string& CameraLabel() const noexcept { return cameraLabel_; }
   const std::shared_ptr<CameraInstance>& Camera() const noexcept {
      return camera_;
   }

   bool WasStopRequested() const noexcept {
      return stopRequested_.load(std::memory_order_acquire);
   }
   void MarkStopRequested() noexcept {
      stopRequested_.store(true, std::memory_order_release);
   }

private:
   explicit SequenceAcquisition(std::shared_ptr<CameraInstance> camera);

   const std::string cameraLabel_;
   const std::shared_ptr<CameraInstance> camera_;
   std::atomic<bool> stopRequested_{false};
};

} // namespace internal
} // namespace mmcore
