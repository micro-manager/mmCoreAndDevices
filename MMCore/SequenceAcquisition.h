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

#include "MMDevice.h"

#include <condition_variable>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace mmcore {
namespace internal {

class CameraInstance;

class SequenceAcquisition {
public:
   struct ChannelInfo {
      // null for intrinsic; non-null = phys cam pointer used for caller-
      // identity matching in callbacks.
      MM::Camera* physCamDevice = nullptr;
      // Snapshot of primary->GetChannelName(idx) at start time.
      std::string channelName;
      // Label of the phys cam (for composite channels). Empty for intrinsic.
      std::string physCamLabel;
   };

   struct ParticipantInfo {
      enum class Kind {
         NotParticipant,    // caller is not in the SA's participant set
         CompositeChannel,  // caller is a phys cam at `index`
         IntrinsicPrimary,  // caller is the primary, which has at least one
                            // intrinsic channel (or is a degenerate single-
                            // channel non-composite)
      };
      Kind kind = Kind::NotParticipant;
      unsigned index = 0;  // valid only when kind == CompositeChannel
   };

   enum class PrepareDisposition {
      FirstOpener,    // caller must open shutter, then call FinishShutterOpen
      WaitForOpener,  // another caller is opening; call WaitForShutterOpened
      AlreadyOpened,  // shutter already open; return immediately
      OpenFailed,     // a previous opener failed; caller should also fail
      NotParticipant, // caller isn't an expected participant
      AlreadyPrepared,// caller already called PrepareForAcq before
   };

   static std::shared_ptr<SequenceAcquisition> Create(
      std::shared_ptr<CameraInstance> camera,
      std::vector<ChannelInfo> channels);

   ~SequenceAcquisition();
   SequenceAcquisition(const SequenceAcquisition&) = delete;
   SequenceAcquisition& operator=(const SequenceAcquisition&) = delete;

   const std::string& CameraLabel() const noexcept { return cameraLabel_; }
   const std::shared_ptr<CameraInstance>& Camera() const noexcept {
      return camera_;
   }

   // Lookup by caller MM::Device* (== MM::Camera*) → participant info.
   // Immutable after construction.
   ParticipantInfo LookupParticipant(const MM::Device* caller) const noexcept;
   bool HasParticipant(const MM::Device* caller) const noexcept;

   bool HasIntrinsicChannel() const noexcept { return hasIntrinsic_; }
   const std::vector<ChannelInfo>& Channels() const noexcept {
      return channels_;
   }
   unsigned NumChannels() const noexcept {
      return static_cast<unsigned>(channels_.size());
   }
   const ChannelInfo& Channel(unsigned n) const { return channels_.at(n); }

   // Mutable state (mutex-protected):
   bool WasStopRequested() const noexcept;

   // Mark stop requested. Returns true iff this call caused a transition to
   // "complete" (i.e., stopRequested && all participants have finished).
   bool MarkStopRequested() noexcept;

   // Returns disposition; see enum. On FirstOpener, caller must invoke
   // FinishShutterOpen exactly once after attempting to open the shutter
   // (success or failure). On WaitForOpener, caller must invoke
   // WaitForShutterOpened to block until the opener completes.
   PrepareDisposition BeginPrepare(const MM::Device* caller);

   // Called by the FirstOpener exactly once, regardless of success.
   void FinishShutterOpen(bool success);

   // Blocks until shutter state is terminal (Opened or OpenFailed). Returns
   // true if Opened, false if OpenFailed.
   bool WaitForShutterOpened();

   // Records that `caller` finished the acquisition. Returns true iff this is
   // the boundary call (last expected participant to finish), in which case
   // the caller is responsible for the auto-shutter close + notification.
   // `caller` not in expectedParticipants_ → returns false.
   // Repeat call from same caller → returns false.
   bool RecordFinish(const MM::Device* caller);

   // True iff stopRequested AND all expected participants have called
   // RecordFinish.
   bool IsComplete() const noexcept;

   void DeferShutterClose();
   bool TakeDeferredShutterClose();

private:
   SequenceAcquisition(std::shared_ptr<CameraInstance> camera,
                       std::vector<ChannelInfo> channels);

   const std::string cameraLabel_;
   const std::shared_ptr<CameraInstance> camera_;
   const std::vector<ChannelInfo> channels_;
   const bool hasIntrinsic_;
   const std::set<const MM::Device*> expectedParticipants_;

   mutable std::mutex mu_;
   std::condition_variable shutterOpenedCv_;
   enum class ShutterState { NotOpened, Opening, Opened, OpenFailed };
   ShutterState shutterState_ = ShutterState::NotOpened;
   bool stopRequested_ = false;
   bool shutterCloseDeferred_ = false;
   std::set<const MM::Device*> readyParticipants_;
   std::set<const MM::Device*> finishedParticipants_;
};

} // namespace internal
} // namespace mmcore
