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

#include "SequenceAcquisition.h"

#include "Devices/CameraInstance.h"

#include <utility>

namespace mmcore {
namespace internal {

namespace {

bool ComputeHasIntrinsic(
   const std::vector<SequenceAcquisition::ChannelInfo>& channels)
{
   for (const auto& ch : channels)
      if (ch.physCamDevice == nullptr)
         return true;
   return false;
}

std::set<const MM::Device*> ComputeExpectedParticipants(
   const std::shared_ptr<CameraInstance>& camera,
   const std::vector<SequenceAcquisition::ChannelInfo>& channels,
   bool hasIntrinsic)
{
   std::set<const MM::Device*> result;
   for (const auto& ch : channels)
      if (ch.physCamDevice)
         result.insert(ch.physCamDevice);
   if (hasIntrinsic)
      result.insert(camera->GetRawPtr());
   return result;
}

} // namespace

std::shared_ptr<SequenceAcquisition>
SequenceAcquisition::Create(std::shared_ptr<CameraInstance> camera,
                            std::vector<ChannelInfo> channels)
{
   return std::shared_ptr<SequenceAcquisition>(
      new SequenceAcquisition(std::move(camera), std::move(channels)));
}

SequenceAcquisition::SequenceAcquisition(
   std::shared_ptr<CameraInstance> camera,
   std::vector<ChannelInfo> channels) :
   cameraLabel_(camera->GetLabel()),
   camera_(std::move(camera)),
   channels_(std::move(channels)),
   hasIntrinsic_(ComputeHasIntrinsic(channels_)),
   expectedParticipants_(
      ComputeExpectedParticipants(camera_, channels_, hasIntrinsic_))
{
}

SequenceAcquisition::~SequenceAcquisition() = default;

SequenceAcquisition::ParticipantInfo
SequenceAcquisition::LookupParticipant(const MM::Device* caller) const noexcept
{
   ParticipantInfo info;
   if (caller == nullptr)
      return info;
   for (unsigned i = 0; i < channels_.size(); ++i) {
      if (channels_[i].physCamDevice == caller) {
         info.kind = ParticipantInfo::Kind::CompositeChannel;
         info.index = i;
         return info;
      }
   }
   if (hasIntrinsic_ && caller == camera_->GetRawPtr()) {
      info.kind = ParticipantInfo::Kind::IntrinsicPrimary;
      return info;
   }
   return info;
}

bool SequenceAcquisition::HasParticipant(
      const MM::Device* caller) const noexcept {
   return caller && expectedParticipants_.count(caller);
}

bool SequenceAcquisition::WasStopRequested() const noexcept
{
   std::lock_guard<std::mutex> g(mu_);
   return stopRequested_;
}

bool SequenceAcquisition::MarkStopRequested() noexcept
{
   std::lock_guard<std::mutex> g(mu_);
   if (stopRequested_)
      return false;
   stopRequested_ = true;
   return finishedParticipants_.size() == expectedParticipants_.size();
}

SequenceAcquisition::PrepareDisposition
SequenceAcquisition::BeginPrepare(const MM::Device* caller)
{
   if (caller == nullptr)
      return PrepareDisposition::NotParticipant;
   if (expectedParticipants_.find(caller) == expectedParticipants_.end())
      return PrepareDisposition::NotParticipant;

   std::lock_guard<std::mutex> g(mu_);
   const bool first = readyParticipants_.insert(caller).second;
   if (!first)
      return PrepareDisposition::AlreadyPrepared;

   switch (shutterState_) {
   case ShutterState::NotOpened:
      shutterState_ = ShutterState::Opening;
      return PrepareDisposition::FirstOpener;
   case ShutterState::Opening:
      return PrepareDisposition::WaitForOpener;
   case ShutterState::Opened:
      return PrepareDisposition::AlreadyOpened;
   case ShutterState::OpenFailed:
      return PrepareDisposition::OpenFailed;
   }
   return PrepareDisposition::NotParticipant;
}

void SequenceAcquisition::FinishShutterOpen(bool success)
{
   {
      std::lock_guard<std::mutex> g(mu_);
      shutterState_ = success ? ShutterState::Opened : ShutterState::OpenFailed;
   }
   shutterOpenedCv_.notify_all();
}

bool SequenceAcquisition::WaitForShutterOpened()
{
   std::unique_lock<std::mutex> g(mu_);
   shutterOpenedCv_.wait(g, [this] {
      return shutterState_ == ShutterState::Opened ||
             shutterState_ == ShutterState::OpenFailed;
   });
   return shutterState_ == ShutterState::Opened;
}

bool SequenceAcquisition::NeedsStartRollback() const noexcept
{
   std::lock_guard<std::mutex> g(mu_);
   return shutterState_ == ShutterState::Opened;
}

bool SequenceAcquisition::RecordFinish(const MM::Device* caller)
{
   if (caller == nullptr)
      return false;
   if (expectedParticipants_.find(caller) == expectedParticipants_.end())
      return false;

   std::lock_guard<std::mutex> g(mu_);
   const bool first = finishedParticipants_.insert(caller).second;
   if (!first)
      return false;
   return finishedParticipants_.size() == expectedParticipants_.size();
}

bool SequenceAcquisition::IsComplete() const noexcept
{
   std::lock_guard<std::mutex> g(mu_);
   return stopRequested_ &&
      finishedParticipants_.size() == expectedParticipants_.size();
}

bool SequenceAcquisition::AllParticipantsFinished() const noexcept
{
   std::lock_guard<std::mutex> g(mu_);
   return finishedParticipants_.size() == expectedParticipants_.size();
}

void SequenceAcquisition::DeferShutterClose()
{
   std::lock_guard<std::mutex> g(mu_);
   shutterCloseDeferred_ = true;
}

bool SequenceAcquisition::TakeDeferredShutterClose()
{
   std::lock_guard<std::mutex> g(mu_);
   return std::exchange(shutterCloseDeferred_, false);
}

} // namespace internal
} // namespace mmcore
