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

std::shared_ptr<SequenceAcquisition>
SequenceAcquisition::Create(std::shared_ptr<CameraInstance> camera)
{
   return std::shared_ptr<SequenceAcquisition>(
      new SequenceAcquisition(std::move(camera)));
}

SequenceAcquisition::SequenceAcquisition(
   std::shared_ptr<CameraInstance> camera) :
   cameraLabel_(camera->GetLabel()),
   camera_(std::move(camera))
{
}

SequenceAcquisition::~SequenceAcquisition() = default;

} // namespace internal
} // namespace mmcore
