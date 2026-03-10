///////////////////////////////////////////////////////////////////////////////
// FILE:          TaskSet.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     MMCore
//-----------------------------------------------------------------------------
// DESCRIPTION:   Base class for grouping tasks for one logical operation.
//
// AUTHOR:        Tomas Hanak, tomas.hanak@teledyne.com, 03/03/2021
//                Andrej Bencur, andrej.bencur@teledyne.com, 03/03/2021
//
// COPYRIGHT:     Teledyne Digital Imaging US, Inc., 2021
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

#include "TaskSet.h"

#include <cassert>

namespace mmcore {
namespace internal {

TaskSet::TaskSet(std::shared_ptr<ThreadPool> pool)
    : pool_(pool),
    semaphore_(std::make_shared<Semaphore>())
{
    assert(pool);
}

TaskSet::~TaskSet() = default;

size_t TaskSet::GetUsedTaskCount() const
{
    return usedTaskCount_;
}

void TaskSet::Execute()
{
   std::vector<Task*> rawTasks;
   rawTasks.reserve(usedTaskCount_);
   for (size_t i = 0; i < usedTaskCount_; ++i)
      rawTasks.push_back(tasks_[i].get());
   pool_->Execute(rawTasks);
}

void TaskSet::Wait()
{
    semaphore_->Wait(usedTaskCount_);
}

} // namespace internal
} // namespace mmcore
