///////////////////////////////////////////////////////////////////////////////
// FILE:          ThreadPool.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     MMCore
//-----------------------------------------------------------------------------
// DESCRIPTION:   A class executing queued tasks on separate threads
//                and scaling number of threads based on hardware.
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

#include "ThreadPool.h"

#include "Task.h"

#include <algorithm>
#include <cassert>
#include <mutex>
#include <thread>

ThreadPool::ThreadPool()
{
    const size_t hwThreadCount = std::max<size_t>(1, std::thread::hardware_concurrency());
    for (size_t n = 0; n < hwThreadCount; ++n)
    {
        auto thread = std::make_unique<std::thread>(&ThreadPool::ThreadFunc, this);
        threads_.push_back(std::move(thread));
    }
}

ThreadPool::~ThreadPool()
{
    {
        std::lock_guard<std::mutex> lock(mx_);
        abortFlag_ = true;
    }
    cv_.notify_all();

    for (const auto& thread : threads_)
        thread->join();
}

size_t ThreadPool::GetSize() const
{
    return threads_.size();
}

void ThreadPool::Execute(Task* task)
{
    assert(task);
    {
        std::lock_guard<std::mutex> lock(mx_);
        if (abortFlag_)
            return;
        queue_.push_back(task);
    }
    cv_.notify_one();
}

void ThreadPool::Execute(const std::vector<Task*>& tasks)
{
    assert(!tasks.empty());

    {
        std::lock_guard<std::mutex> lock(mx_);
        if (abortFlag_)
            return;
        for (Task* task : tasks)
        {
            assert(task);
            queue_.push_back(task);
        }
    }
    cv_.notify_all();
}

void ThreadPool::ThreadFunc()
{
    for (;;)
    {
        Task* task = nullptr;
        {
            std::unique_lock<std::mutex> lock(mx_);
            cv_.wait(lock, [&]() { return abortFlag_ || !queue_.empty(); });
            if (abortFlag_)
                break;
            task = queue_.front();
            queue_.pop_front();
        }
        task->Execute();
        task->Done();
    }
}
