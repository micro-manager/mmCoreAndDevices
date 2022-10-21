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
#include <memory>
#include <mutex>
#include <thread>

ThreadPool::ThreadPool()
    : abortFlag_(false)
{
    const size_t hwThreadCount = std::max<size_t>(1, std::thread::hardware_concurrency());
    for (size_t n = 0; n < hwThreadCount; ++n)
        threads_.push_back(std::make_shared<std::thread>(&ThreadPool::ThreadFunc, this));
}

ThreadPool::~ThreadPool()
{
    {
        std::lock_guard<std::mutex> lock(mx_);
        abortFlag_ = true;
    }
    cv_.notify_all();

    for (const std::shared_ptr<std::thread>& thread : threads_)
        thread->join();
}

size_t ThreadPool::GetSize() const
{
    return threads_.size();
}

void ThreadPool::Execute(Task* task) 
{
    assert(task != NULL);
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
            assert(task != NULL);
            queue_.push_back(task);
        }
    }
    cv_.notify_all();
}

void ThreadPool::ThreadFunc()
{
    for (;;)
    {
        Task* task = NULL;
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
