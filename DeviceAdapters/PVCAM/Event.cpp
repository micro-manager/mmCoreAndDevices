#include "Event.h"

Event::Event()
{
}

Event::Event(bool manualReset, bool signalled)
    : manualReset_(manualReset),
    signalled_(signalled)
{
}

void Event::Set()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        signalled_ = true;
    }
    condVar_.notify_all();
}

void Event::Reset()
{
    std::lock_guard<std::mutex> lock(mutex_);
    signalled_ = false;
}

void Event::Wait()
{
    std::unique_lock<std::mutex> lock(mutex_);

    condVar_.wait(lock, [&]() { return signalled_; });

    if (!manualReset_)
        signalled_ = false;
}

bool Event::Wait(unsigned int timeoutMs)
{
    std::unique_lock<std::mutex> lock(mutex_);

    const bool timedOut = !condVar_.wait_for(
            lock, std::chrono::milliseconds(timeoutMs), [&]() { return signalled_; });
    if (timedOut)
        return false;

    if (!manualReset_)
        signalled_ = false;

    return true;
}
