#include "Event.h"

Event::Event() : manualReset_(false), signalled_(false)
{
}

Event::Event(bool manualReset, bool signalled) : manualReset_(manualReset), signalled_(signalled)
{
}

Event::~Event()
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

bool Event::Wait()
{
    std::unique_lock<std::mutex> lock(mutex_);

    condVar_.wait(lock, [&]() { return signalled_; });

    if (!manualReset_)
        signalled_ = false;

    return true;
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
