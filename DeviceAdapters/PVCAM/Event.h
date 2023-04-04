#ifndef _EVENT_H_
#define _EVENT_H_

#include <condition_variable>
#include <mutex>

/**
* Simple synchronization primitive.
*/
class Event final
{
public:
    /**
    * Creates an auto reset, non-signalled event
    */
    Event();
    /**
    * Creates an event with specific configuration
    * @param manualReset Set this to true to make this a manual reset event
    * @param initialState True to set the event to initial signalled state
    */
    explicit Event(bool manualReset, bool initialState);

    /**
    * Sets the event to signalled state
    */
    void Set();
    /**
    * Resets the event to non-signalled state
    */
    void Reset();
    /**
    * Waits for the event signal. If the event is already signalled
    * the function returns immediately. If the event is non-signalled
    * the function waits until the event becomes signaled.
    * In case of auto-reset event the event is reset automatically.
    */
    void Wait();
    /**
    * Similar to the argument-less overload but allows to set an timeout.
    * @return False if the wait timed out. True otherwise.
    */
    bool Wait(unsigned int timeoutMs);

private:
    bool manualReset_{ false };
    bool signalled_{ false };
    std::mutex mutex_{};
    std::condition_variable condVar_{};
};

#endif // _EVENT_H_