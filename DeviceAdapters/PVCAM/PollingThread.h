#pragma once

// MMDevice
#include "DeviceThreads.h"

class Universal;

/**
* Acquisition thread used for polling acquisition only.
*/
class PollingThread : public MMDeviceThreadBase
{
public:
    explicit PollingThread(Universal* camera);
    virtual ~PollingThread();

    void SetStop(bool stop);
    bool GetStop() const;
    void Start();

    virtual int svc() override; // From MMDeviceThreadBase

private:
    Universal* const camera_;
    bool stop_{ true };
};
