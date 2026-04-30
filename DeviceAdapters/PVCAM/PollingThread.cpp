#include "PollingThread.h"

#include "PVCAMAdapter.h"

PollingThread::PollingThread(Universal* camera)
    : camera_(camera)
{
}

PollingThread::~PollingThread()
{
}

void PollingThread::SetStop(bool stop)
{
    stop_ = stop;
}

bool PollingThread::GetStop() const
{
    return stop_;
}

void PollingThread::Start()
{
    stop_ = false;
    activate();
}

int PollingThread::svc()
{
    int ret = DEVICE_ERR;
    try
    {
        ret = camera_->PollingThreadRun();
    }
    catch(...)
    {
        camera_->LogAdapterMessage(g_Msg_EXCEPTION_IN_THREAD, false);
    }
    return ret;
}
