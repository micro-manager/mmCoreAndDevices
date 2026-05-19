#include "AcqThread.h"

#include "PVCAMAdapter.h"

AcqThread::AcqThread(Universal* camera)
    : camera_(camera)
{
}

AcqThread::~AcqThread()
{
    Stop();
}

void AcqThread::Start()
{
    camera_->LogAdapterMessage("AcqThead starting");
    this->activate();
}

void AcqThread::Stop()
{
    camera_->LogAdapterMessage("AcqThead exit requested");
    requestStop_ = true;
    Resume();
    // Wait for the thread function to exit
    this->wait();
    camera_->LogAdapterMessage("AcqThead exited");
}

void AcqThread::Pause()
{
    // No logging, this is called frequently
    resumeEvent_.Reset();
}

void AcqThread::Resume()
{
    // No logging, this is called frequently
    acqStatus_ = DEVICE_OK;
    resumeEvent_.Set();
}

int AcqThread::AcqStatus() const
{
    return acqStatus_;
}

int AcqThread::svc()
{
    camera_->LogAdapterMessage("AcqThead loop started");

    while (!requestStop_)
    {
        resumeEvent_.Wait();
        if (requestStop_)
            break;

        acqStatus_ = camera_->acquireFrameSeq();
        if (acqStatus_ != DEVICE_OK)
        {
            resumeEvent_.Reset(); // Pause on error
            continue;
        }

        acqStatus_ = camera_->waitForFrameSeq();
        if (acqStatus_ != DEVICE_OK)
        {
            resumeEvent_.Reset(); // Pause on error
            continue;
        }

        // Frame successfully arrived and ready in the buffer.
        // Frame pushed to the MM core from PVCAM callback handler.
    }

    camera_->LogAdapterMessage("AcqThead loop exited");
    return DEVICE_OK;
}
