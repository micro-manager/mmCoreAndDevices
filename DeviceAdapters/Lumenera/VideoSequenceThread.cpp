#include "VideoSequenceThread.h"
#include "LumeneraCamera.h"
#include "Cameras/LucamCamera/LucamCamera.h"

#include "Exceptions/LuXAppsException.h"

VideoSequenceThread::VideoSequenceThread(LumeneraCamera* camera)
	: MMDeviceThreadBase()
	, device_(camera)
	, stopped_(true)
	, remainingInSequence_(0)
{
}

VideoSequenceThread::~VideoSequenceThread()
{
}

int VideoSequenceThread::svc()
{
	int ret = DEVICE_ERR;

	LUXAPPS_TRY
	{
		StartCameraStream();

		do
		{
			ret = device_->captureSequenceImage();
			if (remainingInSequence_ > 0) {
				// If its less than 0 that means run forever
				remainingInSequence_--;
			}
			if (remainingInSequence_ == 0) {
				Stop();
			}
		} 
		while (ret == DEVICE_OK && !IsStopped());

	    if (ret != DEVICE_OK)
		{
			device_->LogMessage("Video acquisition encountered an error\n.");
		}

		StopCameraStream();
	}
	LUXAPPS_CATCH(...)
	{
		StopCameraStream();

		device_->LogMessage(g_Msg_EXCEPTION_IN_THREAD, false);

		ret = DEVICE_ERR;
	}

	//NOTE: In case thread exited with error.
	stopped_ = true;

	device_->sequenceEnded();

	return ret;
}

bool VideoSequenceThread::hasCamera() const
{
	return device_->hasCamera();
}

CameraInterface::LucamAdapter::LucamCamera* VideoSequenceThread::camera() const
{
	return device_->camera();
}

void VideoSequenceThread::StartCameraStream()
{
	if (hasCamera() && !camera()->isStreaming())
	{
		camera()->startStream();
	}
}

void VideoSequenceThread::StopCameraStream()
{
	if (hasCamera() && camera()->isStreaming())
	{
		camera()->stopStream();
	}
}

void VideoSequenceThread::Start(int numImages)
{
	MMThreadGuard g(this->stopLock_);
	remainingInSequence_ = numImages;
	stopped_ = false;
	activate();
}

void VideoSequenceThread::Stop()
{
	MMThreadGuard g(this->stopLock_);
	stopped_ = true;
}

bool VideoSequenceThread::IsStopped() const
{
	MMThreadGuard g(this->stopLock_);
	return stopped_;
}

