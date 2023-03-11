#pragma once

#include "DeviceThreads.h"
#include "MMDevice.h"

class LumeneraCamera;

namespace CameraInterface
{
	namespace LucamAdapter
	{
		class LucamCamera;
	}
}

class VideoSequenceThread : public MMDeviceThreadBase
{
	friend class LumeneraCamera;

public:

	explicit VideoSequenceThread(LumeneraCamera* pCam);
	virtual ~VideoSequenceThread() override;

	virtual int svc() override;

	void Start(int numImages);
	void Stop();
	bool IsStopped() const;

private:

	LumeneraCamera* device_;

	mutable MMThreadLock stopLock_;
	bool stopped_;
	int remainingInSequence_;


	bool hasCamera() const;
	CameraInterface::LucamAdapter::LucamCamera* camera() const;

	void StartCameraStream();
	void StopCameraStream();
};
