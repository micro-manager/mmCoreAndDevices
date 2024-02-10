#ifndef _ARAVIS_CAMERA_H_
#define _ARAVIS_CAMERA_H_

#include "DeviceBase.h"
#include "ImgBuffer.h"
#include "DeviceThreads.h"
#include "arv.h"


class AravisAcquisitionThread;


class AravisCamera : public CCameraBase<AravisCamera>
{
public:
   AravisCamera(const char *serialNumber);
   ~AravisCamera();
};


class AravisAcquisitionThread : public MMDeviceThreadBase
{
public:
   AravisAcquisitionThread(AravisCamera *aCam);
   ~AravisAcquisitionThread();
};

#endif // !_ARAVIS_CAMERA_H_

