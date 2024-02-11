#ifndef _ARAVIS_CAMERA_H_
#define _ARAVIS_CAMERA_H_

#include <stdlib.h>
#include <stdio.h>


#include "DeviceBase.h"
#include "ImgBuffer.h"
#include "DeviceThreads.h"
#include "arv.h"
#include "glib.h"


#define ARV_ERROR 3141  // Should this be something specific?


class AravisAcquisitionThread;


class AravisCamera : public CCameraBase<AravisCamera>
{
public:
  AravisCamera(const char *serialNumber);
  ~AravisCamera();

  // MM required functions.
  int ClearROI();
  int GetBinning() const;
  unsigned GetBitDepth() const;
  double GetExposure() const;
  const unsigned char* GetImageBuffer();
  long GetImageBufferSize() const;
  unsigned GetImageBytesPerPixel() const;
  unsigned GetImageWidth() const;
  unsigned GetImageHeight() const;
  void GetName(char* name) const;
  unsigned GetNumberOfComponents() const;
  int GetROI(unsigned& x, unsigned& y, unsigned& xSize, unsigned& ySize);
  int Initialize();
  int IsExposureSequenceable(bool& isSequenceable) const;  
  int SetBinning(int binSize);
  void SetExposure(double exp);
  int SetROI(unsigned x, unsigned y, unsigned xSize, unsigned ySize);
  int Shutdown();
  int SnapImage();

  // Variables.
  ArvBuffer *a_buffer;
  ArvCamera *a_cam;
  const char *a_cam_name;  

};


class AravisAcquisitionThread : public MMDeviceThreadBase
{
public:
   AravisAcquisitionThread(AravisCamera *aCam);
   ~AravisAcquisitionThread();
};

#endif // !_ARAVIS_CAMERA_H_

