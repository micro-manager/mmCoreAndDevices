#ifndef _ARAVIS_CAMERA_H_
#define _ARAVIS_CAMERA_H_

#include <iostream>
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
  AravisCamera(const char *);
  ~AravisCamera();

  // MMDevice API.
  void GetName(char* name) const;  
  int Initialize();
  int Shutdown();
  
  // MMCamera API.
  int ClearROI();
  int GetBinning() const;
  unsigned GetBitDepth() const;
  double GetExposure() const;
  const unsigned char* GetImageBuffer();
  long GetImageBufferSize() const;
  unsigned GetImageBytesPerPixel() const;
  unsigned GetImageWidth() const;
  unsigned GetImageHeight() const;
  unsigned GetNumberOfComponents() const;
  int GetROI(unsigned& x, unsigned& y, unsigned& xSize, unsigned& ySize);
  int IsExposureSequenceable(bool& isSequenceable) const;  
  int SetBinning(int binSize);
  void SetExposure(double exp);
  int SetROI(unsigned x, unsigned y, unsigned xSize, unsigned ySize);
  int SnapImage();

private:
  bool initialized_;
  long img_buffer_size;
  
  ArvBuffer *arv_buffer;
  ArvCamera *arv_cam;
  char *arv_cam_name;
  unsigned char *img_buffer;
};


class AravisAcquisitionThread : public MMDeviceThreadBase
{
public:
   AravisAcquisitionThread(AravisCamera *aCam);
   ~AravisAcquisitionThread();
};

#endif // !_ARAVIS_CAMERA_H_

