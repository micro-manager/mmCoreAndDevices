/*
  Copyright 2024 Hazen Babcock
  
  Redistribution and use in source and binary forms, with or without modification, 
  are permitted provided that the following conditions are met:
  
  1. Redistributions of source code must retain the above copyright notice, this 
     list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright notice, this 
     list of conditions and the following disclaimer in the documentation and/or 
     other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its contributors may 
     be used to endorse or promote products derived from this software without 
     specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY 
  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES 
  OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT 
  SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED 
  TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR 
  BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN 
  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH 
  DAMAGE.
*/

#ifndef _ARAVIS_CAMERA_H_
#define _ARAVIS_CAMERA_H_

/*
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
*/
#include "DeviceBase.h"
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

  // Continuous acquisition.
  bool IsCapturing();
  int PrepareSequenceAcquisition();
  int StartSequenceAcquisition(long numImages, double interval_ms, bool stopOnOverflow);
  int StartSequenceAcquisition(double interval_ms);
  int StopSequenceAcquisition();

  // Properties.
  int OnAutoBlackLevel(MM::PropertyBase* pProp, MM::ActionType eAct);
  int OnAutoGain(MM::PropertyBase* pProp, MM::ActionType eAct);
  int OnBinning(MM::PropertyBase* pProp, MM::ActionType eAct);
  int OnBlackLevel(MM::PropertyBase* pProp, MM::ActionType eAct);
  int OnGain(MM::PropertyBase* pProp, MM::ActionType eAct);
  int OnGamma(MM::PropertyBase* pProp, MM::ActionType eAct);
  int OnGammaEnable(MM::PropertyBase* pProp, MM::ActionType eAct);
  int OnPixelType(MM::PropertyBase* pProp, MM::ActionType eAct);
  int OnTriggerMode(MM::PropertyBase* pProp, MM::ActionType eAct);
  int OnTriggerSelector(MM::PropertyBase* pProp, MM::ActionType eAct);
  int OnTriggerSource(MM::PropertyBase* pProp, MM::ActionType eAct);

  // Internal.
  void AcquisitionCallback(ArvStreamCallbackType, ArvBuffer *);
  void ArvBufferUpdate(ArvBuffer *aBuffer);
  int ArvCheckError(GError *gerror) const;
  void ArvGetExposure();
  void ArvPixelFormatUpdate(guint32 arvPixelFormat);
  int ArvStartSequenceAcquisition();

  
private:
  bool capturing;
  long counter;
  double exposure_time;
  unsigned img_buffer_bit_depth;
  int img_buffer_bytes_per_pixel;
  int img_buffer_height;
  unsigned img_buffer_number_components;
  size_t img_buffer_number_pixels;
  size_t img_buffer_size;
  int img_buffer_width;
  bool initialized;

  ArvBuffer *arv_buffer;
  ArvCamera *arv_cam;
  char *arv_cam_name;
  ArvDevice *arv_device;
  ArvStream *arv_stream;
  unsigned char *img_buffer;
  const char *pixel_type;
  const char *trigger;
};

#endif // !_ARAVIS_CAMERA_H_

