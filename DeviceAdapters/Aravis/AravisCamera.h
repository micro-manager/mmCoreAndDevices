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

#include <atomic>
#include <mutex>
#include <string>


#define ARV_ERROR 3141  // Should this be something specific?

// Told apart from ARV_ERROR because it is the one a user is most likely to
// hit and least able to diagnose from a number: the camera named in a saved
// configuration is not answering.
#define ARV_ERROR_NO_CAMERA 3142

// The camera answered, but offers nothing this adapter can turn into an image.
#define ARV_ERROR_NO_SUPPORTED_FORMAT 3143

// SnapImage() waits this multiple of the exposure time for a frame, and never
// less than the floor. Generous, because the alternative to a wrong guess is a
// spurious timeout on a slow link -- but finite, because Aravis treats a zero
// timeout as "block forever", which hangs the application.
#define ARV_SNAP_EXPOSURE_FACTOR 5.0
#define ARV_SNAP_MIN_TIMEOUT_US  5000000  // 5 seconds

// The camera's own pixel format, under its GenICam name. Micro-Manager's
// PixelType property is a different thing in a different vocabulary, so it
// gets a different property.
#define ARV_PROP_PIXEL_FORMAT "PixelFormat"


class AravisAcquisitionThread;


class AravisCamera : public CCameraBase<AravisCamera>
{
public:
  AravisCamera(const char *);
  ~AravisCamera();

  // MMDevice API.
  bool Busy() { return false; }
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
  int OnPixelFormat(MM::PropertyBase* pProp, MM::ActionType eAct);
  int OnPixelType(MM::PropertyBase* pProp, MM::ActionType eAct);
  int OnTriggerMode(MM::PropertyBase* pProp, MM::ActionType eAct);
  int OnTriggerSelector(MM::PropertyBase* pProp, MM::ActionType eAct);
  int OnTriggerSource(MM::PropertyBase* pProp, MM::ActionType eAct);

  // Internal.
  void AcquisitionCallback(ArvStreamCallbackType, ArvBuffer *);
  void ArvBufferUpdate(ArvBuffer *aBuffer);
  int ArvCheckError(GError **gerror) const;
  void ArvGeometryUpdate();
  void ArvGetExposure();
  void ArvPixelFormatUpdate(guint32 arvPixelFormat);
  void ArvSequenceFinished();
  int ArvStartSequenceAcquisition();


private:
  // Written by the Micro-Manager thread and read by the Aravis stream
  // callback thread, so plain bool is a data race.
  std::atomic<bool> capturing;
  long counter;

  // How many frames this sequence was asked for; zero or less means until
  // Micro-Manager says stop. Read on the stream thread, written on the
  // Micro-Manager thread before the stream exists.
  std::atomic<long> num_images;

  double exposure_time;

  // Whether this camera has binning at all, answered once by Initialize().
  // Asking Aravis for a feature the camera does not have fails every time, and
  // Micro-Manager asks for binning on a timer, so the answer has to be
  // remembered rather than rediscovered.
  bool has_binning;

  // The rest of what this camera can and cannot do, asked once. Each is a
  // separate question: a camera can have a settable size and a fixed offset,
  // or an exposure time and no frame rate control, and guessing one from
  // another is how the adapter ended up calling features that do not exist.
  bool has_exposure_time;
  bool has_frame_rate;
  bool has_region_offset;
  bool has_settable_region;

  unsigned img_buffer_bit_depth;
  int img_buffer_bytes_per_pixel;
  int img_buffer_height;
  unsigned img_buffer_number_components;
  size_t img_buffer_number_pixels;
  size_t img_buffer_size;

  // Whether the camera's red and blue channels have to be exchanged on the
  // way into Micro-Manager's BGRA buffer.
  bool img_buffer_swap_rb;

  int img_buffer_width;
  bool initialized;

  // Guards img_buffer and the size/format fields describing it. The stream
  // callback may reallocate the buffer while the Micro-Manager thread is
  // reading it.
  mutable std::mutex img_buffer_mutex;

  ArvBuffer *arv_buffer;
  ArvCamera *arv_cam;
  std::string arv_cam_name;
  ArvDevice *arv_device;

  // The format the image description was last built from. Kept so that a
  // format the adapter cannot decode is reported when it changes rather than
  // once per frame.
  guint32 arv_pixel_format;

  ArvStream *arv_stream;
  unsigned char *img_buffer;
  const char *pixel_type;
};

#endif // !_ARAVIS_CAMERA_H_

