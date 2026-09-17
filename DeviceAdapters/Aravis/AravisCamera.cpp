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

#include "AravisCamera.h"

#include "CameraImageMetadata.h"

#include <vector>
#include <string>
#include <algorithm>
#include <cstdint>

// Values for the PixelType property. These are the strings the rest of the
// device adapters use -- see DemoCamera -- and they are deliberately not
// MMDevice's PixelType keyword constants: those belong to the image metadata
// tag of the same name, which MMCore writes itself from the byte depth and the
// component count.
const char *g_PixelType_8bit     = "8bit";
const char *g_PixelType_16bit    = "16bit";
const char *g_PixelType_32bitRGB = "32bitRGB";
const char *g_PixelType_Unknown  = "Unknown";

std::vector<std::string> supportedPixelFormats = {
  "Mono8",
  "Mono10",
  "Mono12",
  "Mono14",
  "Mono16",
  "BayerRG8",
  "BayerRG10",
  "BayerRG12",
  "BayerRG16",
  "RGB8",
  "BGR8"
};


/*
 * Module functions.
 */
MODULE_API void InitializeModuleData()
{
  uint64_t nDevices=0;

  // Debugging.
  // arv_debug_enable("all:1,device");

  // Update and get number of aravis compatible cameras.
  arv_update_device_list();
  nDevices = arv_get_n_devices();
  
  for (int i = 0; i < nDevices; i++)
  {
    RegisterDevice(arv_get_device_id(i), MM::CameraDevice, "Aravis Camera");
  }
}


MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
  return new AravisCamera(deviceName);
}


MODULE_API void DeleteDevice(MM::Device* pDevice)
{
  delete pDevice;
}


// A GenICam increment that could not be read, or that a camera reports as
// zero, is not a step size -- and dividing by it is a division by zero. One is
// the identity, so it is both safe and correct when there is no granularity to
// honour.
static gint ArvIncrement(gint increment)
{
  return (increment > 0) ? increment : 1;
}


// RGB unpacker.
//
// Micro-Manager's RGB32 is BGRA in memory, and its documentation says the
// alpha byte "is not used and should contain zeroes". Neither held before: the
// three colour bytes were copied in the order the camera sent them, so an RGB8
// camera came out with red and blue exchanged, and the alpha byte was left
// holding whatever the image buffer held last -- zeros on a fresh allocation,
// stale pixels once it had been reused.
void rgb_to_rgba(unsigned char *dest, unsigned char *source, size_t size, bool swap_rb)
{
  size_t i;
  size_t dOffset = 0;
  size_t sOffset = 0;

  for (i = 0; i < size; i++){
    if (swap_rb){
      dest[dOffset]     = source[sOffset + 2];
      dest[dOffset + 1] = source[sOffset + 1];
      dest[dOffset + 2] = source[sOffset];
    }
    else{
      memcpy(dest + dOffset, source + sOffset, 3);
    }
    dest[dOffset + 3] = 0;
    sOffset += 3;
    dOffset += 4;
  }
}


// Sequence acquisition callback.
static void
stream_callback (void *user_data, ArvStreamCallbackType type, ArvBuffer *arv_buffer)
{
  AravisCamera *camera = (AravisCamera *) user_data;

  camera->AcquisitionCallback(type, arv_buffer);
}


/*
 * Camera class and methods.
 */
AravisCamera::AravisCamera(const char *name) :
  capturing(false),
  counter(0),
  num_images(-1),
  exposure_time(0.0),
  has_binning(false),
  has_exposure_time(false),
  has_frame_rate(false),
  has_region_offset(false),
  has_settable_region(false),
  img_buffer_bit_depth(0),
  img_buffer_bytes_per_pixel(0),
  img_buffer_height(0),
  img_buffer_number_components(0),
  img_buffer_number_pixels(0),
  img_buffer_size(0),
  img_buffer_swap_rb(false),
  img_buffer_width(0),
  initialized(false),
  arv_buffer(nullptr),
  arv_cam(nullptr),
  arv_cam_name(name ? name : ""),
  arv_device(nullptr),
  arv_pixel_format(0),
  arv_stream(nullptr),
  img_buffer(nullptr),
  pixel_type(nullptr)
{
  // A bare "3141" in a dialog tells a user nothing. Give the two codes this
  // adapter returns some words, and name the camera in the one that means it
  // did not answer -- that is the failure a user meets when a camera in a
  // saved configuration is switched off or has moved to another subnet.
  SetErrorText(ARV_ERROR,
	       "The Aravis library reported an error. The message from the "
	       "camera is in the CoreLog.");
  SetErrorText(ARV_ERROR_NO_CAMERA,
	       ("No camera answered to the id '" + arv_cam_name + "'. Check "
		"that it is powered on, connected, and on the same subnet as "
		"this computer.").c_str());
  SetErrorText(ARV_ERROR_NO_SUPPORTED_FORMAT,
	       "This camera offers no pixel format that the Aravis adapter can "
	       "decode. The formats it does offer are listed in the CoreLog.");

  // The name was previously copied into malloc(strlen(name)) with
  // CDeviceUtils::CopyLimitedString(), which writes strlen(name) + 1 bytes.
  // That put the terminating NUL one byte past the end of the allocation on
  // every camera. A std::string removes the arithmetic entirely.
}


AravisCamera::~AravisCamera()
{
  // Shutdown() is idempotent, and Micro-Manager does not guarantee it ran.
  Shutdown();
  g_clear_object(&arv_cam);
}


// These are in alphabetical order.
void AravisCamera::AcquisitionCallback(ArvStreamCallbackType type, ArvBuffer *cb_arv_buffer)
{
  size_t size;
  unsigned char *cb_arv_buffer_data;
  int inserted = DEVICE_OK;

  MM::CameraImageMetadata md;

  if (!capturing){
    return;
  }
      
  switch (type) {
    /* Do we need this? IDK. */
  case ARV_STREAM_CALLBACK_TYPE_INIT:
    arv_make_thread_realtime (10);
    arv_make_thread_high_priority(-10);
    break;
  case ARV_STREAM_CALLBACK_TYPE_BUFFER_DONE:
    {
    // Pop the completed buffer. This used to sit inside g_assert(), which
    // makes the stream depend on an assertion: built with G_DISABLE_ASSERT the
    // pop would vanish and the stream would starve once its buffers ran out,
    // and a genuine mismatch would abort Micro-Manager rather than be handled.
    ArvBuffer *popped_arv_buffer = arv_stream_pop_buffer(arv_stream);

    if (popped_arv_buffer == NULL){
      LogMessage("Aravis Error, stream returned a NULL buffer", false);
      break;
    }
    if (popped_arv_buffer != cb_arv_buffer){
      // Not expected: the callback reports the buffer the stream just
      // completed, which is the one at the head of the output queue. Trust the
      // popped buffer, since that is the one we now own and must push back.
      LogMessage("Aravis Error, popped buffer is not the completed buffer", false);
    }

    {
      // ArvBufferUpdate() may reallocate img_buffer, and InsertImage() reads
      // it, so both are held under the lock that GetImageBuffer() also takes.
      std::lock_guard<std::mutex> lock(img_buffer_mutex);

      ArvBufferUpdate(popped_arv_buffer);

      // Image metadata.
      md.AddTag(MM::g_Keyword_Metadata_CameraLabel, "");
      md.AddTag(MM::g_Keyword_Metadata_ROI_X, CDeviceUtils::ConvertToString((long)img_buffer_width));
      md.AddTag(MM::g_Keyword_Metadata_ROI_Y, CDeviceUtils::ConvertToString((long)img_buffer_height));
      md.AddTag(MM::g_Keyword_Metadata_ImageNumber, CDeviceUtils::ConvertToString(counter));
      md.AddTag(MM::g_Keyword_Metadata_Exposure, exposure_time);
      md.AddTag(MM::g_Keyword_PixelType, pixel_type);

      // Pass data to MM.
      inserted = GetCoreCallback()->InsertImage(this,
					       img_buffer,
					       img_buffer_width,
					       img_buffer_height,
					       img_buffer_bytes_per_pixel,
					       1,
					       md.Serialize());
    }

    arv_stream_push_buffer(arv_stream, popped_arv_buffer);
    counter += 1;

    // MMDevice's contract: stop on any InsertImage() error. The Core no longer
    // reports a buffer overflow when it was told not to stop on one, so an
    // error here always means the frame could not be delivered.
    if (inserted != DEVICE_OK){
      std::stringstream msg;
      msg << "Aravis Error, stopping the sequence: the Core refused frame "
	  << counter << " with error " << inserted;
      LogMessage(msg.str(), false);
      ArvSequenceFinished();
      break;
    }

    // A finite sequence has to stop itself. Micro-Manager asks for a number of
    // frames and then waits for the camera to say it is done; this used to
    // stream until something else stopped it, so a request for 8 frames
    // delivered hundreds and never finished.
    long wanted = num_images;
    if ((wanted > 0) && (counter >= wanted)){
      ArvSequenceFinished();
    }
    break;
    }
  }
}

void AravisCamera::ArvBufferUpdate(ArvBuffer *aBuffer)
{
  int status;
  size_t arvSize, size;
  guint32 arvPixelFormat;
  unsigned char *arvBufferData;
  
  status = arv_buffer_get_status(aBuffer);
  if (status != 0){
    std::stringstream msg;
    switch (status) {
    case ARV_BUFFER_STATUS_UNKNOWN:
      msg << "Aravis Error, Aravis buffer status is 'UNKNOWN'";
      LogMessage(msg.str(), false);
      return;
    case ARV_BUFFER_STATUS_CLEARED:
      msg << "Aravis Error, Aravis buffer status is 'CLEARED'";
      LogMessage(msg.str(), false);
      return;
    case ARV_BUFFER_STATUS_TIMEOUT:
      msg << "Aravis Error, Aravis buffer status is 'TIMEOUT'";
      LogMessage(msg.str(), false);
      return;
    case ARV_BUFFER_STATUS_MISSING_PACKETS:
      msg << "Aravis Error, Aravis buffer status is 'MISSING PACKETS'";
      LogMessage(msg.str(), false);
      return;
    case ARV_BUFFER_STATUS_WRONG_PACKET_ID:
      msg << "Aravis Error, Aravis buffer status is 'WRONG_PACKET_ID'";
      LogMessage(msg.str(), false);
      return;
    case ARV_BUFFER_STATUS_SIZE_MISMATCH:
      msg << "Aravis Error, Aravis buffer status is 'SIZE_MISMATCH'";
      LogMessage(msg.str(), false);
      return;
    case ARV_BUFFER_STATUS_FILLING:
      msg << "Aravis Error, Aravis buffer status is 'FILLING'";
      LogMessage(msg.str(), false);
      return;
    case ARV_BUFFER_STATUS_ABORTED:
      msg << "Aravis Error, Aravis buffer status is 'ABORTED'";
      LogMessage(msg.str(), false);
      return;
    case ARV_BUFFER_STATUS_PAYLOAD_NOT_SUPPORTED:
      msg << "Aravis Error, Aravis buffer status is 'PAYLOAD_NOT_SUPPORTED'";
      LogMessage(msg.str(), false);
      return;
    default:
      msg << "Aravis Error, Aravis buffer status is 'UNKNOWN_STATUS'";
      LogMessage(msg.str(), false);
      return;
    }
  }

  // Pixel format updates.
  arvPixelFormat = arv_buffer_get_image_pixel_format(aBuffer);
  ArvPixelFormatUpdate(arvPixelFormat);

  // A format with no case in ArvPixelFormatUpdate() leaves these at zero.
  // Zero components is not one, so the copy below would take the RGB path and
  // write four bytes per pixel into a buffer sized for zero. Refuse instead.
  //
  // Silently: this runs once per frame, and ArvPixelFormatUpdate() has just
  // named the format and said it is not implemented. Repeating that here for
  // every frame of a live acquisition buries the rest of the log.
  if ((img_buffer_bytes_per_pixel < 1) || (img_buffer_number_components < 1)){
    return;
  }

  // Image size updates.
  img_buffer_width = (int)arv_buffer_get_image_width(aBuffer);
  img_buffer_height = (int)arv_buffer_get_image_height(aBuffer);
  img_buffer_number_pixels = img_buffer_width * img_buffer_height;

  // Copy buffer to MM.
  arvBufferData = (unsigned char *)arv_buffer_get_data(aBuffer, &arvSize);
  size = img_buffer_width * img_buffer_height * img_buffer_bytes_per_pixel;

  // The source must hold everything the destination is about to be filled
  // with. For the packed RGB formats the camera sends three bytes per pixel
  // and we expand to four, so compare against what will actually be read.
  size_t arvNeeded = (img_buffer_number_components == 1)
    ? size
    : img_buffer_number_pixels * 3;
  if (arvSize < arvNeeded){
    std::stringstream msg;
    msg << "Aravis Error, buffer holds " << arvSize << " bytes but "
	<< arvNeeded << " are needed for a " << img_buffer_width << "x"
	<< img_buffer_height << " image";
    LogMessage(msg.str(), false);
    return;
  }

  if (img_buffer_size != size){
    if (img_buffer != nullptr){
      free(img_buffer);
    }
    img_buffer = (unsigned char *)malloc(size);
    img_buffer_size = size;
  }
  if (img_buffer_number_components == 1){
    memcpy(img_buffer, arvBufferData, size);
  }
  else{
    rgb_to_rgba(img_buffer, arvBufferData, img_buffer_number_pixels,
		img_buffer_swap_rb);
  }
}


// Log and clear an Aravis error, if there is one.
//
// The GError is taken by address, not by value. Taken by value, g_clear_error()
// freed the error but cleared only this function's own copy of the pointer,
// leaving the caller holding a dangling non-NULL pointer. The caller would then
// pass that pointer to its next Aravis call and check it again, reading freed
// memory. Any camera that refused arv_camera_set_region() crashed Micro-Manager
// during Initialize() this way.
int AravisCamera::ArvCheckError(GError **gerror) const
{
  if ((gerror != NULL) && (*gerror != NULL)) {
    std::stringstream msg;
    msg << "Aravis Error: " << (*gerror)->message;
    LogMessage(msg.str(), false);
    g_clear_error(gerror);
    return 1;
  }
  return 0;
}


// Read the camera's region and describe the image with it.
//
// Micro-Manager asks for the image dimensions and the buffer size before it
// has seen a single frame -- it sizes its circular buffer from them -- and
// again immediately after any change to the region or the binning. Both
// questions used to be answered from whatever the last frame had set: nothing
// at all before the first snap, and the previous geometry after every change.
void AravisCamera::ArvGeometryUpdate()
{
  gint gx, gy, gwidth, gheight;
  GError *gerror = nullptr;

  arv_camera_get_region(arv_cam, &gx, &gy, &gwidth, &gheight, &gerror);
  if (ArvCheckError(&gerror)){
    return;
  }

  std::lock_guard<std::mutex> lock(img_buffer_mutex);
  img_buffer_width = (int)gwidth;
  img_buffer_height = (int)gheight;
  img_buffer_number_pixels = (size_t)gwidth * (size_t)gheight;
}


// Call the Aravis library to check exposure time only as needed.
void AravisCamera::ArvGetExposure()
{
  double expTimeUs;
  GError *gerror = nullptr;

  if (!has_exposure_time){
    return;
  }

  expTimeUs = arv_camera_get_exposure_time(arv_cam, &gerror);
  if(!ArvCheckError(&gerror)){
    exposure_time = expTimeUs * 1.0e-3;
  }
}


// Update MM image values based on pixel format.
//
// PixelType here says what the buffer Micro-Manager is handed looks like, in
// the vocabulary the other device adapters use. The camera's own format name
// lives in the PixelFormat property. Previously this set a third set of
// strings ("8bit mono", "8bitRGB") that belonged to neither.
//
// Mono10, Mono12 and Mono14 are all 16bit: two bytes per pixel, lsb aligned,
// with the sensor's real depth reported separately by GetBitDepth().
void AravisCamera::ArvPixelFormatUpdate(guint32 arvPixelFormat)
{
  // ArvBufferUpdate() calls this for every frame. A format the adapter has no
  // case for is a standing fact about the camera, not a per-frame event, so it
  // is worth saying only when the format has actually changed.
  bool format_changed = (arvPixelFormat != arv_pixel_format);
  arv_pixel_format = arvPixelFormat;

  // Only the packed RGB formats set this; it is meaningless for the rest.
  img_buffer_swap_rb = false;

  switch (arvPixelFormat){
  case ARV_PIXEL_FORMAT_MONO_8:
    img_buffer_bit_depth = 8;
    img_buffer_bytes_per_pixel = 1;
    img_buffer_number_components = 1;
    pixel_type = g_PixelType_8bit;
    break;
  case ARV_PIXEL_FORMAT_MONO_10:
    img_buffer_bit_depth = 10;
    img_buffer_bytes_per_pixel = 2;
    img_buffer_number_components = 1;
    pixel_type = g_PixelType_16bit;
    break;
  case ARV_PIXEL_FORMAT_MONO_12:
    img_buffer_bit_depth = 12;
    img_buffer_bytes_per_pixel = 2;
    img_buffer_number_components = 1;
    pixel_type = g_PixelType_16bit;
    break;
  case ARV_PIXEL_FORMAT_MONO_14:
    img_buffer_bit_depth = 14;
    img_buffer_bytes_per_pixel = 2;
    img_buffer_number_components = 1;
    pixel_type = g_PixelType_16bit;
    break;
  case ARV_PIXEL_FORMAT_MONO_16:
    img_buffer_bit_depth = 16;
    img_buffer_bytes_per_pixel = 2;
    img_buffer_number_components = 1;
    pixel_type = g_PixelType_16bit;
    break;

  case ARV_PIXEL_FORMAT_BAYER_RG_8:
    img_buffer_bit_depth = 8;
    img_buffer_bytes_per_pixel = 1;
    img_buffer_number_components = 1;
    pixel_type = g_PixelType_8bit;
    break;
  case ARV_PIXEL_FORMAT_BAYER_RG_10:
    img_buffer_bit_depth = 10;
    img_buffer_bytes_per_pixel = 2;
    img_buffer_number_components = 1;
    pixel_type = g_PixelType_16bit;
    break;
  case ARV_PIXEL_FORMAT_BAYER_RG_12:
    img_buffer_bit_depth = 12;
    img_buffer_bytes_per_pixel = 2;
    img_buffer_number_components = 1;
    pixel_type = g_PixelType_16bit;
    break;
  case ARV_PIXEL_FORMAT_BAYER_RG_16:
    img_buffer_bit_depth = 16;
    img_buffer_bytes_per_pixel = 2;
    img_buffer_number_components = 1;
    pixel_type = g_PixelType_16bit;
    break;

  case ARV_PIXEL_FORMAT_RGB_8_PACKED:
    img_buffer_bit_depth = 8;
    img_buffer_bytes_per_pixel = 4;
    img_buffer_number_components = 4;
    // The camera sends R,G,B and Micro-Manager wants B,G,R,A.
    img_buffer_swap_rb = true;
    pixel_type = g_PixelType_32bitRGB;
    break;
  case ARV_PIXEL_FORMAT_BGR_8_PACKED:
    img_buffer_bit_depth = 8;
    img_buffer_bytes_per_pixel = 4;
    img_buffer_number_components = 4;
    // Already in Micro-Manager's order; only the alpha byte has to be added.
    pixel_type = g_PixelType_32bitRGB;
    break;

  default:
    // Leave the image description in a state callers can recognise as
    // unusable, rather than keeping the previous format's values and
    // describing the new data with them. printf() went to a console that
    // Micro-Manager users do not have; this belongs in the log.
    if (format_changed){
      std::stringstream msg;
      msg << "Aravis Error, pixel format " << (int)arvPixelFormat
	  << " is not implemented";
      LogMessage(msg.str(), false);
    }
    img_buffer_bit_depth = 0;
    img_buffer_bytes_per_pixel = 0;
    img_buffer_number_components = 0;
    pixel_type = g_PixelType_Unknown;
    break;
  }
}


// End a sequence from inside the stream callback.
//
// Deliberately not StopSequenceAcquisition(): that unreffs the stream, which
// stops and joins the stream's own thread -- the thread calling this one. The
// camera is stopped over the control channel, which is a different socket and
// safe from here, and the stream itself is released by whichever
// StopSequenceAcquisition() or Shutdown() comes next.
void AravisCamera::ArvSequenceFinished()
{
  GError *gerror = nullptr;

  if (!capturing.exchange(false)){
    return;
  }

  if (arv_cam != nullptr){
    arv_camera_stop_acquisition(arv_cam, &gerror);
    ArvCheckError(&gerror);
  }

  GetCoreCallback()->AcqFinished(this, 0);
}


int AravisCamera::ArvStartSequenceAcquisition()
{
  int i;
  size_t payload;
  GError *gerror = nullptr;

  counter = 0;
    
  arv_camera_set_acquisition_mode(arv_cam, ARV_ACQUISITION_MODE_CONTINUOUS, &gerror);
  if (!ArvCheckError(&gerror)){
    arv_stream = arv_camera_create_stream(arv_cam, stream_callback, this, &gerror);
    if (ArvCheckError(&gerror)){
      return 1;
    }
  }
  else{
    return 1;
  }
  
  if (ARV_IS_STREAM(arv_stream)){
    payload = arv_camera_get_payload(arv_cam, &gerror);
    if (!ArvCheckError(&gerror)){
      for (i = 0; i < 20; i++)
	arv_stream_push_buffer(arv_stream, arv_buffer_new(payload, NULL));
    }
    arv_camera_start_acquisition(arv_cam, &gerror);
    if (ArvCheckError(&gerror)){
      return 1;
    }
  }
  else{
    return 1;
  }
  capturing = true;
  return 0;
}


int AravisCamera::ClearROI()
{
  gint h,tmp,w;
  GError *gerror = nullptr;

  // A camera whose size is fixed is always at full frame. There is nothing to
  // clear, and asking would fail on every call -- which is what happened at
  // Initialize() on every such camera.
  if (!has_settable_region){
    ArvGeometryUpdate();
    return DEVICE_OK;
  }

  // The 64x64 intermediate this used to set first was superstition: it fails
  // outright on a camera whose minimum width is larger than 64 or whose
  // increment does not divide it, and it was never needed, because
  // arv_camera_set_region() already zeroes OffsetX and OffsetY before writing
  // the new size and restores them afterwards.
  arv_camera_get_height_bounds(arv_cam, &tmp, &h, &gerror);
  ArvCheckError(&gerror);

  arv_camera_get_width_bounds(arv_cam, &tmp, &w, &gerror);
  ArvCheckError(&gerror);

  arv_camera_set_region(arv_cam, 0, 0, w, h, &gerror);
  int ret = ArvCheckError(&gerror) ? ARV_ERROR : DEVICE_OK;

  ArvGeometryUpdate();

  return ret;
}


int AravisCamera::GetBinning() const
{
  gint dx;
  gint dy;
  GError *gerror = nullptr;

  // Micro-Manager calls this whenever it needs the binning factor, which
  // includes once per image while tagging metadata. On a camera without
  // binning the Aravis call fails every time, so a camera that simply does not
  // bin filled the log with "[BinningHorizontal] Not found" during live
  // acquisition. It bins by one, and that needs no camera to answer.
  if (!has_binning){
    return 1;
  }

  arv_camera_get_binning(arv_cam, &dx, &dy, &gerror);
  ArvCheckError(&gerror);

  // dx is always dy for MM? Add check?
  return (int)dx;
}


unsigned AravisCamera::GetBitDepth() const
{
  return img_buffer_bit_depth;
}


double AravisCamera::GetExposure() const
{
  return exposure_time;
}


const unsigned char* AravisCamera::GetImageBuffer()
{
  if (ARV_IS_BUFFER (arv_buffer)) {
    {
      std::lock_guard<std::mutex> lock(img_buffer_mutex);
      ArvBufferUpdate(arv_buffer);
    }
    g_clear_object(&arv_buffer);
    return img_buffer;
  }
  return NULL;
}


long AravisCamera::GetImageBufferSize() const
{
  return img_buffer_number_pixels * img_buffer_bytes_per_pixel;
}


unsigned AravisCamera::GetImageBytesPerPixel() const
{
  return img_buffer_bytes_per_pixel;
}


unsigned AravisCamera::GetImageWidth() const
{
  return (unsigned)img_buffer_width;
}


unsigned AravisCamera::GetImageHeight() const
{
  return (unsigned)img_buffer_height;
}


void AravisCamera::GetName(char *name) const
{
  CDeviceUtils::CopyLimitedString(name, arv_cam_name.c_str());
}


unsigned AravisCamera::GetNumberOfComponents() const
{
  return img_buffer_number_components;
}


int AravisCamera::GetROI(unsigned& x, unsigned& y, unsigned& xSize, unsigned& ySize)
{
  gint gx,gy,gwidth,gheight;
  GError *gerror = nullptr;

  arv_camera_get_region(arv_cam, &gx, &gy, &gwidth, &gheight, &gerror);
  if (ArvCheckError(&gerror)){
    return ARV_ERROR;
  }

  // y was assigned gx, and the two sizes were assigned to themselves, so every
  // caller got a garbage region -- on a camera at 0,0 the first two happened
  // to be right, which is why this survived.
  x = (unsigned)gx;
  y = (unsigned)gy;
  xSize = (unsigned)gwidth;
  ySize = (unsigned)gheight;

  return DEVICE_OK;
}


int AravisCamera::Initialize()
{
  int i,ret;
  GError *gerror = nullptr;

  if(initialized){
    return DEVICE_OK;
  }
  
  arv_cam = arv_camera_new(arv_cam_name.c_str(), &gerror);
  if (ArvCheckError(&gerror) || (arv_cam == nullptr)){
    return ARV_ERROR_NO_CAMERA;
  }

  arv_device = arv_camera_get_device(arv_cam);

  // What this camera can do, asked once, before anything relies on it.
  //
  // The region is two questions, not one. Aravis' region-offset probe reports
  // on OffsetX and OffsetY only, so a camera with a settable size and a fixed
  // offset -- or the reverse -- was handled wrongly in both directions. The
  // size is answered from the GenICam access mode of Width and Height, which
  // is what actually decides whether a write can succeed.
  has_region_offset = arv_camera_is_region_offset_available(arv_cam, &gerror);
  ArvCheckError(&gerror);
  has_settable_region =
    (arv_device_get_feature_access_mode(arv_device, "Width") == ARV_GC_ACCESS_MODE_RW) &&
    (arv_device_get_feature_access_mode(arv_device, "Height") == ARV_GC_ACCESS_MODE_RW);

  has_exposure_time = arv_camera_is_exposure_time_available(arv_cam, &gerror);
  ArvCheckError(&gerror);
  has_frame_rate = arv_camera_is_frame_rate_available(arv_cam, &gerror);
  ArvCheckError(&gerror);

  // Which camera is this? Nothing recorded it before, so a configuration with
  // two cameras of the same model said nothing about which was which, and a
  // support question could not be answered without walking to the microscope.
  auto describe = [&](const char *name, const char *value){
    CreateProperty(name, (value != nullptr) ? value : "", MM::String, true);
  };
  describe(MM::g_Keyword_Description, "Aravis GigE Vision / USB3 Vision camera");

  const char *info;
  info = arv_camera_get_vendor_name(arv_cam, &gerror);
  ArvCheckError(&gerror);
  describe("Vendor", info);

  info = arv_camera_get_model_name(arv_cam, &gerror);
  ArvCheckError(&gerror);
  describe(MM::g_Keyword_CameraName, info);

  info = arv_camera_get_device_serial_number(arv_cam, &gerror);
  ArvCheckError(&gerror);
  describe("SerialNumber", info);

  info = arv_camera_get_device_id(arv_cam, &gerror);
  ArvCheckError(&gerror);
  describe(MM::g_Keyword_CameraID, info);

  // Not standard enough to assume: ask before reading, or a camera without it
  // logs a failure at every open.
  if (arv_camera_is_feature_available(arv_cam, "DeviceVersion", &gerror)){
    ArvCheckError(&gerror);
    info = arv_device_get_string_feature_value(arv_device, "DeviceVersion", &gerror);
    ArvCheckError(&gerror);
    describe(MM::g_Keyword_Version, info);
  }
  ArvCheckError(&gerror);

  // Clear ROI settings that may still be present from a previous session.
  ClearROI();

  // Get starting image size. From the region the camera is actually in, not
  // from the largest one it could be in: a camera opening on a smaller region
  // described every image it produced with the wrong dimensions.
  ArvGeometryUpdate();

  // Set image properties based on current pixel type.
  guint32 arvPixelFormat;
  arvPixelFormat = arv_camera_get_pixel_format(arv_cam, &gerror);
  ArvCheckError(&gerror);
  ArvPixelFormatUpdate(arvPixelFormat);

  // Turn off auto exposure (if available).
  if(arv_camera_is_exposure_auto_available(arv_cam, &gerror)){
    ArvCheckError(&gerror);
    arv_camera_set_exposure_time_auto(arv_cam, ARV_AUTO_OFF, &gerror);
    ArvCheckError(&gerror);  
  }

  // Get current exposure time.
  ArvGetExposure();
  
  // Pixel formats, as two properties, because there are two vocabularies and
  // they are not the same one. PixelFormat selects what the camera sends,
  // under the camera's own GenICam names. PixelType describes the buffer
  // Micro-Manager is handed, in Micro-Manager's names. One property cannot be
  // both, and trying left it flipping between three different sets of strings
  // depending on which code path wrote it last.
  //
  // Sort the camera's formats before either property exists, because what the
  // adapter can decode decides whether this camera can be opened at all.
  guint nPixelFormats;
  std::vector<std::string> pixelFormatValues;
  std::vector<std::string> unsupportedFormats;
  const char **pixelFormats;

  pixelFormats = arv_camera_dup_available_pixel_formats_as_strings(arv_cam, &nPixelFormats, &gerror);
  ArvCheckError(&gerror);
  for(i=0;i<nPixelFormats;i++){
    if (std::find(supportedPixelFormats.begin(), supportedPixelFormats.end(), pixelFormats[i]) != supportedPixelFormats.end()){
      pixelFormatValues.push_back(pixelFormats[i]);
    }
    else{
      unsupportedFormats.push_back(pixelFormats[i]);
    }
  }
  g_free(pixelFormats);

  // Dropped formats used to vanish without a word, so a camera whose list came
  // back short looked like a camera with a short list.
  if (!unsupportedFormats.empty()){
    std::stringstream msg;
    msg << "Aravis: this camera offers " << unsupportedFormats.size()
	<< " pixel format(s) that this adapter does not implement, and which "
	<< "are therefore not offered in " << ARV_PROP_PIXEL_FORMAT << ":";
    for (const std::string &name : unsupportedFormats){
      msg << " " << name;
    }
    LogMessage(msg.str(), false);
  }

  // A camera this adapter cannot decode a single image from is not one it can
  // open. Opening it anyway produced a device reporting zero bytes per pixel,
  // which MMCore turns into "memory requirements not adequate" the first time
  // live acquisition starts -- a message that says nothing about the cause.
  if (pixelFormatValues.empty()){
    std::stringstream msg;
    msg << "Aravis Error, this camera offers no pixel format that this adapter "
	<< "implements. It offers:";
    for (const std::string &name : unsupportedFormats){
      msg << " " << name;
    }
    LogMessage(msg.str(), false);
    return ARV_ERROR_NO_SUPPORTED_FORMAT;
  }

  const char *pixel_format;
  pixel_format = arv_camera_get_pixel_format_as_string (arv_cam, &gerror);
  ArvCheckError(&gerror);

  // A camera can be sitting in a format the adapter cannot decode while also
  // offering one it can -- a packed mono default alongside Mono8, say. Move it
  // rather than open into a state that produces no images.
  if ((pixel_format == nullptr) ||
      (std::find(pixelFormatValues.begin(), pixelFormatValues.end(),
		 pixel_format) == pixelFormatValues.end())){
    std::stringstream msg;
    msg << "Aravis: camera is in pixel format "
	<< ((pixel_format != nullptr) ? pixel_format : "(unreadable)")
	<< ", which this adapter does not implement; switching to "
	<< pixelFormatValues[0];
    LogMessage(msg.str(), false);

    arv_camera_set_pixel_format_from_string(arv_cam, pixelFormatValues[0].c_str(), &gerror);
    if (ArvCheckError(&gerror)){
      return ARV_ERROR_NO_SUPPORTED_FORMAT;
    }

    pixel_format = arv_camera_get_pixel_format_as_string(arv_cam, &gerror);
    ArvCheckError(&gerror);

    arvPixelFormat = arv_camera_get_pixel_format(arv_cam, &gerror);
    ArvCheckError(&gerror);
    ArvPixelFormatUpdate(arvPixelFormat);
    ArvGeometryUpdate();
  }

  CPropertyAction* pAct = new CPropertyAction(this, &AravisCamera::OnPixelFormat);
  ret = CreateProperty(ARV_PROP_PIXEL_FORMAT,
		       (pixel_format != nullptr) ? pixel_format : "",
		       MM::String, false, pAct);
  assert(ret == DEVICE_OK);

  SetAllowedValues(ARV_PROP_PIXEL_FORMAT, pixelFormatValues);

  // Read-only, because it is derived rather than chosen: it follows whatever
  // format the camera is in.
  pAct = new CPropertyAction(this, &AravisCamera::OnPixelType);
  ret = CreateProperty(MM::g_Keyword_PixelType, pixel_type, MM::String, true, pAct);
  assert(ret == DEVICE_OK);
  
  // Binning.
  pAct = new CPropertyAction(this, &AravisCamera::OnBinning);
  ret = CreateProperty(MM::g_Keyword_Binning, "1", MM::Integer, false, pAct);
  assert(ret == DEVICE_OK);

  // Remembered, not just used here: every other binning entry point has to
  // know the answer too, or it will ask Aravis and log a failure each time.
  has_binning = arv_camera_is_binning_available(arv_cam, &gerror);
  ArvCheckError(&gerror);
  std::vector<std::string> binValues;
  if (has_binning){
    gint bmin,bmax,binc;

    //Assuming X/Y symmetric..
    arv_camera_get_x_binning_bounds(arv_cam, &bmin, &bmax, &gerror);
    ArvCheckError(&gerror);

    // Guarded, because binc is the step of the loop below: an increment the
    // camera could not report comes back as zero, and the loop never ends.
    binc = ArvIncrement(arv_camera_get_x_binning_increment(arv_cam, &gerror));
    ArvCheckError(&gerror);

    for (int x = bmin; x <= bmax; x += binc){
      binValues.push_back(std::to_string(x));
    }
  }
  else {
    binValues.push_back("1");
  }
  SetAllowedValues(MM::g_Keyword_Binning, binValues);
  
  // Auto gain.
  gboolean hasAutoGain;
  hasAutoGain = arv_camera_is_gain_auto_available(arv_cam, &gerror);
  ArvCheckError(&gerror);

  if (hasAutoGain){
    pAct = new CPropertyAction(this, &AravisCamera::OnAutoGain);
    ret = CreateProperty("GainAuto", "NA", MM::String, false, pAct);
    std::vector<std::string> autoGainValues = {"AUTO_OFF", "AUTO_ONCE", "AUTO_CONTINUOUS"};
    SetAllowedValues("GainAuto", autoGainValues);
  }

  // Gain.
  gboolean hasGain;
  hasGain = arv_camera_is_gain_available(arv_cam, &gerror);
  ArvCheckError(&gerror);  

  if (hasGain){
    double gmin,gmax;

    arv_camera_get_gain_bounds(arv_cam, &gmin, &gmax, &gerror);
    ArvCheckError(&gerror);
    
    pAct = new CPropertyAction(this, &AravisCamera::OnGain);
    ret = CreateProperty(MM::g_Keyword_Gain, "1.0", MM::Float, false, pAct);
    SetPropertyLimits(MM::g_Keyword_Gain, gmin, gmax);
  }

  // Auto black level.
  gboolean hasAutoBlackLevel;
  hasAutoBlackLevel = arv_camera_is_black_level_auto_available(arv_cam, &gerror);
  ArvCheckError(&gerror);

  if (hasAutoBlackLevel){
    pAct = new CPropertyAction(this, &AravisCamera::OnAutoBlackLevel);
    ret = CreateProperty("BlackLevelAuto", "NA", MM::String, false, pAct);
    assert(ret == DEVICE_OK);
    std::vector<std::string> autoBlackLevelValues = {"AUTO_OFF", "AUTO_ONCE", "AUTO_CONTINUOUS"};
    SetAllowedValues("BlackLevelAuto", autoBlackLevelValues);
  }
  
  // Black level.
  gboolean hasBlackLevel;
  hasBlackLevel = arv_camera_is_black_level_available(arv_cam, &gerror);
  ArvCheckError(&gerror);  

  if (hasBlackLevel){
    double bmin,bmax;

    arv_camera_get_black_level_bounds(arv_cam, &bmin, &bmax, &gerror);
    ArvCheckError(&gerror);
    
    pAct = new CPropertyAction(this, &AravisCamera::OnBlackLevel);
    ret = CreateProperty(MM::g_Keyword_Offset, "1.0", MM::Float, false, pAct);
    assert(ret == DEVICE_OK);
    SetPropertyLimits(MM::g_Keyword_Offset, bmin, bmax);
  }

  // Gamma.
  //
  // Check by getting the feature because if "GammaEnable" is turned off the
  // feature won't appear as available with arv_device_is_feature_avaialable().
  //
  ArvGcNode *hasGamma;
  hasGamma = arv_device_get_feature(arv_device, "Gamma");
  if (hasGamma != NULL){
    double gmin,gmax;

    arv_device_get_float_feature_bounds(arv_device, "Gamma", &gmin, &gmax, &gerror);
    ArvCheckError(&gerror);
    
    pAct = new CPropertyAction(this, &AravisCamera::OnGamma);
    ret = CreateProperty("Gamma", "1.0", MM::Float, false, pAct);
    assert(ret == DEVICE_OK);
    SetPropertyLimits("Gamma", gmin, gmax);    
  }

  // Gamma enable.
  gboolean hasGammaEnable;
  hasGammaEnable = arv_device_is_feature_available(arv_device, "GammaEnable", &gerror);
  ArvCheckError(&gerror);

  if (hasGammaEnable){
    pAct = new CPropertyAction(this, &AravisCamera::OnGammaEnable);
    ret = CreateProperty("GammaEnable", "0", MM::String, false, pAct);
    assert(ret == DEVICE_OK);
    std::vector<std::string> gammaEnableValues = {"0", "1"};
    SetAllowedValues("GammaEnable", gammaEnableValues);
  }
    
  // Trigger mode.
  guint nTriggerModes = 0;
  const char **triggerModes;
  triggerModes = arv_device_dup_available_enumeration_feature_values_as_strings(arv_device, "TriggerMode", &nTriggerModes, &gerror);
  ArvCheckError(&gerror);

  if (nTriggerModes > 1){
    CPropertyAction* pAct = new CPropertyAction(this, &AravisCamera::OnTriggerMode);
    ret = CreateProperty("TriggerMode", "NA", MM::String, false, pAct);
    assert(ret == DEVICE_OK);

    std::vector<std::string> triggerModeValues;
    for(i=0;i<nTriggerModes;i++){
      triggerModeValues.push_back(triggerModes[i]);
    }
    SetAllowedValues("TriggerMode", triggerModeValues);
  }
  g_free(triggerModes);
  
  // Trigger selector.
  guint nTriggerSelectors = 0;
  const char **triggerSelectors;
  triggerSelectors = arv_camera_dup_available_triggers(arv_cam, &nTriggerSelectors, &gerror);
  ArvCheckError(&gerror);

  if (nTriggerSelectors > 1){
    CPropertyAction* pAct = new CPropertyAction(this, &AravisCamera::OnTriggerSelector);
    ret = CreateProperty("TriggerSelector", "NA", MM::String, false, pAct);
    assert(ret == DEVICE_OK);

    std::vector<std::string> triggerSelectorValues;
    for(i=0;i<nTriggerSelectors;i++){
      triggerSelectorValues.push_back(triggerSelectors[i]);
    }
    SetAllowedValues("TriggerSelector", triggerSelectorValues);
  }
  g_free(triggerSelectors);
  
  // Trigger sources.
  guint nTriggerSources = 0;
  const char **triggerSources;
  triggerSources = arv_camera_dup_available_trigger_sources(arv_cam, &nTriggerSources, &gerror);
  ArvCheckError(&gerror);

  if (nTriggerSources > 1){
    CPropertyAction* pAct = new CPropertyAction(this, &AravisCamera::OnTriggerSource);
    ret = CreateProperty("TriggerSource", "NA", MM::String, false, pAct);
    assert(ret == DEVICE_OK);
    
    std::vector<std::string> triggerSourceValues;
    for(i=0;i<nTriggerSources;i++){
      triggerSourceValues.push_back(triggerSources[i]);
    }
    SetAllowedValues("TriggerSource", triggerSourceValues);
  }
  g_free(triggerSources);

  initialized = true;
    
  return DEVICE_OK;
}


// Not sure if these cameras are sequencable or not, going with not.
int AravisCamera::IsExposureSequenceable(bool &isSequencable) const
{
  isSequencable = false;

  return DEVICE_OK;
}


bool AravisCamera::IsCapturing()
{
  return capturing;
}


int AravisCamera::OnAutoBlackLevel(MM::PropertyBase* pProp, MM::ActionType eAct)
{
  GError *gerror = nullptr;

  if (eAct == MM::AfterSet){
    if (!capturing){
      std::string autoBlackLevelMode;
      pProp->Get(autoBlackLevelMode);
      
      if (!autoBlackLevelMode.compare("AUTO_OFF")){
	arv_camera_set_black_level_auto(arv_cam, ARV_AUTO_OFF, &gerror);
      }
      else if (!autoBlackLevelMode.compare("AUTO_ONCE")){
	arv_camera_set_black_level_auto(arv_cam, ARV_AUTO_ONCE, &gerror);
      }
      else if (!autoBlackLevelMode.compare("AUTO_CONTINUOUS")){
	arv_camera_set_black_level_auto(arv_cam, ARV_AUTO_CONTINUOUS, &gerror);
      }
      else{
	printf("Unrecognized auto black level mode %s", autoBlackLevelMode.c_str());
      }
      ArvCheckError(&gerror);
    }
  }
  else if (eAct == MM::BeforeGet) {
    int mode;
    mode = arv_camera_get_black_level_auto(arv_cam, &gerror);
    ArvCheckError(&gerror);

    if (mode == ARV_AUTO_OFF){
      pProp->Set("AUTO_OFF");
    }
    else if (mode == ARV_AUTO_ONCE){
      pProp->Set("AUTO_ONCE");
    }
    else if (mode == ARV_AUTO_CONTINUOUS){
      pProp->Set("AUTO_CONTINUOUS");
    }
  }
  
  return DEVICE_OK;
}


int AravisCamera::OnAutoGain(MM::PropertyBase* pProp, MM::ActionType eAct)
{
  GError *gerror = nullptr;

  if (eAct == MM::AfterSet){
    if (!capturing){
      std::string autoGainMode;
      pProp->Get(autoGainMode);
      
      if (!autoGainMode.compare("AUTO_OFF")){
	arv_camera_set_gain_auto(arv_cam, ARV_AUTO_OFF, &gerror);
      }
      else if (!autoGainMode.compare("AUTO_ONCE")){
	arv_camera_set_gain_auto(arv_cam, ARV_AUTO_ONCE, &gerror);
      }
      else if (!autoGainMode.compare("AUTO_CONTINUOUS")){
	arv_camera_set_gain_auto(arv_cam, ARV_AUTO_CONTINUOUS, &gerror);
      }
      else{
	printf("Unrecognized auto gain mode %s", autoGainMode.c_str());
      }
      ArvCheckError(&gerror);
    }
  }
  else if (eAct == MM::BeforeGet) {
    int mode;
    mode = arv_camera_get_gain_auto(arv_cam, &gerror);
    ArvCheckError(&gerror);

    if (mode == ARV_AUTO_OFF){
      pProp->Set("AUTO_OFF");
    }
    else if (mode == ARV_AUTO_ONCE){
      pProp->Set("AUTO_ONCE");
    }
    else if (mode == ARV_AUTO_CONTINUOUS){
      pProp->Set("AUTO_CONTINUOUS");
    }
  }
  
  return DEVICE_OK;
}

  
int AravisCamera::OnBinning(MM::PropertyBase* pProp, MM::ActionType eAct)
{
  gint bx,by;
  std::string binning;
  GError *gerror = nullptr;

  // Micro-Manager refreshes properties on a timer, so a camera without binning
  // logged an Aravis failure here once per refresh, forever. The property is
  // still offered, fixed at 1, because Micro-Manager expects cameras to have
  // one; it just no longer costs a failed register read to report it.
  if (!has_binning){
    if (eAct == MM::BeforeGet){
      pProp->Set(1L);
    }
    return DEVICE_OK;
  }

  if (eAct == MM::AfterSet){
    if (!capturing){
      pProp->Get(binning);
      bx = std::stoi(binning);
      
      arv_camera_set_binning(arv_cam, bx, bx, &gerror);
      ArvCheckError(&gerror);
      
      // This restores the image size when we decrease the binning.
      ClearROI();
    }    
  }
  else if (eAct == MM::BeforeGet) {
    arv_camera_get_binning(arv_cam, &bx, &by, &gerror);
    ArvCheckError(&gerror);

    std::string bxs = std::to_string(bx);
    pProp->Set(bxs.c_str());
  }
  
  return DEVICE_OK;
}


int AravisCamera::OnBlackLevel(MM::PropertyBase* pProp, MM::ActionType eAct)
{
  double blackLevel;
  GError *gerror = nullptr;

  if (eAct == MM::AfterSet){
    int mode;
    mode = arv_camera_get_black_level_auto(arv_cam, &gerror);
    ArvCheckError(&gerror);

    if (mode == ARV_AUTO_OFF){
      pProp->Get(blackLevel);
      arv_camera_set_black_level(arv_cam, blackLevel, &gerror);
      ArvCheckError(&gerror);
    }
  }
  else if (eAct == MM::BeforeGet){
    blackLevel = arv_camera_get_black_level(arv_cam, &gerror);
    ArvCheckError(&gerror);

    pProp->Set(blackLevel);
  }
  return DEVICE_OK;
}


int AravisCamera::OnGain(MM::PropertyBase* pProp, MM::ActionType eAct)
{
  double gain;
  GError *gerror = nullptr;

  if (eAct == MM::AfterSet){
    int mode;
    mode = arv_camera_get_gain_auto(arv_cam, &gerror);
    ArvCheckError(&gerror);

    if (mode == ARV_AUTO_OFF){
      pProp->Get(gain);	  
      arv_camera_set_gain(arv_cam, gain, &gerror);
      ArvCheckError(&gerror);
    }
  }
  else if (eAct == MM::BeforeGet) {
    gain = arv_camera_get_gain(arv_cam, &gerror);
    ArvCheckError(&gerror);

    pProp->Set(gain);
  }
  return DEVICE_OK;
}


int AravisCamera::OnGamma(MM::PropertyBase* pProp, MM::ActionType eAct)
{
  double gamma;
  GError *gerror = nullptr;

  if (eAct == MM::AfterSet){
    pProp->Get(gamma);
    arv_device_set_float_feature_value(arv_device, "Gamma", gamma, &gerror);
    ArvCheckError(&gerror);
  }
  else if (eAct == MM::BeforeGet){
    gamma = arv_device_get_float_feature_value(arv_device, "Gamma", &gerror);
    ArvCheckError(&gerror);
    pProp->Set(gamma);
  }
  return DEVICE_OK;
}


int AravisCamera::OnGammaEnable(MM::PropertyBase* pProp, MM::ActionType eAct)
{
  gboolean ge;
  std::string gammaEnable;
  GError *gerror = nullptr;

  if (eAct == MM::AfterSet){
    pProp->Get(gammaEnable);
    ge = std::stoi(gammaEnable);
    arv_device_set_boolean_feature_value(arv_device, "GammaEnable", ge, &gerror);
    ArvCheckError(&gerror);
  }
  else if (eAct == MM::BeforeGet){
    ge = arv_device_get_boolean_feature_value(arv_device, "GammaEnable", &gerror);
    ArvCheckError(&gerror);
    gammaEnable = std::to_string(ge);
    pProp->Set(gammaEnable.c_str());
  }
  return DEVICE_OK;
}


int AravisCamera::OnPixelFormat(MM::PropertyBase* pProp, MM::ActionType eAct)
{
  GError *gerror = nullptr;

  if (eAct == MM::AfterSet){
    if (!capturing){
      guint32 arvPixelFormat;
      std::string pixelFormat;
      pProp->Get(pixelFormat);

      // Nothing to do if the camera is already in that format -- and more than
      // nothing to avoid: a camera that offers a single format has no reason
      // to make PixelFormat writable, so writing even the value it already
      // holds fails at the register.
      const char *current;
      current = arv_camera_get_pixel_format_as_string(arv_cam, &gerror);
      if (ArvCheckError(&gerror)){
	return ARV_ERROR;
      }
      if ((current != nullptr) && (pixelFormat == current)){
	return DEVICE_OK;
      }

      arv_camera_set_pixel_format_from_string(arv_cam, pixelFormat.c_str(), &gerror);
      if (ArvCheckError(&gerror)){
	return ARV_ERROR;
      }

      // Read back rather than assume: a camera may round the request, or
      // decline it while leaving the property looking as though it took.
      arvPixelFormat = arv_camera_get_pixel_format(arv_cam, &gerror);
      if (ArvCheckError(&gerror)){
	return ARV_ERROR;
      }
      ArvPixelFormatUpdate(arvPixelFormat);
    }
  }
  else if (eAct == MM::BeforeGet) {
    const char *pixelFormat;
    pixelFormat = arv_camera_get_pixel_format_as_string(arv_cam, &gerror);
    if (ArvCheckError(&gerror)){
      return ARV_ERROR;
    }
    if (pixelFormat != nullptr){
      pProp->Set(pixelFormat);
    }
  }

  return DEVICE_OK;
}


// PixelType is read-only and derived: it says what the buffer Micro-Manager
// receives looks like, which is a consequence of the camera's pixel format
// rather than an independent setting. Selecting a format is PixelFormat's job.
//
// Computing it on every read is also what keeps the adapter from ever writing
// it: GetImageBuffer() used to push this value into the property on the image
// path, which sent "Unknown" back through the format setter and asked the
// camera, once per image, to switch to a pixel format by that name.
int AravisCamera::OnPixelType(MM::PropertyBase* pProp, MM::ActionType eAct)
{
  if (eAct == MM::BeforeGet){
    pProp->Set((pixel_type != nullptr) ? pixel_type : g_PixelType_Unknown);
  }

  return DEVICE_OK;
}


int AravisCamera::OnTriggerMode(MM::PropertyBase* pProp, MM::ActionType eAct)
{
  GError *gerror = nullptr;

  if (eAct == MM::AfterSet){
    if (!capturing){
      std::string mode;
      pProp->Get(mode);

      arv_device_set_string_feature_value(arv_device, "TriggerMode", mode.c_str(), &gerror);
      ArvCheckError(&gerror);
    }
  }
  else if (eAct == MM::BeforeGet) {
    const char *mode;
    mode = arv_device_get_string_feature_value(arv_device, "TriggerMode", &gerror);
    ArvCheckError(&gerror);

    pProp->Set(mode);
  }
  
  return DEVICE_OK;
}


int AravisCamera::OnTriggerSelector(MM::PropertyBase* pProp, MM::ActionType eAct)
{
  GError *gerror = nullptr;

  if (eAct == MM::AfterSet){
    if (!capturing){
      std::string trigger;
      pProp->Get(trigger);

      arv_device_set_string_feature_value(arv_device, "TriggerSelector", trigger.c_str(), &gerror);
      //arv_camera_set_trigger(arv_cam, trigger.c_str(), &gerror);
      ArvCheckError(&gerror);
    }
  }
  else if (eAct == MM::BeforeGet) {
    const char *trigger;
    trigger = arv_device_get_string_feature_value(arv_device, "TriggerSelector", &gerror);
    ArvCheckError(&gerror);

    pProp->Set(trigger);
  }
  
  return DEVICE_OK;
}
  

int AravisCamera::OnTriggerSource(MM::PropertyBase* pProp, MM::ActionType eAct)
{
  GError *gerror = nullptr;

  if (eAct == MM::AfterSet){
    if (!capturing){
      std::string triggerSource;
      pProp->Get(triggerSource);

      arv_camera_set_trigger_source(arv_cam, triggerSource.c_str(), &gerror);
      ArvCheckError(&gerror);
    }
  }
  else if (eAct == MM::BeforeGet) {
    const char *triggerSource;
    triggerSource = arv_camera_get_trigger_source(arv_cam, &gerror);
    ArvCheckError(&gerror);

    pProp->Set(triggerSource);
  }
  
  return DEVICE_OK;
}


int AravisCamera::SetBinning(int binSize)
{
  GError *gerror = nullptr;

  // The Binning property offers only "1" on a camera that cannot bin, so the
  // only value that reaches here is the one it already has.
  if (!has_binning){
    return (binSize == 1) ? DEVICE_OK : DEVICE_UNSUPPORTED_COMMAND;
  }

  arv_camera_set_binning(arv_cam, (gint)binSize, (gint)binSize, &gerror);
  if (ArvCheckError(&gerror)){
    return ARV_ERROR;
  }

  // Binning changes the frame size, and Micro-Manager reads the new one back
  // before the next frame arrives.
  ArvGeometryUpdate();

  return DEVICE_OK;
}


void AravisCamera::SetExposure(double expMs)
{
  double expUs = 1000.0*expMs;
  double min, max;
  GError *gerror = nullptr;

  // Nothing to set, and nothing that could be set: a camera with no
  // ExposureTime feature failed both calls below on every exposure change.
  if (!has_exposure_time){
    return;
  }

  arv_camera_get_exposure_time_bounds(arv_cam, &min, &max, &gerror);
  ArvCheckError(&gerror);

  if (expUs < min){ expUs = min; }
  if (expUs > max){ expUs = max; }

  arv_camera_set_exposure_time(arv_cam, expUs, &gerror);
  ArvCheckError(&gerror);

  // Disable the frame rate limit so the exposure is what paces the camera --
  // but only on a camera that has one to disable. The bounds this used to read
  // first were never looked at; the comment even said they do not change.
  if (has_frame_rate){
    arv_camera_set_frame_rate(arv_cam, -1.0, &gerror);
    ArvCheckError(&gerror);
  }

  ArvGetExposure();
}


int AravisCamera::SetROI(unsigned x, unsigned y, unsigned xSize, unsigned ySize)
{
  gint inc, ix, iy, ixs, iys;
  GError *gerror = nullptr;

  // A camera that has no OffsetX fails the increment query, and Aravis has no
  // increment to return; dividing by what comes back would be a division by
  // zero. One is the identity here, so it is also the safe answer.
  inc = arv_camera_get_x_offset_increment(arv_cam, &gerror);
  ArvCheckError(&gerror);
  ix = ((gint)x/ArvIncrement(inc))*ArvIncrement(inc);

  inc = arv_camera_get_y_offset_increment(arv_cam, &gerror);
  ArvCheckError(&gerror);
  iy = ((gint)y/ArvIncrement(inc))*ArvIncrement(inc);

  inc = arv_camera_get_width_increment(arv_cam, &gerror);
  ArvCheckError(&gerror);
  ixs = ((gint)xSize/ArvIncrement(inc))*ArvIncrement(inc);

  inc = arv_camera_get_height_increment(arv_cam, &gerror);
  ArvCheckError(&gerror);
  iys = ((gint)ySize/ArvIncrement(inc))*ArvIncrement(inc);

  arv_camera_set_region(arv_cam, ix, iy, ixs, iys, &gerror);
  int ret = ArvCheckError(&gerror) ? ARV_ERROR : DEVICE_OK;

  // Either way: Micro-Manager reads the new dimensions straight away, and if
  // the camera rounded the request or refused it outright, what it actually
  // took is what the rest of the application has to work from.
  ArvGeometryUpdate();

  return ret;
}


// Release everything acquired since Initialize(). Idempotent: Micro-Manager
// may call this more than once, and the destructor calls it again.
//
// This used to do nothing at all, which left a running acquisition streaming
// into a callback whose target was about to be freed, and leaked the stream,
// the snap buffer and the image buffer on every load/unload of a configuration.
int AravisCamera::Shutdown()
{
  StopSequenceAcquisition();

  g_clear_object(&arv_stream);
  g_clear_object(&arv_buffer);

  {
    std::lock_guard<std::mutex> lock(img_buffer_mutex);
    if (img_buffer != nullptr){
      free(img_buffer);
      img_buffer = nullptr;
    }
    img_buffer_size = 0;
    img_buffer_number_pixels = 0;
  }

  // arv_device is owned by arv_cam, which the destructor clears; it must not
  // be unreffed here.
  arv_device = nullptr;
  initialized = false;

  return DEVICE_OK;
}


// This should wait until the image is acquired? Maybe it does?
int AravisCamera::SnapImage()
{
  GError *gerror = nullptr;

  // A zero timeout means "no timeout" to Aravis, which pops the buffer with a
  // blocking call. A camera left in hardware trigger mode, or one frame lost
  // to a dropped packet, would then block this thread forever and hang the
  // application with no way back. Wait generously but finitely: several times
  // the exposure, and never less than ARV_SNAP_MIN_TIMEOUT_US.
  guint64 timeout_us = (guint64)(exposure_time * 1000.0 * ARV_SNAP_EXPOSURE_FACTOR);
  if (timeout_us < ARV_SNAP_MIN_TIMEOUT_US){
    timeout_us = ARV_SNAP_MIN_TIMEOUT_US;
  }

  arv_buffer = arv_camera_acquisition(arv_cam, timeout_us, &gerror);
  if (ArvCheckError(&gerror)) return ARV_ERROR;

  if (arv_buffer == nullptr){
    std::stringstream msg;
    msg << "Aravis Error, no image after " << (timeout_us / 1000) << "ms. "
	<< "If this camera is waiting on a hardware trigger, that trigger did "
	<< "not arrive.";
    LogMessage(msg.str(), false);
    return ARV_ERROR;
  }

  return DEVICE_OK;
}


int AravisCamera::StartSequenceAcquisition(long numImages, double interval_ms, bool stopOnOverflow)
{
  // stopOnOverflow is not kept: the Core acts on it itself, and tells this
  // adapter by failing InsertImage(), which the callback already stops on.
  num_images = numImages;

  if (!ArvStartSequenceAcquisition()){
    int ret = GetCoreCallback()->PrepareForAcq(this);
    if (ret != DEVICE_OK) {
      return ret;
    }
    return DEVICE_OK;
  }
  return ARV_ERROR;
}


int AravisCamera::StartSequenceAcquisition(double interval_ms) {
  // The continuous overload: run until Micro-Manager stops it.
  num_images = -1;

  if (!ArvStartSequenceAcquisition()){
    int ret = GetCoreCallback()->PrepareForAcq(this);
    if (ret != DEVICE_OK) {
      return ret;
    }    
    return DEVICE_OK;
  }
  return ARV_ERROR;
}


int AravisCamera::StopSequenceAcquisition()
{
  GError *gerror = nullptr;

  // The stream is released whether or not this call is the one that ended the
  // acquisition: a finite sequence stops itself from the callback, which
  // cannot touch the stream, so something has to clean up afterwards.
  bool was_capturing = capturing.exchange(false);

  if (was_capturing && (arv_cam != nullptr)){
    arv_camera_stop_acquisition(arv_cam, &gerror);
    ArvCheckError(&gerror);
  }

  // Unreffing the stream stops and joins its thread, so no callback can be
  // in flight once this returns. Shutdown() relies on that before it frees
  // the image buffer the callback writes into.
  g_clear_object(&arv_stream);

  if (was_capturing){
    GetCoreCallback()->AcqFinished(this, 0);
  }
  return DEVICE_OK;
}
