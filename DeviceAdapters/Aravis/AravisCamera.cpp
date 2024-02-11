// #pragma warning(push)
// #pragma warning(disable : 4482)
// #pragma warning(disable : 4251) // Note: need to have a C++ interface, i.e., compiler versions need to match!

#include "AravisCamera.h"
#include "ModuleInterface.h"
#include <vector>
#include <string>
#include <algorithm>


/*
 * Module functions.
 */
MODULE_API void InitializeModuleData()
{
  uint64_t nDevices=0;

  // Update and get number of aravis compatible cameras.
  arv_update_device_list();
  nDevices = arv_get_n_devices();
  printf("AAAAA: Aravis Found %d\n", (int)nDevices);
  
  for (int i = 0; i < nDevices; i++)
  {
    RegisterDevice(arv_get_device_id(i), MM::CameraDevice, "Aravis Camera");
  }
}

MODULE_API MM::Device* CreateDevice(const char* deviceId)
{
  return new AravisCamera(deviceId);
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
  delete pDevice;
}


/*
 * Camera class and methods.
 */

AravisCamera::AravisCamera(const char *name) : CCameraBase<AravisCamera>()
{
  a_cam_name = name;
}

AravisCamera::~AravisCamera()
{
  g_clear_object(&a_cam);
}

// These supposed to be in alphabetical order.

int AravisCamera::ClearROI()
{
  return DEVICE_OK;
}

int AravisCamera::GetBinning() const
{
  gint dx;
  gint dy;
  GError *error = NULL;
  
  arv_camera_get_binning(a_cam, &dx, &dy, &error);
  if (error != NULL) {
    printf ("Aravis Error: %s\n", error->message);
    g_clear_error(&error);
  }
  
  // dx is always dy for MM? Add check?  
  return (int)dx;
}

unsigned AravisCamera::GetBitDepth() const
{
  guint32 arvPixelFormat;
  GError *error = NULL;
  
  arvPixelFormat = arv_camera_get_pixel_format(a_cam, &error);
  if (error != NULL) {
    printf ("Aravis Error: %s\n", error->message);
    g_clear_error(&error);
  }
  
  switch (arvPixelFormat){
  case ARV_PIXEL_FORMAT_MONO_8: {
    return 8;
  }
  case ARV_PIXEL_FORMAT_MONO_10: {
    return 10;
  }
  case ARV_PIXEL_FORMAT_MONO_12: {
    return 12;
  }
  case ARV_PIXEL_FORMAT_MONO_14: {
    return 14;
  }
  case ARV_PIXEL_FORMAT_MONO_16: {
    return 16;
  }
  default:{
    printf ("Aravis Error: Pixel Format %d is not implemented\n", (int)arvPixelFormat);
  }    
  }
  return 0;
}

double AravisCamera::GetExposure() const
{
  GError *error = NULL;
  
  return arv_camera_get_exposure_time(a_cam, &error);
  if (error != NULL) {
    printf ("Aravis Error: %s\n", error->message);
    g_clear_error(&error);
  }
}

const unsigned char* AravisCamera::GetImageBuffer()
{
  int i;
  gint w,h;
  size_t size;
  unsigned char *mm_buffer;
  unsigned char *cam_data;

  printf("get image buffer\n");
  if (ARV_IS_BUFFER (a_buffer)) {
    w = arv_buffer_get_image_width(a_buffer);
    h = arv_buffer_get_image_height(a_buffer);
    cam_data = (unsigned char *)arv_buffer_get_data(a_buffer, &size);
    printf("buffer is %ld, %d x %d\n", (long)size, (int)w, (int)h); 

    for(i=0;i<(w*h);i++){
      mm_buffer[i] = cam_data[i];
    }
    g_clear_object(&a_buffer);
    return mm_buffer;
  }
  return NULL;
}

long AravisCamera::GetImageBufferSize() const
{
  gint gx,gy,gwidth,gheight;
  GError *error = NULL;
  arv_camera_get_region(a_cam, &gx, &gy, &gwidth, &gheight, &error);
  if (error != NULL) {
    printf ("Aravis Error: %s\n", error->message);
    g_clear_error(&error);
  }  
  return (long) gwidth * gheight * this->GetImageBytesPerPixel(); 
}

unsigned AravisCamera::GetImageBytesPerPixel() const
{
  return 1;
}

unsigned AravisCamera::GetImageWidth() const
{
  gint w;
  gint h;
  GError *error = NULL;
  
  arv_camera_get_sensor_size(a_cam, &w, &h, &error);
  if (error != NULL) {
    printf ("Aravis Error: %s\n", error->message);
    g_clear_error(&error);
  }  
  return (unsigned)w;
}

unsigned AravisCamera::GetImageHeight() const
{
  gint w;
  gint h;
  GError *error = NULL;
  
  arv_camera_get_sensor_size(a_cam, &w, &h, &error);
  if (error != NULL) {
    printf ("Aravis Error: %s\n", error->message);
    g_clear_error(&error);
  }
  return (unsigned)h;
}

void AravisCamera::GetName(char *name) const
{
  CDeviceUtils::CopyLimitedString(name, a_cam_name); 
}

unsigned AravisCamera::GetNumberOfComponents() const
{
  // Add support for RGB cameras.
  return 1;
}

int AravisCamera::GetROI(unsigned& x, unsigned& y, unsigned& xSize, unsigned& ySize)
{
  gint gx,gy,gwidth,gheight;
  GError *error = NULL;
  
  arv_camera_get_region(a_cam, &gx, &gy, &gwidth, &gheight, &error);
  if (error != NULL) {
    printf ("Aravis Error: %s\n", error->message);
    g_clear_error(&error);
  }
  x = (unsigned)gx;
  y = (unsigned)gx;
  xSize = (unsigned)xSize;
  ySize = (unsigned)ySize;

  return DEVICE_OK;
}

int AravisCamera::Initialize()
{
  GError *error = NULL;
  a_cam = arv_camera_new(a_cam_name, &error);
  if (error != NULL) {
    printf ("Aravis Error: %s\n", error->message);
    g_clear_error(&error);
    return ARV_ERROR;
  }

  arv_camera_set_exposure_time_auto(a_cam, ARV_AUTO_OFF, &error);
  if (error != NULL) {
    printf ("Aravis Error: %s\n", error->message);
    g_clear_error(&error);
  }
  return DEVICE_OK;
}

int AravisCamera::IsExposureSequenceable(bool &isSequencable) const
{
  isSequencable = false;
  return DEVICE_OK;
}

int AravisCamera::SetBinning(int binSize)
{
  GError *error;
  
  arv_camera_set_binning(a_cam, (gint)binSize, (gint)binSize, &error);
  if (error != NULL) {
    printf ("Aravis Error: %s\n", error->message);
    g_clear_error(&error);
  }  
  return DEVICE_OK;
}

void AravisCamera::SetExposure(double exp)
{
  double expUs = 1.0e6*exp;
  double frameRate = 1.0/exp;
  GError *error;

  // Range checking?
  // Frame rate should be slightly slower than exposure time?
  arv_camera_set_frame_rate(a_cam, frameRate, &error);
  if (error != NULL) {
    printf ("Aravis Error: %s\n", error->message);
    g_clear_error(&error);
  }  
  arv_camera_set_exposure_time(a_cam, expUs, &error);
  if (error != NULL) {
    printf ("Aravis Error: %s\n", error->message);
    g_clear_error(&error);
  }  
}

int AravisCamera::SetROI(unsigned x, unsigned y, unsigned xSize, unsigned ySize)
{
  GError *error;
  arv_camera_set_region(a_cam, (gint)x, (gint)y, (gint)xSize, (gint)ySize, &error);
  if (error != NULL) {
    printf ("Aravis Error: %s\n", error->message);
    g_clear_error(&error);
  }    
  return DEVICE_OK;
}

int AravisCamera::Shutdown()
{
  return DEVICE_OK;
}

int AravisCamera::SnapImage()
{
  GError *error = NULL;
  // arv_camera_set_acquisition_mode(a_cam, ARV_ACQUISITION_MODE_SINGLE_FRAME);
  
  AravisCamera::a_buffer = arv_camera_acquisition(a_cam, 0, &error);

  if (error != NULL) {
    printf ("Aravis Error: %s\n", error->message);
    g_clear_error(&error);
    return ARV_ERROR;
  }

  return DEVICE_OK;
}


/*
 * Acquistion thread class and methods.
 */
AravisAcquisitionThread::AravisAcquisitionThread(AravisCamera * aCam)
{
}

AravisAcquisitionThread::~AravisAcquisitionThread()
{
}

// #pragma warning(pop)
