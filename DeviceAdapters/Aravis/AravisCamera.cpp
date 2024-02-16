// #pragma warning(push)
// #pragma warning(disable : 4482)
// #pragma warning(disable : 4251) // Note: need to have a C++ interface, i.e., compiler versions need to match!

#include "AravisCamera.h"
#include "ModuleInterface.h"


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

MODULE_API MM::Device* CreateDevice(const char* deviceName)
{
  printf("ArvCreateDevice %s\n", deviceName);
  return new AravisCamera(deviceName);
}

MODULE_API void DeleteDevice(MM::Device* pDevice)
{
  delete pDevice;
}


/*
 * Camera class and methods.
 */

AravisCamera::AravisCamera(const char *name) :
  CCameraBase<AravisCamera>(),
  img_buffer(nullptr),
  img_buffer_size(0),
  initialized_(false)
{
  printf("ArvCamera %s\n", name);
  arv_cam_name = (char *)malloc(sizeof(char) * strlen(name));
  CDeviceUtils::CopyLimitedString(arv_cam_name, name);
}

AravisCamera::~AravisCamera()
{
  g_clear_object(&arv_cam);
}

// These supposed to be in alphabetical order.

int AravisCamera::ClearROI()
{
  printf("ArvClearROI\n");
  return DEVICE_OK;
}

int AravisCamera::GetBinning() const
{
  gint dx;
  gint dy;
  GError *error = NULL;

  printf("ArvGetBinning\n");
  arv_camera_get_binning(arv_cam, &dx, &dy, &error);
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

  printf("ArvGetBitDepth\n");
  arvPixelFormat = arv_camera_get_pixel_format(arv_cam, &error);
  if (error != NULL) {
    printf ("Aravis Error: %s\n", error->message);
    g_clear_error(&error);
  }
  printf("  %d\n", arvPixelFormat);
    
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
  double expTimeUs;
  GError *error = NULL;

  printf("ArvGetExposure\n");
  expTimeUs = arv_camera_get_exposure_time(arv_cam, &error);
  if (error != NULL) {
    printf ("Aravis Error: %s\n", error->message);
    g_clear_error(&error);
  }
  return expTimeUs * 1.0e3;
}

// This at least needs to allocate mm_buffer..
const unsigned char* AravisCamera::GetImageBuffer()
{
  int i;
  gint w,h;
  size_t size;
  unsigned char *arv_buffer_data;

  printf("ArvGetImageBuffer\n");
  if (ARV_IS_BUFFER (arv_buffer)) {
    w = arv_buffer_get_image_width(arv_buffer);
    h = arv_buffer_get_image_height(arv_buffer);
    arv_buffer_data = (unsigned char *)arv_buffer_get_data(arv_buffer, &size);
    printf("buffer is %ld, %d x %d\n", (long)size, (int)w, (int)h);
    printf("buffer size %ld\n", GetImageBufferSize());

    if (img_buffer_size != size){
      if (img_buffer != nullptr){
	free(img_buffer);
      }
      printf("malloc %ld\n", size);
      img_buffer = (unsigned char *)malloc(size);
    }
    memcpy(img_buffer, arv_buffer_data, size);
    return img_buffer;
  }
  return NULL;
}

long AravisCamera::GetImageBufferSize() const
{
  gint gx,gy,gwidth,gheight;
  GError *error = NULL;

  printf("ArvGetImageBufferSize\n");
  arv_camera_get_region(arv_cam, &gx, &gy, &gwidth, &gheight, &error);
  if (error != NULL) {
    printf ("Aravis Error: %s\n", error->message);
    g_clear_error(&error);
  }
  printf("  %d %d\n", gwidth, gheight);
  return (long) gwidth * gheight * GetImageBytesPerPixel(); 
}

unsigned AravisCamera::GetImageBytesPerPixel() const
{
  printf("ArvGetImageBytesPerPixel\n");
  return 1;
}

unsigned AravisCamera::GetImageWidth() const
{
  gint w;
  gint h;
  GError *error = NULL;

  printf("ArvGetImageWidth\n");
  arv_camera_get_sensor_size(arv_cam, &w, &h, &error);
  if (error != NULL) {
    printf ("Aravis Error: %s\n", error->message);
    g_clear_error(&error);
  }
  printf("  %d %d\n", w, h);
  return (unsigned)w;
}

unsigned AravisCamera::GetImageHeight() const
{
  gint w;
  gint h;
  GError *error = NULL;
  
  printf("ArvGetImageHeight\n");
  arv_camera_get_sensor_size(arv_cam, &w, &h, &error);
  if (error != NULL) {
    printf ("Aravis Error: %s\n", error->message);
    g_clear_error(&error);
  }
  return (unsigned)h;
}

void AravisCamera::GetName(char *name) const
{
  printf("ArvGetName\n");
  CDeviceUtils::CopyLimitedString(name, arv_cam_name);
}

unsigned AravisCamera::GetNumberOfComponents() const
{
  printf("ArvGetNumberOfComponents\n");
  // Add support for RGB cameras.
  return 1;
}

int AravisCamera::GetROI(unsigned& x, unsigned& y, unsigned& xSize, unsigned& ySize)
{
  gint gx,gy,gwidth,gheight;
  GError *error = NULL;

  printf("ArvGetROI\n");
  arv_camera_get_region(arv_cam, &gx, &gy, &gwidth, &gheight, &error);
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
  int ret;
  GError *error = NULL;

  if(initialized_){
    return DEVICE_OK;
  }
  printf("ArvInitialize %s\n", arv_cam_name);
  
  arv_cam = arv_camera_new(arv_cam_name, &error);
  if (error != NULL) {
    printf ("Aravis Error: %s\n", error->message);
    g_clear_error(&error);
    return ARV_ERROR;
  }

  // Turn off auto exposure.
  arv_camera_set_exposure_time_auto(arv_cam, ARV_AUTO_OFF, &error);
  if (error != NULL) {
    printf ("Aravis Error: %s\n", error->message);
    g_clear_error(&error);
  }
  initialized_ = true;

  //void arv_camera_get_x_binning_bounds (ArvCamera* camera, gint* min, gint* max, GError** error);
  ret = CreateIntegerProperty(MM::g_Keyword_Binning, 1, true);
  SetPropertyLimits(MM::g_Keyword_Binning, 1, 1);
  assert(ret == DEVICE_OK);
		
  return DEVICE_OK;
}

int AravisCamera::IsExposureSequenceable(bool &isSequencable) const
{
  isSequencable = false;

  printf("ArvIsExposureSequencable\n");
  return DEVICE_OK;
}

int AravisCamera::SetBinning(int binSize)
{
  GError *error;

  printf("ArvSetBinning\n");
  arv_camera_set_binning(arv_cam, (gint)binSize, (gint)binSize, &error);
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

  printf("ArvSetExposure\n");
  // Range checking?
  // Frame rate should be slightly slower than exposure time?
  arv_camera_set_frame_rate(arv_cam, frameRate, &error);
  if (error != NULL) {
    printf ("Aravis Error: %s\n", error->message);
    g_clear_error(&error);
  }  
  arv_camera_set_exposure_time(arv_cam, expUs, &error);
  if (error != NULL) {
    printf ("Aravis Error: %s\n", error->message);
    g_clear_error(&error);
  }  
}

int AravisCamera::SetROI(unsigned x, unsigned y, unsigned xSize, unsigned ySize)
{
  GError *error;

  printf("ArvSetROI\n");
  arv_camera_set_region(arv_cam, (gint)x, (gint)y, (gint)xSize, (gint)ySize, &error);
  if (error != NULL) {
    printf ("Aravis Error: %s\n", error->message);
    g_clear_error(&error);
  }    
  return DEVICE_OK;
}

int AravisCamera::Shutdown()
{
  printf("Shutdown\n");
  
  return DEVICE_OK;
}

// This should wait until the image is acquired?
int AravisCamera::SnapImage()
{
  GError *error = NULL;

  printf("SnapImage\n");
  // arv_camera_set_acquisition_mode(a_cam, ARV_ACQUISITION_MODE_SINGLE_FRAME);
  arv_buffer = arv_camera_acquisition(arv_cam, 0, &error);

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
