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

  // Debugging.
  //arv_debug_enable("all:3,device");
   
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


int arvCheckError(GError *gerror){
  printf("check\n");
  if (gerror != NULL) {
    printf("Aravis Error: %s\n", gerror->message);
    g_clear_error(&gerror);
    return 1;
  }
  return 0;
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
  CCameraBase<AravisCamera>(),
  capturing(false),
  counter(0),
  exposure_time(0.0),
  img_buffer_bytes_per_pixel(0),
  img_buffer_height(0),
  img_buffer_size(0),
  img_buffer_width(0),
  initialized(false),
  arv_buffer(nullptr),
  arv_cam(nullptr),
  arv_cam_name(nullptr),
  arv_stream(nullptr),
  img_buffer(nullptr)
{
  printf("ArvCamera %s\n", name);
  arv_cam_name = (char *)malloc(sizeof(char) * strlen(name));
  CDeviceUtils::CopyLimitedString(arv_cam_name, name);
}

AravisCamera::~AravisCamera()
{
  g_clear_object(&arv_cam);
}

// These in alphabetical order.

// This leaks alot of memory. Images should be freed?
void AravisCamera::AcquisitionCallback(ArvStreamCallbackType type, ArvBuffer *cb_arv_buffer)
{
  size_t size;
  unsigned char *cb_arv_buffer_data;
  //  unsigned char *cb_img_buffer;

  Metadata md;

  printf("stream_callback %ld\n", counter);

  if (!capturing){
    return;
  }
      
  switch (type) {
  case ARV_STREAM_CALLBACK_TYPE_BUFFER_DONE:
    
    g_assert(cb_arv_buffer == arv_stream_pop_buffer(arv_stream));
    g_assert(cb_arv_buffer != NULL);

    img_buffer_width = (int)arv_buffer_get_image_width(cb_arv_buffer);
    img_buffer_height = (int)arv_buffer_get_image_height(cb_arv_buffer);
    cb_arv_buffer_data = (unsigned char *)arv_buffer_get_data(cb_arv_buffer, &size);
    //    cb_img_buffer = (unsigned char *)malloc(size);
    //memcpy(cb_img_buffer, cb_arv_buffer_data, size);

    // Image metadata.
    md.put("Camera", "");
    md.put(MM::g_Keyword_Metadata_ROI_X, CDeviceUtils::ConvertToString((long)img_buffer_width));
    md.put(MM::g_Keyword_Metadata_ROI_Y, CDeviceUtils::ConvertToString((long)img_buffer_height));
    md.put(MM::g_Keyword_Metadata_ImageNumber, CDeviceUtils::ConvertToString(counter));
    md.put(MM::g_Keyword_Meatdata_Exposure, exposure_time);
    
    // Copy to intermediate buffer
    int ret = GetCoreCallback()->InsertImage(this,
					     //					     cb_img_buffer,
					     cb_arv_buffer_data,
					     img_buffer_width,
					     img_buffer_height,
					     GetImageBytesPerPixel(),
					     1,
					     md.Serialize().c_str(),
					     FALSE);
    if (ret == DEVICE_BUFFER_OVERFLOW) {
      //if circular buffer overflows, just clear it and keep putting stuff in so live mode can continue
      GetCoreCallback()->ClearImageBuffer(this);
    }
    //free(cb_img_buffer);
    
    arv_stream_push_buffer(arv_stream, cb_arv_buffer);
    counter += 1;
    break;
  }
}

// Call the Aravis library to check exposure time only as needed.
void AravisCamera::ArvGetExposure()
{
  double expTimeUs;
  GError *gerror = NULL;

  expTimeUs = arv_camera_get_exposure_time(arv_cam, &gerror);
  if(!arvCheckError(gerror)){
    exposure_time = expTimeUs * 1.0e-3;
  }
}

int AravisCamera::ClearROI()
{
  printf("ArvClearROI\n");
  return DEVICE_OK;
}

int AravisCamera::GetBinning() const
{
  gint dx;
  gint dy;
  GError *gerror = NULL;

  printf("ArvGetBinning\n");
  arv_camera_get_binning(arv_cam, &dx, &dy, &gerror);
  arvCheckError(gerror);
    
  // dx is always dy for MM? Add check?  
  return (int)dx;
}

unsigned AravisCamera::GetBitDepth() const
{
  guint32 arvPixelFormat;
  GError *gerror = NULL;

  printf("ArvGetBitDepth\n");
  arvPixelFormat = arv_camera_get_pixel_format(arv_cam, &gerror);
  arvCheckError(gerror);
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
  return exposure_time;
}

// This at least needs to allocate mm_buffer..
const unsigned char* AravisCamera::GetImageBuffer()
{
  size_t size;
  unsigned char *arv_buffer_data;

  printf("ArvGetImageBuffer\n");
  if (ARV_IS_BUFFER (arv_buffer)) {
    img_buffer_width = (int)arv_buffer_get_image_width(arv_buffer);
    img_buffer_height = (int)arv_buffer_get_image_height(arv_buffer);
    arv_buffer_data = (unsigned char *)arv_buffer_get_data(arv_buffer, &size);
    printf("buffer is %ld, %d x %d\n", (long)size, (int)img_buffer_width, (int)img_buffer_height);
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
  GError *gerror = NULL;

  printf("ArvGetImageBufferSize\n");
  arv_camera_get_region(arv_cam, &gx, &gy, &gwidth, &gheight, &gerror);
  arvCheckError(gerror);

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
  return (unsigned)img_buffer_width;
}

unsigned AravisCamera::GetImageHeight() const
{
  return (unsigned)img_buffer_height;
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
  GError *gerror = NULL;

  printf("ArvGetROI\n");
  arv_camera_get_region(arv_cam, &gx, &gy, &gwidth, &gheight, &gerror);
  arvCheckError(gerror);

  x = (unsigned)gx;
  y = (unsigned)gx;
  xSize = (unsigned)xSize;
  ySize = (unsigned)ySize;

  return DEVICE_OK;
}

int AravisCamera::Initialize()
{
  int ret;
  gint h,w,tmp;
  GError *gerror = NULL;

  if(initialized){
    return DEVICE_OK;
  }
  printf("ArvInitialize %s\n", arv_cam_name);
  
  arv_cam = arv_camera_new(arv_cam_name, &gerror);
  if (arvCheckError(gerror)) return ARV_ERROR;

  // Turn off auto exposure.
  arv_camera_set_exposure_time_auto(arv_cam, ARV_AUTO_OFF, &gerror);
  arvCheckError(gerror);
  initialized = true;

  // Start at full (accessible) chip size. This doesn't work. IDK.
  arv_camera_get_height_bounds(arv_cam, &tmp, &h, &gerror);
  arvCheckError(gerror);

  arv_camera_get_width_bounds(arv_cam, &tmp, &w, &gerror);
  arvCheckError(gerror);

  printf("  %d %d\n", w, h);
  //SetROI(0, 0, 1616, 1240);

  img_buffer_height = (int)h;
  img_buffer_width = (int)w;
  
  //void arv_camera_get_x_binning_bounds (ArvCamera* camera, gint* min, gint* max, GError** error);
  ret = CreateIntegerProperty(MM::g_Keyword_Binning, 1, true);
  SetPropertyLimits(MM::g_Keyword_Binning, 1, 1);
  assert(ret == DEVICE_OK);
  
  ArvGetExposure();
		
  return DEVICE_OK;
}

// Not sure if these cameras are sequencable or not, going with not.
int AravisCamera::IsExposureSequenceable(bool &isSequencable) const
{
  isSequencable = false;

  printf("ArvIsExposureSequencable\n");
  return DEVICE_OK;
}

bool AravisCamera::IsCapturing()
{
  return capturing;
}

int AravisCamera::PrepareSequenceAcqusition()
{
   return DEVICE_OK;
}

int AravisCamera::SetBinning(int binSize)
{
  GError *gerror;

  printf("ArvSetBinning\n");
  arv_camera_set_binning(arv_cam, (gint)binSize, (gint)binSize, &gerror);
  arvCheckError(gerror);

  return DEVICE_OK;
}

void AravisCamera::SetExposure(double exp)
{
  double expUs = 1000.0*exp;
  double frameRate = 1.0/exp;
  GError *gerror;

  printf("ArvSetExposure\n");
  // Range checking?
  // Frame rate should be slightly slower than exposure time?
  arv_camera_set_frame_rate(arv_cam, frameRate, &gerror);
  arvCheckError(gerror);

  arv_camera_set_exposure_time(arv_cam, expUs, &gerror);
  arvCheckError(gerror);

  ArvGetExposure();
}

int AravisCamera::SetROI(unsigned x, unsigned y, unsigned xSize, unsigned ySize)
{
  GError *gerror;

  printf("ArvSetROI %d %d %d %d\n", x, y, xSize, ySize);
  arv_camera_set_region(arv_cam, (gint)x, (gint)y, (gint)xSize, (gint)ySize, &gerror);
  arvCheckError(gerror);

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
  GError *gerror = NULL;

  printf("SnapImage\n");
  // arv_camera_set_acquisition_mode(a_cam, ARV_ACQUISITION_MODE_SINGLE_FRAME);
  arv_buffer = arv_camera_acquisition(arv_cam, 0, &gerror);
  if (arvCheckError(gerror)) return ARV_ERROR;

  return DEVICE_OK;
}

int AravisCamera::StartSequenceAcquisition(long numImages, double interval_ms, bool stopOnOverflow)
{
  GError  *gerror = NULL;

  printf("StartSequenceAcquisition1 %ld %f %d\n", numImages, interval_ms, stopOnOverflow);
  counter = 0;
  
  arv_camera_set_acquisition_mode(arv_cam, ARV_ACQUISITION_MODE_CONTINUOUS, &gerror);
  if (!arvCheckError(gerror)){
    arv_stream = arv_camera_create_stream(arv_cam, stream_callback, this, &gerror);
    arvCheckError(gerror);
  }
  
  if (ARV_IS_STREAM(arv_stream)){
    int i;
    size_t payload;
    
    payload = arv_camera_get_payload(arv_cam, &gerror);
    if (!arvCheckError(gerror)){
      for (i = 0; i < 20; i++)
	arv_stream_push_buffer(arv_stream, arv_buffer_new(payload, NULL));
    }
    arv_camera_start_acquisition(arv_cam, &gerror);
    arvCheckError(gerror);
  }
  else{
    printf("arv error, stream creation failed.\n");
  }
  capturing = true;
  printf("  started1\n");
  return DEVICE_OK;
}

int AravisCamera::StartSequenceAcquisition(double interval_ms) {
  int i;
  size_t payload;
  GError *gerror = NULL;

  printf("StartSequenceAcquisition2 %f\n", interval_ms);
  counter = 0;
    
  arv_camera_set_acquisition_mode(arv_cam, ARV_ACQUISITION_MODE_CONTINUOUS, &gerror);
  if (!arvCheckError(gerror)){
    arv_stream = arv_camera_create_stream(arv_cam, stream_callback, this, &gerror);
    arvCheckError(gerror);
  }

  if (ARV_IS_STREAM(arv_stream)){
    payload = arv_camera_get_payload(arv_cam, &gerror);
    if (!arvCheckError(gerror)){
      for (i = 0; i < 20; i++)
	arv_stream_push_buffer(arv_stream, arv_buffer_new(payload, NULL));
    }
    arv_camera_start_acquisition(arv_cam, &gerror);
    arvCheckError(gerror);
  }
  else{
    printf("stream creation failed.\n");
  }
  capturing = true;
  printf("  started2\n");
  return DEVICE_OK;
}

int AravisCamera::StopSequenceAcquisition()
{
  GError *gerror = NULL;

  printf("StopSequenceAcquisition\n");
  if (capturing){
    capturing = false;
    arv_camera_stop_acquisition(arv_cam, &gerror);
    arvCheckError(gerror);
    g_clear_object(&arv_stream);
    
    GetCoreCallback()->AcqFinished(this, 0);
  }
  return DEVICE_OK;
}


