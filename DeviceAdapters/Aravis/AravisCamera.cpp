// #pragma warning(push)
// #pragma warning(disable : 4482)
// #pragma warning(disable : 4251) // Note: need to have a C++ interface, i.e., compiler versions need to match!

#include "AravisCamera.h"
#include "ModuleInterface.h"

#include <vector>
#include <string>
#include <algorithm>


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
  //arv_debug_enable("all:1,device");

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


int arvCheckError(GError *gerror){
  if (gerror != NULL) {
    printf("Aravis Error: %s\n", gerror->message);
    g_clear_error(&gerror);
    return 1;
  }
  return 0;
}


// RGB unpacker.
void rgb_to_rgba(unsigned char *dest, unsigned char *source, size_t size)
{
  size_t i;
  size_t dOffset = 0;
  size_t sOffset = 0;
  
  for (i = 0; i < size; i++){
    memcpy(dest + dOffset, source + sOffset, 3);
    //dest[dOffset + 3] = 0;
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
  CCameraBase<AravisCamera>(),
  capturing(false),
  counter(0),
  exposure_time(0.0),
  img_buffer_bit_depth(0),
  img_buffer_bytes_per_pixel(0),
  img_buffer_height(0),
  img_buffer_number_components(0),
  img_buffer_number_pixels(0),
  img_buffer_size(0),
  img_buffer_width(0),
  initialized(false),  
  arv_buffer(nullptr),
  arv_cam(nullptr),
  arv_cam_name(nullptr),
  arv_stream(nullptr),
  img_buffer(nullptr),
  pixel_type(nullptr)
{
  arv_cam_name = (char *)malloc(sizeof(char) * strlen(name));
  CDeviceUtils::CopyLimitedString(arv_cam_name, name);
}


AravisCamera::~AravisCamera()
{
  g_clear_object(&arv_cam);
}


// These are in alphabetical order.
void AravisCamera::AcquisitionCallback(ArvStreamCallbackType type, ArvBuffer *cb_arv_buffer)
{
  size_t size;
  unsigned char *cb_arv_buffer_data;

  Metadata md;

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

    // Copy buffer data.
    g_assert(cb_arv_buffer == arv_stream_pop_buffer(arv_stream));
    g_assert(cb_arv_buffer != NULL);
    ArvBufferUpdate(cb_arv_buffer);

    // Image metadata.
    md.put("Camera", "");
    md.put(MM::g_Keyword_Metadata_ROI_X, CDeviceUtils::ConvertToString((long)img_buffer_width));
    md.put(MM::g_Keyword_Metadata_ROI_Y, CDeviceUtils::ConvertToString((long)img_buffer_height));
    md.put(MM::g_Keyword_Metadata_ImageNumber, CDeviceUtils::ConvertToString(counter));
    md.put(MM::g_Keyword_Meatdata_Exposure, exposure_time);
    md.put(MM::g_Keyword_PixelType, pixel_type);
    
    // Pass data to MM.
    int ret = GetCoreCallback()->InsertImage(this,
					     img_buffer,
					     img_buffer_width,
					     img_buffer_height,
					     img_buffer_bytes_per_pixel,
					     1,
					     md.Serialize().c_str(),
					     FALSE);
    if (ret == DEVICE_BUFFER_OVERFLOW) {
      GetCoreCallback()->ClearImageBuffer(this);
    }
    
    arv_stream_push_buffer(arv_stream, cb_arv_buffer);
    counter += 1;
    break;
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
    printf("Error, Aravis buffer status is %d\n", status);
    return;
  }

  // Pixel format updates.
  arvPixelFormat = arv_buffer_get_image_pixel_format(aBuffer);
  ArvPixelFormatUpdate(arvPixelFormat);

  // Image size updates.
  img_buffer_width = (int)arv_buffer_get_image_width(aBuffer);
  img_buffer_height = (int)arv_buffer_get_image_height(aBuffer);
  img_buffer_number_pixels = img_buffer_width * img_buffer_height;

  // Copy buffer to MM.
  arvBufferData = (unsigned char *)arv_buffer_get_data(aBuffer, &arvSize);
  size = img_buffer_width * img_buffer_height * img_buffer_bytes_per_pixel;

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
    rgb_to_rgba(img_buffer, arvBufferData, img_buffer_number_pixels);
  }  
}


// Call the Aravis library to check exposure time only as needed.
void AravisCamera::ArvGetExposure()
{
  double expTimeUs;
  GError *gerror = nullptr;

  expTimeUs = arv_camera_get_exposure_time(arv_cam, &gerror);
  if(!arvCheckError(gerror)){
    exposure_time = expTimeUs * 1.0e-3;
  }
}


// Update MM image values based on pixel format.
void AravisCamera::ArvPixelFormatUpdate(guint32 arvPixelFormat)
{
  switch (arvPixelFormat){
  case ARV_PIXEL_FORMAT_MONO_8:
    img_buffer_bit_depth = 8;
    img_buffer_bytes_per_pixel = 1;
    img_buffer_number_components = 1;
    pixel_type = "8bit mono";
    break;
  case ARV_PIXEL_FORMAT_MONO_10:
    img_buffer_bit_depth = 10;
    img_buffer_bytes_per_pixel = 2;
    img_buffer_number_components = 1;
    pixel_type = "10bit mono";
    break;
  case ARV_PIXEL_FORMAT_MONO_12:
    img_buffer_bit_depth = 10;
    img_buffer_bytes_per_pixel = 2;
    img_buffer_number_components = 1;
    pixel_type = "12bit mono";
    break;
  case ARV_PIXEL_FORMAT_MONO_14:
    img_buffer_bit_depth = 14;
    img_buffer_bytes_per_pixel = 2;
    img_buffer_number_components = 1;
    pixel_type = "14bit mono";
    break;
  case ARV_PIXEL_FORMAT_MONO_16:
    img_buffer_bit_depth = 16;
    img_buffer_bytes_per_pixel = 2;
    img_buffer_number_components = 1;
    pixel_type = "16bit mono";
    break;
    
  case ARV_PIXEL_FORMAT_BAYER_RG_8:
    img_buffer_bit_depth = 8;
    img_buffer_bytes_per_pixel = 1;
    img_buffer_number_components = 1;
    pixel_type = "8bit mono";
    break;
  case ARV_PIXEL_FORMAT_BAYER_RG_10:
    img_buffer_bit_depth = 10;
    img_buffer_bytes_per_pixel = 2;
    img_buffer_number_components = 1;
    pixel_type = "10bit mono";
    break;
  case ARV_PIXEL_FORMAT_BAYER_RG_12:
    img_buffer_bit_depth = 12;
    img_buffer_bytes_per_pixel = 2;
    img_buffer_number_components = 1;
    pixel_type = "12bit mono";
    break;
  case ARV_PIXEL_FORMAT_BAYER_RG_16:
    img_buffer_bit_depth = 16;
    img_buffer_bytes_per_pixel = 2;
    img_buffer_number_components = 1;
    pixel_type = "16bit mono";
    break;
	
  case ARV_PIXEL_FORMAT_RGB_8_PACKED:
    img_buffer_bit_depth = 8;
    img_buffer_bytes_per_pixel = 4;
    img_buffer_number_components = 4;
    pixel_type = "8bitRGB";
    break;
  case ARV_PIXEL_FORMAT_BGR_8_PACKED:
    img_buffer_bit_depth = 8;
    img_buffer_bytes_per_pixel = 4;
    img_buffer_number_components = 4;
    pixel_type = "8bitBGR";
    break;

  default:
    printf ("Aravis Error: Pixel Format %d is not implemented\n", (int)arvPixelFormat);
    break;
  }
}


int AravisCamera::ArvStartSequenceAcquisition()
{
  int i;
  size_t payload;
  GError *gerror = nullptr;

  counter = 0;
    
  arv_camera_set_acquisition_mode(arv_cam, ARV_ACQUISITION_MODE_CONTINUOUS, &gerror);
  if (!arvCheckError(gerror)){
    arv_stream = arv_camera_create_stream(arv_cam, stream_callback, this, &gerror);
    if (arvCheckError(gerror)){
      return 1;
    }
  }
  else{
    return 1;
  }
  
  if (ARV_IS_STREAM(arv_stream)){
    payload = arv_camera_get_payload(arv_cam, &gerror);
    if (!arvCheckError(gerror)){
      for (i = 0; i < 20; i++)
	arv_stream_push_buffer(arv_stream, arv_buffer_new(payload, NULL));
    }
    arv_camera_start_acquisition(arv_cam, &gerror);
    if (arvCheckError(gerror)){
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
  
  arv_camera_set_region(arv_cam, 0, 0, 64, 64, &gerror);
  arvCheckError(gerror);
      
  arv_camera_get_height_bounds(arv_cam, &tmp, &h, &gerror);  
  arvCheckError(gerror);

  arv_camera_get_width_bounds(arv_cam, &tmp, &w, &gerror);
  arvCheckError(gerror);

  arv_camera_set_region(arv_cam, 0, 0, w, h, &gerror);
  arvCheckError(gerror);
    
  return DEVICE_OK;
}


int AravisCamera::GetBinning() const
{
  gint dx;
  gint dy;
  GError *gerror = nullptr;

  arv_camera_get_binning(arv_cam, &dx, &dy, &gerror);
  arvCheckError(gerror);
    
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
  int status;
  size_t arv_size, size;
  gboolean chunks;
  unsigned char *arv_buffer_data;

  if (ARV_IS_BUFFER (arv_buffer)) {
    ArvBufferUpdate(arv_buffer);
    g_clear_object(&arv_buffer);
    SetProperty(MM::g_Keyword_PixelType, pixel_type);
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
  CDeviceUtils::CopyLimitedString(name, arv_cam_name);
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
  arvCheckError(gerror);

  x = (unsigned)gx;
  y = (unsigned)gx;
  xSize = (unsigned)xSize;
  ySize = (unsigned)ySize;

  return DEVICE_OK;
}


int AravisCamera::Initialize()
{
  int i,ret;
  gint tmp;
  GError *gerror = nullptr;

  if(initialized){
    return DEVICE_OK;
  }
  
  arv_cam = arv_camera_new(arv_cam_name, &gerror);
  if (arvCheckError(gerror)) return ARV_ERROR;
  
  // Clear ROI settings that may still be present from a previous session.
  ClearROI();

  // Get starting image size.
  gint h,w;
  arv_camera_get_height_bounds(arv_cam, &tmp, &h, &gerror);  
  arvCheckError(gerror);

  arv_camera_get_width_bounds(arv_cam, &tmp, &w, &gerror);
  arvCheckError(gerror);

  img_buffer_height = (int)h;
  img_buffer_width = (int)w;

  // Set image properties based on current pixel type.
  guint32 arvPixelFormat;
  arvPixelFormat = arv_camera_get_pixel_format(arv_cam, &gerror);
  arvCheckError(gerror);
  ArvPixelFormatUpdate(arvPixelFormat);

  // Turn off auto exposure.
  arv_camera_set_exposure_time_auto(arv_cam, ARV_AUTO_OFF, &gerror);
  arvCheckError(gerror);

  // Get current exposure time.
  ArvGetExposure();
  
  // Pixel formats.
  // FIXME: Camera might start with a format that is not supported.
  const char *pixel_format;
  pixel_format = arv_camera_get_pixel_format_as_string (arv_cam, &gerror);
  arvCheckError(gerror);
  
  CPropertyAction* pAct = new CPropertyAction(this, &AravisCamera::OnPixelType);
  ret = CreateProperty(MM::g_Keyword_PixelType, pixel_format, MM::String, false, pAct);
  assert(ret == DEVICE_OK);

  guint nPixelFormats;
  std::vector<std::string> pixelTypeValues;
  const char **pixelFormats;
      
  pixelFormats = arv_camera_dup_available_pixel_formats_as_strings(arv_cam, &nPixelFormats, &gerror);
  arvCheckError(gerror);
  for(i=0;i<nPixelFormats;i++){
    if (std::find(supportedPixelFormats.begin(), supportedPixelFormats.end(), pixelFormats[i]) != supportedPixelFormats.end()){
      pixelTypeValues.push_back(pixelFormats[i]);
    }
  }
  g_free(pixelFormats);
  SetAllowedValues(MM::g_Keyword_PixelType, pixelTypeValues);
  
  // Binning.
  pAct = new CPropertyAction(this, &AravisCamera::OnBinning);
  ret = CreateProperty(MM::g_Keyword_Binning, "1", MM::Integer, false, pAct);    
  SetPropertyLimits(MM::g_Keyword_Binning, 1, 1);
  assert(ret == DEVICE_OK);
    
  gboolean hasBinning;
  hasBinning = arv_camera_is_binning_available(arv_cam, &gerror);
  arvCheckError(gerror);
  if (hasBinning){
    gint bmin,bmax,binc;

    //Assuming X/Y symmetric..
    arv_camera_get_x_binning_bounds(arv_cam, &bmin, &bmax, &gerror);
    arvCheckError(gerror);

    binc = arv_camera_get_x_binning_increment(arv_cam, &gerror);
    arvCheckError(gerror);

    SetPropertyLimits(MM::g_Keyword_Binning, bmin, bmax);

    for (int x = bmin; x <= bmax; x += binc){
      std::string xs = std::to_string(x);
      AddAllowedValue(MM::g_Keyword_Binning, xs.c_str());
    }
  }
  
  // Auto gain.
  gboolean hasAutoGain;
  hasAutoGain = arv_camera_is_gain_auto_available(arv_cam, &gerror);
  arvCheckError(gerror);

  if (hasAutoGain){
    pAct = new CPropertyAction(this, &AravisCamera::OnAutoGain);
    ret = CreateProperty("GainAuto", "NA", MM::String, false, pAct);
    std::vector<std::string> autoGainValues = {"AUTO_OFF", "AUTO_ONCE", "AUTO_CONTINUOUS"};
    SetAllowedValues("GainAuto", autoGainValues);
  }

  // Gain.
  gboolean hasGain;
  hasGain = arv_camera_is_gain_available(arv_cam, &gerror);
  arvCheckError(gerror);  

  if (hasGain){
    double gmin,gmax;

    arv_camera_get_gain_bounds(arv_cam, &gmin, &gmax, &gerror);
    arvCheckError(gerror);
    
    pAct = new CPropertyAction(this, &AravisCamera::OnGain);
    ret = CreateProperty(MM::g_Keyword_Gain, "1.0", MM::Float, false, pAct);
    SetPropertyLimits(MM::g_Keyword_Gain, gmin, gmax);
  }
  
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
      arvCheckError(gerror);
    }
  }
  else if (eAct == MM::BeforeGet) {
    int mode;
    mode = arv_camera_get_gain_auto(arv_cam, &gerror);
    arvCheckError(gerror);

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

  if (eAct == MM::AfterSet){
    if (!capturing){
      pProp->Get(binning);
      bx = std::stoi(binning);
      
      arv_camera_set_binning(arv_cam, bx, bx, &gerror);
      arvCheckError(gerror);
      
      // This restores the image size when we decrease the binning.
      ClearROI();
    }    
  }
  else if (eAct == MM::BeforeGet) {
    arv_camera_get_binning(arv_cam, &bx, &by, &gerror);
    arvCheckError(gerror);

    std::string bxs = std::to_string(bx);
    pProp->Set(bxs.c_str());
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
    arvCheckError(gerror);

    if (mode == ARV_AUTO_OFF){
      pProp->Get(gain);	  
      arv_camera_set_gain(arv_cam, gain, &gerror);
      arvCheckError(gerror);
    }
  }
  else if (eAct == MM::BeforeGet) {
    gain = arv_camera_get_gain(arv_cam, &gerror);
    arvCheckError(gerror);

    pProp->Set(gain);
  }
  return DEVICE_OK;
}


int AravisCamera::OnPixelType(MM::PropertyBase* pProp, MM::ActionType eAct)
{
  GError *gerror = nullptr;

  if (eAct == MM::AfterSet){
    if (!capturing){
      guint32 arvPixelFormat;
      std::string pixelType;
      pProp->Get(pixelType);
      
      arv_camera_set_pixel_format_from_string(arv_cam, pixelType.c_str(), &gerror);
      arvCheckError(gerror);
      
      arvPixelFormat = arv_camera_get_pixel_format(arv_cam, &gerror);
      arvCheckError(gerror);
      ArvPixelFormatUpdate(arvPixelFormat);
    }
  }
  else if (eAct == MM::BeforeGet) {
    const char *pixelFormat;
    pixelFormat = arv_camera_get_pixel_format_as_string(arv_cam, &gerror);
    arvCheckError(gerror);

    pProp->Set(pixelFormat);
  }
  
  return DEVICE_OK;
}
  

int AravisCamera::PrepareSequenceAcqusition()
{
   return DEVICE_OK;
}


int AravisCamera::SetBinning(int binSize)
{
  GError *gerror = nullptr;

  arv_camera_set_binning(arv_cam, (gint)binSize, (gint)binSize, &gerror);
  arvCheckError(gerror);

  return DEVICE_OK;
}


void AravisCamera::SetExposure(double expMs)
{
  double expUs = 1000.0*expMs;
  double min, max;
  GError *gerror = nullptr;

  arv_camera_get_exposure_time_bounds(arv_cam, &min, &max, &gerror);
  arvCheckError(gerror);

  if (expUs < min){ expUs = min; }
  if (expUs > max){ expUs = max; }
  
  arv_camera_set_exposure_time(arv_cam, expUs, &gerror);
  arvCheckError(gerror);

  // This always returns the same bounds, independent of exposure time..
  arv_camera_get_frame_rate_bounds(arv_cam, &min, &max, &gerror);
  arvCheckError(gerror);

  // arv_camera_set_frame_rate(arv_cam, max, &gerror);
  // This is supposed to disable the frame rate, which will then
  // presumably be set by the exposure time.
  arv_camera_set_frame_rate(arv_cam, -1.0, &gerror);
  arvCheckError(gerror);

  ArvGetExposure();
}


int AravisCamera::SetROI(unsigned x, unsigned y, unsigned xSize, unsigned ySize)
{
  gint inc, ix, iy, ixs, iys;
  GError *gerror = nullptr;

  inc = arv_camera_get_x_offset_increment(arv_cam, &gerror);
  arvCheckError(gerror);
  ix = ((gint)x/inc)*inc;

  inc = arv_camera_get_y_offset_increment(arv_cam, &gerror);
  arvCheckError(gerror);
  iy = ((gint)y/inc)*inc;

  inc = arv_camera_get_width_increment(arv_cam, &gerror);
  arvCheckError(gerror);
  ixs = ((gint)xSize/inc)*inc;

  inc = arv_camera_get_height_increment(arv_cam, &gerror);
  arvCheckError(gerror);
  iys = ((gint)ySize/inc)*inc;
  
  arv_camera_set_region(arv_cam, ix, iy, ixs, iys, &gerror);
  arvCheckError(gerror);

  return DEVICE_OK;
}


int AravisCamera::Shutdown()
{
  return DEVICE_OK;
}


// This should wait until the image is acquired? Maybe it does?
int AravisCamera::SnapImage()
{
  GError *gerror = nullptr;

  arv_buffer = arv_camera_acquisition(arv_cam, 0, &gerror);
  if (arvCheckError(gerror)) return ARV_ERROR;

  return DEVICE_OK;
}


int AravisCamera::StartSequenceAcquisition(long numImages, double interval_ms, bool stopOnOverflow)
{
  if (!ArvStartSequenceAcquisition()){    
    return DEVICE_OK;
  }
  else{
    return ARV_ERROR;
  }
}


int AravisCamera::StartSequenceAcquisition(double interval_ms) {
  if (!ArvStartSequenceAcquisition()){
    return DEVICE_OK;
  }
  else{
    return ARV_ERROR;
  }
}


int AravisCamera::StopSequenceAcquisition()
{
  GError *gerror = nullptr;

  if (capturing){
    capturing = false;
    arv_camera_stop_acquisition(arv_cam, &gerror);
    arvCheckError(gerror);
    g_clear_object(&arv_stream);
    
    GetCoreCallback()->AcqFinished(this, 0);
  }
  return DEVICE_OK;
}
