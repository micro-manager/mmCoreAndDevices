////////////////////////////////////////////////////////////////////////////////////////////////////
// FILE:          QSICameraAdapter.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   QSI camera adapter for use with Micro-Manager
//                                
// COPYRIGHT:     Quantum Scientific Imaging, Inc. 2024
//

#include "QSICameraAdapter.h"
#include "QSIToolkit.h"
#include <algorithm>

using namespace std;

#pragma region [ Local Constants ]

namespace QSIString
{
  // A namespace is being used so there's no chance of name collisions

  const char * const ANTI_BLOOMING = "AntiBlooming";
  const char * const ANTI_BLOOMING_HIGH = "High";
  const char * const ANTI_BLOOMING_NORMAL = "Normal";
  const char * const BINNING_MAX_X = "BinningMaxX";
  const char * const BINNING_MAX_Y = "BinningMaxY";
  const char * const BODY_TEMPERATURE = "BodyTemperature";
  const char * const CONNECT_TO_SERIAL_NUMBER = "ConnectToSerialNumber";
  const char * const CONNECT_TO_SERIAL_NUMBER_ANY = "Any";
  const char * const COOLER_STATE = "CoolerState";
  const char * const COOLER_STATE_ON = "On";
  const char * const COOLER_STATE_OFF = "Off";
  const char * const COOLER_POWER = "CoolerPower";
  const char * const DEVICE_NAME = "QSI Camera";
  const char * const DEVICE_DESCRIPTION = "QSI camera adapter";
  const char * const DRIVER_INFO = "DriverInfo";
  const char * const ELECTRONS_PER_ADU = "ElectronsPerADU";
  const char * const EXPOSURE_MAX = "ExposureDurationMax";
  const char * const EXPOSURE_MIN = "ExposureDurationMin";
  const char * const FAN_MODE = "FanMode";
  const char * const FAN_MODE_OFF = "Off";
  const char * const FAN_MODE_QUIET = "Quiet";
  const char * const FAN_MODE_FULL = "Full";
  const char * const FILTER_WHEEL_POSITION = "FilterWheelPosition";
  const char * const FILTER_WHEEL_POSITIONS = "FilterWheelPositions";
  const char * const FULL_WELL_CAPACITY = "FullWellCapacity";
  const char * const GAIN_MODE = "GainMode";
  const char * const GAIN_MODE_HIGH = "High";
  const char * const GAIN_MODE_LOW = "Low";
  const char * const HAS_FILTER_WHEEL = "HasFilterWheel";
  const char * const HAS_FILTER_WHEEL_NO = "No";
  const char * const HAS_FILTER_WHEEL_YES = "Yes";
  const char * const HAS_SHUTTER = "HasShutter";
  const char * const HAS_SHUTTER_NO = "No";
  const char * const HAS_SHUTTER_YES = "Yes";
  const char * const LED_SETTING = "LEDSetting";
  const char * const LED_SETTING_ENABLED = "Enabled";
  const char * const LED_SETTING_DISABLED = "Disabled";
  const char * const MAX_ADU = "MaxADU";
  const char * const MODEL_NAME = "ModelName";
  const char * const MODEL_NUMBER = "ModelNumber";
  const char * const OPEN_SHUTTER = "OpenShutterDuringExposure";
  const char * const OPEN_SHUTTER_NO = "No";
  const char * const OPEN_SHUTTER_YES = "Yes";
  const char * const PCB_TEMPERATURE = "PCBTemperature";
  const char * const PIXEL_SIZE_X = "PixelSizeX";
  const char * const PIXEL_SIZE_Y = "PixelSizeY";
  const char * const PRE_EXPOSURE_FLUSH = "PreExposureFlush";
  const char * const PRE_EXPOSURE_FLUSH_NONE = "None";
  const char * const PRE_EXPOSURE_FLUSH_MODEST = "Modest";
  const char * const PRE_EXPOSURE_FLUSH_NORMAL = "Normal";
  const char * const PRE_EXPOSURE_FLUSH_AGGRESSIVE = "Aggressive";
  const char * const PRE_EXPOSURE_FLUSH_VERY_AGGRESSIVE = "Very Aggressive";
  const char * const READOUT_MODE_FAST_READOUT = "Fast Readout";
  const char * const READOUT_MODE_HIGH_QUALITY = "High Quality";
  const char * const SERIAL_NUMBER = "SerialNumber";
  const char * const SHUTTER_PRIORITY = "ShutterPriority";
  const char * const SHUTTER_PRIORITY_ELECTRONIC = "Electronic";
  const char * const SHUTTER_PRIORITY_MECHANICAL = "Mechanical";
  const char * const SOUND_SETTING = "SoundSetting";
  const char * const SOUND_SETTING_ENABLED = "Enabled";
  const char * const SOUND_SETTING_DISABLED = "Disabled";
}

#pragma endregion

#pragma region [ Windows DLL Entry Code ]

// Windows DLL entry code
#ifdef WIN32
BOOL APIENTRY DllMain(  HANDLE /*hModule*/, DWORD  ul_reason_for_call, LPVOID /*lpReserved*/ )
{
  switch( ul_reason_for_call )
  {
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
      break;
  }
  return TRUE;
}
#endif

#pragma endregion

#pragma region [ Exported MMDevice API ]

////////////////////////////////////////////////////////////////////////////////////////////////////
// List all supported hardware devices here
//
MODULE_API void InitializeModuleData()
{
  /// Register a device class provided by the device adapter library.
  /**
  * To be called in the device adapter module's implementation of
  * InitializeModuleData().
  *
  * Calling this function indicates that the module provides a device with the
  * given name and type, and provides a user-visible description string.
  */
  RegisterDevice(QSIString::DEVICE_NAME, MM::CameraDevice, QSIString::DEVICE_DESCRIPTION);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
MODULE_API MM::Device * CreateDevice( const char * deviceName )
{
  if( deviceName == 0 )
    return 0;

  // Decide which device class to create based on the deviceName parameter
  if( strcmp( deviceName, QSIString::DEVICE_NAME ) == 0 )
  {
    // Create camera
    return new QSICameraAdapter();
  }

  return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
MODULE_API void DeleteDevice( MM::Device * pDevice )
{
  delete pDevice;
}

#pragma endregion

#pragma region [ QSICameraAdapter - Constructor / Desctructor ]

////////////////////////////////////////////////////////////////////////////////////////////////////
// QSICameraAdapter constructor
//
// Setup default all variables and create device properties required to exist
// before intialization. In this case, no such properties were required. All
// properties will be created in the Initialize() method.
//
// As a general guideline Micro-Manager devices do not access hardware in the
// the constructor. We should do as little as possible in the constructor and
// perform most of the initialization in the Initialize() method.
//
QSICameraAdapter::QSICameraAdapter() :
  m_handle( QSI_HANDLE_INVALID ),
  m_imageBinning( 1 ),
  m_imageMaxX( 1 ),
  m_imageMaxY( 1 ),
  m_imageNumX( 1 ),
  m_imageNumY( 1 ),
  m_imageStartX( 0 ),
  m_imageStartY( 0 ),
  m_initialized( false ),
  m_exposureDuration( 10 ),
  m_exposureDurationMax( 1000 ),
  m_exposureDurationMin( 10 ),
  m_exposureOpenShutter( true ),
  m_pImageBuffer( 0 ),
  m_pixelSizeX( 0 ),
  m_pixelSizeY( 0 ),
  m_status( QSI_OK )
{
  int response;
  
  // Call the base class method to set-up default error codes/messages
  InitializeDefaultErrorMessages();

  // Serial number pre-initialization property
  response = CreateProperty( QSIString::CONNECT_TO_SERIAL_NUMBER, QSIString::CONNECT_TO_SERIAL_NUMBER_ANY, MM::String, false, 0, true );
  assert( response == DEVICE_OK );
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// QSICameraAdapter destructor
//
// If this device used as intended within the Micro-Manager system,
// Shutdown() will be always called before the destructor. But in any case
// we need to make sure that all resources are properly released even if
// Shutdown() was not called.
//
QSICameraAdapter::~QSICameraAdapter()
{
  Shutdown();
}

#pragma endregion

#pragma region [ QSICameraAdapter - MM::Device Methods ]

////////////////////////////////////////////////////////////////////////////////////////////////////
// Obtains device name.
//
void QSICameraAdapter::GetName( char * name ) const
{
  // We just return the name we use for referring to this device adapter.
  CDeviceUtils::CopyLimitedString( name, QSIString::DEVICE_NAME );
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Intializes the hardware.
//
// Typically we access and initialize hardware at this point.
// Device properties are typically created here as well.
//
int QSICameraAdapter::Initialize()
{
  char serialNumber[MM::MaxStrLength];
  int imageBufferSize, response;


  // Don't continue if we've already initialized
  if( m_initialized )
    return DEVICE_OK;


  // Assign meaningful descriptions to error codes
  SetErrorText( QSI_NOTSUPPORTED, "Not supported." );
  SetErrorText( QSI_UNRECOVERABLE, "An unrecoverable internal or device error occurred." );
  SetErrorText( QSI_NOFILTER, "No filterwheel available." );
  SetErrorText( QSI_NOMEMORY, "Out of memory." );
  SetErrorText( QSI_BADROWSIZE, "Invalid row size specified." );
  SetErrorText( QSI_BADCOLSIZE, "Invalid column size specified." );
  SetErrorText( QSI_INVALIDBIN, "Invalid bin factor specified." );
  SetErrorText( QSI_NOASYMBIN, "Cannot asymmetric bin." );
  SetErrorText( QSI_BADEXPOSURE, "Invalid exposure duration specified." );
  SetErrorText( QSI_BADBINSIZE, "Invalid bin factor specified." );
  SetErrorText( QSI_NOEXPOSURE, "No prior exposure taken." );
  SetErrorText( QSI_BADRELAYSTATUS, "An error occurred while getting relay status." );
  SetErrorText( QSI_BADABORTRELAYS, "An error occurred while aborting active relays." );
  SetErrorText( QSI_RELAYERROR, "An error occurred while activating relays." );
  SetErrorText( QSI_INVALIDIMAGEPARAMETER, "One or more image parameters are invalid." );
  SetErrorText( QSI_NOIMAGEAVAILABLE, "There is no image available for download." );
  SetErrorText( QSI_NOTCONNECTED, "The camera has not been connected to." );
  SetErrorText( QSI_INVALIDFILTERNUMBER, "Filter position is invalid." );
  SetErrorText( QSI_RECOVERABLE, "A recoverable internal error occurred." );
  SetErrorText( QSI_CONNECTED, "Camera cannot be connected when making this call." );
  SetErrorText( QSI_INVALIDTEMP, "Invalid temperature specified." );
  SetErrorText( QSI_TRIGGERTIMEOUT	, "Trigger timed out." );
  SetErrorText( QSI_ERROR_NO_CAMERA, "No camera available." );
  SetErrorText( QSI_ERROR_NO_SHUTTER, "No shutter available." );
  SetErrorText( QSI_ERROR_DOWNLOADING, "Camera is busy downloading an image." );
  SetErrorText( QSI_ERROR_INVALID_HANDLE, "The specified handle is not valid." );
  SetErrorText( QSI_ERROR_BAD_ARGUMENT, "One of the arguments specified is not valid." );
  SetErrorText( QSI_ERROR_NULL_POINTER, "A null pointer was passed as an argument." );
  SetErrorText( QSI_ERROR_INTERNAL_ERROR, "An undefined error occurred." );
  SetErrorText( QSI_ERROR_NOT_IMPLEMENTED, "Not yet implemented." );
  SetErrorText( QSI_ERROR_BUFFER_TOO_SMALL, "The specified buffer is too small." );
  SetErrorText( QSI_ERROR_BUFFER_TOO_LARGE, "The specified buffer is too small." );
  SetErrorText( QSI_ERROR_NOT_ENABLED, "The requested mode is not enabled." );
  SetErrorText( QSI_ERROR_CREATE_HANDLE, "C API failed to create handle." );
  SetErrorText( QSI_ERROR_COMMAND_FAILED, "The command failed." );

  // Grab the serial number from the pre-initialization property created in the constructor
  GetProperty( QSIString::CONNECT_TO_SERIAL_NUMBER, serialNumber );

  // Connect to the camera
  QSI_CreateHandle( &m_handle );
  HANDLE_QSI_ERROR( this, m_status );

  if( strcmp( QSIString::CONNECT_TO_SERIAL_NUMBER_ANY, serialNumber ) == 0 )
  {
    m_status = QSI_Connect( m_handle );
    HANDLE_QSI_ERROR( this, m_status );
  }
  else
  {
    m_status = QSI_SetSerialNumber( m_handle, serialNumber, MM::MaxStrLength );
    HANDLE_QSI_ERROR( this, m_status );

    m_status = QSI_Connect( m_handle );
    HANDLE_QSI_ERROR( this, m_status );
  }

  // Get image size info
  m_status = QSI_GetImageSizeX( m_handle, &m_imageMaxX );
  HANDLE_QSI_ERROR( this, m_status );

  m_status = QSI_GetImageSizeY( m_handle, &m_imageMaxY );
  HANDLE_QSI_ERROR( this, m_status );

  m_status = QSI_GetImageStartX( m_handle, &m_imageStartX );
  HANDLE_QSI_ERROR( this, m_status );

  m_status = QSI_GetImageStartY( m_handle, &m_imageStartY );
  HANDLE_QSI_ERROR( this, m_status );

  m_status = QSI_GetImageNumX( m_handle, &m_imageNumX );
  HANDLE_QSI_ERROR( this, m_status );

  m_status = QSI_GetImageNumY( m_handle, &m_imageNumY );
  HANDLE_QSI_ERROR( this, m_status );
    
  // Allocate image buffer and clear it for good measure
  imageBufferSize = m_imageMaxX * m_imageMaxY * QSI_IMAGE_BYTES_PER_PIXEL;

  m_pImageBuffer = static_cast<unsigned short *>( malloc( imageBufferSize ) );

  if( m_pImageBuffer == 0 )
    HANDLE_MM_ERROR( this, DEVICE_OUT_OF_MEMORY, "Out of memory." );

  memset( m_pImageBuffer, 0, imageBufferSize );

  // Setup device properties
  response = AntiBloomingPropertySetup();
  HANDLE_MM_ERROR( this, response, "An error occurred while setting up the AntiBlooming property." );

  response = BinningPropertiesSetup();
  HANDLE_MM_ERROR( this, response, "An error occurred while setting up the Binning property." );
  
  response = BodyTemperaturePropertySetup();
  HANDLE_MM_ERROR( this, response, "An error occurred while setting up the BodyTemperature property." );

  response = CCDTemperaturePropertySetup();
  HANDLE_MM_ERROR( this, response, "An error occurred while setting up the CCDTemperature property." );

  response = CCDTemperatureSetpointPropertySetup();
  HANDLE_MM_ERROR( this, response, "An error occurred while setting up the CCDTemperatureSetpoint property." );

  response = CoolerPowerPropertySetup();
  HANDLE_MM_ERROR( this, response, "An error occurred while setting up the CoolerPower property." );

  response = CoolerStatePropertySetup();
  HANDLE_MM_ERROR( this, response, "An error occurred while setting up the CoolerState property." );
  
  response = DescriptionPropertySetup();
  HANDLE_MM_ERROR( this, response, "An error occurred while setting up the Description property." );

  response = DriverInfoPropertySetup();
  HANDLE_MM_ERROR( this, response, "An error occurred while setting up the DriverInfo property." );

  response = ExposurePropertiesSetup();
  HANDLE_MM_ERROR( this, response, "An error occurred while setting up the Exposure properties." );

  response = FanModePropertySetup();
  HANDLE_MM_ERROR( this, response, "An error occurred while setting up the FanMode property." );
  
  response = FilterWheelPropertiesSetup();
  HANDLE_MM_ERROR( this, response, "An error occurred while setting up the FilterWheel properties." );

  response = FullWellCapacityPropertySetup();
  HANDLE_MM_ERROR( this, response, "An error occurred while setting up the FullWellCapacity property." );

  response = GainPropertiesSetup();
  HANDLE_MM_ERROR( this, response, "An error occurred while setting up the Gain properties." );

  response = LEDEnabledPropertySetup();
  HANDLE_MM_ERROR( this, response, "An error occurred while setting up the LEDEnabled property." );

  response = MaxADUPropertySetup();
  HANDLE_MM_ERROR( this, response, "An error occurred while setting up the MaxADU property." );

  response = ModelNamePropertySetup();
  HANDLE_MM_ERROR( this, response, "An error occurred while setting up the ModelName property." );

  response = ModelNumberPropertySetup();
  HANDLE_MM_ERROR( this, response, "An error occurred while setting up the ModelNumber property." );

  response = OpenShutterPropertySetup();
  HANDLE_MM_ERROR( this, response, "An error occurred while setting up the OpenShutter property." );

  response = PCBTemperaturePropertySetup();
  HANDLE_MM_ERROR( this, response, "An error occurred while setting up the PCBTemperature property." );

  response = PixelSizePropertiesSetup();
  HANDLE_MM_ERROR( this, response, "An error occurred while setting up the PixelSize properties." );

  response = PreExposureFlushPropertySetup();
  HANDLE_MM_ERROR( this, response, "An error occurred while setting up the PreExposureFlush property." );

  response = ReadoutModePropertySetup();
  HANDLE_MM_ERROR( this, response, "An error occurred while setting up the ReadoutMode property." );

  response = SerialNumberPropertySetup();
  HANDLE_MM_ERROR( this, response, "An error occurred while setting up the SerialNumber property." );

  response = ShutterPropertiesSetup();
  HANDLE_MM_ERROR( this, response, "An error occurred while setting up the Shutter properties." );
  
  response = SoundEnabledPropertySetup();
  HANDLE_MM_ERROR( this, response, "An error occurred while setting up the SoundEnabled property." );

  // Synchronize all properties
  response = UpdateStatus();
  HANDLE_MM_ERROR( this, response, "An error occured while synchronizing properties." );

  // Set a flag so we know that we've been properly initialized
  m_initialized = true;

  return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Shuts down (unloads) the device.
//
// Ideally this method will completely unload the device and release all resources.
// Shutdown() may be called multiple times in a row.
//
int QSICameraAdapter::Shutdown()
{
  // Disconnect from camera and release handle
  if( m_handle != QSI_HANDLE_INVALID )
  {
    QSI_Disconnect( m_handle );
    QSI_ReleaseHandle( m_handle );
    m_handle = QSI_HANDLE_INVALID;
  }

  // Free image buffer
  if( m_pImageBuffer != 0 )
  {
    free( m_pImageBuffer );
    m_pImageBuffer = 0;
  }

  m_initialized = false;

  return DEVICE_OK;
}

#pragma endregion

#pragma region [ QSICameraAdapter - MM::Camera Methods ]

////////////////////////////////////////////////////////////////////////////////////////////////////
// Resets the Region of Interest to full frame.
//
int QSICameraAdapter::ClearROI()
{
  m_imageNumX = m_imageMaxX / m_imageBinning;
  m_status = QSI_SetImageNumX( m_handle, m_imageNumX );
  HANDLE_QSI_ERROR( this, m_status );

  m_imageNumY = m_imageMaxY / m_imageBinning;
  m_status = QSI_SetImageNumY( m_handle, m_imageNumY );
  HANDLE_QSI_ERROR( this, m_status );

  m_imageStartX = 0;
  m_status = QSI_SetImageStartX( m_handle, m_imageStartX );
  HANDLE_QSI_ERROR( this, m_status );

  m_imageStartY = 0;
  m_status = QSI_SetImageStartY( m_handle, m_imageStartY );
  HANDLE_QSI_ERROR( this, m_status );

  return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Acquires and inserts an image and it's metadata into MMCore's circular buffer
//
int QSICameraAdapter::InsertImage()
{
  char label[MM::MaxStrLength];
  Metadata metadata;
  const unsigned char * pImageBuffer;
  const char * pSerializedMetadata;

  // Assemble metadata
  this->GetLabel( label );
  
  metadata.PutImageTag(MM::g_Keyword_Metadata_CameraLabel, label );

  pSerializedMetadata =  metadata.Serialize().c_str();

  // Download image
  pImageBuffer = GetImageBuffer();

  // Insert received image into MMCore's circular buffer
  return GetCoreCallback()->InsertImage( this, pImageBuffer, m_imageNumX, m_imageNumY, QSI_IMAGE_BYTES_PER_PIXEL, pSerializedMetadata );
}

////////////////////////////////////////////////////////////////////////////////////////////////////
int QSICameraAdapter::IsExposureSequenceable( bool & seq ) const
{
  seq = false;
  return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Returns the current binning factor.
//
int QSICameraAdapter::GetBinning() const
{
  return m_imageBinning;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Returns the bit depth (dynamic range) of the pixel.
// This does not affect the buffer size, it just gives the client application
// a guideline on how to interpret pixel values.
//
unsigned int QSICameraAdapter::GetBitDepth() const
{
  return QSI_IMAGE_BIT_DEPTH;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Returns the current exposure setting in milliseconds.
//
double QSICameraAdapter::GetExposure() const
{
  return m_exposureDuration;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Returns pixel data.
//
// The calling program will assume the size of the buffer based on the values
// obtained from GetImageBufferSize(), which in turn should be consistent with
// values returned by GetImageWidth(), GetImageHight() and GetImageBytesPerPixel().
// The calling program allso assumes that camera never changes the size of
// the pixel buffer on its own. In other words, the buffer can change only if
// appropriate properties are set (such as binning, pixel type, etc.)
//
const unsigned char * QSICameraAdapter::GetImageBuffer()
{
  int imageSize;

  imageSize = m_imageNumX * m_imageNumY;

  m_status = QSI_ReadImage( m_handle, m_pImageBuffer, imageSize );
  
  if( m_status )
  {
    LogMessage( GetLastQSIError( m_handle ), false );

    // Set all image values to 0x0101 (257) to denote an error occurred
    memset( m_pImageBuffer, 1, m_imageMaxX * m_imageMaxY * QSI_IMAGE_BYTES_PER_PIXEL );
  }

  return reinterpret_cast<unsigned char *>( m_pImageBuffer );
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Returns the size in bytes of the image buffer.
//
long QSICameraAdapter::GetImageBufferSize() const
{
  return m_imageNumX * m_imageNumY * QSI_IMAGE_BYTES_PER_PIXEL;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Returns image buffer pixel depth in bytes.
//
unsigned int QSICameraAdapter::GetImageBytesPerPixel() const
{
  return QSI_IMAGE_BYTES_PER_PIXEL;
} 

////////////////////////////////////////////////////////////////////////////////////////////////////
// Returns image buffer Y-size in pixels.
//
unsigned int QSICameraAdapter::GetImageHeight() const
{
  return m_imageNumY;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Returns image buffer X-size in pixels.
//
unsigned int QSICameraAdapter::GetImageWidth() const
{
  return m_imageNumX;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Returns the actual dimensions of the current ROI.
//
int QSICameraAdapter::GetROI( unsigned int & x, unsigned int & y, unsigned int & xSize, unsigned int & ySize )
{
  x = m_imageStartX;
  y = m_imageStartY;

  xSize = m_imageNumX;
  ySize = m_imageNumY;

  return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Sets binning factor.
//
int QSICameraAdapter::SetBinning( int binF )
{
  return SetProperty( MM::g_Keyword_Binning, CDeviceUtils::ConvertToString( binF ) );
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Sets exposure in milliseconds.
//
void QSICameraAdapter::SetExposure( double exp )
{
  SetProperty( MM::g_Keyword_Exposure, CDeviceUtils::ConvertToString( exp ) );
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Sets the camera Region Of Interest.
// This command will change the dimensions of the image.
// Depending on the hardware capabilities the camera may not be able to configure the
// exact dimensions requested - but should try do as close as possible.
// If the hardware does not have this capability the software should simulate the ROI by
// appropriately cropping each frame.
//
// @param x - top-left corner coordinate
// @param y - top-left corner coordinate
// @param xSize - width
// @param ySize - height
//
int QSICameraAdapter::SetROI( unsigned x, unsigned y, unsigned xSize, unsigned ySize )
{
  if( xSize == 0 && ySize == 0 )
  {
    // According to the sample code, this condition is supposed to clear the ROI
    return ClearROI();
  }
  else
  {
    m_status = QSI_SetImageNumX( m_handle, xSize );
    HANDLE_QSI_ERROR( this, m_status );
    m_imageNumX = xSize;

    m_status = QSI_SetImageNumY( m_handle, ySize );
    HANDLE_QSI_ERROR( this, m_status );
    m_imageNumY = ySize;

    m_status = QSI_SetImageStartX( m_handle, x );
    HANDLE_QSI_ERROR( this, m_status );
    m_imageStartX = x;

    m_status = QSI_SetImageStartY( m_handle, y );
    HANDLE_QSI_ERROR( this, m_status );
    m_imageStartY = y;
  }

  return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Performs exposure.
//
// This function should block during the actual exposure and return immediately afterwards 
// (i.e., before readout).  This behavior is needed for proper synchronization with the shutter.
//
int QSICameraAdapter::SnapImage()
{
  qsi_bool imageReady;

  m_status = QSI_StartExposure( m_handle, m_exposureDuration / 1000.0, m_exposureOpenShutter );
  HANDLE_QSI_ERROR( this, m_status );

  imageReady = false;

  if( m_exposureDuration > 2 )
    CDeviceUtils::SleepMs( static_cast<long>( m_exposureDuration ) - 1 );

  while( !imageReady )
  {
    m_status = QSI_GetImageReady( m_handle, &imageReady );
    HANDLE_QSI_ERROR( this, m_status );
  }

  return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Please implement this yourself and do not rely on the base class implementation
// The Base class implementation is deprecated and will be removed shortly
//
int QSICameraAdapter::StartSequenceAcquisition( double interval )
{
  return base::StartSequenceAcquisition( LONG_MAX, interval, false );            
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Called by BaseSequenceThread during sequence acquisition.
//
int QSICameraAdapter::ThreadRun()
{
  int response;

  response = SnapImage();
  if( response ) return response;

  response = InsertImage();
  if( response ) return response;

  return response;
}

#pragma endregion

#pragma region [ QSICameraAdapter - Property Methods ]

////////////////////////////////////////////////////////////////////////////////////////////////////
int QSICameraAdapter::AntiBloomingPropertyHandler( MM::PropertyBase * pProp, MM::ActionType eAct )
{
  string value;
  qsi_anti_bloom antiBloom;

  if( eAct == MM::AfterSet )
  {
    pProp->Get( value );

    if( value.compare( QSIString::ANTI_BLOOMING_HIGH ) == 0 )
      m_status = QSI_SetAntiBlooming( m_handle, QSI_ANTI_BLOOM_HIGH );
    else
      m_status = QSI_SetAntiBlooming( m_handle, QSI_ANTI_BLOOM_NORMAL );

    HANDLE_QSI_ERROR( this, m_status );
  }
  else if( eAct == MM::BeforeGet )
  {
    m_status = QSI_GetAntiBlooming( m_handle, &antiBloom );
    HANDLE_QSI_ERROR( this, m_status );

    if( antiBloom == QSI_ANTI_BLOOM_HIGH )
      pProp->Set( QSIString::ANTI_BLOOMING_HIGH );
    else
      pProp->Set( QSIString::ANTI_BLOOMING_NORMAL );
  }

  return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
int QSICameraAdapter::AntiBloomingPropertySetup()
{
  qsi_bool canSetAntiBlooming;
  CPropertyAction * propertyAction;
  int response;
  vector<string> values;

  m_status = QSI_GetCanSetAntiBlooming( m_handle, &canSetAntiBlooming );
  HANDLE_QSI_ERROR( this, m_status );

  if( canSetAntiBlooming )
  {
    propertyAction = new CPropertyAction( this, &QSICameraAdapter::AntiBloomingPropertyHandler );
    response = CreateProperty( QSIString::ANTI_BLOOMING, QSIString::ANTI_BLOOMING_NORMAL, MM::String, false, propertyAction );
    assert( response == DEVICE_OK );

    values.push_back( QSIString::ANTI_BLOOMING_NORMAL );
    values.push_back( QSIString::ANTI_BLOOMING_HIGH );

    response = SetAllowedValues( QSIString::ANTI_BLOOMING, values );
    assert( response == DEVICE_OK );
  }

  return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
int QSICameraAdapter::BinningPropertiesSetup()
{
  int maxBinX, maxBinY, maxBin, binX, binY;
  CPropertyAction * propertyAction;
  int response;
  vector<string> values;

  m_status = QSI_GetMaxBinX( m_handle, &maxBinX );
  HANDLE_QSI_ERROR( this, m_status );

  m_status = QSI_GetMaxBinY( m_handle, &maxBinY );
  HANDLE_QSI_ERROR( this, m_status );

  maxBin = std::min<short>( maxBinX, maxBinY );

  m_status = QSI_GetBinX( m_handle, &binX );
  HANDLE_QSI_ERROR( this, m_status );

  m_status = QSI_GetBinY( m_handle, &binY );
  HANDLE_QSI_ERROR( this, m_status );

  // Force symmetric binning
  if( binX != binY )
  {
    m_status = QSI_SetBinY( m_handle, binX );
    HANDLE_QSI_ERROR( this, m_status );
  }

  m_imageBinning = binX;

  propertyAction = new CPropertyAction( this, &QSICameraAdapter::BinningPropertyHandler );
  response = CreateProperty( MM::g_Keyword_Binning, CDeviceUtils::ConvertToString( m_imageBinning ), MM::Integer, false, propertyAction );
  assert( response == DEVICE_OK );

  values.reserve( maxBin );

  for( int i = 0; i < maxBin; i++ )
    values.push_back( CDeviceUtils::ConvertToString( i + 1 ) );

  response = SetAllowedValues( MM::g_Keyword_Binning, values );
  assert( response == DEVICE_OK );

  response = CreateProperty( QSIString::BINNING_MAX_X, CDeviceUtils::ConvertToString( maxBinX ), MM::Integer, true );
  assert( response == DEVICE_OK );

  response = CreateProperty( QSIString::BINNING_MAX_Y, CDeviceUtils::ConvertToString( maxBinY ), MM::Integer, true );
  assert( response == DEVICE_OK );

  return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
int QSICameraAdapter::BinningPropertyHandler( MM::PropertyBase * pProp, MM::ActionType eAct )
{
  long binSize;

  if( eAct == MM::AfterSet )
  {
    pProp->Get( binSize );

    m_imageBinning = (int) binSize;
    
    m_status = QSI_SetBinX( m_handle, (short) binSize );
    HANDLE_QSI_ERROR( this, m_status );

    m_status = QSI_SetBinY( m_handle, (short) binSize );
    HANDLE_QSI_ERROR( this, m_status );

    ClearROI();
  }
  else if( eAct == MM::BeforeGet )
  {
    pProp->Set( (long) m_imageBinning );
  }

  return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
int QSICameraAdapter::BodyTemperaturePropertyHandler( MM::PropertyBase * pProp, MM::ActionType eAct )
{
  double value;

  if( eAct == MM::BeforeGet )
  {
    m_status = QSI_GetHeatSinkTemperature( m_handle, &value );
    HANDLE_QSI_ERROR( this, m_status );;

    pProp->Set( value );
  }

  return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
int QSICameraAdapter::BodyTemperaturePropertySetup()
{
  double value;
  CPropertyAction * propertyAction;
  int response;

  m_status = QSI_GetHeatSinkTemperature( m_handle, &value );
  HANDLE_QSI_ERROR( this, m_status );;

  propertyAction = new CPropertyAction( this, &QSICameraAdapter::BodyTemperaturePropertyHandler );
  response = CreateProperty( QSIString::BODY_TEMPERATURE, CDeviceUtils::ConvertToString( 0.0 ), MM::Float, true, propertyAction );
  assert( response == DEVICE_OK );

  return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
int QSICameraAdapter::CCDTemperaturePropertyHandler( MM::PropertyBase * pProp, MM::ActionType eAct )
{
  double value;

  if( eAct == MM::BeforeGet )
  {
    m_status = QSI_GetCCDTemperature( m_handle, &value );
    HANDLE_QSI_ERROR( this, m_status );;

    pProp->Set( value );
  }

  return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
int QSICameraAdapter::CCDTemperaturePropertySetup()
{
  double value;
  CPropertyAction * propertyAction;
  int response;

  m_status = QSI_GetCCDTemperature( m_handle, &value );
  HANDLE_QSI_ERROR( this, m_status );;

  propertyAction = new CPropertyAction( this, &QSICameraAdapter::CCDTemperaturePropertyHandler );
  response = CreateProperty( MM::g_Keyword_CCDTemperature, CDeviceUtils::ConvertToString( value ), MM::Float, true, propertyAction );
  assert( response == DEVICE_OK );

  return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
int QSICameraAdapter::CCDTemperatureSetpointPropertyHandler( MM::PropertyBase * pProp, MM::ActionType eAct )
{
  double value;

  if( eAct == MM::AfterSet )
  {
    pProp->Get( value );

    m_status = QSI_SetCCDTemperatureSetpoint( m_handle, value );
    HANDLE_QSI_ERROR( this, m_status );;
  }
  else if( eAct == MM::BeforeGet )
  {
    m_status = QSI_GetCCDTemperatureSetpoint( m_handle, &value );
    HANDLE_QSI_ERROR( this, m_status );;

    pProp->Set( value );
  }

  return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
int QSICameraAdapter::CCDTemperatureSetpointPropertySetup()
{
  double value;
  CPropertyAction * propertyAction;
  int response;

  m_status = QSI_GetCCDTemperatureSetpoint( m_handle, &value );
  HANDLE_QSI_ERROR( this, m_status );;

  propertyAction = new CPropertyAction( this, &QSICameraAdapter::CCDTemperatureSetpointPropertyHandler );
  response = CreateProperty( MM::g_Keyword_CCDTemperatureSetPoint, CDeviceUtils::ConvertToString( value ), MM::Float, false, propertyAction );
  assert( response == DEVICE_OK );

  return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
int QSICameraAdapter::CoolerPowerPropertyHandler( MM::PropertyBase * pProp, MM::ActionType eAct )
{
  double value;

  if( eAct == MM::BeforeGet )
  {
    m_status = QSI_GetCoolerPower( m_handle, &value );
    HANDLE_QSI_ERROR( this, m_status );;

    pProp->Set( value );
  }

  return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
int QSICameraAdapter::CoolerPowerPropertySetup()
{
  double value;
  CPropertyAction * propertyAction;
  int response;

  m_status = QSI_GetCoolerPower( m_handle, &value );
  HANDLE_QSI_ERROR( this, m_status );;

  propertyAction = new CPropertyAction( this, &QSICameraAdapter::CoolerPowerPropertyHandler );
  response = CreateProperty( QSIString::COOLER_POWER, CDeviceUtils::ConvertToString( value ), MM::Float, true, propertyAction );
  assert( response == DEVICE_OK );

  return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
int QSICameraAdapter::CoolerStatePropertyHandler( MM::PropertyBase * pProp, MM::ActionType eAct )
{
  std::string value;
  qsi_bool coolerOn;
  const char * newValue;

  if( eAct == MM::AfterSet )
  {
    pProp->Get( value );

    if( value.compare( QSIString::COOLER_STATE_ON ) == 0 )
      coolerOn = 1;
    else
      coolerOn = 0;

    m_status = QSI_SetCoolerOn( m_handle, coolerOn );
    HANDLE_QSI_ERROR( this, m_status );;
  }
  else if( eAct == MM::BeforeGet )
  {
    m_status = QSI_GetCoolerOn( m_handle, &coolerOn );
    HANDLE_QSI_ERROR( this, m_status );;

    if( coolerOn )
      newValue = QSIString::COOLER_STATE_ON;
    else
      newValue = QSIString::COOLER_STATE_OFF;

    pProp->Set( newValue );
  }

  return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
int QSICameraAdapter::CoolerStatePropertySetup()
{
  qsi_bool value;
  const char * valueString;
  CPropertyAction * propertyAction;
  int response;
  vector<string> values;

  m_status = QSI_GetCoolerOn( m_handle, &value );
  HANDLE_QSI_ERROR( this, m_status );;

  if( value )
    valueString = QSIString::COOLER_STATE_ON;
  else
    valueString = QSIString::COOLER_STATE_OFF;

  propertyAction = new CPropertyAction( this, &QSICameraAdapter::CoolerStatePropertyHandler );
  response = CreateProperty( QSIString::COOLER_STATE, valueString, MM::String, false, propertyAction );
  assert( response == DEVICE_OK );

  values.push_back( QSIString::COOLER_STATE_OFF );
  values.push_back( QSIString::COOLER_STATE_ON );

  response = SetAllowedValues( QSIString::COOLER_STATE, values );
  assert( response == DEVICE_OK );

  return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
int QSICameraAdapter::DescriptionPropertySetup()
{
  char value[QSI_LENGTH_DESCRIPTION];
  int response;

  m_status = QSI_GetDescription( m_handle, value, QSI_LENGTH_DESCRIPTION );
  HANDLE_QSI_ERROR( this, m_status );

  response = CreateProperty( MM::g_Keyword_Description, value, MM::String, true );
  assert( response == DEVICE_OK );

  return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
int QSICameraAdapter::DriverInfoPropertySetup()
{
  char value[QSI_LENGTH_DRIVER_INFO];
  int response;

  m_status = QSI_GetDriverInfo( m_handle, value, QSI_LENGTH_DRIVER_INFO );
  HANDLE_QSI_ERROR( this, m_status );

  response = CreateProperty( QSIString::DRIVER_INFO, value, MM::String, true );
  assert( response == DEVICE_OK );

  return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
int QSICameraAdapter::ExposurePropertyHandler( MM::PropertyBase * pProp, MM::ActionType eAct )
{
  double value;

  if( eAct == MM::AfterSet )
  {
    pProp->Get( value );
    m_exposureDuration = value;
  }
  else if( eAct == MM::BeforeGet )
  {
    pProp->Set( m_exposureDuration );
  }

  return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
int QSICameraAdapter::ExposurePropertiesSetup()
{
  CPropertyAction * propertyAction;
  int response;

  m_status = QSI_GetMaxExposureTime( m_handle, &m_exposureDurationMax );
  HANDLE_QSI_ERROR( this, m_status );
  m_exposureDurationMax *= 1000; // Convert to milliseconds

  m_status = QSI_GetMinExposureTime( m_handle, &m_exposureDurationMin );
  HANDLE_QSI_ERROR( this, m_status );
  m_exposureDurationMin *= 1000; // Convert to milliseconds

  if( m_exposureDuration < m_exposureDurationMin )
    m_exposureDuration = m_exposureDurationMin;
  else if( m_exposureDuration > m_exposureDurationMax )
    m_exposureDuration = m_exposureDurationMax;

  response = CreateProperty( QSIString::EXPOSURE_MAX, CDeviceUtils::ConvertToString( m_exposureDurationMax ), MM::String, true );
  assert( response == DEVICE_OK );

  response = CreateProperty( QSIString::EXPOSURE_MIN, CDeviceUtils::ConvertToString( m_exposureDurationMin ), MM::String, true );
  assert( response == DEVICE_OK );

  propertyAction = new CPropertyAction( this, &QSICameraAdapter::ExposurePropertyHandler );
  response = CreateProperty( MM::g_Keyword_Exposure, CDeviceUtils::ConvertToString( m_exposureDuration ), MM::Float, false, propertyAction );
  assert( response == DEVICE_OK );

  response = SetPropertyLimits( MM::g_Keyword_Exposure, m_exposureDurationMin, m_exposureDurationMax );
  assert( response == DEVICE_OK );

  return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
int QSICameraAdapter::FanModePropertyHandler( MM::PropertyBase * pProp, MM::ActionType eAct )
{
  string value;
  qsi_fan_mode fanMode;

  if( eAct == MM::AfterSet )
  {
    pProp->Get( value );

    if( value.compare( QSIString::FAN_MODE_OFF ) == 0 )
      m_status = QSI_SetFanMode( m_handle, QSI_FAN_OFF );
    else if( value.compare( QSIString::FAN_MODE_QUIET ) == 0 )
      m_status = QSI_SetFanMode( m_handle, QSI_FAN_QUIET );
    else
      m_status = QSI_SetFanMode( m_handle, QSI_FAN_FULL );

    HANDLE_QSI_ERROR( this, m_status );
  }
  else if( eAct == MM::BeforeGet )
  {
    m_status = QSI_GetFanMode( m_handle, &fanMode );
    HANDLE_QSI_ERROR( this, m_status );

    if( fanMode == QSI_FAN_OFF )
      pProp->Set( QSIString::FAN_MODE_OFF );
    else if( fanMode == QSI_FAN_QUIET )
      pProp->Set( QSIString::FAN_MODE_QUIET );
    else
      pProp->Set( QSIString::FAN_MODE_FULL );
  }

  return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
int QSICameraAdapter::FanModePropertySetup()
{
  CPropertyAction * propertyAction;
  int response;
  vector<string> values;

  propertyAction = new CPropertyAction( this, &QSICameraAdapter::FanModePropertyHandler );
  response = CreateProperty( QSIString::FAN_MODE, QSIString::FAN_MODE_QUIET, MM::String, false, propertyAction );
  assert( response == DEVICE_OK );

  values.push_back( QSIString::FAN_MODE_OFF );
  values.push_back( QSIString::FAN_MODE_QUIET );
  values.push_back( QSIString::FAN_MODE_FULL );

  response = SetAllowedValues( QSIString::FAN_MODE, values );
  assert( response == DEVICE_OK );

  return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
int QSICameraAdapter::FilterWheelPositionPropertyHandler( MM::PropertyBase * pProp, MM::ActionType eAct )
{
  long value;
  int position;

  if( eAct == MM::AfterSet )
  {
    pProp->Get( value );
    position = static_cast<int>( value );

    m_status = QSI_SetFilterWheelPosition( m_handle, position - 1 );
    HANDLE_QSI_ERROR( this, m_status );
  }
  else if( eAct == MM::BeforeGet )
  {
    m_status = QSI_GetFilterWheelPosition( m_handle, &position );
    HANDLE_QSI_ERROR( this, m_status );

    pProp->Set( CDeviceUtils::ConvertToString( position + 1 ) );
  }

  return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
int QSICameraAdapter::FilterWheelPropertiesSetup()
{
  qsi_bool hasFilterWheel;
  const char * hasFilterWheelString;
  int filterWheelPositionCount;
  CPropertyAction * propertyAction;
  int response;
  vector<string> values;

  m_status = QSI_GetHasFilterWheel( m_handle, &hasFilterWheel );
  HANDLE_QSI_ERROR( this, m_status );

  if( hasFilterWheel )
    hasFilterWheelString = QSIString::HAS_FILTER_WHEEL_YES;
  else
    hasFilterWheelString = QSIString::HAS_FILTER_WHEEL_NO;

  response = CreateProperty( QSIString::HAS_FILTER_WHEEL, hasFilterWheelString , MM::String, true );
  assert( response == DEVICE_OK );

  if( hasFilterWheel )
  {
    m_status = QSI_GetFilterWheelPositionCount( m_handle, &filterWheelPositionCount );
    HANDLE_QSI_ERROR( this, m_status );

    response = CreateProperty( QSIString::FILTER_WHEEL_POSITIONS, CDeviceUtils::ConvertToString( filterWheelPositionCount ), MM::Integer, true );
    assert( response == DEVICE_OK );

    propertyAction = new CPropertyAction( this, &QSICameraAdapter::FilterWheelPositionPropertyHandler );
    response = CreateProperty( QSIString::FILTER_WHEEL_POSITION, CDeviceUtils::ConvertToString( 1 ), MM::Integer, false, propertyAction );
    assert( response == DEVICE_OK );

    for( int i = 0; i < filterWheelPositionCount; i++ )
      values.push_back( CDeviceUtils::ConvertToString( i + 1 ) );

    response = SetAllowedValues( QSIString::FILTER_WHEEL_POSITION, values );
    assert( response == DEVICE_OK );
  }

  return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
int QSICameraAdapter::FullWellCapacityPropertySetup()
{
  double value;
  int response;

  m_status = QSI_GetFullWellCapacity( m_handle, &value );
  HANDLE_QSI_ERROR( this, m_status );

  response = CreateProperty( QSIString::FULL_WELL_CAPACITY, CDeviceUtils::ConvertToString( value ), MM::Float, true );
  assert( response == DEVICE_OK );

  return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
int QSICameraAdapter::GainPropertiesSetup()
{
  qsi_bool canSetGain;
  double gain;
  CPropertyAction * propertyAction;
  int response;
  vector<string> values;

  m_status = QSI_GetCanSetGainMode( m_handle, &canSetGain );
  HANDLE_QSI_ERROR( this, m_status );

  m_status = QSI_GetElectronsPerADU( m_handle, &gain );
  HANDLE_QSI_ERROR( this, m_status );

  if( canSetGain )
  {
    propertyAction = new CPropertyAction( this, &QSICameraAdapter::GainModePropertyHandler );
    response = CreateProperty( QSIString::GAIN_MODE, QSIString::GAIN_MODE_HIGH, MM::String, false, propertyAction );
    assert( response == DEVICE_OK );

    values.push_back( QSIString::GAIN_MODE_HIGH );
    values.push_back( QSIString::GAIN_MODE_LOW );

    response = SetAllowedValues( QSIString::GAIN_MODE, values );
    assert( response == DEVICE_OK );

    propertyAction = new CPropertyAction( this, &QSICameraAdapter::GainPropertyHandler );
    response = CreateProperty( MM::g_Keyword_Gain, CDeviceUtils::ConvertToString( gain ), MM::Float, true, propertyAction );
    assert( response == DEVICE_OK );
  }
  else
  {
    response = CreateProperty( MM::g_Keyword_Gain, CDeviceUtils::ConvertToString( gain ), MM::Float, true );
    assert( response == DEVICE_OK );
  }

  return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
int QSICameraAdapter::GainPropertyHandler( MM::PropertyBase * pProp, MM::ActionType eAct )
{
  double value;

  if( eAct == MM::BeforeGet )
  {
    m_status = QSI_GetElectronsPerADU( m_handle, &value );
    HANDLE_QSI_ERROR( this, m_status );

    pProp->Set( value );
  }

  return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
int QSICameraAdapter::GainModePropertyHandler( MM::PropertyBase * pProp, MM::ActionType eAct )
{
  string value;
  qsi_gain_mode gain;
  int response;

  if( eAct == MM::AfterSet )
  {
    pProp->Get( value );

    if( value.compare( QSIString::GAIN_MODE_HIGH ) == 0 )
      m_status = QSI_SetGainMode( m_handle, QSI_GAIN_MODE_HIGH );
    else
      m_status = QSI_SetGainMode( m_handle, QSI_GAIN_MODE_LOW );

    HANDLE_QSI_ERROR( this, m_status );

    response = UpdateProperty( MM::g_Keyword_Gain );
    HANDLE_MM_ERROR( this, response, "An error occurred while updating the Gain property." );
  }
  else if( eAct == MM::BeforeGet )
  {
    m_status = QSI_GetGainMode( m_handle, &gain );
    HANDLE_QSI_ERROR( this, m_status );

    if( gain == QSI_GAIN_MODE_HIGH )
      pProp->Set( QSIString::GAIN_MODE_HIGH );
    else
      pProp->Set( QSIString::GAIN_MODE_LOW );
  }

  return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
int QSICameraAdapter::LEDEnabledPropertyHandler( MM::PropertyBase * pProp, MM::ActionType eAct )
{
  string value;
  qsi_bool enabled;

  if( eAct == MM::AfterSet )
  {
    pProp->Get( value );

    if( value.compare( QSIString::LED_SETTING_ENABLED ) == 0 )
      enabled = 1;
    else
      enabled = 0;

    m_status = QSI_SetLEDEnabled( m_handle, enabled );
    HANDLE_QSI_ERROR( this, m_status );
  }
  else if( eAct == MM::BeforeGet )
  {
    m_status = QSI_GetLEDEnabled( m_handle, &enabled );
    HANDLE_QSI_ERROR( this, m_status );

    if( enabled )
      pProp->Set( QSIString::LED_SETTING_ENABLED );
    else
      pProp->Set( QSIString::LED_SETTING_DISABLED );
  }

  return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
int QSICameraAdapter::LEDEnabledPropertySetup()
{
  qsi_bool canSetLEDEnabled;
  CPropertyAction * propertyAction;
  int response;
  vector<string> values;

  m_status = QSI_GetCanSetLEDEnabled( m_handle, &canSetLEDEnabled );
  HANDLE_QSI_ERROR( this, m_status );

  if( canSetLEDEnabled )
  {
    propertyAction = new CPropertyAction( this, &QSICameraAdapter::LEDEnabledPropertyHandler );
    response = CreateProperty( QSIString::LED_SETTING, QSIString::LED_SETTING_ENABLED, MM::String, false, propertyAction );
    assert( response == DEVICE_OK );

    values.push_back( QSIString::LED_SETTING_ENABLED );
    values.push_back( QSIString::LED_SETTING_DISABLED );

    response = SetAllowedValues( QSIString::LED_SETTING, values );
    assert( response == DEVICE_OK );
  }

  return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
int QSICameraAdapter::MaxADUPropertySetup()
{
  int value;
  int response;

  m_status = QSI_GetMaxADU( m_handle, &value );
  HANDLE_QSI_ERROR( this, m_status );

  response = CreateProperty( QSIString::MAX_ADU, CDeviceUtils::ConvertToString( value ), MM::Float, true );
  assert( response == DEVICE_OK );

  return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
int QSICameraAdapter::ModelNamePropertySetup()
{
  char value[QSI_LENGTH_MODEL_NAME];
  int response;

  m_status = QSI_GetModelName( m_handle, value, QSI_LENGTH_MODEL_NAME );
  HANDLE_QSI_ERROR( this, m_status );;

  response = CreateProperty( QSIString::MODEL_NAME, value, MM::String, true );
  assert( response == DEVICE_OK );

  return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
int QSICameraAdapter::ModelNumberPropertySetup()
{
  char value[QSI_LENGTH_MODEL_NUMBER];
  int response;

  m_status = QSI_GetModelNumber( m_handle, value, QSI_LENGTH_MODEL_NUMBER );
  HANDLE_QSI_ERROR( this, m_status );;

  response = CreateProperty( QSIString::MODEL_NUMBER, value, MM::String, true );
  assert( response == DEVICE_OK );

  return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
int QSICameraAdapter::OpenShutterPropertyHandler( MM::PropertyBase * pProp, MM::ActionType eAct )
{
  string value;

  if( eAct == MM::AfterSet )
  {
    pProp->Get( value );

    if( value.compare( QSIString::OPEN_SHUTTER_YES ) == 0 )
      m_exposureOpenShutter = true;
    else
      m_exposureOpenShutter = false;
  }
  else if( eAct == MM::BeforeGet )
  {
    if( m_exposureOpenShutter )
      pProp->Set( QSIString::OPEN_SHUTTER_YES );
    else
      pProp->Set( QSIString::OPEN_SHUTTER_NO );
  }

  return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
int QSICameraAdapter::OpenShutterPropertySetup()
{
  CPropertyAction * propertyAction;
  int response;
  vector<string> values;

  propertyAction = new CPropertyAction( this, &QSICameraAdapter::OpenShutterPropertyHandler );
  response = CreateProperty( QSIString::OPEN_SHUTTER, QSIString::OPEN_SHUTTER_YES, MM::String, false, propertyAction  );
  assert( response == DEVICE_OK );

  values.push_back( QSIString::OPEN_SHUTTER_NO );
  values.push_back( QSIString::OPEN_SHUTTER_YES );

  response = SetAllowedValues( QSIString::OPEN_SHUTTER, values );
  assert( response == DEVICE_OK );


  return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
int QSICameraAdapter::PCBTemperaturePropertyHandler( MM::PropertyBase * pProp, MM::ActionType eAct )
{
  double value;

  if( eAct == MM::BeforeGet )
  {
    m_status = QSI_GetPCBTemperature( m_handle, &value );
    HANDLE_QSI_ERROR( this, m_status );

    pProp->Set( CDeviceUtils::ConvertToString( value ) );
  }

  return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
int QSICameraAdapter::PCBTemperaturePropertySetup()
{
  qsi_bool canGetPCBTemperature;
  CPropertyAction * propertyAction;
  int response;

  m_status = QSI_GetCanGetPCBTemperature( m_handle, &canGetPCBTemperature );
  HANDLE_QSI_ERROR( this, m_status );

  if( canGetPCBTemperature )
  {
    propertyAction = new CPropertyAction( this, &QSICameraAdapter::PCBTemperaturePropertyHandler );
    response = CreateProperty( QSIString::PCB_TEMPERATURE, CDeviceUtils::ConvertToString( 0 ), MM::Float, true, propertyAction  );
    assert( response == DEVICE_OK );
  }

  return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
int QSICameraAdapter::PixelSizePropertiesSetup()
{
  int response;

  m_status = QSI_GetPixelSizeX( m_handle, &m_pixelSizeX );
  HANDLE_QSI_ERROR( this, m_status );

  m_status = QSI_GetPixelSizeY( m_handle, &m_pixelSizeY );
  HANDLE_QSI_ERROR( this, m_status );

  response = CreateProperty( QSIString::PIXEL_SIZE_X, CDeviceUtils::ConvertToString( m_pixelSizeX ), MM::Float, true  );
  assert( response == DEVICE_OK );

  response = CreateProperty( QSIString::PIXEL_SIZE_Y, CDeviceUtils::ConvertToString( m_pixelSizeY ), MM::Float, true  );
  assert( response == DEVICE_OK );

  return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
int QSICameraAdapter::PreExposureFlushPropertyHandler( MM::PropertyBase * pProp, MM::ActionType eAct )
{
  string value;
  qsi_pre_exposure_flush preExposureFlush;

  if( eAct == MM::AfterSet )
  {
    pProp->Get( value );

    if( value.compare( QSIString::PRE_EXPOSURE_FLUSH_NONE ) == 0 )
      preExposureFlush = QSI_FLUSH_NONE;
    else if( value.compare( QSIString::PRE_EXPOSURE_FLUSH_MODEST ) == 0 )
      preExposureFlush = QSI_FLUSH_MODEST;
    else if( value.compare( QSIString::PRE_EXPOSURE_FLUSH_AGGRESSIVE ) == 0 )
      preExposureFlush = QSI_FLUSH_AGGRESSIVE;
    else if( value.compare( QSIString::PRE_EXPOSURE_FLUSH_VERY_AGGRESSIVE ) == 0 )
      preExposureFlush = QSI_FLUSH_VERY_AGGRESSIVE;
    else
      preExposureFlush = QSI_FLUSH_NORMAL;

    m_status = QSI_SetPreExposureFlush( m_handle, preExposureFlush );
    HANDLE_QSI_ERROR( this, m_status );
  }
  else if( eAct == MM::BeforeGet )
  {
    m_status = QSI_GetPreExposureFlush( m_handle, &preExposureFlush );
    HANDLE_QSI_ERROR( this, m_status );

    if( preExposureFlush == QSI_FLUSH_NONE )
      pProp->Set( QSIString::PRE_EXPOSURE_FLUSH_NONE );
    else if( preExposureFlush == QSI_FLUSH_MODEST )
      pProp->Set( QSIString::PRE_EXPOSURE_FLUSH_MODEST );
    else if( preExposureFlush == QSI_FLUSH_AGGRESSIVE )
      pProp->Set( QSIString::PRE_EXPOSURE_FLUSH_AGGRESSIVE );
    else if( preExposureFlush == QSI_FLUSH_VERY_AGGRESSIVE )
      pProp->Set( QSIString::PRE_EXPOSURE_FLUSH_VERY_AGGRESSIVE );
    else
      pProp->Set( QSIString::PRE_EXPOSURE_FLUSH_NORMAL );
  }

  return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
int QSICameraAdapter::PreExposureFlushPropertySetup()
{
  qsi_bool canSetPreExposureFlush;
  CPropertyAction * propertyAction;
  int response;
  vector<string> values;

  m_status = QSI_GetCanSetPreExposureFlush( m_handle, &canSetPreExposureFlush );
  HANDLE_QSI_ERROR( this, m_status );

  if( canSetPreExposureFlush )
  {
    propertyAction = new CPropertyAction( this, &QSICameraAdapter::PreExposureFlushPropertyHandler );
    response = CreateProperty( QSIString::PRE_EXPOSURE_FLUSH, QSIString::PRE_EXPOSURE_FLUSH_NORMAL, MM::String, false, propertyAction );
    assert( response == DEVICE_OK );

    values.push_back( QSIString::PRE_EXPOSURE_FLUSH_NONE );
    values.push_back( QSIString::PRE_EXPOSURE_FLUSH_MODEST );
    values.push_back( QSIString::PRE_EXPOSURE_FLUSH_NORMAL );
    values.push_back( QSIString::PRE_EXPOSURE_FLUSH_AGGRESSIVE );
    values.push_back( QSIString::PRE_EXPOSURE_FLUSH_VERY_AGGRESSIVE );

    response = SetAllowedValues( QSIString::PRE_EXPOSURE_FLUSH, values );
    assert( response == DEVICE_OK );
  }

  return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
int QSICameraAdapter::ReadoutModePropertyHandler( MM::PropertyBase * pProp, MM::ActionType eAct )
{
  string value;
  qsi_readout_speed readoutSpeed;

  if( eAct == MM::AfterSet )
  {
    pProp->Get( value );

    if( value.compare( QSIString::READOUT_MODE_FAST_READOUT ) == 0 )
      m_status = QSI_SetReadoutSpeed( m_handle, QSI_READOUT_FAST );
    else
      m_status = QSI_SetReadoutSpeed( m_handle, QSI_READOUT_HIGH_QUALITY );

    HANDLE_QSI_ERROR( this, m_status );
  }
  else if( eAct == MM::BeforeGet )
  {
    m_status = QSI_GetReadoutSpeed( m_handle, &readoutSpeed );
    HANDLE_QSI_ERROR( this, m_status );

    if( readoutSpeed == QSI_READOUT_FAST )
      pProp->Set( QSIString::READOUT_MODE_FAST_READOUT );
    else
      pProp->Set( QSIString::READOUT_MODE_HIGH_QUALITY );
  }

  return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
int QSICameraAdapter::ReadoutModePropertySetup()
{
  qsi_bool canSetReadoutSpeed;
  CPropertyAction * propertyAction;
  int response;
  vector<string> values;

  m_status = QSI_GetCanSetReadoutSpeed( m_handle, &canSetReadoutSpeed );
  HANDLE_QSI_ERROR( this, m_status );

  propertyAction = new CPropertyAction( this, &QSICameraAdapter::ReadoutModePropertyHandler );
  response = CreateProperty( MM::g_Keyword_ReadoutMode, QSIString::READOUT_MODE_HIGH_QUALITY, MM::String, false, propertyAction );
  assert( response == DEVICE_OK );

  values.push_back( QSIString::READOUT_MODE_HIGH_QUALITY );

  if( canSetReadoutSpeed )
    values.push_back( QSIString::READOUT_MODE_FAST_READOUT ); 

  response = SetAllowedValues( MM::g_Keyword_ReadoutMode, values );
  assert( response == DEVICE_OK );

  return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
int QSICameraAdapter::ShutterPropertiesSetup()
{
  qsi_bool hasShutter, canSetShutterPriority;
  const char * hasShutterString;
  CPropertyAction * propertyAction;
  int response;
  vector<string> values;

  m_status = QSI_GetHasShutter( m_handle, &hasShutter );
  HANDLE_QSI_ERROR( this, m_status );

  if( hasShutter )
    hasShutterString = QSIString::HAS_SHUTTER_YES;
  else
    hasShutterString = QSIString::HAS_SHUTTER_NO;

  response = CreateProperty( QSIString::HAS_SHUTTER, hasShutterString, MM::String, true );
  assert( response == DEVICE_OK );

  m_status = QSI_GetCanSetShutterPriority( m_handle, &canSetShutterPriority );
  HANDLE_QSI_ERROR( this, m_status );

  if( hasShutter && canSetShutterPriority )
  {
    propertyAction = new CPropertyAction( this, &QSICameraAdapter::ShutterPriorityPropertyHandler );
    response = CreateProperty( QSIString::SHUTTER_PRIORITY, QSIString::SHUTTER_PRIORITY_ELECTRONIC, MM::String, false, propertyAction );
    assert( response == DEVICE_OK );

    values.push_back( QSIString::SHUTTER_PRIORITY_ELECTRONIC );
    values.push_back( QSIString::SHUTTER_PRIORITY_MECHANICAL );

    response = SetAllowedValues( QSIString::SHUTTER_PRIORITY, values );
    assert( response == DEVICE_OK );
  }

  return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
int QSICameraAdapter::ShutterPriorityPropertyHandler( MM::PropertyBase * pProp, MM::ActionType eAct )
{
  string value;
  qsi_shutter_priority shutterPriority;

  if( eAct == MM::AfterSet )
  {
    pProp->Get( value );

    if( value.compare( QSIString::SHUTTER_PRIORITY_ELECTRONIC ) == 0 )
      shutterPriority = QSI_SHUTTER_PRIORITY_ELECTRONIC;
    else
      shutterPriority = QSI_SHUTTER_PRIORITY_MECHANICAL;

    m_status = QSI_SetShutterPriority( m_handle, shutterPriority );
    HANDLE_QSI_ERROR( this, m_status );
  }
  else if( eAct == MM::BeforeGet )
  {
    m_status = QSI_GetShutterPriority( m_handle, &shutterPriority );
    HANDLE_QSI_ERROR( this, m_status );

    if( shutterPriority == QSI_SHUTTER_PRIORITY_ELECTRONIC )
      pProp->Set( QSIString::SHUTTER_PRIORITY_ELECTRONIC );
    else
      pProp->Set( QSIString::SHUTTER_PRIORITY_MECHANICAL );
  }

  return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
int QSICameraAdapter::SerialNumberPropertySetup()
{
  char serialNumber[QSI_LENGTH_SERIAL_NUMBER];
  int response;

  m_status = QSI_GetSerialNumber( m_handle, serialNumber, QSI_LENGTH_SERIAL_NUMBER );
  HANDLE_QSI_ERROR( this, m_status );

  response = CreateProperty( QSIString::SERIAL_NUMBER, serialNumber, MM::String, true );
  assert( response == DEVICE_OK );

  return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
int QSICameraAdapter::SoundEnabledPropertyHandler( MM::PropertyBase * pProp, MM::ActionType eAct )
{
  string value;
  qsi_bool enabled;

  if( eAct == MM::AfterSet )
  {
    pProp->Get( value );

    if( value.compare( QSIString::SOUND_SETTING_ENABLED ) == 0 )
      enabled = 1;
    else
      enabled = 0;

    m_status = QSI_SetSoundEnabled( m_handle, enabled );
    HANDLE_QSI_ERROR( this, m_status );
  }
  else if( eAct == MM::BeforeGet )
  {
    m_status = QSI_GetSoundEnabled( m_handle, &enabled );
    HANDLE_QSI_ERROR( this, m_status );

    if( enabled )
      pProp->Set( QSIString::SOUND_SETTING_ENABLED );
    else
      pProp->Set( QSIString::SOUND_SETTING_DISABLED );
  }

  return DEVICE_OK;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
int QSICameraAdapter::SoundEnabledPropertySetup()
{
  qsi_bool canSetSoundEnabled;
  CPropertyAction * propertyAction;
  int response;
  vector<string> values;

  m_status = QSI_GetCanSetSoundEnabled( m_handle, &canSetSoundEnabled );
  HANDLE_QSI_ERROR( this, m_status );

  if( canSetSoundEnabled )
  {
    propertyAction = new CPropertyAction( this, &QSICameraAdapter::SoundEnabledPropertyHandler );
    response = CreateProperty( QSIString::SOUND_SETTING, QSIString::SOUND_SETTING_ENABLED, MM::String, false, propertyAction );
    assert( response == DEVICE_OK );

    values.push_back( QSIString::SOUND_SETTING_ENABLED );
    values.push_back( QSIString::SOUND_SETTING_DISABLED );

    response = SetAllowedValues( QSIString::SOUND_SETTING, values );
    assert( response == DEVICE_OK );
  }

  return DEVICE_OK;
}

#pragma endregion
