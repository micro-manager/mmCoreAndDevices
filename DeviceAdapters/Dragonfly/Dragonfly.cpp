///////////////////////////////////////////////////////////////////////////////
// FILE:          Dragonfly.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------

#include "../../MMDevice/ModuleInterface.h"
#include "Dragonfly.h"
#include "DichroicMirror.h"
#include "FilterWheel.h"
#include "ASDWrapper/ASDWrapper.h"
#include "DragonflyStatus.h"
#include "Disk.h"
#include "ConfocalMode.h"
#include "Aperture.h"
#include "CameraPortMirror.h"
#include "Lens.h"
#include "SuperRes.h"
#include "TIRF.h"
#include "ConfigFileHandler.h"

#include "ASDInterface.h"

#include <string>
#include <stdexcept>

using namespace std;

const char* const g_DeviceName = "Andor Dragonfly";
const char* const g_DeviceDescription = "Andor Dragonfly";
const char* const g_DevicePort = "COM Port";
const char* const g_DeviceSerialNumber = "Description | Serial Number";
const char* const g_DeviceSoftwareVersion = "Description | Software Version";

///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////

MODULE_API void InitializeModuleData()
{
  RegisterDevice( g_DeviceName, MM::GenericDevice, g_DeviceDescription );
}

MODULE_API MM::Device * CreateDevice( const char * DeviceName )
{
  if ( DeviceName == nullptr )
  {
    return nullptr;
  }

  // decide which device class to create based on the deviceName parameter
  if ( strcmp( DeviceName, g_DeviceName ) == 0 )
  {
    // create Dragonfly
    MM::Device * vOpenedDevice = new CDragonfly();
    return vOpenedDevice;
  }

  // ...supplied name not recognized
  return nullptr;
}

MODULE_API void DeleteDevice( MM::Device * Device )
{
  delete Device;
}

///////////////////////////////////////////////////////////////////////////////
// CDragonfly
///////////////////////////////////////////////////////////////////////////////

CDragonfly::CDragonfly()
  : Initialized_( false ),
  ConstructionReturnCode_( DEVICE_OK ),
  ASDWrapper_( nullptr ),
  ASDLoader_( nullptr ),
  DichroicMirror_( nullptr ),
  FilterWheel1_( nullptr ),
  FilterWheel2_( nullptr ),
  DragonflyStatus_( nullptr ),
  Disk_( nullptr ),
  ConfocalMode_( nullptr ),
  Aperture_( nullptr ),
  CameraPortMirror_( nullptr ),
  SuperRes_( nullptr ),
  TIRF_( nullptr ),
  ConfigFile_( nullptr )
{
  InitializeDefaultErrorMessages();

  string vContactSupportMessage( "Please contact Andor support and send the latest file present in the Micro-Manager CoreLogs directory." );
#ifdef _M_X64
  SetErrorText( ERR_LIBRARY_LOAD, "Failed to load the Dragonfly library. Make sure AB_ASDx64.dll is present in the Micro-Manager root directory." );
#else
  SetErrorText( ERR_LIBRARY_LOAD, "Failed to load the Dragonfly library. Make sure AB_ASD.dll is present in the Micro-Manager root directory." );
#endif
  SetErrorText( ERR_LIBRARY_INIT, "Dragonfly Library initialisation failed. Make sure the device is connected and you selected the correct COM port." );
  string vMessage = "Dichroic mirror initialisation failed. " + vContactSupportMessage;
  SetErrorText( ERR_DICHROICMIRROR_INIT, vMessage.c_str() );
  vMessage = "Filter wheel 1 initialisation failed. " + vContactSupportMessage;
  SetErrorText( ERR_FILTERWHEEL1_INIT, vMessage.c_str() );
  vMessage = "Filter wheel 2 initialisation failed. " + vContactSupportMessage;
  SetErrorText( ERR_FILTERWHEEL2_INIT, vMessage.c_str() );
  vMessage = "Dragonfly Status class not accessible. " + vContactSupportMessage;
  SetErrorText( ERR_DRAGONFLYSTATUS_INVALID_POINTER, vMessage.c_str() );
  vMessage = "Dragonfly Status initialisation failed. " + vContactSupportMessage;
  SetErrorText( ERR_DRAGONFLYSTATUS_INIT, vMessage.c_str() );
  vMessage = "Disk speed initialisation failed. " + vContactSupportMessage;
  SetErrorText( ERR_DISK_INIT , vMessage.c_str());
  vMessage = "Imaging mode/Power density initialisation failed. " + vContactSupportMessage;
  SetErrorText( ERR_CONFOCALMODE_INIT, vMessage.c_str() );
  vMessage = "Field Aperture initialisation failed. " + vContactSupportMessage;
  SetErrorText( ERR_APERTURE_INIT, vMessage.c_str() );
  vMessage = "Image splitter initialisation failed. " + vContactSupportMessage;
  SetErrorText( ERR_CAMERAPORTMIRROR_INIT, vMessage.c_str() );
  vMessage = "Lens initialisation failed. " + vContactSupportMessage;
  SetErrorText( ERR_LENS_INIT, vMessage.c_str() );
  vMessage = "Power density initialisation failed. " + vContactSupportMessage;
  SetErrorText( ERR_POWERDENSITY_INIT, vMessage.c_str() );
  vMessage = "Super Resolution initialisation failed. " + vContactSupportMessage;
  SetErrorText( ERR_SUPERRES_INIT, vMessage.c_str() );
  vMessage = "TIRF initialisation failed. " + vContactSupportMessage;
  SetErrorText( ERR_TIRF_INIT, vMessage.c_str() );
  SetErrorText( ERR_CONFIGFILEIO_ERROR, "Can't write to Dragonfly configuration file. Please select a file for which you have read and write access." );
  SetErrorText( ERR_COMPORTPROPERTY_CREATION, "Error creating COM Port property" );
  SetErrorText( ERR_CONFIGFILEPROPERTY_CREATION, "Error creating Configuration File path property" );

  // Connect to ASD wrapper
  try
  {
    ASDWrapper_ = new CASDWrapper();
  }
  catch ( exception& vException )
  {
    ASDWrapper_ = nullptr;
    vMessage = "Error loading the Dragonfly library. Caught Exception with message: ";
    vMessage += vException.what();
    LogMessage( vMessage );
    ConstructionReturnCode_ = ERR_LIBRARY_LOAD;
  }

  // COM Port property
  int vRet = CreatePropertyWithHandler( g_DevicePort, "Undefined", MM::String, false, &CDragonfly::OnPort, true );
  if ( vRet == DEVICE_OK )
  {
    string vComPortBaseName;
    string vComPortAddress;
    for ( int i = 1; i <= 64; i++ )
    {
      vComPortBaseName = "COM" + to_string( static_cast< long long >( i ) );
      vComPortAddress = "\\\\.\\" + vComPortBaseName;
      HANDLE hCom = CreateFile( vComPortAddress.c_str(),
        GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL, NULL );

      if ( hCom != INVALID_HANDLE_VALUE )
      {
        AddAllowedValue( g_DevicePort, vComPortBaseName.c_str() );
        CloseHandle( hCom );
      }
    }
  }
  else
  {
    ConstructionReturnCode_ = ERR_COMPORTPROPERTY_CREATION;
  }

  // Config file property
  try
  {
    ConfigFile_ = new CConfigFileHandler( this );
  }
  catch ( std::exception& )
  {
    ConstructionReturnCode_ = ERR_CONFIGFILEPROPERTY_CREATION;
  }
}

CDragonfly::~CDragonfly()
{
  delete ConfigFile_;
  delete ASDWrapper_;
}

void CDragonfly::UpdatePropertyUI( const char* PropertyName, const char* PropertyValue )
{
  GetCoreCallback()->OnPropertyChanged( this, PropertyName, PropertyValue );
}

int CDragonfly::Initialize()
{
  if ( ConstructionReturnCode_ != DEVICE_OK )
  {
    return ConstructionReturnCode_;
  }

  if ( Initialized_ )
  {
    return DEVICE_OK;
  }

  // Load Config file
  int vRet = ConfigFile_->LoadConfig();
  if ( vRet != DEVICE_OK )
  {
    return vRet;
  }

  // Remove the leading space from the port name if there's any before connecting
  size_t vFirstValidCharacter = Port_.find_first_not_of( " " );
  vRet = Connect( Port_.substr( vFirstValidCharacter ) );
  if ( vRet != DEVICE_OK )
  {
    return vRet;
  }

  vRet = InitializeComponents();
  if ( vRet != DEVICE_OK )
  {
    // MicroManager doesn't call Shutdown on initilisation fail so we do it ourselves to keep the adapter in a clean state
    Shutdown();
    return vRet;
  }

  Initialized_ = true;
  return vRet;
}

int CDragonfly::Shutdown()
{
  if ( ConstructionReturnCode_ != DEVICE_OK )
  {
    return DEVICE_OK;
  }

  // It's important to reset pointers and clear lists in case there's a problem in the next Initialize call
  delete DichroicMirror_;
  DichroicMirror_ = nullptr;
  delete FilterWheel1_;
  FilterWheel1_ = nullptr;
  delete FilterWheel2_;
  FilterWheel2_ = nullptr;
  delete DragonflyStatus_;
  DragonflyStatus_ = nullptr;
  delete Disk_;
  Disk_ = nullptr;
  delete ConfocalMode_;
  ConfocalMode_ = nullptr;
  delete Aperture_;
  Aperture_ = nullptr;
  delete CameraPortMirror_;
  CameraPortMirror_ = nullptr;
  list<CLens*>::iterator vLensIt = Lens_.begin();
  while ( vLensIt != Lens_.end() )
  {
    delete *vLensIt;
    vLensIt++;
  }
  Lens_.clear();
  delete SuperRes_;
  SuperRes_ = nullptr;
  delete TIRF_;
  TIRF_ = nullptr;

  Disconnect();

  Initialized_ = false;

  return DEVICE_OK;
}

/**
* Obtains device name.
* Required by the MM::Device API.
*/
void CDragonfly::GetName( char * Name ) const
{
  // We just return the name we use for referring to this device adapter.
  CDeviceUtils::CopyLimitedString( Name, g_DeviceName );
}

bool CDragonfly::Busy()
{
  return false;
}

int CDragonfly::OnPort( MM::PropertyBase* Prop, MM::ActionType Act )
{
  int vRet = DEVICE_OK;
  if ( Act == MM::BeforeGet )
  {
    Prop->Set( Port_.c_str() );
  }
  else if ( Act == MM::AfterSet )
  {
    Prop->Get( Port_ );
  }
  return vRet;
}

int CDragonfly::Connect( const string& Port )
{
  // Try connecting to the Dragonfly
  string vPort( "\\\\.\\" + Port );
  ASDWrapper_->CreateASDLoader( vPort.c_str(), ASD_DF, &ASDLoader_ );
  if ( ASDLoader_ == nullptr )
  {
    string vMessage( "CreateASDLoader failed on port " + Port );
    LogMessage( vMessage );
    return ERR_LIBRARY_INIT;
  }

  return DEVICE_OK;
}

int CDragonfly::Disconnect()
{
  if ( ASDLoader_ != nullptr )
  {
    ASDWrapper_->DeleteASDLoader( ASDLoader_ );
    ASDLoader_ = nullptr;
  }

  return DEVICE_OK;
}

int CDragonfly::InitializeComponents()
{
  IASDInterface* vASDInterface = ASDLoader_->GetASDInterface();
  IASDInterface2* vASDInterface2 = ASDLoader_->GetASDInterface2();
  IASDInterface3* vASDInterface3 = ASDLoader_->GetASDInterface3();

  // Description property
  int vRet = CreateProperty( MM::g_Keyword_Description, g_DeviceDescription, MM::String, true );
  if ( vRet != DEVICE_OK )
  {
    LogMessage( "Error encountered when creating Description property" );
    return vRet;
  }

  // Serial number property
  string vSerialNumber = vASDInterface->GetSerialNumber();
  vRet = CreateProperty( g_DeviceSerialNumber, vSerialNumber.c_str(), MM::String, true );
  if ( vRet != DEVICE_OK )
  {
    LogMessage( "Error encountered when creating " + string( g_DeviceSerialNumber ) + " property" );
    return vRet;
  }

  // Software version property
  string vSoftwareVersion = vASDInterface->GetSoftwareVersion();
  vRet = CreateProperty( g_DeviceSoftwareVersion, vSoftwareVersion.c_str(), MM::String, true );
  if ( vRet != DEVICE_OK )
  {
    LogMessage( "Error encountered when creating " + string( g_DeviceSoftwareVersion ) + " property" );
    return vRet;
  }

  // Status
  vRet = CreateDragonflyStatus( vASDInterface3 );
  if ( vRet != DEVICE_OK )
  {
    return vRet;
  }
  
  // Dichroic mirror component
  vRet = CreateDichroicMirror( vASDInterface );
  if ( vRet != DEVICE_OK )
  {
    return vRet;
  }

  // Filter wheel 1 component
  vRet = CreateFilterWheel( vASDInterface, FilterWheel1_, WheelIndex1, ERR_FILTERWHEEL1_INIT );
  if ( vRet != DEVICE_OK )
  {
    return vRet;
  }

  // Filter wheel 2 component
  vRet = CreateFilterWheel( vASDInterface, FilterWheel2_, WheelIndex2, ERR_FILTERWHEEL2_INIT );
  if ( vRet != DEVICE_OK )
  {
    return vRet;
  }

  // Disk component
  vRet = CreateDisk( vASDInterface );
  if ( vRet != DEVICE_OK )
  {
    return vRet;
  }

  // Confocal mode component
  vRet = CreateConfocalMode( vASDInterface3 );
  if ( vRet != DEVICE_OK )
  {
    return vRet;
  }

  // Aperture component
  vRet = CreateAperture( vASDInterface2 );
  if ( vRet != DEVICE_OK )
  {
    return vRet;
  }

  // Camera port mirror component
  vRet = CreateCameraPortMirror( vASDInterface2 );
  if ( vRet != DEVICE_OK )
  {
    return vRet;
  }

  // Lens components
  for ( int vLensIndex = lt_Lens1; vLensIndex < lt_LensMax; ++vLensIndex )
  {
    vRet = CreateLens( vASDInterface2, vLensIndex );
    if ( vRet != DEVICE_OK )
    {
      return vRet;
    }
  }

  // Super Resolution component
  vRet = CreateSuperRes( vASDInterface3 );
  if ( vRet != DEVICE_OK )
  {
    return vRet;
  }

  // TIRF component
  vRet = CreateTIRF( vASDInterface3 );
  if ( vRet != DEVICE_OK )
  {
    return vRet;
  }
  
  return vRet;
}

int CDragonfly::CreateDragonflyStatus( IASDInterface3* ASDInterface3 )
{
  int vErrorCode = DEVICE_OK;
  try
  {
    IStatusInterface* vASDStatus = ASDInterface3->GetStatus();
    if ( vASDStatus != nullptr )
    {
      DragonflyStatus_ = new CDragonflyStatus( vASDStatus, this );
    }
    else
    {
      vErrorCode = ERR_DRAGONFLYSTATUS_INVALID_POINTER;
    }
  }
  catch ( exception& vException )
  {
    string vMessage( "Error loading the Dragonfly Status. Caught Exception with message: " );
    vMessage += vException.what();
    LogMessage( vMessage );
    vErrorCode = ERR_DRAGONFLYSTATUS_INIT;
  }
  return vErrorCode;
}

int CDragonfly::CreateDichroicMirror( IASDInterface* ASDInterface )
{
  int vErrorCode = DEVICE_OK;
  if ( ASDInterface->IsDichroicAvailable() )
  {
    try
    {
      IDichroicMirrorInterface* vASDDichroicMirror = ASDInterface->GetDichroicMirror();
      if ( vASDDichroicMirror != nullptr )
      {
        DichroicMirror_ = new CDichroicMirror( vASDDichroicMirror, this );
      }
      else
      {
        LogMessage( "Dichroic mirror ASD SDK pointer invalid" );
        vErrorCode = ERR_DICHROICMIRROR_INIT;
      }
    }
    catch ( exception& vException )
    {
      string vMessage( "Error loading the Dichroic mirror. Caught Exception with message: " );
      vMessage += vException.what();
      LogMessage( vMessage );
      vErrorCode = ERR_DICHROICMIRROR_INIT;
    }
  }
  else
  {
    LogMessage( "Dichroic mirror not available", true );
  }
  return vErrorCode;
}

int CDragonfly::CreateFilterWheel( IASDInterface* ASDInterface, CFilterWheel*& FilterWheel, TWheelIndex WheelIndex, unsigned int ErrorCode )
{
  int vErrorCode = DEVICE_OK;
  if ( ASDInterface->IsFilterWheelAvailable( WheelIndex ) )
  {
    try
    {
      IFilterWheelInterface* vASDFilterWheel = ASDInterface->GetFilterWheel( WheelIndex );
      if ( vASDFilterWheel != nullptr )
      {
        FilterWheel = new CFilterWheel( WheelIndex, vASDFilterWheel, DragonflyStatus_, ConfigFile_, this );
      }
      else
      {
        LogMessage( "Filter wheel " + to_string( static_cast< long long >( WheelIndex ) ) + " ASD SDK pointer invalid" );
        vErrorCode = ErrorCode;
      }
    }
    catch ( exception& vException )
    {
      string vMessage( "Error loading the filter wheel " + to_string( static_cast< long long >( WheelIndex ) ) + ". Caught Exception with message: " );
      vMessage += vException.what();
      LogMessage( vMessage );
      vErrorCode = ErrorCode;
    }
  }
  else
  {
    LogMessage( "Filter wheel " + to_string( static_cast< long long >( WheelIndex ) ) + " not available", true );
  }
  return vErrorCode;
}

int CDragonfly::CreateDisk( IASDInterface* ASDInterface )
{
  int vErrorCode = DEVICE_OK;
  if ( ASDInterface->IsDiskAvailable() )
  {
    try
    {
      IDiskInterface2* vASDDisk = ASDInterface->GetDisk_v2();
      if ( vASDDisk != nullptr )
      {
        Disk_ = new CDisk( vASDDisk, ConfigFile_, this );
      }
      else
      {
        LogMessage( "Spinning disk ASD SDK pointer invalid" );
        vErrorCode = ERR_DISK_INIT;
      }
    }
    catch ( exception& vException )
    {
      string vMessage( "Error loading the Spinning disk. Caught Exception with message: " );
      vMessage += vException.what();
      LogMessage( vMessage );
      vErrorCode = ERR_DISK_INIT;
    }
  }
  else
  {
    LogMessage( "Spinning disk not available", true );
  }
  return vErrorCode;
}

int CDragonfly::CreateConfocalMode( IASDInterface3* ASDInterface )
{
  int vErrorCode = DEVICE_OK;
  if ( ASDInterface->IsImagingModeAvailable() )
  {
    try
    {
      IConfocalModeInterface3* vASDConfocalMode = ASDInterface->GetImagingMode();
      if ( vASDConfocalMode != nullptr )
      {
        IIllLensInterface* vIllLensInterface = nullptr;
        for ( int vLensIndex = lt_Lens1; vLensIndex < lt_LensMax && vIllLensInterface == nullptr; ++vLensIndex )
        {
          if ( ASDInterface->IsIllLensAvailable( (TLensType)vLensIndex ) )
          {
            vIllLensInterface = ASDInterface->GetIllLens( (TLensType)vLensIndex );
          }
        }
        if ( vIllLensInterface == nullptr )
        {
          LogMessage( "Couldn't find any valid instance of Power Density" );
        }
        ConfocalMode_ = new CConfocalMode( vASDConfocalMode, vIllLensInterface, this );
      }
      else
      {
        LogMessage( "Confocal mode ASD SDK pointer invalid" );
        vErrorCode = ERR_CONFOCALMODE_INIT;
      }     
    }
    catch ( exception& vException )
    {
      string vMessage( "Error loading the Confocal mode. Caught Exception with message: " );
      vMessage += vException.what();
      LogMessage( vMessage );
      vErrorCode = ERR_CONFOCALMODE_INIT;
    }
  }
  else
  {
    LogMessage( "Confocal mode not available", true );
  }
  return vErrorCode;
}

int CDragonfly::CreateAperture( IASDInterface2* ASDInterface )
{
  int vErrorCode = DEVICE_OK;
  if ( ASDInterface->IsApertureAvailable() )
  {
    try
    {
      IApertureInterface* vAperture = ASDInterface->GetAperture();
      if ( vAperture != nullptr )
      {
        Aperture_ = new CAperture( vAperture, this );
      }
      else
      {
        LogMessage( "Aperture ASD SDK pointer invalid" );
        vErrorCode = ERR_APERTURE_INIT;
      }
    }
    catch ( exception& vException )
    {
      string vMessage( "Error loading the Aperture. Caught Exception with message: " );
      vMessage += vException.what();
      LogMessage( vMessage );
      vErrorCode = ERR_APERTURE_INIT;
    }
  }
  else
  {
    LogMessage( "Aperture not available", true );
  }
  return vErrorCode;
}

int CDragonfly::CreateCameraPortMirror( IASDInterface2* ASDInterface )
{
  int vErrorCode = DEVICE_OK;
  if ( ASDInterface->IsCameraPortMirrorAvailable() )
  {
    try
    {
      ICameraPortMirrorInterface* vCameraPortMirror = ASDInterface->GetCameraPortMirror();
      if ( vCameraPortMirror != nullptr )
      {
        CameraPortMirror_ = new CCameraPortMirror( vCameraPortMirror, DragonflyStatus_, this );
      }
      else
      {
        LogMessage( "Camera port mirror ASD SDK pointer invalid" );
        vErrorCode = ERR_CAMERAPORTMIRROR_INIT;
      }
    }
    catch ( exception& vException )
    {
      string vMessage( "Error loading the Camera port mirror. Caught Exception with message: " );
      vMessage += vException.what();
      LogMessage( vMessage );
      vErrorCode = ERR_CAMERAPORTMIRROR_INIT;
    }
  }
  else
  {
    LogMessage( "Camera port mirror not available", true );
  }
  return vErrorCode;
}

int CDragonfly::CreateLens( IASDInterface2* ASDInterface, int LensIndex )
{
  int vErrorCode = DEVICE_OK;
  if ( ASDInterface->IsLensAvailable( (TLensType)LensIndex ) )
  {
    try
    {
      ILensInterface* vLensInterface = ASDInterface->GetLens( (TLensType)LensIndex );
      if ( vLensInterface != nullptr )
      {
        CLens* vLens = new CLens( vLensInterface, LensIndex, this );
        if ( vLens != nullptr )
        {
          Lens_.push_back( vLens );
        }
      }
      else
      {
        LogMessage( "Lens " + to_string( static_cast< long long >( LensIndex ) ) + " ASD SDK pointer invalid" );
        vErrorCode = ERR_LENS_INIT;
      }
    }
    catch ( exception& vException )
    {
      string vMessage( "Error loading the Lens " + to_string( static_cast< long long >( LensIndex ) ) + ". Caught Exception with message: " );
      vMessage += vException.what();
      LogMessage( vMessage );
      vErrorCode = ERR_LENS_INIT;
    }
  }
  else
  {
    LogMessage( "Lens " + to_string( static_cast< long long >( LensIndex ) ) + " not available", true );
  }
  return vErrorCode;
}

int CDragonfly::CreateSuperRes( IASDInterface3* ASDInterface )
{
  int vErrorCode = DEVICE_OK;
  if ( ASDInterface->IsSuperResAvailable() )
  {
    try
    {
      ISuperResInterface* vSuperRes = ASDInterface->GetSuperRes();
      if ( vSuperRes != nullptr )
      {
        SuperRes_ = new CSuperRes( vSuperRes, this );
      }
      else
      {
        LogMessage( "Super resolution ASD SDK pointer invalid" );
        vErrorCode = ERR_SUPERRES_INIT;
      }
    }
    catch ( exception& vException )
    {
      string vMessage( "Error loading the Super resolution. Caught Exception with message: " );
      vMessage += vException.what();
      LogMessage( vMessage );
      vErrorCode = ERR_SUPERRES_INIT;
    }
  }
  else
  {
    LogMessage( "Super resolution not available", true );
  }
  return vErrorCode;
}

int CDragonfly::CreateTIRF( IASDInterface3* ASDInterface )
{
  int vErrorCode = DEVICE_OK;
  if ( ASDInterface->IsTIRFAvailable() )
  {
    try
    {
      ITIRFInterface* vTIRF = ASDInterface->GetTIRF();
      if ( vTIRF != nullptr )
      {
        TIRF_ = new CTIRF( vTIRF, ConfigFile_, this );
      }
      else
      {
        LogMessage( "TIRF ASD SDK pointer invalid" );
        vErrorCode = ERR_TIRF_INIT;
      }
    }
    catch ( exception& vException )
    {
      string vMessage( "Error loading TIRF. Caught Exception with message: " );
      vMessage += vException.what();
      LogMessage( vMessage );
      vErrorCode = ERR_TIRF_INIT;
    }
  }
  else
  {
    LogMessage( "TIRF not available", true );
  }
  return vErrorCode;
}

void CDragonfly::LogComponentMessage( const std::string& Message )
{
  LogMessage( Message );
}
