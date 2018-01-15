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
#include "PowerDensity.h"
#include "SuperRes.h"
#include "TIRF.h"

#include "ASDInterface.h"

#include <string>
#include <stdexcept>
#include <vector>

using namespace std;

const char* const g_DeviceName = "Dragonfly";
const char* const g_DeviceDescription = "Andor Dragonfly Device Adapter";
const char* const g_DevicePort = "COM Port";
const char* const g_DeviceSerialNumber = "Serial Number";
const char* const g_DeviceProductID = "Product ID";
const char* const g_DeviceSoftwareVersion = "Software Version";

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
  ASDWrapper_( nullptr ),
  ASDLoader_( nullptr ),
  ASDLibraryConnected_( false ),
  DichroicMirror_( nullptr ),
  FilterWheel1_( nullptr ),
  FilterWheel2_( nullptr ),
  DragonflyStatus_( nullptr ),
  Disk_( nullptr ),
  ConfocalMode_( nullptr ),
  Aperture_( nullptr ),
  CameraPortMirror_( nullptr ),
  SuperRes_( nullptr ),
  TIRF_( nullptr )
{
  InitializeDefaultErrorMessages();

  string vContactSupportMessage( "Please contact Andor support and send the latest file present in the Micro-Manager CoreLogs directory." );
#ifdef _M_X64
  SetErrorText( ERR_LIBRARY_LOAD, "Failed to load the ASD library. Make sure AB_ASDx64.dll is present in the Micro-Manager root directory." );
#else
  SetErrorText( ERR_LIBRARY_LOAD, "Failed to load the ASD library. Make sure AB_ASD.dll is present in the Micro-Manager root directory." );
#endif
  SetErrorText( ERR_LIBRARY_INIT, "ASD Library initialisation failed. Make sure the device is connected and you selected the correct COM port." );
  string vMessage = "Dichroic mirror initialisation failed. " + vContactSupportMessage;
  SetErrorText( ERR_DICHROICMIRROR_INIT, vMessage.c_str() );
  vMessage = "Filter wheel 1 initialisation failed. " + vContactSupportMessage;
  SetErrorText( ERR_FILTERWHEEL1_INIT, vMessage.c_str() );
  vMessage = "Filter wheel 2 initialisation failed. " + vContactSupportMessage;
  SetErrorText( ERR_FILTERWHEEL2_INIT, vMessage.c_str() );
  vMessage = "ASD Status class not accessible. " + vContactSupportMessage;
  SetErrorText( ERR_DRAGONFLYSTATUS_INVALID_POINTER, vMessage.c_str() );
  vMessage = "ASD Status initialisation failed. " + vContactSupportMessage;
  SetErrorText( ERR_DRAGONFLYSTATUS_INIT, vMessage.c_str() );
  vMessage = "Disk speed initialisation failed. " + vContactSupportMessage;
  SetErrorText( ERR_DISK_INIT , vMessage.c_str());
  vMessage = "Confocal mode initialisation failed. " + vContactSupportMessage;
  SetErrorText( ERR_CONFOCALMODE_INIT, vMessage.c_str() );
  vMessage = "Aperture initialisation failed. " + vContactSupportMessage;
  SetErrorText( ERR_APERTURE_INIT, vMessage.c_str() );
  vMessage = "Camera port mirror initialisation failed. " + vContactSupportMessage;
  SetErrorText( ERR_CAMERAPORTMIRROR_INIT, vMessage.c_str() );
  vMessage = "Lens initialisation failed. " + vContactSupportMessage;
  SetErrorText( ERR_LENS_INIT, vMessage.c_str() );
  vMessage = "Power density initialisation failed. " + vContactSupportMessage;
  SetErrorText( ERR_POWERDENSITY_INIT, vMessage.c_str() );
  vMessage = "Super Resolution initialisation failed. " + vContactSupportMessage;
  SetErrorText( ERR_SUPERRES_INIT, vMessage.c_str() );
  vMessage = "TIRF initialisation failed. " + vContactSupportMessage;
  SetErrorText( ERR_TIRF_INIT, vMessage.c_str() );

  // Connect to ASD wrapper
  try
  {
    ASDWrapper_ = new CASDWrapper();
    ASDLibraryConnected_ = true;
  }
  catch ( exception& vException )
  {
    ASDWrapper_ = nullptr;
    vMessage = "Error loading the ASD library. Caught Exception with message: ";
    vMessage += vException.what();
    LogMessage( vMessage );
  }

  // COM Port property
  int vRet = CreatePropertyWithHandler( g_DevicePort, "Undefined", MM::String, false, &CDragonfly::OnPort, true );
  if ( vRet == DEVICE_OK )
  {
    for ( int i = 1;i <= 20;++i )
    {
      string vCOMPort;
      if ( i < 10 )
      {
        vCOMPort = " ";
      }
      vCOMPort += "COM" + to_string( i );
      AddAllowedValue( g_DevicePort, vCOMPort.c_str() );
    }
  }
  else
  {
    LogMessage( "Error creating " + string( g_DevicePort ) + " property" );
  }

}

CDragonfly::~CDragonfly()
{
  delete ASDWrapper_;
}

int CDragonfly::Initialize()
{
  if ( !ASDLibraryConnected_ )
  {
    return ERR_LIBRARY_LOAD;
  }

  if ( Initialized_ )
  {
    return DEVICE_OK;
  }

  // Description property
  int vRet = CreateProperty( MM::g_Keyword_Description, g_DeviceDescription, MM::String, true );
  if ( vRet != DEVICE_OK )
  {
    return vRet;
  }
     
  // remove the leading space from the port name if there's any before connecting
  size_t vFirstValidCharacter = Port_.find_first_not_of( " " );
  if ( Connect( Port_.substr( vFirstValidCharacter ) ) == DEVICE_OK )
  {
    vRet = InitializeComponents();
  }
  else
  {
    return ERR_LIBRARY_INIT;
  }

  Initialized_ = true;
  return DEVICE_OK;
}

int CDragonfly::Shutdown()
{
  if ( !ASDLibraryConnected_ )
  {
    return ERR_LIBRARY_LOAD;
  }

  delete DichroicMirror_;
  delete FilterWheel1_;
  delete FilterWheel2_;
  delete DragonflyStatus_;
  delete Disk_;
  delete ConfocalMode_;
  delete Aperture_;
  delete CameraPortMirror_;
  vector<CLens*>::iterator vLensIt = Lens_.begin();
  while ( vLensIt != Lens_.end() )
  {
    delete *vLensIt;
    vLensIt++;
  }
  vector<CPowerDensity*>::iterator vPowerDensityIt = PowerDensity_.begin();
  while ( vPowerDensityIt != PowerDensity_.end() )
  {
    delete *vPowerDensityIt;
    vPowerDensityIt++;
  }
  delete SuperRes_;
  delete TIRF_;
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
  ASDWrapper_->CreateASDLoader( vPort.c_str(), TASDType::ASD_DF, &ASDLoader_ );
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

  LogMessage( "Getting Serial Number" );
  // Serial number property
  string vSerialNumber = vASDInterface->GetSerialNumber();
  int vRet = CreateProperty( g_DeviceSerialNumber, vSerialNumber.c_str(), MM::String, true );
  if ( vRet != DEVICE_OK )
  {
    return vRet;
  }

  LogMessage( "Getting Product ID" );
  // Product ID property
  string vProductID = vASDInterface->GetProductID();
  vRet = CreateProperty( g_DeviceProductID, vProductID.c_str(), MM::String, true );
  if ( vRet != DEVICE_OK )
  {
    return vRet;
  }

  LogMessage( "Getting Software Version" );
  // Software version property
  string vSoftwareVersion = vASDInterface->GetSoftwareVersion();
  vRet = CreateProperty( g_DeviceSoftwareVersion, vSoftwareVersion.c_str(), MM::String, true );
  if ( vRet != DEVICE_OK )
  {
    return vRet;
  }

  LogMessage( "Creating Dragonfly Status" );
  vRet = CreateDragonflyStatus( vASDInterface3 );
  if ( vRet != DEVICE_OK )
  {
    return vRet;
  }

  LogMessage( "Creating Dichroic Mirror" );
  // Dichroic mirror component
  vRet = CreateDichroicMirror( vASDInterface );
  if ( vRet != DEVICE_OK )
  {
    return vRet;
  }

  LogMessage( "Creating Filter Wheel 1" );
  // Filter wheel 1 component
  vRet = CreateFilterWheel( vASDInterface, FilterWheel1_, WheelIndex1, ERR_FILTERWHEEL1_INIT );
  if ( vRet != DEVICE_OK )
  {
    return vRet;
  }

  LogMessage( "Creating Filter Wheel 2" );
  // Filter wheel 2 component
  vRet = CreateFilterWheel( vASDInterface, FilterWheel2_, WheelIndex2, ERR_FILTERWHEEL2_INIT );
  if ( vRet != DEVICE_OK )
  {
    return vRet;
  }

  LogMessage( "Creating Disk" );
  // Disk component
  vRet = CreateDisk( vASDInterface );
  if ( vRet != DEVICE_OK )
  {
    return vRet;
  }

  LogMessage( "Creating Confocal Mode" );
  // Confocal mode component
  vRet = CreateConfocalMode( vASDInterface3 );
  if ( vRet != DEVICE_OK )
  {
    return vRet;
  }

  LogMessage( "Creating Aperture" );
  // Aperture component
  vRet = CreateAperture( vASDInterface2 );
  if ( vRet != DEVICE_OK )
  {
    return vRet;
  }

  LogMessage( "Creating Camera Port Mirror" );
  // Camera port mirror component
  vRet = CreateCameraPortMirror( vASDInterface2 );
  if ( vRet != DEVICE_OK )
  {
    return vRet;
  }

  LogMessage( "Creating Lenses" );
  // Lens components
  for ( int vLensIndex = lt_Lens1; vLensIndex < lt_LensMax; ++vLensIndex )
  {
    vRet = CreateLens( vASDInterface2, vLensIndex );
    if ( vRet != DEVICE_OK )
    {
      return vRet;
    }
  }

  LogMessage( "Creating Power Density" );
  // Power density components
  for ( int vLensIndex = lt_Lens1; vLensIndex < lt_LensMax; ++vLensIndex )
  {
    vRet = CreatePowerDensity( vASDInterface3, vLensIndex );
    if ( vRet != DEVICE_OK )
    {
      return vRet;
    }
  }

  LogMessage( "Creating Super Resolution" );
  // Super Resolution component
  vRet = CreateSuperRes( vASDInterface3 );
  if ( vRet != DEVICE_OK )
  {
    return vRet;
  }

  LogMessage( "Creating TIRF" );
  // TIRF component
  vRet = CreateTIRF( vASDInterface3 );
  if ( vRet != DEVICE_OK )
  {
    return vRet;
  }

  return DEVICE_OK;
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
    string vMessage( "Error loading the ASD Status. Caught Exception with message: " );
    vMessage += vException.what();
    LogMessage( vMessage );
    vErrorCode = ERR_DRAGONFLYSTATUS_INIT;
  }
  return vErrorCode;
}

int CDragonfly::CreateDichroicMirror( IASDInterface* ASDInterface )
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
      LogMessage( "Dichroic mirror not detected" );
    }
  }
  catch ( exception& vException )
  {
    string vMessage( "Error loading the Dichroic mirror. Caught Exception with message: " );
    vMessage += vException.what();
    LogMessage( vMessage );
    return ERR_DICHROICMIRROR_INIT;
  }
  return DEVICE_OK;
}

int CDragonfly::CreateFilterWheel( IASDInterface* ASDInterface, CFilterWheel*& FilterWheel, TWheelIndex WheelIndex, unsigned int ErrorCode )
{
  try
  {
    IFilterWheelInterface* vASDFilterWheel = ASDInterface->GetFilterWheel( WheelIndex );
    if ( vASDFilterWheel != nullptr )
    {
      FilterWheel = new CFilterWheel( WheelIndex, vASDFilterWheel, DragonflyStatus_, this );
    }
    else
    {
      LogMessage( "Filter wheel " + to_string( WheelIndex ) + " not detected" );
    }
  }
  catch ( exception& vException )
  {
    string vMessage( "Error loading the filter wheel " + to_string( WheelIndex ) + ". Caught Exception with message: " );
    vMessage += vException.what();
    LogMessage( vMessage );
    return ErrorCode;
  }
  return DEVICE_OK;
}

int CDragonfly::CreateDisk( IASDInterface* ASDInterface )
{
  try
  {
    IDiskInterface2* vASDDisk = ASDInterface->GetDisk_v2();
    if ( vASDDisk != nullptr )
    {
      Disk_ = new CDisk( vASDDisk, this );
    }
    else
    {
      LogMessage( "Spinning disk not detected" );
    }
  }
  catch ( exception& vException )
  {
    string vMessage( "Error loading the Spinning disk. Caught Exception with message: " );
    vMessage += vException.what();
    LogMessage( vMessage );
    return ERR_DISK_INIT;
  }
  return DEVICE_OK;
}

int CDragonfly::CreateConfocalMode( IASDInterface3* ASDInterface )
{
  try
  {
    IConfocalModeInterface3* vASDConfocalMode = ASDInterface->GetImagingMode();
    if ( vASDConfocalMode != nullptr )
    {
      ConfocalMode_ = new CConfocalMode( vASDConfocalMode, this );
    }
    else
    {
      LogMessage( "Confocal mode not detected" );
    }
  }
  catch ( exception& vException )
  {
    string vMessage( "Error loading the Confocal mode. Caught Exception with message: " );
    vMessage += vException.what();
    LogMessage( vMessage );
    return ERR_CONFOCALMODE_INIT;
  }
  return DEVICE_OK;
}

int CDragonfly::CreateAperture( IASDInterface2* ASDInterface )
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
      LogMessage( "Aperture not detected" );
    }
  }
  catch ( exception& vException )
  {
    string vMessage( "Error loading the Aperture. Caught Exception with message: " );
    vMessage += vException.what();
    LogMessage( vMessage );
    return ERR_APERTURE_INIT;
  }
  return DEVICE_OK;
}

int CDragonfly::CreateCameraPortMirror( IASDInterface2* ASDInterface )
{
  try
  {
    ICameraPortMirrorInterface* vCameraPortMirror = ASDInterface->GetCameraPortMirror();
    if ( vCameraPortMirror != nullptr )
    {
      CameraPortMirror_ = new CCameraPortMirror( vCameraPortMirror, this );
    }
    else
    {
      LogMessage( "Camera port mirror not detected" );
    }
  }
  catch ( exception& vException )
  {
    string vMessage( "Error loading the Camera port mirror. Caught Exception with message: " );
    vMessage += vException.what();
    LogMessage( vMessage );
    return ERR_CAMERAPORTMIRROR_INIT;
  }
  return DEVICE_OK;
}

int CDragonfly::CreateLens( IASDInterface2* ASDInterface, int LensIndex )
{
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
        LogMessage( "Lens " + to_string( LensIndex ) + " not detected" );
      }
    }
    catch ( exception& vException )
    {
      string vMessage( "Error loading the Lens " + to_string( LensIndex ) + ". Caught Exception with message: " );
      vMessage += vException.what();
      LogMessage( vMessage );
      return ERR_LENS_INIT;
    }
  }
  return DEVICE_OK;
}

int CDragonfly::CreatePowerDensity( IASDInterface3* ASDInterface, int LensIndex )
{
  LogMessage( "CREATE POWER DENSITY: Is Ill Lens Available for Lens " + to_string(LensIndex) );
  if ( ASDInterface->IsIllLensAvailable( (TLensType)LensIndex ) )
  {
    try
    {
      LogMessage( "CREATE POWER DENSITY: GetIllLens" );
      IIllLensInterface* vIllLensInterface = ASDInterface->GetIllLens( (TLensType)LensIndex );
      if ( vIllLensInterface != nullptr )
      {
        LogMessage( "CREATE POWER DENSITY: New CPowerDensity" );
        CPowerDensity* vPowerDensity = new CPowerDensity( vIllLensInterface, LensIndex, this );
        if ( vPowerDensity != nullptr )
        {
          PowerDensity_.push_back( vPowerDensity );
        }
      }
      else
      {
        LogMessage( "Power density " + to_string( LensIndex ) + " not detected" );
      }
    }
    catch ( exception& vException )
    {
      string vMessage( "Error loading the Power density " + to_string( LensIndex ) + ". Caught Exception with message: " );
      vMessage += vException.what();
      LogMessage( vMessage );
      return ERR_POWERDENSITY_INIT;
    }
  }
  else
  {
    LogMessage( "CREATE POWER DENSITY: Ill Lens NOT Available" );
  }
  return DEVICE_OK;
}

int CDragonfly::CreateSuperRes( IASDInterface3* ASDInterface )
{
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
        LogMessage( "Super resolution not detected" );
      }
    }
    catch ( exception& vException )
    {
      string vMessage( "Error loading the Super resolution. Caught Exception with message: " );
      vMessage += vException.what();
      LogMessage( vMessage );
      return ERR_SUPERRES_INIT;
    }
  }
  return DEVICE_OK;
}

int CDragonfly::CreateTIRF( IASDInterface3* ASDInterface )
{
  if ( ASDInterface->IsTIRFAvailable() )
  {
    try
    {
      ITIRFInterface* vTIRF = ASDInterface->GetTIRF();
      if ( vTIRF != nullptr )
      {
        TIRF_ = new CTIRF( vTIRF, this );
      }
      else
      {
        LogMessage( "TIRF not detected" );
      }
    }
    catch ( exception& vException )
    {
      string vMessage( "Error loading TIRF. Caught Exception with message: " );
      vMessage += vException.what();
      LogMessage( vMessage );
      return ERR_TIRF_INIT;
    }
  }
  return DEVICE_OK;
}

void CDragonfly::LogComponentMessage( const std::string& Message )
{
  LogMessage( Message );
}
