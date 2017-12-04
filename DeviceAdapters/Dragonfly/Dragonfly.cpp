///////////////////////////////////////////////////////////////////////////////
// FILE:          Dragonfly.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------

#include "../../MMDevice/ModuleInterface.h"
#include "Dragonfly.h"
#include "DichroicMirror.h"

#include "ASDWrapper.h"
#include "ASDInterface.h"

#include <string>
#include <stdexcept>
#include <vector>

using namespace std;

const char * const g_DeviceName = "Dragonfly";
const char * const g_DeviceDescription = "Andor Dragonfly Device Adapter";
const char * const g_DeviceSerialNumber = "Serial Number";
const char * const g_DeviceProductID = "Product ID";
const char * const g_DeviceSoftwareVersion = "Software Version";

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
  DeviceConnected_( false )
{
  InitializeDefaultErrorMessages();

  string vContactSupportMessage( "Please contact Andor support and send the latest file present in the Micro-Manager CoreLogs directory." );
#ifdef _M_X64
  SetErrorText( ERR_LIBRARY_LOAD, "Failed to load the ASD library. Make sure AB_ASDx64.dll is present in the Micro-Manager root directory." );
#else
  SetErrorText( ERR_LIBRARY_LOAD, "Failed to load the ASD library. Make sure AB_ASD.dll is present in the Micro-Manager root directory." );
#endif
  SetErrorText( ERR_LIBRARY_INIT, "ASD Library initialisation failed. Make sure the device is connected." );
  string vMessage = "Dichroic Mirror initialisation failed. " + vContactSupportMessage;
  SetErrorText( ERR_DICHROICMIRROR_INIT, vMessage.c_str() );

  // Connect to ASD wrapper
  try
  {
    ASDWrapper_ = new CASDWrapper();
    ASDLibraryConnected_ = true;
  }
  catch ( exception& vException )
  {
    vMessage = "Error loading the ASD library. Caught Exception with message: ";
    vMessage += vException.what();
    LogMessage( vMessage );
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

  // COM Port property
  vRet = CreatePropertyWithHandler( MM::g_Keyword_Port, "Undefined", MM::String, false, &CDragonfly::OnPort );
  if ( vRet != DEVICE_OK )
  {
    return vRet;
  }

  for ( int i = 1;i <= 20;++i )
  {
    string vCOMPort;
    if ( i < 10 )
    {
      vCOMPort = " ";
    }
    vCOMPort += "COM" + to_string( i );
    AddAllowedValue( MM::g_Keyword_Port, vCOMPort.c_str() );
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

  if ( Initialized_ )
  {
    Initialized_ = false;
  }

  delete DichroicMirror_;
  Disconnect();

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
    if ( !DeviceConnected_ )
    {
      string vNewPort;
      Prop->Get( vNewPort );
      // remove the leading space from the port name if there's any before connecting
      size_t vFirstValidCharacter = vNewPort.find_first_not_of( " " );
      if ( Connect( vNewPort.substr( vFirstValidCharacter ) ) == DEVICE_OK )
      {
        Port_ = vNewPort;
        vRet = InitializeComponents();
      }
    }
    else
    {
      Prop->Set( Port_.c_str() );
    }
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
  DeviceConnected_ = true;

  return DEVICE_OK;
}

int CDragonfly::Disconnect()
{
  if ( ASDLoader_ != nullptr )
  {
    ASDWrapper_->DeleteASDLoader( ASDLoader_ );
    ASDLoader_ = nullptr;
  }
  DeviceConnected_ = false;
  return DEVICE_OK;
}

int CDragonfly::InitializeComponents()
{
  IASDInterface* vASDInterface = ASDLoader_->GetASDInterface();

  // Serial number property
  string vSerialNumber = vASDInterface->GetSerialNumber();
  int vRet = CreateProperty( g_DeviceSerialNumber, vSerialNumber.c_str(), MM::String, true );
  if ( vRet != DEVICE_OK )
  {
    return vRet;
  }

  // Product ID property
  string vProductID = vASDInterface->GetProductID();
  vRet = CreateProperty( g_DeviceProductID, vProductID.c_str(), MM::String, true );
  if ( vRet != DEVICE_OK )
  {
    return vRet;
  }

  // Software version property
  string vSoftwareVersion = vASDInterface->GetSoftwareVersion();
  vRet = CreateProperty( g_DeviceSoftwareVersion, vSoftwareVersion.c_str(), MM::String, true );
  if ( vRet != DEVICE_OK )
  {
    return vRet;
  }

  try
  {
    IDichroicMirrorInterface* vASDDichroicMirror = vASDInterface->GetDichroicMirror();
    if ( vASDDichroicMirror != nullptr )
    {
      DichroicMirror_ = new CDichroicMirror( vASDDichroicMirror, this );
    }
    else
    {
      LogMessage( "Dichroic mirror not detected" );
    }
  }
  catch( exception& vException )
  {
    string vMessage( "Error loading the dichroic mirror. Caught Exception with message: " );
    vMessage += vException.what();
    LogMessage( vMessage );
    return ERR_DICHROICMIRROR_INIT;
  }
  
  return DEVICE_OK;
}

void CDragonfly::LogComponentMessage( const std::string& Message )
{
  LogMessage( Message );
}
