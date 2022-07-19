///////////////////////////////////////////////////////////////////////////////
// FILE:          IntegratedLaserEngine.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   IntegratedLaserEngine controller adapter
//
// Based off the AndorLaserCombiner adapter from Karl Hoover, UCSF
//

#include "ALC_REV.h"
#include "IntegratedLaserEngine.h"
#include "SingleILE.h"
#include "DualILE.h"
#include "ILEWrapper/ILEWrapper.h"
#include "Lasers.h"
#include "VeryLowPower.h"


// Properties
const char* const g_DeviceListProperty = "Device";
const char* const g_ResetDeviceProperty = "Interlock Reset";

// Property values
const char* const g_Undefined = "Undefined";
const char* const g_PropertyOn = "On";
const char* const g_PropertyOff = "Off";


///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////

MODULE_API void InitializeModuleData()
{
  RegisterDevice( CSingleILE::g_DeviceName, MM::ShutterDevice, CSingleILE::g_DeviceDescription );
  RegisterDevice( CDualILE::g_DualDeviceName, MM::ShutterDevice, CDualILE::g_DualDeviceDescription );
  RegisterDevice( CDualILE::g_Dual700DeviceName, MM::ShutterDevice, CDualILE::g_Dual700DeviceDescription );
}

MODULE_API MM::Device* CreateDevice(const char* DeviceName)
{
  if ( DeviceName == 0 )
  {
    return 0;
  }

  if ( ( strcmp( DeviceName, CSingleILE::g_DeviceName ) == 0 ) )
  {
    // Single ILE
    CSingleILE* vIntegratedLaserEngine = new CSingleILE();
    return vIntegratedLaserEngine;
  }

  if ( ( strcmp( DeviceName, CDualILE::g_DualDeviceName ) == 0 ) )
  {
    // Dual ILE
    CDualILE* vIntegratedLaserEngine = new CDualILE( false );
    return vIntegratedLaserEngine;
  }

  if ( ( strcmp( DeviceName, CDualILE::g_Dual700DeviceName ) == 0 ) )
  {
    // Dual ILE 700
    CDualILE* vIntegratedLaserEngine = new CDualILE( true );
    return vIntegratedLaserEngine;
  }

  return 0;
}

MODULE_API void DeleteDevice(MM::Device* Device)
{
  delete Device;
}

///////////////////////////////////////////////////////////////////////////////
// Controller implementation
///////////////////////////////////////////////////////////////////////////////

CIntegratedLaserEngine::CIntegratedLaserEngine( const std::string& Description, int NbDevices ) :
  Initialized_( false ),
  ChangedTime_( 0.0 ),
  ILEWrapper_( nullptr ),
  ILEDevice_( nullptr ),
  Lasers_( nullptr ),
  VeryLowPower_( nullptr ),
  ResetDeviceProperty_( nullptr ),
  ConstructionReturnCode_( DEVICE_OK ),
  ClassIVInterlockActive_( false ),
  KeyInterlockActive_( false )
{
  if ( NbDevices <= 0 )
  {
    throw std::logic_error( "Number of requested ILE devices null or negative" );
  }

  InitializeDefaultErrorMessages();

#ifdef _M_X64
  SetErrorText( ERR_LIBRARY_LOAD, "Failed to load the ILE library. Make sure AB_ALC_REV64.dll is present in the Micro-Manager root directory and it is version 1.1.1.1 or later." );
#else
  SetErrorText( ERR_LIBRARY_LOAD, "Failed to load the Dragonfly library. Make sure AB_ALC_REV.dll is present in the Micro-Manager root directory and it is version 1.1.1.1 or later." );
#endif
  SetErrorText( ERR_PORTS_INIT, "Ports initialisation failed" );
  SetErrorText( ERR_ACTIVEBLANKING_INIT, "Active Blanking initialisation failed" );
  SetErrorText( ERR_LOWPOWERMODE_INIT, "Low Power mode initialisation failed" );
  SetErrorText( ERR_LASERS_INIT, "Lasers initialisation failed" );
  SetErrorText( ERR_VERYLOWPOWER_INIT, "Very Low Power initialisation failed" );
  SetErrorText( ERR_DEVICE_INDEXINVALID, "Device index invalid" );
  SetErrorText( ERR_DEVICE_CONNECTIONFAILED, "Connection to the device failed" );
  SetErrorText( ERR_DEVICE_RECONNECTIONFAILED, "Error occured during the reconnection with the device. Please try again or reload the configuration." );

  SetErrorText( ERR_LASER_STATE_READ, "Reading laser state failed" );
  SetErrorText( ERR_INTERLOCK, "Interlock triggered" );
  SetErrorText( ERR_CLASSIV_INTERLOCK, "Class IV interlock triggered" );
  SetErrorText( ERR_KEY_INTERLOCK, "Key interlock triggered" );
  SetErrorText( ERR_DEVICE_NOT_CONNECTED, "Device not connected" );
  SetErrorText( ERR_LASER_SET, "Setting laser power failed" );
  SetErrorText( ERR_SETCONTROLMODE, "Setting control mode failed" );
  SetErrorText( ERR_SETLASERSHUTTER, "Failed to open or close the laser shutter!" );

  SetErrorText( ERR_ACTIVEBLANKING_SET, "Setting active blanking failed" );
  SetErrorText( ERR_ACTIVEBLANKING_GETNBLINES, "Getting the number of lines for Active Blanking failed" );
  SetErrorText( ERR_ACTIVEBLANKING_GETSTATE, "Getting Active Blanking state failed" );

  SetErrorText( ERR_LOWPOWERMODE_SET, "Setting low power mode failed" );
  SetErrorText( ERR_LOWPOWERMODE_GET, "Getting low power mode state failed" );

  SetErrorText( ERR_PORTS_SET, "Setting port failed" );
  SetErrorText( ERR_PORTS_GET, "Getting current port index failed" );

  SetErrorText( ERR_VERYLOWPOWER_SET, "Setting very low power failed" );
  SetErrorText( ERR_VERYLOWPOWER_GET, "Getting very low power state failed" );

  // Load the library
  try
  {
    ILEWrapper_ = LoadILEWrapper( this );
  }
  catch ( std::exception& vException )
  {
    ILEWrapper_ = nullptr;
    std::string vMessage = "Error loading the ILE library. Caught Exception with message: ";
    vMessage += vException.what();
    LogMessage( vMessage );
    ConstructionReturnCode_ = ERR_LIBRARY_LOAD;
  }

  // Create pre-initialization properties:
  // -------------------------------------

  // Description
  CreateStringProperty( MM::g_Keyword_Description, Description.c_str(), true );
 
  // Devices
  if ( ILEWrapper_ != nullptr )
  {
    ILEWrapper_->GetListOfDevices( DeviceList_ );
    DevicesNames_.assign( NbDevices, g_Undefined );
    if ( NbDevices == 1 )
    {
      CreateDeviceSelectionProperty( 0, 0 );
    }
    else
    {
      for ( int vDeviceIndex = 0; vDeviceIndex < NbDevices; ++vDeviceIndex )
      {
        CreateDeviceSelectionProperty( vDeviceIndex + 1, vDeviceIndex );
      }
    }
  }
  
  EnableDelay(); // Signals that the delay setting will be used
  UpdateStatus();
}

CIntegratedLaserEngine::~CIntegratedLaserEngine()
{
  // Unload the library
  UnloadILEWrapper();
}

void CIntegratedLaserEngine::CreateDeviceSelectionProperty( int DeviceID, int DeviceIndex )
{
  std::string vInitialDevice = g_Undefined;
  if ( !DeviceList_.empty() )
  {
    vInitialDevice = *( DeviceList_.begin() );
  }
  std::string vPropertyName = g_DeviceListProperty;
  if ( DeviceID > 0 )
  {
    vPropertyName += std::to_string( static_cast<long long>( DeviceID ) );
  }
  CPropertyActionEx* pAct = new CPropertyActionEx( this, &CIntegratedLaserEngine::OnDeviceChange, DeviceIndex );
  CreateStringProperty( vPropertyName.c_str(), vInitialDevice.c_str(), false, pAct, true );
  std::vector<std::string> vDevices;
  CILEWrapper::TDeviceList::const_iterator vDeviceIt = DeviceList_.begin();
  while ( vDeviceIt != DeviceList_.end() )
  {
    vDevices.push_back( *vDeviceIt );
    ++vDeviceIt;
  }
  SetAllowedValues( vPropertyName.c_str(), vDevices );
  DevicesNames_[DeviceIndex] = vInitialDevice;
}

bool CIntegratedLaserEngine::Busy()
{
  MM::MMTime vInterval = GetCurrentMMTime() - ChangedTime_;
  MM::MMTime vDelay( GetDelayMs()*1000.0 );
  if ( vInterval < vDelay )
  {
    return true;
  }
  else
  {
    return false;
  }
}

void CIntegratedLaserEngine::GetName(char* Name) const
{
  CDeviceUtils::CopyLimitedString( Name, GetDeviceName().c_str() );
}

int CIntegratedLaserEngine::Initialize()
{
  if ( ConstructionReturnCode_ != DEVICE_OK )
  {
    return ConstructionReturnCode_;
  }
  if ( Initialized_ )
  {
    return DEVICE_OK;
  }
  
  // Connecting to the device
  try
  {
    if ( !CreateILE() )
    {
      LogMessage( "CreateILE failed" );
      return DEVICE_NOT_CONNECTED;
    }
  }
  catch ( std::string& exs )
  {
    LogMessage( exs.c_str() );
    SetErrorText( DEVICE_LOCALLY_DEFINED_ERROR, exs.c_str() );
    //CodeUtility::DebugOutput(exs.c_str());
    return DEVICE_LOCALLY_DEFINED_ERROR;
  }

  // Reset device property
  CPropertyAction* vAct = new CPropertyAction( this, &CIntegratedLaserEngine::OnResetDevice );
  CreateStringProperty( g_ResetDeviceProperty, g_PropertyOff, true, vAct );
  std::vector<std::string> vEnabledValues;
  vEnabledValues.push_back( g_PropertyOn );
  vEnabledValues.push_back( g_PropertyOff );
  SetAllowedValues( g_ResetDeviceProperty, vEnabledValues );

  // Lasers
  int vRet = InitalizeLasers();
  if( vRet != DEVICE_OK )
  {
    return vRet;
  }

  // Very Low Power
  vRet = InitializeVeryLowPower();
  if( vRet != DEVICE_OK )
  {
    return vRet;
  }
  
  // Ports
  vRet = InitializePorts();
  if ( vRet != DEVICE_OK )
  {
    return vRet;
  }

  // Active Blanking
  vRet = InitializeActiveBlanking();
  if ( vRet != DEVICE_OK )
  {
    return vRet;
  }

  // Low Power Mode
  vRet = InitializeLowPowerMode();
  if ( vRet != DEVICE_OK )
  {
    return vRet;
  }

  Initialized_ = true;
  return DEVICE_OK;
}

int CIntegratedLaserEngine::InitalizeLasers()
{
  IALC_REV_ILEPowerManagement* vLowPowerMode = ILEWrapper_->GetILEPowerManagementInterface( ILEDevice_ );
  IALC_REV_Laser2* vLaserInterface = ILEDevice_->GetLaserInterface2();
  IALC_REV_ILE* vILE = ILEDevice_->GetILEInterface();
  if( vLaserInterface != nullptr && vLowPowerMode != nullptr && vILE != nullptr )
  {
    try
    {
      Lasers_ = new CLasers( vLaserInterface, vLowPowerMode, vILE, this );
    }
    catch( std::exception& vException )
    {
      std::string vMessage( "Error loading the Lasers. Caught Exception with message: " );
      vMessage += vException.what();
      LogMessage( vMessage );
      return ERR_LASERS_INIT;
    }
  }
  else
  {
    if( vLaserInterface == nullptr )
    {
      LogMessage( "Laser interface pointer invalid" );
    }
    if( vLowPowerMode == nullptr )
    {
      LogMessage( "Low power interface pointer invalid" );
    }
    if( vILE == nullptr )
    {
      LogMessage( "ILE interface pointer invalid" );
    }
  }
  return DEVICE_OK;
}

int CIntegratedLaserEngine::InitializeVeryLowPower()
{
  IALC_REV_ILEPowerManagement* vLowPowerMode = ILEWrapper_->GetILEPowerManagementInterface( ILEDevice_ );
  if( vLowPowerMode != nullptr )
  {
    bool vVeryLowPowerPresent = false;
    if( !vLowPowerMode->IsCoherenceModePresent( &vVeryLowPowerPresent ) )
    {
      LogMessage( "ILE Power IsCoherenceModePresent failed" );
      return ERR_VERYLOWPOWER_INIT;
    }
    if( vVeryLowPowerPresent )
    {
      try
      {
        VeryLowPower_ = new CVeryLowPower( vLowPowerMode, this );
      }
      catch( std::exception& vException )
      {
        std::string vMessage( "Error loading Very Low Power. Caught Exception with message: " );
        vMessage += vException.what();
        LogMessage( vMessage );
        return ERR_VERYLOWPOWER_INIT;
      }
    }
    else
    {
      LogMessage( "Very Low Power not present", true );
    }
  }
  else
  {
    LogMessage( "ILE Power interface pointer invalid" );
  }
  return DEVICE_OK;
}

int CIntegratedLaserEngine::Shutdown()
{
  if ( ConstructionReturnCode_ != DEVICE_OK )
  {
    return DEVICE_OK;
  }
  if ( !Initialized_ )
  {
    return DEVICE_OK;
  }

  delete VeryLowPower_;
  VeryLowPower_ = nullptr;
  delete Lasers_;
  Lasers_ = nullptr;
  DeleteILE();
  
  Initialized_ = false;
  return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// Action interface
///////////////////////////////////////////////////////////////////////////////

int CIntegratedLaserEngine::OnDeviceChange( MM::PropertyBase* Prop, MM::ActionType Act, long DeviceIndex )
{
  if ( DeviceIndex >= DevicesNames_.size() )
  {
    return ERR_DEVICE_INDEXINVALID;
  }
  if ( Act == MM::BeforeGet )
  {
    Prop->Set( DevicesNames_[DeviceIndex].c_str() );    
  }
  else if ( Act == MM::AfterSet )
  {
    Prop->Get( DevicesNames_[DeviceIndex] );
  }
  return DEVICE_OK;
}

int CIntegratedLaserEngine::OnResetDevice( MM::PropertyBase* Prop, MM::ActionType Act )
{
  if ( ResetDeviceProperty_ == nullptr )
  {
    ResetDeviceProperty_ = Prop;
    if( GetClassIVAndKeyInterlockStatus() != DEVICE_OK )
    {
      ActivateInterlock();
    }
  }
  int vRet = DEVICE_OK;
  if ( Act == MM::BeforeGet )
  {
    // Displaying "off" since the only time when the property is "on" is while the reset happens
    // If we are here then we are not going through a reset so making sure we set the value back to "off" is a good idea (in case reset failed)
    Prop->Set( g_PropertyOff );
  }
  else if ( Act == MM::AfterSet )
  {
    std::string vValue;
    Prop->Get( vValue );
    if ( vValue == g_PropertyOn )
    {
      // Disconnect from the ILE interface
      LogMessage( "Disconnecting from the ILE interface", true );
      DisconnectILEInterfaces();
      if( VeryLowPower_ != nullptr )
      {
        VeryLowPower_->UpdateILEInterface( nullptr );
      }
      if( Lasers_ != nullptr )
      {
        Lasers_->UpdateILEInterface( nullptr, nullptr, nullptr );
      }

      // Disconnect the device
      LogMessage( "Disconnecting ILE", true );
      DeleteILE();

      // Reconnect the device
      LogMessage( "Reconnecting ILE", true );
      try
      {
        if ( !CreateILE() )
        {
          LogMessage( "CreateILE failed" );
          return ERR_DEVICE_CONNECTIONFAILED;
        }
      }
      catch ( std::string& exs )
      {
        vRet = DEVICE_LOCALLY_DEFINED_ERROR;
        LogMessage( exs.c_str() );
        SetErrorText( DEVICE_LOCALLY_DEFINED_ERROR, exs.c_str() );
        //CodeUtility::DebugOutput(exs.c_str());
        return vRet;
      }

      // Reconnect to ILE interface
      LogMessage( "Reconnecting to the ILE interface", true );
      vRet = ReconnectLasers();
      if ( vRet == DEVICE_OK )
      {
        vRet = ReconnectILEInterfaces();
      }
      if( vRet == DEVICE_OK )
      {
        // Handling very low power last since it enforces an update of the laser range
        vRet = ReconnectVeryLowPower();
      }
      if ( vRet == DEVICE_OK )
      {
        ClassIVInterlockActive_ = false;
        KeyInterlockActive_ = false;
        Prop->Set( g_PropertyOff );
        MM::Property* pChildProperty = ( MM::Property* )Prop;
        pChildProperty->SetReadOnly( true );
      }
    }
  }
  return vRet;
}

int CIntegratedLaserEngine::ReconnectLasers()
{
  int vRet = DEVICE_OK;
  if( Lasers_ != nullptr )
  {
    IALC_REV_ILEPowerManagement* vLowPowerMode = ILEWrapper_->GetILEPowerManagementInterface( ILEDevice_ );
    IALC_REV_Laser2* vLaserInterface = ILEDevice_->GetLaserInterface2();
    IALC_REV_ILE* vILE = ILEDevice_->GetILEInterface();
    if( vLowPowerMode != nullptr && vLaserInterface != nullptr && vILE != nullptr )
    {
      vRet = Lasers_->UpdateILEInterface( vLaserInterface, vLowPowerMode, vILE );
    }
    else
    {
      if( vLaserInterface == nullptr )
      {
        LogMessage( "Laser interface pointer invalid" );
      }
      if( vLowPowerMode == nullptr )
      {
        LogMessage( "Low power interface pointer invalid" );
      }
      if( vILE == nullptr )
      {
        LogMessage( "ILE interface pointer invalid" );
      }
      vRet = ERR_DEVICE_RECONNECTIONFAILED;
    }
  }
  return vRet;
}

int CIntegratedLaserEngine::ReconnectVeryLowPower()
{
  int vRet = DEVICE_OK;
  if( VeryLowPower_ != nullptr )
  {
    IALC_REV_ILEPowerManagement* vLowPowerMode = ILEWrapper_->GetILEPowerManagementInterface( ILEDevice_ );
    if( vLowPowerMode != nullptr )
    {
      vRet = VeryLowPower_->UpdateILEInterface( vLowPowerMode );
    }
    else
    {
      LogMessage( "ILE Power interface pointer invalid" );
      vRet = ERR_DEVICE_RECONNECTIONFAILED;
    }
  }
  return vRet;
}

///////////////////////////////////////////////////////////////////////////////
// Shutter API
///////////////////////////////////////////////////////////////////////////////

int CIntegratedLaserEngine::SetOpen(bool Open)
{
  if ( Lasers_ != nullptr )
  {
    return Lasers_->SetOpen( Open );
  }
  return DEVICE_OK;
}

int CIntegratedLaserEngine::GetOpen(bool& Open)
{
  Open = false;
  if ( Lasers_ != nullptr )
  {
    return Lasers_->GetOpen( Open );
  }
  return DEVICE_OK;
}

/**
 * ON for DeltaT milliseconds.  Other implementations of Shutter don't
 * implement this.  Is this perhaps because this blocking call is not
 * appropriate?
 */
int CIntegratedLaserEngine::Fire(double DeltaT)
{
  int vRet = SetOpen( true );
  if( vRet == DEVICE_OK )
  {
    CDeviceUtils::SleepMs( ( long )( DeltaT + .5 ) );
    vRet = SetOpen( false );
  }
  return vRet;
}

///////////////////////////////////////////////////////////////////////////////
// Public helper functions
///////////////////////////////////////////////////////////////////////////////

void CIntegratedLaserEngine::LogMMMessage( std::string Message, bool DebugOnly )
{
  LogMessage( Message, DebugOnly );
}

MM::MMTime CIntegratedLaserEngine::GetCurrentTime()
{
  return GetCurrentMMTime();
}

void CIntegratedLaserEngine::CheckAndUpdateLasers()
{
  if ( Lasers_ != nullptr )
  {
    Lasers_->CheckAndUpdateLasers();
  }
}

void CIntegratedLaserEngine::ActivateInterlock()
{
  LogMessage( "Modifying Reset property state", true );
  if ( ResetDeviceProperty_ != nullptr )
  {
    MM::Property* pChildProperty = ( MM::Property* )ResetDeviceProperty_;
    pChildProperty->SetReadOnly( false );
  }
}

void CIntegratedLaserEngine::ActiveClassIVInterlock()
{
  LogMessage( "Class IV interlock detected", true );
  ClassIVInterlockActive_ = true;
  ActivateInterlock();
  LogMessage( "Class IV interlock activated", true );
}

void CIntegratedLaserEngine::ActiveKeyInterlock()
{
  LogMessage( "Key interlock detected", true );
  KeyInterlockActive_ = true;
  ActivateInterlock();
  LogMessage( "Key interlock activated", true );
}

int CIntegratedLaserEngine::GetClassIVAndKeyInterlockStatus()
{
  if ( ClassIVInterlockActive_ )
  {
    return ERR_CLASSIV_INTERLOCK;    
  }
  if ( KeyInterlockActive_ ) 
  {
    return ERR_KEY_INTERLOCK;
  }
  return DEVICE_OK;
}

void CIntegratedLaserEngine::UpdatePropertyUI( const char* PropertyName, const char* PropertyValue )
{
  GetCoreCallback()->OnPropertyChanged( this, PropertyName, PropertyValue );
}
