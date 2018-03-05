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
#include "ILEWrapper/ILEWrapper.h"
#include "Ports.h"
#include "ActiveBlanking.h"
#include "LowPowerMode.h"
#include "Lasers.h"

// Properties
const char* const g_DeviceName = "IntegratedLaserEngine";
const char* const g_DeviceDescription = "Integrated Laser Engine";
const char* const g_DeviceListProperty = "Device";
const char* const g_ResetDeviceProperty = "Reset device connection";

// Property values
const char* const g_PropertyOn = "On";
const char* const g_PropertyOff = "Off";


///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////

MODULE_API void InitializeModuleData()
{
  RegisterDevice( g_DeviceName, MM::ShutterDevice, g_DeviceDescription );
}

MODULE_API MM::Device* CreateDevice(const char* DeviceName)
{
  if ( DeviceName == 0 )
  {
    return 0;
  }

  if ( ( strcmp( DeviceName, g_DeviceName ) == 0 ) )
  {
    // create Controller
    CIntegratedLaserEngine* vIntegratedLaserEngine = new CIntegratedLaserEngine();
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

CIntegratedLaserEngine::CIntegratedLaserEngine() :
  Initialized_( false ),
  ChangedTime_( 0.0 ),
  ILEWrapper_( nullptr ),
  ILEDevice_( nullptr ),
  Ports_( nullptr ),
  ActiveBlanking_( nullptr ),
  LowPowerMode_( nullptr ),
  Lasers_( nullptr ),
  ResetDeviceProperty_( nullptr )

{
  // Load the library
  ILEWrapper_ = LoadILEWrapper( this );

  InitializeDefaultErrorMessages();

  SetErrorText( ERR_PORTS_INIT, "Ports initialisation failed" );
  SetErrorText( ERR_ACTIVEBLANKING_INIT, "Active Blanking initialisation failed" );
  SetErrorText( ERR_LOWPOWERMODE_INIT, "Low Power mode initialisation failed" );
  SetErrorText( ERR_LASERS_INIT, "Lasers initialisation failed" );
  SetErrorText( ERR_INTERLOCK, "Interlock triggered" );
  SetErrorText( ERR_CLASSIV_INTERLOCK, "Class IV interlock triggered" );
  SetErrorText( ERR_DEVICE_NOT_CONNECTED, "Device reconnecting. Please wait." );

  // Create pre-initialization properties:
  // -------------------------------------

  // Description
  CreateStringProperty( MM::g_Keyword_Description, g_DeviceDescription, true );

  // Devices
  ILEWrapper_->GetListOfDevices( DeviceList_ );
  std::string vInitialDevice = "Undefined";
  if ( !DeviceList_.empty() )
  {
    vInitialDevice = *( DeviceList_.begin() );
  }
  CPropertyAction* pAct = new CPropertyAction( this, &CIntegratedLaserEngine::OnDeviceChange );
  CreateStringProperty( g_DeviceListProperty, vInitialDevice.c_str(), false, pAct, true );
  std::vector<std::string> vDevices;
  CILEWrapper::TDeviceList::const_iterator vDeviceIt = DeviceList_.begin();
  while ( vDeviceIt != DeviceList_.end() )
  {
    vDevices.push_back( *vDeviceIt );
    ++vDeviceIt;
  }
  SetAllowedValues( g_DeviceListProperty, vDevices );
  
  EnableDelay(); // Signals that the delay setting will be used
  UpdateStatus();
  LogMessage( std::string( g_DeviceName ) + " ctor OK", true );
}

CIntegratedLaserEngine::~CIntegratedLaserEngine()
{
  Shutdown();
  // Unload the library
  UnloadILEWrapper();
  LogMessage( std::string( g_DeviceName ) + " dtor OK", true );
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
  CDeviceUtils::CopyLimitedString( Name, g_DeviceName );
}

int CIntegratedLaserEngine::Initialize()
{
  int vRet = DEVICE_OK;

  // Connecting to the device
  try
  {
    if ( !ILEWrapper_->CreateILE( &ILEDevice_, DeviceName_.c_str() ) )
    {
      LogMessage( "CreateILE failed" );
      return DEVICE_NOT_CONNECTED;
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

  // Reset device
  CPropertyAction* vAct = new CPropertyAction( this, &CIntegratedLaserEngine::OnResetDevice );
  CreateStringProperty( g_ResetDeviceProperty, g_PropertyOff, true, vAct );
  std::vector<std::string> vEnabledValues;
  vEnabledValues.push_back( g_PropertyOn );
  vEnabledValues.push_back( g_PropertyOff );
  SetAllowedValues( g_ResetDeviceProperty, vEnabledValues );


  // Lasers
  IALC_REV_ILEPowerManagement* vLowPowerMode = ILEWrapper_->GetILEPowerManagementInterface( ILEDevice_ );
  IALC_REV_Laser2* vLaserInterface = ILEDevice_->GetLaserInterface2();
  IALC_REV_ILE* vILE = ILEDevice_->GetILEInterface();
  if ( vLaserInterface != nullptr )
  {
    try
    {
      Lasers_ = new CLasers( vLaserInterface, vLowPowerMode, vILE, this );
    }
    catch ( std::exception& vException )
    {
      std::string vMessage( "Error loading the Lasers. Caught Exception with message: " );
      vMessage += vException.what();
      LogMessage( vMessage );
      return ERR_LASERS_INIT;
    }
  }
  else
  {
    LogMessage( "Laser interface pointer invalid" );
  }
  
  // Ports
  IALC_REV_Port* vPortInterface = ILEDevice_->GetPortInterface();
  if ( vPortInterface != nullptr )
  {
    try
    {
      Ports_ = new CPorts( vPortInterface, this );
    }
    catch ( std::exception& vException )
    {
      std::string vMessage( "Error loading the Ports. Caught Exception with message: " );
      vMessage += vException.what();
      LogMessage( vMessage );
      return ERR_PORTS_INIT;
    }
  }
  else
  {
    LogMessage( "Port interface pointer invalid" );
  }

  // Active Blanking
  IALC_REV_ILEActiveBlankingManagement* vActiveBlanking = ILEWrapper_->GetILEActiveBlankingManagementInterface( ILEDevice_ );
  if ( vActiveBlanking != nullptr )
  {
    bool vActiveBlankingPresent = false;
    if ( !vActiveBlanking->IsActiveBlankingManagementPresent( &vActiveBlankingPresent ) )
    {
      LogMessage( "Active Blanking IsActiveBlankingManagementPresent failed" );
      return ERR_ACTIVEBLANKING_INIT;
    }
    if ( vActiveBlankingPresent )
    {
      try
      {
        ActiveBlanking_ = new CActiveBlanking( vActiveBlanking, this );
      }
      catch ( std::exception& vException )
      {
        std::string vMessage( "Error loading Active Blanking. Caught Exception with message: " );
        vMessage += vException.what();
        LogMessage( vMessage );
        return ERR_ACTIVEBLANKING_INIT;
      }
    }
  }
  else
  {
    LogMessage( "Active Blanking interface pointer invalid" );
  }

  // Low Power Mode
  if ( vLowPowerMode != nullptr )
  {
    bool vLowPowerModePresent = false;
    if ( !vLowPowerMode->IsLowPowerPresent( &vLowPowerModePresent ) )
    {
      LogMessage( "ILE Power IsLowPowerPresent failed" );
      return ERR_LOWPOWERMODE_INIT;
    }
    if ( vLowPowerModePresent )
    {
      try
      {
        LowPowerMode_ = new CLowPowerMode( vLowPowerMode, this );
      }
      catch ( std::exception& vException )
      {
        std::string vMessage( "Error loading Low Power mode. Caught Exception with message: " );
        vMessage += vException.what();
        LogMessage( vMessage );
        return ERR_LOWPOWERMODE_INIT;
      }
    }
  }
  else
  {
    LogMessage( "ILE Power interface pointer invalid" );
  }

  Initialized_ = true;
  return DEVICE_OK;
}

int CIntegratedLaserEngine::Shutdown()
{
  if ( Initialized_ )
  {
    Initialized_ = false;
    delete LowPowerMode_;
    delete ActiveBlanking_;
    delete Ports_;
    delete Lasers_;
    ILEWrapper_->DeleteILE( ILEDevice_ );
  }
  return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// Action interface
///////////////////////////////////////////////////////////////////////////////

int CIntegratedLaserEngine::OnDeviceChange( MM::PropertyBase* Prop, MM::ActionType Act )
{
  if ( Act == MM::BeforeGet )
  {
    Prop->Set( DeviceName_.c_str() );    
  }
  else if ( Act == MM::AfterSet )
  {
    Prop->Get( DeviceName_ );
  }
  return DEVICE_OK;
}

int CIntegratedLaserEngine::OnResetDevice( MM::PropertyBase* Prop, MM::ActionType Act )
{
  if ( ResetDeviceProperty_ == nullptr )
  {
    ResetDeviceProperty_ = Prop;
  }
  int vRet = DEVICE_OK;
  if ( Act == MM::AfterSet )
  {
    std::string vValue;
    Prop->Get( vValue );
    if ( vValue == g_PropertyOn )
    {
      // Disconnect from the ILE interface
      Ports_->UpdateILEInterface( nullptr );
      ActiveBlanking_->UpdateILEInterface( nullptr );
      LowPowerMode_->UpdateILEInterface( nullptr );
      Lasers_->UpdateILEInterface( nullptr, nullptr, nullptr );

      // Disconnect the device
      ILEWrapper_->DeleteILE( ILEDevice_ );
      ILEDevice_ = nullptr;

      // Reconnect the device
      try
      {
        if ( !ILEWrapper_->CreateILE( &ILEDevice_, DeviceName_.c_str() ) )
        {
          LogMessage( "CreateILE failed" );
          return DEVICE_ERR;
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
      IALC_REV_Port* vPortInterface = ILEDevice_->GetPortInterface();
      IALC_REV_ILEActiveBlankingManagement* vActiveBlanking = ILEWrapper_->GetILEActiveBlankingManagementInterface( ILEDevice_ );
      IALC_REV_ILEPowerManagement* vLowPowerMode = ILEWrapper_->GetILEPowerManagementInterface( ILEDevice_ );
      IALC_REV_Laser2* vLaserInterface = ILEDevice_->GetLaserInterface2();
      IALC_REV_ILE* vILE = ILEDevice_->GetILEInterface();

      Ports_->UpdateILEInterface( vPortInterface );
      ActiveBlanking_->UpdateILEInterface( vActiveBlanking );
      LowPowerMode_->UpdateILEInterface( vLowPowerMode );
      Lasers_->UpdateILEInterface( vLaserInterface, vLowPowerMode, vILE );

      Prop->Set( g_PropertyOff );
      MM::Property* pChildProperty = ( MM::Property* )Prop;
      pChildProperty->SetReadOnly( true );
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
    Lasers_->GetOpen( Open );
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
  SetOpen( true );
  CDeviceUtils::SleepMs( (long)( DeltaT + .5 ) );
  SetOpen( false );
  return DEVICE_OK;
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

void CIntegratedLaserEngine::ActiveClassIVInterlock()
{
  if ( ResetDeviceProperty_ != nullptr )
  {
    MM::Property* pChildProperty = ( MM::Property* )ResetDeviceProperty_;
    pChildProperty->SetReadOnly( false );
  }
}

void CIntegratedLaserEngine::UpdatePropertyUI( const char* PropertyName, const char* PropertyValue )
{
  GetCoreCallback()->OnPropertyChanged( this, PropertyName, PropertyValue );
}