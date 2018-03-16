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
#include "PortsConfiguration.h"
#include "DualILEPorts.h"
#include "ALC_REV_ILE2.h"

// Properties
const char* const g_DeviceName = "Andor ILE";
const char* const g_DualDeviceName = "Andor Dual ILE";
const char* const g_Dual700DeviceName = "Andor Dual ILE 700";
const char* const g_DeviceDescription = "Integrated Laser Engine";
const char* const g_DualDeviceDescription = "Dual Integrated Laser Engine";
const char* const g_Dual700DeviceDescription = "Dual Integrated Laser Engine 700";
const char* const g_DeviceListProperty = "Device";
const char* const g_ResetDeviceProperty = "Reset device connection";
const char* const g_ILE700Property = "ILE 700 Device";
const char* const g_PortsConfigurationProperty = "Ports configuration";

// Property values
const char* const g_Undefined = "Undefined";
const char* const g_PropertyOn = "On";
const char* const g_PropertyOff = "Off";
const char* const g_PropertyYes = "Yes";
const char* const g_PropertyNo = "No";
const char* const g_PortConfigDragonfly = "Dragonfly";
const char* const g_PortConfigTest = "Test";


///////////////////////////////////////////////////////////////////////////////
// Exported MMDevice API
///////////////////////////////////////////////////////////////////////////////

MODULE_API void InitializeModuleData()
{
  RegisterDevice( g_DeviceName, MM::ShutterDevice, g_DeviceDescription );
  RegisterDevice( g_DualDeviceName, MM::ShutterDevice, g_DualDeviceDescription );
  RegisterDevice( g_Dual700DeviceName, MM::ShutterDevice, g_Dual700DeviceDescription );
}

MODULE_API MM::Device* CreateDevice(const char* DeviceName)
{
  if ( DeviceName == 0 )
  {
    return 0;
  }

  if ( ( strcmp( DeviceName, g_DeviceName ) == 0 ) )
  {
    // Single ILE
    CSingleILE* vIntegratedLaserEngine = new CSingleILE();
    return vIntegratedLaserEngine;
  }

  if ( ( strcmp( DeviceName, g_DualDeviceName ) == 0 ) )
  {
    // Dual ILE
    CDualILE* vIntegratedLaserEngine = new CDualILE( false );
    return vIntegratedLaserEngine;
  }

  if ( ( strcmp( DeviceName, g_Dual700DeviceName ) == 0 ) )
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
  ResetDeviceProperty_( nullptr )
{
  if ( NbDevices <= 0 )
  {
    throw std::logic_error( "Number of requested ILE devices null or negative" );
  }

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
  SetErrorText( ERR_ACTIVEBLANKING_SET, "Setting active blanking failed" );
  SetErrorText( ERR_DEVICE_INDEXINVALID, "Device index invalid" );
  SetErrorText( ERR_DEVICE_CONNECTIONFAILED, "Connection to the device failed" );
  SetErrorText( ERR_LOWPOWERMODE_SET, "Setting low power mode failed" );
  SetErrorText( ERR_PORTS_SET, "Setting port failed" );

  // Create pre-initialization properties:
  // -------------------------------------

  // Description
  CreateStringProperty( MM::g_Keyword_Description, Description.c_str(), true );
 
  // Devices
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
  
  EnableDelay(); // Signals that the delay setting will be used
  UpdateStatus();
  LogMessage( std::string( g_DeviceName ) + " ctor OK", true );
}

CIntegratedLaserEngine::~CIntegratedLaserEngine()
{
  // Unload the library
  UnloadILEWrapper();
  LogMessage( std::string( g_DeviceName ) + " dtor OK", true );
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
    vPropertyName += std::to_string( DeviceID );
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
  int vRet = InitializePorts();
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

int CIntegratedLaserEngine::Shutdown()
{
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
  }
  int vRet = DEVICE_OK;
  if ( Act == MM::AfterSet )
  {
    std::string vValue;
    Prop->Get( vValue );
    if ( vValue == g_PropertyOn )
    {
      // Disconnect from the ILE interface
      DisconnectILEInterfaces();
      Lasers_->UpdateILEInterface( nullptr, nullptr, nullptr );

      // Disconnect the device
      DeleteILE();

      // Reconnect the device
      try
      {
        if ( CreateILE() )
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
      IALC_REV_ILEPowerManagement* vLowPowerMode = ILEWrapper_->GetILEPowerManagementInterface( ILEDevice_ );
      IALC_REV_Laser2* vLaserInterface = ILEDevice_->GetLaserInterface2();
      IALC_REV_ILE* vILE = ILEDevice_->GetILEInterface();
      Lasers_->UpdateILEInterface( vLaserInterface, vLowPowerMode, vILE );
      ReconnectILEInterfaces();

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

///////////////////////////////////////////////////////////////////////////////
// SINGLE ILE
///////////////////////////////////////////////////////////////////////////////

CSingleILE::CSingleILE() :
  CIntegratedLaserEngine( g_DeviceDescription, 1 ),
  Ports_( nullptr ),
  ActiveBlanking_( nullptr ),
  LowPowerMode_( nullptr )
{
}

CSingleILE::~CSingleILE()
{
}

std::string CSingleILE::GetDeviceName() const
{
  return g_DeviceName;
}

int CSingleILE::Shutdown()
{
  delete LowPowerMode_;
  LowPowerMode_ = nullptr;
  delete ActiveBlanking_;
  ActiveBlanking_ = nullptr;
  delete Ports_;
  Ports_ = nullptr;
  return CIntegratedLaserEngine::Shutdown();
}

bool CSingleILE::CreateILE()
{
  bool vRet = false;
  if ( DevicesNames_.size() > 0 )
  {
    vRet = ILEWrapper_->CreateILE( &ILEDevice_, DevicesNames_[0].c_str() );
  }
  return vRet;
}

void CSingleILE::DeleteILE()
{
  ILEWrapper_->DeleteILE( ILEDevice_ );
  ILEDevice_ = nullptr;
}

void CSingleILE::DisconnectILEInterfaces()
{
  LowPowerMode_->UpdateILEInterface( nullptr );
  ActiveBlanking_->UpdateILEInterface( nullptr );
  Ports_->UpdateILEInterface( nullptr );
}

void CSingleILE::ReconnectILEInterfaces()
{
  IALC_REV_Port* vPortInterface = ILEDevice_->GetPortInterface();
  IALC_REV_ILEActiveBlankingManagement* vActiveBlanking = ILEWrapper_->GetILEActiveBlankingManagementInterface( ILEDevice_ );
  IALC_REV_ILEPowerManagement* vLowPowerMode = ILEWrapper_->GetILEPowerManagementInterface( ILEDevice_ );
  Ports_->UpdateILEInterface( vPortInterface );
  ActiveBlanking_->UpdateILEInterface( vActiveBlanking );
  LowPowerMode_->UpdateILEInterface( vLowPowerMode );
}

int CSingleILE::InitializePorts()
{
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
  return DEVICE_OK;
}

int CSingleILE::InitializeActiveBlanking()
{
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
  return DEVICE_OK;
}

int CSingleILE::InitializeLowPowerMode()
{
  IALC_REV_ILEPowerManagement* vLowPowerMode = ILEWrapper_->GetILEPowerManagementInterface( ILEDevice_ );
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
  return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// DUAL ILE
///////////////////////////////////////////////////////////////////////////////

CDualILE::CDualILE( bool ILE700 ) :
  CIntegratedLaserEngine( ( ILE700 ? g_Dual700DeviceDescription : g_DualDeviceDescription ), 2 ), 
  ILE700_( ILE700 ),
  PortsConfiguration_( g_Undefined ),
  Ports_( nullptr )
{
  SetErrorText( ERR_DUALPORTS_PORTCONFIGCORRUPTED, "Changing port failed. Port configuration is corrupted" );
  SetErrorText( ERR_DUALPORTS_PORTCHANGEFAIL, "Changing port failed." );
  
  // Create pre-initialization properties:
  // -------------------------------------

  // Ports configuration
  std::vector<std::string> vValues;
  std::vector<CPortsConfiguration>::const_iterator vPortsConfigurationIt = GetPortsConfigurationList().begin();
  while ( vPortsConfigurationIt != GetPortsConfigurationList().end() )
  {
    vValues.push_back( vPortsConfigurationIt->GetName() );
    ++vPortsConfigurationIt;
  }
  PortsConfiguration_ = vValues[0];
  CPropertyAction* pAct = new CPropertyAction( this, &CDualILE::OnPortsConfigurationChange );
  CreateStringProperty( g_PortsConfigurationProperty, PortsConfiguration_.c_str(), false, pAct, true );
  SetAllowedValues( g_PortsConfigurationProperty, vValues );
}

CDualILE::~CDualILE()
{
}

std::string CDualILE::GetDeviceName() const
{
  std::string vName = g_DualDeviceName;
  if ( ILE700_ )
  {
    vName = g_Dual700DeviceName;
  }
  return vName;
}

int CDualILE::Shutdown()
{
  delete Ports_;
  Ports_ = nullptr;
  return CIntegratedLaserEngine::Shutdown();
}

int CDualILE::OnPortsConfigurationChange( MM::PropertyBase* Prop, MM::ActionType Act )
{
  if ( Act == MM::BeforeGet )
  {
    Prop->Set( PortsConfiguration_.c_str() );
  }
  else if ( Act == MM::AfterSet )
  {
    Prop->Get( PortsConfiguration_ );
  }
  return DEVICE_OK;
}

bool CDualILE::CreateILE()
{
  bool vRet = false;
  if ( DevicesNames_.size() > 1 )
  {
    vRet = ILEWrapper_->CreateDualILE( &ILEDevice_, DevicesNames_[0].c_str(), DevicesNames_[1].c_str(), ILE700_ );
  }
  return vRet;
}

void CDualILE::DeleteILE()
{
  ILEWrapper_->DeleteDualILE( ILEDevice_ );
  ILEDevice_ = nullptr;
}

void CDualILE::DisconnectILEInterfaces()
{
  Ports_->UpdateILEInterface( nullptr, nullptr );
}

void CDualILE::ReconnectILEInterfaces()
{
  IALC_REV_ILE2* vILE2 = ILEWrapper_->GetILEInterface2( ILEDevice_ );
  IALC_REVObject3 *vILEDevice1, *vILEDevice2;
  if ( vILE2->GetInterface( &vILEDevice1, &vILEDevice2 ) )
  {
    IALC_REV_Port* vDualPortInterface = ILEDevice_->GetPortInterface();
    Ports_->UpdateILEInterface( vDualPortInterface, vILE2 );
  }
}

int CDualILE::InitializePorts()
{
  IALC_REV_ILE2* vILE2 = ILEWrapper_->GetILEInterface2( ILEDevice_ );
  IALC_REVObject3 *vILEDevice1, *vILEDevice2;
  if ( vILE2->GetInterface( &vILEDevice1, &vILEDevice2 ) )
  {
    IALC_REV_Port* vDualPortInterface = ILEDevice_->GetPortInterface();
    if ( vDualPortInterface!= nullptr )
    {
      CPortsConfiguration *vPortsConfiguration = nullptr;
      std::vector<CPortsConfiguration>& vPortsConfigurationList = GetPortsConfigurationList();
      std::vector<CPortsConfiguration>::iterator vPortsConfigurationIt = vPortsConfigurationList.begin();
      while ( vPortsConfiguration == nullptr && vPortsConfigurationIt != vPortsConfigurationList.end() )
      {
        if ( vPortsConfigurationIt->GetName() == PortsConfiguration_ )
        {
          vPortsConfiguration = &*vPortsConfigurationIt;
        }
        else
        {
          ++vPortsConfigurationIt;
        }
      }
      try
      {
        Ports_ = new CDualILEPorts( vDualPortInterface, vILE2, vPortsConfiguration, this );
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
  }
  else
  {
    LogMessage( "Retrieving Dual port intefaces failed" );
  }
  return DEVICE_OK;
}

int CDualILE::InitializeActiveBlanking()
{
  return DEVICE_OK;
}

int CDualILE::InitializeLowPowerMode()
{
  return DEVICE_OK;
}