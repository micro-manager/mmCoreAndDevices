///////////////////////////////////////////////////////////////////////////////
// FILE:          IntegratedLaserEngine.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   IntegratedLaserEngine controller adapter
//
// Based off the AndorLaserCombiner adapter from Karl Hoover, UCSF
//

#ifdef WIN32
#include <windows.h>
#endif

#include "../../MMDevice/MMDevice.h"
#include "boost/lexical_cast.hpp"
#include "ALC_REV.h"
#include "IntegratedLaserEngine.h"
#include "ILEWrapper.h"


#ifndef _isnan
#ifdef __GNUC__
#include <cmath>
using std::isnan;
#elif _MSC_VER  // MSVC.
#include <float.h>
#define isnan _isnan
#endif
#endif


// Properties
const char* const g_DeviceName = "IntegratedLaserEngine";
const char* const g_DeviceDescription = "Integrated Laser Engine";
const char* const g_DeviceListProperty = "Device";
const char* const g_HoursProperty = "Hours";
const char* const g_IsLinearProperty = "IsLinear";
const char* const g_PowerReadbackProperty = "LaserPowerReadback";
const char* const g_LaserStateProperty = "LaserState";
const char* const g_MaximumLaserPowerProperty = "MaximumLaserPower";
const char* const g_EnableProperty = "PowerEnable";
const char* const g_PowerSetpointProperty = "PowerSetpoint";

// Enable states
const char* const g_LaserEnableOn = "On";
const char* const g_LaserEnableOff = "Off";
const char* const g_LaserEnableTTL = "External TTL";

// Laser tates
const char* const g_LaserNotAvailable = "Laser Not Available";
const char* const g_LaserWarmingUp = "Laser Warming up";
const char* const g_LaserReady = "Laser Ready";
const char* const g_LaserInterlockError = "Interlock Error Detected";
const char* const g_LaserPowerError= "Power Error Detected";
const char* const g_LaserClassIVInterlockError = "Class IV Interlock Error Detected";
const char* const g_LaserUnknownState = "Unknown State";


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
// ~~~~~~~~~~~~~~~~~~~~~~~~~~

CIntegratedLaserEngine::CIntegratedLaserEngine() :
  Initialized_( false ),
  Busy_( false ),
  Error_( DEVICE_OK ),
  ChangedTime_( 0.0 ),
  ILEWrapper_( nullptr ),
  NumberOfLasers_( 0 ),
  OpenRequest_( false ),
  LaserPort_( 0 ),
  ILEDevice_( nullptr )
{
  // Load the library
  ILEWrapper_ = LoadILEWrapper( this );

  for ( int il = 0; il < MaxLasers + 1; ++il )
  {
    PowerSetPoint_[il] = 0;
    IsLinear_[il] = false;
    Enable_[il] = g_LaserEnableOn;
  }

  InitializeDefaultErrorMessages();

  // Create pre-initialization properties:
  // -------------------------------------

  // Description
  CreateStringProperty( MM::g_Keyword_Description, g_DeviceDescription, true );

  // Devices
  ILEWrapper_->GetListOfDevices( DeviceList_ );
  std::string vInitialDevice = "Undefined";
  if ( !DeviceList_.empty() )
  {
    vInitialDevice = DeviceList_.begin()->first;
  }
  CPropertyAction* pAct = new CPropertyAction( this, &CIntegratedLaserEngine::OnDeviceChange );
  CreateStringProperty( g_DeviceListProperty, vInitialDevice.c_str(), false, pAct, true );
  std::vector<std::string> vDevices;
  CILEWrapper::TDeviceList::const_iterator vDeviceIt = DeviceList_.begin();
  while ( vDeviceIt != DeviceList_.end() )
  {
    vDevices.push_back( vDeviceIt->first );
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

  try
  {
    if ( !ILEWrapper_->CreateILE( &ILEDevice_, DeviceName_.c_str() ) )
    {
      LogMessage( "CreateILE failed" );
      return DEVICE_ERR;
    }
    LaserInterface_ = ILEDevice_->GetLaserInterface2();
    if ( LaserInterface_ == nullptr )
    {
      throw std::runtime_error( "GetLaserInterface failed" );
    }

    NumberOfLasers_ = LaserInterface_->Initialize();
    LogMessage( ( "in CIntegratedLaserEngine::Initialize, NumberOfLasers_ =" + boost::lexical_cast<std::string, int>( NumberOfLasers_ ) ), true );
    CDeviceUtils::SleepMs( 100 );

    TLaserState state[10];
    memset( (void*)state, 0, 10 * sizeof( state[0] ) );

    //Andor says that lasers can take up to 90 seconds to initialize.
    MM::TimeoutMs vTimerOut( GetCurrentMMTime(), 91000 );
    int iloop = 0;

    for ( ;;)
    {
      bool vFinishWaiting = true;
      for ( int vLaserIndex = 1; vLaserIndex <= NumberOfLasers_; ++vLaserIndex )
      {
        if ( 0 == state[vLaserIndex] )
        {
          LaserInterface_->GetLaserState(vLaserIndex, state + vLaserIndex);
          switch( state[vLaserIndex] )
          {
          case ELaserState::ALC_NOT_AVAILABLE: // ALC_NOT_AVAILABLE ( 0) Laser is not Available
            vFinishWaiting = false;
            break;
          case ELaserState::ALC_WARM_UP: //ALC_WARM_UP ( 1) Laser Warming Up
            LogMessage( " laser " + boost::lexical_cast<std::string, int>( vLaserIndex ) + " is warming up", true );
            break;
          case ELaserState::ALC_READY: //ALC_READY ( 2) Laser is ready
            LogMessage( " laser " + boost::lexical_cast<std::string, int>( vLaserIndex ) + " is ready ", true );
            break;
          case ELaserState::ALC_INTERLOCK_ERROR: //ALC_INTERLOCK_ERROR ( 3) Interlock Error Detected
            LogMessage( " laser " + boost::lexical_cast<std::string, int>( vLaserIndex ) + " encountered interlock error ", false );
            break;
          case ELaserState::ALC_POWER_ERROR: //ALC_POWER_ERROR ( 4) Power Error Detected
            LogMessage( " laser " + boost::lexical_cast<std::string, int>( vLaserIndex ) + " encountered power error ", false );
            break;
          }
        }
      }
      if ( vFinishWaiting )
      {
        break;
      }
      else
      {
        if ( vTimerOut.expired( GetCurrentMMTime() ) )
        {
          LogMessage( " some lasers did not respond", false );
          break;
        }
        iloop++;
      }
      CDeviceUtils::SleepMs( 100 );
    }

    GenerateALCProperties();
    GenerateReadOnlyIDProperties();
  }
  catch ( std::string& exs )
  {
    vRet = DEVICE_LOCALLY_DEFINED_ERROR;
    LogMessage( exs.c_str() );
    SetErrorText( DEVICE_LOCALLY_DEFINED_ERROR, exs.c_str() );
    //CodeUtility::DebugOutput(exs.c_str());
    return vRet;
  }

  Initialized_ = true;
  return HandleErrors();
}

/////////////////////////////////////////////
// Property Generators
/////////////////////////////////////////////

std::string CIntegratedLaserEngine::BuildPropertyName( const std::string& BasePropertyName, int Wavelength )
{
  return std::to_string( Wavelength ) + "-" + BasePropertyName;
}

void CIntegratedLaserEngine::GenerateALCProperties()
{
  CPropertyActionEx* pAct; 
  std::string vPropertyName;
  int vWavelength;

  // 1 based index for the lasers
  for ( int vLaserIndex = 1; vLaserIndex < NumberOfLasers_ + 1; ++vLaserIndex )
  {
    vWavelength = Wavelength( vLaserIndex );
    pAct = new CPropertyActionEx( this, &CIntegratedLaserEngine::OnPowerSetpoint, vLaserIndex );
    vPropertyName = BuildPropertyName( g_PowerSetpointProperty, vWavelength );
    CreateProperty( vPropertyName.c_str(), "0", MM::Float, false, pAct );

    float vFullScale = 10.00;
    // Set the limits as interrogated from the laser controller
    LogMessage( "Range for " + vPropertyName + "= [0," + boost::lexical_cast<std::string, float>( vFullScale ) + "]", true );
    SetPropertyLimits( vPropertyName.c_str(), 0, vFullScale );  // Volts

    pAct = new CPropertyActionEx( this, &CIntegratedLaserEngine::OnMaximumLaserPower, vLaserIndex );
    vPropertyName = BuildPropertyName( g_MaximumLaserPowerProperty, vWavelength );
    CreateProperty( vPropertyName.c_str(), std::to_string( vFullScale ).c_str(), MM::Integer, true, pAct );

    // Readbacks
    pAct = new CPropertyActionEx( this, &CIntegratedLaserEngine::OnPowerReadback, vLaserIndex );
    vPropertyName = BuildPropertyName( g_PowerReadbackProperty, vWavelength );
    CreateProperty( vPropertyName.c_str(), "0", MM::Float, true, pAct );

    // States
    pAct = new CPropertyActionEx( this, &CIntegratedLaserEngine::OnLaserState, vLaserIndex );
    vPropertyName = BuildPropertyName( g_LaserStateProperty, vWavelength );
    CreateStringProperty( vPropertyName.c_str(), g_LaserUnknownState, true, pAct );

    // Enable
    pAct = new CPropertyActionEx( this, &CIntegratedLaserEngine::OnEnable, vLaserIndex );
    vPropertyName = BuildPropertyName( g_EnableProperty, vWavelength );
    EnableStates_[vLaserIndex].clear();
    EnableStates_[vLaserIndex].push_back( g_LaserEnableOn );
    EnableStates_[vLaserIndex].push_back( g_LaserEnableOff );
    if ( AllowsExternalTTL( vLaserIndex ) )
    {
      EnableStates_[vLaserIndex].push_back( g_LaserEnableTTL );
    }
    CreateProperty( vPropertyName.c_str(), EnableStates_[vLaserIndex][0].c_str(), MM::String, false, pAct );
    SetAllowedValues( vPropertyName.c_str(), EnableStates_[vLaserIndex] );
  }
}

void CIntegratedLaserEngine::GenerateReadOnlyIDProperties()
{
  CPropertyActionEx* pAct;
  std::string vPropertyName;
  int vWavelength;

  // 1 based index
  for ( int vLaserIndex = 1; vLaserIndex < NumberOfLasers_ + 1; ++vLaserIndex )
  {
    vWavelength = Wavelength( vLaserIndex );
    pAct = new CPropertyActionEx( this, &CIntegratedLaserEngine::OnHours, vLaserIndex );
    vPropertyName = BuildPropertyName( g_HoursProperty, vWavelength );
    CreateProperty( vPropertyName.c_str(), "", MM::String, true, pAct );

    pAct = new CPropertyActionEx( this, &CIntegratedLaserEngine::OnIsLinear, vLaserIndex );
    vPropertyName = BuildPropertyName( g_IsLinearProperty, vWavelength );
    CreateProperty( vPropertyName.c_str(), "", MM::String, true, pAct );
  }
}

int CIntegratedLaserEngine::Shutdown()
{
  if ( Initialized_ )
  {
    Initialized_ = false;
    ILEWrapper_->DeleteILE( ILEDevice_ );
  }
  return HandleErrors();
}

///////////////////////////////////////////////////////////////////////////////
// Action handlers
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
  return HandleErrors();
}

/**
 * Current laser head power output.
 * <p>
 * Output of 0, could be due to laser being put in Standby using
 * SaveLifetime, or a fault with the laser head.  If power is more than
 * a few percent lower than MaximumLaserPower, it also indicates a
 * faulty laser head, but some lasers can take up to 5 minutes to warm
 * up (most warm up in 2 minutes).
 *
 * @see OnMaximumLaserPower(MM::PropertyBase* Prop, MM::ActionType Act, long LaserIndex)
 */
int CIntegratedLaserEngine::OnPowerReadback(MM::PropertyBase* Prop, MM::ActionType Act, long LaserIndex)
{
  if ( Act == MM::BeforeGet )
  {
    double vValue = PowerReadback( (int)LaserIndex );
    LogMessage( " PowerReadback" + boost::lexical_cast<std::string, long>( Wavelength( LaserIndex ) ) + "  = " + boost::lexical_cast<std::string, double>( vValue ), true );
    Prop->Set( vValue );
  }
  else if ( Act == MM::AfterSet )
  {
    // This is a read-only property!
  }
  return HandleErrors();
}

/**
 * AOTF intensity setting.  Actual power output may or may not be
 * linear.
 *
 * @see OnIsLinear(MM::PropertyBase* Prop, MM::ActionType Act, long LaserIndex)
 */
int CIntegratedLaserEngine::OnPowerSetpoint(MM::PropertyBase* Prop, MM::ActionType Act, long  LaserIndex)
{
  double vPowerSetpoint;
  if ( Act == MM::BeforeGet )
  {
    vPowerSetpoint = (double)PowerSetpoint( LaserIndex );
    LogMessage( "from equipment: PowerSetpoint" + boost::lexical_cast<std::string, long>( Wavelength( LaserIndex ) ) + "  = " + boost::lexical_cast<std::string, double>( vPowerSetpoint ), true );
    Prop->Set( vPowerSetpoint );
  }
  else if ( Act == MM::AfterSet )
  {
    Prop->Get( vPowerSetpoint );
    LogMessage( "to equipment: PowerSetpoint" + boost::lexical_cast<std::string, long>( Wavelength( LaserIndex ) ) + "  = " + boost::lexical_cast<std::string, double>( vPowerSetpoint ), true );
    PowerSetpoint( LaserIndex, static_cast<float>( vPowerSetpoint ) );
    if ( OpenRequest_ )
      SetOpen();

    //Prop->Set(achievedSetpoint);  ---- for quantization....
  }
  return HandleErrors();
}

/**
 * Laser bulb hours.
 * <p>
 * Indicates laser expires life to plan warranty contracts.
 * 
 * @see OnPowerReadback(MM::PropertyBase* Prop, MM::ActionType Act, long LaserIndex)
 */
int CIntegratedLaserEngine::OnHours(MM::PropertyBase* Prop, MM::ActionType Act, long LaserIndex)
{
  if ( Act == MM::BeforeGet )
  {
    int vValue = 0;
    LaserInterface_->GetLaserHours( LaserIndex, &vValue );
    LogMessage( "Hours" + boost::lexical_cast<std::string, long>( Wavelength( LaserIndex ) ) + "  = " + boost::lexical_cast<std::string, int>( vValue ), true );
    Prop->Set( (long)vValue );
  }
  else if ( Act == MM::AfterSet )
  {
    // This is a read-only property!
  }
  return HandleErrors();
}

/**
 * Reads whether linear correction algorithm is being applied to AOTF
 * by PowerSetpoint, otherwise AOTF output is sigmoid.
 * <p>
 * Requires firmware 2.
 *
 * @see OnPowerSetpoint(MM::PropertyBase* Prop, MM::ActionType Act, long LaserIndex)
 */
int CIntegratedLaserEngine::OnIsLinear(MM::PropertyBase* Prop, MM::ActionType Act, long LaserIndex) 
{
  if ( Act == MM::BeforeGet )
  {
    int vValue;
    LaserInterface_->IsLaserOutputLinearised( LaserIndex, &vValue );
    IsLinear_[LaserIndex] = ( vValue == 1 );
    LogMessage( "IsLinear" + boost::lexical_cast<std::string, long>( Wavelength( LaserIndex ) ) + "  = " + boost::lexical_cast<std::string, int>( vValue ), true );
    Prop->Set( (long)vValue );
  }
  else if ( Act == MM::AfterSet )
  {
    // This is a read-only property!
  }
  return HandleErrors();
}

/**
 * Laser rated operating power in milli-Watts.
 *
 * @see OnPowerReadback(MM::PropertyBase* Prop, MM::ActionType Act, long LaserIndex)
 */
int CIntegratedLaserEngine::OnMaximumLaserPower(MM::PropertyBase* Prop, MM::ActionType Act, long LaserIndex)
{
  if ( Act == MM::BeforeGet )
  {
    int vValue = PowerFullScale(LaserIndex);
    LogMessage("PowerFullScale" + boost::lexical_cast<std::string, long>(Wavelength(LaserIndex)) + "  = " + boost::lexical_cast<std::string,int>(vValue), true);
    Prop->Set((long)vValue);
  }
  else if ( Act == MM::AfterSet )
  {
    // This is a read-only property!
  }
  return HandleErrors();
}

/**
 * Laser state.
 */
int CIntegratedLaserEngine::OnLaserState(MM::PropertyBase* Prop, MM::ActionType Act, long LaserIndex)
{
  if ( Act == MM::BeforeGet )
  {
    TLaserState vState;
    LaserInterface_->GetLaserState( LaserIndex, &vState );
    const char* vStateString = g_LaserUnknownState;
    switch ( vState )
    {
    case ELaserState::ALC_NOT_AVAILABLE:
      vStateString = g_LaserNotAvailable;
      break;
    case ELaserState::ALC_WARM_UP:
      vStateString = g_LaserWarmingUp;
      break;
    case ELaserState::ALC_READY:
      vStateString = g_LaserReady;
      break;
    case ELaserState::ALC_INTERLOCK_ERROR:
      vStateString = g_LaserInterlockError;
      break;
    case ELaserState::ALC_POWER_ERROR:
      vStateString = g_LaserPowerError;
      break;
    case ELaserState::ALC_CLASS_IV_INTERLOCK_ERROR:
      vStateString = g_LaserClassIVInterlockError;
      break;
    }
    LogMessage( " LaserState" + boost::lexical_cast<std::string, long>( Wavelength( LaserIndex ) ) + "  = " + boost::lexical_cast<std::string, int>( vState ) + "[" + vStateString + "]", true );
    Prop->Set( vStateString );
  }
  else if ( Act == MM::AfterSet )
  {
    // This is a read-only property!
  }
   return HandleErrors();
}

/**
 * Logical shutter to allow selection of laser line.  It can also set
 * the laser to TTL mode, if the laser supports it.
 * <p>
 * TTL mode requires firmware 2.
 */
int CIntegratedLaserEngine::OnEnable(MM::PropertyBase* Prop, MM::ActionType Act, long LaserIndex)
{
  if ( Act == MM::BeforeGet )
  {
    // Not calling GetControlMode() from ALC SDK, since it may slow
    // down acquisition while switching channels
    Prop->Set( Enable_[LaserIndex].c_str() );
  }
  else if ( Act == MM::AfterSet )
  {
    std::string vEnable;
    Prop->Get( vEnable );
    if ( Enable_[LaserIndex].compare( vEnable ) != 0 )
    {
      // Update the laser control mode if we are changing to, or
      // from External TTL mode
      if ( vEnable.compare( g_LaserEnableTTL ) == 0 )
      {
        LaserInterface_->SetControlMode( LaserIndex, TTL_PULSED );
      }
      else if ( Enable_[LaserIndex].compare( g_LaserEnableTTL ) == 0 )
      {
        LaserInterface_->SetControlMode( LaserIndex, CW );
      }

      Enable_[LaserIndex] = vEnable;
      LogMessage( "Enable" + boost::lexical_cast<std::string, long>( Wavelength( LaserIndex ) ) + " = " + Enable_[LaserIndex], true );
      if ( OpenRequest_ )
      {
        SetOpen();
      }
    }
  }
  return HandleErrors();
}

int CIntegratedLaserEngine::HandleErrors()
{
  int vLastError = Error_;
  Error_ = DEVICE_OK;
  return vLastError;
}

//********************
// Shutter API
//********************

int CIntegratedLaserEngine::SetOpen(bool Open)
{
  for( int vLaserIndex = 1; vLaserIndex <= NumberOfLasers_; ++vLaserIndex)
  {
    if ( Open )
    {
      double vFullScale = 10.00; // Volts instead of milliWatts, and  double instead of int
      bool vLaserOn = ( PowerSetpoint( vLaserIndex ) > 0 ) && ( Enable_[vLaserIndex].compare( g_LaserEnableOff ) != 0 );
      double vPercentScale = 0.;
      if ( vLaserOn )
      {
        vPercentScale = 100.*PowerSetpoint( vLaserIndex ) / vFullScale;
      }

      if ( 100. < vPercentScale )
      {
        vPercentScale = 100.;
      }
      LogMessage( "SetLas" + boost::lexical_cast<std::string, long>( vLaserIndex ) + "  = " + boost::lexical_cast<std::string, double>( vPercentScale ) + "(" + boost::lexical_cast<std::string, bool>( vLaserOn ) + ")", true );

      TLaserState vLaserState;
      LaserInterface_->GetLaserState( vLaserIndex, &vLaserState );
      if ( vLaserOn && ( vLaserState != ELaserState::ALC_READY ) )
      {
        std::string vMessage = "Laser # " + boost::lexical_cast<std::string, int>( vLaserIndex ) + " is not ready!";
        // laser is not ready!
        LogMessage( vMessage.c_str(), false );
        // GetCoreCallback()->PostError(std::make_pair<int,std::string>(DEVICE_ERR,vMessage));
      }

      if ( vLaserState > ELaserState::ALC_NOT_AVAILABLE )
      {
        LogMessage( "setting Laser " + boost::lexical_cast<std::string, int>( Wavelength( vLaserIndex ) ) + " to " + boost::lexical_cast<std::string, double>( vPercentScale ) + "% full scale", true );
        LaserInterface_->SetLas_I( vLaserIndex, vPercentScale, vLaserOn );
      }
    }
    LogMessage( "set shutter " + boost::lexical_cast<std::string, bool>( Open ), true );
    bool vSuccess = LaserInterface_->SetLas_Shutter( Open );
    if ( !vSuccess )
    {
      LogMessage( "set shutter " + boost::lexical_cast<std::string, bool>( Open ) + " failed", false );
    }
  }

  OpenRequest_ = Open;

  return DEVICE_OK;
}

int CIntegratedLaserEngine::GetOpen(bool& Open)
{
  // todo check that all requested lasers are 'ready'
  Open = OpenRequest_; // && Ready();
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
  return HandleErrors();
}

int CIntegratedLaserEngine::Wavelength(const int LaserIndex )
{
  int vValue = 0;
  LaserInterface_->GetWavelength( LaserIndex, &vValue );
  return vValue;
}

int CIntegratedLaserEngine::PowerFullScale(const int LaserIndex )
{
  int vValue = 0;
  LaserInterface_->GetPower( LaserIndex, &vValue );
  return vValue;
}

float CIntegratedLaserEngine::PowerReadback(const int LaserIndex )
{
  double vValue = 0.;
  LaserInterface_->GetCurrentPower( LaserIndex, &vValue );
  if ( isnan( vValue ) )
  {
    LogMessage( "invalid PowerReadback on # " + boost::lexical_cast<std::string, int>( LaserIndex ), false );
    vValue = 0.;
  }
  return (float)vValue;
}

float CIntegratedLaserEngine::PowerSetpoint(const int LaserIndex )
{
  return PowerSetPoint_[LaserIndex];
}

void  CIntegratedLaserEngine::PowerSetpoint(const int LaserIndex, const float Value)
{
  PowerSetPoint_[LaserIndex] = Value;
}

bool CIntegratedLaserEngine::AllowsExternalTTL(const int LaserIndex )
{
  int vValue = 0;
  LaserInterface_->IsControlModeAvailable( LaserIndex, &vValue);
  return (vValue == 1);
}

bool CIntegratedLaserEngine::Ready(const int LaserIndex )
{
  TLaserState vState = ELaserState::ALC_NOT_AVAILABLE;
  bool vRet = LaserInterface_->GetLaserState( LaserIndex, &vState );
  return vRet && ( ELaserState::ALC_READY == vState );
}

void CIntegratedLaserEngine::LogMMMessage( std::string Message )
{
  LogMessage( Message );
}