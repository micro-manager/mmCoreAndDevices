///////////////////////////////////////////////////////////////////////////////
// FILE:          Lasers.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------

#include "boost/lexical_cast.hpp"
#include "ALC_REV.h"
#include "Lasers.h"
#include "IntegratedLaserEngine.h"

// Properties
const char* const g_EnableProperty = "Power Enable";
const char* const g_PowerSetpointProperty = "Power Setpoint";

// Enable states
const char* const g_LaserEnableOn = "On";
const char* const g_LaserEnableOff = "Off";
const char* const g_LaserEnableTTL = "External TTL";


CLasers::CLasers( IALC_REV_Laser2 *LaserInterface, CIntegratedLaserEngine* MMILE ) :
  LaserInterface_( LaserInterface ),
  MMILE_( MMILE ),
  NumberOfLasers_( 0 ),
  OpenRequest_( false )
{
  for ( int il = 0; il < MaxLasers + 1; ++il )
  {
    PowerSetPoint_[il] = 0;
    Enable_[il] = g_LaserEnableOn;
  }

  NumberOfLasers_ = LaserInterface_->Initialize();
  MMILE_->LogMMMessage( ( "in CLasers constructor, NumberOfLasers_ =" + boost::lexical_cast<std::string, int>( NumberOfLasers_ ) ), true );
  CDeviceUtils::SleepMs( 100 );

  TLaserState state[10];
  memset( (void*)state, 0, 10 * sizeof( state[0] ) );

  // Lasers can take up to 90 seconds to initialize
  MM::TimeoutMs vTimerOut( MMILE_->GetCurrentTime(), 91000 );
  int iloop = 0;

  for ( ;;)
  {
    bool vFinishWaiting = true;
    for ( int vLaserIndex = 1; vLaserIndex <= NumberOfLasers_; ++vLaserIndex )
    {
      if ( 0 == state[vLaserIndex] )
      {
        LaserInterface_->GetLaserState( vLaserIndex, state + vLaserIndex );
        switch ( state[vLaserIndex] )
        {
        case ELaserState::ALC_NOT_AVAILABLE:
          vFinishWaiting = false;
          break;
        case ELaserState::ALC_WARM_UP:
          MMILE_->LogMMMessage( " laser " + boost::lexical_cast<std::string, int>( vLaserIndex ) + " is warming up", true );
          break;
        case ELaserState::ALC_READY:
          MMILE_->LogMMMessage( " laser " + boost::lexical_cast<std::string, int>( vLaserIndex ) + " is ready ", true );
          break;
        case ELaserState::ALC_INTERLOCK_ERROR:
          MMILE_->LogMMMessage( " laser " + boost::lexical_cast<std::string, int>( vLaserIndex ) + " encountered interlock error ", false );
          break;
        case ELaserState::ALC_POWER_ERROR:
          MMILE_->LogMMMessage( " laser " + boost::lexical_cast<std::string, int>( vLaserIndex ) + " encountered power error ", false );
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
      if ( vTimerOut.expired( MMILE_->GetCurrentTime() ) )
      {
        MMILE_->LogMMMessage( " some lasers did not respond", false );
        break;
      }
      iloop++;
    }
    CDeviceUtils::SleepMs( 100 );
  }

  GenerateALCProperties();
}

CLasers::~CLasers()
{
}

///////////////////////////////////////////////////////////////////////////////
// Generate properties
///////////////////////////////////////////////////////////////////////////////

std::string CLasers::BuildPropertyName( const std::string& BasePropertyName, int Wavelength )
{
  return "Laser " + std::to_string( Wavelength ) + "-" + BasePropertyName;
}

void CLasers::GenerateALCProperties()
{
  CPropertyActionEx* vAct; 
  std::string vPropertyName;
  int vWavelength;

  // 1 based index for the lasers
  for ( int vLaserIndex = 1; vLaserIndex < NumberOfLasers_ + 1; ++vLaserIndex )
  {
    vWavelength = Wavelength( vLaserIndex );
    vAct = new CPropertyActionEx( this, &CLasers::OnPowerSetpoint, vLaserIndex );
    vPropertyName = BuildPropertyName( g_PowerSetpointProperty, vWavelength );
    MMILE_->CreateProperty( vPropertyName.c_str(), "0", MM::Float, false, vAct );

    float vFullScale = 10.00;
    // Set the limits as interrogated from the laser controller
    MMILE_->LogMMMessage( "Range for " + vPropertyName + "= [0," + boost::lexical_cast<std::string, float>( vFullScale ) + "]", true );
    MMILE_->SetPropertyLimits( vPropertyName.c_str(), 0, vFullScale );  // Volts

    // Enable
    vAct = new CPropertyActionEx( this, &CLasers::OnEnable, vLaserIndex );
    vPropertyName = BuildPropertyName( g_EnableProperty, vWavelength );
    EnableStates_[vLaserIndex].clear();
    EnableStates_[vLaserIndex].push_back( g_LaserEnableOn );
    EnableStates_[vLaserIndex].push_back( g_LaserEnableOff );
    if ( AllowsExternalTTL( vLaserIndex ) )
    {
      EnableStates_[vLaserIndex].push_back( g_LaserEnableTTL );
    }
    MMILE_->CreateProperty( vPropertyName.c_str(), EnableStates_[vLaserIndex][0].c_str(), MM::String, false, vAct );
    MMILE_->SetAllowedValues( vPropertyName.c_str(), EnableStates_[vLaserIndex] );
  }
}

///////////////////////////////////////////////////////////////////////////////
// Actions
///////////////////////////////////////////////////////////////////////////////

/**
* AOTF intensity setting.  Actual power output may or may not be
* linear.
*/
int CLasers::OnPowerSetpoint(MM::PropertyBase* Prop, MM::ActionType Act, long  LaserIndex)
{
  double vPowerSetpoint;
  if ( Act == MM::BeforeGet )
  {
    vPowerSetpoint = (double)PowerSetpoint( LaserIndex );
    MMILE_->LogMMMessage( "from equipment: PowerSetpoint" + boost::lexical_cast<std::string, long>( Wavelength( LaserIndex ) ) + "  = " + boost::lexical_cast<std::string, double>( vPowerSetpoint ), true );
    Prop->Set( vPowerSetpoint );
  }
  else if ( Act == MM::AfterSet )
  {
    Prop->Get( vPowerSetpoint );
    MMILE_->LogMMMessage( "to equipment: PowerSetpoint" + boost::lexical_cast<std::string, long>( Wavelength( LaserIndex ) ) + "  = " + boost::lexical_cast<std::string, double>( vPowerSetpoint ), true );
    PowerSetpoint( LaserIndex, static_cast<float>( vPowerSetpoint ) );
    if ( OpenRequest_ )
      SetOpen();

    //Prop->Set(achievedSetpoint);  ---- for quantization....
  }
  return DEVICE_OK;
}

/**
 * Logical shutter to allow selection of laser line.  It can also set
 * the laser to TTL mode, if the laser supports it.
 * <p>
 * TTL mode requires firmware 2.
 */
int CLasers::OnEnable(MM::PropertyBase* Prop, MM::ActionType Act, long LaserIndex)
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
      MMILE_->LogMMMessage( "Enable" + boost::lexical_cast<std::string, long>( Wavelength( LaserIndex ) ) + " = " + Enable_[LaserIndex], true );
      if ( OpenRequest_ )
      {
        SetOpen();
      }
    }
  }
  return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// Shutter API
///////////////////////////////////////////////////////////////////////////////

int CLasers::SetOpen(bool Open)
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
      MMILE_->LogMMMessage( "SetLas" + boost::lexical_cast<std::string, long>( vLaserIndex ) + "  = " + boost::lexical_cast<std::string, double>( vPercentScale ) + "(" + boost::lexical_cast<std::string, bool>( vLaserOn ) + ")", true );

      TLaserState vLaserState;
      LaserInterface_->GetLaserState( vLaserIndex, &vLaserState );
      if ( vLaserOn && ( vLaserState != ELaserState::ALC_READY ) )
      {
        std::string vMessage = "Laser # " + boost::lexical_cast<std::string, int>( vLaserIndex ) + " is not ready!";
        // laser is not ready!
        MMILE_->LogMMMessage( vMessage.c_str(), false );
        // GetCoreCallback()->PostError(std::make_pair<int,std::string>(DEVICE_ERR,vMessage));
      }

      if ( vLaserState > ELaserState::ALC_NOT_AVAILABLE )
      {
        MMILE_->LogMMMessage( "setting Laser " + boost::lexical_cast<std::string, int>( Wavelength( vLaserIndex ) ) + " to " + boost::lexical_cast<std::string, double>( vPercentScale ) + "% full scale", true );
        if ( !LaserInterface_->SetLas_I( vLaserIndex, vPercentScale, vLaserOn ) )
        {
          MMILE_->LogMMMessage( std::string( "Setting Laser power for laser " + std::to_string( vLaserIndex ) + " failed with value [" ) + std::to_string( vPercentScale ) + "]" );
        }
      }
    }
    MMILE_->LogMMMessage( "set shutter " + boost::lexical_cast<std::string, bool>( Open ), true );
    bool vSuccess = LaserInterface_->SetLas_Shutter( Open );
    if ( !vSuccess )
    {
      MMILE_->LogMMMessage( "set shutter " + boost::lexical_cast<std::string, bool>( Open ) + " failed", false );
    }
  }

  OpenRequest_ = Open;

  return DEVICE_OK;
}

int CLasers::GetOpen(bool& Open)
{
  // todo check that all requested lasers are 'ready'
  Open = OpenRequest_; // && Ready();
  return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// Update lasers
///////////////////////////////////////////////////////////////////////////////

void CLasers::CheckAndUpdateLasers()
{

}

///////////////////////////////////////////////////////////////////////////////
// Private functions
///////////////////////////////////////////////////////////////////////////////

int CLasers::Wavelength(const int LaserIndex )
{
  int vValue = 0;
  LaserInterface_->GetWavelength( LaserIndex, &vValue );
  return vValue;
}

float CLasers::PowerSetpoint(const int LaserIndex )
{
  return PowerSetPoint_[LaserIndex];
}

void  CLasers::PowerSetpoint(const int LaserIndex, const float Value)
{
  PowerSetPoint_[LaserIndex] = Value;
}

bool CLasers::AllowsExternalTTL(const int LaserIndex )
{
  int vValue = 0;
  LaserInterface_->IsControlModeAvailable( LaserIndex, &vValue);
  return (vValue == 1);
}
