///////////////////////////////////////////////////////////////////////////////
// FILE:          Lasers.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------

#include "ALC_REV.h"
#include "Lasers.h"
#include "IntegratedLaserEngine.h"

// Properties
const char* const g_EnableProperty = "Power Enable";
const char* const g_PowerSetpointProperty = "Power Setpoint";
const char* const g_InterlockStatus = "Interlock Flag Status";

// Property values
const char* const g_LaserEnableOn = "On";
const char* const g_LaserEnableOff = "Off";
const char* const g_LaserEnableTTL = "External TTL";

const char* const g_InterlockInactive = "Inactive";
const char* const g_InterlockActive = "Active";
const char* const g_InterlockClassIVActive = "Class IV interlock active. Device reset required.";
const char* const g_InterlockKeyActive = "Key interlock active. Device reset required.";

CLasers::CLasers( IALC_REV_Laser2 *LaserInterface, IALC_REV_ILEPowerManagement* PowerInterface, IALC_REV_ILE* ILEInterface, CIntegratedLaserEngine* MMILE ) :
  LaserInterface_( LaserInterface ),
  PowerInterface_( PowerInterface ),
  ILEInterface_( ILEInterface ),
  MMILE_( MMILE ),
  NumberOfLasers_( 0 ),
  OpenRequest_( false ),
  DisplayedInterlockState_( NO_INTERLOCK ),
#ifdef _ACTIVATE_DUMMYTEST_
  InterlockTEMP_( false ),
  ClassIVInterlockTEMP_( false ),
  KeyInterlockTEMP_( false ),
#endif
  InterlockStatusMonitor_( nullptr )
{
  if ( LaserInterface_ == nullptr )
  {
    throw std::logic_error( "CLasers: Pointer to Laser interface invalid" );
  }
  if ( PowerInterface_ == nullptr )
  {
    throw std::logic_error( "CLasers: Pointer to Power interface invalid" );
  }
  if ( ILEInterface_ == nullptr )
  {
    throw std::logic_error( "CLasers: Pointer to ILE interface invalid" );
  }
  if ( MMILE_ == nullptr )
  {
    throw std::logic_error( "CLasers: Pointer tomain class invalid" );
  }

  for ( int vLaserIndex = 0; vLaserIndex < MaxLasers + 1; ++vLaserIndex )
  {
    PowerSetPoint_[vLaserIndex] = 0;
    Enable_[vLaserIndex] = g_LaserEnableOn;
    LaserRange_[vLaserIndex].PowerMin = LaserRange_[vLaserIndex].PowerMax = 0;
  }

  NumberOfLasers_ = LaserInterface_->Initialize();
  MMILE_->LogMMMessage( ( "in CLasers constructor, NumberOfLasers_ =" + std::to_string( static_cast<long long>( NumberOfLasers_ ) ) ), true );
  CDeviceUtils::SleepMs( 100 );

  WaitOnLaserWarmingUp();
  GenerateProperties();

  InterlockStatusMonitor_ = new CInterlockStatusMonitor( MMILE_ );
  InterlockStatusMonitor_->activate();
}

CLasers::~CLasers()
{
  delete InterlockStatusMonitor_;
}

///////////////////////////////////////////////////////////////////////////////
// Generate properties
///////////////////////////////////////////////////////////////////////////////

void CLasers::WaitOnLaserWarmingUp()
{
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
        case ALC_NOT_AVAILABLE:
          vFinishWaiting = false;
          break;
        case ALC_WARM_UP:
          MMILE_->LogMMMessage( " laser " + std::to_string( static_cast<long long>( vLaserIndex ) ) + " is warming up", true );
          break;
        case ALC_READY:
          MMILE_->LogMMMessage( " laser " + std::to_string( static_cast<long long>( vLaserIndex ) ) + " is ready ", true );
          break;
        case ALC_INTERLOCK_ERROR:
          MMILE_->LogMMMessage( " laser " + std::to_string( static_cast<long long>( vLaserIndex ) ) + " encountered interlock error ", false );
          break;
        case ALC_POWER_ERROR:
          MMILE_->LogMMMessage( " laser " + std::to_string( static_cast<long long>( vLaserIndex ) ) + " encountered power error ", false );
          break;
        case ALC_CLASS_IV_INTERLOCK_ERROR:
          MMILE_->LogMMMessage( " laser " + std::to_string( static_cast<long long>( vLaserIndex ) ) + " encountered class IV interlock error ", false );
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

}

std::string CLasers::BuildPropertyName( const std::string& BasePropertyName, int Wavelength )
{
  std::string vPropertyName = "Laser " + std::to_string( static_cast<long long>( Wavelength ) ) + "-" + BasePropertyName;

  int vIndex = 0;
  char vValue[MM::MaxStrLength];
  while ( MMILE_->GetProperty( vPropertyName.c_str(), vValue ) == DEVICE_OK )
  {
    // Property already exists, build a new name
    vIndex++;
    vPropertyName = "Laser " + std::to_string( static_cast<long long>( Wavelength ) ) + "_" + std::to_string( static_cast<long long>( vIndex ) ) + "-" + BasePropertyName;
  }
  return vPropertyName;
}

void CLasers::GenerateProperties()
{
  CPropertyActionEx* vActEx; 
  std::string vPropertyName;
  int vWavelength;

  // 1 based index for the lasers
  for ( int vLaserIndex = 1; vLaserIndex < NumberOfLasers_ + 1; ++vLaserIndex )
  {
    vWavelength = Wavelength( vLaserIndex );
    vActEx = new CPropertyActionEx( this, &CLasers::OnPowerSetpoint, vLaserIndex );
    vPropertyName = BuildPropertyName( g_PowerSetpointProperty, vWavelength );
    MMILE_->CreateProperty( vPropertyName.c_str(), "0", MM::Float, false, vActEx );

    // Set the limits as interrogated from the laser controller
    MMILE_->LogMMMessage( "Range for " + vPropertyName + "= [0,100]", true );
    MMILE_->SetPropertyLimits( vPropertyName.c_str(), 0, 100 );
    PropertyPointers_[vPropertyName] = nullptr;

    // Enable
    vActEx = new CPropertyActionEx( this, &CLasers::OnEnable, vLaserIndex );
    vPropertyName = BuildPropertyName( g_EnableProperty, vWavelength );
    EnableStates_[vLaserIndex].clear();
    EnableStates_[vLaserIndex].push_back( g_LaserEnableOn );
    EnableStates_[vLaserIndex].push_back( g_LaserEnableOff );
    if ( AllowsExternalTTL( vLaserIndex ) )
    {
      EnableStates_[vLaserIndex].push_back( g_LaserEnableTTL );
    }
    MMILE_->CreateProperty( vPropertyName.c_str(), EnableStates_[vLaserIndex][0].c_str(), MM::String, false, vActEx );
    MMILE_->SetAllowedValues( vPropertyName.c_str(), EnableStates_[vLaserIndex] );
    PropertyPointers_[vPropertyName] = nullptr;
  }

  CPropertyAction* vAct = new CPropertyAction( this, &CLasers::OnInterlockStatus );
  MMILE_->CreateStringProperty( g_InterlockStatus, g_InterlockInactive, true, vAct );

#ifdef _ACTIVATE_DUMMYTEST_
  std::vector<std::string> vEnabledValues;
  vEnabledValues.push_back( g_LaserEnableOn );
  vEnabledValues.push_back( g_LaserEnableOff );
  vAct = new CPropertyAction( this, &CLasers::OnInterlock );
  MMILE_->CreateStringProperty( "Interlock TEST Activate", g_LaserEnableOff, false, vAct );
  MMILE_->SetAllowedValues( "Interlock TEST Activate", vEnabledValues );

  vAct = new CPropertyAction( this, &CLasers::OnClassIVInterlock );
  MMILE_->CreateStringProperty( "Interlock TEST Activate Class IV", g_LaserEnableOff, false, vAct );
  MMILE_->SetAllowedValues( "Interlock TEST Activate Class IV", vEnabledValues );

  vAct = new CPropertyAction( this, &CLasers::OnKeyInterlock );
  MMILE_->CreateStringProperty( "Interlock TEST Activate Key", g_LaserEnableOff, false, vAct );
  MMILE_->SetAllowedValues( "Interlock TEST Activate Key", vEnabledValues );
#endif
  UpdateLasersRange();
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
  if ( PropertyPointers_.find( Prop->GetName() ) != PropertyPointers_.end() && PropertyPointers_[Prop->GetName()] == nullptr )
  {
    PropertyPointers_[Prop->GetName()] = Prop;
  }
  double vPowerSetpoint;
  if ( Act == MM::BeforeGet )
  {
    vPowerSetpoint = (double)PowerSetpoint( LaserIndex );
    MMILE_->LogMMMessage( "from equipment: PowerSetpoint" + std::to_string( static_cast<long long>( Wavelength( LaserIndex ) ) ) + "  = " + std::to_string( static_cast<long double>( vPowerSetpoint ) ), true );
    Prop->Set( vPowerSetpoint );
  }
  else if ( Act == MM::AfterSet )
  {
    int vInterlockStatus = MMILE_->GetClassIVAndKeyInterlockStatus();
    if ( vInterlockStatus != DEVICE_OK )
    {
      return vInterlockStatus;
    }
    if ( LaserInterface_ == nullptr )
    {
      return ERR_DEVICE_NOT_CONNECTED;
    }
    if ( !IsInterlockTriggered( LaserIndex ) )
    {
      Prop->Get( vPowerSetpoint );
      MMILE_->LogMMMessage( "to equipment: PowerSetpoint" + std::to_string( static_cast<long long>( Wavelength( LaserIndex ) ) ) + "  = " + std::to_string( static_cast<long double>( vPowerSetpoint ) ), true );
      PowerSetpoint( LaserIndex, static_cast<float>( vPowerSetpoint ) );
      if ( OpenRequest_ )
        return SetOpen();
    }
    else
    {
      if( IsKeyInterlockTriggered( LaserIndex ) )
      {
        MMILE_->ActiveKeyInterlock();
        return ERR_KEY_INTERLOCK;
      }
      else if ( IsClassIVInterlockTriggered() )
      {
        MMILE_->ActiveClassIVInterlock();
        return ERR_CLASSIV_INTERLOCK;
      }
      else
      {
        return ERR_INTERLOCK;
      }
    }

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
  if ( PropertyPointers_.find( Prop->GetName() ) != PropertyPointers_.end() && PropertyPointers_[Prop->GetName()] == nullptr )
  {
    PropertyPointers_[Prop->GetName()] = Prop;
  }
  if ( Act == MM::BeforeGet )
  {
    // Not calling GetControlMode() from ALC SDK, since it may slow
    // down acquisition while switching channels
    Prop->Set( Enable_[LaserIndex].c_str() );
  }
  else if ( Act == MM::AfterSet )
  {
    int vInterlockStatus = MMILE_->GetClassIVAndKeyInterlockStatus();
    if ( vInterlockStatus != DEVICE_OK )
    {
      return vInterlockStatus;
    }
    if ( LaserInterface_ == nullptr )
    {
      return ERR_DEVICE_NOT_CONNECTED;
    }
    if ( !IsInterlockTriggered( LaserIndex ) )
    {
      std::string vEnable;
      Prop->Get( vEnable );
      if ( Enable_[LaserIndex].compare( vEnable ) != 0 )
      {
        // Update the laser control mode if we are changing to, or
        // from External TTL mode
        if ( vEnable.compare( g_LaserEnableTTL ) == 0 )
        {
          if ( !LaserInterface_->SetControlMode( LaserIndex, TTL_PULSED ) )
          {
            return ERR_SETCONTROLMODE;
          }
        }
        else if ( Enable_[LaserIndex].compare( g_LaserEnableTTL ) == 0 )
        {
          if ( !LaserInterface_->SetControlMode( LaserIndex, CW ) )
          {
            return ERR_SETCONTROLMODE;
          }
        }

        Enable_[LaserIndex] = vEnable;
        MMILE_->LogMMMessage( "Enable" + std::to_string( static_cast<long long>( Wavelength( LaserIndex ) ) ) + " = " + Enable_[LaserIndex], true );
        if ( OpenRequest_ )
        {
          return SetOpen();
        }
      }
    }
    else
    {
      if ( IsKeyInterlockTriggered( LaserIndex ) )
      {
        MMILE_->ActiveKeyInterlock();
        return ERR_KEY_INTERLOCK;
      }
      else if ( IsClassIVInterlockTriggered() )
      {
        MMILE_->ActiveClassIVInterlock();
        return ERR_CLASSIV_INTERLOCK;
      }
      else
      {
        return ERR_INTERLOCK;
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
  int vInterlockStatus = MMILE_->GetClassIVAndKeyInterlockStatus();
  if ( vInterlockStatus != DEVICE_OK )
  {
    return vInterlockStatus;
  }
  if ( LaserInterface_ == nullptr )
  {
    return ERR_DEVICE_NOT_CONNECTED;
  }

  for( int vLaserIndex = 1; vLaserIndex <= NumberOfLasers_; ++vLaserIndex)
  {
    if ( Open )
    {
      bool vLaserOn = ( PowerSetpoint( vLaserIndex ) > 0 ) && ( Enable_[vLaserIndex].compare( g_LaserEnableOff ) != 0 );
      double vPercentScale = 0.;
      if ( vLaserOn )
      {
        vPercentScale = PowerSetpoint( vLaserIndex );
      }
      double vPower = ( vPercentScale / 100. ) * ( LaserRange_[vLaserIndex].PowerMax - LaserRange_[vLaserIndex].PowerMin ) + LaserRange_[vLaserIndex].PowerMin;

      MMILE_->LogMMMessage( "SetLas" + std::to_string( static_cast<long long>( vLaserIndex ) ) + "  = " + std::to_string( static_cast<long double>( vPower ) ) + "(" + std::to_string( static_cast<long long>( vLaserOn ) ) + ")", true );

      TLaserState vLaserState;
      if ( LaserInterface_->GetLaserState( vLaserIndex, &vLaserState ) )
      {
        if ( vLaserOn && ( vLaserState != ALC_READY ) )
        {
          std::string vMessage = "Laser # " + std::to_string( static_cast<long long>( vLaserIndex ) ) + " is not ready!";
          // laser is not ready!
          MMILE_->LogMMMessage( vMessage.c_str(), false );
          // GetCoreCallback()->PostError(std::make_pair<int,std::string>(DEVICE_ERR,vMessage));
        }

        if ( vLaserState > ALC_NOT_AVAILABLE )
        {
          MMILE_->LogMMMessage( "setting Laser " + std::to_string( static_cast<long long>( Wavelength( vLaserIndex ) ) ) + " to " + std::to_string( static_cast<long double>( vPower ) ) + "% full scale", true );
          if ( !LaserInterface_->SetLas_I( vLaserIndex, vPower, vLaserOn ) )
          {
            MMILE_->LogMMMessage( std::string( "Setting Laser power for laser " + std::to_string( static_cast<long long>( vLaserIndex ) ) + " failed with value [" ) + std::to_string( static_cast<long double>( vPower ) ) + "]" );
            return ERR_LASER_SET;
          }
        }
      }
      else
      {
        return ERR_LASER_STATE_READ;
      }
    }
    MMILE_->LogMMMessage( "set shutter " + std::to_string( static_cast<long long>( Open ) ), true );
    bool vSuccess = LaserInterface_->SetLas_Shutter( Open );
    if ( !vSuccess )
    {
      MMILE_->LogMMMessage( "set shutter " + std::to_string( static_cast<long long>( Open ) ) + " failed", false );
      return ERR_SETLASERSHUTTER;
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
  UpdateLasersRange();
}

void CLasers::UpdateLasersRange()
{
  if ( PowerInterface_ != nullptr )
  {
    for ( int vLaserIndex = 1; vLaserIndex < NumberOfLasers_ + 1; ++vLaserIndex )
    {
      PowerInterface_->GetPowerRange( vLaserIndex, &( LaserRange_[vLaserIndex].PowerMin ), &( LaserRange_[vLaserIndex].PowerMax ) );
      MMILE_->LogMMMessage( "New range for laser " + std::to_string( static_cast<long long>( vLaserIndex ) ) + " [" + std::to_string( static_cast<long double>( LaserRange_[vLaserIndex].PowerMin ) ) + ", " + std::to_string( static_cast<long double>( LaserRange_[vLaserIndex].PowerMax ) ) + "]", true );
    }
    if( OpenRequest_ )
    {
      SetOpen( OpenRequest_ );
    }
  }
}

int CLasers::UpdateILEInterface( IALC_REV_Laser2 *LaserInterface, IALC_REV_ILEPowerManagement* PowerInterface, IALC_REV_ILE* ILEInterface )
{
  LaserInterface_ = LaserInterface;
  PowerInterface_ = PowerInterface;
  ILEInterface_ = ILEInterface;
  if ( LaserInterface_ != nullptr && PowerInterface_ != nullptr && ILEInterface_ != nullptr )
  {
    int vNbLasers = LaserInterface_->Initialize();
    if ( vNbLasers <= 0 )
    {
      return ERR_LASERS_INIT;
    }
    WaitOnLaserWarmingUp();
    UpdateLasersRange();

    std::string vPropertyName;
    int vWavelength;
    for ( int vLaserIndex = 1; vLaserIndex <= vNbLasers; vLaserIndex++ )
    {
      vWavelength = Wavelength( vLaserIndex );

      Enable_[vLaserIndex] = g_LaserEnableOn;
      vPropertyName = BuildPropertyName( g_PowerSetpointProperty, vWavelength );
      if ( PropertyPointers_.find( vPropertyName ) != PropertyPointers_.end() && PropertyPointers_[vPropertyName] != nullptr )
      {
        PropertyPointers_[vPropertyName]->Set( Enable_[vLaserIndex].c_str() );
      }

      PowerSetpoint( vLaserIndex, 0. );
      vPropertyName = BuildPropertyName( g_EnableProperty, vWavelength );
      if ( PropertyPointers_.find( vPropertyName ) != PropertyPointers_.end() && PropertyPointers_[vPropertyName] != nullptr )
      {
        PropertyPointers_[vPropertyName]->Set( "0" );
      }
    }
  }
  return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// Private functions
///////////////////////////////////////////////////////////////////////////////

int CLasers::Wavelength(const int LaserIndex )
{
  int vInterlockStatus = MMILE_->GetClassIVAndKeyInterlockStatus();
  if ( vInterlockStatus != DEVICE_OK )
  {
    return vInterlockStatus;
  }
  if ( LaserInterface_ == nullptr )
  {
    return ERR_DEVICE_NOT_CONNECTED;
  }

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
  if ( LaserInterface_ == nullptr )
  {
    return false;
  }

  int vValue = 0;
  LaserInterface_->IsControlModeAvailable( LaserIndex, &vValue);
  return (vValue == 1);
}

///////////////////////////////////////////////////////////////////////////////
// Interlock functions
///////////////////////////////////////////////////////////////////////////////

bool CLasers::IsKeyInterlockTriggered(int LaserIndex)
{
  if( LaserInterface_ == nullptr )
  {
    return false;
  }

  bool vInterlockError = false;
  TLaserState vLaserState;
  if( LaserInterface_->GetLaserState( LaserIndex, &vLaserState ) )
  {
    if( vLaserState == ALC_POWER_ERROR )
    {
      vInterlockError = true;
    }
  }
#ifdef _ACTIVATE_DUMMYTEST_
  vInterlockError = KeyInterlockTEMP_;
#endif
  return vInterlockError;
}

bool CLasers::IsInterlockTriggered( int LaserIndex )
{
  if ( LaserInterface_ == nullptr )
  {
    return false;
  }

  bool vInterlockError = false;
  TLaserState vLaserState;
  if ( LaserInterface_->GetLaserState( LaserIndex, &vLaserState ) )
  {
    if ( vLaserState == ALC_INTERLOCK_ERROR || vLaserState == ALC_POWER_ERROR || vLaserState == ALC_CLASS_IV_INTERLOCK_ERROR )
    {
      vInterlockError = true;
    }
  }
#ifdef _ACTIVATE_DUMMYTEST_
  vInterlockError = InterlockTEMP_ | ClassIVInterlockTEMP_ | KeyInterlockTEMP_;
#endif
  return vInterlockError;
}

bool CLasers::IsClassIVInterlockTriggered()
{
  if ( ILEInterface_ == nullptr )
  {
    return false;
  }

  bool vActive = false;
  ILEInterface_->IsClassIVInterlockFlagActive( &vActive );
#ifdef _ACTIVATE_DUMMYTEST_
  vActive = ClassIVInterlockTEMP_;
#endif
  return vActive;
}

#ifdef _ACTIVATE_DUMMYTEST_
int CLasers::OnInterlock( MM::PropertyBase* Prop, MM::ActionType Act )
{
  if ( Act == MM::AfterSet )
  {
    std::string vValue;
    Prop->Get( vValue );
    if ( vValue == g_LaserEnableOn )
    {
      InterlockTEMP_ = true;
    }
    else
    {
      InterlockTEMP_ = false;
    }
  }
  return DEVICE_OK;
}

int CLasers::OnClassIVInterlock( MM::PropertyBase* Prop, MM::ActionType Act )
{
  if ( Act == MM::AfterSet )
  {
    std::string vValue;
    Prop->Get( vValue );
    if ( vValue == g_LaserEnableOn )
    {
      ClassIVInterlockTEMP_ = true;
    }
    else
    {
      ClassIVInterlockTEMP_ = false;
    }
  }
  return DEVICE_OK;
}

int CLasers::OnKeyInterlock( MM::PropertyBase* Prop, MM::ActionType Act )
{
  if( Act == MM::AfterSet )
  {
    std::string vValue;
    Prop->Get( vValue );
    if( vValue == g_LaserEnableOn )
    {
      KeyInterlockTEMP_ = true;
    }
    else
    {
      KeyInterlockTEMP_ = false;
    }
  }
  return DEVICE_OK;
}
#endif

void CLasers::DisplayKeyInterlockMessage( MM::PropertyBase* Prop )
{
  if( DisplayedInterlockState_ != KEY_INTERLOCK )
  {
    DisplayedInterlockState_ = KEY_INTERLOCK;
    Prop->Set( g_InterlockKeyActive );
    MMILE_->UpdatePropertyUI( g_InterlockStatus, g_InterlockKeyActive );
    MMILE_->ActiveKeyInterlock();
  }
}

void CLasers::DisplayClassIVInterlockMessage( MM::PropertyBase* Prop )
{
  if( DisplayedInterlockState_ != CLASSIV_INTERLOCK ) 
  {
    DisplayedInterlockState_ = CLASSIV_INTERLOCK;
    Prop->Set( g_InterlockClassIVActive );
    MMILE_->UpdatePropertyUI( g_InterlockStatus, g_InterlockClassIVActive );
    MMILE_->ActiveClassIVInterlock();
  }
}

void CLasers::DisplayInterlockMessage( MM::PropertyBase* Prop )
{
  if( DisplayedInterlockState_ != INTERLOCK )
  {
    DisplayedInterlockState_ = INTERLOCK;
    Prop->Set( g_InterlockActive );
    MMILE_->UpdatePropertyUI( g_InterlockStatus, g_InterlockActive );
  }
}

void CLasers::DisplayNoInterlockMessage( MM::PropertyBase* Prop )
{
  if( DisplayedInterlockState_ != NO_INTERLOCK )
  {
    DisplayedInterlockState_ = NO_INTERLOCK;
    Prop->Set( g_InterlockInactive );
    MMILE_->UpdatePropertyUI( g_InterlockStatus, g_InterlockInactive );
  }
}

int CLasers::OnInterlockStatus( MM::PropertyBase* Prop, MM::ActionType Act )
{
  if ( Act == MM::BeforeGet )
  {
    int vMainClassInterlockState = MMILE_->GetClassIVAndKeyInterlockStatus();
    if( vMainClassInterlockState != DEVICE_OK || IsInterlockTriggered( 1 ) ) 
    {
      if( vMainClassInterlockState == ERR_KEY_INTERLOCK || IsKeyInterlockTriggered( 1 ) ) 
      {
        DisplayKeyInterlockMessage( Prop );
      }
      else if( vMainClassInterlockState == ERR_CLASSIV_INTERLOCK || IsClassIVInterlockTriggered() )
      {
        DisplayClassIVInterlockMessage( Prop );
      }
      else 
      {
        DisplayInterlockMessage( Prop );
      }
    }
    else 
    {
      DisplayNoInterlockMessage( Prop );
    }
  }
  return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// Interlock thread
///////////////////////////////////////////////////////////////////////////////


CInterlockStatusMonitor::CInterlockStatusMonitor( CIntegratedLaserEngine* MMILE )
  :MMILE_( MMILE ),
  KeepRunning_( true )
{
}

CInterlockStatusMonitor::~CInterlockStatusMonitor()
{
  KeepRunning_ = false;
  wait();
}

int CInterlockStatusMonitor::svc()
{
  while ( KeepRunning_ )
  {
    MMILE_->UpdateProperty( g_InterlockStatus );
    Sleep( 500 );
  }
  return 0;
}
