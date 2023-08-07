///////////////////////////////////////////////////////////////////////////////
// FILE:          Lasers.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------

#include "ALC_REV.h"
#include "Lasers.h"
#include "IntegratedLaserEngine.h"
#include <numeric>
#include <algorithm>

// Properties
const char* const g_EnableProperty = "Power Enable";
const char* const g_PowerSetpointProperty = "Power Setpoint";
const char* const g_InterlockStatus = "Interlock Flag Status";

// Interlock status property values
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

  int vNumberOfLasers = LaserInterface_->Initialize();
  MMILE_->LogMMMessage( "in CLasers constructor, number of lasers =" + std::to_string( static_cast<long long>( vNumberOfLasers ) ), true );
  NumberOfLasers_ = vNumberOfLasers > 0 ? vNumberOfLasers : 0;

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
  std::vector<TLaserState> vState( NumberOfLasers_ + 1, ALC_NOT_AVAILABLE );

  // Lasers can take up to 90 seconds to initialize
  MM::TimeoutMs vTimerOut( MMILE_->GetCurrentTime(), 91000 );

  while ( true )
  {
    bool vFinishWaiting = true;
    for ( int vLaserIndex = 1; vLaserIndex <= NumberOfLasers_; ++vLaserIndex )
    {
      if ( ALC_NOT_AVAILABLE == vState[vLaserIndex] )
      {
        LaserInterface_->GetLaserState( vLaserIndex, &( vState[vLaserIndex] ) );
        switch ( vState[vLaserIndex] )
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

    if ( vTimerOut.expired( MMILE_->GetCurrentTime() ) )
    {
      MMILE_->LogMMMessage( " some lasers did not respond", false );
      break;
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
  LasersState_.resize( max( LasersState_.size(), NumberOfLasers_ + 1 ) );
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
#ifdef SHOW_LASER_ENABLE_PROPERTY
    vActEx = new CPropertyActionEx( this, &CLasers::OnEnable, vLaserIndex );
    vPropertyName = BuildPropertyName( g_EnableProperty, vWavelength );
    std::vector<std::string> vEnableStates;
    vEnableStates.push_back( g_LaserEnableOn );
    vEnableStates.push_back( g_LaserEnableOff );
    if ( AllowsExternalTTL( vLaserIndex ) )
    {
      vEnableStates.push_back( g_LaserEnableTTL );
    }
    MMILE_->CreateProperty( vPropertyName.c_str(), vEnableStates[0].c_str(), MM::String, false, vActEx );
    MMILE_->SetAllowedValues( vPropertyName.c_str(), vEnableStates );
    PropertyPointers_[vPropertyName] = nullptr;
#endif
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
    int vRet = CheckInterlock( LaserIndex );
    if ( vRet == DEVICE_OK )
    {
      float vOldValue = PowerSetpoint( LaserIndex );
      Prop->Get( vPowerSetpoint );
      MMILE_->LogMMMessage( "to equipment: PowerSetpoint" + std::to_string( static_cast<long long>( Wavelength( LaserIndex ) ) ) + "  = " + std::to_string( static_cast<long double>( vPowerSetpoint ) ), true );
      PowerSetpoint( LaserIndex, static_cast<float>( vPowerSetpoint ) );
      if ( OpenRequest_ )
      {
        vRet = SetOpen();
        if ( vRet != DEVICE_OK )
        {
          PowerSetpoint( LaserIndex, vOldValue );
          Prop->Set( vOldValue );
          return vRet;
        }
      }
      else
      {
        // Update HW setpoint even when shutter closed
        double vPercentScale = PowerSetpoint( LaserIndex );
        double vPower = ( vPercentScale / 100. ) * ( LasersState_[LaserIndex].LaserRange_.PowerMax - LasersState_[LaserIndex].LaserRange_.PowerMin ) + LasersState_[LaserIndex].LaserRange_.PowerMin;

        MMILE_->LogMMMessage( "SetLas" + std::to_string( static_cast<long long>( LaserIndex ) ) + "  = " + std::to_string( static_cast<long double>( vPower ) ), true );

        TLaserState vLaserState;
        if ( LaserInterface_->GetLaserState( LaserIndex, &vLaserState ) )
        {
          if ( vLaserState > ALC_NOT_AVAILABLE )
          {
            MMILE_->LogMMMessage( "Setting Laser " + std::to_string( static_cast<long long>( Wavelength( LaserIndex ) ) ) + " to " + std::to_string( static_cast<long double>( vPower ) ) + "% full scale", true );
            if ( !LaserInterface_->SetLas_I( LaserIndex, vPower, false ) )
            {
              MMILE_->LogMMMessage( std::string( "Setting Laser power for laser " + std::to_string( static_cast<long long>( LaserIndex ) ) + " failed with value [" ) + std::to_string( static_cast<long double>( vPower ) ) + "]" );
              return ERR_LASER_SET;
            }
          }
        }
        else
        {
          return ERR_LASER_STATE_READ;
        }
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
  std::string& vCurrentLaserEnableState = LasersState_[LaserIndex].Enable_;
  if ( Act == MM::BeforeGet )
  {
    // Not calling GetControlMode() from ALC SDK, since it may slow
    // down acquisition while switching channels
    Prop->Set( vCurrentLaserEnableState.c_str() );
  }
  else if ( Act == MM::AfterSet )
  {
    int vRet = CheckInterlock( LaserIndex );
    if ( vRet == DEVICE_OK )
    {
      std::string vNewLaserEnableState;
      Prop->Get( vNewLaserEnableState );
      if ( vCurrentLaserEnableState != vNewLaserEnableState )
      {
        // Update the laser control mode if we are changing to, or
        // from External TTL mode
        if ( vNewLaserEnableState == g_LaserEnableTTL )
        {
          // Enable TTL mode
          MMILE_->LogMMMessage( "Set Laser Control mode to [PULSED]", true );
          if ( !LaserInterface_->SetControlMode( LaserIndex, EXTERNALMODE::TTL_PULSED ) )
          {
            return ERR_SETCONTROLMODE;
          }
        }
        else if ( vCurrentLaserEnableState == g_LaserEnableTTL )
        {
          // Disable TTL mode
          MMILE_->LogMMMessage( "Set Laser Control mode to [CW]", true );
          if ( !LaserInterface_->SetControlMode( LaserIndex, EXTERNALMODE::CW ) )
          {
            return ERR_SETCONTROLMODE;
          }
        }

        vCurrentLaserEnableState = vNewLaserEnableState;
        MMILE_->LogMMMessage( "Enable" + std::to_string( static_cast<long long>( Wavelength( LaserIndex ) ) ) + " = " + vNewLaserEnableState, true );
        if ( OpenRequest_ )
        {
          return SetOpen();
        }
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
  static const std::string vOn{ "On" };
  static const std::string vOff{ "Off" };

  int vInterlockStatus = MMILE_->GetClassIVAndKeyInterlockStatus();
  if ( vInterlockStatus != DEVICE_OK )
  {
    return vInterlockStatus;
  }
  if ( LaserInterface_ == nullptr )
  {
    return ERR_DEVICE_NOT_CONNECTED;
  }

  // If we close the shutter, do it before modifying laser values
  if ( !Open )
  {
    if ( int vRet = ChangeDeviceShutterState( Open ) != DEVICE_OK )
    {
      return vRet;
    }
  }

  std::vector<int> vSortedLaserIndicesByPower = GetLasersSortedByPower();

  // Update laser values
  for ( int vLaserIndex : vSortedLaserIndicesByPower )
  {
    if ( Open )
    {
      bool vLaserOn = ( PowerSetpoint( vLaserIndex ) > 0 ) && ( LasersState_[vLaserIndex].Enable_ != g_LaserEnableOff );
      double vPercentScale = 0.;
      if ( vLaserOn )
      {
        vPercentScale = PowerSetpoint( vLaserIndex );
      }
      double vPower = ( vPercentScale / 100. ) * ( LasersState_[vLaserIndex].LaserRange_.PowerMax - LasersState_[vLaserIndex].LaserRange_.PowerMin ) + LasersState_[vLaserIndex].LaserRange_.PowerMin;

      MMILE_->LogMMMessage( "SetLas" + std::to_string( static_cast<long long>( vLaserIndex ) ) + "  = " + std::to_string( static_cast<long double>( vPower ) ) + " (" + (vLaserOn ? vOn : vOff) + ")", true );

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
          MMILE_->LogMMMessage( "Setting Laser " + std::to_string( static_cast<long long>( Wavelength( vLaserIndex ) ) ) + " to " + std::to_string( static_cast<long double>( vPower ) ) + "% full scale", true );
          if ( !LaserInterface_->SetLas_I( vLaserIndex, vPower, vLaserOn ) )
          {
            std::string vMessage = "Setting Laser power for laser " + std::to_string( static_cast< long long >( vLaserIndex ) ) + " failed";
            int vProhibited;
            if ( LaserInterface_->WasLaserIlluminationProhibitedOnLastChange( vLaserIndex, &vProhibited ) && vProhibited == 1 )
            {
              MMILE_->LogMMMessage( vMessage + " because the maximum power limit was exceeded" );
              return ERR_MAX_POWER_LIMIT_EXCEEDED;
            }
            else
            {
              MMILE_->LogMMMessage( vMessage + " for value [" + std::to_string( static_cast< long double >( vPower ) ) + "]" );
              return ERR_LASER_SET;
            }
          }
        }
      }
      else
      {
        return ERR_LASER_STATE_READ;
      }
    }
  }

  // If we open the shutter, do it only once all laser values are set
  if ( Open )
  {
    if ( int vRet = ChangeDeviceShutterState( Open ) != DEVICE_OK )
    {
      return vRet;
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
    for ( int vLaserIndex = 1; vLaserIndex <= NumberOfLasers_; ++vLaserIndex )
    {
      TLaserRange& vLaserRange = LasersState_[vLaserIndex].LaserRange_;
      PowerInterface_->GetPowerRange( vLaserIndex, &( vLaserRange.PowerMin ), &( vLaserRange.PowerMax ) );
      MMILE_->LogMMMessage( "New range for laser " + std::to_string( static_cast<long long>( vLaserIndex ) ) +
            " [" + std::to_string( static_cast<long double>( vLaserRange.PowerMin ) ) +
            ", " + std::to_string( static_cast<long double>( vLaserRange.PowerMax ) ) + "]", true );
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
    if ( vNbLasers != NumberOfLasers_ )
    {
      LaserInterface_ = nullptr;
      PowerInterface_ = nullptr;
      ILEInterface_ = nullptr;
      return ERR_LASERS_INIT;
    }
    WaitOnLaserWarmingUp();
    UpdateLasersRange();

    std::string vPropertyName;
    int vWavelength;
    for ( int vLaserIndex = 1; vLaserIndex <= vNbLasers; vLaserIndex++ )
    {
      vWavelength = Wavelength( vLaserIndex );

      LasersState_[vLaserIndex].Enable_ = g_LaserEnableOn;
      vPropertyName = BuildPropertyName( g_EnableProperty, vWavelength );
      if ( PropertyPointers_.find( vPropertyName ) != PropertyPointers_.end() && PropertyPointers_[vPropertyName] != nullptr )
      {
        PropertyPointers_[vPropertyName]->Set( LasersState_[vLaserIndex].Enable_.c_str() );
      }

      PowerSetpoint( vLaserIndex, 0. );
      vPropertyName = BuildPropertyName( g_PowerSetpointProperty, vWavelength );
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
  return LasersState_[LaserIndex].PowerSetPoint_;
}

void CLasers::PowerSetpoint(const int LaserIndex, const float Value)
{
  LasersState_[LaserIndex].PowerSetPoint_ = Value;
}

std::vector<int> CLasers::GetLasersSortedByPower() const
{
  auto vLaserPowerComp = [this]( int laserIndex1, int laserIndex2 ) { return LasersState_[laserIndex1].PowerSetPoint_ < LasersState_[laserIndex2].PowerSetPoint_; };

  std::vector<int> vLaserIndices( NumberOfLasers_ );
  std::iota( vLaserIndices.begin(), vLaserIndices.end(), 1 );
  std::sort( vLaserIndices.begin(), vLaserIndices.end(), vLaserPowerComp );

  /*
  std::vector<int> vLaserIndices;
  for ( int vLaserIndex = 1; vLaserIndex <= NumberOfLasers_; ++vLaserIndex )
  {
    std::vector<int>::const_iterator vIndexIt = std::upper_bound( vLaserIndices.begin(), vLaserIndices.end(), vLaserIndex, vLaserPowerComp );
    vLaserIndices.insert( vIndexIt, vLaserIndex );
  }
  */

  return vLaserIndices;
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

int CLasers::ChangeDeviceShutterState( bool Open )
{
  static const std::string vOpen{ "Open" };
  static const std::string vClose{ "Close" };

  MMILE_->LogMMMessage( "Set shutter [" + (Open ? vOpen : vClose) + "]", true);
  bool vSuccess = LaserInterface_->SetLas_Shutter( Open );
  if ( !vSuccess )
  {
    MMILE_->LogMMMessage( "Set shutter [" + (Open ? vOpen : vClose) + "] failed" );
    return ERR_SETLASERSHUTTER;
  }
  return DEVICE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// Interlock functions
///////////////////////////////////////////////////////////////////////////////

int CLasers::CheckInterlock( int LaserIndex )
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

  if ( IsInterlockTriggered( LaserIndex ) )
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

    return ERR_INTERLOCK;
  }

  return DEVICE_OK;
}

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
