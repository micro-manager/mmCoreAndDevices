///////////////////////////////////////////////////////////////////////////////
// FILE:          LowPowerMode.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------

#include "LowPowerMode.h"
#include "IntegratedLaserEngine.h"
#include "Ports.h"
#include "ALC_REV.h"
#include <exception>

const char* const g_PropertyBaseName = "Low Power Mode [X 0.1]";
const char* const g_On = "On";
const char* const g_Off = "Off";

CLowPowerMode::CLowPowerMode( IALC_REV_ILEPowerManagement* PowerInterface, CIntegratedLaserEngine* MMILE ) :
  PowerInterface_( PowerInterface ),
  MMILE_( MMILE ),
  CurrentLowPowerPosition_( false ),
  PropertyPointer_( nullptr )
{
  if ( PowerInterface_ == nullptr )
  {
    throw std::logic_error( "CLowPowerMode: Pointer to ILE Power interface invalid" );
  }
  if ( MMILE_ == nullptr )
  {
    throw std::logic_error( "CLowPowerMode: Pointer tomain class invalid" );
  }

  int vLowPowerPortIndex;
  if ( !PowerInterface_->GetLowPowerPort( &vLowPowerPortIndex ) )
  {
    throw std::runtime_error( "ILE GetLowPowerPort failed" );
  }

  if ( vLowPowerPortIndex < 1 )
  {
    throw std::runtime_error( "Low Power port index invalid [" + std::to_string( static_cast<long long>( vLowPowerPortIndex ) ) + "]" );
  }

  if ( !PowerInterface_->GetLowPowerState( &CurrentLowPowerPosition_ ) )
  {
    throw std::runtime_error( "ILE GetLowPowerState failed" );
  }

  // Create property
  char vPortName[2];
  vPortName[1] = 0;
  vPortName[0] = CPorts::PortIndexToName( vLowPowerPortIndex );
  std::string vPropertyName = std::string( "Port " ) + vPortName + "-" + g_PropertyBaseName;

  std::vector<std::string> vAllowedValues;
  vAllowedValues.push_back( g_On );
  vAllowedValues.push_back( g_Off );

  CPropertyAction* vAct = new CPropertyAction( this, &CLowPowerMode::OnValueChange );
  MMILE_->CreateStringProperty( vPropertyName.c_str(), CurrentLowPowerPosition_ ? g_On : g_Off, false, vAct );
  MMILE_->SetAllowedValues( vPropertyName.c_str(), vAllowedValues );
}

CLowPowerMode::~CLowPowerMode()
{
}

int CLowPowerMode::SetDevice( bool NewPosition )
{
  if ( PowerInterface_ == nullptr )
  {
    return ERR_DEVICE_NOT_CONNECTED;
  }

  bool vEnabled;
  if ( !PowerInterface_->IsLowPowerEnabled( &vEnabled ) )
  {
    MMILE_->LogMMMessage( "ILE IsLowPowerEnabled FAILED" );
    return ERR_LOWPOWERMODE_GET;
  }

  if ( !vEnabled )
  {
    // Wrong port, ignore command
    MMILE_->LogMMMessage( "ILE Low Power not enabled", true );
    return ERR_LOWPOWERMODE_NOT_ENABLED;
  }

  bool vCurrentDeviceState;
  if ( !PowerInterface_->GetLowPowerState( &vCurrentDeviceState ) )
  {
    MMILE_->LogMMMessage( "ILE GetLowPowerState FAILED" );
    return ERR_LOWPOWERMODE_GET;
  }

  MMILE_->LogMMMessage( "Current Low Power state: [" + std::string( vCurrentDeviceState ? g_On : g_Off ) + "]", true );

  if ( NewPosition != vCurrentDeviceState )
  {
    MMILE_->LogMMMessage( "Set Low Power state to [" + std::string( NewPosition ? g_On : g_Off ) + "]", true );
    if ( !PowerInterface_->SetLowPowerState( NewPosition ) )
    {
      MMILE_->LogMMMessage( "Turning Low Power state " + std::string( NewPosition ? g_On : g_Off ) + " FAILED" );
      return ERR_LOWPOWERMODE_SET;
    }

    MMILE_->CheckAndUpdateLasers();
  }

  return DEVICE_OK;
}

int CLowPowerMode::OnValueChange( MM::PropertyBase * Prop, MM::ActionType Act )
{
  if ( PropertyPointer_ == nullptr )
  {
    PropertyPointer_ = Prop;
  }

  if ( Act == MM::BeforeGet )
  {
    Prop->Set( CurrentLowPowerPosition_ ? g_On : g_Off );
  }
  else if ( Act == MM::AfterSet )
  {
    int vInterlockStatus = MMILE_->GetClassIVAndKeyInterlockStatus();
    if ( vInterlockStatus != DEVICE_OK )
    {
      return vInterlockStatus;
    }
    if ( PowerInterface_ == nullptr )
    {
      return ERR_DEVICE_NOT_CONNECTED;
    }

    std::string vValue;
    Prop->Get( vValue );

    bool vEnable = ( vValue == g_On );
    int vRet = SetDevice( vEnable );

    if ( vRet == ERR_LOWPOWERMODE_NOT_ENABLED )
    {
      // Ignore when Low Power is not enabled (wrong current port) to allow MD acquisition to continue
      vRet = DEVICE_OK;
    }

    if ( vRet != DEVICE_OK )
    {
      // If we can't change the device's state, revert the selection to the previous position
      MMILE_->LogMMMessage( "Couldn't change Low Power state, reverting the UI position to [" + std::string( CurrentLowPowerPosition_ ? g_On : g_Off ) + "]" );
      Prop->Set( CurrentLowPowerPosition_ ? g_On : g_Off );
      return vRet;
    }

    CurrentLowPowerPosition_ = vEnable;
  }

  return DEVICE_OK;
}

int CLowPowerMode::UpdateILEInterface( IALC_REV_ILEPowerManagement* PowerInterface )
{
  int vRet = DEVICE_OK;

  if ( PowerInterface != PowerInterface_ )
  {
    PowerInterface_ = PowerInterface;

    if ( PowerInterface_ != nullptr )
    {
      MMILE_->LogMMMessage( "Resetting Low Power mode to device state", true );

      if ( PowerInterface_->GetLowPowerState( &CurrentLowPowerPosition_ ) )
      {
        MMILE_->LogMMMessage( "Low Power mode device state [" + std::string( CurrentLowPowerPosition_ ? g_On : g_Off ) + "]", true );
        if ( PropertyPointer_ != nullptr )
        {
          PropertyPointer_->Set( CurrentLowPowerPosition_ ? g_On : g_Off );
        }
      }
      else
      {
        MMILE_->LogMMMessage( "ILE GetLowPowerState FAILED" );
        vRet = ERR_LOWPOWERMODE_GET;
      }
    }
  }

  return vRet;
}


void CLowPowerMode::CheckAndUpdate()
{
  // Update the device state on port change
  SetDevice( CurrentLowPowerPosition_ );
}