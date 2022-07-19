///////////////////////////////////////////////////////////////////////////////
// FILE:          VeryLowPower.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------

#include "VeryLowPower.h"
#include "IntegratedLaserEngine.h"
#include "Ports.h"
#include "ALC_REV.h"
#include <exception>

const char* const g_PropertyName = "Very Low Power";
const char* const g_On = "On";
const char* const g_Off = "Off";

CVeryLowPower::CVeryLowPower( IALC_REV_ILEPowerManagement* PowerInterface, CIntegratedLaserEngine* MMILE ) :
  PowerInterface_( PowerInterface ),
  MMILE_( MMILE ),
  VeryLowPowerActive_( false ),
  PropertyPointer_( nullptr )
{
  if ( PowerInterface_ == nullptr )
  {
    throw std::logic_error( "CVeryLowPower: Pointer to ILE Power interface invalid" );
  }
  if ( MMILE_ == nullptr )
  {
    throw std::logic_error( "CVeryLowPower: Pointer tomain class invalid" );
  }

  // Forcing the value on initialisation
  if( !PowerInterface_->SetCoherenceMode( VeryLowPowerActive_ ) )
  {
    throw std::runtime_error( "SetCoherenceMode failed" );
  }

  // Create property
  std::vector<std::string> vAllowedValues;
  vAllowedValues.push_back( g_On );
  vAllowedValues.push_back( g_Off );
  CPropertyAction* vAct = new CPropertyAction( this, &CVeryLowPower::OnValueChange );
  MMILE_->CreateStringProperty( g_PropertyName, VeryLowPowerActive_ ? g_On : g_Off, false, vAct );
  MMILE_->SetAllowedValues( g_PropertyName, vAllowedValues );
}

CVeryLowPower::~CVeryLowPower()
{
}

int CVeryLowPower::OnValueChange( MM::PropertyBase * Prop, MM::ActionType Act )
{
  if ( PropertyPointer_ == nullptr )
  {
    PropertyPointer_ = Prop;
  }
  if ( Act == MM::BeforeGet )
  {
    Prop->Set( VeryLowPowerActive_ ? g_On : g_Off );
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
    if ( PowerInterface_->SetCoherenceMode( vEnable ) )
    {
      VeryLowPowerActive_ = vEnable;
      MMILE_->CheckAndUpdateLasers();
    }
    else
    {
      MMILE_->LogMMMessage( std::string( vEnable ? "Enabling" : "Disabling" ) + " low power state FAILED" );
      return ERR_VERYLOWPOWER_SET;
    }
  }
  return DEVICE_OK;
}

int CVeryLowPower::UpdateILEInterface( IALC_REV_ILEPowerManagement* PowerInterface )
{
  PowerInterface_ = PowerInterface;
  if ( PowerInterface_ != nullptr )
  {
    if ( PowerInterface_->SetCoherenceMode( VeryLowPowerActive_ ) )
    {
      MMILE_->LogMMMessage( "Resetting very low power device's state to [" + std::string( VeryLowPowerActive_ ? g_On : g_Off ) + "]", true );
      MMILE_->CheckAndUpdateLasers();
    }
    else
    {
      return ERR_VERYLOWPOWER_SET;
    }
  }
  return DEVICE_OK;
}