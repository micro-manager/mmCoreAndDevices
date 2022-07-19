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
  LowPowerModeActive_( false ),
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
    throw std::runtime_error( "ILE Power GetLowPowerPort failed" );
  }

  if ( vLowPowerPortIndex < 1 )
  {
    throw std::runtime_error( "Low Power port index invalid [" + std::to_string( static_cast<long long>( vLowPowerPortIndex ) ) + "]" );
  }

  if ( !PowerInterface_->GetLowPowerState( &LowPowerModeActive_ ) )
  {
    throw std::runtime_error( "ILE Power GetLowPowerState failed" );
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
  MMILE_->CreateStringProperty( vPropertyName.c_str(), LowPowerModeActive_ ? g_On : g_Off, false, vAct );
  MMILE_->SetAllowedValues( vPropertyName.c_str(), vAllowedValues );
}

CLowPowerMode::~CLowPowerMode()
{
}

int CLowPowerMode::OnValueChange( MM::PropertyBase * Prop, MM::ActionType Act )
{
  if ( PropertyPointer_ == nullptr )
  {
    PropertyPointer_ = Prop;
  }
  if ( Act == MM::BeforeGet )
  {
    Prop->Set( LowPowerModeActive_ ? g_On : g_Off );
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
    if ( PowerInterface_->SetLowPowerState( vEnable ) )
    {
      LowPowerModeActive_ = vEnable;
      MMILE_->CheckAndUpdateLasers();
    }
    else
    {
      MMILE_->LogMMMessage( std::string( vEnable ? "Enabling" : "Disabling" ) + " low power state FAILED" );
      return ERR_LOWPOWERMODE_SET;
    }
  }
  return DEVICE_OK;
}

int CLowPowerMode::UpdateILEInterface( IALC_REV_ILEPowerManagement* PowerInterface )
{
  PowerInterface_ = PowerInterface;
  if ( PowerInterface_ != nullptr )
  {
    if ( PowerInterface_->GetLowPowerState( &LowPowerModeActive_ ) )
    {
      MMILE_->LogMMMessage( "Resetting low power mode to device state [" + std::string( LowPowerModeActive_ ? g_On : g_Off ) + "]", true );
      if ( PropertyPointer_ != nullptr )
      {
        PropertyPointer_->Set( LowPowerModeActive_ ? g_On : g_Off );
      }
    }
    else
    {
      return ERR_LOWPOWERMODE_GET;
    }
  }
  return DEVICE_OK;
}