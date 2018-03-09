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
  MMILE_( MMILE )
{
  if ( PowerInterface_ == nullptr )
  {
    throw std::logic_error( "CLowPowerMode: Pointer to ILE Power interface invalid" );
  }

  int vLowPowerPortIndex;
  if ( !PowerInterface_->GetLowPowerPort( &vLowPowerPortIndex ) )
  {
    throw std::runtime_error( "ILE Power GetLowPowerPort failed" );
  }

  if ( vLowPowerPortIndex < 1 )
  {
    throw std::runtime_error( "Low Power port index invalid [" + std::to_string( vLowPowerPortIndex ) + "]" );
  }

  bool vActive;
  if ( !PowerInterface_->GetLowPowerState( &vActive ) )
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
  MMILE_->CreateStringProperty( vPropertyName.c_str(), vActive ? g_On : g_Off, false, vAct );
  MMILE_->SetAllowedValues( vPropertyName.c_str(), vAllowedValues );
}

CLowPowerMode::~CLowPowerMode()
{
}

int CLowPowerMode::OnValueChange( MM::PropertyBase * Prop, MM::ActionType Act )
{
  if ( Act == MM::AfterSet )
  {
    if ( PowerInterface_ == nullptr )
    {
      return ERR_DEVICE_NOT_CONNECTED;
    }

    std::string vValue;
    Prop->Get( vValue );
    bool vEnabled = ( vValue == g_On );
    if ( PowerInterface_->SetLowPowerState( vEnabled ) )
    {
      MMILE_->CheckAndUpdateLasers();
    }
    else
    {
      MMILE_->LogMMMessage( std::string( vEnabled ? "Enabling" : "Disabling" ) + " low power state FAILED" );
      return DEVICE_ERR;
    }
  }
  return DEVICE_OK;
}

void CLowPowerMode::UpdateILEInterface( IALC_REV_ILEPowerManagement* PowerInterface )
{
  PowerInterface_ = PowerInterface;
}