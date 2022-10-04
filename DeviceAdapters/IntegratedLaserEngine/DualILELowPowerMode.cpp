///////////////////////////////////////////////////////////////////////////////
// FILE:          DualILELowPowerMode.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------

#include "DualILELowPowerMode.h"
#include "DualILE.h"
#include "PortsConfiguration.h"
#include "ALC_REV.h"
#include <exception>

const char* const g_PropertyBaseName = "Low Power Mode [X 0.1]";
const char* const g_On = "On";
const char* const g_Off = "Off";

CDualILELowPowerMode::CDualILELowPowerMode( IALC_REV_ILEPowerManagement* Unit1PowerInterface, IALC_REV_ILEPowerManagement* Unit2PowerInterface, const CPortsConfiguration* PortsConfiguration, CDualILE* MMILE ) :
  Unit1PowerInterface_( Unit1PowerInterface ),
  Unit2PowerInterface_( Unit2PowerInterface ),
  PortsConfiguration_( PortsConfiguration ),
  MMILE_( MMILE ),
  Unit1Active_( false ),
  Unit2Active_( false )
{
  if ( Unit1PowerInterface_ == nullptr && Unit2PowerInterface_ == nullptr )
  {
    throw std::logic_error( "CDualILELowPowerMode: Pointers to ILE Power interface for both units are invalid" );
  }
  if ( PortsConfiguration_ == nullptr )
  {
    throw std::logic_error( "CDualILELowPowerMode: Pointer to Ports configuration invalid" );
  }
  if ( MMILE_ == nullptr )
  {
    throw std::logic_error( "CDualILELowPowerMode: Pointer to main class invalid" );
  }

  int vUnit1LowPowerPortIndex = 0;
  if ( Unit1PowerInterface_ )
  {
    if ( !Unit1PowerInterface_->GetLowPowerPort( &vUnit1LowPowerPortIndex ) )
    {
      throw std::runtime_error( "ILE Power GetLowPowerPort for unit1 failed" );
    }
    if ( vUnit1LowPowerPortIndex < 1 )
    {
      throw std::runtime_error( "Low Power port index for unit1 invalid [" + std::to_string( static_cast<long long>( vUnit1LowPowerPortIndex ) ) + "]" );
    }
    if ( !Unit1PowerInterface_->GetLowPowerState( &Unit1Active_ ) )
    {
      throw std::runtime_error( "ILE Power GetLowPowerState for unit1 failed" );
    }
    MMILE_->LogMMMessage( "Low power mode port for unit1: " + std::to_string( static_cast<long long>( vUnit1LowPowerPortIndex ) ) + " - " + ( Unit1Active_ ? g_On : g_Off ), true );
  }
  else
  {
    MMILE_->LogMMMessage( "Pointer to Low power mode for unit1 is invalid" );
  }

  int vUnit2LowPowerPortIndex = 0;
  if ( Unit2PowerInterface_ )
  {
    if ( !Unit2PowerInterface_->GetLowPowerPort( &vUnit2LowPowerPortIndex ) )
    {
      throw std::runtime_error( "ILE Power GetLowPowerPort for unit2 failed" );
    }
    if ( vUnit2LowPowerPortIndex < 1 )
    {
      throw std::runtime_error( "Low Power port index for unit2 invalid [" + std::to_string( static_cast<long long>( vUnit2LowPowerPortIndex ) ) + "]" );
    }
    if ( !Unit2PowerInterface_->GetLowPowerState( &Unit2Active_ ) )
    {
      throw std::runtime_error( "ILE Power GetLowPowerState for unit2 failed" );
    }
    MMILE_->LogMMMessage( "Low power mode port for unit2: " + std::to_string( static_cast<long long>( vUnit2LowPowerPortIndex ) ) + " - " + ( Unit2Active_ ? g_On : g_Off ), true );
  }
  else
  {
    MMILE_->LogMMMessage( "Pointer to Low power mode for unit2 is invalid" );
  }

  std::vector<std::string> vPortNames = PortsConfiguration_->GetPortList();

  // Create properties
  if ( !vPortNames.empty() )
  {
    std::vector<std::string> vAllowedValues;
    vAllowedValues.push_back( g_On );
    vAllowedValues.push_back( g_Off );

    for ( const std::string& vPortName : vPortNames )
    {
      std::vector<int> vUnitsPorts;
      std::vector<int> vUnitsProperty;
      PortsConfiguration_->GetUnitPortsForMergedPort( vPortName, &vUnitsPorts );
      if ( Unit1PowerInterface_ && vUnitsPorts[0] == vUnit1LowPowerPortIndex )
      {
        vUnitsProperty.push_back( 0 );
        MMILE_->LogMMMessage( "Port " + vPortName + " low power mode available for unit1", true );
      }
      if ( Unit2PowerInterface_ && vUnitsPorts[1] == vUnit2LowPowerPortIndex )
      {
        vUnitsProperty.push_back( 1 );
        MMILE_->LogMMMessage( "Port " + vPortName + " low power mode available for unit2", true );
      }
      if ( !vUnitsProperty.empty() )
      {
        // Create a property if at least one of the units' port in the current virtual port supports low power mode
        std::string vPropertyName = std::string( "Port " ) + vPortName + "-" + g_PropertyBaseName;
        CPropertyAction* vAct = new CPropertyAction( this, &CDualILELowPowerMode::OnValueChange );
        MMILE_->CreateStringProperty( vPropertyName.c_str(), ( ( Unit1Active_ || Unit2Active_ ) ? g_On : g_Off ), false, vAct );
        MMILE_->SetAllowedValues( vPropertyName.c_str(), vAllowedValues );
        UnitsPropertyMap_[vPropertyName] = vUnitsProperty;
        PropertyPointers_[vPropertyName] = nullptr;
      }
    }
  }
}

CDualILELowPowerMode::~CDualILELowPowerMode()
{
}

bool CDualILELowPowerMode::GetCachedValueForProperty( const std::string& PropertyName )
{
  bool vLowPowerModeActive = false;
  const std::vector<int>& vUnitsProperty = UnitsPropertyMap_[PropertyName];
  for ( int vUnitIndex : vUnitsProperty )
  {
    if ( vUnitIndex == 0 )
    {
      vLowPowerModeActive |= Unit1Active_;
    }
    if ( vUnitIndex == 1 )
    {
      vLowPowerModeActive |= Unit2Active_;
    }
  }
  return vLowPowerModeActive;
}

int CDualILELowPowerMode::SetDevice( IALC_REV_ILEPowerManagement* PowerInterface, int UnitIndex, bool NewPosition, bool& UpdatedPosition )
{
  UpdatedPosition = false;

  if ( PowerInterface == nullptr )
  {
    return ERR_DEVICE_NOT_CONNECTED;
  }

  bool vEnabled;
  if ( !PowerInterface->IsLowPowerEnabled( &vEnabled ) )
  {
    MMILE_->LogMMMessage( "ILE IsLowPowerEnabled for unit" + std::to_string( UnitIndex ) + " FAILED" );
    return ERR_LOWPOWERMODE_GET;
  }

  if ( !vEnabled )
  {
    // Wrong port, ignore command
    MMILE_->LogMMMessage( "ILE Low Power not enabled for unit" + std::to_string( UnitIndex ), true);
    return ERR_LOWPOWERMODE_NOT_ENABLED;
  }

  bool vCurrentDeviceState;
  if ( !PowerInterface->GetLowPowerState( &vCurrentDeviceState ) )
  {
    MMILE_->LogMMMessage( "ILE GetLowPowerState for unit" + std::to_string( UnitIndex ) + " FAILED" );
    return ERR_LOWPOWERMODE_GET;
  }

  MMILE_->LogMMMessage( "Current Dual Low Power state for unit" + std::to_string( UnitIndex ) + ": [" + std::string( vCurrentDeviceState ? g_On : g_Off ) + "] ", true );

  if ( NewPosition != vCurrentDeviceState )
  {
    MMILE_->LogMMMessage( "Set Dual Low Power state for unit" + std::to_string( UnitIndex ) + " to [" + std::string( NewPosition ? g_On : g_Off ) + "]", true );
    if ( !PowerInterface->SetLowPowerState( NewPosition ) )
    {
      MMILE_->LogMMMessage( "Turning Low Power state " + std::string( NewPosition ? g_On : g_Off ) + " for unit" + std::to_string( UnitIndex ) + " FAILED" );
      return ERR_LOWPOWERMODE_SET;
    }
    UpdatedPosition = true;
  }

  return DEVICE_OK;
}

int CDualILELowPowerMode::OnValueChange( MM::PropertyBase * Prop, MM::ActionType Act )
{
  int vRet = DEVICE_OK;
  std::string vPropertyName = Prop->GetName();

  if ( PropertyPointers_.find( vPropertyName ) == PropertyPointers_.end() )
  {
    return DEVICE_OK;
  }

  if ( PropertyPointers_[vPropertyName] == nullptr )
  {
    PropertyPointers_[vPropertyName] = Prop;
  }

  if ( Act == MM::BeforeGet )
  {
    Prop->Set( GetCachedValueForProperty( vPropertyName ) ? g_On : g_Off );
  }
  else if ( Act == MM::AfterSet )
  {
    int vInterlockStatus = MMILE_->GetClassIVAndKeyInterlockStatus();
    if ( vInterlockStatus != DEVICE_OK )
    {
      return vInterlockStatus;
    }
    if ( Unit1PowerInterface_ == nullptr && Unit2PowerInterface_ == nullptr )
    {
      return ERR_DEVICE_NOT_CONNECTED;
    }

    std::string vValue;
    Prop->Get( vValue );
    bool vEnable = ( vValue == g_On );

    const std::vector<int>& vUnitsProperty = UnitsPropertyMap_[vPropertyName];

    std::vector<std::tuple<IALC_REV_ILEPowerManagement*, unsigned int, bool*>> vUnitsPreviousValues;
    bool vPowerStateUpdated = false;
    for ( int vUnitIndex : vUnitsProperty )
    {
      IALC_REV_ILEPowerManagement* vPowerInterface = nullptr;
      bool *vUnitValuePtr = nullptr;
      if ( vUnitIndex == 0 )
      {
        vPowerInterface = Unit1PowerInterface_;
        vUnitValuePtr = &Unit1Active_;
      }
      if ( vUnitIndex == 1 )
      {
        vPowerInterface = Unit2PowerInterface_;
        vUnitValuePtr = &Unit2Active_;
      }

      bool vUnitPowerStateUpdated;
      vRet = SetDevice( vPowerInterface, vUnitIndex + 1, vEnable, vUnitPowerStateUpdated );
      vPowerStateUpdated |= vUnitPowerStateUpdated;

      if ( vRet == ERR_LOWPOWERMODE_NOT_ENABLED || vRet == ERR_DEVICE_NOT_CONNECTED )
      {
        // Ignore when Low Power is not enabled (wrong current port)
        // Also ignore the case where only one device's pointer is invalid
        vRet = DEVICE_OK;
      }

      if ( vRet == DEVICE_OK )
      {
        vUnitsPreviousValues.push_back( { vPowerInterface,  vUnitIndex + 1, vUnitValuePtr } );
      }
      else
      {
        // One of the units failed, no need to continue but reset whatever value was set for other units
        for ( const auto& vPreviousValue : vUnitsPreviousValues )
        {
          SetDevice(
            std::get< IALC_REV_ILEPowerManagement*>( vPreviousValue ),
            std::get< unsigned int>( vPreviousValue ),
            *std::get< bool*>( vPreviousValue ),
            vUnitPowerStateUpdated );
        }

        bool vPreviousPosition = GetCachedValueForProperty( vPropertyName );
        MMILE_->LogMMMessage( "Couldn't change Low Power state, reverting the UI position for property " + vPropertyName + " to [" + std::string( vPreviousPosition ? g_On : g_Off ) + "]" );
        Prop->Set( vPreviousPosition ? g_On : g_Off );
        break;
      }
    }

    if ( vRet == DEVICE_OK )
    {
      // Update stored values
      for ( const auto& vPreviousValue : vUnitsPreviousValues )
      {
        *std::get< bool*>( vPreviousValue ) = vEnable;
      }

      if ( vPowerStateUpdated )
      {
        MMILE_->CheckAndUpdateLasers();
      }
    }
  }

  return vRet;
}

int CDualILELowPowerMode::UpdateILEInterface( IALC_REV_ILEPowerManagement* Unit1PowerInterface, IALC_REV_ILEPowerManagement* Unit2PowerInterface )
{
  int vRet = DEVICE_OK;

  if ( Unit1PowerInterface != Unit1PowerInterface_ || Unit2PowerInterface != Unit2PowerInterface_ )
  {
    Unit1PowerInterface_ = Unit1PowerInterface;
    Unit2PowerInterface_ = Unit2PowerInterface;

    if ( Unit1PowerInterface_ != nullptr || Unit2PowerInterface_ != nullptr )
    {
      MMILE_->LogMMMessage( "Resetting Low Power mode to device state", true );

      if ( Unit1PowerInterface_ != nullptr )
      {
        if ( !Unit1PowerInterface_->GetLowPowerState( &Unit1Active_ ) )
        {
          MMILE_->LogMMMessage( "ILE GetLowPowerState for unit1 FAILED" );
          vRet = ERR_LOWPOWERMODE_GET;
        }
        else
        {
          MMILE_->LogMMMessage( "Device state for unit1: " + std::string( Unit1Active_ ? g_On : g_Off ), true );
        }
      }
      else
      {
        MMILE_->LogMMMessage( "Pointer to Low Power mode for unit1 is invalid" );
      }

      if ( Unit2PowerInterface_ != nullptr )
      {
        if ( !Unit2PowerInterface_->GetLowPowerState( &Unit2Active_ ) )
        {
          MMILE_->LogMMMessage( "ILE GetLowPowerState for unit2 FAILED" );
          vRet = ERR_LOWPOWERMODE_GET;
        }
        else
        {
          MMILE_->LogMMMessage( "Device state for unit2: " + std::string( Unit2Active_ ? g_On : g_Off ), true );
        }
      }
      else
      {
        MMILE_->LogMMMessage( "Pointer to Low Power mode for unit2 is invalid" );
      }

      for ( const auto& vPropertyPointer : PropertyPointers_ )
      {
        if ( vPropertyPointer.second != nullptr )
        {
          vPropertyPointer.second->Set( GetCachedValueForProperty( vPropertyPointer.first ) ? g_On : g_Off );
        }
      }
    }
  }

  return vRet;
}

void CDualILELowPowerMode::CheckAndUpdate()
{
  // Update the device state on port change
  bool vUpdatedPosition = false;

  if ( Unit1PowerInterface_ )
  {
    SetDevice( Unit1PowerInterface_, 1, Unit1Active_, vUpdatedPosition );
  }

  if ( Unit2PowerInterface_ )
  {
    bool vUnitUpdatedPosition = false;
    SetDevice( Unit2PowerInterface_, 2, Unit2Active_, vUnitUpdatedPosition );
    vUpdatedPosition |= vUnitUpdatedPosition;
  }

  if ( vUpdatedPosition )
  {
    MMILE_->CheckAndUpdateLasers();
  }
}
