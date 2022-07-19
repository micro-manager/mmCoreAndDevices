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

  int vUnit1LowPowerPortIndex = 0, vUnit2LowPowerPortIndex = 0;
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

  std::vector<std::string> vPortNames = PortsConfiguration_->GetPortList();

  // Create properties
  if ( !vPortNames.empty() )
  {
    std::vector<std::string> vAllowedValues;
    vAllowedValues.push_back( g_On );
    vAllowedValues.push_back( g_Off );

    std::vector<std::string>::const_iterator vPortIt = vPortNames.begin();
    std::vector<int> vUnitsPorts;
    long vPropertyIndex = 0;
    while ( vPortIt != vPortNames.end() )
    {
      std::vector<int> vUnitsProperty;
      PortsConfiguration_->GetUnitPortsForMergedPort( *vPortIt, &vUnitsPorts );
      if ( Unit1PowerInterface_ && vUnitsPorts[0] == vUnit1LowPowerPortIndex )
      {
        vUnitsProperty.push_back( 0 );
        MMILE_->LogMMMessage( "Port " + *vPortIt + " low power mode available for unit1", true );
      }
      if ( Unit2PowerInterface_ && vUnitsPorts[1] == vUnit2LowPowerPortIndex )
      {
        vUnitsProperty.push_back( 1 );
        MMILE_->LogMMMessage( "Port " + *vPortIt + " low power mode available for unit2", true );
      }
      if ( !vUnitsProperty.empty() )
      {
        // Create a property if at least one of the units' port in the current virtual port supports low power mode
        std::string vPropertyName = std::string( "Port " ) + *vPortIt + "-" + g_PropertyBaseName;
        CPropertyAction* vAct = new CPropertyAction( this, &CDualILELowPowerMode::OnValueChange );
        MMILE_->CreateStringProperty( vPropertyName.c_str(), ( ( Unit1Active_ || Unit2Active_ ) ? g_On : g_Off ), false, vAct );
        MMILE_->SetAllowedValues( vPropertyName.c_str(), vAllowedValues );
        ++vPropertyIndex;
        UnitsPropertyMap_[vPropertyName] = vUnitsProperty;
        PropertyPointers_[vPropertyName] = nullptr;
      }
      ++vPortIt;
    }
  }
}

CDualILELowPowerMode::~CDualILELowPowerMode()
{
}

bool CDualILELowPowerMode::GetCachedValueForProperty( const std::string& PropertyName )
{
  bool vLowPowerModeActive = false;
  std::vector<int>* vUnitsProperty = &( UnitsPropertyMap_[PropertyName] );
  std::vector<int>::const_iterator vUnitsIt = vUnitsProperty->begin();
  while ( vUnitsIt != vUnitsProperty->end() )
  {
    if ( *vUnitsIt == 0 )
    {
      vLowPowerModeActive |= Unit1Active_;
    }
    if ( *vUnitsIt == 1 )
    {
      vLowPowerModeActive |= Unit2Active_;
    }
    ++vUnitsIt;
  }
  return vLowPowerModeActive;
}

int CDualILELowPowerMode::OnValueChange( MM::PropertyBase * Prop, MM::ActionType Act )
{
  int vRet = DEVICE_OK;
  if ( PropertyPointers_.find( Prop->GetName() ) != PropertyPointers_.end() && PropertyPointers_[Prop->GetName()] == nullptr )
  {
    PropertyPointers_[Prop->GetName()] = Prop;
  }
  if ( Act == MM::BeforeGet )
  {
    Prop->Set( GetCachedValueForProperty( Prop->GetName() ) ? g_On : g_Off );
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
    std::string vPropertyName = Prop->GetName();
    std::map<std::string, std::vector<int>>::const_iterator vPropertyIt = UnitsPropertyMap_.find( vPropertyName );
    if ( vPropertyIt == UnitsPropertyMap_.end() )
    {
      MMILE_->LogMMMessage( "Low Power Mode: Property not found in property map [" + vPropertyName + "]" );
      return DEVICE_ERR;
    }
    std::string vValue;
    Prop->Get( vValue );
    bool vEnable = ( vValue == g_On );
    std::vector<int>* vUnitsProperty = &(UnitsPropertyMap_[vPropertyName]);
    std::vector<int>::const_iterator vUnitsIt = vUnitsProperty->begin();
    bool vPowerStateUpdated = false;
    while ( vUnitsIt != vUnitsProperty->end() )
    {
      IALC_REV_ILEPowerManagement* vPowerInterface = nullptr;
      bool* vUnitActive = nullptr;
      if ( *vUnitsIt == 0 )
      {
        vPowerInterface = Unit1PowerInterface_;
        vUnitActive = &Unit1Active_;
      }
      if ( *vUnitsIt == 1 )
      {
        vPowerInterface = Unit2PowerInterface_;
        vUnitActive = &Unit2Active_;
      }
      if ( vPowerInterface && vPowerInterface->SetLowPowerState( vEnable ) )
      {
        *vUnitActive = vEnable;
        vPowerStateUpdated = true;
      }
      else
      {
        std::string vErrorLogBase = std::string( vEnable ? "Enabling" : "Disabling" ) + " low power state for unit" + std::to_string( static_cast<long long>( *vUnitsIt + 1 ) ) + " FAILED.";
        if ( vPowerInterface )
        {
          MMILE_->LogMMMessage( vErrorLogBase + " Pointer to ILE power interface invalid." );
        }
        else
        {
          MMILE_->LogMMMessage( vErrorLogBase );
        }
        vRet = ERR_LOWPOWERMODE_SET;
      }
      ++vUnitsIt;
    }
    if ( vPowerStateUpdated )
    {
      MMILE_->CheckAndUpdateLasers();
    }
  }
  return vRet;
}

int CDualILELowPowerMode::UpdateILEInterface( IALC_REV_ILEPowerManagement* Unit1PowerInterface, IALC_REV_ILEPowerManagement* Unit2PowerInterface )
{
  Unit1PowerInterface_ = Unit1PowerInterface;
  Unit2PowerInterface_ = Unit2PowerInterface;

  if ( Unit1PowerInterface_ != nullptr || Unit2PowerInterface_ != nullptr )
  {
    if ( Unit1PowerInterface_ )
    {
      if ( !Unit1PowerInterface_->GetLowPowerState( &Unit1Active_ ) )
      {
        return ERR_LOWPOWERMODE_GET;
      }
    }
    if ( Unit2PowerInterface_ )
    {
      if ( !Unit2PowerInterface_->GetLowPowerState( &Unit2Active_ ) )
      {
        return ERR_LOWPOWERMODE_GET;
      }
    }
    MMILE_->LogMMMessage( "Resetting low power mode to device state [" + std::string( Unit1Active_ ? g_On : g_Off ) + ", " + ( Unit2Active_ ? g_On : g_Off ) + "]", true );
    std::map<std::string, std::vector<int>>::const_iterator vPropertyIt = UnitsPropertyMap_.begin();
    while ( vPropertyIt != UnitsPropertyMap_.end() )
    {
      if ( PropertyPointers_.find( vPropertyIt->first ) != PropertyPointers_.end() && PropertyPointers_[vPropertyIt->first] != nullptr )
      {
        PropertyPointers_[vPropertyIt->first]->Set( GetCachedValueForProperty( vPropertyIt->first ) ? g_On : g_Off );
      }
      ++vPropertyIt;
    }
  }
  return DEVICE_OK;
}