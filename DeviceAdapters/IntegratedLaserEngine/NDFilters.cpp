///////////////////////////////////////////////////////////////////////////////
// FILE:          NDFilters.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------

#include "NDFilters.h"
#include "IntegratedLaserEngine.h"
#include "Ports.h"
#include "ALC_REV.h"
#include <exception>
#include <string>

const char* const g_PropertyBaseName = "ND Filters";
const char* const g_1x = "100%";

CNDFilters::CNDFilters( IALC_REV_ILEPowerManagement2* PowerInterface, CIntegratedLaserEngine* MMILE ) :
  PowerInterface_( PowerInterface ),
  MMILE_( MMILE ),
  CurrentFilterPosition_( 0 ),
  PropertyPointer_( nullptr )
{
  if ( PowerInterface_ == nullptr )
  {
    throw std::logic_error( "CNDFilters: Pointer to ILE Power interface invalid" );
  }
  if ( MMILE_ == nullptr )
  {
    throw std::logic_error( "CNDFilters: Pointer to main class invalid" );
  }

#if 0
  // Removing the check for Low Power enabled to ensure we retrieve the current
  // state of the device on init regardless of the selected port
  bool vEnabled;
  if ( !PowerInterface_->IsLowPowerEnabled( &vEnabled ) )
  {
    throw std::runtime_error( "ILE IsLowPowerEnabled failed" );
  }

  bool vLowPowerActive = false;
  if ( vEnabled )
  {
    if ( !PowerInterface_->GetLowPowerState( &vLowPowerActive ) )
    {
      throw std::runtime_error( "ILE GetLowPowerState failed" );
    }
  }
#else
  bool vLowPowerActive;
  if ( !PowerInterface_->GetLowPowerState( &vLowPowerActive ) )
  {
    throw std::runtime_error( "ILE GetLowPowerState failed" );
  }
#endif

  int vNumLevels;
  if ( !PowerInterface_->GetNumberOfLowPowerLevels( &vNumLevels ) )
  {
    throw std::runtime_error( "ILE GetNumberOfLowPowerLevels failed" );
  }

  int vLowPowerPortIndex;
  if ( !PowerInterface_->GetLowPowerPort( &vLowPowerPortIndex ) )
  {
    throw std::runtime_error( "ILE GetLowPowerPort failed" );
  }

  if ( vLowPowerPortIndex < 1 )
  {
    throw std::runtime_error( "Low Power port index invalid [" + std::to_string( static_cast< long long >( vLowPowerPortIndex ) ) + "]" );
  }

  FilterPositions_.push_back( g_1x );
  for ( int vLevel = 1; vLevel < vNumLevels + 1; ++vLevel )
  {
    double vPercentage;
    if ( PowerInterface_->GetLowPowerPercentage( vLevel, &vPercentage ) )
    {
      std::string str = std::to_string( vPercentage );
      // Transform double to int (why not cast from double to int then???)
      str.erase( str.find_last_not_of( '0' ) + 1, std::string::npos );
      str.erase( str.find_last_not_of( '.' ) + 1, std::string::npos );
      FilterPositions_.push_back( str + "%" );
    }
  }

  if ( vLowPowerActive )
  {
    if ( !PowerInterface_->GetLowPowerLevel( &CurrentFilterPosition_ ) )
    {
      throw std::runtime_error( "ILE GetLowPowerLevel failed" );
    }
  }

  // Create property
  char vPortName[2];
  vPortName[1] = 0;
  vPortName[0] = CPorts::PortIndexToName( vLowPowerPortIndex );
  std::string vPropertyName = std::string( "Port " ) + vPortName + "-" + g_PropertyBaseName;
  CPropertyAction* vAct = new CPropertyAction( this, &CNDFilters::OnValueChange );
  MMILE_->CreateStringProperty( vPropertyName.c_str(), FilterPositions_[CurrentFilterPosition_].c_str(), false, vAct);
  MMILE_->SetAllowedValues( vPropertyName.c_str(), FilterPositions_ );
}

CNDFilters::~CNDFilters()
{
}

int CNDFilters::SetDevice( int NewPosition )
{
  if ( PowerInterface_ == nullptr )
  {
    return ERR_DEVICE_NOT_CONNECTED;
  }

  bool vEnabled;
  if ( !PowerInterface_->IsLowPowerEnabled( &vEnabled ) )
  {
    MMILE_->LogMMMessage( "ILE IsLowPowerEnabled FAILED" );
    return ERR_NDFILTERS_GET;
  }

  if ( !vEnabled )
  {
    // Wrong port, ignore command
    MMILE_->LogMMMessage( "ILE Low Power not enabled", true );
    return ERR_NDFILTERS_NOT_ENABLED;
  }

  bool vCurrentDeviceState;
  if ( !PowerInterface_->GetLowPowerState( &vCurrentDeviceState ) )
  {
    MMILE_->LogMMMessage( "ILE GetLowPowerState FAILED" );
    return ERR_NDFILTERS_GET;
  }

  MMILE_->LogMMMessage( "Current Low Power state: [" + std::string( vCurrentDeviceState ? "ON" : "OFF" ) + "]", true);

  bool vDevicePositionChanged = false;
  if ( FilterPositions_[NewPosition] == g_1x )
  {
    // Turn Low Power off if needed
    if ( vCurrentDeviceState )
    {
      MMILE_->LogMMMessage( "Turn Low Power OFF", true );
      if ( !PowerInterface_->SetLowPowerState( false ) )
      {
        MMILE_->LogMMMessage( "Turning Low Power OFF FAILED" );
        return ERR_NDFILTERS_SET;
      }
      vDevicePositionChanged = true;
    }
  }
  else
  {
    // Turn Low Power on if needed
    if ( !vCurrentDeviceState )
    {
      MMILE_->LogMMMessage( "Turn Low Power ON", true );
      if ( !PowerInterface_->SetLowPowerState( true ) )
      {
        MMILE_->LogMMMessage( "Turning Low Power ON FAILED" );
        return ERR_NDFILTERS_SET;
      }
      vDevicePositionChanged = true;
    }

    int vCurrentPosition;
    if ( !PowerInterface_->GetLowPowerLevel( &vCurrentPosition ) )
    {
      MMILE_->LogMMMessage( "ILE GetLowPowerLevel FAILED" );
      return ERR_NDFILTERS_GET;
    }

    MMILE_->LogMMMessage( "Current Low Power level: [" + std::to_string( vCurrentPosition ) + "]", true );

    // Change Low Power level if needed
    if ( vCurrentPosition != NewPosition )
    {
      MMILE_->LogMMMessage( "Set Low Power level to [" + std::to_string( NewPosition ) + "]", true );
      if ( !PowerInterface_->SetLowPowerLevel( NewPosition ) )
      {
        std::string vErrorMessage = "Changing ND Filters to position [" + std::to_string( NewPosition ) + "] FAILED";
        MMILE_->LogMMMessage( vErrorMessage.c_str() );

        if ( !vCurrentDeviceState )
        {
          MMILE_->LogMMMessage( "Reverting Low Power state to OFF", true );
          if ( !PowerInterface_->SetLowPowerState( false ) )
          {
            MMILE_->LogMMMessage( "Turning Low Power OFF FAILED" );
            return ERR_NDFILTERS_SET;
          }
        }
        return ERR_NDFILTERS_SET;
      }
     vDevicePositionChanged = true;
    }
  }

  if ( vDevicePositionChanged )
  {
    MMILE_->CheckAndUpdateLasers();
  }

  return DEVICE_OK;
}

int CNDFilters::OnValueChange( MM::PropertyBase * Prop, MM::ActionType Act )
{
  if ( PropertyPointer_ == nullptr )
  {
    PropertyPointer_ = Prop;
  }

  if ( Act == MM::BeforeGet )
  {
    Prop->Set( FilterPositions_[CurrentFilterPosition_].c_str());
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
    const auto vCurrentSelectionIt = std::find( FilterPositions_.begin(), FilterPositions_.end(), vValue );
    if ( vCurrentSelectionIt == FilterPositions_.end() )
    {
      return ERR_DEVICE_INDEXINVALID;
    }

    int vPosition = static_cast< int >( std::distance( FilterPositions_.begin(), vCurrentSelectionIt ) );
    int vRet = SetDevice( vPosition );

    if ( vRet == ERR_NDFILTERS_NOT_ENABLED )
    {
      // Ignore when ND filter is not enabled (wrong current port) to allow MD acquisition to continue
      vRet = DEVICE_OK;
    }

    if ( vRet != DEVICE_OK )
    {
      // If we can't change the device's state, revert the selection to the previous position
      MMILE_->LogMMMessage( "Couldn't change Low Power state, reverting the UI position to [" + FilterPositions_[CurrentFilterPosition_] + "]");
      Prop->Set( FilterPositions_[CurrentFilterPosition_].c_str() );
      return vRet;
    }
    CurrentFilterPosition_ = vPosition;
  }

  return DEVICE_OK;
}

int CNDFilters::UpdateILEInterface( IALC_REV_ILEPowerManagement2* PowerInterface )
{
  int vRet = DEVICE_OK;

  if ( PowerInterface != PowerInterface_ )
  {
    PowerInterface_ = PowerInterface;

    if ( PowerInterface_ != nullptr )
    {
      MMILE_->LogMMMessage( "Resetting ND Filters position to device state", true );

      bool vLowPowerActive = false;
      if ( !PowerInterface_->GetLowPowerState( &vLowPowerActive ) )
      {
        MMILE_->LogMMMessage( "ILE GetLowPowerState FAILED" );
        vRet = ERR_NDFILTERS_GET;
      }

      int vCurrentFilterPosition = 0;
      if ( vRet == DEVICE_OK && vLowPowerActive )
      {
        if ( !PowerInterface_->GetLowPowerLevel( &vCurrentFilterPosition ) )
        {
          MMILE_->LogMMMessage( "ILE GetLowPowerLevel FAILED" );
          vRet = ERR_NDFILTERS_GET;
        }
      }

      if ( vRet == DEVICE_OK )
      {
        if ( vCurrentFilterPosition >= 0 && vCurrentFilterPosition < FilterPositions_.size() )
        {
          CurrentFilterPosition_ = vCurrentFilterPosition;
          MMILE_->LogMMMessage( "ND Filters device state [" + std::to_string( CurrentFilterPosition_ ) + "]", true );
          if ( PropertyPointer_ != nullptr )
          {
            PropertyPointer_->Set( FilterPositions_[CurrentFilterPosition_].c_str() );
          }
        }
        else
        {
          MMILE_->LogMMMessage( "Invalid ND Filters device state [" + std::to_string( vCurrentFilterPosition ) + "]", true );
          vRet = DEVICE_ERR;
        }
      }
    }
  }

  return vRet;
}

void CNDFilters::CheckAndUpdate()
{
  // Update the device state on port change
  // If the previous port wasn't supporting ND filter, any requested ND filter position
  // would have been ignored
  SetDevice( CurrentFilterPosition_ );
}