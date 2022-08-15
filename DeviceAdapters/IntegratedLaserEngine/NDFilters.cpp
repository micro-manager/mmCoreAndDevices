///////////////////////////////////////////////////////////////////////////////
// FILE:          NDFilters.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------

#include "NDFilters.h"
#include "IntegratedLaserEngine.h"
#include "ALC_REV.h"
#include <exception>
#include <string>

const char* const g_PropertyName = "ND Filters";
const char* const g_1x = "100%";

CNDFilters::CNDFilters( IALC_REV_ILEPowerManagement2* PowerInterface, CIntegratedLaserEngine* MMILE ) :
  PowerInterface_( PowerInterface ),
  MMILE_( MMILE )
{
  if ( PowerInterface_ == nullptr )
  {
    throw std::logic_error( "CNDFilters: Pointer to ILE Power interface invalid" );
  }
  if ( MMILE_ == nullptr )
  {
    throw std::logic_error( "CNDFilters: Pointer to main class invalid" );
  }

  bool vEnabled;
  if ( !PowerInterface_->IsActivationModeEnabled( &vEnabled ) )
  {
    throw std::runtime_error( "ILE IsActivationModeEnabled failed" );
  }
  int vNumLevels;
  if ( !PowerInterface_->GetNumberOfLowPowerLevels( &vNumLevels ) )
  {
    throw std::runtime_error( "ILE GetNumberOfLowPowerLevels failed" );
  }

  FilterPositions_.push_back( g_1x );
  for ( int vLevel = 1; vLevel < vNumLevels + 1; ++vLevel )
  {
    double vPercentage;
    if ( PowerInterface_->GetLowPowerPercentage( vLevel, &vPercentage ) )
    {
      std::string str = std::to_string( vPercentage );
      str.erase( str.find_last_not_of( '0' ) + 1, std::string::npos );
      str.erase( str.find_last_not_of( '.' ) + 1, std::string::npos );
      FilterPositions_.push_back( str + "%" );
    }
  }

  CurrentFilterPosition_ = 0;
  if ( vEnabled )
  {
    if ( !PowerInterface_->GetLowPowerLevel( &CurrentFilterPosition_ ) )
    {
      throw std::runtime_error( "ILE GetLowPowerLevel failed" );
    }
  }

  // Create property
  CPropertyAction* vAct = new CPropertyAction( this, &CNDFilters::OnValueChange );
  MMILE_->CreateStringProperty( g_PropertyName, FilterPositions_[CurrentFilterPosition_].c_str(), false, vAct );
  MMILE_->SetAllowedValues( g_PropertyName, FilterPositions_ );
}

CNDFilters::~CNDFilters()
{
}

int CNDFilters::SetDevice( int NewPosition, bool Initialisation )
{
  if ( FilterPositions_[NewPosition] == g_1x )
  {
    if ( !PowerInterface_->EnableActivationMode( false ) )
    {
      MMILE_->LogMMMessage( "Disabling activation mode FAILED" );
      return ERR_NDFILTERS_SET;
    }
  }
  else
  {
    if ( Initialisation || FilterPositions_[CurrentFilterPosition_] == g_1x )
    {
      if ( !PowerInterface_->EnableActivationMode( true ) )
      {
        MMILE_->LogMMMessage( "Enabling activation mode FAILED" );
        return ERR_NDFILTERS_SET;
      }
    }
    if ( !PowerInterface_->SetLowPowerLevel( NewPosition ) )
    {
      std::string vErrorMessage = "Changing ND Filters to position [" + std::to_string( NewPosition ) + "] FAILED";
      MMILE_->LogMMMessage( vErrorMessage.c_str() );
      return ERR_NDFILTERS_SET;
    }
  }

  MMILE_->CheckAndUpdateLasers();

  return DEVICE_OK;
}

int CNDFilters::OnValueChange( MM::PropertyBase * Prop, MM::ActionType Act )
{
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
    int vRet = SetDevice( vPosition, false );
    if ( vRet != DEVICE_OK )
    {
      return vRet;
    }
    CurrentFilterPosition_ = vPosition;
  }

  return DEVICE_OK;
}

int CNDFilters::UpdateILEInterface( IALC_REV_ILEPowerManagement2* PowerInterface )
{
  if ( PowerInterface != PowerInterface_ )
  {
    PowerInterface_ = PowerInterface;
    if ( PowerInterface_ != nullptr )
    {
      int vRet = SetDevice( CurrentFilterPosition_, true );
      if ( vRet != DEVICE_OK )
      {
        return vRet;
      }
    }
    MMILE_->LogMMMessage( "Resetting ND Filters device's state to [" + FilterPositions_[CurrentFilterPosition_] + "]", true );
  }
  
  return DEVICE_OK;
}