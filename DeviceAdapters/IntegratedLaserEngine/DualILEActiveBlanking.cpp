///////////////////////////////////////////////////////////////////////////////
// FILE:          DualILEActiveBlanking.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------

#include "DualILEActiveBlanking.h"
#include "DualILE.h"
#include "PortsConfiguration.h"
#include "ALC_REV.h"
#include "ALC_REV_ILE2.h"
#include <exception>
#include <algorithm>

const char* const g_PropertyName = "Ports Active Blanking";
const char* const g_On = "On";
const char* const g_Off = "Off";

CDualILEActiveBlanking::CDualILEActiveBlanking( IALC_REV_ILE4* DualActiveBlankingInterface, const CPortsConfiguration* PortsConfiguration, CDualILE* MMILE ) :
  DualActiveBlankingInterface_( DualActiveBlankingInterface ),
  PortsConfiguration_( PortsConfiguration ),
  MMILE_( MMILE ),
  Unit1EnabledPattern_( 0 ),
  Unit2EnabledPattern_( 0 ),
  Unit1ActiveBlankingPresent_( false ),
  Unit2ActiveBlankingPresent_( false ),
  PropertyPointer_( nullptr )
{
  // Check pointers validity
  if ( DualActiveBlankingInterface_ == nullptr )
  {
    throw std::logic_error( "CDualILEActiveBlanking: Pointer to Dual Active Blanking interface invalid" );
  }
  if ( PortsConfiguration_ == nullptr )
  {
    throw std::logic_error( "CDualILEActiveBlanking: Pointer to Ports configuration invalid" );
  }
  if ( MMILE_ == nullptr )
  {
    throw std::logic_error( "CDualILEActiveBlanking: Pointer to main class invalid" );
  }

  // Get presence of Active Blanking for each unit
  if ( !DualActiveBlankingInterface_->IsActiveBlankingManagementPresent( 0, &Unit1ActiveBlankingPresent_ ) )
  {
    throw std::runtime_error( "DualActiveBlankingInterface IsActiveBlankingManagementPresent failed for unit1" );
  }
  if ( !DualActiveBlankingInterface_->IsActiveBlankingManagementPresent( 1, &Unit2ActiveBlankingPresent_ ) )
  {
    throw std::runtime_error( "DualActiveBlankingInterface IsActiveBlankingManagementPresent failed for unit2" );
  }

  // Get state and number of lines for each unit
  if ( Unit1ActiveBlankingPresent_ )
  {
    if ( !DualActiveBlankingInterface_->GetActiveBlankingState( 0, &Unit1EnabledPattern_ ) )
    {
      throw std::runtime_error( "DualActiveBlankingInterface GetActiveBlankingState failed for unit1" );
    }
  }
  if ( Unit2ActiveBlankingPresent_ )
  {
    if ( !DualActiveBlankingInterface_->GetActiveBlankingState( 1, &Unit2EnabledPattern_ ) )
    {
      throw std::runtime_error( "DualActiveBlankingInterface GetActiveBlankingState failed for unit2" );
    }
  }

  // Create property
  std::vector<std::string> vAllowedValues;
  vAllowedValues.push_back( g_On );
  vAllowedValues.push_back( g_Off );
  CPropertyAction* vAct = new CPropertyAction( this, &CDualILEActiveBlanking::OnValueChange );
  MMILE_->CreateStringProperty( g_PropertyName, Unit1EnabledPattern_ | Unit2EnabledPattern_ ? g_On : g_Off, false, vAct );
  MMILE_->SetAllowedValues( g_PropertyName, vAllowedValues );
}

CDualILEActiveBlanking::~CDualILEActiveBlanking()
{

}

int CDualILEActiveBlanking::OnValueChange( MM::PropertyBase * Prop, MM::ActionType Act )
{
  if ( PropertyPointer_ == nullptr )
  {
    PropertyPointer_ = Prop;
  }
  if ( Act == MM::BeforeGet )
  {
    Prop->Set( Unit1EnabledPattern_ | Unit2EnabledPattern_ ? g_On : g_Off );
  }
  else if ( Act == MM::AfterSet )
  {
    int vInterlockStatus = MMILE_->GetClassIVAndKeyInterlockStatus();
    if ( vInterlockStatus != DEVICE_OK )
    {
      return vInterlockStatus;
    }
    if ( DualActiveBlankingInterface_ == nullptr )
    {
      return ERR_DEVICE_NOT_CONNECTED;
    }
    std::string vValue;
    Prop->Get( vValue );
    Unit2EnabledPattern_ = Unit1EnabledPattern_ = ( vValue == g_On ) ? 0xFF : 0;
    if ( Unit1ActiveBlankingPresent_ && !DualActiveBlankingInterface_->SetActiveBlankingState( 0, Unit1EnabledPattern_ ) )
    {
      return ERR_ACTIVEBLANKING_SET;
    }
    if ( Unit2ActiveBlankingPresent_ && !DualActiveBlankingInterface_->SetActiveBlankingState( 1, Unit2EnabledPattern_ ) )
    {
      return ERR_ACTIVEBLANKING_SET;
    }

  }
  return DEVICE_OK;
}

int CDualILEActiveBlanking::UpdateILEInterface( IALC_REV_ILE4* DualActiveBlankingInterface )
{
  DualActiveBlankingInterface_ = DualActiveBlankingInterface;
  if ( DualActiveBlankingInterface_ != nullptr )
  {
    // Get state and number of lines for each unit
    if ( Unit1ActiveBlankingPresent_ )
    {
      if ( !DualActiveBlankingInterface_->GetActiveBlankingState( 0, &Unit1EnabledPattern_ ) )
      {
        return ERR_ACTIVEBLANKING_GETSTATE;
      }
    }
    if ( Unit2ActiveBlankingPresent_ )
    {
      if ( !DualActiveBlankingInterface_->GetActiveBlankingState( 1, &Unit2EnabledPattern_ ) )
      {
        return ERR_ACTIVEBLANKING_GETSTATE;
      }
    }

    MMILE_->LogMMMessage( "Resetting active blanking to device state [" + std::to_string( static_cast<long long>( Unit1EnabledPattern_ ) ) + ", " + std::to_string( static_cast<long long>( Unit2EnabledPattern_ ) ) + "]", true );
    if ( PropertyPointer_ != nullptr )
    {
      PropertyPointer_->Set( Unit1EnabledPattern_ | Unit2EnabledPattern_ ? g_On : g_Off );
    }
  }
  return DEVICE_OK;
}
