///////////////////////////////////////////////////////////////////////////////
// FILE:          DualILEActiveBlanking.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------

#include "DualILEActiveBlanking.h"
#include "IntegratedLaserEngine.h"
#include "PortsConfiguration.h"
#include "ALC_REV.h"
#include "ALC_REV_ILE2.h"
#include <exception>
#include <algorithm>

const char* const g_PropertyBaseName = "Active Blanking";
const char* const g_On = "On";
const char* const g_Off = "Off";

CDualILEActiveBlanking::CDualILEActiveBlanking( IALC_REV_ILE4* DualActiveBlankingInterface, const CPortsConfiguration* PortsConfiguration, CIntegratedLaserEngine* MMILE ) :
  DualActiveBlankingInterface_( DualActiveBlankingInterface ),
  PortsConfiguration_( PortsConfiguration ),
  MMILE_( MMILE ),
  Unit1EnabledPattern_( 0 ),
  Unit2EnabledPattern_( 0 ),
  Unit1ActiveBlankingPresent_( false ),
  Unit2ActiveBlankingPresent_( false ),
  Unit1NbLines_( 0 ),
  Unit2NbLines_( 0 )
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
    if ( !DualActiveBlankingInterface_->GetNumberOfLines( 0, &Unit1NbLines_ ) )
    {
      throw std::runtime_error( "DualActiveBlankingInterface GetNumberOfLines failed for unit1" );
    }
  }
  if ( Unit2ActiveBlankingPresent_ )
  {
    if ( !DualActiveBlankingInterface_->GetActiveBlankingState( 1, &Unit2EnabledPattern_ ) )
    {
      throw std::runtime_error( "DualActiveBlankingInterface GetActiveBlankingState failed for unit2" );
    }
    if ( !DualActiveBlankingInterface_->GetNumberOfLines( 1, &Unit2NbLines_ ) )
    {
      throw std::runtime_error( "DualActiveBlankingInterface GetNumberOfLines failed for unit2" );
    }
  }

  PortNames_ = PortsConfiguration_->GetPortList();

  if ( !PortNames_.empty() )
  {
/*
    // Synchronise the units' active blanking states for each combined port
    if ( Unit1ActiveBlankingPresent_ && Unit2ActiveBlankingPresent_ )
    {
      std::vector<std::string>::const_iterator vPort = PortNames_.begin();
      std::vector<int> vUnitsPorts;
      while ( vPort != PortNames_.end() )
      {
        PortsConfiguration_->GetUnitPortsForMergedPort( *vPort, &vUnitsPorts );
        bool vUnit1 = IsLineEnabledForSinglePort( 0, vUnitsPorts[0] );
        bool vUnit2 = IsLineEnabledForSinglePort( 1, vUnitsPorts[1] );
        MMILE_->LogMMMessage( "Active Blanking - Port " + *vPort + ": Unit1/" + std::to_string( vUnitsPorts[0] ) + " = " + ( vUnit1 ? g_On : g_Off ), true);
        MMILE_->LogMMMessage( "Active Blanking - Port " + *vPort + ": Unit2/" + std::to_string( vUnitsPorts[1] ) + " = " + ( vUnit2 ? g_On : g_Off ), true );
        if ( vUnit1 != vUnit2 )
        {
          // Ports have different states so turn them off (safest choice)
          if ( vUnit1 )
          {
            SetLineStateForSinglePort( 0, vUnitsPorts[0], false );
            if ( !DualActiveBlankingInterface_->SetActiveBlankingState( 0, Unit1EnabledPattern_ ) )
            {
              throw std::runtime_error( "DualActiveBlankingInterface SetActiveBlankingState failed for unit1" );
            }
          }
          else
          {
            SetLineStateForSinglePort( 1, vUnitsPorts[1], false );
            if ( !DualActiveBlankingInterface_->SetActiveBlankingState( 1, Unit2EnabledPattern_ ) )
            {
              throw std::runtime_error( "DualActiveBlankingInterface SetActiveBlankingState failed for unit2" );
            }
          }
        }
        vPort++;
      }
    }
*/
    // Create properties
    std::vector<std::string> vAllowedValues;
    vAllowedValues.push_back( g_On );
    vAllowedValues.push_back( g_Off );
    std::string vPropertyName;
    for (int vPortIndex = 0; vPortIndex < PortNames_.size(); ++vPortIndex )
    {
      vPropertyName = BuildProperty( PortNames_[vPortIndex] );
      bool vEnabled = IsLineEnabledForDualPort( PortNames_[vPortIndex] );
      CPropertyActionEx* vAct = new CPropertyActionEx( this, &CDualILEActiveBlanking::OnValueChange, vPortIndex );
      MMILE_->CreateStringProperty( vPropertyName.c_str(), vEnabled ? g_On : g_Off, false, vAct );
      MMILE_->SetAllowedValues( vPropertyName.c_str(), vAllowedValues );
      PropertyPointers_[vPropertyName] = nullptr;
    }
  }
}

CDualILEActiveBlanking::~CDualILEActiveBlanking()
{

}

std::string CDualILEActiveBlanking::BuildProperty( const std::string& PortName ) const
{
  return "Port " + PortName + "-" + g_PropertyBaseName;
}

bool CDualILEActiveBlanking::IsLineEnabledForSinglePort( int Unit, int Line ) const
{
  bool vEnabled = false;
  int vNbLines = 0;
  int vEnabledPattern = 0;
  if ( Unit == 0 )
  {
    vNbLines = Unit1NbLines_;
    vEnabledPattern = Unit1EnabledPattern_;
  }
  else
  {
    vNbLines = Unit2NbLines_;
    vEnabledPattern = Unit2EnabledPattern_;
  }

  // Line is 1-based
  if ( Line > 0 && Line <= vNbLines )
  {
    int vMask = 1;
    for ( int vIt = 1; vIt < Line; vIt++ )
    {
      vMask <<= 1;
    }
    vEnabled = ( vEnabledPattern & vMask ) != 0;
    MMILE_->LogMMMessage( "Active Blanking for line " + std::to_string( static_cast<long long>( Line ) ) + " for unit " + std::to_string( static_cast<long long>( Unit + 1 ) ) + " = " + ( vEnabled ? g_On : g_Off ) + " - pattern = " + std::to_string( static_cast<long long>( vEnabledPattern ) )  + " mask = " + std::to_string( static_cast<long long>( vMask ) ), true );
  }
  return vEnabled;
}

bool CDualILEActiveBlanking::IsLineEnabledForDualPort( const std::string& PortName ) const
{
  std::vector<std::string>::const_iterator vPort = std::find( PortNames_.begin(), PortNames_.end(), PortName );

  if ( vPort != PortNames_.end() )
  {
    std::vector<int> vUnitsPorts;
    PortsConfiguration_->GetUnitPortsForMergedPort( *vPort, &vUnitsPorts );
    return IsLineEnabledForSinglePort( 0, vUnitsPorts[0] ) || IsLineEnabledForSinglePort( 1, vUnitsPorts[1] );
  }
  return false;
}

void CDualILEActiveBlanking::SetLineStateForSinglePort( int Unit, int Line, bool Enable )
{
  int vNbLines = 0;
  int* vEnabledPattern = nullptr;
  if ( Unit == 0 )
  {
    vNbLines = Unit1NbLines_;
    vEnabledPattern = &Unit1EnabledPattern_;
  }
  else
  {
    vNbLines = Unit2NbLines_;
    vEnabledPattern = &Unit2EnabledPattern_;
  }
  
  // Line is 1-based
  if ( Line > 0 && Line <= vNbLines )
  {
    std::string vLogStart = "Changing line " + std::to_string( static_cast<long long>( Line ) ) + " for unit " + std::to_string( static_cast<long long>( Unit + 1 ) );
    MMILE_->LogMMMessage( vLogStart + " - Old pattern = " + std::to_string( static_cast<long long>( *vEnabledPattern ) ), true );
    int vMask = 1;
    for ( int vIt = 1; vIt < Line; vIt++ )
    {
      vMask <<= 1;
    }
    if ( Enable )
    {
      *vEnabledPattern |= vMask;
    }
    else
    {
      vMask ^= 0xFF;
      *vEnabledPattern &= vMask;
    }
    MMILE_->LogMMMessage( vLogStart + " - New pattern = " + std::to_string( static_cast<long long>( *vEnabledPattern ) ) + " mask = " + std::to_string( static_cast<long long>( vMask ) ), true );
  }
}

void CDualILEActiveBlanking::SetLineStateForDualPort( const std::string& PortName, bool Enable )
{
  std::vector<std::string>::const_iterator vPort = std::find( PortNames_.begin(), PortNames_.end(), PortName );

  if ( vPort != PortNames_.end() )
  {
    std::vector<int> vUnitsPorts;
    PortsConfiguration_->GetUnitPortsForMergedPort( *vPort, &vUnitsPorts );
    SetLineStateForSinglePort( 0, vUnitsPorts[0], Enable );
    SetLineStateForSinglePort( 1, vUnitsPorts[1], Enable );
  }
}

int CDualILEActiveBlanking::OnValueChange( MM::PropertyBase * Prop, MM::ActionType Act, long PortIndex )
{
  if ( PropertyPointers_.find( Prop->GetName() ) != PropertyPointers_.end() && PropertyPointers_[Prop->GetName()] == nullptr )
  {
    PropertyPointers_[Prop->GetName()] = Prop;
  }
  if ( Act == MM::BeforeGet )
  {
    // Commenting this in case 2 virtual ports share the same physical port. In this case changing the active blanking state
    // of one of the virtual ports will sometimes change the state of other virtual ports using one of its physical ports.
    //if ( PortIndex < PortNames_.size() )
    //{
    //  std::string vPortName = PortNames_[PortIndex];    
    //  Prop->Set( IsLineEnabledForDualPort( vPortName ) ? g_On : g_Off );
    //}
  }
  else if ( Act == MM::AfterSet )
  {
    if ( DualActiveBlankingInterface_ == nullptr )
    {
      return ERR_DEVICE_NOT_CONNECTED;
    }

    if ( PortIndex < PortNames_.size() )
    {
      std::string vPortName = PortNames_[PortIndex];
      std::string vValue;
      Prop->Get( vValue );
      bool vRequestEnabled = ( vValue == g_On );
      SetLineStateForDualPort( vPortName, vRequestEnabled );
      // TEST
      std::vector<int> vUnitsPorts;
      PortsConfiguration_->GetUnitPortsForMergedPort( vPortName, &vUnitsPorts );
      bool vUnit1 = IsLineEnabledForSinglePort( 0, vUnitsPorts[0] );
      bool vUnit2 = IsLineEnabledForSinglePort( 1, vUnitsPorts[1] );
      MMILE_->LogMMMessage( "Changing Active Blanking - Port " + vPortName + ": Unit1/" + std::to_string( static_cast<long long>( vUnitsPorts[0] ) ) + " = " + ( vUnit1 ? g_On : g_Off ) );
      MMILE_->LogMMMessage( "Changing Active Blanking - Port " + vPortName + ": Unit2/" + std::to_string( static_cast<long long>( vUnitsPorts[1] ) ) + " = " + ( vUnit2 ? g_On : g_Off ) );
      // TEST
      if ( !DualActiveBlankingInterface_->SetActiveBlankingState( 0, Unit1EnabledPattern_ ) )
      {
        LogSetActiveBlankingError( 0, vPortName, vRequestEnabled );
        return ERR_ACTIVEBLANKING_SET;
      }
      if ( !DualActiveBlankingInterface_->SetActiveBlankingState( 1, Unit2EnabledPattern_ ) ) 
      {
        LogSetActiveBlankingError( 1, vPortName, vRequestEnabled );
        return ERR_ACTIVEBLANKING_SET;
      }
    }
  }
  return DEVICE_OK;
}

void CDualILEActiveBlanking::LogSetActiveBlankingError( int Unit, const std::string& PortName, bool Enabling )
{
  std::vector<int> vUnitsPorts;
  PortsConfiguration_->GetUnitPortsForMergedPort( PortName, &vUnitsPorts );
  std::string vEnableString = "Enabling";
  if ( !Enabling )
  {
    vEnableString = "Disabling";
  }
  MMILE_->LogMMMessage( vEnableString + " Active Blanking on unit" + std::to_string( static_cast<long long>( Unit + 1 ) ) + " [line " + std::to_string( static_cast<long long>( vUnitsPorts[Unit] ) ) + "] for port " + PortName + " FAILED" );
}

void CDualILEActiveBlanking::UpdateILEInterface( IALC_REV_ILE4* DualActiveBlankingInterface )
{
  DualActiveBlankingInterface_ = DualActiveBlankingInterface;
  if ( DualActiveBlankingInterface_ != nullptr )
  {
    // Get state and number of lines for each unit
    if ( Unit1ActiveBlankingPresent_ )
    {
      DualActiveBlankingInterface_->GetActiveBlankingState( 0, &Unit1EnabledPattern_ );
    }
    if ( Unit2ActiveBlankingPresent_ )
    {
      DualActiveBlankingInterface_->GetActiveBlankingState( 1, &Unit2EnabledPattern_ );
    }

    MMILE_->LogMMMessage( "Resetting active blanking to device state [" + std::to_string( static_cast<long long>( Unit1EnabledPattern_ ) ) + ", " + std::to_string( static_cast<long long>( Unit2EnabledPattern_ ) ) + "]", true );
    std::string vPropertyName;
    std::vector<std::string>::const_iterator vPortIt = PortNames_.begin();
    while ( vPortIt != PortNames_.end() )
    {
      vPropertyName = BuildProperty( *vPortIt );
      if ( PropertyPointers_.find(vPropertyName) != PropertyPointers_.end() && PropertyPointers_[vPropertyName] != nullptr )
      {
        PropertyPointers_[vPropertyName]->Set( IsLineEnabledForDualPort( *vPortIt ) ? g_On : g_Off );
      }
      ++vPortIt;
    }
  }
}
