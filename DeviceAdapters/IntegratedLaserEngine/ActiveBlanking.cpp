///////////////////////////////////////////////////////////////////////////////
// FILE:          ActiveBlanking.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------

#include "ActiveBlanking.h"
#include "IntegratedLaserEngine.h"
#include "Ports.h"
#include "ALC_REV.h"
#include <exception>

const char* const g_PropertyBaseName = "Active Blanking";
const char* const g_On = "On";
const char* const g_Off = "Off";

CActiveBlanking::CActiveBlanking( IALC_REV_ILEActiveBlankingManagement* ActiveBlankingInterface, CIntegratedLaserEngine* MMILE ) :
  ActiveBlankingInterface_( ActiveBlankingInterface ),
  MMILE_( MMILE )
{
  if ( ActiveBlankingInterface_ == nullptr )
  {
    throw std::logic_error( "CActiveBlanking: Pointer to Active Blanking interface invalid" );
  }
  if ( MMILE_ == nullptr )
  {
    throw std::logic_error( "CActiveBlanking: Pointer tomain class invalid" );
  }

  int vNbLines;
  if ( !ActiveBlankingInterface_->GetNumberOfLines( &vNbLines ) )
  {
    throw std::runtime_error( "ActiveBlankingInterface GetNumberOfLines failed" );
  }

  if ( vNbLines > 0 )
  {
    if ( !ActiveBlankingInterface_->GetActiveBlankingState( &EnabledPattern_ ) )
    {
      throw std::runtime_error( "ActiveBlankingInterface GetActiveBlankingState failed" );
    }

    // Create properties
    std::vector<std::string> vAllowedValues;
    vAllowedValues.push_back( g_On );
    vAllowedValues.push_back( g_Off );
    char vLineName[2];
    vLineName[1] = 0;
    std::string vPropertyName;
    for ( int vLineIndex = 0; vLineIndex < vNbLines; vLineIndex++ )
    {
      vLineName[0] = CPorts::PortIndexToName( vLineIndex + 1 ); // port indices are 1-based but line indices are 0-based
      vPropertyName = "Port " + std::string( vLineName ) + "-" + g_PropertyBaseName;
      PropertyLineIndexMap_[vPropertyName] = vLineIndex;
      PropertyPointers_[vPropertyName] = nullptr;
      bool vEnabled = IsLineEnabled( vLineIndex );
      CPropertyAction* vAct = new CPropertyAction( this, &CActiveBlanking::OnValueChange );
      MMILE_->CreateStringProperty( vPropertyName.c_str(), vEnabled ? g_On : g_Off, false, vAct );
      MMILE_->SetAllowedValues( vPropertyName.c_str(), vAllowedValues );
    }
  }
}

CActiveBlanking::~CActiveBlanking()
{

}

bool CActiveBlanking::IsLineEnabled( int Line ) const
{
  if ( Line < PropertyLineIndexMap_.size() )
  {
    int vMask = 1;
    for ( int vIt = 0; vIt < Line; vIt++ )
    {
      vMask <<= 1;
    }
    return ( EnabledPattern_ & vMask ) != 0;
  }
  return false;
}

void CActiveBlanking::ChangeLineState( int Line )
{
  if ( Line < PropertyLineIndexMap_.size() )
  {
    int vMask = 1;
    for ( int vIt = 0; vIt < Line; vIt++ )
    {
      vMask <<= 1;
    }
    EnabledPattern_ ^= vMask;
  }
}

int CActiveBlanking::OnValueChange( MM::PropertyBase * Prop, MM::ActionType Act )
{
  if ( PropertyPointers_.find( Prop->GetName() ) != PropertyPointers_.end() && PropertyPointers_[Prop->GetName()] == nullptr )
  {
    PropertyPointers_[Prop->GetName()] = Prop;
  }
  if ( Act == MM::BeforeGet )
  {
    if ( PropertyLineIndexMap_.find( Prop->GetName() ) != PropertyLineIndexMap_.end() )
    {
      int vLineIndex = PropertyLineIndexMap_[Prop->GetName()];
      bool vEnabled = IsLineEnabled( vLineIndex );
      Prop->Set( vEnabled ? g_On : g_Off );
    }
  }
  else if ( Act == MM::AfterSet )
  {
    int vInterlockStatus = MMILE_->GetClassIVAndKeyInterlockStatus();
    if ( vInterlockStatus != DEVICE_OK )
    {
      return vInterlockStatus;
    }
    if ( ActiveBlankingInterface_ == nullptr )
    {
      return ERR_DEVICE_NOT_CONNECTED;
    }

    if ( PropertyLineIndexMap_.find( Prop->GetName() ) != PropertyLineIndexMap_.end() )
    {
      int vLineIndex = PropertyLineIndexMap_[Prop->GetName()];
      std::string vValue;
      Prop->Get( vValue );
      bool vRequestEnabled = ( vValue == g_On );
      bool vEnabled = IsLineEnabled( vLineIndex );
      if ( vEnabled != vRequestEnabled )
      {
        ChangeLineState( vLineIndex );
        if ( !ActiveBlankingInterface_->SetActiveBlankingState( EnabledPattern_ ) )
        {
          if ( vRequestEnabled )
          {
            MMILE_->LogMMMessage( "Enabling Active Blanking for line " + std::to_string( static_cast<long long>( vLineIndex ) ) + " FAILED" );
          }
          else
          {
            MMILE_->LogMMMessage( "Disabling Active Blanking for line " + std::to_string( static_cast<long long>( vLineIndex ) ) + " FAILED" );
          }
          return ERR_ACTIVEBLANKING_SET;
        }
      }
    }
  }
  return DEVICE_OK;
}

int CActiveBlanking::UpdateILEInterface( IALC_REV_ILEActiveBlankingManagement* ActiveBlankingInterface )
{
  ActiveBlankingInterface_ = ActiveBlankingInterface;
  if ( ActiveBlankingInterface_ != nullptr )
  {
    int vNbLines;
    if ( ActiveBlankingInterface_->GetNumberOfLines( &vNbLines ) )
    {
      if ( ActiveBlankingInterface_->GetActiveBlankingState( &EnabledPattern_ ) )
      {
        MMILE_->LogMMMessage( "Resetting active blanking to device state [" + std::to_string( static_cast<long long>( EnabledPattern_ ) ) + "]", true );
        int vLineIndex = 0;
        std::map<std::string, int>::const_iterator vPropertyIt = PropertyLineIndexMap_.begin();
        while ( vPropertyIt != PropertyLineIndexMap_.end() )
        {
          if ( vPropertyIt->second < vNbLines && PropertyPointers_.find( vPropertyIt->first.c_str() ) != PropertyPointers_.end() && PropertyPointers_[vPropertyIt->first] != nullptr )
          {
            PropertyPointers_[vPropertyIt->first]->Set( IsLineEnabled( vLineIndex ) ? g_On : g_Off );
          }
          ++vLineIndex;
          ++vPropertyIt;
        }
      }
      else
      {
        return ERR_ACTIVEBLANKING_GETSTATE;
      }
    }
    else
    {
      return ERR_ACTIVEBLANKING_GETNBLINES;
    }
  }
  return DEVICE_OK;
}
