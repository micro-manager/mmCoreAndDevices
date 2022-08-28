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
const int CActiveBlanking::LineMaskSize_ = sizeof( CActiveBlanking::EnabledPattern_ ) * CHAR_BIT;

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

  if ( !ActiveBlankingInterface_->GetNumberOfLines( &NumberOfLines_ ) )
  {
    throw std::runtime_error( "ActiveBlankingInterface GetNumberOfLines failed" );
  }

  if ( NumberOfLines_ <= 0 || NumberOfLines_ > LineMaskSize_ )
  {
    MMILE_->LogMMMessage( "Invalid number of lines returned by ActiveBlankingInterface GetNumberOfLines [" + std::to_string( NumberOfLines_ ) + "]", true );
    NumberOfLines_ = 0;
  }
  else
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
    for ( int vLineIndex = 0; vLineIndex < NumberOfLines_; vLineIndex++ )
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
  // Note: Line is 0-based
  int vMask = 1 << Line;
  return ( EnabledPattern_ & vMask ) != 0;
}

void CActiveBlanking::ChangeLineState( int Line )
{
  // Note: Line is 0-based
  int vMask = 1 << Line;
  EnabledPattern_ ^= vMask;
}

int CActiveBlanking::OnValueChange( MM::PropertyBase * Prop, MM::ActionType Act )
{
  std::string vPropName = Prop->GetName();

  if ( PropertyPointers_.find( vPropName ) == PropertyPointers_.end()
    || PropertyLineIndexMap_.find( vPropName ) == PropertyLineIndexMap_.end() )
  {
    return DEVICE_OK;
  }

  if ( PropertyPointers_[vPropName] == nullptr )
  {
    PropertyPointers_[vPropName] = Prop;
  }

  if ( Act == MM::BeforeGet )
  {
    bool vEnabled = IsLineEnabled( PropertyLineIndexMap_[vPropName] );
    Prop->Set( vEnabled ? g_On : g_Off );
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

    int vLineIndex = PropertyLineIndexMap_[vPropName];
    bool vCurrentlyEnabled = IsLineEnabled( vLineIndex );

    std::string vValue;
    Prop->Get( vValue );
    bool vRequestEnabled = ( vValue == g_On );

    if ( vCurrentlyEnabled != vRequestEnabled )
    {
      ChangeLineState( vLineIndex );
      if ( !ActiveBlankingInterface_->SetActiveBlankingState( EnabledPattern_ ) )
      {
        std::string message = "Active Blanking for line " + std::to_string( static_cast< long long >( vLineIndex ) ) + " FAILED";
        if ( vRequestEnabled )
        {
          MMILE_->LogMMMessage( "Enabling " + message );
        }
        else
        {
          MMILE_->LogMMMessage( "Disabling " + message );
        }
        return ERR_ACTIVEBLANKING_SET;
      }
    }
  }
  return DEVICE_OK;
}

int CActiveBlanking::UpdateILEInterface( IALC_REV_ILEActiveBlankingManagement* ActiveBlankingInterface )
{
  ActiveBlankingInterface_ = ActiveBlankingInterface;
  if ( ActiveBlankingInterface_ != nullptr && NumberOfLines_ > 0 )
  {
    int vNbLines;
    if ( bool vGetNbLinesSuccess = ActiveBlankingInterface_->GetNumberOfLines( &vNbLines ) 
      && vNbLines == NumberOfLines_ )
    {
      if ( ActiveBlankingInterface_->GetActiveBlankingState( &EnabledPattern_ ) )
      {
        MMILE_->LogMMMessage( "Resetting active blanking to device state [" + std::to_string( static_cast<long long>( EnabledPattern_ ) ) + "]", true );
        for ( auto const& vLineIndex : PropertyLineIndexMap_ )
        {
          if ( PropertyPointers_[vLineIndex.first] != nullptr )
          {
            PropertyPointers_[vLineIndex.first]->Set( IsLineEnabled( vLineIndex.second ) ? g_On : g_Off );
          }
        }
      }
      else
      {
        return ERR_ACTIVEBLANKING_GETSTATE;
      }
    }
    else
    {
      if ( vGetNbLinesSuccess )
      {
        MMILE_->LogMMMessage( "Invalid number of lines returned by ActiveBlankingInterface GetNumberOfLines [" + std::to_string( vNbLines ) + "], expected [" + std::to_string( NumberOfLines_ ) + "]", true );
      }
      return ERR_ACTIVEBLANKING_GETNBLINES;
    }
  }
  return DEVICE_OK;
}
