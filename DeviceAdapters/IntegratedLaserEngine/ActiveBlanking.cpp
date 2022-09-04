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

const char* const g_PropertyBaseName = "Laser Active Blanking";
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

#ifdef USE_PORT_SPECIFIC_ACTIVE_BLANKING
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
#else
    // Retrieve Active Blanking value
    bool vActiveBlankingEnabled = InitialiseActiveBlanking();

    // Create property
    PropertyLineIndexMap_[g_PropertyBaseName] = 0;
    PropertyPointers_[g_PropertyBaseName] = nullptr;

    std::vector<std::string> vAllowedValues;
    vAllowedValues.push_back( g_On );
    vAllowedValues.push_back( g_Off );

    CPropertyAction* vAct = new CPropertyAction( this, &CActiveBlanking::OnValueChange );
    MMILE_->CreateStringProperty( g_PropertyBaseName, vActiveBlankingEnabled ? g_On : g_Off, false, vAct );
    MMILE_->SetAllowedValues( g_PropertyBaseName, vAllowedValues );
#endif
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

bool CActiveBlanking::IsActiveBlankingEnabled() const
{
  return IsLineEnabled( 0 );
}

void CActiveBlanking::EnableActiveBlanking()
{
  // Set all ports to 1
  if ( NumberOfLines_ < LineMaskSize_ - 1 )
  {
    EnabledPattern_ = ( 1 << NumberOfLines_ ) - 1;
  }
  else if ( NumberOfLines_ == LineMaskSize_ - 1 )
  {
    //0x7FFFFFFF for 32b int
    EnabledPattern_ = ~( 1 << NumberOfLines_ );
  }
  else if ( NumberOfLines_ == LineMaskSize_ )
  {
    //0xFFFFFFFF for 32b int
    EnabledPattern_ = ~( EnabledPattern_ & 0 );
  }
}

void CActiveBlanking::DisableActiveBlanking()
{
  EnabledPattern_ = 0;
}

bool CActiveBlanking::InitialiseActiveBlanking()
{
  bool vActiveBlankingEnabled = false;
  if ( NumberOfLines_ < 2 )
  {
    // Use the only port available (port A)
    vActiveBlankingEnabled = IsLineEnabled( 0 );
  }
  else
  {
    // Use port B if it exists
    vActiveBlankingEnabled = IsLineEnabled( 1 );
  }

  // Ensure all ports are properly initialised
  EnabledPattern_ = 0;
  if ( vActiveBlankingEnabled )
  {
    EnableActiveBlanking();
  }

  int vInterlockStatus = MMILE_->GetClassIVAndKeyInterlockStatus();
  if ( vInterlockStatus == DEVICE_OK )
  {
    if ( !ActiveBlankingInterface_->SetActiveBlankingState( EnabledPattern_ ) )
    {
      throw std::runtime_error( "ActiveBlankingInterface SetActiveBlankingState failed on initialisation" );
    }
  }

  return vActiveBlankingEnabled;
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
#ifdef USE_PORT_SPECIFIC_ACTIVE_BLANKING
      ChangeLineState( vLineIndex );
#else
      if ( vRequestEnabled )
      {
        EnableActiveBlanking();
      }
      else
      {
        DisableActiveBlanking();
      }
#endif
      MMILE_->LogMMMessage( "Set Active Blanking state to [" + std::to_string( EnabledPattern_ ) + "]", true );
      if ( !ActiveBlankingInterface_->SetActiveBlankingState( EnabledPattern_ ) )
      {
#ifdef USE_PORT_SPECIFIC_ACTIVE_BLANKING
        std::string message = "Active Blanking for line " + std::to_string( static_cast< long long >( vLineIndex ) ) + " FAILED";
#else
        std::string message = "Active Blanking FAILED";
#endif
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
#ifdef USE_PORT_SPECIFIC_ACTIVE_BLANKING
        MMILE_->LogMMMessage( "Resetting active blanking to device state [" + std::to_string( static_cast<long long>( EnabledPattern_ ) ) + "]", true );
        for ( auto const& vLineIndex : PropertyLineIndexMap_ )
        {
          if ( PropertyPointers_[vLineIndex.first] != nullptr )
          {
            PropertyPointers_[vLineIndex.first]->Set( IsLineEnabled( vLineIndex.second ) ? g_On : g_Off );
          }
        }
#else
        bool vActiveBlankingEnabled = InitialiseActiveBlanking();
        MMILE_->LogMMMessage( "Resetting active blanking to device state [" + vActiveBlankingEnabled ? std::string("Enabled") : std::string( "Disabled" ) + "]", true);
        if ( PropertyPointers_[g_PropertyBaseName] != nullptr )
        {
          PropertyPointers_[g_PropertyBaseName]->Set( vActiveBlankingEnabled ? g_On : g_Off );
        }
#endif
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
