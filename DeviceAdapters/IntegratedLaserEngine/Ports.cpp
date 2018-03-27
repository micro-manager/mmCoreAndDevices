///////////////////////////////////////////////////////////////////////////////
// FILE:          Ports.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------

#include "Ports.h"
#include "IntegratedLaserEngine.h"
#include "ALC_REV.h"
#include <exception>

const char* const g_PropertyName = "Output Port";

CPorts::CPorts( IALC_REV_Port* PortInterface, CIntegratedLaserEngine* MMILE ) :
  PortInterface_( PortInterface ),
  MMILE_( MMILE ),
  NbPorts_( 0 ),
  CurrentPortIndex_( 0 ),
  PropertyPointer_( nullptr )
{
  if ( PortInterface_ == nullptr )
  {
    throw std::logic_error( "CPorts: Pointer to Port interface invalid" );
  }
  if ( MMILE_ == nullptr )
  {
    throw std::logic_error( "CPorts: Pointer tomain class invalid" );
  }

  NbPorts_ = PortInterface_->InitializePort();

  if ( NbPorts_ > 0 )
  {
    // Build the list of ports starting with "A"
    std::vector<std::string> vPorts;
    char vPortName[2];
    vPortName[1] = 0;
    for ( int vPortIndex = 1; vPortIndex < NbPorts_ + 1; vPortIndex++ )
    {
      vPortName[0] = PortIndexToName( vPortIndex );
      vPorts.push_back( vPortName );
    }

    vPortName[0] = 'A';
    if ( PortInterface_->GetPortIndex( &CurrentPortIndex_ ) )
    {
      vPortName[0] = PortIndexToName( CurrentPortIndex_ );
    }
    else
    {
      throw std::runtime_error( "CPorts: GetPortIndex FAILED" );
    }

    CPropertyAction* vAct = new CPropertyAction( this, &CPorts::OnPortChange );
    int vRet = MMILE_->CreateStringProperty( g_PropertyName, vPortName, false, vAct );
    if ( vRet != DEVICE_OK )
    {
      throw std::runtime_error( "Error creating " + std::string( g_PropertyName ) + " property" );
    }
    MMILE_->SetAllowedValues( g_PropertyName, vPorts );
  }
}

CPorts::~CPorts()
{

}

char CPorts::PortIndexToName( int PortIndex )
{
  return (char)( 'A' + PortIndex - 1 );
}

int CPorts::PortNameToIndex( char PortName )
{
  return PortName - 'A' + 1;
}

int CPorts::OnPortChange( MM::PropertyBase * Prop, MM::ActionType Act )
{
  if ( PropertyPointer_ == nullptr )
  {
    PropertyPointer_ = Prop;
  }
  if ( Act == MM::BeforeGet )
  {
    char vPortName[2];
    vPortName[1] = 0;
    vPortName[0] = PortIndexToName( CurrentPortIndex_ );
    Prop->Set( vPortName );
  }
  else if ( Act == MM::AfterSet )
  {
    int vInterlockStatus = MMILE_->GetClassIVAndKeyInterlockStatus();
    if ( vInterlockStatus != DEVICE_OK )
    {
      return vInterlockStatus;
    }
    if ( PortInterface_ == nullptr )
    {
      return ERR_DEVICE_NOT_CONNECTED;
    }

    std::string vValue;
    Prop->Get( vValue );
    char vPortName = vValue[0];
    int vPortIndex = PortNameToIndex( vPortName );
    if ( PortInterface_->SetPortIndex( vPortIndex ) )
    {
      CurrentPortIndex_ = vPortIndex;
      MMILE_->CheckAndUpdateLasers();
    }
    else
    {
      MMILE_->LogMMMessage( "Changing port to port " + vValue + " FAILED" );
      return ERR_PORTS_SET;
    }
  }
  return DEVICE_OK;
}

int CPorts::UpdateILEInterface( IALC_REV_Port* PortInterface )
{
  PortInterface_ = PortInterface;
  if ( PortInterface_ != nullptr )
  {
    PortInterface_->InitializePort();
    char vPortName[2];
    vPortName[1] = 0;
    if ( PortInterface_->GetPortIndex( &CurrentPortIndex_ ) )
    {
      MMILE_->LogMMMessage( "Resetting current port to device state [" + std::to_string( static_cast<long long>( CurrentPortIndex_ ) ) + "]", true );
      vPortName[0] = PortIndexToName( CurrentPortIndex_ );
      if ( PropertyPointer_ != nullptr )
      {
        PropertyPointer_->Set( vPortName );
      }
    }
    else
    {
      return ERR_PORTS_GET;
    }
  }
  return DEVICE_OK;
}