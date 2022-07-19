///////////////////////////////////////////////////////////////////////////////
// FILE:          DualILEPorts.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------

#include "DualILEPorts.h"
#include "DualILE.h"
#include "ALC_REV.h"
#include "ALC_REV_ILE2.h"
#include "PortsConfiguration.h"
#include <exception>

const char* const g_PropertyName = "Output Port";

CDualILEPorts::CDualILEPorts( IALC_REV_Port* DualPortInterface, IALC_REV_ILE2* ILE2Interface, CPortsConfiguration* PortsConfiguration, CDualILE* MMILE ) :
  DualPortInterface_( DualPortInterface ),
  ILE2Interface_( ILE2Interface ),
  PortsConfiguration_( PortsConfiguration ),
  MMILE_( MMILE ),
  NbPortsUnit1_( 0 ),
  NbPortsUnit2_( 0 ),
  CurrentPortName_( "" ),
  PropertyPointer_( nullptr )
{
  if ( DualPortInterface_ == nullptr )
  {
    throw std::logic_error( "CDualILEPorts: Pointer to the dual Port interface invalid" );
  }
  if ( ILE2Interface_ == nullptr )
  {
    throw std::logic_error( "CDualILEPorts: Pointer to the ILE interface 2 invalid" );
  }
  if ( PortsConfiguration_ == nullptr )
  {
    throw std::logic_error( "CDualILEPorts: Pointer to Port configuration invalid" );
  }
  if ( MMILE_ == nullptr )
  {
    throw std::logic_error( "CDualILEPorts: Pointer tomain class invalid" );
  }

  DualPortInterface_->InitializePort();
  if ( !ILE2Interface_->GetNumberOfPorts( &NbPortsUnit1_, &NbPortsUnit2_ ) )
  {
    throw std::runtime_error( "CDualILEPorts: Get number of ports failed" );
  }

  std::vector<std::string> vPortList = PortsConfiguration_->GetPortList();

  if ( vPortList.size() > 0 )
  {
    int vUnit1Port, vUnit2Port;
    ILE2Interface_->GetPortIndex( &vUnit1Port, &vUnit2Port );
    CurrentPortName_ = PortsConfiguration_->FindMergedPortForUnitPort( vUnit1Port, vUnit2Port );
    if ( CurrentPortName_ == "" )
    {
      // The combination of ports is invalid for the port configuration so we initialise them
      CurrentPortName_ = vPortList[0];
      MMILE_->LogMMMessage( "Current port combination [" + std::to_string( static_cast<long long>( vUnit1Port ) ) + "," + std::to_string( static_cast<long long>( vUnit1Port ) ) + "] doesn't correspond to any of the ports in the configuration. Changing it to " + CurrentPortName_, true );
      if ( ChangePort( CurrentPortName_ ) != DEVICE_OK )
      {
        throw std::runtime_error( "Error changing port to " + CurrentPortName_ );
      }
    }

    CPropertyAction* vAct = new CPropertyAction( this, &CDualILEPorts::OnPortChange );
    int vRet = MMILE_->CreateStringProperty( g_PropertyName, CurrentPortName_.c_str(), false, vAct );
    if ( vRet != DEVICE_OK )
    {
      throw std::runtime_error( "Error creating " + std::string( g_PropertyName ) + " property" );
    }
    MMILE_->SetAllowedValues( g_PropertyName, vPortList );
  }
  else
  {
    throw std::runtime_error( "Ports configuration invalid or not found." );
  }
}

CDualILEPorts::~CDualILEPorts()
{

}

int CDualILEPorts::ChangePort( const std::string& PortName )
{
  std::vector<int> vPortIndices;
  PortsConfiguration_->GetUnitPortsForMergedPort( PortName, &vPortIndices );

  if ( vPortIndices.size() >= 2 )
  {
    if ( vPortIndices[0] <= NbPortsUnit1_ && vPortIndices[1] <= NbPortsUnit2_ )
    {
      // Updating ports
      MMILE_->LogMMMessage( "Setting port indices to [" + std::to_string( static_cast<long long>( vPortIndices[0] ) ) + "," + std::to_string( static_cast<long long>( vPortIndices[1] ) ) + "]", true );
      if ( ILE2Interface_->SetPortIndex( vPortIndices[0], vPortIndices[1] ) )
      {
        CurrentPortName_ = PortName;
        // Updating lasers
        MMILE_->CheckAndUpdateLasers();
      }
      else
      {
        MMILE_->LogMMMessage( "Changing merged port to " + PortName + " [" + std::to_string( static_cast<long long>( vPortIndices[0] ) ) + ", " + std::to_string( static_cast<long long>( vPortIndices[1] ) ) + "] FAILED" );
        return ERR_DUALPORTS_PORTCHANGEFAIL;
      }
    }
    else
    {
      std::string vMessage = "Changing merged port to " + PortName + " FAILED. ";
      vMessage += "Number of ports [" + std::to_string( static_cast<long long>( NbPortsUnit1_ ) ) + ", " + std::to_string( static_cast<long long>( NbPortsUnit2_ ) ) + "] - ";
      vMessage += "Requested ports [" + std::to_string( static_cast<long long>( vPortIndices[0] ) ) + ", " + std::to_string( static_cast<long long>( vPortIndices[1] ) ) + "]";
      MMILE_->LogMMMessage( vMessage );
      return ERR_DUALPORTS_PORTCONFIGCORRUPTED;
    }
  }
  else
  {
    MMILE_->LogMMMessage( "Changing merged port to " + PortName + " FAILED. The number of units in the configuration is invalid [" + std::to_string( static_cast<long long>( vPortIndices.size() ) ) + "]" );
    return ERR_DUALPORTS_PORTCONFIGCORRUPTED;
  }
  return DEVICE_OK;
}

int CDualILEPorts::OnPortChange( MM::PropertyBase * Prop, MM::ActionType Act )
{
  if ( PropertyPointer_ == nullptr )
  {
    PropertyPointer_ = Prop;
  }
  if ( Act == MM::BeforeGet )
  {
    Prop->Set( CurrentPortName_.c_str() );
  }
  else if ( Act == MM::AfterSet )
  {
    int vInterlockStatus = MMILE_->GetClassIVAndKeyInterlockStatus();
    if ( vInterlockStatus != DEVICE_OK )
    {
      return vInterlockStatus;
    }
    if ( ILE2Interface_ == nullptr )
    {
      return ERR_DEVICE_NOT_CONNECTED;
    }

    std::string vValue;
    Prop->Get( vValue );
    return ChangePort( vValue );
  }
  return DEVICE_OK;
}

int CDualILEPorts::UpdateILEInterface( IALC_REV_Port* DualPortInterface, IALC_REV_ILE2* ILE2Interface )
{
  DualPortInterface_ = DualPortInterface;
  ILE2Interface_ = ILE2Interface;

  if ( ILE2Interface_ != nullptr && DualPortInterface_ != nullptr )
  {
    DualPortInterface_->InitializePort();
    int vUnit1Port, vUnit2Port;
    if ( ILE2Interface_->GetPortIndex( &vUnit1Port, &vUnit2Port ) )
    {
      std::string vCurrentPortName = PortsConfiguration_->FindMergedPortForUnitPort( vUnit1Port, vUnit2Port );
      if ( vCurrentPortName != "" )
      {
        CurrentPortName_ = vCurrentPortName;
        if ( PropertyPointer_ != nullptr )
        {
          PropertyPointer_->Set( CurrentPortName_.c_str() );
        }
        MMILE_->LogMMMessage( "Resetting curren port to device state [" + CurrentPortName_ + " (" + std::to_string( static_cast<long long>( vUnit1Port ) ) + ", " + std::to_string( static_cast<long long>( vUnit2Port ) ) + ")]", true );
      }
    }
    else
    {
      return ERR_PORTS_GET;
    }
  }
  return DEVICE_OK;
}