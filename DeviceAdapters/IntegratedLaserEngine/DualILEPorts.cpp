///////////////////////////////////////////////////////////////////////////////
// FILE:          DualILEPorts.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------

#include "DualILEPorts.h"
#include "IntegratedLaserEngine.h"
#include "ALC_REV.h"
#include "ALC_REV_ILE2.h"
#include "PortsConfiguration.h"
#include <exception>

const char* const g_PropertyName = "Output Port";

CDualILEPorts::CDualILEPorts( IALC_REV_Port* DualPortInterface, IALC_REV_ILE2* ILE2Interface, CPortsConfiguration* PortsConfiguration, CIntegratedLaserEngine* MMILE ) :
  DualPortInterface_( DualPortInterface ),
  ILE2Interface_( ILE2Interface ),
  PortsConfiguration_( PortsConfiguration ),
  MMILE_( MMILE ),
  NbPortsUnit1_( 0 ),
  NbPortsUnit2_( 0 )
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

  DualPortInterface_->InitializePort();
  if ( !ILE2Interface_->GetNumberOfPorts( &NbPortsUnit1_, &NbPortsUnit2_ ) )
  {
    throw std::runtime_error( "CDualILEPorts: Get number of ports failed" );
  }

  std::vector<std::string> vPortList = PortsConfiguration_->GetPortList();

  if ( vPortList.size() > 0 )
  {
    CPropertyAction* vAct = new CPropertyAction( this, &CDualILEPorts::OnPortChange );
    int vRet = MMILE_->CreateStringProperty( g_PropertyName, vPortList[0].c_str(), false, vAct );
    if ( vRet != DEVICE_OK )
    {
      throw std::runtime_error( "Error creating " + std::string( g_PropertyName ) + " property" );
    }
    MMILE_->SetAllowedValues( g_PropertyName, vPortList );
  }
}

CDualILEPorts::~CDualILEPorts()
{

}

int CDualILEPorts::OnPortChange( MM::PropertyBase * Prop, MM::ActionType Act )
{
  if ( Act == MM::AfterSet )
  {
    if ( ILE2Interface_ == nullptr )
    {
      return ERR_DEVICE_NOT_CONNECTED;
    }

    std::string vValue;
    Prop->Get( vValue );
    std::vector<int> vPortIndices;
    PortsConfiguration_->GetUnitPortsForMergedPort( vValue, &vPortIndices );

    if ( vPortIndices.size() >= 2 )
    {      
      if ( vPortIndices[0] <= NbPortsUnit1_ && vPortIndices[1] <= NbPortsUnit2_ )
      {
        // Updating ports
        if ( ILE2Interface_->SetPortIndex( vPortIndices[0], vPortIndices[1] ) )
        {
          // Updating lasers
          MMILE_->CheckAndUpdateLasers();
        }
        else
        {
          MMILE_->LogMMMessage( "Changing merged port to " + vValue + " [" + std::to_string( static_cast< long long >( vPortIndices[0] ) ) + ", " + std::to_string( static_cast< long long >( vPortIndices[1] ) ) + "] FAILED" );
          return ERR_DUALPORTS_PORTCHANGEFAIL;
        }
      }
      else
      {
        std::string vMessage = "Changing merged port to " + vValue + " FAILED. ";
        vMessage += "Number of ports [" + std::to_string( static_cast< long long >( NbPortsUnit1_ ) ) + ", " + std::to_string( static_cast< long long >( NbPortsUnit2_ ) ) + "] - ";
        vMessage += "Requested ports [" + std::to_string( static_cast< long long >( vPortIndices[0] ) ) + ", " + std::to_string( static_cast< long long >( vPortIndices[1] ) ) + "]";
        MMILE_->LogMMMessage( vMessage );
        return ERR_DUALPORTS_PORTCONFIGCORRUPTED;
      }
    }
    else
    {
      MMILE_->LogMMMessage( "Changing merged port to " + vValue + " FAILED. The number of units in the configuration is invalid [" + std::to_string( vPortIndices.size() ) +  "]" );
      return ERR_DUALPORTS_PORTCONFIGCORRUPTED;
    }
  }
  return DEVICE_OK;
}

void CDualILEPorts::UpdateILEInterface( IALC_REV_Port* DualPortInterface, IALC_REV_ILE2* ILE2Interface )
{
  DualPortInterface_ = DualPortInterface;
  ILE2Interface_ = ILE2Interface;
}