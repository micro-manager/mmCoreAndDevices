///////////////////////////////////////////////////////////////////////////////
// FILE:          DualILEPorts.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------

#include "DualILEPorts.h"
#include "IntegratedLaserEngine.h"
#include "ALC_REV.h"
#include "PortsConfiguration.h"
#include <exception>

const char* const g_PropertyName = "Output Port";

CDualILEPorts::CDualILEPorts( IALC_REV_Port* Unit1PortInterface, IALC_REV_Port* Unit2PortInterface, CPortsConfiguration* PortsConfiguration, CIntegratedLaserEngine* MMILE ) :
  Unit1PortInterface_( Unit1PortInterface ),
  Unit2PortInterface_( Unit2PortInterface ),
  PortsConfiguration_( PortsConfiguration ),
  MMILE_( MMILE ),
  NbPorts_( 0 )
{
  if ( Unit1PortInterface_ == nullptr )
  {
    throw std::logic_error( "CDualILEPorts: Pointer to the Port interface of Unit1 invalid" );
  }

  if ( Unit2PortInterface_ == nullptr )
  {
    throw std::logic_error( "CDualILEPorts: Pointer to the Port interface of Unit2 invalid" );
  }

  if ( PortsConfiguration_ == nullptr )
  {
    throw std::logic_error( "CDualILEPorts: Pointer to Port configuration invalid" );
  }

  Unit1PortInterface_->InitializePort();
  Unit2PortInterface_->InitializePort();

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
    if ( Unit1PortInterface_ == nullptr || Unit2PortInterface_ == nullptr )
    {
      return ERR_DEVICE_NOT_CONNECTED;
    }

    std::string vValue;
    Prop->Get( vValue );
    std::vector<int> vPortIndices;
    PortsConfiguration_->GetUnitPortsForMergedPort( vValue, &vPortIndices );

    if ( vPortIndices.size() >= 2 )
    {
      int Unit1PreviousPortIndex;
      bool vPreviousPortRetrieved = Unit1PortInterface_->GetPortIndex( &Unit1PreviousPortIndex );
      if ( Unit1PortInterface_->SetPortIndex( vPortIndices[0] ) )
      {
        if ( Unit2PortInterface_->SetPortIndex( vPortIndices[1] ) )
        {
          MMILE_->CheckAndUpdateLasers();
        }
        else
        {
          MMILE_->LogMMMessage( "Changing port of Unit2 to port " + vValue + " FAILED" );
          if ( vPreviousPortRetrieved )
          {
            Unit1PortInterface_->SetPortIndex( Unit1PreviousPortIndex );
          }
          return DEVICE_ERR;
        }
      }
      else
      {
        MMILE_->LogMMMessage( "Changing port of Unit1 to port " + vValue + " FAILED" );
        return DEVICE_ERR;
      }
    }
    else
    {
      return DEVICE_ERR;
    }
  }
  return DEVICE_OK;
}

void CDualILEPorts::UpdateILEInterface( IALC_REV_Port* Unit1PortInterface, IALC_REV_Port* Unit2PortInterface )
{
  Unit1PortInterface_ = Unit1PortInterface;
  Unit2PortInterface_ = Unit2PortInterface;
}