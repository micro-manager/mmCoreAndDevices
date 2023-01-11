///////////////////////////////////////////////////////////////////////////////
// FILE:          ALC_REV_PortWrapper.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------

#include "ALC_REV_PortWrapper.h"
#include "ILESDKLock.h"
#include <stdexcept>


CALC_REV_PortWrapper::CALC_REV_PortWrapper( IALC_REV_Port* ALC_REV_Port ) :
  ALC_REV_Port_( ALC_REV_Port )
{
  if ( ALC_REV_Port_ == nullptr )
  {
    throw std::logic_error( "IALC_REV_Port pointer passed to CALC_REV_PortWrapper is null" );
  }
}

CALC_REV_PortWrapper::~CALC_REV_PortWrapper()
{
}

///////////////////////////////////////////////////////////////////////////////
// IALC_REV_Port
///////////////////////////////////////////////////////////////////////////////

int CALC_REV_PortWrapper::InitializePort( void )
{
  CILESDKLock vSDKLock;
  return ALC_REV_Port_->InitializePort();
}

bool CALC_REV_PortWrapper::GetNumberOfPorts( int *NumberOfPorts )
{
  CILESDKLock vSDKLock;
  return ALC_REV_Port_->GetNumberOfPorts( NumberOfPorts );
}

bool CALC_REV_PortWrapper::GetPortIndex( int *PortIndex )
{
  CILESDKLock vSDKLock;
  return ALC_REV_Port_->GetPortIndex( PortIndex );
}

bool CALC_REV_PortWrapper::SetPortIndex( int PortIndex )
{
  CILESDKLock vSDKLock;
  return ALC_REV_Port_->SetPortIndex( PortIndex );
}

