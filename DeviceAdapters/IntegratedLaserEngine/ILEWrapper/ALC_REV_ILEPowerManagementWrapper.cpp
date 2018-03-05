///////////////////////////////////////////////////////////////////////////////
// FILE:          ALC_REV_ILEPowerManagementWrapper.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------

#include "ALC_REV_ILEPowerManagementWrapper.h"
#include "ILESDKLock.h"
#include <stdexcept>


CALC_REV_ILEPowerManagementWrapper::CALC_REV_ILEPowerManagementWrapper( IALC_REV_ILEPowerManagement* ALC_REV_ILEPowerManagement ) :
  ALC_REV_ILEPowerManagement_( ALC_REV_ILEPowerManagement )
{
  if ( ALC_REV_ILEPowerManagement_ == nullptr )
  {
    throw std::logic_error( "IALC_REV_ILEPowerManagement pointer passed to CALC_REV_ILEPowerManagementWrapper is null" );
  }
}

CALC_REV_ILEPowerManagementWrapper::~CALC_REV_ILEPowerManagementWrapper()
{
}

///////////////////////////////////////////////////////////////////////////////
// IALC_REV_ILEPowerManagement
///////////////////////////////////////////////////////////////////////////////

int CALC_REV_ILEPowerManagementWrapper::GetNumberOfLasers()
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILEPowerManagement_->GetNumberOfLasers();
}

bool CALC_REV_ILEPowerManagementWrapper::IsLowPowerPresent( bool *Present )
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILEPowerManagement_->IsLowPowerPresent( Present );
}

bool CALC_REV_ILEPowerManagementWrapper::IsLowPowerEnabled( bool *Enabled )
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILEPowerManagement_->IsLowPowerEnabled( Enabled );
}

bool CALC_REV_ILEPowerManagementWrapper::GetLowPowerState( bool *Active )
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILEPowerManagement_->GetLowPowerState( Active );
}

bool CALC_REV_ILEPowerManagementWrapper::SetLowPowerState( bool Activate )
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILEPowerManagement_->SetLowPowerState( Activate );
}

bool CALC_REV_ILEPowerManagementWrapper::GetLowPowerPort( int *PortIndex )
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILEPowerManagement_->GetLowPowerPort( PortIndex );
}

bool CALC_REV_ILEPowerManagementWrapper::IsCoherenceModePresent( bool *Present )
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILEPowerManagement_->IsCoherenceModePresent( Present );
}

bool CALC_REV_ILEPowerManagementWrapper::IsCoherenceModeActive( bool *Active )
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILEPowerManagement_->IsCoherenceModeActive( Active );
}

bool CALC_REV_ILEPowerManagementWrapper::SetCoherenceMode( bool Active )
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILEPowerManagement_->SetCoherenceMode( Active );
}

bool CALC_REV_ILEPowerManagementWrapper::GetPowerRange( int LaserIndex, double *PowerMinPercentage, double *PowerMaxPercentage )
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILEPowerManagement_->GetPowerRange( LaserIndex, PowerMinPercentage, PowerMaxPercentage );
}

bool CALC_REV_ILEPowerManagementWrapper::GetLowPowerDetails( int LaserIndex, int *LowPowerPort, int *StepsPerRotation, int *StepsToHome, int *InsertHome )
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILEPowerManagement_->GetLowPowerDetails( LaserIndex, LowPowerPort, StepsPerRotation, StepsToHome, InsertHome );
}
