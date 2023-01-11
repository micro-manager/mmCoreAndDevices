///////////////////////////////////////////////////////////////////////////////
// FILE:          ALC_REV_ILEPowerManagement2Wrapper.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------

#include "ALC_REV_ILEPowerManagement2Wrapper.h"
#include "ILESDKLock.h"
#include <stdexcept>


CALC_REV_ILEPowerManagement2Wrapper::CALC_REV_ILEPowerManagement2Wrapper( IALC_REV_ILEPowerManagement2* ALC_REV_ILEPowerManagement2 ) :
  ALC_REV_ILEPowerManagement2_( ALC_REV_ILEPowerManagement2 )
{
  if ( ALC_REV_ILEPowerManagement2_ == nullptr )
  {
    throw std::logic_error( "IALC_REV_ILEPowerManagement2 pointer passed to CALC_REV_ILEPowerManagement2Wrapper is null" );
  }
}

CALC_REV_ILEPowerManagement2Wrapper::~CALC_REV_ILEPowerManagement2Wrapper()
{
}

///////////////////////////////////////////////////////////////////////////////
// IALC_REV_ILEPowerManagement
///////////////////////////////////////////////////////////////////////////////

int CALC_REV_ILEPowerManagement2Wrapper::GetNumberOfLasers()
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILEPowerManagement2_->GetNumberOfLasers();
}

bool CALC_REV_ILEPowerManagement2Wrapper::IsLowPowerPresent( bool *Present )
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILEPowerManagement2_->IsLowPowerPresent( Present );
}

bool CALC_REV_ILEPowerManagement2Wrapper::IsLowPowerEnabled( bool *Enabled )
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILEPowerManagement2_->IsLowPowerEnabled( Enabled );
}

bool CALC_REV_ILEPowerManagement2Wrapper::GetLowPowerState( bool *Active )
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILEPowerManagement2_->GetLowPowerState( Active );
}

bool CALC_REV_ILEPowerManagement2Wrapper::SetLowPowerState( bool Activate )
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILEPowerManagement2_->SetLowPowerState( Activate );
}

bool CALC_REV_ILEPowerManagement2Wrapper::GetLowPowerPort( int *PortIndex )
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILEPowerManagement2_->GetLowPowerPort( PortIndex );
}

bool CALC_REV_ILEPowerManagement2Wrapper::IsCoherenceModePresent( bool *Present )
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILEPowerManagement2_->IsCoherenceModePresent( Present );
}

bool CALC_REV_ILEPowerManagement2Wrapper::IsCoherenceModeActive( bool *Active )
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILEPowerManagement2_->IsCoherenceModeActive( Active );
}

bool CALC_REV_ILEPowerManagement2Wrapper::SetCoherenceMode( bool Active )
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILEPowerManagement2_->SetCoherenceMode( Active );
}

bool CALC_REV_ILEPowerManagement2Wrapper::GetPowerRange( int LaserIndex, double *PowerMinPercentage, double *PowerMaxPercentage )
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILEPowerManagement2_->GetPowerRange( LaserIndex, PowerMinPercentage, PowerMaxPercentage );
}

bool CALC_REV_ILEPowerManagement2Wrapper::GetLowPowerDetails( int LaserIndex, int *LowPowerPort, int *StepsPerRotation, int *StepsToHome, int *InsertHome )
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILEPowerManagement2_->GetLowPowerDetails( LaserIndex, LowPowerPort, StepsPerRotation, StepsToHome, InsertHome );
}

///////////////////////////////////////////////////////////////////////////////
// IALC_REV_ILEPowerManagement2
///////////////////////////////////////////////////////////////////////////////

bool CALC_REV_ILEPowerManagement2Wrapper::GetNumberOfLowPowerLevels( int *NumLevels )
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILEPowerManagement2_->GetNumberOfLowPowerLevels( NumLevels );
}

bool CALC_REV_ILEPowerManagement2Wrapper::GetLowPowerPercentage( int LowPowerIndex, double *Percentage )
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILEPowerManagement2_->GetLowPowerPercentage( LowPowerIndex, Percentage );
}

bool CALC_REV_ILEPowerManagement2Wrapper::GetLowPowerLevel( int *LowPowerIndex )
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILEPowerManagement2_->GetLowPowerLevel( LowPowerIndex );
}

bool CALC_REV_ILEPowerManagement2Wrapper::SetLowPowerLevel( int LowPowerIndex )
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILEPowerManagement2_->SetLowPowerLevel( LowPowerIndex );
}

bool CALC_REV_ILEPowerManagement2Wrapper::IsActivationModePresent( bool *Present )
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILEPowerManagement2_->IsActivationModePresent( Present );
}

bool CALC_REV_ILEPowerManagement2Wrapper::IsActivationModeEnabled( bool *Enabled )
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILEPowerManagement2_->IsActivationModeEnabled( Enabled );
}

bool CALC_REV_ILEPowerManagement2Wrapper::EnableActivationMode( bool Enabled )
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILEPowerManagement2_->EnableActivationMode( Enabled );
}
