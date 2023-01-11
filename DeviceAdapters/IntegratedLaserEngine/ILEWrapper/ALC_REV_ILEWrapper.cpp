///////////////////////////////////////////////////////////////////////////////
// FILE:          ALC_REV_ILEWrapper.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------

#include "ALC_REV_ILEWrapper.h"
#include "ILESDKLock.h"
#include <stdexcept>


CALC_REV_ILEWrapper::CALC_REV_ILEWrapper( IALC_REV_ILE* ALC_REV_ILE ) :
  ALC_REV_ILE_( ALC_REV_ILE )
{
  if ( ALC_REV_ILE_ == nullptr )
  {
    throw std::logic_error( "IALC_REV_ILE pointer passed to CALC_REV_ILEWrapper is null" );
  }
}

CALC_REV_ILEWrapper::~CALC_REV_ILEWrapper()
{
}

///////////////////////////////////////////////////////////////////////////////
// IALC_REV_ILE
///////////////////////////////////////////////////////////////////////////////

bool CALC_REV_ILEWrapper::SetFLICRThreshold( int Percent )
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILE_->SetFLICRThreshold( Percent );
}

bool CALC_REV_ILEWrapper::GetFLICRThreshold( int *Percent )
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILE_->GetFLICRThreshold( Percent );
}

bool CALC_REV_ILEWrapper::GetLaserWarmUpTime( int LaserIndex, int *Minutes )
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILE_->GetLaserWarmUpTime( LaserIndex, Minutes );
}

bool CALC_REV_ILEWrapper::GetHours( double *Hours, double *Lifetime )
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILE_->GetHours( Hours, Lifetime );
}

bool CALC_REV_ILEWrapper::GetLaserHours( int LaserIndex, double *Hours )
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILE_->GetLaserHours( LaserIndex, Hours );
}

bool CALC_REV_ILEWrapper::GetAttenuationWheelDetails( int LaserIndex, int *AttenuationWheelPresent, int *StepsPerRotation, int *StepsToHome )
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILE_->GetAttenuationWheelDetails( LaserIndex, AttenuationWheelPresent, StepsPerRotation, StepsToHome );
}

bool CALC_REV_ILEWrapper::GetPowerIntoInputFibre( double *Power_mW )
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILE_->GetPowerIntoInputFibre( Power_mW );
}

bool CALC_REV_ILEWrapper::SetPowerIntoInputFibre( double Power_mW )
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILE_->SetPowerIntoInputFibre( Power_mW );
}

bool CALC_REV_ILEWrapper::IsClassIVInterlockFlagActive( bool *Active )
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILE_->IsClassIVInterlockFlagActive( Active );
}

bool CALC_REV_ILEWrapper::ClearClassIVInterlockFlag()
{
  CILESDKLock vSDKLock;
  return ALC_REV_ILE_->ClearClassIVInterlockFlag();
}
