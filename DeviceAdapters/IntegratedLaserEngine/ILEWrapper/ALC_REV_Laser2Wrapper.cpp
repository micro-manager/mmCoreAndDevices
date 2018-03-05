///////////////////////////////////////////////////////////////////////////////
// FILE:          ALC_REV_Laser2Wrapper.cpp
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------

#include "ALC_REV_Laser2Wrapper.h"
#include "ILESDKLock.h"
#include <stdexcept>


CALC_REV_Laser2Wrapper::CALC_REV_Laser2Wrapper( IALC_REV_Laser2* ALC_REV_Laser2 ) :
  ALC_REV_Laser2_( ALC_REV_Laser2 )
{
  if ( ALC_REV_Laser2_ == nullptr )
  {
    throw std::logic_error( "IALC_REV_Laser2 pointer passed to CALC_REV_Laser2Wrapper is null" );
  }
}

CALC_REV_Laser2Wrapper::~CALC_REV_Laser2Wrapper()
{
}

///////////////////////////////////////////////////////////////////////////////
// IALC_REV_Laser2
///////////////////////////////////////////////////////////////////////////////

int CALC_REV_Laser2Wrapper::Initialize( void )
{
  CILESDKLock vSDKLock;
  return ALC_REV_Laser2_->Initialize();
}

int CALC_REV_Laser2Wrapper::GetNumberOfLasers()
{
  CILESDKLock vSDKLock;
  return ALC_REV_Laser2_->GetNumberOfLasers();
}

bool CALC_REV_Laser2Wrapper::GetWavelength( int LaserIndex, int *Wavelength )
{
  CILESDKLock vSDKLock;
  return ALC_REV_Laser2_->GetWavelength( LaserIndex, Wavelength );
}

bool CALC_REV_Laser2Wrapper::GetPower( int LaserIndex, int *Power )
{
  CILESDKLock vSDKLock;
  return ALC_REV_Laser2_->GetPower( LaserIndex, Power );
}

bool CALC_REV_Laser2Wrapper::IsLaserOutputLinearised( int LaserIndex, int *Linearised )
{
  CILESDKLock vSDKLock;
  return ALC_REV_Laser2_->IsLaserOutputLinearised( LaserIndex, Linearised );
}

bool CALC_REV_Laser2Wrapper::IsEnabled( int LaserIndex, int *Enabled )
{
  CILESDKLock vSDKLock;
  return ALC_REV_Laser2_->IsEnabled( LaserIndex, Enabled );
}

bool CALC_REV_Laser2Wrapper::Enable( int LaserIndex )
{
  CILESDKLock vSDKLock;
  return ALC_REV_Laser2_->Enable( LaserIndex );
}

bool CALC_REV_Laser2Wrapper::Disable( int LaserIndex )
{
  CILESDKLock vSDKLock;
  return ALC_REV_Laser2_->Disable( LaserIndex );
}

bool CALC_REV_Laser2Wrapper::IsControlModeAvailable( int LaserIndex, int *Available )
{
  CILESDKLock vSDKLock;
  return ALC_REV_Laser2_->IsControlModeAvailable( LaserIndex, Available );
}

bool CALC_REV_Laser2Wrapper::GetControlMode( int LaserIndex, int *ControlMode )
{
  CILESDKLock vSDKLock;
  return ALC_REV_Laser2_->GetControlMode( LaserIndex, ControlMode );
}

bool CALC_REV_Laser2Wrapper::SetControlMode( int LaserIndex, int Mode )
{
  CILESDKLock vSDKLock;
  return ALC_REV_Laser2_->SetControlMode( LaserIndex, Mode );
}

bool CALC_REV_Laser2Wrapper::GetLaserState( int LaserIndex, TLaserState *LaserState )
{
  CILESDKLock vSDKLock;
  return ALC_REV_Laser2_->GetLaserState( LaserIndex, LaserState );
}

bool CALC_REV_Laser2Wrapper::GetLaserHours( int LaserIndex, int *Hours )
{
  CILESDKLock vSDKLock;
  return ALC_REV_Laser2_->GetLaserHours( LaserIndex, Hours );
}

bool CALC_REV_Laser2Wrapper::GetCurrentPower( int LaserIndex, double *CurrentPower )
{
  CILESDKLock vSDKLock;
  return ALC_REV_Laser2_->GetCurrentPower( LaserIndex, CurrentPower );
}

bool CALC_REV_Laser2Wrapper::SetLas_W( int Wavelength, double Power, bool On )
{
  CILESDKLock vSDKLock;
  return ALC_REV_Laser2_->SetLas_W( Wavelength, Power, On );
}

bool CALC_REV_Laser2Wrapper::SetLas_I( int LaserIndex, double Power, bool On )
{
  CILESDKLock vSDKLock;
  return ALC_REV_Laser2_->SetLas_I( LaserIndex, Power, On );
}

bool CALC_REV_Laser2Wrapper::GetLas_W( int Wavelength, double *Power, bool *On )
{
  CILESDKLock vSDKLock;
  return ALC_REV_Laser2_->GetLas_W( Wavelength, Power, On );
}

bool CALC_REV_Laser2Wrapper::GetLas_I( int LaserIndex, double *Power, bool *On )
{
  CILESDKLock vSDKLock;
  return ALC_REV_Laser2_->GetLas_I( LaserIndex, Power, On );
}

bool CALC_REV_Laser2Wrapper::SetLas_Shutter( bool Open )
{
  CILESDKLock vSDKLock;
  return ALC_REV_Laser2_->SetLas_Shutter( Open );
}

bool CALC_REV_Laser2Wrapper::GetNumberOfPorts( int *NumberOfPorts )
{
  CILESDKLock vSDKLock;
  return ALC_REV_Laser2_->GetNumberOfPorts( NumberOfPorts );
}

bool CALC_REV_Laser2Wrapper::GetPowerLimit( int PortIndex, double *PowerLimit_mW )
{
  CILESDKLock vSDKLock;
  return ALC_REV_Laser2_->GetPowerLimit( PortIndex, PowerLimit_mW );
}

bool CALC_REV_Laser2Wrapper::GetPortForPowerLimit( int *Port )
{
  CILESDKLock vSDKLock;
  return ALC_REV_Laser2_->GetPortForPowerLimit( Port );
}

bool CALC_REV_Laser2Wrapper::SetPortForPowerLimit( int Port )
{
  CILESDKLock vSDKLock;
  return ALC_REV_Laser2_->SetPortForPowerLimit( Port );
}

bool CALC_REV_Laser2Wrapper::GetCurrentPowerIntoFiber( double *Power_mW )
{
  CILESDKLock vSDKLock;
  return ALC_REV_Laser2_->GetCurrentPowerIntoFiber( Power_mW );
}

bool CALC_REV_Laser2Wrapper::CalculatePowerIntoFibre( int LaserIndex, double PercentPower, double *Power_mW )
{
  CILESDKLock vSDKLock;
  return ALC_REV_Laser2_->CalculatePowerIntoFibre( LaserIndex, PercentPower, Power_mW );
}

bool CALC_REV_Laser2Wrapper::GetPowerStatus( int *PowerStatus )
{
  CILESDKLock vSDKLock;
  return ALC_REV_Laser2_->GetPowerStatus( PowerStatus );
}

bool CALC_REV_Laser2Wrapper::WasLaserIlluminationProhibitedOnLastChange( int LaserIndex, int *Prohibited )
{
  CILESDKLock vSDKLock;
  return ALC_REV_Laser2_->WasLaserIlluminationProhibitedOnLastChange( LaserIndex, Prohibited );
}

bool CALC_REV_Laser2Wrapper::IsLaserIlluminationProhibited( int LaserIndex, int *Prohibited )
{
  CILESDKLock vSDKLock;
  return ALC_REV_Laser2_->IsLaserIlluminationProhibited( LaserIndex, Prohibited );
}

