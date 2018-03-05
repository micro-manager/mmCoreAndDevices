///////////////////////////////////////////////////////////////////////////////
// FILE:          ALC_REV_Laser2Wrapper.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------

#ifndef _ALC_REV_LASER2WRAPPER_H_
#define _ALC_REV_LASER2WRAPPER_H_

#include "ALC_REV.h"

class CALC_REV_Laser2Wrapper : public IALC_REV_Laser2
{
public:
  CALC_REV_Laser2Wrapper( IALC_REV_Laser2* ALC_REV_Laser2 );
  ~CALC_REV_Laser2Wrapper();

  // IALC_REV_Laser2
  int Initialize( void );
  int GetNumberOfLasers();
  bool GetWavelength( int LaserIndex, int *Wavelength );
  bool GetPower( int LaserIndex, int *Power );
  bool IsLaserOutputLinearised( int LaserIndex, int *Linearised );
  bool IsEnabled( int LaserIndex, int *Enabled );
  bool Enable( int LaserIndex );
  bool Disable( int LaserIndex );
  bool IsControlModeAvailable( int LaserIndex, int *Available );
  bool GetControlMode( int LaserIndex, int *ControlMode );
  bool SetControlMode( int LaserIndex, int Mode );
  bool GetLaserState( int LaserIndex, TLaserState *LaserState );
  bool GetLaserHours( int LaserIndex, int *Hours );
  bool GetCurrentPower( int LaserIndex, double *CurrentPower );
  bool SetLas_W( int Wavelength, double Power, bool On );
  bool SetLas_I( int LaserIndex, double Power, bool On );
  bool GetLas_W( int Wavelength, double *Power, bool *On );
  bool GetLas_I( int LaserIndex, double *Power, bool *On );
  bool SetLas_Shutter( bool Open );
  bool GetNumberOfPorts( int *NumberOfPorts );
  bool GetPowerLimit( int PortIndex, double *PowerLimit_mW );
  bool GetPortForPowerLimit( int *Port );
  bool SetPortForPowerLimit( int Port );
  bool GetCurrentPowerIntoFiber( double *Power_mW );
  bool CalculatePowerIntoFibre( int LaserIndex, double PercentPower, double *Power_mW );
  bool GetPowerStatus( int *PowerStatus );
  bool WasLaserIlluminationProhibitedOnLastChange( int LaserIndex, int *Prohibited );
  bool IsLaserIlluminationProhibited( int LaserIndex, int *Prohibited );

private:
  IALC_REV_Laser2* ALC_REV_Laser2_;
};

#endif
