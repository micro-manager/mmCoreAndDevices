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
  int __stdcall Initialize( void );
  int __stdcall GetNumberOfLasers();
  bool __stdcall GetWavelength( int LaserIndex, int *Wavelength );
  bool __stdcall GetPower( int LaserIndex, int *Power );
  bool __stdcall IsLaserOutputLinearised( int LaserIndex, int *Linearised );
  bool __stdcall IsEnabled( int LaserIndex, int *Enabled );
  bool __stdcall Enable( int LaserIndex );
  bool __stdcall Disable( int LaserIndex );
  bool __stdcall IsControlModeAvailable( int LaserIndex, int *Available );
  bool __stdcall GetControlMode( int LaserIndex, int *ControlMode );
  bool __stdcall SetControlMode( int LaserIndex, int Mode );
  bool __stdcall GetLaserState( int LaserIndex, TLaserState *LaserState );
  bool __stdcall GetLaserHours( int LaserIndex, int *Hours );
  bool __stdcall GetCurrentPower( int LaserIndex, double *CurrentPower );
  bool __stdcall SetLas_W( int Wavelength, double Power, bool On );
  bool __stdcall SetLas_I( int LaserIndex, double Power, bool On );
  bool __stdcall GetLas_W( int Wavelength, double *Power, bool *On );
  bool __stdcall GetLas_I( int LaserIndex, double *Power, bool *On );
  bool __stdcall SetLas_Shutter( bool Open );
  bool __stdcall GetNumberOfPorts( int *NumberOfPorts );
  bool __stdcall GetPowerLimit( int PortIndex, double *PowerLimit_mW );
  bool __stdcall GetPortForPowerLimit( int *Port );
  bool __stdcall SetPortForPowerLimit( int Port );
  bool __stdcall GetCurrentPowerIntoFiber( double *Power_mW );
  bool __stdcall CalculatePowerIntoFibre( int LaserIndex, double PercentPower, double *Power_mW );
  bool __stdcall GetPowerStatus( int *PowerStatus );
  bool __stdcall WasLaserIlluminationProhibitedOnLastChange( int LaserIndex, int *Prohibited );
  bool __stdcall IsLaserIlluminationProhibited( int LaserIndex, int *Prohibited );

private:
  IALC_REV_Laser2* ALC_REV_Laser2_;
};

#endif
