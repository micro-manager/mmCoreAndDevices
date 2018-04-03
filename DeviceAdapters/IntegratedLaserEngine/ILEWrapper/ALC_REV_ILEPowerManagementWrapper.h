///////////////////////////////////////////////////////////////////////////////
// FILE:          ALC_REV_ILEPowerManagementWrapper.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------

#ifndef _ALC_REV_ILEPOWERMGMTWRAPPER_H_
#define _ALC_REV_ILEPOWERMGMTWRAPPER_H_

#include "ALC_REV.h"

class CALC_REV_ILEPowerManagementWrapper : public IALC_REV_ILEPowerManagement
{
public:
  CALC_REV_ILEPowerManagementWrapper( IALC_REV_ILEPowerManagement* ALC_REV_ILEPowerManagement );
  ~CALC_REV_ILEPowerManagementWrapper();

  // IALC_REV_ILEPowerManagement
  int __stdcall GetNumberOfLasers();
  bool __stdcall IsLowPowerPresent( bool *Present );
  bool __stdcall IsLowPowerEnabled( bool *Enabled );
  bool __stdcall GetLowPowerState( bool *Active );
  bool __stdcall SetLowPowerState( bool Activate );
  bool __stdcall GetLowPowerPort( int *PortIndex );
  bool __stdcall IsCoherenceModePresent( bool *Present );
  bool __stdcall IsCoherenceModeActive( bool *Active );
  bool __stdcall SetCoherenceMode( bool Active );
  bool __stdcall GetPowerRange( int LaserIndex, double *PowerMinPercentage, double *PowerMaxPercentage );
  bool __stdcall GetLowPowerDetails( int LaserIndex, int *LowPowerPort, int *StepsPerRotation, int *StepsToHome, int *InsertHome );

private:
  IALC_REV_ILEPowerManagement* ALC_REV_ILEPowerManagement_;
};

#endif
