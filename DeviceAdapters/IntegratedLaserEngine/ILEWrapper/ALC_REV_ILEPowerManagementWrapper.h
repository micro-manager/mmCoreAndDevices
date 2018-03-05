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
  int GetNumberOfLasers();
  bool IsLowPowerPresent( bool *Present );
  bool IsLowPowerEnabled( bool *Enabled );
  bool GetLowPowerState( bool *Active );
  bool SetLowPowerState( bool Activate );
  bool GetLowPowerPort( int *PortIndex );
  bool IsCoherenceModePresent( bool *Present );
  bool IsCoherenceModeActive( bool *Active );
  bool SetCoherenceMode( bool Active );
  bool GetPowerRange( int LaserIndex, double *PowerMinPercentage, double *PowerMaxPercentage );
  bool GetLowPowerDetails( int LaserIndex, int *LowPowerPort, int *StepsPerRotation, int *StepsToHome, int *InsertHome );

private:
  IALC_REV_ILEPowerManagement* ALC_REV_ILEPowerManagement_;
};

#endif
