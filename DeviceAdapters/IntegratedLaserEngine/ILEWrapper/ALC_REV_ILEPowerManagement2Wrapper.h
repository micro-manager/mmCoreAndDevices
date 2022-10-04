///////////////////////////////////////////////////////////////////////////////
// FILE:          ALC_REV_ILEPowerManagement2Wrapper.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------

#ifndef _ALC_REV_ILEPOWERMGMT2WRAPPER_H_
#define _ALC_REV_ILEPOWERMGMT2WRAPPER_H_

#include "ALC_REV.h"

class CALC_REV_ILEPowerManagement2Wrapper : public IALC_REV_ILEPowerManagement2
{
public:
  CALC_REV_ILEPowerManagement2Wrapper( IALC_REV_ILEPowerManagement2* ALC_REV_ILEPowerManagement2 );
  ~CALC_REV_ILEPowerManagement2Wrapper();

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

  // IALC_REV_ILEPowerManagement2
  bool __stdcall GetNumberOfLowPowerLevels( int *NumLevels );
  bool __stdcall GetLowPowerPercentage( int LowPowerIndex, double *Percentage );
  bool __stdcall GetLowPowerLevel( int *LowPowerIndex );
  bool __stdcall SetLowPowerLevel( int LowPowerIndex );
  bool __stdcall IsActivationModePresent( bool *Present );
  bool __stdcall IsActivationModeEnabled( bool *Enabled );
  bool __stdcall EnableActivationMode( bool Enabled );

private:
  IALC_REV_ILEPowerManagement2* ALC_REV_ILEPowerManagement2_;
};

#endif
