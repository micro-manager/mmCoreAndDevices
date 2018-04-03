///////////////////////////////////////////////////////////////////////////////
// FILE:          ALC_REV_ILEWrapper.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------

#ifndef _ALC_REV_ILEWRAPPER_H_
#define _ALC_REV_ILEWRAPPER_H_

#include "ALC_REV.h"

class CALC_REV_ILEWrapper : public IALC_REV_ILE
{
public:
  CALC_REV_ILEWrapper( IALC_REV_ILE* ALC_REV_ILE );
  ~CALC_REV_ILEWrapper();

  // IALC_REV_ILE
  bool __stdcall SetFLICRThreshold( int Percent );
  bool __stdcall GetFLICRThreshold( int *Percent );
  bool __stdcall GetLaserWarmUpTime( int LaserIndex, int *Minutes );
  bool __stdcall GetHours( double *Hours, double *Lifetime );
  bool __stdcall GetLaserHours( int LaserIndex, double *Hours );
  bool __stdcall GetAttenuationWheelDetails( int LaserIndex, int *AttenuationWheelPresent, int *StepsPerRotation, int *StepsToHome );
  bool __stdcall GetPowerIntoInputFibre( double *Power_mW );
  bool __stdcall SetPowerIntoInputFibre( double Power_mW );
  bool __stdcall IsClassIVInterlockFlagActive( bool *Active );
  bool __stdcall ClearClassIVInterlockFlag();

private:
  IALC_REV_ILE* ALC_REV_ILE_;
};

#endif
