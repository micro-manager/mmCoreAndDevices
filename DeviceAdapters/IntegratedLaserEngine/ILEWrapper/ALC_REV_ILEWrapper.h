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
  bool SetFLICRThreshold( int Percent );
  bool GetFLICRThreshold( int *Percent );
  bool GetLaserWarmUpTime( int LaserIndex, int *Minutes );
  bool GetHours( double *Hours, double *Lifetime );
  bool GetLaserHours( int LaserIndex, double *Hours );
  bool GetAttenuationWheelDetails( int LaserIndex, int *AttenuationWheelPresent, int *StepsPerRotation, int *StepsToHome );
  bool GetPowerIntoInputFibre( double *Power_mW );
  bool SetPowerIntoInputFibre( double Power_mW );
  bool IsClassIVInterlockFlagActive( bool *Active );
  bool ClearClassIVInterlockFlag();

private:
  IALC_REV_ILE* ALC_REV_ILE_;
};

#endif
