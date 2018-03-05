///////////////////////////////////////////////////////////////////////////////
// FILE:          Lasers.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   IntegratedLaserEngine controller adapter
//
// Based off the AndorLaserCombiner adapter from Karl Hoover, UCSF
//
//

#ifndef _LASERS_H_
#define _LASERS_H_

#include "Property.h"
const int MaxLasers = 10;

class IALC_REV_Laser2;
class CIntegratedLaserEngine;

class CLasers
{
public:
  CLasers( IALC_REV_Laser2 *LaserInterface, CIntegratedLaserEngine* MMILE );
  ~CLasers();

  // Actions
  typedef MM::ActionEx<CLasers> CPropertyActionEx;
  typedef MM::Action<CLasers> CPropertyAction;
  int OnPowerSetpoint( MM::PropertyBase* Prop, MM::ActionType Act, long LaserIndex );
  int OnEnable( MM::PropertyBase* Prop, MM::ActionType Act, long LaserIndex );

  // Shutter API
  int SetOpen( bool Open = true );
  int GetOpen( bool& Open );
  
  void CheckAndUpdateLasers();

private:   
  IALC_REV_Laser2 *LaserInterface_;
  CIntegratedLaserEngine* MMILE_;
  int NumberOfLasers_;
  float PowerSetPoint_[MaxLasers+1];  // 1-based arrays therefore +1
  std::string Enable_[MaxLasers+1];
  std::vector<std::string> EnableStates_[MaxLasers+1];
  enum EXTERNALMODE
  {
    CW,
    TTL_PULSED
  };
  bool OpenRequest_;

  void GenerateALCProperties();
  std::string BuildPropertyName( const std::string& BasePropertyName, int Wavelength );
  int Wavelength( const int LaserIndex );  // nano-meters
  bool AllowsExternalTTL( const int LaserIndex );
  float PowerSetpoint( const int LaserIndex );  // milli-Watts
  void PowerSetpoint( const int LaserIndex, const float Value );  // milli-Watts
};


#endif
