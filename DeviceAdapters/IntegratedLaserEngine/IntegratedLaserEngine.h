///////////////////////////////////////////////////////////////////////////////
// FILE:          IntegratedLaserEngine.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   IntegratedLaserEngine controller adapter
//
// Based off the AndorLaserCombiner adapter from Karl Hoover, UCSF
//
//

#ifndef _INTEGRATEDLASERENGINE_H_
#define _INTEGRATEDLASERENGINE_H_

#include "../../MMDevice/MMDevice.h"
#include "../../MMDevice/DeviceBase.h"
#include "../../MMDevice/DeviceUtils.h"
#include <string>
#include <vector>
#include "ILEWrapper.h"
const int MaxLasers = 10;

//////////////////////////////////////////////////////////////////////////////
// Error codes
//


class CIntegratedLaserEngine : public CShutterBase<CIntegratedLaserEngine>
{
public:
  CIntegratedLaserEngine();
  ~CIntegratedLaserEngine();

  // MMDevice API
  int Initialize();
  int Shutdown();

  void GetName( char* Name ) const;
  bool Busy();

  // Action interface
  int OnDeviceChange( MM::PropertyBase* Prop, MM::ActionType Act );
  int OnPowerSetpoint( MM::PropertyBase* Prop, MM::ActionType Act, long LaserIndex );
  int OnPowerReadback( MM::PropertyBase* Prop, MM::ActionType Act, long LaserIndex );

  // Read-only properties
  int OnHours( MM::PropertyBase* Prop, MM::ActionType Act, long LaserIndex );
  int OnIsLinear( MM::PropertyBase* Prop, MM::ActionType Act, long LaserIndex );
  int OnMaximumLaserPower( MM::PropertyBase* Prop, MM::ActionType Act, long LaserIndex );
  int OnLaserState( MM::PropertyBase* Prop, MM::ActionType Act, long LaserIndex );
  int OnEnable( MM::PropertyBase* Prop, MM::ActionType Act, long LaserIndex );

  // Shutter API
  int SetOpen( bool Open = true );
  int GetOpen( bool& Open );
  int Fire( double DeltaT );

private:   
  IALC_REVObject3 *ILEDevice_;
  CILEWrapper::TDeviceList DeviceList_;
  std::string DeviceName_;

  /** Implementation instance shared with PiezoStage. */
  CILEWrapper* ILEWrapper_;
    // todo -- can move these to the implementation
  int HandleErrors();
  CIntegratedLaserEngine& operator = ( CIntegratedLaserEngine& /*rhs*/ )
  {
    assert( false );
    return *this;
  }

  void GenerateALCProperties();
  void GenerateReadOnlyIDProperties();

  int Error_;
  bool Initialized_;
  bool Busy_;
  MM::MMTime ChangedTime_;
  int NumberOfLasers_;
  float PowerSetPoint_[MaxLasers+1];  // 1-based arrays therefore +1
  bool IsLinear_[MaxLasers+1];
  std::string Enable_[MaxLasers+1];
  std::vector<std::string> EnableStates_[MaxLasers+1];
  enum EXTERNALMODE
  {
    CW,
    TTL_PULSED
  };
  bool OpenRequest_;
  unsigned char LaserPort_;  // First two bits of DOUT (0 or 1 or 2) IFF multiPortUnitPresent_

  std::string BuildPropertyName( const std::string& BasePropertyName, int Wavelength );
  int Wavelength( const int LaserIndex );  // nano-meters
  int PowerFullScale( const int LaserIndex );  // Unitless - TODO Should be percentage IFF IsLinear_
  bool Ready( const int LaserIndex );
  float PowerReadback( const int LaserIndex );  // milli-Watts
  bool AllowsExternalTTL( const int LaserIndex );
  float PowerSetpoint( const int LaserIndex );  // milli-Watts
  void PowerSetpoint( const int LaserIndex, const float Value );  // milli-Watts
};


#endif // _AndorLaserCombiner_H_
