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
#include "ILEWrapperInterface.h"

#define ERR_PORTS_INIT 101
#define ERR_ACTIVEBLANKING_INIT 102
#define ERR_LOWPOWERMODE_INIT 103
#define ERR_LASERS_INIT 104

class IALC_REVObject3;
class IALC_REV_Laser2;
class CPorts;
class CActiveBlanking;
class CLowPowerMode;
class CLasers;

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

  void LogMMMessage( std::string Message, bool DebugOnly = false );
  MM::MMTime GetCurrentTime();

  void CheckAndUpdateLasers();

private:   
  IALC_REVObject3 *ILEDevice_;
  IALC_REV_Laser2 *LaserInterface_;
  IILEWrapperInterface::TDeviceList DeviceList_;
  std::string DeviceName_;
  CPorts* Ports_;
  CActiveBlanking* ActiveBlanking_;
  CLowPowerMode* LowPowerMode_;
  CLasers* Lasers_;


  /** Implementation instance shared with PiezoStage. */
  IILEWrapperInterface* ILEWrapper_;
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
  float PowerSetPoint_[10+1];  // 1-based arrays therefore +1
  bool IsLinear_[10+1];
  std::string Enable_[10+1];
  std::vector<std::string> EnableStates_[10+1];
  enum EXTERNALMODE
  {
    CW,
    TTL_PULSED
  };
  bool OpenRequest_;

  std::string BuildPropertyName( const std::string& BasePropertyName, int Wavelength );
  int Wavelength( const int LaserIndex );  // nano-meters
  int PowerFullScale( const int LaserIndex );  // Unitless - TODO Should be percentage IFF IsLinear_
  bool Ready( const int LaserIndex );
  float PowerReadback( const int LaserIndex );  // milli-Watts
  bool AllowsExternalTTL( const int LaserIndex );
  float PowerSetpoint( const int LaserIndex );  // milli-Watts
  void PowerSetpoint( const int LaserIndex, const float Value );  // milli-Watts
};


#endif
