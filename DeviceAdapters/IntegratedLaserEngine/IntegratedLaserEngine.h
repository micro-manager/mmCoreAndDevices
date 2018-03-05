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

  // Shutter API
  int SetOpen( bool Open = true );
  int GetOpen( bool& Open );
  int Fire( double DeltaT );

  // Helper functions
  void LogMMMessage( std::string Message, bool DebugOnly = false );
  MM::MMTime GetCurrentTime();

  void CheckAndUpdateLasers();

private:   
  IILEWrapperInterface* ILEWrapper_;
  IALC_REVObject3 *ILEDevice_;
  IILEWrapperInterface::TDeviceList DeviceList_;
  std::string DeviceName_;
  CPorts* Ports_;
  CActiveBlanking* ActiveBlanking_;
  CLowPowerMode* LowPowerMode_;
  CLasers* Lasers_;

  bool Initialized_;
  MM::MMTime ChangedTime_;

  CIntegratedLaserEngine& operator = ( CIntegratedLaserEngine& /*rhs*/ )
  {
    assert( false );
    return *this;
  }
};


#endif
