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
#include "../../MMDevice/DeviceThreads.h"

const int MaxLasers = 10;

class IALC_REV_Laser2;
class IALC_REV_ILEPowerManagement;
class CIntegratedLaserEngine;
class CInterlockStatusMonitor;
//#define _ACTIVATE_DUMMYTEST_

class CLasers
{
public:
  CLasers( IALC_REV_Laser2 *LaserInterface, IALC_REV_ILEPowerManagement* PowerInterface, IALC_REV_ILE* ILEInterface, CIntegratedLaserEngine* MMILE );
  ~CLasers();

  // Actions
  typedef MM::ActionEx<CLasers> CPropertyActionEx;
  typedef MM::Action<CLasers> CPropertyAction;
  int OnPowerSetpoint( MM::PropertyBase* Prop, MM::ActionType Act, long LaserIndex );
  int OnEnable( MM::PropertyBase* Prop, MM::ActionType Act, long LaserIndex );
#ifdef _ACTIVATE_DUMMYTEST_
  int OnInterlock( MM::PropertyBase* Prop, MM::ActionType Act );
  int OnClassIVInterlock( MM::PropertyBase* Prop, MM::ActionType Act );
#endif
  int OnInterlockStatus( MM::PropertyBase* Prop, MM::ActionType Act );

  // Shutter API
  int SetOpen( bool Open = true );
  int GetOpen( bool& Open );
  
  void CheckAndUpdateLasers();
  void UpdateILEInterface( IALC_REV_Laser2 *LaserInterface, IALC_REV_ILEPowerManagement* PowerInterface, IALC_REV_ILE* ILEInterface );

private:   
  IALC_REV_Laser2 *LaserInterface_;
  IALC_REV_ILEPowerManagement* PowerInterface_;
  IALC_REV_ILE* ILEInterface_;
  CIntegratedLaserEngine* MMILE_;
  int NumberOfLasers_;
  float PowerSetPoint_[MaxLasers + 1];  // 1-based arrays therefore +1
  std::string Enable_[MaxLasers + 1];
  std::vector<std::string> EnableStates_[MaxLasers + 1];
  struct TLaserRange
  {
    double PowerMin; 
    double PowerMax;
  };
  TLaserRange LaserRange_[MaxLasers + 1];
  enum EXTERNALMODE
  {
    CW,
    TTL_PULSED
  };
  bool OpenRequest_;
  bool Interlock_;
  bool ClassIVInterlock_;
  CInterlockStatusMonitor* InterlockStatusMonitor_;
#ifdef _ACTIVATE_DUMMYTEST_
  bool InterlockTEMP_;
  bool ClassIVInterlockTEMP_;
#endif

  void GenerateProperties();
  std::string BuildPropertyName( const std::string& BasePropertyName, int Wavelength );
  int Wavelength( const int LaserIndex );  // nano-meters
  bool AllowsExternalTTL( const int LaserIndex );
  float PowerSetpoint( const int LaserIndex );  // milli-Watts
  void PowerSetpoint( const int LaserIndex, const float Value );  // milli-Watts
  void UpdateLasersRange();
  bool IsInterlockTriggered( int LaserIndex );
  bool IsClassIVInterlockTriggered();
};


class CInterlockStatusMonitor : public MMDeviceThreadBase
{
public:
  CInterlockStatusMonitor( CIntegratedLaserEngine* MMILE );
  virtual ~CInterlockStatusMonitor();

  int svc();

private:
  CIntegratedLaserEngine* MMILE_;
  bool KeepRunning_;
};
#endif
