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

class IALC_REV_Laser2;
class IALC_REV_ILEPowerManagement;
class CIntegratedLaserEngine;
class CInterlockStatusMonitor;
//#define _ACTIVATE_DUMMYTEST_

// Laser state property values
const char* const g_LaserEnableOn = "On";
const char* const g_LaserEnableOff = "Off";
const char* const g_LaserEnableTTL = "External TTL";

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
  int OnKeyInterlock( MM::PropertyBase* Prop, MM::ActionType Act );
#endif
  int OnInterlockStatus( MM::PropertyBase* Prop, MM::ActionType Act );

  // Shutter API
  int SetOpen( bool Open = true );
  int GetOpen( bool& Open );
  
  void CheckAndUpdateLasers();
  int UpdateILEInterface( IALC_REV_Laser2 *LaserInterface, IALC_REV_ILEPowerManagement* PowerInterface, IALC_REV_ILE* ILEInterface );

private:
  struct TLaserRange
  {
    double PowerMin = 0;
    double PowerMax = 0;
  };

  struct CLaserState
  {
    float PowerSetPoint_ = 0;
    std::string Enable_ = g_LaserEnableOn;
    TLaserRange LaserRange_;
  };

  IALC_REV_Laser2 *LaserInterface_;
  IALC_REV_ILEPowerManagement* PowerInterface_;
  IALC_REV_ILE* ILEInterface_;
  CIntegratedLaserEngine* MMILE_;
  std::size_t NumberOfLasers_;
  std::vector<CLaserState> LasersState_;
  enum EXTERNALMODE
  {
    CW,
    TTL_PULSED
  };
  bool OpenRequest_;
  CInterlockStatusMonitor* InterlockStatusMonitor_;
#ifdef _ACTIVATE_DUMMYTEST_
  bool InterlockTEMP_;
  bool ClassIVInterlockTEMP_;
  bool KeyInterlockTEMP_;
#endif
  std::map<std::string, MM::PropertyBase *> PropertyPointers_;
  enum INTERLOCKSTATE
  {
    NO_INTERLOCK,
    INTERLOCK,
    KEY_INTERLOCK,
    CLASSIV_INTERLOCK
  };
  INTERLOCKSTATE DisplayedInterlockState_;

  void GenerateProperties();
  std::string BuildPropertyName( const std::string& BasePropertyName, int Wavelength );
  int Wavelength( const int LaserIndex );  // nano-meters
  bool AllowsExternalTTL( const int LaserIndex );
  float PowerSetpoint( const int LaserIndex );  // milli-Watts
  void PowerSetpoint( const int LaserIndex, const float Value );  // milli-Watts
  std::vector<int> GetLasersSortedByPower() const;
  void UpdateLasersRange();
  int CheckInterlock( int LaserIndex );
  bool IsKeyInterlockTriggered( int LaserIndex );
  bool IsInterlockTriggered( int LaserIndex );
  bool IsClassIVInterlockTriggered();
  void WaitOnLaserWarmingUp();
  void DisplayKeyInterlockMessage( MM::PropertyBase* Prop );
  void DisplayClassIVInterlockMessage( MM::PropertyBase* Prop );
  void DisplayInterlockMessage( MM::PropertyBase* Prop );
  void DisplayNoInterlockMessage( MM::PropertyBase* Prop );
  int ChangeDeviceShutterState( bool Open );
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
