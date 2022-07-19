///////////////////////////////////////////////////////////////////////////////
// FILE:          LowPowerMode.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------

#ifndef _LOWPOWERMODE_H_
#define _LOWPOWERMODE_H_

#include "Property.h"
#include <map>

class IALC_REV_ILEPowerManagement;
class CIntegratedLaserEngine;

class CLowPowerMode
{
public:
  CLowPowerMode( IALC_REV_ILEPowerManagement* PowerInterface, CIntegratedLaserEngine* MMILE );
  ~CLowPowerMode();

  int OnValueChange( MM::PropertyBase * Prop, MM::ActionType Act );
  typedef MM::Action<CLowPowerMode> CPropertyAction;

  int UpdateILEInterface( IALC_REV_ILEPowerManagement* PowerInterface );

private:
  IALC_REV_ILEPowerManagement* PowerInterface_;
  CIntegratedLaserEngine* MMILE_;
  bool LowPowerModeActive_;
  MM::PropertyBase* PropertyPointer_;
};

#endif