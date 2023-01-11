///////////////////////////////////////////////////////////////////////////////
// FILE:          VeryLowPower.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------

#ifndef _VERYLOWPOWER_H_
#define _VERYLOWPOWER_H_

#include "Property.h"
#include <map>

class IALC_REV_ILEPowerManagement;
class CIntegratedLaserEngine;

class CVeryLowPower
{
public:
  CVeryLowPower( IALC_REV_ILEPowerManagement* PowerInterface, CIntegratedLaserEngine* MMILE );
  ~CVeryLowPower();

  int OnValueChange( MM::PropertyBase * Prop, MM::ActionType Act );
  typedef MM::Action<CVeryLowPower> CPropertyAction;

  int UpdateILEInterface( IALC_REV_ILEPowerManagement* PowerInterface );

private:
  IALC_REV_ILEPowerManagement* PowerInterface_;
  CIntegratedLaserEngine* MMILE_;
  bool VeryLowPowerActive_;
  MM::PropertyBase* PropertyPointer_;
};

#endif