///////////////////////////////////////////////////////////////////////////////
// FILE:          DualILELowPowerMode.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------

#ifndef _DUALILELOWPOWERMODE_H_
#define _DUALILELOWPOWERMODE_H_

#include "Property.h"
#include <map>

class IALC_REV_ILEPowerManagement;
class CPortsConfiguration;
class CIntegratedLaserEngine;

class CDualILELowPowerMode
{
public:
  CDualILELowPowerMode( IALC_REV_ILEPowerManagement* Unit1PowerInterface, IALC_REV_ILEPowerManagement* Unit2PowerInterface, const CPortsConfiguration* PortsConfiguration, CIntegratedLaserEngine* MMILE );
  ~CDualILELowPowerMode();

  int OnValueChange( MM::PropertyBase * Prop, MM::ActionType Act, long UnitsPropertyIndex );
  typedef MM::ActionEx<CDualILELowPowerMode> CPropertyActionEx;

  void UpdateILEInterface( IALC_REV_ILEPowerManagement* Unit1PowerInterface, IALC_REV_ILEPowerManagement* Unit2PowerInterface );

private:
  IALC_REV_ILEPowerManagement* Unit1PowerInterface_;
  IALC_REV_ILEPowerManagement* Unit2PowerInterface_;
  const CPortsConfiguration* PortsConfiguration_;
  CIntegratedLaserEngine* MMILE_;
  std::vector<std::string> PortNames_;
  std::vector<std::vector<int>> UnitsPropertyMap_;
  bool Unit1Active_;
  bool Unit2Active_;
};

#endif