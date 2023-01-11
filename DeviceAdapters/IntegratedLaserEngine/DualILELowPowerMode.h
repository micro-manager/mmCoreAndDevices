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
class CDualILE;

class CDualILELowPowerMode
{
public:
  CDualILELowPowerMode( IALC_REV_ILEPowerManagement* Unit1PowerInterface, IALC_REV_ILEPowerManagement* Unit2PowerInterface, const CPortsConfiguration* PortsConfiguration, CDualILE* MMILE );
  ~CDualILELowPowerMode();

  int OnValueChange( MM::PropertyBase * Prop, MM::ActionType Act );
  typedef MM::Action<CDualILELowPowerMode> CPropertyAction;

  int UpdateILEInterface( IALC_REV_ILEPowerManagement* Unit1PowerInterface, IALC_REV_ILEPowerManagement* Unit2PowerInterface );
  void CheckAndUpdate();

private:
  IALC_REV_ILEPowerManagement* Unit1PowerInterface_;
  IALC_REV_ILEPowerManagement* Unit2PowerInterface_;
  const CPortsConfiguration* PortsConfiguration_;
  CDualILE* MMILE_;
  std::map<std::string, std::vector<int>> UnitsPropertyMap_;
  std::map<std::string, MM::PropertyBase *> PropertyPointers_;
  bool Unit1Active_;
  bool Unit2Active_;

  bool GetCachedValueForProperty( const std::string& PropertyName );
  int SetDevice( IALC_REV_ILEPowerManagement* PowerInterface, int UnitIndex, bool NewPosition, bool& UpdatedPosition );
};

#endif