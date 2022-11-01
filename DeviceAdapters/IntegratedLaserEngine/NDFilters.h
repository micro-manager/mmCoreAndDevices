///////////////////////////////////////////////////////////////////////////////
// FILE:          NDFilters.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------

#ifndef _NDFILTERS_H_
#define _NDFILTERS_H_

#include "Property.h"
#include <map>

class IALC_REV_ILEPowerManagement2;
class CIntegratedLaserEngine;

class CNDFilters
{
public:
  CNDFilters( IALC_REV_ILEPowerManagement2* PowerInterface, CIntegratedLaserEngine* MMILE );
  ~CNDFilters();

  int OnValueChange( MM::PropertyBase * Prop, MM::ActionType Act );
  typedef MM::Action<CNDFilters> CPropertyAction;

  int UpdateILEInterface( IALC_REV_ILEPowerManagement2* PowerInterface );
  void CheckAndUpdate();

private:
  IALC_REV_ILEPowerManagement2* PowerInterface_;
  CIntegratedLaserEngine* MMILE_;
  int CurrentFilterPosition_;
  std::vector<std::string> FilterPositions_;
  MM::PropertyBase* PropertyPointer_;

  int SetDevice( int Position );
};

#endif