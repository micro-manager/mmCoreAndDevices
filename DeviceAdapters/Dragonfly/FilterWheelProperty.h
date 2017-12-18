///////////////////////////////////////////////////////////////////////////////
// FILE:          FilterWheelProperty.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _FILTERWHEELPROPERTY_H_
#define _FILTERWHEELPROPERTY_H_

#include "MMDeviceConstants.h"
#include "Property.h"

class IFilterWheelDeviceInterface;
class CDragonfly;

class CFilterWheelProperty
{
public:
  CFilterWheelProperty( IFilterWheelDeviceInterface* ASDInterface, CDragonfly* MMDragonfly, const std::string& PropertyName, const std::string& ComponentName );
  ~CFilterWheelProperty();

  int OnPositionChange( MM::PropertyBase * Prop, MM::ActionType Act );
  typedef MM::Action<CFilterWheelProperty> CPropertyAction;

private:
  IFilterWheelDeviceInterface* FilterWheelDevice_;
  CDragonfly* MMDragonfly_;
  const std::string PropertyName_;
  const std::string ComponentName_;

  typedef std::map<unsigned int, std::string> TPositionNameMap;
  TPositionNameMap PositionNames_;

  bool RetrievePositionsFromFilterConfig();
  bool RetrievePositionsWithoutDescriptions();
  bool SetPropertyValueFromDevicePosition( MM::PropertyBase* Prop );
};

#endif