///////////////////////////////////////////////////////////////////////////////
// FILE:          PositionComponentInterface.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _POSITIONCOMPONENTINTERFACE_H_
#define _POSITIONCOMPONENTINTERFACE_H_

#include "MMDeviceConstants.h"
#include "Property.h"
#include "PositionComponentHelper.h"

class CDragonfly;
class IFilterSet;

class IPositionComponentInterface
{
public:
  IPositionComponentInterface( CDragonfly* MMDragonfly, const std::string& PropertyName, bool AddIndexToPositionNames );
  ~IPositionComponentInterface();

  int OnPositionChange( MM::PropertyBase * Prop, MM::ActionType Act );
  typedef MM::Action<IPositionComponentInterface> CPropertyAction;

protected:
  typedef CPositionComponentHelper::TPositionNameMap TPositionNameMap;
  TPositionNameMap PositionNames_;
  std::string PropertyName_;
  CDragonfly* MMDragonfly_;

  void Initialise();

  virtual bool GetPosition( unsigned int& Position ) = 0;
  virtual bool SetPosition( unsigned int Position ) = 0;
  virtual bool GetLimits( unsigned int& MinPosition, unsigned int&MaxPosition ) = 0;
  virtual IFilterSet* GetFilterSet() = 0;
  
  virtual bool UpdateAllowedValues() { return false; } // return true if allowed values have been updated

  const TPositionNameMap& GetPositionNameMap() const { return PositionNames_; }

private:
  bool Initialised_;
  bool AddIndexToPositionNames_;

  int SetPropertyValueFromDevicePosition( MM::PropertyBase* Prop );
};

#endif