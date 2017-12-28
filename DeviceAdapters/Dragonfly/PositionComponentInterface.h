///////////////////////////////////////////////////////////////////////////////
// FILE:          PositionComponent.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _POSITIONCOMPONENTINTERFACE_H_
#define _POSITIONCOMPONENTINTERFACE_H_

#include "MMDeviceConstants.h"
#include "Property.h"

class CDragonfly;
class IFilterSet;

class IPositionComponentInterface
{
public:
  IPositionComponentInterface( CDragonfly* MMDragonfly, const std::string& PropertyName );
  ~IPositionComponentInterface();

  int OnPositionChange( MM::PropertyBase * Prop, MM::ActionType Act );
  typedef MM::Action<IPositionComponentInterface> CPropertyAction;

protected:
  std::string PropertyName_;
  
  void Initialise();

  virtual bool GetPosition( unsigned int& Position ) = 0;
  virtual bool SetPosition( unsigned int Position ) = 0;
  virtual bool GetLimits( unsigned int& MinPosition, unsigned int&MaxPosition ) = 0;
  virtual IFilterSet* GetFilterSet() = 0;

private:
  CDragonfly* MMDragonfly_;
  bool Initialised_;

  typedef std::map<unsigned int, std::string> TPositionNameMap;
  TPositionNameMap PositionNames_;

  bool RetrievePositionsFromFilterSet();
  bool RetrievePositionsWithoutDescriptions();
  bool SetPropertyValueFromDevicePosition( MM::PropertyBase* Prop );
};

#endif