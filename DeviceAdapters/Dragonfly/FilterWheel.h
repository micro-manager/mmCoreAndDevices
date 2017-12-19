///////////////////////////////////////////////////////////////////////////////
// FILE:          FilterWheel.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _FILTERWHEEL_H_
#define _FILTERWHEEL_H_


#include "MMDeviceConstants.h"
#include "Property.h"

#include "ComponentInterface.h"
#include "FilterWheelDeviceInterface.h"

class CDragonflyStatus;
class CDragonfly;
class CFilterWheelProperty;

class CFilterWheel : public IFilterWheelDeviceInterface
{
public:
  CFilterWheel( TWheelIndex WheelIndex, IFilterWheelInterface* FilterWheelInterface, const CDragonflyStatus* DragonflyStatus, CDragonfly* MMDragonfly );
  ~CFilterWheel();

  int OnModeChange( MM::PropertyBase * Prop, MM::ActionType Act );
  typedef MM::Action<CFilterWheel> CPropertyAction;

  bool GetPosition( unsigned int& Position );
  bool SetPosition( unsigned int Position );
  bool GetLimits( unsigned int& MinPosition, unsigned int& MaxPosition );
  IFilterConfigInterface* GetFilterConfigInterface();

private:
  TWheelIndex WheelIndex_;
  IFilterWheelInterface* FilterWheelInterface_;
  IFilterWheelModeInterface* FilterWheelMode_;
  const CDragonflyStatus* DragonflyStatus_;
  CDragonfly* MMDragonfly_;
  CFilterWheelProperty* FilterWheelProperty_;

  const std::string ComponentName_;
  const std::string FilterModeProperty_;
  const std::string RFIDStatusProperty_;

  void CreateModeProperty();
  void CreateRFIDStatusProperty();
  const char* GetStringFromMode( TFilterWheelMode Mode ) const;
  TFilterWheelMode GetModeFromString( const std::string& ModeString ) const;
};

#endif