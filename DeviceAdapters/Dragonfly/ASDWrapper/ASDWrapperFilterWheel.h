///////////////////////////////////////////////////////////////////////////////
// FILE:          ASDWrapperFilterWheel.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _ASDWRAPPERFILTERWHEEL_H_
#define _ASDWRAPPERFILTERWHEEL_H_

#include "ASDInterface.h"

class CASDWrapperFilterConfig;
class CASDWrapperFilterWheelMode;

class CASDWrapperFilterWheel : public IFilterWheelInterface
{
public:
  CASDWrapperFilterWheel( IFilterWheelInterface* FilterWheelInterface );
  ~CASDWrapperFilterWheel();

  // IDichroicMirrorInterface
  bool GetPosition( unsigned int& Position );
  bool SetPosition( unsigned int Position );
  bool GetLimits( unsigned int& MinPosition, unsigned int& MaxPosition );
  IFilterWheelSpeedInterface* GetFilterWheelSpeedInterface();
  IFilterConfigInterface* GetFilterConfigInterface();
  IFilterWheelModeInterface* GetFilterWheelModeInterface();

private:
  IFilterWheelInterface* FilterWheelInterface_;
  CASDWrapperFilterConfig* FilterConfigWrapper_;
  CASDWrapperFilterWheelMode* FilterWheelModeWrapper_;
};
#endif