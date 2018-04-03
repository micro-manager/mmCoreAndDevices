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

  // IFilterWheelInterface
  bool __stdcall GetPosition( unsigned int& Position );
  bool __stdcall SetPosition( unsigned int Position );
  bool __stdcall GetLimits( unsigned int& MinPosition, unsigned int& MaxPosition );
  IFilterWheelSpeedInterface* __stdcall GetFilterWheelSpeedInterface();
  IFilterConfigInterface* __stdcall GetFilterConfigInterface();
  IFilterWheelModeInterface* __stdcall GetFilterWheelModeInterface();

private:
  IFilterWheelInterface* FilterWheelInterface_;
  CASDWrapperFilterConfig* FilterConfigWrapper_;
  CASDWrapperFilterWheelMode* FilterWheelModeWrapper_;
};
#endif