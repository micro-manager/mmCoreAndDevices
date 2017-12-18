///////////////////////////////////////////////////////////////////////////////
// FILE:          ASDWrapperFilterWheelMode.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _ASDWRAPPERFILTERWHEELMODE_H_
#define _ASDWRAPPERFILTERWHEELMODE_H_

#include "ASDInterface.h"


class CASDWrapperFilterWheelMode : public IFilterWheelModeInterface
{
public:
  CASDWrapperFilterWheelMode( IFilterWheelModeInterface* FilterWheelModeInterface );
  ~CASDWrapperFilterWheelMode();

  // IDichroicMirrorInterface
  bool GetMode( TFilterWheelMode& FilterWheelMode );
  bool SetMode( TFilterWheelMode FilterWheelMode );

private:
  IFilterWheelModeInterface* FilterWheelModeInterface_;
};
#endif