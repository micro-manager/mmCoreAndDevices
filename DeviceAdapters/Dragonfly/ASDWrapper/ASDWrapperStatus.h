///////////////////////////////////////////////////////////////////////////////
// FILE:          ASDWrapperStatus.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _ASDWRAPPERSTATUS_H_
#define _ASDWRAPPERSTATUS_H_

#include "ComponentInterface.h"

class CASDWrapperStatus : public IStatusInterface
{
public:
  CASDWrapperStatus( IStatusInterface* StatusInterface );
  ~CASDWrapperStatus();

  // IStatusInterface
  bool GetStatusCode( unsigned int *Status );
  bool IsStandbyActive();
  bool ActivateStandby();
  bool WakeFromStandby();

private:
  IStatusInterface* StatusInterface_;
};
#endif