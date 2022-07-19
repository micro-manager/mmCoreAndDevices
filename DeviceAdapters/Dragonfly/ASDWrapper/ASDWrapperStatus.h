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
  bool __stdcall GetStatusCode( unsigned int *Status );
  bool __stdcall IsStandbyActive();
  bool __stdcall ActivateStandby();
  bool __stdcall WakeFromStandby();

private:
  IStatusInterface* StatusInterface_;
};
#endif