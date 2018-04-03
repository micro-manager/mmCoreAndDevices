///////////////////////////////////////////////////////////////////////////////
// FILE:          ASDWrapperIllLens.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _ASDWRAPPERILLLENS_H_
#define _ASDWRAPPERILLLENS_H_

#include "ASDInterface.h"

class CASDWrapperFilterSet;

class CASDWrapperIllLens : public IIllLensInterface
{
public:
  CASDWrapperIllLens( IIllLensInterface* IllLensInterface );
  ~CASDWrapperIllLens();

  // IIllLensInterface
  bool __stdcall GetPosition( unsigned int& Position );
  bool __stdcall SetPosition( unsigned int Position );
  bool __stdcall GetLimits( unsigned int& MinPosition, unsigned int& MaxPosition );
  bool __stdcall IsRestrictionEnabled();
  bool __stdcall GetRestrictedRange( unsigned int &MinPosition, unsigned int &MaxPosition );
  bool __stdcall RegisterForNotificationOnRangeRestriction( INotify *Notify );
  IFilterSet* __stdcall GetLensConfigInterface();

private:
  IIllLensInterface* IllLensInterface_;
  CASDWrapperFilterSet* FilterSetWrapper_;
};
#endif