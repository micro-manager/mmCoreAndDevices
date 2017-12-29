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
  bool GetPosition( unsigned int& Position );
  bool SetPosition( unsigned int Position );
  bool GetLimits( unsigned int& MinPosition, unsigned int& MaxPosition );
  bool IsRestrictionEnabled();
  bool GetRestrictedRange( unsigned int &MinPosition, unsigned int &MaxPosition );
  bool RegisterForNotificationOnRangeRestriction( INotify *Notify );
  IFilterSet* GetLensConfigInterface();

private:
  IIllLensInterface* IllLensInterface_;
  CASDWrapperFilterSet* FilterSetWrapper_;
};
#endif