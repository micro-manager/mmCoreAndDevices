///////////////////////////////////////////////////////////////////////////////
// FILE:          ASDWrapperFilterSet.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _ASDWRAPPERFILTERSET_H_
#define _ASDWRAPPERFILTERSET_H_

#include "ASDConfigInterface.h"

class CASDWrapperFilterSet : public IFilterSet
{
public:
  CASDWrapperFilterSet( IFilterSet* FilterSetInterface );
  ~CASDWrapperFilterSet();

  // IFilterSet
  bool __stdcall GetDescription( char *Description, unsigned int StringLength );
  bool __stdcall GetFilterDescription( unsigned int Position, char *Description, unsigned int StringLength );
  bool __stdcall GetLimits( unsigned int &MinPosition, unsigned int &MaxPosition );

private:
  IFilterSet* FilterSetInterface_;
};
#endif