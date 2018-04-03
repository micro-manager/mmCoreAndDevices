///////////////////////////////////////////////////////////////////////////////
// FILE:          ASDWrapperFilterConfig.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _ASDWRAPPERFILTERCONFIG_H_
#define _ASDWRAPPERFILTERCONFIG_H_

#include "ASDConfigInterface.h"

class CASDWrapperFilterSet;

class CASDWrapperFilterConfig : public IFilterConfigInterface
{
public:
  CASDWrapperFilterConfig( IFilterConfigInterface* FilterConfigInterface );
  ~CASDWrapperFilterConfig();

  // IFilterConfigInterface
  IFilterSet* __stdcall GetFilterSet();
  bool __stdcall GetPositionOfFilterSetInRepository( unsigned int *Position );
  bool __stdcall ExchangeFilterSet( unsigned int Position );
  IFilterRepository* __stdcall GetFilterRepository();

private:
  IFilterConfigInterface* FilterConfigInterface_;
  CASDWrapperFilterSet* FilterSetWrapper_;
};
#endif