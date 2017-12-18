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
  IFilterSet* GetFilterSet();
  bool GetPositionOfFilterSetInRepository( unsigned int *Position );
  bool ExchangeFilterSet( unsigned int Position );
  IFilterRepository* GetFilterRepository();

private:
  IFilterConfigInterface* FilterConfigInterface_;
  CASDWrapperFilterSet* FilterSetWrapper_;
};
#endif