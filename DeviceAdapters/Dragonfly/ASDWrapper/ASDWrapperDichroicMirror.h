///////////////////////////////////////////////////////////////////////////////
// FILE:          ASDWrapperDichroicMirror.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _ASDWRAPPERDICHROICMIRROR_H_
#define _ASDWRAPPERDICHROICMIRROR_H_

#include "ASDInterface.h"

class CASDWrapperFilterConfig;

class CASDWrapperDichroicMirror : public IDichroicMirrorInterface
{
public:
  CASDWrapperDichroicMirror( IDichroicMirrorInterface* DichroicMirrorInterface );
  ~CASDWrapperDichroicMirror();

  // IDichroicMirrorInterface
  bool __stdcall GetPosition( unsigned int& Position );
  bool __stdcall SetPosition( unsigned int Position );
  bool __stdcall GetLimits( unsigned int& MinPosition, unsigned int& MaxPosition );
  IFilterConfigInterface* __stdcall GetFilterConfigInterface();

private:
  IDichroicMirrorInterface* DichroicMirrorInterface_;
  CASDWrapperFilterConfig* FilterConfigWrapper_;
};
#endif