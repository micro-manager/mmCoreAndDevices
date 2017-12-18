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
  bool GetPosition( unsigned int& Position );
  bool SetPosition( unsigned int Position );
  bool GetLimits( unsigned int& MinPosition, unsigned int& MaxPosition );
  IFilterConfigInterface* GetFilterConfigInterface();

private:
  IDichroicMirrorInterface* DichroicMirrorInterface_;
  CASDWrapperFilterConfig* FilterConfigWrapper_;
};
#endif