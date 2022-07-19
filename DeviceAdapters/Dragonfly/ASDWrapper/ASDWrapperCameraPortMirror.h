///////////////////////////////////////////////////////////////////////////////
// FILE:          ASDWrapperCameraPortMirror.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _ASDWRAPPERCAMERAPORTMIRROR_H_
#define _ASDWRAPPERCAMERAPORTMIRROR_H_

#include "ASDInterface.h"

class CASDWrapperFilterSet;

class CASDWrapperCameraPortMirror : public ICameraPortMirrorInterface
{
public:
  CASDWrapperCameraPortMirror( ICameraPortMirrorInterface* CameraPortMirrorInterface );
  ~CASDWrapperCameraPortMirror();

  // ICameraPortMirrorInterface
  bool __stdcall GetPosition( unsigned int& Position );
  bool __stdcall SetPosition( unsigned int Position );
  bool __stdcall GetLimits( unsigned int& MinPosition, unsigned int& MaxPosition );
  bool __stdcall IsSplitFieldMirrorPresent();
  IFilterSet* __stdcall GetCameraPortMirrorConfigInterface();

private:
  ICameraPortMirrorInterface* CameraPortMirrorInterface_;
  CASDWrapperFilterSet* FilterSetWrapper_;
};
#endif