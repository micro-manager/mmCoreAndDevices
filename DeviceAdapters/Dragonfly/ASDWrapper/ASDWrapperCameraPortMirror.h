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
  bool GetPosition( unsigned int& Position );
  bool SetPosition( unsigned int Position );
  bool GetLimits( unsigned int& MinPosition, unsigned int& MaxPosition );
  bool IsSplitFieldMirrorPresent();
  IFilterSet* GetCameraPortMirrorConfigInterface();

private:
  ICameraPortMirrorInterface* CameraPortMirrorInterface_;
  CASDWrapperFilterSet* FilterSetWrapper_;
};
#endif