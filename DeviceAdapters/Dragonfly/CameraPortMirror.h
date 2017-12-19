///////////////////////////////////////////////////////////////////////////////
// FILE:          CameraPortMirror.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _CAMERAPORTMIRROR_H_
#define _CAMERAPORTMIRROR_H_

#include "MMDeviceConstants.h"
#include "Property.h"

class CDragonfly;
class ICameraPortMirrorInterface;

class CCameraPortMirror
{
public:
  CCameraPortMirror( ICameraPortMirrorInterface* CameraPortMirrorInterface, CDragonfly* MMDragonfly );
  ~CCameraPortMirror();

  int OnPositionChange( MM::PropertyBase * Prop, MM::ActionType Act );
  typedef MM::Action<CCameraPortMirror> CPropertyAction;

private:
  ICameraPortMirrorInterface* CameraPortMirrorInterface_;
  CDragonfly* MMDragonfly_;

  typedef std::map<unsigned int, std::string> TPositionNameMap;
  TPositionNameMap PositionNames_;

  bool RetrievePositionsFromFilterSet();
  bool RetrievePositionsWithoutDescriptions();
  bool SetPropertyValueFromDevicePosition( MM::PropertyBase* Prop );
};

#endif