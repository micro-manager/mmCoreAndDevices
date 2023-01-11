///////////////////////////////////////////////////////////////////////////////
// FILE:          CameraPortMirror.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _CAMERAPORTMIRROR_H_
#define _CAMERAPORTMIRROR_H_

#include "MMDeviceConstants.h"
#include "Property.h"

#include "PositionComponentInterface.h"

class CDragonflyStatus;
class CDragonfly;
class ICameraPortMirrorInterface;

class CCameraPortMirror : public IPositionComponentInterface
{
public:
  CCameraPortMirror( ICameraPortMirrorInterface* CameraPortMirrorInterface, const CDragonflyStatus* DragonflyStatus, CDragonfly* MMDragonfly );
  ~CCameraPortMirror();

protected:
  virtual bool GetPosition( unsigned int& Position );
  bool SetPosition( unsigned int Position );
  bool GetLimits( unsigned int& MinPosition, unsigned int& MaxPosition );
  IFilterSet* GetFilterSet();

private:
  ICameraPortMirrorInterface* CameraPortMirrorInterface_;
  CDragonfly* MMDragonfly_;
  const CDragonflyStatus* DragonflyStatus_;
  const std::string RFIDStatusProperty_;

  void CreateRFIDStatusProperty();
};

#endif