///////////////////////////////////////////////////////////////////////////////
// FILE:          DragonflyStatus.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _DRAGONFLYSTATUS_H_
#define _DRAGONFLYSTATUS_H_

#include "ComponentInterface.h"

class IStatusInterface;
class CDragonfly;

class CDragonflyStatus
{
public:
  CDragonflyStatus( IStatusInterface* StatusInterface, CDragonfly* MMDragonfly );
  ~CDragonflyStatus();

  bool IsRFIDPresentForWheel( TWheelIndex WheelIndex ) const;
  bool IsRFIDReadForWheel( TWheelIndex WheelIndex ) const;
  bool IsRFIDPresentForCameraPortMirror() const;
  bool IsRFIDReadForCameraPortMirror() const;
  bool IsExternalDiskSync() const;

private:
  IStatusInterface* StatusInterface_;
  CDragonfly* MMDragonfly_;
};

#endif