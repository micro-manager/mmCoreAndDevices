///////////////////////////////////////////////////////////////////////////////
// FILE:          DichroicMirror.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _DICHROICMIRROR_H_
#define _DICHROICMIRROR_H_

#include "MMDeviceConstants.h"
#include "Property.h"

#include "FilterWheelDeviceInterface.h"

class IDichroicMirrorInterface;
class CDragonfly;
class CFilterWheelProperty;

class CDichroicMirror : public IFilterWheelDeviceInterface
{
public:
  CDichroicMirror( IDichroicMirrorInterface* DichroicMirrorInterface, CDragonfly* MMDragonfly );
  ~CDichroicMirror();

  bool GetPosition( unsigned int& Position );
  bool SetPosition( unsigned int Position );
  bool GetLimits( unsigned int& MinPosition, unsigned int& MaxPosition );
  IFilterConfigInterface* GetFilterConfigInterface();

private:
  IDichroicMirrorInterface* DichroicMirrorInterface_;
  CDragonfly* MMDragonfly_;
  CFilterWheelProperty* FilterWheelProperty_;
};

#endif