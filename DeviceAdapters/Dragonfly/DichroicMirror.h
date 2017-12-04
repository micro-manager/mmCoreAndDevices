///////////////////////////////////////////////////////////////////////////////
// FILE:          DichroicMirror.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _DICHROICMIRROR_H_
#define _DICHROICMIRROR_H_

#include "MMDeviceConstants.h"
#include "Property.h"

class IDichroicMirrorInterface;
class CDragonfly;

class CDichroicMirror
{
public:
  CDichroicMirror( IDichroicMirrorInterface* DichroicMirrorInterface, CDragonfly* MMDragonfly );
  ~CDichroicMirror();

  int OnPositionChange( MM::PropertyBase * Prop, MM::ActionType Act );
  typedef MM::Action<CDichroicMirror> CPropertyAction;

private:
  IDichroicMirrorInterface* DichroicMirrorInterface_;
  CDragonfly* MMDragonfly_;

  typedef std::map<unsigned int, std::string> TPositionNameMap;
  TPositionNameMap PositionNames_;

  bool RetrievePositionsFromFilterConfig();
  bool RetrievePositionsWithoutDescriptions();
  bool SetPropertyValueFromDevicePosition( MM::PropertyBase* Prop );
};

#endif