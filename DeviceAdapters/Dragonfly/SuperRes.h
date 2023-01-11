///////////////////////////////////////////////////////////////////////////////
// FILE:          SuperRes.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _SUPERRES_H_
#define _SUPERRES_H_

#include "MMDeviceConstants.h"
#include "Property.h"

class CDragonfly;
class ISuperResInterface;

class CSuperRes
{
public:
  CSuperRes( ISuperResInterface* SuperResInterface, CDragonfly* MMDragonfly );
  ~CSuperRes();

  int OnPositionChange( MM::PropertyBase * Prop, MM::ActionType Act );
  typedef MM::Action<CSuperRes> CPropertyAction;

private:
  CDragonfly* MMDragonfly_;
  ISuperResInterface* SuperResInterface_;

  typedef std::map<unsigned int, std::string> TPositionNameMap;
  TPositionNameMap PositionNames_;

  bool RetrievePositions();
  int SetPropertyValueFromDevicePosition( MM::PropertyBase* Prop );
};

#endif