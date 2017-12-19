///////////////////////////////////////////////////////////////////////////////
// FILE:          Aperture.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _APERTURE_H_
#define _APERTURE_H_

#include "MMDeviceConstants.h"
#include "Property.h"

class CDragonfly;
class IApertureInterface;

class CAperture
{
public:
  CAperture( IApertureInterface* ApertureInterface, CDragonfly* MMDragonfly );
  ~CAperture();

  int OnPositionChange( MM::PropertyBase * Prop, MM::ActionType Act );
  typedef MM::Action<CAperture> CPropertyAction;

private:
  IApertureInterface* ApertureInterface_;
  CDragonfly* MMDragonfly_;

  typedef std::map<unsigned int, std::string> TPositionNameMap;
  TPositionNameMap PositionNames_;

  bool RetrievePositionsFromFilterSet();
  bool RetrievePositionsWithoutDescriptions();
  bool SetPropertyValueFromDevicePosition( MM::PropertyBase* Prop );
};

#endif