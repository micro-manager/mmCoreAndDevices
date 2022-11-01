///////////////////////////////////////////////////////////////////////////////
// FILE:          BTIRF.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _BTIRF_H_
#define _BTIRF_H_

#include "MMDeviceConstants.h"
#include "Property.h"

#include <memory>

class CDragonfly;
class IASDInterface4;
class IBorealisTIRFInterface;
class CBTIRFCriticalAngleProperty;

class CBTIRF
{
public:
  CBTIRF( IASDInterface4* ASDInterface4, CDragonfly* MMDragonfly );
  ~CBTIRF();

private:
  CDragonfly* MMDragonfly_;
  std::unique_ptr<CBTIRFCriticalAngleProperty> BTIRF60CriticalAngle_;
  std::unique_ptr<CBTIRFCriticalAngleProperty> BTIRF100CriticalAngle_;
};


class CBTIRFCriticalAngleProperty
{
public:
  CBTIRFCriticalAngleProperty( IBorealisTIRFInterface* BTIRFInterface, CDragonfly* MMDragonfly );
  ~CBTIRFCriticalAngleProperty();
  typedef MM::Action<CBTIRFCriticalAngleProperty> CPropertyAction;
  int OnChange( MM::PropertyBase * Prop, MM::ActionType Act );

private:
  CDragonfly* MMDragonfly_;
  IBorealisTIRFInterface* BTIRFInterface_;
  std::string PropertyName_;
};
#endif