///////////////////////////////////////////////////////////////////////////////
// FILE:          ConfocalMode.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _CONFOCALMODE_H_
#define _CONFOCALMODE_H_

#include "Property.h"

class IConfocalModeInterface3;
class CDragonfly;

class CConfocalMode
{
public:
  CConfocalMode( IConfocalModeInterface3* ConfocalModeInterface, CDragonfly* MMDragonfly );
  ~CConfocalMode();

  int OnValueChange( MM::PropertyBase * Prop, MM::ActionType Act );
  typedef MM::Action<CConfocalMode> CPropertyAction;

private:
  IConfocalModeInterface3* ConfocalModeInterface_;
  CDragonfly* MMDragonfly_;
  std::string ConfocalHCName_;
  std::string ConfocalHSName_;
};

#endif