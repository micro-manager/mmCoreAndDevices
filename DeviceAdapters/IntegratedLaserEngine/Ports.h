///////////////////////////////////////////////////////////////////////////////
// FILE:          Ports.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------

#ifndef _PORTS_H_
#define _PORTS_H_

#include "Property.h"

class IALC_REV_Port;
class CIntegratedLaserEngine;

class CPorts
{
public:
  CPorts( IALC_REV_Port* PortInterface, CIntegratedLaserEngine* MMILE );
  ~CPorts();

  static char PortIndexToName( int PortIndex );

  int OnPortChange( MM::PropertyBase * Prop, MM::ActionType Act );
  typedef MM::Action<CPorts> CPropertyAction;

private:
  IALC_REV_Port* PortInterface_;
  int NbPorts_;
  CIntegratedLaserEngine* MMILE_;
};

#endif