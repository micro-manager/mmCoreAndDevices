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

  /**
  * Transform 1-based port index to port name character
  */
  static char PortIndexToName( int PortIndex );
  /**
  * Transform port name character to 1-based port index
  */
  static int PortNameToIndex( char PortName );

  int OnPortChange( MM::PropertyBase * Prop, MM::ActionType Act );
  typedef MM::Action<CPorts> CPropertyAction;

  int UpdateILEInterface( IALC_REV_Port* PortInterface );

private:
  IALC_REV_Port* PortInterface_;
  int NbPorts_;
  CIntegratedLaserEngine* MMILE_;
  int CurrentPortIndex_;
  MM::PropertyBase* PropertyPointer_;
};

#endif