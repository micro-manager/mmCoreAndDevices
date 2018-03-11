///////////////////////////////////////////////////////////////////////////////
// FILE:          Ports.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------

#ifndef _DUALILEPORTS_H_
#define _DUALILEPORTS_H_

#include "Property.h"

class IALC_REV_Port;
class CPortsConfiguration;
class CIntegratedLaserEngine;

class CDualILEPorts
{
public:
  CDualILEPorts( IALC_REV_Port* Unit1PortInterface, IALC_REV_Port* Unit2PortInterface, CPortsConfiguration* PortsConfiguration, CIntegratedLaserEngine* MMILE );
  ~CDualILEPorts();

  int OnPortChange( MM::PropertyBase * Prop, MM::ActionType Act );
  typedef MM::Action<CDualILEPorts> CPropertyAction;

  void UpdateILEInterface( IALC_REV_Port* Unit1PortInterface, IALC_REV_Port* Unit2PortInterface );

private:
  IALC_REV_Port* Unit1PortInterface_;
  IALC_REV_Port* Unit2PortInterface_;
  CPortsConfiguration* PortsConfiguration_;
  CIntegratedLaserEngine* MMILE_;
  int NbPorts_;
};

#endif