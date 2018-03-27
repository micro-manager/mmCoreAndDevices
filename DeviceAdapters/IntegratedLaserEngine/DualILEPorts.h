///////////////////////////////////////////////////////////////////////////////
// FILE:          Ports.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------

#ifndef _DUALILEPORTS_H_
#define _DUALILEPORTS_H_

#include "Property.h"

class IALC_REV_Port;
class IALC_REV_ILE2;
class CPortsConfiguration;
class CIntegratedLaserEngine;

class CDualILEPorts
{
public:
  CDualILEPorts( IALC_REV_Port* DualPortInterface, IALC_REV_ILE2* ILE2Interface, CPortsConfiguration* PortsConfiguration, CIntegratedLaserEngine* MMILE );
  ~CDualILEPorts();

  int OnPortChange( MM::PropertyBase * Prop, MM::ActionType Act );
  typedef MM::Action<CDualILEPorts> CPropertyAction;

  void UpdateILEInterface( IALC_REV_Port* DualPortInterface, IALC_REV_ILE2* ILE2Interface );

private:
  IALC_REV_Port* DualPortInterface_;
  IALC_REV_ILE2* ILE2Interface_;
  CPortsConfiguration* PortsConfiguration_;
  CIntegratedLaserEngine* MMILE_;
  int NbPortsUnit1_;
  int NbPortsUnit2_;
  std::string CurrentPortName_;
  MM::PropertyBase* PropertyPointer_;

  int ChangePort( const std::string& PortName );
};

#endif