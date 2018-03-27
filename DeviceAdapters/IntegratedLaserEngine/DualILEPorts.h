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
class CDualILE;

class CDualILEPorts
{
public:
  CDualILEPorts( IALC_REV_Port* DualPortInterface, IALC_REV_ILE2* ILE2Interface, CPortsConfiguration* PortsConfiguration, CDualILE* MMILE );
  ~CDualILEPorts();

  int OnPortChange( MM::PropertyBase * Prop, MM::ActionType Act );
  typedef MM::Action<CDualILEPorts> CPropertyAction;

  int UpdateILEInterface( IALC_REV_Port* DualPortInterface, IALC_REV_ILE2* ILE2Interface );

private:
  IALC_REV_Port* DualPortInterface_;
  IALC_REV_ILE2* ILE2Interface_;
  CPortsConfiguration* PortsConfiguration_;
  CDualILE* MMILE_;
  int NbPortsUnit1_;
  int NbPortsUnit2_;
  std::string CurrentPortName_;
  MM::PropertyBase* PropertyPointer_;

  int ChangePort( const std::string& PortName );
};

#endif