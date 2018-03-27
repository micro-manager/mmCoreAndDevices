///////////////////////////////////////////////////////////////////////////////
// FILE:          DualILEActiveBlanking.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------

#ifndef _DUALILEACTIVEBLANKING_H_
#define _DUALILEACTIVEBLANKING_H_

#include "Property.h"
#include <map>

class IALC_REV_ILE4;
class CDualILE;
class CPortsConfiguration;

class CDualILEActiveBlanking
{
public:
  CDualILEActiveBlanking( IALC_REV_ILE4* DualActiveBlankingInterface, const CPortsConfiguration* PortsConfiguration, CDualILE* MMILE );
  ~CDualILEActiveBlanking();

  int OnValueChange( MM::PropertyBase * Prop, MM::ActionType Act );
  typedef MM::Action<CDualILEActiveBlanking> CPropertyAction;

  int UpdateILEInterface( IALC_REV_ILE4* DualActiveBlankingInterface );

private:
  IALC_REV_ILE4* DualActiveBlankingInterface_;
  const CPortsConfiguration* PortsConfiguration_;
  CDualILE* MMILE_;
  MM::PropertyBase* PropertyPointer_;
  int Unit1EnabledPattern_;
  int Unit2EnabledPattern_;
  bool Unit1ActiveBlankingPresent_;
  bool Unit2ActiveBlankingPresent_;
};

#endif