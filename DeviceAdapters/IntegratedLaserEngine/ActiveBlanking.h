///////////////////////////////////////////////////////////////////////////////
// FILE:          ActiveBlanking.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------

#ifndef _ACTIVEBLANKING_H_
#define _ACTIVEBLANKING_H_

#include "Property.h"
#include <map>

class IALC_REV_ILEActiveBlankingManagement;
class CIntegratedLaserEngine;

class CActiveBlanking
{
public:
  CActiveBlanking( IALC_REV_ILEActiveBlankingManagement* ActiveBlankingInterface, CIntegratedLaserEngine* MMILE );
  ~CActiveBlanking();

  int OnValueChange( MM::PropertyBase * Prop, MM::ActionType Act );
  typedef MM::Action<CActiveBlanking> CPropertyAction;

  int UpdateILEInterface( IALC_REV_ILEActiveBlankingManagement* ActiveBlankingInterface );

private:
  IALC_REV_ILEActiveBlankingManagement* ActiveBlankingInterface_;
  CIntegratedLaserEngine* MMILE_;
  std::map<std::string, int> PropertyLineIndexMap_;
  std::map<std::string, MM::PropertyBase *> PropertyPointers_;
  int EnabledPattern_;

  bool IsLineEnabled( int Line ) const;
  void ChangeLineState( int Line );
};

#endif