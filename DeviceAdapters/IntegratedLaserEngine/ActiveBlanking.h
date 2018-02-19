///////////////////////////////////////////////////////////////////////////////
// FILE:          Ports.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------

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

private:
  IALC_REV_ILEActiveBlankingManagement* ActiveBlankingInterface_;
  CIntegratedLaserEngine* MMILE_;
  std::map<std::string, int> PropertyLineIndexMap_;
  int EnabledPattern_;

  bool IsLineEnabled( int EnabledPattern, int Line ) const;
  void UpdateEnabledPattern( int Line, bool Enabled );
};