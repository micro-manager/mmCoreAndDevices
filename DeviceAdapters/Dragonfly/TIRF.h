///////////////////////////////////////////////////////////////////////////////
// FILE:          TIRF.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _TIRF_H_
#define _TIRF_H_

#include "MMDeviceConstants.h"
#include "Property.h"

class CDragonfly;
class ITIRFInterface;
class CTIRFModeIntSubProperty;
class CTIRFModeFloatSubProperty;
class CPenetrationWrapper;
class CHILOObliqueAngleWrapper;
class COffsetWrapper;
class IConfigFileHandler;

class CTIRF
{
public:
  CTIRF( ITIRFInterface* TIRFInterface, IConfigFileHandler* ConfigFileHandler, CDragonfly* MMDragonfly );
  ~CTIRF();
  typedef MM::Action<CTIRF> CPropertyAction;
  int OnTIRFModeChange( MM::PropertyBase * Prop, MM::ActionType Act );
  int OnOpticalPathwayChange( MM::PropertyBase * Prop, MM::ActionType Act );
  int OnBandwidthChange( MM::PropertyBase * Prop, MM::ActionType Act );

private:
  CDragonfly* MMDragonfly_;
  IConfigFileHandler* ConfigFileHandler_;
  ITIRFInterface* TIRFInterface_;
  int Magnification_, ScopeID_;
  double NumericalAperture_, RefractiveIndex_;
  CTIRFModeIntSubProperty* PenetrationProperty_;
  CPenetrationWrapper* PenetrationWrapper_;
  CTIRFModeFloatSubProperty* HILOObliqueAngleProperty_;
  CHILOObliqueAngleWrapper* HILOObliqueAngleWrapper_;
  CTIRFModeIntSubProperty* CriticalAngleOffsetProperty_;
  COffsetWrapper* CriticalAngleOffsetWrapper_;

  bool GetTIRFModeNameFromIndex( int TIRFModeIndex, const char** TIRFModeName );
  bool GetTIFRModeIndexFromName( const std::string& TIRFModeName, int* TIRFModeIndex );
  bool GetScopeNameFromIndex( int ScopeID, const char** ScopeName );
  bool GetScopeIndexFromName( const std::string& ScopeName, int* ScopeIndex );
  void UpdateTIRFModeSelection( const std::string& TIRFModeIndex );
};

#endif