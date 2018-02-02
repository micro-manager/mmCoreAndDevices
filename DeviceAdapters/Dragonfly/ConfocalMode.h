///////////////////////////////////////////////////////////////////////////////
// FILE:          ConfocalMode.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _CONFOCALMODE_H_
#define _CONFOCALMODE_H_

#include "Property.h"
#include "PositionComponentHelper.h"
#include "ComponentInterface.h"

class IConfocalModeInterface3;
class IIllLensInterface;
class CDragonfly;

class CConfocalMode
{
public:
  CConfocalMode( IConfocalModeInterface3* ConfocalModeInterface, IIllLensInterface* IllLensInterface, CDragonfly* MMDragonfly );
  ~CConfocalMode();

  int OnValueChange( MM::PropertyBase * Prop, MM::ActionType Act );
  typedef MM::Action<CConfocalMode> CPropertyAction;

private:
  IConfocalModeInterface3* ConfocalModeInterface_;
  IIllLensInterface* IllLensInterface_;
  CDragonfly* MMDragonfly_;
  std::string ConfocalHCName_;
  std::string ConfocalHSName_;
  struct TDevicePosition { TConfocalMode ConfocalMode; unsigned int PowerDensity; } ;
  typedef std::map<std::string, TDevicePosition> TPositionNameMap;
  TPositionNameMap PositionNameMap_;
  std::string ConfocalModePropertyName_;

  int SetDeviceConfocalMode( TConfocalMode ConfocalMode );
  std::string BuildPropertyValueFromDeviceValue( const std::string& ConfocalModeBaseName, const std::string& PowerDensityName );
  void AddValuesForConfocalMode( TConfocalMode ConfocalMode, const std::string& ConfocalModeBaseName, const CPositionComponentHelper::TPositionNameMap& PowerDensityPositionNames );
  void AddValue( TConfocalMode ConfocalMode, const std::string& ConfocalModeBaseName, unsigned int PowerDensity, const std::string& PowerDensityName );
  int SetDeviceFromPropertyValue( const std::string& PropertValue );
};

#endif