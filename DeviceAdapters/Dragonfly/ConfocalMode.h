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

//////////////////////////////////////////////////////////////////////////////
// IConfocalMode
//////////////////////////////////////////////////////////////////////////////

class IConfocalMode
{
public:
  virtual bool IsTIRFSelected() const = 0;
};

//////////////////////////////////////////////////////////////////////////////
// CConfocalMode
//////////////////////////////////////////////////////////////////////////////

class CConfocalMode : public IConfocalMode
{
public:
  CConfocalMode( IConfocalModeInterface3* ConfocalModeInterface, IConfocalModeInterface4* ConfocalModeInterface4, IIllLensInterface* IllLensInterface, IBorealisTIRFInterface* vBorealisTIRF60Interface, CDragonfly* MMDragonfly );
  ~CConfocalMode();

  int OnValueChange( MM::PropertyBase * Prop, MM::ActionType Act );
  typedef MM::Action<CConfocalMode> CPropertyAction;

  bool IsTIRFSelected() const override;

private:
  IConfocalModeInterface3* ConfocalModeInterface3_;
  IConfocalModeInterface4* ConfocalModeInterface4_;
  IIllLensInterface* IllLensInterface_;
  IBorealisTIRFInterface* BorealisTIRF60Interface_;
  CDragonfly* MMDragonfly_;
  std::string ConfocalHCName_;
  std::string ConfocalHSName_;
  struct TDevicePosition { TConfocalMode ConfocalMode; unsigned int PowerDensity; } ;
  typedef std::map<std::string, TDevicePosition> TPositionNameMap;
  TPositionNameMap PositionNameMap_;
  std::string ConfocalModePropertyName_;
  TConfocalMode CurrentConfocalMode_;

  int SetDeviceConfocalMode( TConfocalMode ConfocalMode );
  std::string BuildPropertyValueFromDeviceValue( const std::string& ConfocalModeBaseName, const std::string& PowerDensityName );
  void AddValuesForConfocalMode( TConfocalMode ConfocalMode, const std::string& ConfocalModeBaseName, const CPositionComponentHelper::TPositionNameMap& PowerDensityPositionNames );
  void AddValue( TConfocalMode ConfocalMode, const std::string& ConfocalModeBaseName, unsigned int PowerDensity, const std::string& PowerDensityName );
  int SetDeviceFromPropertyValue( const std::string& PropertValue );
  bool IsTIRFMode( TConfocalMode ConfocalMode ) const;
};

#endif