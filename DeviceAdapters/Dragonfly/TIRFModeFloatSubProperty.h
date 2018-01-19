///////////////////////////////////////////////////////////////////////////////
// FILE:          TIRFModeFloatSubProperty.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _TIRFMODEFLOATSUBPROPERTY_H_
#define _TIRFMODEFLOATSUBPROPERTY_H_

#include "MMDeviceConstants.h"
#include "Property.h"
#include "ComponentInterface.h"

class CDragonfly;


class CFloatDeviceWrapper
{
public:
  virtual bool GetLimits( float* Min, float* Max ) = 0;
  virtual bool Get( float* Value ) = 0;
  virtual bool Set( float Value ) = 0;
};

class CHILOObliqueAngleWrapper : public CFloatDeviceWrapper
{
public:
  CHILOObliqueAngleWrapper( ITIRFInterface* TIRFInterface ) : TIRFInterface_( TIRFInterface ) {}
  bool GetLimits( float* Min, float* Max )
  {
    int vMin, vMax;
    bool vRet = TIRFInterface_->GetObliqueAngleLimit( &vMin, &vMax );
    *Min = vMin / 1000.;
    *Max = vMax / 1000.;
    return vRet;
  }
  bool Get( float* Value )
  {
    int vValue;
    bool vRet = TIRFInterface_->GetObliqueAngle_mdeg( &vValue );
    *Value = vValue / 1000.;
    return vRet;
  }
  bool Set( float Value )
  {
    return TIRFInterface_->SetObliqueAngle_mdeg( Value*1000 );
  }
private:
  ITIRFInterface* TIRFInterface_;
};


class CTIRFModeFloatSubProperty
{
public:
  CTIRFModeFloatSubProperty( CFloatDeviceWrapper* DeviceWrapper, CDragonfly* MMDragonfly, const std::string& PropertyName );
  ~CTIRFModeFloatSubProperty();
  typedef MM::Action<CTIRFModeFloatSubProperty> CPropertyAction;
  int OnChange( MM::PropertyBase * Prop, MM::ActionType Act );
  void SetReadOnly( bool ReadOnly );

private:
  CDragonfly* MMDragonfly_;
  CFloatDeviceWrapper* DeviceWrapper_;
  std::string PropertyName_;
  MM::Property* MMProp_;

  bool SetPropertyValueFromDeviceValue( MM::PropertyBase* Prop );
};

#endif