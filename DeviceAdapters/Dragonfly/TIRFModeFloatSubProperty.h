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
#include "TIRFMode.h"

class CDragonfly;
class IConfigFileHandler;


class CFloatDeviceWrapper
{
public:
  virtual bool GetLimits( float* Min, float* Max ) = 0;
  virtual bool Get( float* Value ) = 0;
  virtual bool Set( float Value ) = 0;
  virtual ETIRFMode Mode() const = 0;
};

class CHILOObliqueAngleWrapper : public CFloatDeviceWrapper
{
public:
  CHILOObliqueAngleWrapper( ITIRFInterface* TIRFInterface ) : TIRFInterface_( TIRFInterface ) {}
  bool GetLimits( float* Min, float* Max )
  {
    int vMin, vMax;
    bool vRet = TIRFInterface_->GetObliqueAngleLimit( &vMin, &vMax );
    *Min =  vMin / 1000.f;
    *Max =  vMax / 1000.f;
    return vRet;
  }
  bool Get( float* Value )
  {
    int vValue;
    bool vRet = TIRFInterface_->GetObliqueAngle_mdeg( &vValue );
    *Value = vValue / 1000.f;
    return vRet;
  }
  bool Set( float Value )
  {
    return TIRFInterface_->SetObliqueAngle_mdeg( (int)(Value*1000) );
  }
  ETIRFMode Mode() const { return HiLoObliqueAngle; }
private:
  ITIRFInterface* TIRFInterface_;
};


class CTIRFModeFloatSubProperty
{
public:
  CTIRFModeFloatSubProperty( CFloatDeviceWrapper* DeviceWrapper, IConfigFileHandler* ConfigFileHandler, CDragonfly* MMDragonfly, const std::string& PropertyName );
  ~CTIRFModeFloatSubProperty();
  typedef MM::Action<CTIRFModeFloatSubProperty> CPropertyAction;
  int OnChange( MM::PropertyBase * Prop, MM::ActionType Act );
  void ModeSelected( ETIRFMode SelectedTIRFMode );
  bool IsModeSelected();

private:
  CDragonfly* MMDragonfly_;
  IConfigFileHandler* ConfigFileHandler_;
  CFloatDeviceWrapper* DeviceWrapper_;
  std::string PropertyName_;
  MM::Property* MMProp_;
  float BufferedUserSelectionValue_;
  ETIRFMode SelectedTIRFMode_;

  int SetDeviceValue( MM::PropertyBase* Prop, float RequestedValue );
  int SetPropertyValueFromDeviceValue( MM::PropertyBase* Prop );
  void SetPropertyValue( MM::PropertyBase* Prop, double NewValue );
  void SetBufferedUserSelectionValue( float NewValue );
};

#endif