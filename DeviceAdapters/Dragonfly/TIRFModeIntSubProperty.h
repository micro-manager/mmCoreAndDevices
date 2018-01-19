///////////////////////////////////////////////////////////////////////////////
// FILE:          TIRFModeIntSubProperty.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _TIRFMODEINTSUBPROPERTY_H_
#define _TIRFMODEINTSUBPROPERTY_H_

#include "MMDeviceConstants.h"
#include "Property.h"
#include "ComponentInterface.h"

class CDragonfly;


class CIntDeviceWrapper
{
public:
  virtual bool GetLimits( int* Min, int* Max ) = 0;
  virtual bool Get( int* Value ) = 0;
  virtual bool Set( int Value ) = 0;
};

class CPenetrationWrapper : public CIntDeviceWrapper
{
public:
  CPenetrationWrapper( ITIRFInterface* TIRFInterface ) : TIRFInterface_( TIRFInterface ) {}
  bool GetLimits( int* Min, int* Max )
  {
    return TIRFInterface_->GetPenetrationLimit( Min, Max );
  }
  bool Get( int* Value )
  {
    return TIRFInterface_->GetPenetration_nm( Value );
  }
  bool Set( int Value )
  {
    return TIRFInterface_->SetPenetration_nm( Value );
  }
private:
  ITIRFInterface* TIRFInterface_;
};

class COffsetWrapper : public CIntDeviceWrapper
{
public:
  COffsetWrapper( ITIRFInterface* TIRFInterface ) : TIRFInterface_( TIRFInterface ) {}
  bool GetLimits( int* Min, int* Max )
  {
    return TIRFInterface_->GetOffsetLimit( Min, Max );
  }
  bool Get( int* Value )
  {
    return TIRFInterface_->GetOffset( Value );
  }
  bool Set( int Value )
  {
    return TIRFInterface_->SetOffset( Value );
  }
private:
  ITIRFInterface* TIRFInterface_;
};

class CTIRFModeIntSubProperty
{
public:
  CTIRFModeIntSubProperty( CIntDeviceWrapper* DeviceWrapper, CDragonfly* MMDragonfly, const std::string& PropertyName );
  ~CTIRFModeIntSubProperty();
  typedef MM::Action<CTIRFModeIntSubProperty> CPropertyAction;
  int OnChange( MM::PropertyBase * Prop, MM::ActionType Act );
  void SetReadOnly( bool ReadOnly );

private:
  CDragonfly* MMDragonfly_;
  CIntDeviceWrapper* DeviceWrapper_;
  std::string PropertyName_;
  MM::Property* MMProp_;

  bool SetPropertyValueFromDeviceValue( MM::PropertyBase* Prop );
};

#endif