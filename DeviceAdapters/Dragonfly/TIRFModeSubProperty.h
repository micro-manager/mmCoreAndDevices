///////////////////////////////////////////////////////////////////////////////
// FILE:          TIRFModeSubProperty.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _TIRFMODESUBPROPERTY_H_
#define _TIRFMODESUBPROPERTY_H_

#include "MMDeviceConstants.h"
#include "Property.h"
#include "ComponentInterface.h"

class CDragonfly;

class CDeviceWrapper
{
public:
  virtual bool GetLimits( int* Min, int* Max ) = 0;
  virtual bool Get( int* Value ) = 0;
  virtual bool Set( int Value ) = 0;
};

class CPenetrationWrapper : public CDeviceWrapper
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

class CObliqueAngleWrapper : public CDeviceWrapper
{
public:
  CObliqueAngleWrapper( ITIRFInterface* TIRFInterface ) : TIRFInterface_( TIRFInterface ) {}
  bool GetLimits( int* Min, int* Max )
  {
    return TIRFInterface_->GetObliqueAngleLimit( Min, Max );
  }
  bool Get( int* Value )
  {
    return TIRFInterface_->GetObliqueAngle_mdeg( Value );
  }
  bool Set( int Value )
  {
    return TIRFInterface_->SetObliqueAngle_mdeg( Value );
  }
private:
  ITIRFInterface* TIRFInterface_;
};

class COffsetWrapper : public CDeviceWrapper
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

class CTIRFModeSubProperty
{
public:
  CTIRFModeSubProperty( CDeviceWrapper* DeviceWrapper, CDragonfly* MMDragonfly, const std::string& PropertyName );
  ~CTIRFModeSubProperty();
  typedef MM::Action<CTIRFModeSubProperty> CPropertyAction;
  int OnChange( MM::PropertyBase * Prop, MM::ActionType Act );

private:
  CDragonfly* MMDragonfly_;
  CDeviceWrapper* DeviceWrapper_;
  std::string PropertyName_;

  bool SetPropertyValueFromDeviceValue( MM::PropertyBase* Prop );
};

#endif