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
#include "TIRFMode.h"
#include "Dragonfly.h"

class CDragonfly;
class IConfigFileHandler;


class CIntDeviceWrapper
{
public:
  virtual bool GetLimits( int* Min, int* Max ) = 0;
  virtual bool Get( int* Value ) = 0;
  virtual bool Set( int Value ) = 0;
  virtual ETIRFMode Mode() const = 0;
};

class CPenetrationWrapper : public CIntDeviceWrapper
{
public:
  CPenetrationWrapper( ITIRFInterface* TIRFInterface, CDragonfly* MMDragonfly ) : TIRFInterface_( TIRFInterface ), MMDragonfly_( MMDragonfly ) {}
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
    MMDragonfly_->LogComponentMessage( "Set TIRF Penetration to [" + std::to_string( Value ) + "]", true );
    return TIRFInterface_->SetPenetration_nm( Value );
  }
  ETIRFMode Mode() const { return Penetration; }
private:
  ITIRFInterface* TIRFInterface_;
  CDragonfly* MMDragonfly_;
};

class COffsetWrapper : public CIntDeviceWrapper
{
public:
  COffsetWrapper( ITIRFInterface* TIRFInterface, CDragonfly* MMDragonfly ) : TIRFInterface_( TIRFInterface ), MMDragonfly_( MMDragonfly ) {}
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
    MMDragonfly_->LogComponentMessage( "Set TIRF Offset to [" + std::to_string( Value ) + "]", true );
    return TIRFInterface_->SetOffset( Value );
  }
  ETIRFMode Mode() const { return CriticalAngle; }
private:
  ITIRFInterface* TIRFInterface_;
  CDragonfly* MMDragonfly_;
};

class CTIRFModeIntSubProperty
{
public:
  CTIRFModeIntSubProperty( CIntDeviceWrapper* DeviceWrapper, IConfigFileHandler* ConfigFileHandler, CDragonfly* MMDragonfly, const std::string& PropertyName );
  ~CTIRFModeIntSubProperty();
  typedef MM::Action<CTIRFModeIntSubProperty> CPropertyAction;
  int OnChange( MM::PropertyBase * Prop, MM::ActionType Act );
  void ModeSelected( ETIRFMode SelectedTIRFMode );
  bool IsModeSelected();

private:
  CDragonfly* MMDragonfly_;
  IConfigFileHandler* ConfigFileHandler_;
  CIntDeviceWrapper* DeviceWrapper_;
  std::string PropertyName_;
  MM::Property* MMProp_;
  int BufferedUserSelectionValue_;
  long BufferedUIValue_;
  ETIRFMode SelectedTIRFMode_;
  bool SetPropertyValueFromDeviceOnce_;

  int SetDeviceValue( MM::PropertyBase* Prop, int RequestedValue );
  int SetPropertyValueFromDeviceValue( MM::PropertyBase* Prop );
  void SetPropertyValue( MM::PropertyBase* Prop, long NewValue );
  void SetBufferedUserSelectionValue( int NewValue );
};

#endif