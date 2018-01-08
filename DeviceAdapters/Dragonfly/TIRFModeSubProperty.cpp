#include "TIRFModeSubProperty.h"
#include "Dragonfly.h"

#include <stdexcept>

using namespace std;

CTIRFModeSubProperty::CTIRFModeSubProperty( CDeviceWrapper* DeviceWrapper, CDragonfly* MMDragonfly, const string& PropertyName )
  : MMDragonfly_( MMDragonfly ),
  DeviceWrapper_( DeviceWrapper ),
  PropertyName_( PropertyName ),
  MMProp_( nullptr )
{
  int vMin, vMax, vValue;
  bool vValueRetrieved = DeviceWrapper_->GetLimits( &vMin, &vMax );
  if ( !vValueRetrieved )
  {
    throw std::runtime_error( "Failed to retrieve " + PropertyName_ + " limits" );
  }
  vValueRetrieved = DeviceWrapper_->Get( &vValue );
  if ( !vValueRetrieved )
  {
    throw std::runtime_error( "Failed to retrieve the current value for " + PropertyName_ );
  }

  // Create the MM property for Disk speed
  CPropertyAction* vAct = new CPropertyAction( this, &CTIRFModeSubProperty::OnChange );
  MMDragonfly_->CreateIntegerProperty( PropertyName.c_str(), vValue, false, vAct );
  MMDragonfly_->SetPropertyLimits( PropertyName.c_str(), vMin, vMax );
}

CTIRFModeSubProperty::~CTIRFModeSubProperty()
{
}

void CTIRFModeSubProperty::SetReadOnly( bool ReadOnly )
{
  if ( MMProp_ )
  {
    MMProp_->SetReadOnly( ReadOnly );
  }
}

int CTIRFModeSubProperty::OnChange( MM::PropertyBase * Prop, MM::ActionType Act )
{
  if ( MMProp_ == nullptr )
  {
    MMProp_ = dynamic_cast<MM::Property *>(Prop);
  }
  if ( Act == MM::BeforeGet )
  {
    if ( !SetPropertyValueFromDeviceValue( Prop ) )
    {
      return DEVICE_ERR;
    }
  }
  else if ( Act == MM::AfterSet )
  {
    long vRequestedValue;
    Prop->Get( vRequestedValue );
    int vMin, vMax;
    bool vLimitsRetrieved = DeviceWrapper_->GetLimits( &vMin, &vMax );
    if ( vLimitsRetrieved )
    {
      if ( vRequestedValue >= (long)vMin && vRequestedValue <= (long)vMax )
      {
        DeviceWrapper_->Set( vRequestedValue );
      }
      else
      {
        MMDragonfly_->LogComponentMessage( "Requested " + PropertyName_ + " value is out of bound. Ignoring request." );
      }
    }
    else
    {
      MMDragonfly_->LogComponentMessage( "Failed to retrieve " + PropertyName_ + " limits" );
    }
  }

  return DEVICE_OK;
}

bool CTIRFModeSubProperty::SetPropertyValueFromDeviceValue( MM::PropertyBase* Prop )
{
  bool vValueSet = false;
  int vMin, vMax;
  bool vLimitsRetrieved = DeviceWrapper_->GetLimits( &vMin, &vMax );
  if ( vLimitsRetrieved )
  {
    Prop->SetLimits( vMin, vMax );
    int vValue;
    if ( DeviceWrapper_->Get( &vValue ) )
    {
      Prop->Set( (long)vValue );
      vValueSet = true;
    }
    else
    {
      MMDragonfly_->LogComponentMessage( "Failed to retrieve the current value for " + PropertyName_ );
    }
  }
  else
  {
    MMDragonfly_->LogComponentMessage( "Failed to retrieve " + PropertyName_ + " limits" );
  }

  return vValueSet;
}
