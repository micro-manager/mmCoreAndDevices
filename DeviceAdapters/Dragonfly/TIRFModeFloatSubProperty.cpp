#include "TIRFModeFloatSubProperty.h"
#include "Dragonfly.h"
#include "ConfigFileHandlerInterface.h"

#include <stdexcept>

using namespace std;

CTIRFModeFloatSubProperty::CTIRFModeFloatSubProperty( CFloatDeviceWrapper* DeviceWrapper, IConfigFileHandler* ConfigFileHandler, CDragonfly* MMDragonfly, const string& PropertyName )
  : MMDragonfly_( MMDragonfly ),
  ConfigFileHandler_( ConfigFileHandler ),
  DeviceWrapper_( DeviceWrapper ),
  PropertyName_( PropertyName ),
  MMProp_( nullptr )
{
  float vMin, vMax, vValue = 0;
  bool vValueRetrieved = DeviceWrapper_->GetLimits( &vMin, &vMax );
  if ( !vValueRetrieved )
  {
    throw std::runtime_error( "Failed to retrieve " + PropertyName_ + " limits" );
  }
  string vValueFromConfigFile;
  vValueRetrieved = ConfigFileHandler_->LoadPropertyValue( PropertyName_, vValueFromConfigFile );
  if ( vValueRetrieved )
  {
    try
    {
      vValue = stof( vValueFromConfigFile );
    }
    catch ( exception& e )
    {
      MMDragonfly_->LogComponentMessage( "Error raised from stof() when reading value from config file for " + PropertyName_ + ". Loading value from device instead. Exception raised: " + e.what() );
      vValueRetrieved = false;
    }
  }
  if ( !vValueRetrieved )
  {
    vValueRetrieved = DeviceWrapper_->Get( &vValue );
    if ( !vValueRetrieved )
    {
      throw std::runtime_error( "Failed to retrieve the current value for " + PropertyName_ );
    }
    ConfigFileHandler_->SavePropertyValue( PropertyName_, to_string( vValue ) );
  }
  BufferedValue_ = vValue;

  // Create the MM property for Disk speed
  CPropertyAction* vAct = new CPropertyAction( this, &CTIRFModeFloatSubProperty::OnChange );
  MMDragonfly_->CreateFloatProperty( PropertyName.c_str(), vValue, false, vAct );
  MMDragonfly_->SetPropertyLimits( PropertyName.c_str(), vMin, vMax );
  // Enforce call to the action method to initialise MMProp_
  MMDragonfly_->SetProperty( PropertyName.c_str(), to_string(vValue).c_str() );
}

CTIRFModeFloatSubProperty::~CTIRFModeFloatSubProperty()
{
}

void CTIRFModeFloatSubProperty::SetReadOnly( bool /*ReadOnly*/ )
{
  if ( MMProp_ )
  {
    //MMProp_->SetReadOnly( ReadOnly );
  }
}

int CTIRFModeFloatSubProperty::OnChange( MM::PropertyBase * Prop, MM::ActionType Act )
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
    double vRequestedValue;
    Prop->Get( vRequestedValue );
    float vMin, vMax;
    bool vLimitsRetrieved = DeviceWrapper_->GetLimits( &vMin, &vMax );
    if ( vLimitsRetrieved )
    {
      if ( vRequestedValue >= (double)vMin && vRequestedValue <= (double)vMax )
      {
        if ( !DeviceWrapper_->Set( (float)vRequestedValue ) )
        {
          float vDeviceValue;
          DeviceWrapper_->Get( &vDeviceValue );
          Prop->Set( (double)vDeviceValue );
        }
        else
        {
          BufferedValue_ = (float) vRequestedValue;
          // Save the value requested by the user to the config file
          ConfigFileHandler_->SavePropertyValue( PropertyName_, to_string( vRequestedValue ) );
        }
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

bool CTIRFModeFloatSubProperty::SetPropertyValueFromDeviceValue( MM::PropertyBase* Prop )
{
  bool vValueSet = false;
  float vMin, vMax;
  bool vLimitsRetrieved = DeviceWrapper_->GetLimits( &vMin, &vMax );
  if ( vLimitsRetrieved )
  {
    Prop->SetLimits( vMin, vMax );
    float vValue;
    if ( DeviceWrapper_->Get( &vValue ) )
    {
      Prop->Set( (double)vValue );
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

void CTIRFModeFloatSubProperty::ModeSelected()
{
  if ( MMProp_ )
  {
    double vCurrentValue;
    MMProp_->Get( vCurrentValue );
    if ( BufferedValue_ != vCurrentValue )
    {
      MMProp_->Set( (double)BufferedValue_ );
      DeviceWrapper_->Set( BufferedValue_ );
    }
  }
}