#include "TIRFModeIntSubProperty.h"
#include "Dragonfly.h"
#include "ConfigFileHandlerInterface.h"

#include <stdexcept>

using namespace std;

CTIRFModeIntSubProperty::CTIRFModeIntSubProperty( CIntDeviceWrapper* DeviceWrapper, IConfigFileHandler* ConfigFileHandler, CDragonfly* MMDragonfly, const string& PropertyName )
  : MMDragonfly_( MMDragonfly ),
  ConfigFileHandler_( ConfigFileHandler ),
  DeviceWrapper_( DeviceWrapper ),
  PropertyName_( PropertyName ),
  MMProp_( nullptr ),
  BufferedValue_( 0 )
{
  int vMin, vMax, vValue;
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
      vValue = stoi( vValueFromConfigFile );
    }
    catch ( exception& e )
    {
      MMDragonfly_->LogComponentMessage( "Error raised from stoi() when reading value from config file for " + PropertyName_ + ". Loading value from device instead. Exception raised: " + e.what() );
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
  CPropertyAction* vAct = new CPropertyAction( this, &CTIRFModeIntSubProperty::OnChange );
  MMDragonfly_->CreateIntegerProperty( PropertyName.c_str(), vValue, false, vAct );
  MMDragonfly_->SetPropertyLimits( PropertyName.c_str(), vMin, vMax );
  // Enforce call to the action method to initialise MMProp_
  MMDragonfly_->SetProperty( PropertyName.c_str(), to_string(vValue).c_str() );
}

CTIRFModeIntSubProperty::~CTIRFModeIntSubProperty()
{
}

void CTIRFModeIntSubProperty::SetReadOnly( bool ReadOnly )
{
  if ( MMProp_ )
  {
    //MMProp_->SetReadOnly( ReadOnly );
  }
}

int CTIRFModeIntSubProperty::OnChange( MM::PropertyBase * Prop, MM::ActionType Act )
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
        if ( !DeviceWrapper_->Set( vRequestedValue ) )
        {
          int vDeviceValue;
          DeviceWrapper_->Get( &vDeviceValue );
          Prop->Set( (long)vDeviceValue );
        }
        else
        {
          BufferedValue_ = vRequestedValue;
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

bool CTIRFModeIntSubProperty::SetPropertyValueFromDeviceValue( MM::PropertyBase* Prop )
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

void CTIRFModeIntSubProperty::ModeSelected()
{
  if ( MMProp_ )
  {
    long vCurrentValue;
    MMProp_->Get( vCurrentValue );
    if ( BufferedValue_ != vCurrentValue )
    {
      MMProp_->Set( (long)BufferedValue_ );
      DeviceWrapper_->Set( BufferedValue_ );
    }
  }
}