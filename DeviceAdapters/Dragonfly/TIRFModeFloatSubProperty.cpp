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
  MMProp_( nullptr ),
  BufferedUserSelectionValue_( 0.f ),
  BufferedUIValue_( 0.f ),
  SelectedTIRFMode_( ETIRFMode::UnknownTIRFMode )
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
  BufferedUserSelectionValue_ = vValue;
  BufferedUIValue_ = vValue;

  // Create the MM property for Disk speed
  CPropertyAction* vAct = new CPropertyAction( this, &CTIRFModeFloatSubProperty::OnChange );
  MMDragonfly_->CreateFloatProperty( PropertyName_.c_str(), vValue, false, vAct );
  MMDragonfly_->SetPropertyLimits( PropertyName_.c_str(), vMin, vMax );
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
  int vRet = DEVICE_OK;
  if ( MMProp_ == nullptr )
  {
    MMProp_ = dynamic_cast<MM::Property *>(Prop);
  }
  if ( Act == MM::AfterSet )
  {
    double vRequestedValue;
    Prop->Get( vRequestedValue );
    if ( !IsModeSelected() )
    {
      // The current mode is not selected, backup the request and reset the UI to the previously set value
      // The device will be updated with the user request next time the mode is selected
      BufferedUserSelectionValue_ = (float)vRequestedValue;
      SetPropertyValue( Prop, BufferedUIValue_ );
    }
    else
    {
      BufferedUIValue_ = vRequestedValue;
      if ( !SetDeviceValue( Prop, (float)vRequestedValue ) )
      {
        vRet = DEVICE_ERR;
      }
    }
  }

  return DEVICE_OK;
}

bool CTIRFModeFloatSubProperty::SetDeviceValue( MM::PropertyBase* Prop, float RequestedValue )
{
  bool vValueSet = false;
  float vMin, vMax;
  bool vLimitsRetrieved = DeviceWrapper_->GetLimits( &vMin, &vMax );
  if ( vLimitsRetrieved )
  {
    if ( RequestedValue >= vMin && RequestedValue <= vMax )
    {
      if ( !DeviceWrapper_->Set( RequestedValue ) )
      {
        // Failed to set the device. Best is to refresh the UI with the device's current value.
        vValueSet = SetPropertyValueFromDeviceValue( Prop );
        if ( vValueSet )
        {
          double vNewValue;
          Prop->Get( vNewValue );
          BufferedUserSelectionValue_ = (float)vNewValue;
        }
      }
      else
      {
        BufferedUserSelectionValue_ = RequestedValue;
        // Save the new value to the config file
        ConfigFileHandler_->SavePropertyValue( PropertyName_, to_string( BufferedUserSelectionValue_ ) );
        vValueSet = true;
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
  return vValueSet;
}

void CTIRFModeFloatSubProperty::SetPropertyValue( MM::PropertyBase* Prop, double NewValue )
{
  Prop->Set( NewValue );
  BufferedUIValue_ = NewValue;
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
      SetPropertyValue( Prop, (double)vValue );
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

void CTIRFModeFloatSubProperty::ModeSelected( ETIRFMode SelectedTIRFMode )
{
  SelectedTIRFMode_ = SelectedTIRFMode;
  if ( IsModeSelected() )
  {
    if ( MMProp_ )
    {
      double vCurrentValue;
      MMProp_->Get( vCurrentValue );
      if ( BufferedUserSelectionValue_ != vCurrentValue )
      {
        SetPropertyValue( MMProp_, (double)BufferedUserSelectionValue_ );
        SetDeviceValue( MMProp_, BufferedUserSelectionValue_ );
      }
    }
    else
    {
      MMDragonfly_->SetProperty( PropertyName_.c_str(), to_string( BufferedUserSelectionValue_ ).c_str() );
    }
  }
}

bool CTIRFModeFloatSubProperty::IsModeSelected()
{
  return SelectedTIRFMode_ == DeviceWrapper_->Mode();
}
