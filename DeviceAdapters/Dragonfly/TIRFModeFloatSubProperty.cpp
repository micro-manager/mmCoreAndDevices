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
  SelectedTIRFMode_( UnknownTIRFMode )
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
    ConfigFileHandler_->SavePropertyValue( PropertyName_, to_string( static_cast< long long >( vValue ) ) );
  }
  BufferedUserSelectionValue_ = vValue;

  // Create the MM property for Disk speed
  CPropertyAction* vAct = new CPropertyAction( this, &CTIRFModeFloatSubProperty::OnChange );
  MMDragonfly_->CreateFloatProperty( PropertyName_.c_str(), vValue, false, vAct );
  MMDragonfly_->SetPropertyLimits( PropertyName_.c_str(), vMin, vMax );
}

CTIRFModeFloatSubProperty::~CTIRFModeFloatSubProperty()
{
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
      // The current mode is not selected, backup the request
      // The device will be updated with the user request next time the mode is selected
      SetBufferedUserSelectionValue( (float)vRequestedValue );
    }
    else
    {
      vRet = SetDeviceValue( Prop, (float)vRequestedValue );
    }
  }

  return vRet;
}

void CTIRFModeFloatSubProperty::SetBufferedUserSelectionValue( float NewValue )
{
  BufferedUserSelectionValue_ = NewValue;
  // Save the new value to the config file
  ConfigFileHandler_->SavePropertyValue( PropertyName_, to_string( static_cast< long long >( BufferedUserSelectionValue_ ) ) );
}

int CTIRFModeFloatSubProperty::SetDeviceValue( MM::PropertyBase* Prop, float RequestedValue )
{
  int vRet = DEVICE_ERR;
  float vMin, vMax;
  bool vLimitsRetrieved = DeviceWrapper_->GetLimits( &vMin, &vMax );
  if ( vLimitsRetrieved )
  {
    if ( RequestedValue >= vMin && RequestedValue <= vMax )
    {
      if ( !DeviceWrapper_->Set( RequestedValue ) )
      {
        // Failed to set the device. Best is to refresh the UI with the device's current value.
        vRet = DEVICE_CAN_NOT_SET_PROPERTY;
        MMDragonfly_->LogComponentMessage( "Failed to set " + PropertyName_ + " position [" + to_string( static_cast< long long >( RequestedValue ) ) + "]" );
        if ( SetPropertyValueFromDeviceValue( Prop ) == DEVICE_OK )
        {
          double vNewValue;
          Prop->Get( vNewValue );
          SetBufferedUserSelectionValue( (float)vNewValue );
        }
      }
      else
      {
        SetBufferedUserSelectionValue( RequestedValue );
        vRet = DEVICE_OK;
      }
    }
    else
    {
      MMDragonfly_->LogComponentMessage( "Requested " + PropertyName_ + " value is out of bound. Ignoring request." );
      vRet = DEVICE_INVALID_PROPERTY_VALUE;
    }
  }
  else
  {
    MMDragonfly_->LogComponentMessage( "Failed to retrieve " + PropertyName_ + " limits" );
    vRet = DEVICE_ERR;
  }
  return vRet;
}

void CTIRFModeFloatSubProperty::SetPropertyValue( MM::PropertyBase* Prop, double NewValue )
{
  Prop->Set( NewValue );
}

int CTIRFModeFloatSubProperty::SetPropertyValueFromDeviceValue( MM::PropertyBase* Prop )
{
  int vRet = DEVICE_ERR;
  float vMin, vMax;
  bool vLimitsRetrieved = DeviceWrapper_->GetLimits( &vMin, &vMax );
  if ( vLimitsRetrieved )
  {
    Prop->SetLimits( vMin, vMax );
    float vValue;
    if ( DeviceWrapper_->Get( &vValue ) )
    {
      SetPropertyValue( Prop, (double)vValue );
      vRet = DEVICE_OK;
    }
    else
    {
      MMDragonfly_->LogComponentMessage( "Failed to retrieve the current value for " + PropertyName_ );
      vRet = DEVICE_ERR;
    }
  }
  else
  {
    MMDragonfly_->LogComponentMessage( "Failed to retrieve " + PropertyName_ + " limits" );
    vRet = DEVICE_ERR;
  }

  return vRet;
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
      MMDragonfly_->SetProperty( PropertyName_.c_str(), to_string( static_cast< long long >( BufferedUserSelectionValue_ ) ).c_str() );
    }
  }
}

bool CTIRFModeFloatSubProperty::IsModeSelected()
{
  return SelectedTIRFMode_ == DeviceWrapper_->Mode();
}
