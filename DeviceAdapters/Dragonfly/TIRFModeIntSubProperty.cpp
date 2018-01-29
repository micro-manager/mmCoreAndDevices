#include "TIRFModeIntSubProperty.h"
#include "Dragonfly.h"
#include "ConfigFileHandlerInterface.h"
#include "TIRFMode.h"

#include <stdexcept>

using namespace std;

CTIRFModeIntSubProperty::CTIRFModeIntSubProperty( CIntDeviceWrapper* DeviceWrapper, IConfigFileHandler* ConfigFileHandler, CDragonfly* MMDragonfly, const string& PropertyName )
  : MMDragonfly_( MMDragonfly ),
  ConfigFileHandler_( ConfigFileHandler ),
  DeviceWrapper_( DeviceWrapper ),
  PropertyName_( PropertyName ),
  MMProp_( nullptr ),
  BufferedUserSelectionValue_( 0 ),
  BufferedUIValue_( 0 ),
  SelectedTIRFMode_( ETIRFMode::UnknownTIRFMode ),
  SetPropertyValueFromDeviceOnce_( false )
{
  int vMin, vMax, vValue = 0;
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
  BufferedUserSelectionValue_ = vValue;
  BufferedUIValue_ = vValue;

  // Create the MM property for Disk speed
  CPropertyAction* vAct = new CPropertyAction( this, &CTIRFModeIntSubProperty::OnChange );
  MMDragonfly_->CreateIntegerProperty( PropertyName_.c_str(), vValue, false, vAct );
  MMDragonfly_->SetPropertyLimits( PropertyName_.c_str(), vMin, vMax );
}

CTIRFModeIntSubProperty::~CTIRFModeIntSubProperty()
{
}

void CTIRFModeIntSubProperty::SetReadOnly( bool /*ReadOnly*/ )
{
  if ( MMProp_ )
  {
    //MMProp_->SetReadOnly( ReadOnly );
  }
}

int CTIRFModeIntSubProperty::OnChange( MM::PropertyBase * Prop, MM::ActionType Act )
{
  int vRet = DEVICE_OK;
  if ( MMProp_ == nullptr )
  {
    MMProp_ = dynamic_cast<MM::Property *>(Prop);
  }
  if ( Act == MM::BeforeGet )
  {
    if ( SetPropertyValueFromDeviceOnce_ )
    {
      if ( SetPropertyValueFromDeviceValue( Prop ) )
      {
        SetPropertyValueFromDeviceOnce_ = false;
      }
      else
      {
        vRet = DEVICE_ERR;
      }
    }    
  }
  else if ( Act == MM::AfterSet )
  {
    long vRequestedValue;
    Prop->Get( vRequestedValue );
    if ( !IsModeSelected() )
    {
      // The current mode is not selected, backup the request and reset the UI to the previously set value
      // The device will be updated with the user request next time the mode is selected
      BufferedUserSelectionValue_ = (int)vRequestedValue;
      SetPropertyValue( Prop, BufferedUIValue_ );
    }
    else
    {
      BufferedUIValue_ = vRequestedValue;
      if ( !SetDeviceValue( Prop, (int)vRequestedValue ) )
      {
        vRet = DEVICE_ERR;
      }
    }
  }

  return vRet;
}

bool CTIRFModeIntSubProperty::SetDeviceValue( MM::PropertyBase* Prop, int RequestedValue )
{
  bool vValueSet = false;
  int vMin, vMax;
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
          long vNewValue;
          Prop->Get( vNewValue );
          BufferedUserSelectionValue_ = (int)vNewValue;
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

void CTIRFModeIntSubProperty::SetPropertyValue( MM::PropertyBase* Prop, long NewValue )
{
  Prop->Set( NewValue );
  BufferedUIValue_ = NewValue;
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
      SetPropertyValue( Prop, (long)vValue );
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

void CTIRFModeIntSubProperty::ModeSelected( ETIRFMode SelectedTIRFMode )
{
  SelectedTIRFMode_ = SelectedTIRFMode;
  if ( IsModeSelected() )
  {
    if ( MMProp_ )
    {
      long vCurrentValue;
      MMProp_->Get( vCurrentValue );
      if ( BufferedUserSelectionValue_ != vCurrentValue )
      {
        SetPropertyValue( MMProp_, (long)BufferedUserSelectionValue_ );
        SetDeviceValue( MMProp_, BufferedUserSelectionValue_ );
      }
    }
    else
    {
      MMDragonfly_->SetProperty( PropertyName_.c_str(), to_string( BufferedUserSelectionValue_ ).c_str() ); 
    }
  }
  else
  {
    if ( SelectedTIRFMode == ETIRFMode::CriticalAngle && DeviceWrapper_->Mode() == ETIRFMode::Penetration )
    {
      // Critical Angle has just been selected, get the new value from the device
      if ( MMProp_ )
      {
        SetPropertyValueFromDeviceValue( MMProp_ );
      }
      else
      {
        // Activate BeforeGet in OnChange() once to let it update the property value from the device
        SetPropertyValueFromDeviceOnce_ = true;
      }
    }
  }
}

bool CTIRFModeIntSubProperty::IsModeSelected()
{
  return SelectedTIRFMode_ == DeviceWrapper_->Mode();
}
