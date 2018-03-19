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
  SelectedTIRFMode_( UnknownTIRFMode ),
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
    ConfigFileHandler_->SavePropertyValue( PropertyName_, to_string( static_cast< long long >( vValue ) ) );
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
      vRet = SetPropertyValueFromDeviceValue( Prop );
      if ( vRet == DEVICE_OK )
      {
        SetPropertyValueFromDeviceOnce_ = false;
      }
    }    
  }
  else if ( Act == MM::AfterSet )
  {
    long vRequestedValue;
    Prop->Get( vRequestedValue );
    if ( !IsModeSelected() )
    {
      // The current mode is not selected, backup the request
      // The device will be updated with the user request next time the mode is selected
      SetBufferedUserSelectionValue( (int)vRequestedValue );
      if ( SelectedTIRFMode_ == CriticalAngle && DeviceWrapper_->Mode() == Penetration )
      {
        // Special case for when Critical Angle is selected and the Penetration is changed
        // Since the user is really not supposed to do this we prefer reset the UI to prevent confusions
        SetPropertyValue( Prop, BufferedUIValue_ );
      }
    }
    else
    {
      BufferedUIValue_ = vRequestedValue;
      vRet = SetDeviceValue( Prop, (int)vRequestedValue );
    }
  }

  return vRet;
}

void CTIRFModeIntSubProperty::SetBufferedUserSelectionValue( int NewValue )
{
  BufferedUserSelectionValue_ = NewValue;
  // Save the new value to the config file
  ConfigFileHandler_->SavePropertyValue( PropertyName_, to_string( static_cast< long long >( BufferedUserSelectionValue_ ) ) );
}

int CTIRFModeIntSubProperty::SetDeviceValue( MM::PropertyBase* Prop, int RequestedValue )
{
  int vRet = DEVICE_ERR;
  int vMin, vMax;
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
          long vNewValue;
          Prop->Get( vNewValue );
          SetBufferedUserSelectionValue( (int)vNewValue );
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

void CTIRFModeIntSubProperty::SetPropertyValue( MM::PropertyBase* Prop, long NewValue )
{
  Prop->Set( NewValue );
  BufferedUIValue_ = NewValue;
}

int CTIRFModeIntSubProperty::SetPropertyValueFromDeviceValue( MM::PropertyBase* Prop )
{
  int vRet = DEVICE_ERR;
  int vMin, vMax;
  bool vLimitsRetrieved = DeviceWrapper_->GetLimits( &vMin, &vMax );
  if ( vLimitsRetrieved )
  {
    Prop->SetLimits( vMin, vMax );
    int vValue;
    if ( DeviceWrapper_->Get( &vValue ) )
    {
      SetPropertyValue( Prop, (long)vValue );
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
      MMDragonfly_->SetProperty( PropertyName_.c_str(), to_string( static_cast< long long >( BufferedUserSelectionValue_ ) ).c_str() );
    }
  }
  else
  {
    if ( SelectedTIRFMode_ == CriticalAngle && DeviceWrapper_->Mode() == Penetration )
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
