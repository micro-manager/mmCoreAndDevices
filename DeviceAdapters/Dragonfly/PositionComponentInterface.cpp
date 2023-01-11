#include "PositionComponentInterface.h"
#include "Dragonfly.h"
#include "ASDConfigInterface.h"

#include <stdexcept>
#include <map>

using namespace std;

bool IsCharacterNumerical( char Character )
{
  return ( Character >= 48 && Character <= 57 );
}

IPositionComponentInterface::IPositionComponentInterface( CDragonfly* MMDragonfly, const string& PropertyName, bool AddIndexToPositionNames )
  : MMDragonfly_( MMDragonfly ),
  PropertyName_( PropertyName ),
  Initialised_( false ),
  AddIndexToPositionNames_( AddIndexToPositionNames )
{
}

IPositionComponentInterface::~IPositionComponentInterface()
{
}

void IPositionComponentInterface::Initialise()
{
  if ( Initialised_ ) return;

  // Retrieve values from the device
  IFilterSet* vFilterSet = GetFilterSet();
  if ( vFilterSet == nullptr )
  {
    MMDragonfly_->LogComponentMessage( "Invalid FilterSet pointer for " + PropertyName_ );
  }
  if ( vFilterSet == nullptr || !CPositionComponentHelper::RetrievePositionsFromFilterSet( vFilterSet, PositionNames_, AddIndexToPositionNames_ ) )
  {
    unsigned int vMinValue, vMaxValue;
    if ( GetLimits( vMinValue, vMaxValue ) )
    {
      CPositionComponentHelper::RetrievePositionsWithoutDescriptions( vMinValue, vMaxValue, PositionNames_ );
    }
    else
    {
      throw runtime_error( "Failed to retrieve " + PropertyName_ + " positions" );
    }
  }

  // Retrieve the current position from the device
  unsigned int vPosition;
  if ( !GetPosition( vPosition ) )
  {
    throw runtime_error( "Failed to read the current " + PropertyName_ + " position" );
  }

  // Retrieve the current position name
  string vCurrentPositionName = "Undefined";
  if ( PositionNames_.find( vPosition ) != PositionNames_.end() )
  {
    vCurrentPositionName = PositionNames_[vPosition];
  }
  else
  {
    MMDragonfly_->LogComponentMessage( "Current " + PropertyName_ + " position invalid" );
  }

  // Create the MM property
  CPropertyAction* vAct = new CPropertyAction( this, &IPositionComponentInterface::OnPositionChange );
  int vRet = MMDragonfly_->CreateProperty( PropertyName_.c_str(), vCurrentPositionName.c_str(), MM::String, false, vAct );
  if ( vRet != DEVICE_OK )
  {
    throw runtime_error( "Error creating " + PropertyName_ + " property" );
  }

  // Populate the possible positions
  TPositionNameMap::const_iterator vIt = PositionNames_.begin();
  while ( vIt != PositionNames_.end() )
  {
    MMDragonfly_->AddAllowedValue( PropertyName_.c_str(), vIt->second.c_str() );
    vIt++;
  }

  Initialised_ = true;
}

int IPositionComponentInterface::OnPositionChange( MM::PropertyBase* Prop, MM::ActionType Act )
{
  if ( !Initialised_ ) return DEVICE_ERR;

  int vRet = DEVICE_OK;
  if ( Act == MM::BeforeGet )
  {
    if ( UpdateAllowedValues() )
    {
      // If allowed values have been updated, read the current property from the device
      vRet = SetPropertyValueFromDevicePosition( Prop );
    }
  }
  else if ( Act == MM::AfterSet )
  {
    // Search the requested position in the map of existing positions
    string vRequestedPosition;
    Prop->Get( vRequestedPosition );
    bool vFound = false;
    TPositionNameMap::const_iterator vIt = PositionNames_.begin();
    while ( !vFound && vIt != PositionNames_.end() )
    {
      if ( vIt->second == vRequestedPosition )
      {
        vFound = true;
      }
      else
      {
        vIt++;
      }
    }
    if ( vFound )
    {
      // Update device position
      if ( !SetPosition( vIt->first ) )
      {
        MMDragonfly_->LogComponentMessage( "Failed to set the position for property " + PropertyName_ + " [" + to_string( static_cast< long long >( vIt->first ) ) + "]" );
        vRet = DEVICE_CAN_NOT_SET_PROPERTY;
      }
    }
    else
    {
      // Reset position displayed in the UI to the current device position
      MMDragonfly_->LogComponentMessage( "Unknown " + PropertyName_ + " position requested [" + vRequestedPosition + "]. Ignoring request." );
      SetPropertyValueFromDevicePosition( Prop );
      vRet = DEVICE_INVALID_PROPERTY_VALUE;
    }
  }
  return vRet;
}

int IPositionComponentInterface::SetPropertyValueFromDevicePosition( MM::PropertyBase* Prop )
{
  int vRet = DEVICE_ERR;
  unsigned int vPosition;
  if ( GetPosition( vPosition ) )
  {
    if ( PositionNames_.find( vPosition ) != PositionNames_.end() )
    {
      Prop->Set( PositionNames_[vPosition].c_str() );
      vRet = DEVICE_OK;
    }
    else
    {
      MMDragonfly_->LogComponentMessage( "Current " + PropertyName_ + " position invalid [ " + to_string( static_cast< long long >( vPosition ) ) + " ]" );
      vRet = DEVICE_UNKNOWN_POSITION;
    }
  }
  else
  {
    MMDragonfly_->LogComponentMessage( "Failed to read the current " + PropertyName_ + " position" );
    vRet = DEVICE_ERR;
  }

  return vRet;
}
