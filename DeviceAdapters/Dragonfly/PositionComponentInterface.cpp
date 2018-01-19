#include "PositionComponentInterface.h"
#include "Dragonfly.h"
#include "ASDConfigInterface.h"

#include <stdexcept>
#include <map>

using namespace std;

IPositionComponentInterface::IPositionComponentInterface( CDragonfly* MMDragonfly, const string& PropertyName )
  : MMDragonfly_( MMDragonfly ),
  PropertyName_( PropertyName ),
  Initialised_( false )
{
}

IPositionComponentInterface::~IPositionComponentInterface()
{
}

void IPositionComponentInterface::Initialise()
{
  if ( Initialised_ ) return;

  // Retrieve values from the device
  if ( !RetrievePositionsFromFilterSet() )
  {
    if ( !RetrievePositionsWithoutDescriptions() )
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
  MMDragonfly_->CreateProperty( PropertyName_.c_str(), vCurrentPositionName.c_str(), MM::String, false, vAct );

  // Populate the possible positions
  TPositionNameMap::const_iterator vIt = PositionNames_.begin();
  while ( vIt != PositionNames_.end() )
  {
    MMDragonfly_->AddAllowedValue( PropertyName_.c_str(), vIt->second.c_str() );
    vIt++;
  }

  Initialised_ = true;
}

bool IPositionComponentInterface::RetrievePositionsFromFilterSet()
{
  const static unsigned int vStringLength = 64;
  bool vPositionsRetrieved = false;
  IFilterSet* vFilterSet = GetFilterSet();
  if ( vFilterSet != nullptr )
  {
    unsigned int vMinPos, vMaxPos;
    if ( vFilterSet->GetLimits( vMinPos, vMaxPos ) )
    {
      char vDescription[vStringLength];
      unsigned int vUndefinedIndex = 1;
      for ( unsigned int vIndex = vMinPos; vIndex <= vMaxPos; vIndex++ )
      {
        string vPositionName;
        if ( vFilterSet->GetFilterDescription( vIndex, vDescription, vStringLength ) == false )
        {
          vPositionName += "Undefined Position " + to_string(vUndefinedIndex);
          vUndefinedIndex++;
        }
        else
        {
          vPositionName += vDescription;
        }
        PositionNames_[vIndex] = vPositionName;
      }

      vPositionsRetrieved = true;
    }
  }
  else
  {
    MMDragonfly_->LogComponentMessage( "Invalid FilterSet pointer for " + PropertyName_ );
  }
  return vPositionsRetrieved;
}

bool IPositionComponentInterface::RetrievePositionsWithoutDescriptions()
{
  bool vPositionsRetrieved = false;
  unsigned int vMinValue, vMaxValue;
  if ( GetLimits( vMinValue, vMaxValue ) )
  {
    for ( unsigned int vIndex = vMinValue; vIndex <= vMaxValue; vIndex++ )
    {
      string vPositionName = "Undefined Position " + to_string( vIndex );
      PositionNames_[vIndex] = vPositionName;
    }

    vPositionsRetrieved = true;
  }
  return vPositionsRetrieved;
}

int IPositionComponentInterface::OnPositionChange( MM::PropertyBase* Prop, MM::ActionType Act )
{
  if ( !Initialised_ ) return DEVICE_ERR;

  if ( Act == MM::BeforeGet )
  {
    if ( UpdateAllowedValues() )
    {
      // If allowed values have been updated, read the current property from the device
      SetPropertyValueFromDevicePosition( Prop );
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
      SetPosition( vIt->first );
    }
    else
    {
      // Reset position displayed in the UI to the current device position
      MMDragonfly_->LogComponentMessage( "Unknown " + PropertyName_ + " position requested [" + vRequestedPosition + "]. Ignoring request." );
      SetPropertyValueFromDevicePosition( Prop );
    }
  }
  return DEVICE_OK;
}

bool IPositionComponentInterface::SetPropertyValueFromDevicePosition( MM::PropertyBase* Prop )
{
  bool vValueSet = false;
  unsigned int vPosition;
  if ( GetPosition( vPosition ) )
  {
    if ( PositionNames_.find( vPosition ) != PositionNames_.end() )
    {
      Prop->Set( PositionNames_[vPosition].c_str() );
      vValueSet = true;
    }
    else
    {
      MMDragonfly_->LogComponentMessage( "Current " + PropertyName_ + " position invalid [ " + to_string(vPosition) + " ]" );
    }
  }
  else
  {
    MMDragonfly_->LogComponentMessage( "Failed to read the current " + PropertyName_ + " position" );
  }

  return vValueSet;
}
