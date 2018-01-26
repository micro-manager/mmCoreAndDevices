#include "FilterWheelProperty.h"
#include "Dragonfly.h"
#include "FilterWheelDeviceInterface.h"
#include "ASDConfigInterface.h"

#include <stdexcept>
#include <map>

using namespace std;

CFilterWheelProperty::CFilterWheelProperty( IFilterWheelDeviceInterface* FilterWheelDevice, CDragonfly* MMDragonfly, const string& PropertyName, const string& ComponentName )
  : FilterWheelDevice_( FilterWheelDevice ),
  MMDragonfly_( MMDragonfly ),
  PropertyName_( PropertyName ),
  ComponentName_( ComponentName )
{
  // Retrieve values from the device
  if ( !RetrievePositionsFromFilterConfig() )
  {
    if ( !RetrievePositionsWithoutDescriptions() )
    {
      throw std::runtime_error( "Failed to retrieve " + ComponentName_ + " positions" );
    }
  }

  // Retrieve the current position from the device
  unsigned int vPosition;
  if ( !FilterWheelDevice_->GetPosition( vPosition ) )
  {
    throw std::runtime_error( "Failed to read the current " + ComponentName_ + " position" );
  }

  // Retrieve the position name
  string vCurrentPositionName = "Undefined";
  if ( PositionNames_.find( vPosition ) != PositionNames_.end() )
  {
    vCurrentPositionName = PositionNames_[vPosition];
  }
  else
  {
    MMDragonfly_->LogComponentMessage( "Current " + ComponentName_ + " position invalid" );
  }

  // Create the MM property
  CPropertyAction* vAct = new CPropertyAction( this, &CFilterWheelProperty::OnPositionChange );
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

}

CFilterWheelProperty::~CFilterWheelProperty()
{

}

bool CFilterWheelProperty::RetrievePositionsFromFilterConfig()
{
  const static unsigned int vStringLength = 64;
  bool vPositionsRetrieved = false;
  IFilterConfigInterface* vFilterConfigInterface = FilterWheelDevice_->GetFilterConfigInterface();
  if ( vFilterConfigInterface != nullptr )
  {
    IFilterSet *vFilterSet = vFilterConfigInterface->GetFilterSet();
    if ( vFilterSet != nullptr )
    {
      unsigned int vMinPos, vMaxPos;
      if ( vFilterSet->GetLimits( vMinPos, vMaxPos ) )
      {
        char vDescription[vStringLength];
        for ( unsigned int vIndex = vMinPos; vIndex <= vMaxPos; vIndex++ )
        {
          string vPositionName( to_string( vIndex ) + " - " );
          if ( vFilterSet->GetFilterDescription( vIndex, vDescription, vStringLength ) == false )
          {
            vPositionName += "Empty";
          }
          else
          {
            vPositionName += FilterWheelDevice_->ParseDescription( vDescription );
          }
          PositionNames_[vIndex] = vPositionName;
        }

        vPositionsRetrieved = true;
      }
    }
  }
  return vPositionsRetrieved;
}

bool CFilterWheelProperty::RetrievePositionsWithoutDescriptions()
{
  bool vPositionsRetrieved = false;
  unsigned int vMinValue, vMaxValue;
  if ( FilterWheelDevice_->GetLimits( vMinValue, vMaxValue ) )
  {
    for ( unsigned int vIndex = vMinValue; vIndex <= vMaxValue; vIndex++ )
    {
      string vPositionName = to_string( vIndex ) + " - Unknown";
      PositionNames_[vIndex] = vPositionName;
    }

    vPositionsRetrieved = true;
  }
  return vPositionsRetrieved;
}

int CFilterWheelProperty::OnPositionChange( MM::PropertyBase* Prop, MM::ActionType Act )
{
  if ( Act == MM::AfterSet )
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
      FilterWheelDevice_->SetPosition( vIt->first );
    }
    else
    {
      // Reset position displayed in the UI to the current device position
      MMDragonfly_->LogComponentMessage( "Unknown " + ComponentName_ + " position requested [" + vRequestedPosition + "]. Ignoring request." );
      SetPropertyValueFromDevicePosition( Prop );
    }
  }
  return DEVICE_OK;
}

bool CFilterWheelProperty::SetPropertyValueFromDevicePosition( MM::PropertyBase* Prop )
{
  bool vValueSet = false;
  unsigned int vPosition;
  if ( FilterWheelDevice_->GetPosition( vPosition ) )
  {
    if ( PositionNames_.find( vPosition ) != PositionNames_.end() )
    {
      Prop->Set( PositionNames_[vPosition].c_str() );
      vValueSet = true;
    }
    else
    {
      MMDragonfly_->LogComponentMessage( "Current " + ComponentName_ + " position invalid [ " + to_string(vPosition) + " ]" );
    }
  }
  else
  {
    MMDragonfly_->LogComponentMessage( "Failed to read the current " + ComponentName_ + " position" );
  }

  return vValueSet;
}
