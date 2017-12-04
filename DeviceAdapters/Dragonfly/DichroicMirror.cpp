#include "DichroicMirror.h"
#include "Dragonfly.h"

#include "ASDInterface.h"
#include "ASDConfigInterface.h"

#include <stdexcept>
#include <map>

using namespace std;

const char * const g_DichroicMirrorPosition = "Dichroic Mirror";

CDichroicMirror::CDichroicMirror( IDichroicMirrorInterface* DichroicMirrorInterface, CDragonfly* MMDragonfly )
  : DichroicMirrorInterface_( DichroicMirrorInterface ),
  MMDragonfly_( MMDragonfly )
{
  // Retrieve values from the device
  if( !RetrievePositionsFromFilterConfig() )
  {
    if ( !RetrievePositionsWithoutDescriptions() )
    {
      throw std::runtime_error( "Failed to retrieve Dichroic mirror positions" );
    }
  }

  // Retrieve the current position from the device
  unsigned int vPosition;
  if ( !DichroicMirrorInterface_->GetPosition( vPosition ) )
  {
    throw std::runtime_error( "Failed to read the current Dichroic mirror position" );
  }

  // Create the MM property
  CPropertyAction* vAct = new CPropertyAction( this, &CDichroicMirror::OnPositionChange );
  MMDragonfly_->CreateProperty( g_DichroicMirrorPosition, "Undefined", MM::String, false, vAct );

  // Populate the possible positions
  TPositionNameMap::const_iterator vIt = PositionNames_.begin();
  while ( vIt != PositionNames_.end() )
  {
    MMDragonfly_->AddAllowedValue( g_DichroicMirrorPosition, vIt->second.c_str() );
    vIt++;
  }

  // Initialise the position
  if ( PositionNames_.find( vPosition ) != PositionNames_.end() )
  {
    MMDragonfly_->SetProperty( g_DichroicMirrorPosition, PositionNames_[vPosition].c_str() );
  }
  else
  {
    MMDragonfly_->LogComponentMessage( "Current Dichroic mirror position invalid" );
  }
}

CDichroicMirror::~CDichroicMirror()
{

}

bool CDichroicMirror::RetrievePositionsFromFilterConfig()
{
  const static unsigned int vStringLength = 64;
  bool vPositionsRetrieved = false;
  IFilterConfigInterface* vFilterConfigInterface = DichroicMirrorInterface_->GetFilterConfigInterface();
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
          string vPositionName(to_string( vIndex ) + " - ");
          if ( vFilterSet->GetFilterDescription( vIndex, vDescription, vStringLength ) == false )
          {
            vPositionName += "Empty";
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
  }
  return vPositionsRetrieved;
}

bool CDichroicMirror::RetrievePositionsWithoutDescriptions()
{
  bool vPositionsRetrieved = false;
  unsigned int vMinValue, vMaxValue;
  if ( DichroicMirrorInterface_->GetLimits( vMinValue, vMaxValue ) )
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


int CDichroicMirror::OnPositionChange( MM::PropertyBase* Prop, MM::ActionType Act )
{
  if ( Act == MM::BeforeGet )
  {
    SetPropertyValueFromDevicePosition( Prop );
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
      DichroicMirrorInterface_->SetPosition( vIt->first );
    }
    else
    {
      // Reset position displayed in the UI to the current device position
      MMDragonfly_->LogComponentMessage( "Unknown Dichroic mirror position requested [" + vRequestedPosition + "]. Ignoring request." );
      SetPropertyValueFromDevicePosition( Prop );
    }
  }
  return DEVICE_OK;
}

bool CDichroicMirror::SetPropertyValueFromDevicePosition( MM::PropertyBase* Prop )
{
  bool vValueSet = false;
  unsigned int vPosition;
  if ( DichroicMirrorInterface_->GetPosition( vPosition ) )
  {
    if ( PositionNames_.find( vPosition ) != PositionNames_.end() )
    {
      Prop->Set( PositionNames_[vPosition].c_str() );
      vValueSet = true;
    }
    else
    {
      MMDragonfly_->LogComponentMessage( "Current Dichroic mirror position invalid" );
    }
  }
  else
  {
    MMDragonfly_->LogComponentMessage( "Failed to read the current Dichroic mirror position" );
  }

  return vValueSet;
}