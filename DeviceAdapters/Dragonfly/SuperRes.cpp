#include "SuperRes.h"
#include "Dragonfly.h"

#include <stdexcept>
#include <map>

using namespace std;

const char* const g_SuperResPropertyName = "Super Resolution 3D";
const char* const g_SuperResInLightPath = "In light path";
const char* const g_SuperResNotInLightPath = "Not in light path";

CSuperRes::CSuperRes( ISuperResInterface* SuperResInterface, CDragonfly* MMDragonfly )
  : SuperResInterface_( SuperResInterface ),
  MMDragonfly_( MMDragonfly )
{
  // Retrieve values from the device
  if ( !RetrievePositions() )
  {
    throw runtime_error( "Failed to retrieve Super Resolution positions" );
  }

  // Retrieve the current position from the device
  unsigned int vPosition;
  if ( !SuperResInterface_->GetPosition( vPosition ) )
  {
    throw runtime_error( "Failed to read the current Super Resolution position" );
  }
  
  // Retrieve the current position name
  string vCurrentPositionName = "Undefined";
  if ( PositionNames_.find( vPosition ) != PositionNames_.end() )
  {
    vCurrentPositionName = PositionNames_[vPosition];
  }
  else
  {
    MMDragonfly_->LogComponentMessage( "Current Super Resolution position invalid" );
  }

  // Create the MM property
  CPropertyAction* vAct = new CPropertyAction( this, &CSuperRes::OnPositionChange );
  int vRet = MMDragonfly_->CreateProperty( g_SuperResPropertyName, vCurrentPositionName.c_str(), MM::String, false, vAct );
  if ( vRet != DEVICE_OK )
  {
    throw runtime_error( "Error creating " + string( g_SuperResPropertyName ) + " property" );
  }

  // Populate the possible positions
  TPositionNameMap::const_iterator vIt = PositionNames_.begin();
  while ( vIt != PositionNames_.end() )
  {
    MMDragonfly_->AddAllowedValue( g_SuperResPropertyName, vIt->second.c_str() );
    vIt++;
  }
}

CSuperRes::~CSuperRes()
{
}

bool CSuperRes::RetrievePositions()
{
  bool vPositionsRetrieved = false;
  unsigned int vMinValue, vMaxValue;
  if ( SuperResInterface_->GetLimits( vMinValue, vMaxValue ) )
  {
    for ( unsigned int vIndex = vMinValue; vIndex <= vMaxValue; vIndex++ )
    {
      string vPositionName;
      if ( vIndex == 1 )
      {
        vPositionName = g_SuperResInLightPath;
      }
      else if ( vIndex == 2 )
      {
        vPositionName = g_SuperResNotInLightPath;
      }
      else
      {
        vPositionName = "Undefined Position " + to_string( static_cast< long long >( vIndex ) );
      }

      PositionNames_[vIndex] = vPositionName;
    }

    vPositionsRetrieved = true;
  }
  return vPositionsRetrieved;
}

int CSuperRes::OnPositionChange( MM::PropertyBase* Prop, MM::ActionType Act )
{
  int vRet = DEVICE_OK;
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
      if ( !SuperResInterface_->SetPosition( vIt->first ) )
      {
        MMDragonfly_->LogComponentMessage( "Failed to set Super Resolution position [" + to_string( static_cast< long long >( vIt->first ) ) + "]" );
        vRet = DEVICE_CAN_NOT_SET_PROPERTY;
      }
    }
    else
    {
      // Reset position displayed in the UI to the current device position
      MMDragonfly_->LogComponentMessage( "Unknown Super Resolution position requested [" + vRequestedPosition + "]. Ignoring request." );
      SetPropertyValueFromDevicePosition( Prop );
      vRet = DEVICE_INVALID_PROPERTY_VALUE;
    }
  }
  return vRet;
}

int CSuperRes::SetPropertyValueFromDevicePosition( MM::PropertyBase* Prop )
{
  int vRet = DEVICE_ERR;
  unsigned int vPosition;
  if ( SuperResInterface_->GetPosition( vPosition ) )
  {
    if ( PositionNames_.find( vPosition ) != PositionNames_.end() )
    {
      Prop->Set( PositionNames_[vPosition].c_str() );
      vRet = DEVICE_OK;
    }
    else
    {
      MMDragonfly_->LogComponentMessage( "Current Super Resolution position invalid [ " + to_string( static_cast< long long >( vPosition ) ) + " ]" );
      vRet = DEVICE_UNKNOWN_POSITION;
    }
  }
  else
  {
    MMDragonfly_->LogComponentMessage( "Failed to read the current Super Resolution position" );
    vRet = DEVICE_ERR;
  }

  return vRet;
}
