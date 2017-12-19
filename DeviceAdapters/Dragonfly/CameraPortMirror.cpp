#include "CameraPortMirror.h"
#include "Dragonfly.h"
#include "ASDConfigInterface.h"
#include "ComponentInterface.h"

#include <stdexcept>
#include <map>

using namespace std;

const char* const g_CameraPortMirrorPropertyName = "Camera Port Mirror";

CCameraPortMirror::CCameraPortMirror( ICameraPortMirrorInterface* CameraPortMirrorInterface, CDragonfly* MMDragonfly )
  : CameraPortMirrorInterface_( CameraPortMirrorInterface ),
  MMDragonfly_( MMDragonfly )
{
  // Retrieve values from the device
  if ( !RetrievePositionsFromFilterSet() )
  {
    if ( !RetrievePositionsWithoutDescriptions() )
    {
      throw std::runtime_error( "Failed to retrieve Camera port mirror positions" );
    }
  }

  // Retrieve the current position from the device
  unsigned int vPosition;
  if ( !CameraPortMirrorInterface_->GetPosition( vPosition ) )
  {
    throw std::runtime_error( "Failed to read the current Camera port mirror position" );
  }

  // Create the MM property
  CPropertyAction* vAct = new CPropertyAction( this, &CCameraPortMirror::OnPositionChange );
  MMDragonfly_->CreateProperty( g_CameraPortMirrorPropertyName, "Undefined", MM::String, false, vAct );

  // Populate the possible positions
  TPositionNameMap::const_iterator vIt = PositionNames_.begin();
  while ( vIt != PositionNames_.end() )
  {
    MMDragonfly_->AddAllowedValue( g_CameraPortMirrorPropertyName, vIt->second.c_str() );
    vIt++;
  }

  // Initialise the position
  if ( PositionNames_.find( vPosition ) != PositionNames_.end() )
  {
    MMDragonfly_->SetProperty( g_CameraPortMirrorPropertyName, PositionNames_[vPosition].c_str() );
  }
  else
  {
    MMDragonfly_->LogComponentMessage( "Current Camera port mirror position invalid" );
  }
}

CCameraPortMirror::~CCameraPortMirror()
{

}

bool CCameraPortMirror::RetrievePositionsFromFilterSet()
{
  const static unsigned int vStringLength = 64;
  bool vPositionsRetrieved = false;
  IFilterSet* vFilterSet = CameraPortMirrorInterface_->GetCameraPortMirrorConfigInterface();
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
  return vPositionsRetrieved;
}

bool CCameraPortMirror::RetrievePositionsWithoutDescriptions()
{
  bool vPositionsRetrieved = false;
  unsigned int vMinValue, vMaxValue;
  if ( CameraPortMirrorInterface_->GetLimits( vMinValue, vMaxValue ) )
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

int CCameraPortMirror::OnPositionChange( MM::PropertyBase* Prop, MM::ActionType Act )
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
      CameraPortMirrorInterface_->SetPosition( vIt->first );
    }
    else
    {
      // Reset position displayed in the UI to the current device position
      MMDragonfly_->LogComponentMessage( "Unknown Camera port mirror position requested [" + vRequestedPosition + "]. Ignoring request." );
      SetPropertyValueFromDevicePosition( Prop );
    }
  }
  return DEVICE_OK;
}

bool CCameraPortMirror::SetPropertyValueFromDevicePosition( MM::PropertyBase* Prop )
{
  bool vValueSet = false;
  unsigned int vPosition;
  if ( CameraPortMirrorInterface_->GetPosition( vPosition ) )
  {
    if ( PositionNames_.find( vPosition ) != PositionNames_.end() )
    {
      Prop->Set( PositionNames_[vPosition].c_str() );
      vValueSet = true;
    }
    else
    {
      MMDragonfly_->LogComponentMessage( "Current Camera port mirror position invalid [ " + to_string(vPosition) + " ]" );
    }
  }
  else
  {
    MMDragonfly_->LogComponentMessage( "Failed to read the current Camera port mirror position" );
  }

  return vValueSet;
}
