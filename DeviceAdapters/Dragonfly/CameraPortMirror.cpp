#include "CameraPortMirror.h"
#include "Dragonfly.h"
#include "DragonflyStatus.h"

#include "ASDInterface.h"

const char* const g_RFIDStatusPropertyName = "Image Splitter RFID Status";

CCameraPortMirror::CCameraPortMirror( ICameraPortMirrorInterface* CameraPortMirrorInterface, const CDragonflyStatus* DragonflyStatus, CDragonfly* MMDragonfly )
  : IPositionComponentInterface( MMDragonfly, "Image Splitter", false ),
  CameraPortMirrorInterface_( CameraPortMirrorInterface ),
  MMDragonfly_( MMDragonfly ),
  DragonflyStatus_( DragonflyStatus )
{
  Initialise();
  // Create and initialise the RFID status property
  CreateRFIDStatusProperty();
}

CCameraPortMirror::~CCameraPortMirror()
{
}

void CCameraPortMirror::CreateRFIDStatusProperty()
{
  if ( DragonflyStatus_ != nullptr )
  {
    char vPropertyValue[32];
    if ( !DragonflyStatus_->IsRFIDPresentForCameraPortMirror() )
    {
      strncpy( vPropertyValue, "Not present", 32 );
    }
    else
    {
      if ( DragonflyStatus_->IsRFIDReadForCameraPortMirror() )
      {
        strncpy( vPropertyValue, "Present and Read", 32 );
      }
      else
      {
        strncpy( vPropertyValue, "Present but Read failed", 32 );
      }
    }
    int vRet = MMDragonfly_->CreateProperty( g_RFIDStatusPropertyName, vPropertyValue, MM::String, true );
    if ( vRet != DEVICE_OK )
    {
      MMDragonfly_->LogComponentMessage( "Error creating " + std::string(g_RFIDStatusPropertyName) +  " property" );
    }
  }
  else
  {
    throw std::logic_error( "Dragonfly status not initialised before " + PropertyName_ );
  }
}

bool CCameraPortMirror::GetPosition( unsigned int& Position )
{
  return CameraPortMirrorInterface_->GetPosition( Position );
}
bool CCameraPortMirror::SetPosition( unsigned int Position )
{
  return CameraPortMirrorInterface_->SetPosition( Position );
}
bool CCameraPortMirror::GetLimits( unsigned int& MinPosition, unsigned int& MaxPosition )
{
  return CameraPortMirrorInterface_->GetLimits( MinPosition, MaxPosition );
}
IFilterSet* CCameraPortMirror::GetFilterSet()
{
  return CameraPortMirrorInterface_->GetCameraPortMirrorConfigInterface();
}