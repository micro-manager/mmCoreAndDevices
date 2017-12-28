#include "CameraPortMirror.h"

#include "ASDInterface.h"

CCameraPortMirror::CCameraPortMirror( ICameraPortMirrorInterface* CameraPortMirrorInterface, CDragonfly* MMDragonfly )
  : IPositionComponentInterface( MMDragonfly, "Camera Port Mirror" ),
  CameraPortMirrorInterface_( CameraPortMirrorInterface ),
  MMDragonfly_( MMDragonfly )
{
  Initialise();
}

CCameraPortMirror::~CCameraPortMirror()
{
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