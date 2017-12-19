#include "ASDWrapperCameraPortMirror.h"
#include "ASDSDKLock.h"
#include "ASDWrapperFilterSet.h"

CASDWrapperCameraPortMirror::CASDWrapperCameraPortMirror( ICameraPortMirrorInterface* CameraPortMirrorInterface )
  : CameraPortMirrorInterface_( CameraPortMirrorInterface ),
  FilterSetWrapper_( nullptr )
{
  if ( CameraPortMirrorInterface_ == nullptr )
  {
    throw std::exception( "Invalid pointer to CameraPortMirrorInterface" );
  }
}

CASDWrapperCameraPortMirror::~CASDWrapperCameraPortMirror()
{
  delete FilterSetWrapper_;
}

///////////////////////////////////////////////////////////////////////////////
// ICameraPortMirrorInterface
///////////////////////////////////////////////////////////////////////////////

bool CASDWrapperCameraPortMirror::GetPosition( unsigned int& Position )
{
  CASDSDKLock vSDKLock;
  return CameraPortMirrorInterface_->GetPosition( Position );
}

bool CASDWrapperCameraPortMirror::SetPosition( unsigned int Position )
{
  CASDSDKLock vSDKLock;
  return CameraPortMirrorInterface_->SetPosition( Position );
}

bool CASDWrapperCameraPortMirror::GetLimits( unsigned int& MinPosition, unsigned int& MaxPosition )
{
  CASDSDKLock vSDKLock;
  return CameraPortMirrorInterface_->GetLimits( MinPosition, MaxPosition );
}

bool CASDWrapperCameraPortMirror::IsSplitFieldMirrorPresent()
{
  CASDSDKLock vSDKLock;
  return CameraPortMirrorInterface_->IsSplitFieldMirrorPresent();
}

IFilterSet* CASDWrapperCameraPortMirror::GetCameraPortMirrorConfigInterface()
{
  if ( FilterSetWrapper_ == nullptr )
  {
    CASDSDKLock vSDKLock;
    FilterSetWrapper_ = new CASDWrapperFilterSet( CameraPortMirrorInterface_->GetCameraPortMirrorConfigInterface() );
  }
  return FilterSetWrapper_;
}
