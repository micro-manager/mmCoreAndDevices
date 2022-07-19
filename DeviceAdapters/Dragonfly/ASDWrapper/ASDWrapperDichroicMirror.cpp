#include "ASDWrapperDichroicMirror.h"
#include "ASDSDKLock.h"
#include "ASDWrapperFilterConfig.h"

CASDWrapperDichroicMirror::CASDWrapperDichroicMirror( IDichroicMirrorInterface* DichroicMirrorInterface )
  : DichroicMirrorInterface_( DichroicMirrorInterface ),
  FilterConfigWrapper_( nullptr )
{
  if ( DichroicMirrorInterface_ == nullptr )
  {
    throw std::exception( "Invalid pointer to DichroicMirrorInterface" );
  }
}

CASDWrapperDichroicMirror::~CASDWrapperDichroicMirror()
{
  delete FilterConfigWrapper_;
}

///////////////////////////////////////////////////////////////////////////////
// IDichroicMirrorInterface
///////////////////////////////////////////////////////////////////////////////

bool CASDWrapperDichroicMirror::GetPosition( unsigned int& Position )
{
  CASDSDKLock vSDKLock;
  return DichroicMirrorInterface_->GetPosition( Position );
}

bool CASDWrapperDichroicMirror::SetPosition( unsigned int Position )
{
  CASDSDKLock vSDKLock;
  return DichroicMirrorInterface_->SetPosition( Position );
}

bool CASDWrapperDichroicMirror::GetLimits( unsigned int& MinPosition, unsigned int& MaxPosition )
{
  CASDSDKLock vSDKLock;
  return DichroicMirrorInterface_->GetLimits( MinPosition, MaxPosition );
}

IFilterConfigInterface* CASDWrapperDichroicMirror::GetFilterConfigInterface()
{
  if ( FilterConfigWrapper_ == nullptr )
  {
    CASDSDKLock vSDKLock;
    FilterConfigWrapper_ = new CASDWrapperFilterConfig( DichroicMirrorInterface_->GetFilterConfigInterface() );
  }
  return FilterConfigWrapper_;
}
