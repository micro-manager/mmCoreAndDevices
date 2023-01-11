#include "ASDWrapperFilterWheelMode.h"
#include "ASDSDKLock.h"


CASDWrapperFilterWheelMode::CASDWrapperFilterWheelMode( IFilterWheelModeInterface* FilterWheelModeInterface )
  : FilterWheelModeInterface_( FilterWheelModeInterface )
{
  if ( FilterWheelModeInterface_ == nullptr )
  {
    throw std::exception( "Invalid pointer to FilterWheelModeInterface" );
  }
}

CASDWrapperFilterWheelMode::~CASDWrapperFilterWheelMode()
{
}

///////////////////////////////////////////////////////////////////////////////
// IFilterWheelModeInterface
///////////////////////////////////////////////////////////////////////////////

bool CASDWrapperFilterWheelMode::GetMode( TFilterWheelMode& FilterWheelMode )
{
  CASDSDKLock vSDKLock;
  return FilterWheelModeInterface_->GetMode( FilterWheelMode );
}

bool CASDWrapperFilterWheelMode::SetMode( TFilterWheelMode FilterWheelMode )
{
  CASDSDKLock vSDKLock;
  return FilterWheelModeInterface_->SetMode( FilterWheelMode );
}
