#include "ASDWrapperFilterWheel.h"
#include "ASDSDKLock.h"
#include "ASDWrapperFilterConfig.h"
#include "ASDWrapperFilterWheelMode.h"

CASDWrapperFilterWheel::CASDWrapperFilterWheel( IFilterWheelInterface* FilterWheelInterface )
  : FilterWheelInterface_( FilterWheelInterface ),
  FilterConfigWrapper_( nullptr ),
  FilterWheelModeWrapper_( nullptr )
{
  if ( FilterWheelInterface_ == nullptr )
  {
    throw std::exception( "Invalid pointer to FilterWheelInterface" );
  }
}

CASDWrapperFilterWheel::~CASDWrapperFilterWheel()
{
  delete FilterConfigWrapper_;
  delete FilterWheelModeWrapper_;
}

///////////////////////////////////////////////////////////////////////////////
// IFilterWheelInterface
///////////////////////////////////////////////////////////////////////////////

bool CASDWrapperFilterWheel::GetPosition( unsigned int& Position )
{
  CASDSDKLock vSDKLock;
  return FilterWheelInterface_->GetPosition( Position );
}

bool CASDWrapperFilterWheel::SetPosition( unsigned int Position )
{
  CASDSDKLock vSDKLock;
  return FilterWheelInterface_->SetPosition( Position );
}

bool CASDWrapperFilterWheel::GetLimits( unsigned int& MinPosition, unsigned int& MaxPosition )
{
  CASDSDKLock vSDKLock;
  return FilterWheelInterface_->GetLimits( MinPosition, MaxPosition );
}

IFilterWheelSpeedInterface* CASDWrapperFilterWheel::GetFilterWheelSpeedInterface()
{
  throw std::logic_error( "GetFilterWheelSpeedInterface() wrapper function not implemented" );
}

IFilterConfigInterface* CASDWrapperFilterWheel::GetFilterConfigInterface()
{
  if ( FilterConfigWrapper_ == nullptr )
  {
    CASDSDKLock vSDKLock;
    FilterConfigWrapper_ = new CASDWrapperFilterConfig( FilterWheelInterface_->GetFilterConfigInterface() );
  }
  return FilterConfigWrapper_;
}

IFilterWheelModeInterface* CASDWrapperFilterWheel::GetFilterWheelModeInterface()
{
  if ( FilterWheelModeWrapper_ == nullptr )
  {
    CASDSDKLock vSDKLock;
    FilterWheelModeWrapper_ = new CASDWrapperFilterWheelMode( FilterWheelInterface_->GetFilterWheelModeInterface() );
  }
  return FilterWheelModeWrapper_;
}
