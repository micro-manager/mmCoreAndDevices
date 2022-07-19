#include "ASDWrapperStatus.h"
#include "ASDSDKLock.h"


CASDWrapperStatus::CASDWrapperStatus( IStatusInterface* StatusInterface )
  : StatusInterface_( StatusInterface )
{
  if ( StatusInterface_ == nullptr )
  {
    throw std::exception( "Invalid pointer to StatusInterface" );
  }
}

CASDWrapperStatus::~CASDWrapperStatus()
{
}

///////////////////////////////////////////////////////////////////////////////
// IStatusInterface
///////////////////////////////////////////////////////////////////////////////

bool CASDWrapperStatus::GetStatusCode( unsigned int *Status )
{
  CASDSDKLock vSDKLock;
  return StatusInterface_->GetStatusCode( Status );
}

bool CASDWrapperStatus::IsStandbyActive()
{
  CASDSDKLock vSDKLock;
  return StatusInterface_->IsStandbyActive();
}

bool CASDWrapperStatus::ActivateStandby()
{
  CASDSDKLock vSDKLock;
  return StatusInterface_->ActivateStandby();
}

bool CASDWrapperStatus::WakeFromStandby()
{
  CASDSDKLock vSDKLock;
  return StatusInterface_->WakeFromStandby();
}

