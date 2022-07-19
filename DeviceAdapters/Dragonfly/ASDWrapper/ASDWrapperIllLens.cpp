#include "ASDWrapperIllLens.h"
#include "ASDSDKLock.h"
#include "ASDWrapperFilterSet.h"


CASDWrapperIllLens::CASDWrapperIllLens( IIllLensInterface* IllLensInterface )
  : IllLensInterface_( IllLensInterface ),
  FilterSetWrapper_( nullptr )
{
  if ( IllLensInterface_ == nullptr )
  {
    throw std::exception( "Invalid pointer to IllLensInterface" );
  }
}

CASDWrapperIllLens::~CASDWrapperIllLens()
{
  delete FilterSetWrapper_;
}

///////////////////////////////////////////////////////////////////////////////
// IIllLensInterface
///////////////////////////////////////////////////////////////////////////////

bool CASDWrapperIllLens::GetPosition( unsigned int& Position )
{
  CASDSDKLock vSDKLock;
  return IllLensInterface_->GetPosition( Position );
}

bool CASDWrapperIllLens::SetPosition( unsigned int Position )
{
  CASDSDKLock vSDKLock;
  return IllLensInterface_->SetPosition( Position );
}

bool CASDWrapperIllLens::GetLimits( unsigned int& MinPosition, unsigned int& MaxPosition )
{
  CASDSDKLock vSDKLock;
  return IllLensInterface_->GetLimits( MinPosition, MaxPosition );
}

bool CASDWrapperIllLens::IsRestrictionEnabled()
{
  CASDSDKLock vSDKLock;
  return IllLensInterface_->IsRestrictionEnabled();
}

bool CASDWrapperIllLens::GetRestrictedRange( unsigned int &MinPosition, unsigned int &MaxPosition )
{
  CASDSDKLock vSDKLock;
  return IllLensInterface_->GetRestrictedRange( MinPosition, MaxPosition );
}

bool CASDWrapperIllLens::RegisterForNotificationOnRangeRestriction( INotify *Notify )
{
  CASDSDKLock vSDKLock;
  return IllLensInterface_->RegisterForNotificationOnRangeRestriction( Notify );
}

IFilterSet* CASDWrapperIllLens::GetLensConfigInterface()
{
  if ( FilterSetWrapper_ == nullptr )
  {
    CASDSDKLock vSDKLock;
    FilterSetWrapper_ = new CASDWrapperFilterSet( IllLensInterface_->GetLensConfigInterface() );
  }
  return FilterSetWrapper_;
}
