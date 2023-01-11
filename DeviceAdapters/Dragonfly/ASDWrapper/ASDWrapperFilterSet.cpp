#include "ASDWrapperFilterSet.h"
#include "ASDSDKLock.h"



CASDWrapperFilterSet::CASDWrapperFilterSet( IFilterSet* FilterSetInterface )
  : FilterSetInterface_( FilterSetInterface )
{
  if ( FilterSetInterface_ == nullptr )
  {
    throw std::exception( "Invalid pointer to FilterSetInterface" );
  }
}

CASDWrapperFilterSet::~CASDWrapperFilterSet()
{
}

///////////////////////////////////////////////////////////////////////////////
// IFilterSet
///////////////////////////////////////////////////////////////////////////////

bool CASDWrapperFilterSet::GetDescription( char *Description, unsigned int StringLength )
{
  CASDSDKLock vSDKLock;
  return FilterSetInterface_->GetDescription( Description, StringLength );
}

bool CASDWrapperFilterSet::GetFilterDescription( unsigned int Position, char *Description, unsigned int StringLength )
{
  CASDSDKLock vSDKLock;
  return FilterSetInterface_->GetFilterDescription( Position, Description, StringLength );
}

bool CASDWrapperFilterSet::GetLimits( unsigned int &MinPosition, unsigned int &MaxPosition )
{
  CASDSDKLock vSDKLock;
  return FilterSetInterface_->GetLimits( MinPosition, MaxPosition );
}

