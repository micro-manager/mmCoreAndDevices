#include "ASDWrapperFilterConfig.h"
#include "ASDWrapperFilterSet.h"
#include "ASDSDKLock.h"



CASDWrapperFilterConfig::CASDWrapperFilterConfig( IFilterConfigInterface* FilterConfigInterface )
  : FilterConfigInterface_( FilterConfigInterface ),
  FilterSetWrapper_( nullptr )
{
  if ( FilterConfigInterface_ == nullptr )
  {
    throw std::exception( "Invalid pointer to FilterConfigInterface" );
  }
}

CASDWrapperFilterConfig::~CASDWrapperFilterConfig()
{
  delete FilterSetWrapper_;
}

///////////////////////////////////////////////////////////////////////////////
// IFilterConfigInterface
///////////////////////////////////////////////////////////////////////////////

IFilterSet* CASDWrapperFilterConfig::GetFilterSet()
{
  if ( FilterSetWrapper_ == nullptr )
  {
    CASDSDKLock vSDKLock;
    FilterSetWrapper_ = new CASDWrapperFilterSet( FilterConfigInterface_->GetFilterSet() );
  }
  return FilterSetWrapper_;
}

bool CASDWrapperFilterConfig::GetPositionOfFilterSetInRepository( unsigned int *Position )
{
  CASDSDKLock vSDKLock;
  return FilterConfigInterface_->GetPositionOfFilterSetInRepository( Position );
}

bool CASDWrapperFilterConfig::ExchangeFilterSet( unsigned int Position )
{
  CASDSDKLock vSDKLock;
  return FilterConfigInterface_->ExchangeFilterSet( Position );
}

IFilterRepository* CASDWrapperFilterConfig::GetFilterRepository()
{
  throw std::logic_error( "GetFilterRepository() wrapper function not implemented" );
}

