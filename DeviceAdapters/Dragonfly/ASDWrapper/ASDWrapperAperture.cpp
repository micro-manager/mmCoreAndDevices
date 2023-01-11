#include "ASDWrapperAperture.h"
#include "ASDSDKLock.h"
#include "ASDWrapperFilterSet.h"

CASDWrapperAperture::CASDWrapperAperture( IApertureInterface* ApertureInterface )
  : ApertureInterface_( ApertureInterface ),
  FilterSetWrapper_( nullptr )
{
  if ( ApertureInterface_ == nullptr )
  {
    throw std::exception( "Invalid pointer to ApertureInterface" );
  }
}

CASDWrapperAperture::~CASDWrapperAperture()
{
  delete FilterSetWrapper_;
}

///////////////////////////////////////////////////////////////////////////////
// IApertureInterface
///////////////////////////////////////////////////////////////////////////////

bool CASDWrapperAperture::GetPosition( unsigned int& Position )
{
  CASDSDKLock vSDKLock;
  return ApertureInterface_->GetPosition( Position );
}

bool CASDWrapperAperture::SetPosition( unsigned int Position )
{
  CASDSDKLock vSDKLock;
  return ApertureInterface_->SetPosition( Position );
}

bool CASDWrapperAperture::GetLimits( unsigned int& MinPosition, unsigned int& MaxPosition )
{
  CASDSDKLock vSDKLock;
  return ApertureInterface_->GetLimits( MinPosition, MaxPosition );
}

bool CASDWrapperAperture::IsSplitFieldAperturePresent()
{
  CASDSDKLock vSDKLock;
  return ApertureInterface_->IsSplitFieldAperturePresent();
}

IFilterSet* CASDWrapperAperture::GetApertureConfigInterface()
{
  if ( FilterSetWrapper_ == nullptr )
  {
    CASDSDKLock vSDKLock;
    FilterSetWrapper_ = new CASDWrapperFilterSet( ApertureInterface_->GetApertureConfigInterface() );
  }
  return FilterSetWrapper_;
}
