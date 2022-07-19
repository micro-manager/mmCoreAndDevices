#include "ASDWrapperLens.h"
#include "ASDSDKLock.h"
#include "ASDWrapperFilterSet.h"

CASDWrapperLens::CASDWrapperLens( ILensInterface* LensInterface )
  : LensInterface_( LensInterface ),
  FilterSetWrapper_( nullptr )
{
  if ( LensInterface_ == nullptr )
  {
    throw std::exception( "Invalid pointer to LensInterface" );
  }
}

CASDWrapperLens::~CASDWrapperLens()
{
  delete FilterSetWrapper_;
}

///////////////////////////////////////////////////////////////////////////////
// ILensInterface
///////////////////////////////////////////////////////////////////////////////

bool CASDWrapperLens::GetPosition( unsigned int& Position )
{
  CASDSDKLock vSDKLock;
  return LensInterface_->GetPosition( Position );
}

bool CASDWrapperLens::SetPosition( unsigned int Position )
{
  CASDSDKLock vSDKLock;
  return LensInterface_->SetPosition( Position );
}

bool CASDWrapperLens::GetLimits( unsigned int& MinPosition, unsigned int& MaxPosition )
{
  CASDSDKLock vSDKLock;
  return LensInterface_->GetLimits( MinPosition, MaxPosition );
}

IFilterSet* CASDWrapperLens::GetLensConfigInterface()
{
  if ( FilterSetWrapper_ == nullptr )
  {
    CASDSDKLock vSDKLock;
    FilterSetWrapper_ = new CASDWrapperFilterSet( LensInterface_->GetLensConfigInterface() );
  }
  return FilterSetWrapper_;
}
