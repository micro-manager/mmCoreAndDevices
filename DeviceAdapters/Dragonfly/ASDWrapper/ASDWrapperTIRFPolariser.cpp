#include "ASDWrapperTIRFPolariser.h"
#include "ASDSDKLock.h"

CASDWrapperTIRFPolariser::CASDWrapperTIRFPolariser( ITIRFPolariserInterface* TIRFPolariserInterface )
  : TIRFPolariserInterface_( TIRFPolariserInterface )
{
  if ( TIRFPolariserInterface_ == nullptr )
  {
    throw std::exception( "Invalid pointer to TIRFPolariserInterface" );
  }
}

CASDWrapperTIRFPolariser::~CASDWrapperTIRFPolariser()
{
}

///////////////////////////////////////////////////////////////////////////////
// ITIRFPolariserInterface
///////////////////////////////////////////////////////////////////////////////

bool CASDWrapperTIRFPolariser::GetPosition( unsigned int& Position )
{
  CASDSDKLock vSDKLock;
  return TIRFPolariserInterface_->GetPosition( Position );
}

bool CASDWrapperTIRFPolariser::SetPosition( unsigned int Position )
{
  CASDSDKLock vSDKLock;
  return TIRFPolariserInterface_->SetPosition( Position );
}

bool CASDWrapperTIRFPolariser::GetLimits( unsigned int& MinPosition, unsigned int& MaxPosition )
{
  CASDSDKLock vSDKLock;
  return TIRFPolariserInterface_->GetLimits( MinPosition, MaxPosition );
}
