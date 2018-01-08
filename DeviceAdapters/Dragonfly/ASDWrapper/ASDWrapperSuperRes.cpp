#include "ASDWrapperSuperRes.h"
#include "ASDSDKLock.h"

CASDWrapperSuperRes::CASDWrapperSuperRes( ISuperResInterface* SuperResInterface )
  : SuperResInterface_( SuperResInterface )
{
  if ( SuperResInterface_ == nullptr )
  {
    throw std::exception( "Invalid pointer to SuperResInterface" );
  }
}

CASDWrapperSuperRes::~CASDWrapperSuperRes()
{
}

///////////////////////////////////////////////////////////////////////////////
// ISuperResInterface
///////////////////////////////////////////////////////////////////////////////

bool CASDWrapperSuperRes::GetPosition( unsigned int& Position )
{
  CASDSDKLock vSDKLock;
  return SuperResInterface_->GetPosition( Position );
}

bool CASDWrapperSuperRes::SetPosition( unsigned int Position )
{
  CASDSDKLock vSDKLock;
  return SuperResInterface_->SetPosition( Position );
}

bool CASDWrapperSuperRes::GetLimits( unsigned int& MinPosition, unsigned int& MaxPosition )
{
  CASDSDKLock vSDKLock;
  return SuperResInterface_->GetLimits( MinPosition, MaxPosition );
}
