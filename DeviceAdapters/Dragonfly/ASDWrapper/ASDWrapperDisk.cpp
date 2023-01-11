#include "ASDWrapperDisk.h"
#include "ASDSDKLock.h"


CASDWrapperDisk::CASDWrapperDisk( IDiskInterface2* DiskInterface )
  : DiskInterface_( DiskInterface )
{
  if ( DiskInterface_ == nullptr )
  {
    throw std::exception( "Invalid pointer to DiskInterface" );
  }
}

CASDWrapperDisk::~CASDWrapperDisk()
{
}

///////////////////////////////////////////////////////////////////////////////
// IDiskInterface
///////////////////////////////////////////////////////////////////////////////

bool CASDWrapperDisk::GetSpeed( unsigned int &Speed )
{
  CASDSDKLock vSDKLock;
  return DiskInterface_->GetSpeed( Speed );
}

bool CASDWrapperDisk::SetSpeed( unsigned int Speed )
{
  CASDSDKLock vSDKLock;
  return DiskInterface_->SetSpeed( Speed );
}

bool CASDWrapperDisk::IncreaseSpeed()
{
  CASDSDKLock vSDKLock;
  return DiskInterface_->IncreaseSpeed();
}

bool CASDWrapperDisk::DecreaseSpeed()
{
  CASDSDKLock vSDKLock;
  return DiskInterface_->DecreaseSpeed();
}

bool CASDWrapperDisk::GetLimits( unsigned int &Min, unsigned int &Max )
{
  CASDSDKLock vSDKLock;
  return DiskInterface_->GetLimits( Min, Max );
}

bool CASDWrapperDisk::Start()
{
  CASDSDKLock vSDKLock;
  return DiskInterface_->Start();
}

bool CASDWrapperDisk::Stop()
{
  CASDSDKLock vSDKLock;
  return DiskInterface_->Stop();
}

bool CASDWrapperDisk::IsSpinning()
{
  CASDSDKLock vSDKLock;
  return DiskInterface_->IsSpinning();
}

///////////////////////////////////////////////////////////////////////////////
// IDiskInterface2
///////////////////////////////////////////////////////////////////////////////

bool CASDWrapperDisk::GetScansPerRevolution( unsigned int *NumberOfScans )
{
  CASDSDKLock vSDKLock;
  return DiskInterface_->GetScansPerRevolution( NumberOfScans );
}

