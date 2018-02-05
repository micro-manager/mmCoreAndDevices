#include "DiskStatus.h"

#include "ComponentInterface.h"
#include "Dragonfly.h"
#include "DiskSpeedSimulator.h"
#include "DiskSpeedState.h"

using namespace std;

IDiskStatus* CreateDiskStatus( IDiskInterface2* DiskInterface, CDragonfly* MMDragonfly, CDiskSimulator* DiskSimulator )
{
  return new CDiskStatus( DiskInterface, MMDragonfly, DiskSimulator );
}

CDiskStatus::CDiskStatus( IDiskInterface2* DiskSpeedInterface, CDragonfly* MMDragonfly, CDiskSimulator* DiskSimulator )
  : DiskInterface_( DiskSpeedInterface ),
  MMDragonfly_( MMDragonfly ),
  RequestedSpeed_( 0 ),
  CurrentSpeed_( 0 ),
  ChangingSpeedState_( new CChangingSpeedState( this ) ),
  AtSpeedState_( new CAtSpeedState( this ) ),
  StoppingState_( new CStoppingState( this ) ),
  StoppedState_( new CStoppedState( this ) ),
  CurrentState_( nullptr ),
  DiskSimulator_( DiskSimulator )
{
  SetState( StoppedState_ );
}

CDiskStatus::~CDiskStatus()
{
  delete ChangingSpeedState_;
  delete AtSpeedState_;
  delete StoppingState_;
  delete StoppedState_;
}

///////////////////////////////////////////////////////////////////////////////
// Inherited from IDiskStatus
///////////////////////////////////////////////////////////////////////////////

void CDiskStatus::RegisterObserver( CDiskStateChange* Observer )
{
  Observers_.push_back( Observer );
}

void CDiskStatus::UnregisterObserver( CDiskStateChange* Observer )
{
  Observers_.remove( Observer );
}

void CDiskStatus::RegisterErrorObserver( CDiskStateError* Observer )
{
  ErrorObservers_.push_back( Observer );
}

void CDiskStatus::UnregisterErrorObserver( CDiskStateError* Observer )
{
  ErrorObservers_.remove( Observer );
}

void CDiskStatus::Start()
{
  CurrentState_->Start();
}

void CDiskStatus::ChangeSpeed( unsigned int NewRequestedSpeed )
{
  RequestedSpeed_ = NewRequestedSpeed;
  CurrentState_->ChangeSpeed();
}

void CDiskStatus::Stop()
{
  CurrentState_->Stop();
}

void CDiskStatus::UpdateFromDevice()
{
  CurrentState_->UpdateFromDevice();
}

bool CDiskStatus::IsChangingSpeed() const
{
  return CurrentState_ == ChangingSpeedState_;
}

bool CDiskStatus::IsAtSpeed() const
{
  return CurrentState_ == AtSpeedState_;
}

bool CDiskStatus::IsStopping() const
{
  return CurrentState_ == StoppingState_;
}

bool CDiskStatus::IsStopped() const
{
  return CurrentState_ == StoppedState_;
}

unsigned int CDiskStatus::GetCurrentSpeed() const
{
  return CurrentSpeed_;
}

unsigned int CDiskStatus::GetRequestedSpeed() const
{
  return RequestedSpeed_;
}

void  CDiskStatus::ErrorEncountered( const string& ErrorMessage )
{
  NotifyStateError( ErrorMessage );
}

///////////////////////////////////////////////////////////////////////////////
// CDiskStatus methods
///////////////////////////////////////////////////////////////////////////////

bool CDiskStatus::ReadIsSpinningFromDevice() const
{
  return DiskInterface_->IsSpinning();
}

unsigned int CDiskStatus::ReadCurrentSpeedFromDevice()
{
  //DiskSimulator_->GetSpeed( CurrentSpeed_ );
  DiskInterface_->GetSpeed( CurrentSpeed_ );
  return CurrentSpeed_;
}

CDiskSpeedState* CDiskStatus::GetChangingSpeedState()
{
  return ChangingSpeedState_;
}

CDiskSpeedState* CDiskStatus::GetAtSpeedState()
{
  return AtSpeedState_;
}

CDiskSpeedState* CDiskStatus::GetStoppingState()
{
  return StoppingState_;
}

CDiskSpeedState* CDiskStatus::GetStoppedState()
{
  return StoppedState_;
}

void CDiskStatus::SetState( CDiskSpeedState* NewState )
{
  CurrentState_ = NewState;
  CurrentState_->Initialise();
  NotifyStateChange();
}

void CDiskStatus::NotifyStateChange()
{
  list<CDiskStateChange*>::iterator vObserverIt = Observers_.begin();
  while ( vObserverIt != Observers_.end() )
  {
    if ( *vObserverIt != nullptr )
    {
      ( *vObserverIt )->Notify();
    }
    vObserverIt++;
  }
}

void CDiskStatus::NotifyStateError( const string& ErrorMessage )
{
  list<CDiskStateError*>::iterator vObserverIt = ErrorObservers_.begin();
  while ( vObserverIt != ErrorObservers_.end() )
  {
    if ( *vObserverIt != nullptr )
    {
      ( *vObserverIt )->Notify( ErrorMessage );
    }
    vObserverIt++;
  }
}
