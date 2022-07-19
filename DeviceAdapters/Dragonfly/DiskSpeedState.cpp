#include "DiskSpeedState.h"
#include "DiskStatus.h"
#include <algorithm>

using namespace std;

///////////////////////////////////////////////////////////////////////////////
// CDiskSpeedState
///////////////////////////////////////////////////////////////////////////////

CDiskSpeedState::CDiskSpeedState( CDiskStatus* DiskStatus )
  : DiskStatus_( DiskStatus )
{
}

CDiskSpeedState::~CDiskSpeedState()
{
}

///////////////////////////////////////////////////////////////////////////////
// CChangingSpeedState
///////////////////////////////////////////////////////////////////////////////

CChangingSpeedState::CChangingSpeedState( CDiskStatus* DiskStatus )
  : CDiskSpeedState( DiskStatus ),
  PreviousDiskStateUnknown_( true ),
  DiskSpeedIncreasing_( true ),
  MinSpeedReached_( 0 ),
  MaxSpeedReached_( 0 ),
  DiskSpeedNotChangingOnce_( false ),
  DiskSpeedNotChangingTwice_( false )
{ }

void CChangingSpeedState::Initialise()
{
  DiskStatus_->ReadCurrentSpeedFromDevice();
  PreviousDiskStateUnknown_ = true;
  MinSpeedReached_ = 0;
  MaxSpeedReached_ = DiskStatus_->GetRequestedSpeed() + 1000;
  DiskSpeedNotChangingOnce_ = false;
  DiskSpeedNotChangingTwice_ = false;
}

void CChangingSpeedState::Start()
{ }

void CChangingSpeedState::ChangeSpeed()
{ 
  // Change speed to the same state. This will notify observers and initialise the state again.
  DiskStatus_->SetState( DiskStatus_->GetChangingSpeedState() );
}

void CChangingSpeedState::Stop()
{
  // Move to Stopping state
  DiskStatus_->SetState( DiskStatus_->GetStoppingState() );
}

void CChangingSpeedState::UpdateFromDevice()
{
  // If speed is stable, move to At Speed state
  if ( IsSpeedStable() )
  {
    DiskStatus_->SetState( DiskStatus_->GetAtSpeedState() );
  }
}

void CChangingSpeedState::SpeedHasIncreased( unsigned int PreviousSpeed, unsigned int CurrentSpeed)
{
  DiskSpeedNotChangingOnce_ = false;
  DiskSpeedNotChangingTwice_ = false;
  if ( !PreviousDiskStateUnknown_ )
  {
    if ( !DiskSpeedIncreasing_ )
    {
      // The speed was previously decreasing, we may have reached a local min
      unsigned int vRequestedSpeed = DiskStatus_->GetRequestedSpeed();
      if ( CurrentSpeed > vRequestedSpeed )
      {
        // Unlikely case where we are above the requested speed
        if ( PreviousSpeed >= vRequestedSpeed )
        {
          // Boolean was badly initialised previously or a new requested value has been set (?)
        }
        else
        {
          // We might be so close to the requested speed that we are looping around it between 2 ticks (unlikely though)
          MinSpeedReached_ = PreviousSpeed;
        }
      }
      else
      {
        // We are below the requested speed therefore we reached a local minimum
        MinSpeedReached_ = PreviousSpeed;
      }
    }
    else
    {
      // Speed continues to increase, we do nothing
    }
  }
  else
  {
    PreviousDiskStateUnknown_ = false;
  }
  DiskSpeedIncreasing_ = true;
}

void CChangingSpeedState::SpeedHasDecreased( unsigned int PreviousSpeed, unsigned int CurrentSpeed )
{
  DiskSpeedNotChangingOnce_ = false;
  DiskSpeedNotChangingTwice_ = false;
  if ( !PreviousDiskStateUnknown_ )
  {
    if ( DiskSpeedIncreasing_ )
    {
      // The speed was previously increasing, we may have reached a local max
      unsigned int vRequestedSpeed = DiskStatus_->GetRequestedSpeed();
      if ( CurrentSpeed < vRequestedSpeed )
      {
        // Unlikely case where we are below the requested speed
        if ( PreviousSpeed <= vRequestedSpeed )
        {
          // Boolean was badly initialised previously
        }
        else
        {
          // We might be so close to the requested speed that we are looping around it between 2 ticks (unlikely though)
          MaxSpeedReached_ = PreviousSpeed;
        }
      }
      else
      {
        // We are above the requested speed therefore we reached a local maximum
        MaxSpeedReached_ = PreviousSpeed;
      }
    }
    else
    {
      // Speed continues to decrease, we do nothing
    }
  }
  else
  {
    PreviousDiskStateUnknown_ = false;
  }
  DiskSpeedIncreasing_ = false;
}

void CChangingSpeedState::SpeedUnchanged( unsigned int CurrentSpeed )
{
  if ( DiskSpeedNotChangingOnce_ )
  {
    // The speed hasn't changed in 2 ticks
    unsigned int vRequestedSpeed = DiskStatus_->GetRequestedSpeed();
    if ( CurrentSpeed == vRequestedSpeed )
    {
      // We reached the requested speed
      MinSpeedReached_ = vRequestedSpeed;
      MaxSpeedReached_ = vRequestedSpeed;
    }
    else
    {
      // The disk is not changing speed even though it should
      // Something's wrong, we report it to the user
      DiskSpeedNotChangingTwice_ = true;
      DiskStatus_->ErrorEncountered( "Error: Disk speed not changing" );
    }
  }
  DiskSpeedNotChangingOnce_ = true;
}

bool CChangingSpeedState::IsSpeedStable()
{
  unsigned int vPreviousSpeed = DiskStatus_->GetCurrentSpeed();
  unsigned int vCurrentSpeed = DiskStatus_->ReadCurrentSpeedFromDevice();

  if ( vCurrentSpeed > vPreviousSpeed )
  {
    SpeedHasIncreased( vPreviousSpeed, vCurrentSpeed );
  }
  else if ( vCurrentSpeed < vPreviousSpeed )
  {
    SpeedHasDecreased( vPreviousSpeed, vCurrentSpeed );
  }
  else if ( vCurrentSpeed == vPreviousSpeed )
  {
    SpeedUnchanged( vCurrentSpeed );
  }

  unsigned int vRequestedSpeed = DiskStatus_->GetRequestedSpeed();
  if ( MinSpeedReached_ >= GetTargetRangeMin( vRequestedSpeed ) && MaxSpeedReached_ <= GetTargetRangeMax( vRequestedSpeed ) )
  {
    return true;
  }
  return false;
}

#define _ABSOLUTE_SPEED_RANGE_
unsigned int CChangingSpeedState::GetTargetRangeMin(unsigned int RequestedSpeed) const
{
#ifdef _ABSOLUTE_SPEED_RANGE_
  return RequestedSpeed - 20;
#else
  return RequestedSpeed * ( 100 - DynamicRangePercent_ ) / 100;
#endif
}

unsigned int CChangingSpeedState::GetTargetRangeMax( unsigned int RequestedSpeed ) const
{
#ifdef _ABSOLUTE_SPEED_RANGE_
  return RequestedSpeed + 20;
#else
  return RequestedSpeed * ( 100 + DynamicRangePercent_ ) / 100;
#endif
}

///////////////////////////////////////////////////////////////////////////////
// CAtSpeedState
///////////////////////////////////////////////////////////////////////////////

CAtSpeedState::CAtSpeedState( CDiskStatus* DiskStatus )
  : CDiskSpeedState( DiskStatus )
{ }

void CAtSpeedState::Initialise()
{ }

void CAtSpeedState::Start()
{ }

void CAtSpeedState::ChangeSpeed()
{
  // Move to Changing Speed state
  DiskStatus_->SetState( DiskStatus_->GetChangingSpeedState() );
}

void CAtSpeedState::Stop()
{
  // Move to Stopping state
  DiskStatus_->SetState( DiskStatus_->GetStoppingState() );
}

void CAtSpeedState::UpdateFromDevice()
{ }

///////////////////////////////////////////////////////////////////////////////
// CStoppingState
///////////////////////////////////////////////////////////////////////////////

CStoppingState::CStoppingState( CDiskStatus* DiskStatus )
  : CDiskSpeedState( DiskStatus )
{ }

void CStoppingState::Initialise()
{ }

void CStoppingState::Start()
{
  // Move to Changing Speed state
  DiskStatus_->SetState( DiskStatus_->GetChangingSpeedState() );
}

void CStoppingState::ChangeSpeed()
{ }

void CStoppingState::Stop()
{ }

void CStoppingState::UpdateFromDevice()
{
  // Read Is Spinning from device
  bool vIsSpinning = DiskStatus_->ReadIsSpinningFromDevice();
  
  // If spinning is false, move to Stopped state
  if ( !vIsSpinning )
  {
    DiskStatus_->SetState( DiskStatus_->GetStoppedState() );
  }
}

///////////////////////////////////////////////////////////////////////////////
// CStoppedState
///////////////////////////////////////////////////////////////////////////////

CStoppedState::CStoppedState( CDiskStatus* DiskStatus )
  : CDiskSpeedState( DiskStatus )
{ }

void CStoppedState::Initialise()
{ }

void CStoppedState::Start()
{
  // Move to Changing Speed state
  DiskStatus_->SetState( DiskStatus_->GetChangingSpeedState() );
}

void CStoppedState::ChangeSpeed()
{ }

void CStoppedState::Stop()
{ }

void CStoppedState::UpdateFromDevice()
{ }
