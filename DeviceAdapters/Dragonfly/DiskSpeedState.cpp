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
  DiskSpeedStableOnce_( false ),
  DiskSpeedStableTwice_( false )
{ }

void CChangingSpeedState::Initialise()
{
  DiskStatus_->ReadCurrentSpeedFromDevice();
  PreviousDiskStateUnknown_ = true;
  MinSpeedReached_ = 0;
  MaxSpeedReached_ = DiskStatus_->GetRequestedSpeed() + 1000;
  DiskSpeedStableOnce_ = false;
  DiskSpeedStableTwice_ = false;
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

bool CChangingSpeedState::IsSpeedStable()
{
  unsigned int vPreviousSpeed = DiskStatus_->GetCurrentSpeed();
  unsigned int vCurrentSpeed = DiskStatus_->ReadCurrentSpeedFromDevice();
  unsigned int vRequestedSpeed = DiskStatus_->GetRequestedSpeed();

  if ( vCurrentSpeed > vPreviousSpeed )
  {
    DiskSpeedStableOnce_ = false;
    DiskSpeedStableTwice_ = false;
    // Speed is increasing
    if ( !PreviousDiskStateUnknown_ )
    {
      if ( !DiskSpeedIncreasing_ )
      {
        // The speed was previously decreasing, we may have reached a local min
        if ( vCurrentSpeed > vRequestedSpeed )
        {
          // Unlikely case where we are above the requested speed
          if ( vPreviousSpeed >= vRequestedSpeed )
          {
            // Boolean was badly initialised previously or a new requested value has been set (?)
          }
          else
          {
            // We might be so close to the requested speed that we are looping around it between 2 ticks (unlikely though)
            MinSpeedReached_ = vPreviousSpeed;
          }
        }
        else
        {
          // We are below the requested speed therefore we reached a local minimum
          MinSpeedReached_ = vPreviousSpeed;
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
  else if ( vCurrentSpeed < vPreviousSpeed )
  {
    DiskSpeedStableOnce_ = false;
    DiskSpeedStableTwice_ = false;
    // Speed is decreasing
    if ( !PreviousDiskStateUnknown_ )
    {
      if ( DiskSpeedIncreasing_ )
      {
        // The speed was previously increasing, we may have reached a local max
        if ( vCurrentSpeed < vRequestedSpeed )
        {
          // Unlikely case where we are below the requested speed
          if ( vPreviousSpeed <= vRequestedSpeed )
          {
            // Boolean was badly initialised previously
          }
          else
          {
            // We might be so close to the requested speed that we are looping around it between 2 ticks (unlikely though)
            MaxSpeedReached_ = vPreviousSpeed;
          }
        }
        else
        {
          // We are above the requested speed therefore we reached a local maximum
          MaxSpeedReached_ = vPreviousSpeed;
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
  else if ( vCurrentSpeed == vPreviousSpeed )
  {
    // Speed hasn't changed
    if ( DiskSpeedStableOnce_ )
    {
      // The speed hasn't changed in 2 ticks
      if ( vCurrentSpeed == vRequestedSpeed )
      {
        // We reached the requested speed
        MinSpeedReached_ = vRequestedSpeed;
        MaxSpeedReached_ = vRequestedSpeed;
      }
      else
      {
        // The disk is not changing speed even though it should
        // Something's wrong, we report it to the user
        DiskSpeedStableTwice_ = true;
      }
    }
    DiskSpeedStableOnce_ = true;
  }
  vPreviousSpeed = vCurrentSpeed;
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
