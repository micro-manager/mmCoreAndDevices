#include "DiskSpeedSimulator.h"

#include "Dragonfly.h"

using namespace std;

class MutexHandler
{
public:
  MutexHandler( boost::mutex& Mutex )
    : Mutex_( Mutex )
  {
    Mutex_.lock();
  }
  ~MutexHandler()
  {
    Mutex_.unlock();
  }
private:
  boost::mutex& Mutex_;
};

CDiskSimulator::CDiskSimulator( IDiskInterface2* DiskInterface, CDragonfly* MMDragonfly )
  : DiskInterface_( DiskInterface ),
  MMDragonfly_( MMDragonfly ),
  CurrentSpeed_( 0 ),
  Step_( 0 ),
  RequestedSpeed_( 0 ),
  ErrorRange_( 0 ),
  TargetSpeed_( 0 )
{
  SetSpeed( 6000 );
}
bool CDiskSimulator::SetSpeed( unsigned int Speed )
{
  static int vMaxRange = 200;
  if ( DiskInterface_->SetSpeed( Speed ) )
  {
    MutexHandler vHandle( Mutex_ );
    RequestedSpeed_ = Speed;
    if ( CurrentSpeed_ < RequestedSpeed_ )
    {
      ErrorRange_ = vMaxRange;
    }
    else
    {
      ErrorRange_ = -vMaxRange;
    }
    TargetSpeed_ = Speed + ErrorRange_;
    Step_ = ( TargetSpeed_ - CurrentSpeed_ ) / 10;
    return true;
  }
  return false;
}
void CDiskSimulator::UpdateSpeed()
{
  int vRand = ( rand() % 21 ) - 10;
  CurrentSpeed_ += Step_ * ( 100 + vRand ) / 100;
  if ( ( TargetSpeed_ - RequestedSpeed_ ) * ( TargetSpeed_ - CurrentSpeed_ ) <= 0 )
  {
    // We passed the target
    CurrentSpeed_ = TargetSpeed_;
  }

  int vDistanceFromTarget = TargetSpeed_ - CurrentSpeed_;
  // If we are close to the target
  if ( abs( vDistanceFromTarget ) <= 10 )
  {
    ErrorRange_ /= -2;
    TargetSpeed_ = RequestedSpeed_ + ErrorRange_;
    vDistanceFromTarget = TargetSpeed_ - CurrentSpeed_;
    Step_ = vDistanceFromTarget / 10;
    if ( abs( vDistanceFromTarget ) <= 2 )
    {
      // The new target is too close to the requested speed, we stop the algorithm
      TargetSpeed_ = RequestedSpeed_;
      Step_ = 0;
      return;
    }
  }
  if ( abs( vDistanceFromTarget ) < 400 )
  {
    Step_ = vDistanceFromTarget / 10;
  }
  if ( abs( Step_ ) < 20 )
  {
    Step_ = ( Step_ < 0 ? -20 : 20 );
  }
}
bool CDiskSimulator::GetSpeed( unsigned int &Speed )
{
  if ( DiskInterface_->GetSpeed( Speed ) )
  {
    MutexHandler vHandle( Mutex_ );
    UpdateSpeed();
    Speed = CurrentSpeed_;
    return true;
  }
  return false;
}
