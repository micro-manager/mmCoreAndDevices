#include "ASDSDKLock.h"

boost::timed_mutex CASDSDKLock::gsSDKMutex;

CASDSDKLock::CASDSDKLock()
{ 
  if ( !gsSDKMutex.try_lock_for( boost::chrono::seconds( 30 ) ) )
  {
    throw std::runtime_error( "Locking access to the ASD SDK failed" );
  }
}

CASDSDKLock::~CASDSDKLock()
{
  gsSDKMutex.unlock();
}