#include "ASDSDKLock.h"

std::timed_mutex CASDSDKLock::gsSDKMutex;

CASDSDKLock::CASDSDKLock()
{ 
  if ( !gsSDKMutex.try_lock_for( std::chrono::seconds( 60 ) ) )
  {
    throw std::runtime_error( "Locking access to the ASD SDK failed" );
  }
}

CASDSDKLock::~CASDSDKLock()
{
  gsSDKMutex.unlock();
}