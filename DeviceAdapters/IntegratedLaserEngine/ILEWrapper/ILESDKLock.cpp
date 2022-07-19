#include "ILESDKLock.h"

boost::timed_mutex CILESDKLock::gsSDKMutex;

CILESDKLock::CILESDKLock()
{ 
  if ( !gsSDKMutex.try_lock_for( boost::chrono::seconds( 30 ) ) )
  {
    throw std::runtime_error( "locking access to the ile sdk failed" );
  }
}

CILESDKLock::~CILESDKLock()
{
  gsSDKMutex.unlock();
}