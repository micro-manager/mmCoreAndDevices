#include "ILESDKLock.h"

std::mutex CILESDKLock::gsSDKMutex;

CILESDKLock::CILESDKLock()
{ 
  if ( !gsSDKMutex.try_lock() )
  {
    throw std::runtime_error( "Locking access to the ILE SDK failed" );
  }
}

CILESDKLock::~CILESDKLock()
{
  gsSDKMutex.unlock();
}