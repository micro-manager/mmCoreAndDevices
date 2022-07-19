///////////////////////////////////////////////////////////////////////////////
// FILE:          ASDSDKLock.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _ASDSDKLOCK_H_
#define _ASDSDKLOCK_H_

#include "boost\thread.hpp"

class CASDSDKLock
{
public:
  CASDSDKLock();
  ~CASDSDKLock();

private:
  static boost::timed_mutex gsSDKMutex;
};
#endif