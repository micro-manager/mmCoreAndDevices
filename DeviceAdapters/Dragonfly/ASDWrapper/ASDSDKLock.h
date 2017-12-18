///////////////////////////////////////////////////////////////////////////////
// FILE:          ASDSDKLock.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _ASDSDKLOCK_H_
#define _ASDSDKLOCK_H_

#include <mutex>

class CASDSDKLock
{
public:
  CASDSDKLock();
  ~CASDSDKLock();

private:
  static std::timed_mutex gsSDKMutex;
};
#endif