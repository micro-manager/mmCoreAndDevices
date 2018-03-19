///////////////////////////////////////////////////////////////////////////////
// FILE:          ILESDKLock.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _ILESDKLOCK_H_
#define _ILESDKLOCK_H_

#include "boost\thread.hpp"

class CILESDKLock
{
public:
  CILESDKLock();
  ~CILESDKLock();

private:
  static boost::timed_mutex gsSDKMutex;
};
#endif