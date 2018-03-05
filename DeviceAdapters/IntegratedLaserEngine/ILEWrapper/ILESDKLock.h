///////////////////////////////////////////////////////////////////////////////
// FILE:          ILESDKLock.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _ILESDKLOCK_H_
#define _ILESDKLOCK_H_

#include <mutex>

class CILESDKLock
{
public:
  CILESDKLock();
  ~CILESDKLock();

private:
  static std::mutex gsSDKMutex;
};
#endif