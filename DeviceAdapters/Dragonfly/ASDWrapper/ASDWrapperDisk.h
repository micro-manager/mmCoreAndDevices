///////////////////////////////////////////////////////////////////////////////
// FILE:          ASDWrapperDisk.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _ASDWRAPPERDISK_H_
#define _ASDWRAPPERDISK_H_

#include "ComponentInterface.h"

class CASDWrapperDisk : public IDiskInterface2
{
public:
  CASDWrapperDisk( IDiskInterface2* DiskInterface );
  ~CASDWrapperDisk();

  // IDiskInterface
  bool GetSpeed( unsigned int &Speed );
  bool SetSpeed( unsigned int Speed );
  bool IncreaseSpeed();
  bool DecreaseSpeed();
  bool GetLimits( unsigned int &Min, unsigned int &Max );
  bool Start();
  bool Stop();
  bool IsSpinning();

  // IDiskInterface2
  bool GetScansPerRevolution( unsigned int *NumberOfScans );

private:
  IDiskInterface2* DiskInterface_;
};
#endif