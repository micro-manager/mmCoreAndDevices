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
  bool __stdcall GetSpeed( unsigned int &Speed );
  bool __stdcall SetSpeed( unsigned int Speed );
  bool __stdcall IncreaseSpeed();
  bool __stdcall DecreaseSpeed();
  bool __stdcall GetLimits( unsigned int &Min, unsigned int &Max );
  bool __stdcall Start();
  bool __stdcall Stop();
  bool __stdcall IsSpinning();

  // IDiskInterface2
  bool __stdcall GetScansPerRevolution( unsigned int *NumberOfScans );

private:
  IDiskInterface2* DiskInterface_;
};
#endif