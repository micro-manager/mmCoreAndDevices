///////////////////////////////////////////////////////////////////////////////
// FILE:          DiskSpeedSimulator.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _DISKSPEEDSIMULATOR_H_
#define _DISKSPEEDSIMULATOR_H_

#include "boost\thread.hpp"

class IDiskInterface2;
class CDragonfly;

class CDiskSimulator
{
public:
  CDiskSimulator( IDiskInterface2* DiskInterface, CDragonfly* MMDragonfly );
  bool SetSpeed( unsigned int Speed );
  bool GetSpeed( unsigned int &Speed );
private:
  IDiskInterface2* DiskInterface_;
  CDragonfly* MMDragonfly_;
  int CurrentSpeed_;
  int Step_;
  int RequestedSpeed_;
  int TargetSpeed_;
  int ErrorRange_;
  boost::mutex Mutex_;

  void UpdateSpeed();
};
#endif