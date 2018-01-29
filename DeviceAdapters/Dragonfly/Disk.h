///////////////////////////////////////////////////////////////////////////////
// FILE:          Disk.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _DISK_H_
#define _DISK_H_

#include "Property.h"
#include "../../MMDevice/DeviceThreads.h"

class IDiskInterface2;
class CDragonfly;
class CDiskStatusMonitor;

class CDisk
{
public:
  CDisk( IDiskInterface2* DiskInterface, CDragonfly* MMDragonfly );
  ~CDisk();

  int OnSpeedChange( MM::PropertyBase * Prop, MM::ActionType Act );
  int OnStatusChange( MM::PropertyBase * Prop, MM::ActionType Act );
  int OnMonitorStatusChange( MM::PropertyBase * Prop, MM::ActionType Act );
  typedef MM::Action<CDisk> CPropertyAction;

private:
  IDiskInterface2* DiskInterface_;
  CDragonfly* MMDragonfly_;
  CDiskStatusMonitor* DiskStatusMonitor_;
  unsigned int RequestedSpeed_;
  bool RequestedSpeedAchieved_;
  bool StopRequested_;
  bool StopWitnessed_;
  bool FrameScanTimeUpdated_;

  double CalculateFrameScanTime( unsigned int Speed, unsigned int ScansPerRevolution );
};


class CDiskStatusMonitor : public MMDeviceThreadBase
{
public:
  CDiskStatusMonitor( CDragonfly* MMDragonfly );
  virtual ~CDiskStatusMonitor();

  int svc();

private:
  CDragonfly* MMDragonfly_;
  bool KeepRunning_;
};
#endif