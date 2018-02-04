///////////////////////////////////////////////////////////////////////////////
// FILE:          Disk.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _DISK_H_
#define _DISK_H_

#include "Property.h"
#include "../../MMDevice/DeviceThreads.h"
#include "DiskSpeedSimulator.h"

class IDiskInterface2;
class IConfigFileHandler;
class CDragonfly;
class CDiskStatusMonitor;

class CDisk
{
public:
  CDisk( IDiskInterface2* DiskInterface, IConfigFileHandler* ConfigFileHandler, CDragonfly* MMDragonfly );
  ~CDisk();

  int OnSpeedChange( MM::PropertyBase * Prop, MM::ActionType Act );
  int OnStatusChange( MM::PropertyBase * Prop, MM::ActionType Act );
  int OnMonitorStatusChange( MM::PropertyBase * Prop, MM::ActionType Act );
  typedef MM::Action<CDisk> CPropertyAction;

private:
  IDiskInterface2* DiskInterface_;
  IConfigFileHandler* ConfigFileHandler_;
  CDragonfly* MMDragonfly_;
  CDiskStatusMonitor* DiskStatusMonitor_;
  unsigned int RequestedSpeed_;
  bool RequestedSpeedAchieved_;
  bool StopRequested_;
  bool StopWitnessed_;
  bool FrameScanTimeUpdated_;
  unsigned int TargetRangeMin_;
  unsigned int TargetRangeMax_;
  bool DiskSpeedIncreasing_;
  bool DiskSpeedStableOnce_;
  bool DiskSpeedStableTwice_;
  unsigned int PreviousSpeed_;
  unsigned int MaxSpeedReached_;
  unsigned int MinSpeedReached_;
  unsigned int ScansPerRevolution_;

  CDiskSimulator DiskSimulator_;

  double CalculateFrameScanTime( unsigned int Speed, unsigned int ScansPerRevolution );
  void UpdateSpeedRange();
  bool IsSpeedWithinMargin( unsigned int CurrentSpeed );
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