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
class CDiskStatus;
class CDiskStateChange;

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
  unsigned int ScansPerRevolution_;
  CDiskStatus* DiskStatus_;
  CDiskStateChange* SpeedMonitorStateChangeObserver_;
  CDiskStateChange* StatusMonitorStateChangeObserver_;
  CDiskStateChange* FrameScanTimeStateChangeObserver_;

  CDiskSimulator DiskSimulator_;

  double CalculateFrameScanTime( unsigned int Speed, unsigned int ScansPerRevolution );
};


class CDiskStatusMonitor : public MMDeviceThreadBase
{
public:
  CDiskStatusMonitor( CDragonfly* MMDragonfly, CDiskStatus* DiskStatus );
  virtual ~CDiskStatusMonitor();

  int svc();

private:
  CDragonfly* MMDragonfly_;
  CDiskStatus* DiskStatus_;
  bool KeepRunning_;
};
#endif