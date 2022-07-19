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
#include "boost\thread.hpp"

class IDiskInterface2;
class IConfigFileHandler;
class CDragonfly;
class CDiskStatusMonitor;
class IDiskStatus;
class CDiskStateChange;
class CDiskStateError;

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
  IDiskStatus* DiskStatus_;
  CDiskStateChange* SpeedMonitorStateChangeObserver_;
  CDiskStateChange* StatusMonitorStateChangeObserver_;
  CDiskStateChange* FrameScanTimeStateChangeObserver_;
  CDiskStateError* StatusMonitorStateErrorObserver_;
  boost::mutex DiskStatusMutex_;

  CDiskSimulator DiskSimulator_;

  double CalculateFrameScanTime( unsigned int Speed, unsigned int ScansPerRevolution );
};


class CDiskStatusMonitor : public MMDeviceThreadBase
{
public:
  CDiskStatusMonitor( CDragonfly* MMDragonfly, IDiskStatus* DiskStatus, boost::mutex& DiskStatusMutex );
  virtual ~CDiskStatusMonitor();

  int svc();

private:
  CDragonfly* MMDragonfly_;
  IDiskStatus* DiskStatus_;
  boost::mutex& DiskStatusMutex_;
  bool KeepRunning_;
};
#endif