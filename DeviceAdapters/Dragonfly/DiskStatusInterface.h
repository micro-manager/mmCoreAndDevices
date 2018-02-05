///////////////////////////////////////////////////////////////////////////////
// FILE:          DiskStatusInterface.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _DISKSTATUSINTERFACE_H_
#define _DISKSTATUSINTERFACE_H_

class IDiskInterface2;
class CDragonfly;

class CDiskSimulator;

class CDiskStateChange
{
public:
  CDiskStateChange() : StateChangeNotified_( true ) {}
  void Notify() { StateChangeNotified_ = true; }
  bool HasBeenNotified() 
  { 
    if (StateChangeNotified_)
    { 
      StateChangeNotified_ = false;
      return true;
    } 
    return false;
  }
private:
  bool StateChangeNotified_;
};

class IDiskStatus
{
public:
  virtual ~IDiskStatus() = 0 {};

  virtual void RegisterObserver( CDiskStateChange* Observer ) = 0;
  virtual void UnregisterObserver( CDiskStateChange* Observer ) = 0;

  virtual void Start() = 0;
  virtual void ChangeSpeed( unsigned int NewRequestedSpeed ) = 0;
  virtual void Stop() = 0;
  virtual void UpdateFromDevice() = 0;

  virtual bool IsChangingSpeed() const = 0;
  virtual bool IsAtSpeed() const = 0;
  virtual bool IsStopping() const = 0;
  virtual bool IsStopped() const = 0;

  virtual unsigned int GetCurrentSpeed() const = 0;
  virtual unsigned int GetRequestedSpeed() const = 0;
};

IDiskStatus* CreateDiskStatus( IDiskInterface2* DiskInterface, CDragonfly* MMDragonfly, CDiskSimulator* DiskSimulator );
#endif