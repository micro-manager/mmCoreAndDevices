///////////////////////////////////////////////////////////////////////////////
// FILE:          DiskStatus.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _DISKSTATUS_H_
#define _DISKSTATUS_H_
#include <list>

class IDiskInterface2;
class CDragonfly;
class CDiskSpeedState;
class CChangingSpeedState;
class CAtSpeedState;
class CStoppingState;
class CStoppedState;

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

class CDiskStatus
{
public:
  CDiskStatus( IDiskInterface2* DiskInterface, CDragonfly* MMDragonfly, CDiskSimulator* DiskSimulator );
  ~CDiskStatus();

  void RegisterObserver( CDiskStateChange* Observer );

  void Start();
  void ChangeSpeed( unsigned int NewRequestedSpeed );
  void Stop();
  void UpdateFromDevice();

  unsigned int GetCurrentSpeed() const;
  unsigned int ReadCurrentSpeedFromDevice();
  bool ReadIsSpinningFromDevice() const;
  unsigned int GetRequestedSpeed() const;

  bool IsChangingSpeed() const;
  bool IsAtSpeed() const;
  bool IsStopping() const;
  bool IsStopped() const;
  CDiskSpeedState* GetChangingSpeedState( );
  CDiskSpeedState* GetAtSpeedState();
  CDiskSpeedState* GetStoppingState();
  CDiskSpeedState* GetStoppedState();
  void SetState( CDiskSpeedState* NewState );


private:
  IDiskInterface2* DiskInterface_;
  CDragonfly* MMDragonfly_;
  unsigned int RequestedSpeed_;
  unsigned int CurrentSpeed_;
  CChangingSpeedState* ChangingSpeedState_;
  CAtSpeedState* AtSpeedState_;
  CStoppingState* StoppingState_;
  CStoppedState* StoppedState_;
  CDiskSpeedState* CurrentState_;

  CDiskSimulator* DiskSimulator_;
  std::list<CDiskStateChange*> Observers_;

  void NotifyStateChange();
};
#endif