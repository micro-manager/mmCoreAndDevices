///////////////////////////////////////////////////////////////////////////////
// FILE:          DiskStatus.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _DISKSTATUS_H_
#define _DISKSTATUS_H_
#include "DiskStatusInterface.h"
#include <list>

class CDiskSpeedState;
class CChangingSpeedState;
class CAtSpeedState;
class CStoppingState;
class CStoppedState;

class CDiskStatus : public IDiskStatus
{
public:
  CDiskStatus( IDiskInterface2* DiskInterface, CDragonfly* MMDragonfly, CDiskSimulator* DiskSimulator );
  ~CDiskStatus();

  // Inherited from IDiskStatus
  void RegisterObserver( CDiskStateChange* Observer );
  void UnregisterObserver( CDiskStateChange* Observer );
  void RegisterErrorObserver( CDiskStateError* Observer );
  void UnregisterErrorObserver( CDiskStateError* Observer );

  void Start();
  void ChangeSpeed( unsigned int NewRequestedSpeed );
  void Stop();
  void UpdateFromDevice();

  bool IsChangingSpeed() const;
  bool IsAtSpeed() const;
  bool IsStopping() const;
  bool IsStopped() const;

  unsigned int GetCurrentSpeed() const;
  unsigned int GetRequestedSpeed() const;

  // CDiskStatus methods
  unsigned int ReadCurrentSpeedFromDevice();
  bool ReadIsSpinningFromDevice() const;

  CDiskSpeedState* GetChangingSpeedState( );
  CDiskSpeedState* GetAtSpeedState();
  CDiskSpeedState* GetStoppingState();
  CDiskSpeedState* GetStoppedState();
  void SetState( CDiskSpeedState* NewState );

  void ErrorEncountered( const std::string& ErrorMessage );

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
  std::list<CDiskStateChange*> Observers_;
  std::list<CDiskStateError*> ErrorObservers_;

  CDiskSimulator* DiskSimulator_;

  void NotifyStateChange();
  void NotifyStateError(const std::string& ErrorMessage);
};
#endif