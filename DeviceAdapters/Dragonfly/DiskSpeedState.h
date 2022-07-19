///////////////////////////////////////////////////////////////////////////////
// FILE:          DiskSpeedtate.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _DISKSPEEDSTATE_H_
#define _DISKSPEEDSTATE_H_

class CDiskStatus;

class CDiskSpeedState
{
public:
  CDiskSpeedState( CDiskStatus* DiskStatus );
  virtual ~CDiskSpeedState();
  virtual void Initialise() = 0;
  virtual void Start() = 0;
  virtual void ChangeSpeed() = 0;
  virtual void Stop() = 0;
  virtual void UpdateFromDevice() = 0;

protected:
  CDiskStatus* DiskStatus_;
};

class CChangingSpeedState : public CDiskSpeedState
{
public:
  CChangingSpeedState( CDiskStatus* DiskStatus );
  void Initialise();
  void Start();
  void ChangeSpeed();
  void Stop();
  void UpdateFromDevice();

private:
  static const unsigned int DynamicRangePercent_ = 1;
  unsigned int MinSpeedReached_;
  unsigned int MaxSpeedReached_;
  bool DiskSpeedNotChangingOnce_;
  bool DiskSpeedNotChangingTwice_;
  bool PreviousDiskStateUnknown_;
  bool DiskSpeedIncreasing_;

  bool IsSpeedStable();
  void SpeedHasIncreased( unsigned int PreviousSpeed, unsigned int CurrentSpeed );
  void SpeedHasDecreased( unsigned int PreviousSpeed, unsigned int CurrentSpeed );
  void SpeedUnchanged( unsigned int CurrentSpeed );
  unsigned int GetTargetRangeMin( unsigned int RequestedSpeed ) const;
  unsigned int GetTargetRangeMax( unsigned int RequestedSpeed ) const;
};

class CAtSpeedState : public CDiskSpeedState
{
public:
  CAtSpeedState( CDiskStatus* DiskStatus );
  void Initialise();
  void Start();
  void ChangeSpeed();
  void Stop();
  void UpdateFromDevice();
};

class CStoppingState : public CDiskSpeedState
{
public:
  CStoppingState( CDiskStatus* DiskStatus );
  void Initialise();
  void Start();
  void ChangeSpeed();
  void Stop();
  void UpdateFromDevice();
};

class CStoppedState : public CDiskSpeedState
{
public:
  CStoppedState( CDiskStatus* DiskStatus );
  void Initialise();
  void Start();
  void ChangeSpeed();
  void Stop();
  void UpdateFromDevice();
};

#endif