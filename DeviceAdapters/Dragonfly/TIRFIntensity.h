///////////////////////////////////////////////////////////////////////////////
// FILE:          TIRFIntensity.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _TIRFINTENSITY_H_
#define _TIRFINTENSITY_H_

#include <mutex>

#include "Property.h"
#include "../../MMDevice/DeviceThreads.h"


class ITIRFIntensityInterface;
class CDragonfly;
class CTIRFIntensityMonitor;

class CTIRFIntensity
{
public:
  CTIRFIntensity( ITIRFIntensityInterface* TIRFIntensity, CDragonfly* MMDragonfly );
  ~CTIRFIntensity();

  int OnMonitorStatusChange( MM::PropertyBase * Prop, MM::ActionType Act );
  typedef MM::Action<CTIRFIntensity> CPropertyAction;

  void UpdateFromDevice();

private:
  ITIRFIntensityInterface* TIRFIntensity_;
  CDragonfly* MMDragonfly_;
  std::unique_ptr<CTIRFIntensityMonitor> TIRFIntensityMonitor_;
  std::mutex TIRFIntensityMutex_;

  int CurrentTIRFIntensity_ = 0;
  int TIRFIntensityMin_ = 0;
  int TIRFIntensityMax_ = 0;
};


class CTIRFIntensityMonitor : public MMDeviceThreadBase
{
public:
  CTIRFIntensityMonitor( CDragonfly* MMDragonfly, CTIRFIntensity* TIRFIntensity );
  virtual ~CTIRFIntensityMonitor();

  int svc();

private:
  CDragonfly* MMDragonfly_;
  CTIRFIntensity* TIRFIntensity_;
  bool KeepRunning_;
};
#endif