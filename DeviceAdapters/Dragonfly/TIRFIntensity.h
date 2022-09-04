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
class IConfocalMode;
class CDragonfly;
class CTIRFIntensityMonitor;

class CTIRFIntensity
{
public:
  CTIRFIntensity( ITIRFIntensityInterface* TIRFIntensity, IConfocalMode* ConfocalMode, CDragonfly* MMDragonfly );
  ~CTIRFIntensity();

  int OnIntensityUpdate( MM::PropertyBase * Prop, MM::ActionType Act );
  int OnMonitoringUpdate( MM::PropertyBase * Prop, MM::ActionType Act );
  typedef MM::Action<CTIRFIntensity> CPropertyAction;

  void UpdateFromDevice();

private:
  ITIRFIntensityInterface* TIRFIntensity_;
  IConfocalMode* ConfocalMode_;
  CDragonfly* MMDragonfly_;
  std::unique_ptr<CTIRFIntensityMonitor> TIRFIntensityMonitor_;
  std::mutex TIRFIntensityMutex_;

  int CurrentTIRFIntensity_ = 0;
  std::atomic<bool> AllowMonitoring_ = true;
};


class CTIRFIntensityMonitor : public MMDeviceThreadBase
{
public:
  CTIRFIntensityMonitor( CTIRFIntensity* TIRFIntensity );
  virtual ~CTIRFIntensityMonitor();

  int svc();

private:
  CTIRFIntensity* TIRFIntensity_;
  bool KeepRunning_;
};
#endif