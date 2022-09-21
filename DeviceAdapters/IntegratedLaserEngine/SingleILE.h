///////////////////////////////////////////////////////////////////////////////
// FILE:          SingleILE.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   IntegratedLaserEngine controller adapter
//
// Based off the AndorLaserCombiner adapter from Karl Hoover, UCSF
//
//

#ifndef _SINGLEILE_H_
#define _SINGLEILE_H_

#include <string>
#include "IntegratedLaserEngine.h"

class CPorts;
class CActiveBlanking;
class CNDFilters;
class CLowPowerMode;

class CSingleILE : public CIntegratedLaserEngine
{
public:
  CSingleILE();
  virtual ~CSingleILE();

  int Shutdown();
  void CheckAndUpdateLowPowerMode() override;

  static const char* const g_DeviceName;
  static const char* const g_DeviceDescription;

protected:

private:
  CPorts* Ports_;
  CActiveBlanking* ActiveBlanking_;
  CNDFilters* NDFilters_;
  CLowPowerMode* LowPowerMode_;

  std::string GetDeviceName() const;
  bool CreateILE();
  void DeleteILE();
  int InitializePorts();
  int InitializeActiveBlanking();
  int InitializeNDFilters();
  int InitializeLowPowerMode();
  void DisconnectILEInterfaces();
  int ReconnectILEInterfaces();
  int ReconnectNDFilters();
};

#endif
