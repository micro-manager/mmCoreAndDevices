///////////////////////////////////////////////////////////////////////////////
// FILE:          DualILE.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   IntegratedLaserEngine controller adapter
//
// Based off the AndorLaserCombiner adapter from Karl Hoover, UCSF
//
//

#ifndef _DUALILE_H_
#define _DUALILE_H_

#include <string>
#include "IntegratedLaserEngine.h"

#define ERR_DUALPORTS_PORTCHANGEFAIL 301
#define ERR_DUALPORTS_PORTCONFIGCORRUPTED 302
#define ERR_DUALILE_GETINTERFACE 303
#define ERR_LOWPOWERPRESENT 304


class CDualILEPorts;
class CPortsConfiguration;
class CDualILEActiveBlanking;
class CDualILELowPowerMode;

class CDualILE : public CIntegratedLaserEngine
{
public:
  CDualILE( bool ILE700 );
  virtual ~CDualILE();

  int Shutdown();

  static const char* const g_DualDeviceName;
  static const char* const g_Dual700DeviceName;
  static const char* const g_DualDeviceDescription;
  static const char* const g_Dual700DeviceDescription;

protected:

private:
  bool ILE700_;
  CDualILEPorts* Ports_;
  CPortsConfiguration* PortsConfiguration_;
  CDualILEActiveBlanking* ActiveBlanking_;
  CDualILELowPowerMode* LowPowerMode_;

  std::string GetDeviceName() const;
  bool CreateILE();
  void DeleteILE();
  int InitializePorts();
  int InitializeActiveBlanking();
  int InitializeLowPowerMode();
  void DisconnectILEInterfaces();
  int ReconnectILEInterfaces();
};

#endif
