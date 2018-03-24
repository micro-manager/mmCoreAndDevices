///////////////////////////////////////////////////////////////////////////////
// FILE:          IntegratedLaserEngine.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
// DESCRIPTION:   IntegratedLaserEngine controller adapter
//
// Based off the AndorLaserCombiner adapter from Karl Hoover, UCSF
//
//

#ifndef _INTEGRATEDLASERENGINE_H_
#define _INTEGRATEDLASERENGINE_H_

#include "../../MMDevice/MMDevice.h"
#include "../../MMDevice/DeviceBase.h"
#include "../../MMDevice/DeviceUtils.h"
#include <string>
#include <vector>
#include "ILEWrapperInterface.h"

#define ERR_PORTS_INIT 101
#define ERR_ACTIVEBLANKING_INIT 102
#define ERR_LOWPOWERMODE_INIT 103
#define ERR_LASERS_INIT 104
#define ERR_DEVICE_INDEXINVALID 105
#define ERR_DEVICE_CONNECTIONFAILED 106
#define ERR_INTERLOCK 201
#define ERR_CLASSIV_INTERLOCK 202
#define ERR_DEVICE_NOT_CONNECTED 203
#define ERR_DUALPORTS_PORTCHANGEFAIL 301
#define ERR_DUALPORTS_PORTCONFIGCORRUPTED 302
#define ERR_ACTIVEBLANKING_SET 401
#define ERR_LOWPOWERMODE_SET 501
#define ERR_PORTS_SET 601

class IALC_REVObject3;
class IALC_REV_Laser2;
class CPorts;
class CActiveBlanking;
class CLowPowerMode;
class CLasers;

class CIntegratedLaserEngine : public CShutterBase<CIntegratedLaserEngine>
{
public:
  CIntegratedLaserEngine( const std::string& Description, int NbDevices );
  virtual ~CIntegratedLaserEngine();

  // MMDevice API
  int Initialize();
  virtual int Shutdown();

  void GetName( char* Name ) const;
  bool Busy();

  // Action interface
  int OnDeviceChange( MM::PropertyBase* Prop, MM::ActionType Act, long DeviceIndex );
  int OnResetDevice( MM::PropertyBase* Prop, MM::ActionType Act );

  // Shutter API
  int SetOpen( bool Open = true );
  int GetOpen( bool& Open );
  int Fire( double DeltaT );

  // Helper functions
  void LogMMMessage( std::string Message, bool DebugOnly = false );
  MM::MMTime GetCurrentTime();

  void CheckAndUpdateLasers();
  void ActiveClassIVInterlock();
  void UpdatePropertyUI( const char* PropertyName, const char* PropertyValue );

protected:
  IILEWrapperInterface* ILEWrapper_;
  IALC_REVObject3 *ILEDevice_;
  std::vector<std::string> DevicesNames_;

private:   
  IILEWrapperInterface::TDeviceList DeviceList_;
  CLasers* Lasers_;
  MM::PropertyBase* ResetDeviceProperty_;

  bool Initialized_;
  MM::MMTime ChangedTime_;

  CIntegratedLaserEngine& operator = ( CIntegratedLaserEngine& /*rhs*/ )
  {
    assert( false );
    return *this;
  }

  virtual std::string GetDeviceName() const = 0;
  void CreateDeviceSelectionProperty( int DeviceID, int DeviceIndex );
  virtual bool CreateILE() = 0;
  virtual void DeleteILE() = 0;
  virtual int InitializePorts() = 0;
  virtual int InitializeActiveBlanking() = 0;
  virtual int InitializeLowPowerMode() = 0;
  virtual void DisconnectILEInterfaces() = 0;
  virtual void ReconnectILEInterfaces() = 0;
};

class CSingleILE : public CIntegratedLaserEngine
{
public:
  CSingleILE();
  virtual ~CSingleILE();

  int Shutdown();

protected:

private:
  CPorts* Ports_;
  CActiveBlanking* ActiveBlanking_;
  CLowPowerMode* LowPowerMode_;

  std::string GetDeviceName() const;
  bool CreateILE();
  void DeleteILE();
  int InitializePorts();
  int InitializeActiveBlanking();
  int InitializeLowPowerMode();
  void DisconnectILEInterfaces();
  void ReconnectILEInterfaces();
};

class CDualILEPorts;
class CPortsConfiguration;
class CDualILEActiveBlanking;

class CDualILE : public CIntegratedLaserEngine
{
public:
  CDualILE( bool ILE700 );
  virtual ~CDualILE();

  int Shutdown();

  //Action interface
  typedef MM::Action<CDualILE> CPropertyAction;
  
protected:

private:
  bool ILE700_;
  CDualILEPorts* Ports_;
  CPortsConfiguration* PortsConfiguration_;
  CDualILEActiveBlanking* ActiveBlanking_;

  std::string GetDeviceName() const;
  bool CreateILE();
  void DeleteILE();
  int InitializePorts();
  int InitializeActiveBlanking();
  int InitializeLowPowerMode();
  void DisconnectILEInterfaces();
  void ReconnectILEInterfaces();
};

#endif
