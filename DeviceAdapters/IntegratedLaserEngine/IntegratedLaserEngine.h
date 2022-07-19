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

#include "../../MMDevice/DeviceBase.h"
#include <string>
#include <vector>
#include "ILEWrapperInterface.h"


#define ERR_LIBRARY_LOAD 101
#define ERR_PORTS_INIT 102
#define ERR_ACTIVEBLANKING_INIT 103
#define ERR_LOWPOWERMODE_INIT 104
#define ERR_LASERS_INIT 105
#define ERR_VERYLOWPOWER_INIT 106
#define ERR_DEVICE_INDEXINVALID 107
#define ERR_DEVICE_CONNECTIONFAILED 108
#define ERR_DEVICE_RECONNECTIONFAILED 109

#define ERR_LASER_STATE_READ 201
#define ERR_INTERLOCK 202
#define ERR_CLASSIV_INTERLOCK 203
#define ERR_KEY_INTERLOCK 204
#define ERR_DEVICE_NOT_CONNECTED 205
#define ERR_LASER_SET 206
#define ERR_SETCONTROLMODE 207
#define ERR_SETLASERSHUTTER 208

#define ERR_ACTIVEBLANKING_SET 401
#define ERR_ACTIVEBLANKING_GETNBLINES 402
#define ERR_ACTIVEBLANKING_GETSTATE 403

#define ERR_LOWPOWERMODE_SET 501
#define ERR_LOWPOWERMODE_GET 502

#define ERR_PORTS_SET 601
#define ERR_PORTS_GET 602

#define ERR_VERYLOWPOWER_SET 701
#define ERR_VERYLOWPOWER_GET 702

class IALC_REVObject3;
class CLasers;
class CVeryLowPower;

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
  void ActiveKeyInterlock();
  void UpdatePropertyUI( const char* PropertyName, const char* PropertyValue );
  int GetClassIVAndKeyInterlockStatus();

protected:
  IILEWrapperInterface* ILEWrapper_;
  IALC_REVObject3 *ILEDevice_;
  std::vector<std::string> DevicesNames_;
  int ConstructionReturnCode_;

private:   
  IILEWrapperInterface::TDeviceList DeviceList_;
  CLasers* Lasers_;
  CVeryLowPower* VeryLowPower_;
  MM::PropertyBase* ResetDeviceProperty_;
  bool ClassIVInterlockActive_;
  bool KeyInterlockActive_;

  bool Initialized_;
  MM::MMTime ChangedTime_;

  CIntegratedLaserEngine& operator = ( CIntegratedLaserEngine& /*rhs*/ )
  {
    assert( false );
    return *this;
  }

  void CreateDeviceSelectionProperty( int DeviceID, int DeviceIndex );
  void ActivateInterlock();

  virtual std::string GetDeviceName() const = 0;
  virtual bool CreateILE() = 0;
  virtual void DeleteILE() = 0;

  int InitalizeLasers();
  int InitializeVeryLowPower();
  virtual int InitializePorts() = 0;
  virtual int InitializeActiveBlanking() = 0;
  virtual int InitializeLowPowerMode() = 0;

  virtual void DisconnectILEInterfaces() = 0;

  int ReconnectLasers();
  int ReconnectVeryLowPower();
  virtual int ReconnectILEInterfaces() = 0;
};

#endif
