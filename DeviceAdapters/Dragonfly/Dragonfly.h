///////////////////////////////////////////////////////////////////////////////
// FILE:          Dragonfly.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _DRAGONFLY_H_
#define _DRAGONFLY_H_

#include "../../MMDevice/DeviceBase.h"
//#include "../../MMDevice/DeviceThreads.h"
#include "ComponentInterface.h"


//////////////////////////////////////////////////////////////////////////////
// Dragonfly class
//////////////////////////////////////////////////////////////////////////////

class CASDWrapper;
class CDichroicMirror;
class CFilterWheel;
class CDragonflyStatus;
class CDisk;
class CConfocalMode;
class CAperture;
class CCameraPortMirror;
class CLens;
class CPowerDensity;

class IASDLoader;
class IASDInterface;
class IASDInterface2;
class IASDInterface3;

#define ERR_LIBRARY_LOAD 101
#define ERR_LIBRARY_INIT 102
#define ERR_DICHROICMIRROR_INIT 103
#define ERR_FILTERWHEEL1_INIT 104
#define ERR_FILTERWHEEL2_INIT 105
#define ERR_DRAGONFLYSTATUS_INVALID_POINTER 106
#define ERR_DRAGONFLYSTATUS_INIT 107
#define ERR_DISK_INIT 108
#define ERR_CONFOCALMODE_INIT 109
#define ERR_APERTURE_INIT 110
#define ERR_CAMERAPORTMIRROR_INIT 111
#define ERR_LENS_INIT 112
#define ERR_POWERDENSITY_INIT 113

class CDragonfly : public CGenericBase<CDragonfly>
{
public:
  CDragonfly();
  ~CDragonfly();

  int Initialize();
  int Shutdown();

  void GetName( char * Name ) const;
  bool Busy();

  int OnPort( MM::PropertyBase* Prop, MM::ActionType Act );

  void LogComponentMessage( const std::string& Message );

private:
  bool ASDLibraryConnected_;
  bool Initialized_;
  std::string Port_;
  bool DeviceConnected_;

  CASDWrapper* ASDWrapper_;
  CDichroicMirror* DichroicMirror_;
  CFilterWheel* FilterWheel1_;
  CFilterWheel* FilterWheel2_;
  CDragonflyStatus* DragonflyStatus_;
  CDisk* Disk_;
  CConfocalMode* ConfocalMode_;
  CAperture* Aperture_;
  CCameraPortMirror* CameraPortMirror_;
  std::vector<CLens*> Lens_;
  std::vector<CPowerDensity*> PowerDensity_;
  
  IASDLoader* ASDLoader_;

  int Connect( const std::string& Port );
  int Disconnect();
  int InitializeComponents();
  
  int CreateDragonflyStatus( IASDInterface3* ASDInterface );
  int CreateDichroicMirror( IASDInterface* ASDInterface );
  int CreateFilterWheel( IASDInterface* ASDInterface, CFilterWheel*& FilterWheel, TWheelIndex WheelIndex, unsigned int ErrorCode );
  int CreateDisk( IASDInterface* ASDInterface );
  int CreateConfocalMode( IASDInterface3* ASDInterface );
  int CreateAperture( IASDInterface2* ASDInterface );
  int CreateCameraPortMirror( IASDInterface2* ASDInterface );
  int CreateLens( IASDInterface2* ASDInterface, int LensIndex );
  int CreatePowerDensity( IASDInterface3* ASDInterface, int LensIndex );
};

#endif
