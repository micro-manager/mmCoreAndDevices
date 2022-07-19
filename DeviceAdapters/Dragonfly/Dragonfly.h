///////////////////////////////////////////////////////////////////////////////
// FILE:          Dragonfly.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _DRAGONFLY_H_
#define _DRAGONFLY_H_

#include "../../MMDevice/DeviceBase.h"
#include "ComponentInterface.h"
#include <list>


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
class CSuperRes;
class CTIRF;
class CConfigFileHandler;

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
#define ERR_SUPERRES_INIT 114
#define ERR_TIRF_INIT 115
#define ERR_CONFIGFILEIO_ERROR 116
#define ERR_COMPORTPROPERTY_CREATION 117
#define ERR_CONFIGFILEPROPERTY_CREATION 118

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
  void UpdatePropertyUI( const char* PropertyName, const char* PropertyValue );

private:
  bool Initialized_;
  std::string Port_;
  int ConstructionReturnCode_;

  CASDWrapper* ASDWrapper_;
  CDichroicMirror* DichroicMirror_;
  CFilterWheel* FilterWheel1_;
  CFilterWheel* FilterWheel2_;
  CDragonflyStatus* DragonflyStatus_;
  CDisk* Disk_;
  CConfocalMode* ConfocalMode_;
  CAperture* Aperture_;
  CCameraPortMirror* CameraPortMirror_;
  std::list<CLens*> Lens_;
  CSuperRes* SuperRes_;
  CTIRF* TIRF_;
  CConfigFileHandler* ConfigFile_;
  
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
  int CreateSuperRes( IASDInterface3* ASDInterface );
  int CreateTIRF( IASDInterface3* ASDInterface );
};

#endif
