///////////////////////////////////////////////////////////////////////////////
// FILE:          ASDWrapperInterface.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _ASDWRAPPERINTERFACE_H_
#define _ASDWRAPPERINTERFACE_H_

#include "ASDInterface.h"
#include <map>

class CASDWrapperDichroicMirror;
class CASDWrapperFilterWheel;
class CASDWrapperDisk;
class CASDWrapperStatus;
class CASDWrapperConfocalMode;

class CASDWrapperInterface : public IASDInterface3
{
public:
  CASDWrapperInterface( IASDInterface3* ASDInterface );
  ~CASDWrapperInterface();

  // IASDInterface
  const char* GetSerialNumber() const;
  const char* GetProductID() const;
  const char* GetSoftwareVersion() const;
  const char* GetSoftwareBuildTime() const;
  bool IsDichroicAvailable();
  IDichroicMirrorInterface* GetDichroicMirror();
  bool IsDiskAvailable();
  IDiskInterface* GetDisk();
  bool IsFilterWheelAvailable( TWheelIndex FilterIndex );
  IFilterWheelInterface* GetFilterWheel( TWheelIndex FilterIndex );
  bool IsBrightFieldPortAvailable();
  IConfocalModeInterface2* GetBrightFieldPort();
  IDiskInterface2* GetDisk_v2();

  // IASDInterface2
  bool IsApertureAvailable();
  IApertureInterface* GetAperture();
  bool IsCameraPortMirrorAvailable();
  ICameraPortMirrorInterface* GetCameraPortMirror();
  bool IsLensAvailable( TLensType LensIndex );
  ILensInterface* GetLens( TLensType LensIndex );
  int GetModelID();

  // IASDInterface3
  bool IsIllLensAvailable( TLensType LensIndex );
  IIllLensInterface* GetIllLens( TLensType LensIndex );
  bool IsEPIPolariserAvailable();
  IEPIPolariserInterface*	GetEPIPolariser();
  bool IsTIRFPolariserAvailable();
  ITIRFPolariserInterface* GetTIRFPolariser();
  bool IsEmissionIrisAvailable();
  IEmissionIrisInterface* GetEmissionIris();
  bool IsSuperResAvailable();
  ISuperResInterface* GetSuperRes();
  bool IsImagingModeAvailable();
  IConfocalModeInterface3* GetImagingMode();
  bool IsTIRFAvailable();
  ITIRFInterface* GetTIRF();
  IStatusInterface* GetStatus();
  IFrontPanelLEDInterface* GetFrontPanelLED();

private:
  IASDInterface3* ASDInterface_;
  CASDWrapperDichroicMirror* DichroicMirrorWrapper_;
  std::map<TWheelIndex, CASDWrapperFilterWheel*> FilterWheelWrappers_;
  CASDWrapperDisk* DiskWrapper_;
  CASDWrapperStatus* StatusWrapper_;
  CASDWrapperConfocalMode* ConfocalModeWrapper_;
};

#endif