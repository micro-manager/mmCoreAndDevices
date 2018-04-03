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
class CASDWrapperAperture;
class CASDWrapperCameraPortMirror;
class CASDWrapperLens;
class CASDWrapperIllLens;
class CASDWrapperSuperRes;
class CASDWrapperTIRF;
class CASDWrapperTIRFPolariser;

class CASDWrapperInterface : public IASDInterface3
{
public:
  CASDWrapperInterface( IASDInterface3* ASDInterface );
  ~CASDWrapperInterface();

  // IASDInterface
  const char* __stdcall GetSerialNumber() const;
  const char* __stdcall GetProductID() const;
  const char* __stdcall GetSoftwareVersion() const;
  const char* __stdcall GetSoftwareBuildTime() const;
  bool __stdcall IsDichroicAvailable();
  IDichroicMirrorInterface* __stdcall GetDichroicMirror();
  bool __stdcall IsDiskAvailable();
  IDiskInterface* __stdcall GetDisk();
  bool __stdcall IsFilterWheelAvailable( TWheelIndex FilterIndex );
  IFilterWheelInterface* __stdcall GetFilterWheel( TWheelIndex FilterIndex );
  bool __stdcall IsBrightFieldPortAvailable();
  IConfocalModeInterface2* __stdcall GetBrightFieldPort();
  IDiskInterface2* __stdcall GetDisk_v2();

  // IASDInterface2
  bool __stdcall IsApertureAvailable();
  IApertureInterface* __stdcall GetAperture();
  bool __stdcall IsCameraPortMirrorAvailable();
  ICameraPortMirrorInterface* __stdcall GetCameraPortMirror();
  bool __stdcall IsLensAvailable( TLensType LensIndex );
  ILensInterface* __stdcall GetLens( TLensType LensIndex );
  int __stdcall GetModelID();

  // IASDInterface3
  bool __stdcall IsIllLensAvailable( TLensType LensIndex );
  IIllLensInterface* __stdcall GetIllLens( TLensType LensIndex );
  bool __stdcall IsEPIPolariserAvailable();
  IEPIPolariserInterface*	__stdcall GetEPIPolariser();
  bool __stdcall IsTIRFPolariserAvailable();
  ITIRFPolariserInterface* __stdcall GetTIRFPolariser();
  bool __stdcall IsEmissionIrisAvailable();
  IEmissionIrisInterface* __stdcall GetEmissionIris();
  bool __stdcall IsSuperResAvailable();
  ISuperResInterface* __stdcall GetSuperRes();
  bool __stdcall IsImagingModeAvailable();
  IConfocalModeInterface3* __stdcall GetImagingMode();
  bool __stdcall IsTIRFAvailable();
  ITIRFInterface* __stdcall GetTIRF();
  IStatusInterface* __stdcall GetStatus();
  IFrontPanelLEDInterface* __stdcall GetFrontPanelLED();

private:
  IASDInterface3* ASDInterface_;
  CASDWrapperDichroicMirror* DichroicMirrorWrapper_;
  std::map<TWheelIndex, CASDWrapperFilterWheel*> FilterWheelWrappers_;
  CASDWrapperDisk* DiskWrapper_;
  CASDWrapperStatus* StatusWrapper_;
  CASDWrapperConfocalMode* ConfocalModeWrapper_;
  CASDWrapperAperture* ApertureWrapper_;
  CASDWrapperCameraPortMirror* CameraPortMirrorWrapper_;
  std::map<TLensType, CASDWrapperLens*> LensWrappers_;
  std::map<TLensType, CASDWrapperIllLens*> IllLensWrappers_;
  CASDWrapperSuperRes* SuperResWrapper_;
  CASDWrapperTIRF* TIRFWrapper_;
  CASDWrapperTIRFPolariser* TIRFPolariserWrapper_;
};

#endif