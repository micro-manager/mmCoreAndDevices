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
class CASDWrapperConfocalMode4;
class CASDWrapperBorealisTIRF;
class CASDWrapperTIRFIntensity;

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

class CASDWrapperInterface4 : public IASDInterface4
{
public:
  CASDWrapperInterface4( IASDInterface4* ASDInterface );
  ~CASDWrapperInterface4();

  // IASDInterface
  const char* __stdcall GetSerialNumber() const { return ASDWrapperInterface_.GetSerialNumber(); }
  const char* __stdcall GetProductID() const { return ASDWrapperInterface_.GetProductID(); }
  const char* __stdcall GetSoftwareVersion() const { return ASDWrapperInterface_.GetSoftwareVersion(); }
  const char* __stdcall GetSoftwareBuildTime() const { return ASDWrapperInterface_.GetSoftwareBuildTime(); }
  bool __stdcall IsDichroicAvailable() { return ASDWrapperInterface_.IsDichroicAvailable(); }
  IDichroicMirrorInterface* __stdcall GetDichroicMirror() { return ASDWrapperInterface_.GetDichroicMirror(); }
  bool __stdcall IsDiskAvailable() { return ASDWrapperInterface_.IsDiskAvailable(); }
  IDiskInterface* __stdcall GetDisk() { return ASDWrapperInterface_.GetDisk(); }
  bool __stdcall IsFilterWheelAvailable( TWheelIndex FilterIndex ) { return ASDWrapperInterface_.IsFilterWheelAvailable( FilterIndex ); }
  IFilterWheelInterface* __stdcall GetFilterWheel( TWheelIndex FilterIndex ) { return ASDWrapperInterface_.GetFilterWheel( FilterIndex ); }
  bool __stdcall IsBrightFieldPortAvailable() { return ASDWrapperInterface_.IsBrightFieldPortAvailable(); }
  IConfocalModeInterface2* __stdcall GetBrightFieldPort() { return ASDWrapperInterface_.GetBrightFieldPort(); }
  IDiskInterface2* __stdcall GetDisk_v2() { return ASDWrapperInterface_.GetDisk_v2(); }

  // IASDInterface2
  bool __stdcall IsApertureAvailable() { return ASDWrapperInterface_.IsApertureAvailable(); }
  IApertureInterface* __stdcall GetAperture() { return ASDWrapperInterface_.GetAperture(); }
  bool __stdcall IsCameraPortMirrorAvailable() { return ASDWrapperInterface_.IsCameraPortMirrorAvailable(); }
  ICameraPortMirrorInterface* __stdcall GetCameraPortMirror() { return ASDWrapperInterface_.GetCameraPortMirror(); }
  bool __stdcall IsLensAvailable( TLensType LensIndex ) { return ASDWrapperInterface_.IsLensAvailable( LensIndex ); }
  ILensInterface* __stdcall GetLens( TLensType LensIndex ) { return ASDWrapperInterface_.GetLens( LensIndex ); }
  int __stdcall GetModelID() { return ASDWrapperInterface_.GetModelID(); }

  // IASDInterface3
  bool __stdcall IsIllLensAvailable( TLensType LensIndex ) { return ASDWrapperInterface_.IsIllLensAvailable( LensIndex ); }
  IIllLensInterface* __stdcall GetIllLens( TLensType LensIndex ) { return ASDWrapperInterface_.GetIllLens( LensIndex ); }
  bool __stdcall IsEPIPolariserAvailable() { return ASDWrapperInterface_.IsEPIPolariserAvailable(); }
  IEPIPolariserInterface* __stdcall GetEPIPolariser() { return ASDWrapperInterface_.GetEPIPolariser(); }
  bool __stdcall IsTIRFPolariserAvailable() { return ASDWrapperInterface_.IsTIRFPolariserAvailable(); }
  ITIRFPolariserInterface* __stdcall GetTIRFPolariser() { return ASDWrapperInterface_.GetTIRFPolariser(); }
  bool __stdcall IsEmissionIrisAvailable() { return ASDWrapperInterface_.IsEmissionIrisAvailable(); }
  IEmissionIrisInterface* __stdcall GetEmissionIris() { return ASDWrapperInterface_.GetEmissionIris(); }
  bool __stdcall IsSuperResAvailable() { return ASDWrapperInterface_.IsSuperResAvailable(); }
  ISuperResInterface* __stdcall GetSuperRes() { return ASDWrapperInterface_.GetSuperRes(); }
  bool __stdcall IsImagingModeAvailable() { return ASDWrapperInterface_.IsImagingModeAvailable(); }
  IConfocalModeInterface3* __stdcall GetImagingMode() { return ASDWrapperInterface_.GetImagingMode(); }
  bool __stdcall IsTIRFAvailable() { return ASDWrapperInterface_.IsTIRFAvailable(); }
  ITIRFInterface* __stdcall GetTIRF() { return ASDWrapperInterface_.GetTIRF(); }
  IStatusInterface* __stdcall GetStatus() { return ASDWrapperInterface_.GetStatus(); }
  IFrontPanelLEDInterface* __stdcall GetFrontPanelLED() { return ASDWrapperInterface_.GetFrontPanelLED(); }

  // IASDInterface4
  IConfocalModeInterface4* __stdcall GetImagingMode2();
  bool __stdcall IsBorealisTIRF100Available();
  IBorealisTIRFInterface* __stdcall GetBorealisTIRF100();
  bool __stdcall IsBorealisTIRF60Available();
  IBorealisTIRFInterface* __stdcall GetBorealisTIRF60();

private:
  CASDWrapperInterface ASDWrapperInterface_;
  IASDInterface4* ASDInterface4_;
  CASDWrapperConfocalMode4* ConfocalMode4Wrapper_;
  CASDWrapperBorealisTIRF* BorealisTIRF100Wrapper_;
  CASDWrapperBorealisTIRF* BorealisTIRF60Wrapper_;
};

class CASDWrapperInterface6 : public IASDInterface6
{
public:
  CASDWrapperInterface6( IASDInterface6* ASDInterface );
  ~CASDWrapperInterface6();

  // IASDInterface
  const char* __stdcall GetSerialNumber() const { return ASDWrapperInterface4_.GetSerialNumber(); }
  const char* __stdcall GetProductID() const { return ASDWrapperInterface4_.GetProductID(); }
  const char* __stdcall GetSoftwareVersion() const { return ASDWrapperInterface4_.GetSoftwareVersion(); }
  const char* __stdcall GetSoftwareBuildTime() const { return ASDWrapperInterface4_.GetSoftwareBuildTime(); }
  bool __stdcall IsDichroicAvailable() { return ASDWrapperInterface4_.IsDichroicAvailable(); }
  IDichroicMirrorInterface* __stdcall GetDichroicMirror() { return ASDWrapperInterface4_.GetDichroicMirror(); }
  bool __stdcall IsDiskAvailable() { return ASDWrapperInterface4_.IsDiskAvailable(); }
  IDiskInterface* __stdcall GetDisk() { return ASDWrapperInterface4_.GetDisk(); }
  bool __stdcall IsFilterWheelAvailable( TWheelIndex FilterIndex ) { return ASDWrapperInterface4_.IsFilterWheelAvailable( FilterIndex ); }
  IFilterWheelInterface* __stdcall GetFilterWheel( TWheelIndex FilterIndex ) { return ASDWrapperInterface4_.GetFilterWheel( FilterIndex ); }
  bool __stdcall IsBrightFieldPortAvailable() { return ASDWrapperInterface4_.IsBrightFieldPortAvailable(); }
  IConfocalModeInterface2* __stdcall GetBrightFieldPort() { return ASDWrapperInterface4_.GetBrightFieldPort(); }
  IDiskInterface2* __stdcall GetDisk_v2() { return ASDWrapperInterface4_.GetDisk_v2(); }

  // IASDInterface2
  bool __stdcall IsApertureAvailable() { return ASDWrapperInterface4_.IsApertureAvailable(); }
  IApertureInterface* __stdcall GetAperture() { return ASDWrapperInterface4_.GetAperture(); }
  bool __stdcall IsCameraPortMirrorAvailable() { return ASDWrapperInterface4_.IsCameraPortMirrorAvailable(); }
  ICameraPortMirrorInterface* __stdcall GetCameraPortMirror() { return ASDWrapperInterface4_.GetCameraPortMirror(); }
  bool __stdcall IsLensAvailable( TLensType LensIndex ) { return ASDWrapperInterface4_.IsLensAvailable( LensIndex ); }
  ILensInterface* __stdcall GetLens( TLensType LensIndex ) { return ASDWrapperInterface4_.GetLens( LensIndex ); }
  int __stdcall GetModelID() { return ASDWrapperInterface4_.GetModelID(); }

  // IASDInterface3
  bool __stdcall IsIllLensAvailable( TLensType LensIndex ) { return ASDWrapperInterface4_.IsIllLensAvailable( LensIndex ); }
  IIllLensInterface* __stdcall GetIllLens( TLensType LensIndex ) { return ASDWrapperInterface4_.GetIllLens( LensIndex ); }
  bool __stdcall IsEPIPolariserAvailable() { return ASDWrapperInterface4_.IsEPIPolariserAvailable(); }
  IEPIPolariserInterface* __stdcall GetEPIPolariser() { return ASDWrapperInterface4_.GetEPIPolariser(); }
  bool __stdcall IsTIRFPolariserAvailable() { return ASDWrapperInterface4_.IsTIRFPolariserAvailable(); }
  ITIRFPolariserInterface* __stdcall GetTIRFPolariser() { return ASDWrapperInterface4_.GetTIRFPolariser(); }
  bool __stdcall IsEmissionIrisAvailable() { return ASDWrapperInterface4_.IsEmissionIrisAvailable(); }
  IEmissionIrisInterface* __stdcall GetEmissionIris() { return ASDWrapperInterface4_.GetEmissionIris(); }
  bool __stdcall IsSuperResAvailable() { return ASDWrapperInterface4_.IsSuperResAvailable(); }
  ISuperResInterface* __stdcall GetSuperRes() { return ASDWrapperInterface4_.GetSuperRes(); }
  bool __stdcall IsImagingModeAvailable() { return ASDWrapperInterface4_.IsImagingModeAvailable(); }
  IConfocalModeInterface3* __stdcall GetImagingMode() { return ASDWrapperInterface4_.GetImagingMode(); }
  bool __stdcall IsTIRFAvailable() { return ASDWrapperInterface4_.IsTIRFAvailable(); }
  ITIRFInterface* __stdcall GetTIRF() { return ASDWrapperInterface4_.GetTIRF(); }
  IStatusInterface* __stdcall GetStatus() { return ASDWrapperInterface4_.GetStatus(); }
  IFrontPanelLEDInterface* __stdcall GetFrontPanelLED() { return ASDWrapperInterface4_.GetFrontPanelLED(); }

  // IASDInterface4
  IConfocalModeInterface4* __stdcall GetImagingMode2() { return ASDWrapperInterface4_.GetImagingMode2(); }
  bool __stdcall IsBorealisTIRF100Available() { return ASDWrapperInterface4_.IsBorealisTIRF100Available(); }
  IBorealisTIRFInterface* __stdcall GetBorealisTIRF100() { return ASDWrapperInterface4_.GetBorealisTIRF100(); }
  bool __stdcall IsBorealisTIRF60Available() { return ASDWrapperInterface4_.IsBorealisTIRF60Available(); }
  IBorealisTIRFInterface* __stdcall GetBorealisTIRF60() { return ASDWrapperInterface4_.GetBorealisTIRF60(); }

  // IASDInterface5
  const char* __stdcall GetSoftwareVersion2( int ID ) const;
  const char* __stdcall GetSoftwareBuildTime2( int ID ) const;

  // IASDInterface6
  bool __stdcall IsTIRFIntensityAvailable();
  ITIRFIntensityInterface* __stdcall GetTIRFIntensity();

private:
  CASDWrapperInterface4 ASDWrapperInterface4_;
  IASDInterface6* ASDInterface6_;
  CASDWrapperTIRFIntensity* TIRFIntensityWrapper_;
};

#endif