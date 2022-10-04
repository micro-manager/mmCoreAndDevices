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

class CASDWrapperInterface : public IASDInterface6
{
public:
  CASDWrapperInterface(IASDInterface3* ASDInterface);
  CASDWrapperInterface(IASDInterface4* ASDInterface);
  CASDWrapperInterface(IASDInterface6* ASDInterface);
  ~CASDWrapperInterface();

  bool IsASDInterface4Available() const { return ASDInterface4_ != nullptr; }
  bool IsASDInterface6Available() const { return ASDInterface6_ != nullptr; }

  void InitialiseConfocalMode();

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

  // IASDInterface4
  IConfocalModeInterface4* __stdcall GetImagingMode2();
  bool __stdcall IsBorealisTIRF100Available();
  IBorealisTIRFInterface* __stdcall GetBorealisTIRF100();
  bool __stdcall IsBorealisTIRF60Available();
  IBorealisTIRFInterface* __stdcall GetBorealisTIRF60();

  // IASDInterface5
  const char* __stdcall GetSoftwareVersion2(int ID) const;
  const char* __stdcall GetSoftwareBuildTime2(int ID) const;

  // IASDInterface6
  bool __stdcall IsTIRFIntensityAvailable();
  ITIRFIntensityInterface* __stdcall GetTIRFIntensity();

private:
  IASDInterface3* ASDInterface3_ = nullptr;
  IASDInterface4* ASDInterface4_ = nullptr;
  IASDInterface6* ASDInterface6_ = nullptr;

  CASDWrapperDichroicMirror* DichroicMirrorWrapper_ = nullptr;
  std::map<TWheelIndex, CASDWrapperFilterWheel*> FilterWheelWrappers_;
  CASDWrapperDisk* DiskWrapper_ = nullptr;
  CASDWrapperStatus* StatusWrapper_ = nullptr;
  CASDWrapperConfocalMode* ConfocalModeWrapper_ = nullptr;
  CASDWrapperAperture* ApertureWrapper_ = nullptr;
  CASDWrapperCameraPortMirror* CameraPortMirrorWrapper_ = nullptr;
  std::map<TLensType, CASDWrapperLens*> LensWrappers_;
  std::map<TLensType, CASDWrapperIllLens*> IllLensWrappers_;
  CASDWrapperSuperRes* SuperResWrapper_ = nullptr;
  CASDWrapperTIRF* TIRFWrapper_ = nullptr;
  CASDWrapperTIRFPolariser* TIRFPolariserWrapper_ = nullptr;

  CASDWrapperBorealisTIRF* BorealisTIRF100Wrapper_ = nullptr;
  CASDWrapperBorealisTIRF* BorealisTIRF60Wrapper_ = nullptr;

  CASDWrapperTIRFIntensity* TIRFIntensityWrapper_ = nullptr;
};

#endif