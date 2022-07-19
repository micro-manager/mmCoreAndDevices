#include "ASDWrapperInterface.h"
#include "ASDSDKLock.h"
#include "ASDWrapperDichroicMirror.h"
#include "ASDWrapperFilterWheel.h"
#include "ASDWrapperDisk.h"
#include "ASDWrapperStatus.h"
#include "ASDWrapperConfocalMode.h"
#include "ASDWrapperAperture.h"
#include "ASDWrapperCameraPortMirror.h"
#include "ASDWrapperLens.h"
#include "ASDWrapperIllLens.h"
#include "ASDWrapperSuperRes.h"
#include "ASDWrapperTIRF.h"
#include "ASDWrapperTIRFPolariser.h"

CASDWrapperInterface::CASDWrapperInterface( IASDInterface3* ASDInterface )
  : ASDInterface_( ASDInterface ),
  DichroicMirrorWrapper_( nullptr ),
  DiskWrapper_( nullptr ),
  StatusWrapper_( nullptr ),
  ConfocalModeWrapper_( nullptr ),
  ApertureWrapper_( nullptr ),
  CameraPortMirrorWrapper_( nullptr ),
  SuperResWrapper_( nullptr ),
  TIRFWrapper_( nullptr ),
  TIRFPolariserWrapper_( nullptr )
{
  if ( ASDInterface_ == nullptr )
  {
    throw std::exception( "Invalid pointer to ASDInterface" );
  }
}

CASDWrapperInterface::~CASDWrapperInterface()
{
  delete DichroicMirrorWrapper_;
  std::map<TWheelIndex, CASDWrapperFilterWheel*>::iterator vWheelIt = FilterWheelWrappers_.begin();
  while ( vWheelIt != FilterWheelWrappers_.end() )
  {
    delete vWheelIt->second;
    vWheelIt++;
  }
  delete DiskWrapper_;
  delete StatusWrapper_;
  delete ConfocalModeWrapper_;
  delete ApertureWrapper_;
  delete CameraPortMirrorWrapper_;
  std::map<TLensType, CASDWrapperLens*>::iterator vLensIt = LensWrappers_.begin();
  while ( vLensIt != LensWrappers_.end() )
  {
    delete vLensIt->second;
    vLensIt++;
  }
  std::map<TLensType, CASDWrapperIllLens*>::iterator vIllLensIt = IllLensWrappers_.begin();
  while ( vIllLensIt != IllLensWrappers_.end() )
  {
    delete vIllLensIt->second;
    vIllLensIt++;
  }
  delete SuperResWrapper_;
  delete TIRFWrapper_;
  delete TIRFPolariserWrapper_;
}

///////////////////////////////////////////////////////////////////////////////
// IASDInterface
///////////////////////////////////////////////////////////////////////////////

const char* CASDWrapperInterface::GetSerialNumber() const
{
  CASDSDKLock vSDKLock;
  return ASDInterface_->GetSerialNumber();
}

const char* CASDWrapperInterface::GetProductID() const
{
  CASDSDKLock vSDKLock;
  return ASDInterface_->GetProductID();
}

const char* CASDWrapperInterface::GetSoftwareVersion() const
{
  CASDSDKLock vSDKLock;
  return ASDInterface_->GetSoftwareVersion();
}

const char* CASDWrapperInterface::GetSoftwareBuildTime() const
{
  CASDSDKLock vSDKLock;
  return ASDInterface_->GetSoftwareBuildTime();
}

bool CASDWrapperInterface::IsDichroicAvailable()
{
  CASDSDKLock vSDKLock;
  return ASDInterface_->IsDichroicAvailable();
}

IDichroicMirrorInterface* CASDWrapperInterface::GetDichroicMirror()
{
  if ( DichroicMirrorWrapper_ == nullptr )
  {
    CASDSDKLock vSDKLock;
    DichroicMirrorWrapper_ = new CASDWrapperDichroicMirror( ASDInterface_->GetDichroicMirror() );
  }
  return DichroicMirrorWrapper_;
}

bool CASDWrapperInterface::IsDiskAvailable()
{
  CASDSDKLock vSDKLock;
  return ASDInterface_->IsDiskAvailable();
}

IDiskInterface* CASDWrapperInterface::GetDisk()
{
  if ( DiskWrapper_ == nullptr )
  {
    CASDSDKLock vSDKLock;
    DiskWrapper_ = new CASDWrapperDisk( ASDInterface_->GetDisk_v2() );
  }
  return DiskWrapper_;
}

bool CASDWrapperInterface::IsFilterWheelAvailable( TWheelIndex FilterIndex )
{
  CASDSDKLock vSDKLock;
  return ASDInterface_->IsFilterWheelAvailable( FilterIndex );
}

IFilterWheelInterface* CASDWrapperInterface::GetFilterWheel( TWheelIndex FilterIndex )
{
  if ( FilterWheelWrappers_.find( FilterIndex ) == FilterWheelWrappers_.end() )
  {
    CASDSDKLock vSDKLock;
    CASDWrapperFilterWheel* FilterWheelWrapper_ = new CASDWrapperFilterWheel( ASDInterface_->GetFilterWheel( FilterIndex ) );
    FilterWheelWrappers_[FilterIndex] = FilterWheelWrapper_;
  }
  return FilterWheelWrappers_[FilterIndex];
}

bool CASDWrapperInterface::IsBrightFieldPortAvailable()
{
  CASDSDKLock vSDKLock;
  return ASDInterface_->IsBrightFieldPortAvailable();
}

IConfocalModeInterface2* CASDWrapperInterface::GetBrightFieldPort()
{
  if ( ConfocalModeWrapper_ == nullptr )
  {
    CASDSDKLock vSDKLock;
    ConfocalModeWrapper_ = new CASDWrapperConfocalMode( ASDInterface_->GetImagingMode() );
  }
  return ConfocalModeWrapper_;
}

IDiskInterface2* CASDWrapperInterface::GetDisk_v2()
{
  if ( DiskWrapper_ == nullptr )
  {
    CASDSDKLock vSDKLock;
    DiskWrapper_ = new CASDWrapperDisk( ASDInterface_->GetDisk_v2() );
  }
  return DiskWrapper_;
}

///////////////////////////////////////////////////////////////////////////////
// IASDInterface2
///////////////////////////////////////////////////////////////////////////////

bool CASDWrapperInterface::IsApertureAvailable()
{
  CASDSDKLock vSDKLock;
  return ASDInterface_->IsApertureAvailable();
}

IApertureInterface* CASDWrapperInterface::GetAperture()
{
  if ( ApertureWrapper_ == nullptr )
  {
    CASDSDKLock vSDKLock;
    ApertureWrapper_ = new CASDWrapperAperture( ASDInterface_->GetAperture() );
  }
  return ApertureWrapper_;
}

bool CASDWrapperInterface::IsCameraPortMirrorAvailable()
{
  CASDSDKLock vSDKLock;
  return ASDInterface_->IsCameraPortMirrorAvailable();
}

ICameraPortMirrorInterface* CASDWrapperInterface::GetCameraPortMirror()
{
  if ( CameraPortMirrorWrapper_ == nullptr )
  {
    CASDSDKLock vSDKLock;
    CameraPortMirrorWrapper_ = new CASDWrapperCameraPortMirror( ASDInterface_->GetCameraPortMirror() );
  }
  return CameraPortMirrorWrapper_;
}

bool CASDWrapperInterface::IsLensAvailable( TLensType LensIndex )
{
  CASDSDKLock vSDKLock;
  return ASDInterface_->IsLensAvailable( LensIndex );
}

ILensInterface* CASDWrapperInterface::GetLens( TLensType LensIndex )
{
  if ( LensWrappers_.find( LensIndex ) == LensWrappers_.end() )
  {
    CASDSDKLock vSDKLock;
    CASDWrapperLens* vLensWrapper_ = new CASDWrapperLens( ASDInterface_->GetLens( LensIndex ) );
    LensWrappers_[LensIndex] = vLensWrapper_;
  }
  return LensWrappers_[LensIndex];
}

int CASDWrapperInterface::GetModelID()
{
  CASDSDKLock vSDKLock;
  return ASDInterface_->GetModelID();
}

///////////////////////////////////////////////////////////////////////////////
// IASDInterface3
///////////////////////////////////////////////////////////////////////////////

bool CASDWrapperInterface::IsIllLensAvailable( TLensType LensIndex )
{
  CASDSDKLock vSDKLock;
  return ASDInterface_->IsIllLensAvailable( LensIndex );
}

IIllLensInterface* CASDWrapperInterface::GetIllLens( TLensType LensIndex )
{
  if ( IllLensWrappers_.find( LensIndex ) == IllLensWrappers_.end() )
  {
    CASDSDKLock vSDKLock;
    CASDWrapperIllLens* vIllLensWrapper_ = new CASDWrapperIllLens( ASDInterface_->GetIllLens( LensIndex ) );
    IllLensWrappers_[LensIndex] = vIllLensWrapper_;
  }
  return IllLensWrappers_[LensIndex];
}

bool CASDWrapperInterface::IsEPIPolariserAvailable()
{
  CASDSDKLock vSDKLock;
  return ASDInterface_->IsEPIPolariserAvailable();
}

IEPIPolariserInterface*	CASDWrapperInterface::GetEPIPolariser()
{
  throw std::logic_error( "GetEPIPolariser() wrapper function not implemented" );
}

bool CASDWrapperInterface::IsTIRFPolariserAvailable()
{
  CASDSDKLock vSDKLock;
  return ASDInterface_->IsTIRFPolariserAvailable();
}

ITIRFPolariserInterface* CASDWrapperInterface::GetTIRFPolariser()
{
  if ( TIRFPolariserWrapper_ == nullptr )
  {
    CASDSDKLock vSDKLock;
    TIRFPolariserWrapper_ = new CASDWrapperTIRFPolariser( ASDInterface_->GetTIRFPolariser() );
  }
  return TIRFPolariserWrapper_;
}

bool CASDWrapperInterface::IsEmissionIrisAvailable()
{
  CASDSDKLock vSDKLock;
  return ASDInterface_->IsEmissionIrisAvailable();
}

IEmissionIrisInterface* CASDWrapperInterface::GetEmissionIris()
{
  throw std::logic_error( "GetEmissionIris() wrapper function not implemented" );
}

bool CASDWrapperInterface::IsSuperResAvailable()
{
  CASDSDKLock vSDKLock;
  return ASDInterface_->IsSuperResAvailable();
}

ISuperResInterface* CASDWrapperInterface::GetSuperRes()
{
  if ( SuperResWrapper_ == nullptr )
  {
    CASDSDKLock vSDKLock;
    SuperResWrapper_ = new CASDWrapperSuperRes( ASDInterface_->GetSuperRes() );
  }
  return SuperResWrapper_;
}

bool CASDWrapperInterface::IsImagingModeAvailable()
{
  CASDSDKLock vSDKLock;
  return ASDInterface_->IsImagingModeAvailable();
}

IConfocalModeInterface3* CASDWrapperInterface::GetImagingMode()
{
  if ( ConfocalModeWrapper_ == nullptr )
  {
    CASDSDKLock vSDKLock;
    ConfocalModeWrapper_ = new CASDWrapperConfocalMode( ASDInterface_->GetImagingMode() );
  }
  return ConfocalModeWrapper_;
}

bool CASDWrapperInterface::IsTIRFAvailable()
{
  CASDSDKLock vSDKLock;
  return ASDInterface_->IsTIRFAvailable();
}

ITIRFInterface* CASDWrapperInterface::GetTIRF()
{
  if ( TIRFWrapper_ == nullptr )
  {
    CASDSDKLock vSDKLock;
    TIRFWrapper_ = new CASDWrapperTIRF( ASDInterface_->GetTIRF() );
  }
  return TIRFWrapper_;
}

IStatusInterface* CASDWrapperInterface::GetStatus()
{
  if ( StatusWrapper_ == nullptr )
  {
    CASDSDKLock vSDKLock;
    StatusWrapper_ = new CASDWrapperStatus( ASDInterface_->GetStatus() );
  }
  return StatusWrapper_;
}

IFrontPanelLEDInterface* CASDWrapperInterface::GetFrontPanelLED()
{
  throw std::logic_error( "GetFrontPanelLED() wrapper function not implemented" );
}
