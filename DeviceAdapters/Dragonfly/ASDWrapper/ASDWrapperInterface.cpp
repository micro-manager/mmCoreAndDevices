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
#include "ASDWrapperBorealisTIRF.h"
#include "ASDWrapperTIRFIntensity.h"

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

///////////////////////////////////////////////////////////////////////////////
// IASDInterface4
///////////////////////////////////////////////////////////////////////////////

CASDWrapperInterface4::CASDWrapperInterface4( IASDInterface4* ASDInterface )
  : ASDWrapperInterface_( ASDInterface ),
  ASDInterface4_( ASDInterface ),
  ConfocalMode4Wrapper_( nullptr ),
  BorealisTIRF100Wrapper_( nullptr ),
  BorealisTIRF60Wrapper_( nullptr )
{
  if ( ASDInterface4_ == nullptr )
  {
    throw std::exception("Invalid pointer to ASDInterface4");
  }
}

CASDWrapperInterface4::~CASDWrapperInterface4()
{
  delete ConfocalMode4Wrapper_;
  delete BorealisTIRF100Wrapper_;
  delete BorealisTIRF60Wrapper_;
}

IConfocalModeInterface4* CASDWrapperInterface4::GetImagingMode2()
{
  if ( ConfocalMode4Wrapper_ == nullptr )
  {
    CASDSDKLock vSDKLock;
    ConfocalMode4Wrapper_ = new CASDWrapperConfocalMode4( ASDInterface4_->GetImagingMode2() );
  }
  return ConfocalMode4Wrapper_;
}

bool CASDWrapperInterface4::IsBorealisTIRF100Available()
{
  CASDSDKLock vSDKLock;
  return ASDInterface4_->IsBorealisTIRF100Available();
}

IBorealisTIRFInterface* CASDWrapperInterface4::GetBorealisTIRF100()
{
  if ( BorealisTIRF100Wrapper_ == nullptr )
  {
    CASDSDKLock vSDKLock;
    BorealisTIRF100Wrapper_ = new CASDWrapperBorealisTIRF( ASDInterface4_->GetBorealisTIRF100() );
  }
  return BorealisTIRF100Wrapper_;
}

bool CASDWrapperInterface4::IsBorealisTIRF60Available()
{
  CASDSDKLock vSDKLock;
  return ASDInterface4_->IsBorealisTIRF60Available();
}

IBorealisTIRFInterface* CASDWrapperInterface4::GetBorealisTIRF60()
{
  if ( BorealisTIRF60Wrapper_ == nullptr )
  {
    CASDSDKLock vSDKLock;
    BorealisTIRF60Wrapper_ = new CASDWrapperBorealisTIRF( ASDInterface4_->GetBorealisTIRF60() );
  }
  return BorealisTIRF60Wrapper_;
}

///////////////////////////////////////////////////////////////////////////////
// IASDInterface6
///////////////////////////////////////////////////////////////////////////////

CASDWrapperInterface6::CASDWrapperInterface6( IASDInterface6* ASDInterface )
  : ASDWrapperInterface4_( ASDInterface ),
  ASDInterface6_( ASDInterface ),
  TIRFIntensityWrapper_( nullptr )
{
  if ( ASDInterface6_ == nullptr )
  {
    throw std::exception("Invalid pointer to ASDInterface6");
  }
}

CASDWrapperInterface6::~CASDWrapperInterface6()
{
  delete TIRFIntensityWrapper_;
}

const char* CASDWrapperInterface6::GetSoftwareVersion2( int ID ) const
{
  CASDSDKLock vSDKLock;
  return ASDInterface6_->GetSoftwareVersion2( ID );
}

const char* CASDWrapperInterface6::GetSoftwareBuildTime2( int ID ) const
{
  CASDSDKLock vSDKLock;
  return ASDInterface6_->GetSoftwareBuildTime2( ID );
}

bool CASDWrapperInterface6::IsTIRFIntensityAvailable()
{
  CASDSDKLock vSDKLock;
  return ASDInterface6_->IsTIRFIntensityAvailable();
}

ITIRFIntensityInterface* CASDWrapperInterface6::GetTIRFIntensity()
{
  if ( TIRFIntensityWrapper_ == nullptr )
  {
    CASDSDKLock vSDKLock;
    TIRFIntensityWrapper_ = new CASDWrapperTIRFIntensity( ASDInterface6_->GetTIRFIntensity() );
  }
  return TIRFIntensityWrapper_;
}
