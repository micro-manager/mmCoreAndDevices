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
  : ASDInterface3_( ASDInterface )
{
  if ( ASDInterface3_ == nullptr )
  {
    throw std::exception( "Invalid pointer to ASDInterface3" );
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

  // IASDInterface4
  delete BorealisTIRF100Wrapper_;
  delete BorealisTIRF60Wrapper_;

  // IASDInterface6
  delete TIRFIntensityWrapper_;
}

void CASDWrapperInterface::InitialiseConfocalMode()
{
  if ( ConfocalModeWrapper_ == nullptr )
  {
    CASDSDKLock vSDKLock;
    if ( IsASDInterface4Available() )
      ConfocalModeWrapper_ = new CASDWrapperConfocalMode( ASDInterface4_->GetImagingMode2() );
    else
      ConfocalModeWrapper_ = new CASDWrapperConfocalMode( ASDInterface3_->GetImagingMode() );
  }
}

///////////////////////////////////////////////////////////////////////////////
// IASDInterface
///////////////////////////////////////////////////////////////////////////////

const char* CASDWrapperInterface::GetSerialNumber() const
{
  CASDSDKLock vSDKLock;
  return ASDInterface3_->GetSerialNumber();
}

const char* CASDWrapperInterface::GetProductID() const
{
  CASDSDKLock vSDKLock;
  return ASDInterface3_->GetProductID();
}

const char* CASDWrapperInterface::GetSoftwareVersion() const
{
  CASDSDKLock vSDKLock;
  return ASDInterface3_->GetSoftwareVersion();
}

const char* CASDWrapperInterface::GetSoftwareBuildTime() const
{
  CASDSDKLock vSDKLock;
  return ASDInterface3_->GetSoftwareBuildTime();
}

bool CASDWrapperInterface::IsDichroicAvailable()
{
  CASDSDKLock vSDKLock;
  return ASDInterface3_->IsDichroicAvailable();
}

IDichroicMirrorInterface* CASDWrapperInterface::GetDichroicMirror()
{
  if ( DichroicMirrorWrapper_ == nullptr )
  {
    CASDSDKLock vSDKLock;
    DichroicMirrorWrapper_ = new CASDWrapperDichroicMirror( ASDInterface3_->GetDichroicMirror() );
  }
  return DichroicMirrorWrapper_;
}

bool CASDWrapperInterface::IsDiskAvailable()
{
  CASDSDKLock vSDKLock;
  return ASDInterface3_->IsDiskAvailable();
}

IDiskInterface* CASDWrapperInterface::GetDisk()
{
  if ( DiskWrapper_ == nullptr )
  {
    CASDSDKLock vSDKLock;
    DiskWrapper_ = new CASDWrapperDisk( ASDInterface3_->GetDisk_v2() );
  }
  return DiskWrapper_;
}

bool CASDWrapperInterface::IsFilterWheelAvailable( TWheelIndex FilterIndex )
{
  CASDSDKLock vSDKLock;
  return ASDInterface3_->IsFilterWheelAvailable( FilterIndex );
}

IFilterWheelInterface* CASDWrapperInterface::GetFilterWheel( TWheelIndex FilterIndex )
{
  if ( FilterWheelWrappers_.find( FilterIndex ) == FilterWheelWrappers_.end() )
  {
    CASDSDKLock vSDKLock;
    CASDWrapperFilterWheel* FilterWheelWrapper_ = new CASDWrapperFilterWheel( ASDInterface3_->GetFilterWheel( FilterIndex ) );
    FilterWheelWrappers_[FilterIndex] = FilterWheelWrapper_;
  }
  return FilterWheelWrappers_[FilterIndex];
}

bool CASDWrapperInterface::IsBrightFieldPortAvailable()
{
  CASDSDKLock vSDKLock;
  return ASDInterface3_->IsBrightFieldPortAvailable();
}

IConfocalModeInterface2* CASDWrapperInterface::GetBrightFieldPort()
{
  InitialiseConfocalMode();
  return ConfocalModeWrapper_;
}

IDiskInterface2* CASDWrapperInterface::GetDisk_v2()
{
  if ( DiskWrapper_ == nullptr )
  {
    CASDSDKLock vSDKLock;
    DiskWrapper_ = new CASDWrapperDisk( ASDInterface3_->GetDisk_v2() );
  }
  return DiskWrapper_;
}

///////////////////////////////////////////////////////////////////////////////
// IASDInterface2
///////////////////////////////////////////////////////////////////////////////

bool CASDWrapperInterface::IsApertureAvailable()
{
  CASDSDKLock vSDKLock;
  return ASDInterface3_->IsApertureAvailable();
}

IApertureInterface* CASDWrapperInterface::GetAperture()
{
  if ( ApertureWrapper_ == nullptr )
  {
    CASDSDKLock vSDKLock;
    ApertureWrapper_ = new CASDWrapperAperture( ASDInterface3_->GetAperture() );
  }
  return ApertureWrapper_;
}

bool CASDWrapperInterface::IsCameraPortMirrorAvailable()
{
  CASDSDKLock vSDKLock;
  return ASDInterface3_->IsCameraPortMirrorAvailable();
}

ICameraPortMirrorInterface* CASDWrapperInterface::GetCameraPortMirror()
{
  if ( CameraPortMirrorWrapper_ == nullptr )
  {
    CASDSDKLock vSDKLock;
    CameraPortMirrorWrapper_ = new CASDWrapperCameraPortMirror( ASDInterface3_->GetCameraPortMirror() );
  }
  return CameraPortMirrorWrapper_;
}

bool CASDWrapperInterface::IsLensAvailable( TLensType LensIndex )
{
  CASDSDKLock vSDKLock;
  return ASDInterface3_->IsLensAvailable( LensIndex );
}

ILensInterface* CASDWrapperInterface::GetLens( TLensType LensIndex )
{
  if ( LensWrappers_.find( LensIndex ) == LensWrappers_.end() )
  {
    CASDSDKLock vSDKLock;
    CASDWrapperLens* vLensWrapper_ = new CASDWrapperLens( ASDInterface3_->GetLens( LensIndex ) );
    LensWrappers_[LensIndex] = vLensWrapper_;
  }
  return LensWrappers_[LensIndex];
}

int CASDWrapperInterface::GetModelID()
{
  CASDSDKLock vSDKLock;
  return ASDInterface3_->GetModelID();
}

///////////////////////////////////////////////////////////////////////////////
// IASDInterface3
///////////////////////////////////////////////////////////////////////////////

bool CASDWrapperInterface::IsIllLensAvailable( TLensType LensIndex )
{
  CASDSDKLock vSDKLock;
  return ASDInterface3_->IsIllLensAvailable( LensIndex );
}

IIllLensInterface* CASDWrapperInterface::GetIllLens( TLensType LensIndex )
{
  if ( IllLensWrappers_.find( LensIndex ) == IllLensWrappers_.end() )
  {
    CASDSDKLock vSDKLock;
    CASDWrapperIllLens* vIllLensWrapper_ = new CASDWrapperIllLens( ASDInterface3_->GetIllLens( LensIndex ) );
    IllLensWrappers_[LensIndex] = vIllLensWrapper_;
  }
  return IllLensWrappers_[LensIndex];
}

bool CASDWrapperInterface::IsEPIPolariserAvailable()
{
  CASDSDKLock vSDKLock;
  return ASDInterface3_->IsEPIPolariserAvailable();
}

IEPIPolariserInterface* CASDWrapperInterface::GetEPIPolariser()
{
  throw std::logic_error( "GetEPIPolariser() wrapper function not implemented" );
}

bool CASDWrapperInterface::IsTIRFPolariserAvailable()
{
  CASDSDKLock vSDKLock;
  return ASDInterface3_->IsTIRFPolariserAvailable();
}

ITIRFPolariserInterface* CASDWrapperInterface::GetTIRFPolariser()
{
  if ( TIRFPolariserWrapper_ == nullptr )
  {
    CASDSDKLock vSDKLock;
    TIRFPolariserWrapper_ = new CASDWrapperTIRFPolariser( ASDInterface3_->GetTIRFPolariser() );
  }
  return TIRFPolariserWrapper_;
}

bool CASDWrapperInterface::IsEmissionIrisAvailable()
{
  CASDSDKLock vSDKLock;
  return ASDInterface3_->IsEmissionIrisAvailable();
}

IEmissionIrisInterface* CASDWrapperInterface::GetEmissionIris()
{
  throw std::logic_error( "GetEmissionIris() wrapper function not implemented" );
}

bool CASDWrapperInterface::IsSuperResAvailable()
{
  CASDSDKLock vSDKLock;
  return ASDInterface3_->IsSuperResAvailable();
}

ISuperResInterface* CASDWrapperInterface::GetSuperRes()
{
  if ( SuperResWrapper_ == nullptr )
  {
    CASDSDKLock vSDKLock;
    SuperResWrapper_ = new CASDWrapperSuperRes( ASDInterface3_->GetSuperRes() );
  }
  return SuperResWrapper_;
}

bool CASDWrapperInterface::IsImagingModeAvailable()
{
  CASDSDKLock vSDKLock;
  return ASDInterface3_->IsImagingModeAvailable();
}

IConfocalModeInterface3* CASDWrapperInterface::GetImagingMode()
{
  InitialiseConfocalMode();
  return ConfocalModeWrapper_;
}

bool CASDWrapperInterface::IsTIRFAvailable()
{
  CASDSDKLock vSDKLock;
  return ASDInterface3_->IsTIRFAvailable();
}

ITIRFInterface* CASDWrapperInterface::GetTIRF()
{
  if ( TIRFWrapper_ == nullptr )
  {
    CASDSDKLock vSDKLock;
    TIRFWrapper_ = new CASDWrapperTIRF( ASDInterface3_->GetTIRF() );
  }
  return TIRFWrapper_;
}

IStatusInterface* CASDWrapperInterface::GetStatus()
{
  if ( StatusWrapper_ == nullptr )
  {
    CASDSDKLock vSDKLock;
    StatusWrapper_ = new CASDWrapperStatus( ASDInterface3_->GetStatus() );
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

CASDWrapperInterface::CASDWrapperInterface( IASDInterface4* ASDInterface )
  : CASDWrapperInterface( dynamic_cast< IASDInterface3* >( ASDInterface ) )
{
  ASDInterface4_ = ASDInterface;
  if ( ASDInterface4_ == nullptr )
  {
    throw std::exception( "Invalid pointer to ASDInterface4" );
  }
}

IConfocalModeInterface4* CASDWrapperInterface::GetImagingMode2()
{
  InitialiseConfocalMode();
  return ConfocalModeWrapper_;
}

bool CASDWrapperInterface::IsBorealisTIRF100Available()
{
  CASDSDKLock vSDKLock;
  return ASDInterface4_->IsBorealisTIRF100Available();
}

IBorealisTIRFInterface* CASDWrapperInterface::GetBorealisTIRF100()
{
  if ( IsBorealisTIRF100Available() )
  {
    if ( BorealisTIRF100Wrapper_ == nullptr )
    {
      CASDSDKLock vSDKLock;
      BorealisTIRF100Wrapper_ = new CASDWrapperBorealisTIRF( ASDInterface4_->GetBorealisTIRF100() );
    }
  }
  return BorealisTIRF100Wrapper_;
}

bool CASDWrapperInterface::IsBorealisTIRF60Available()
{
  CASDSDKLock vSDKLock;
  return ASDInterface4_->IsBorealisTIRF60Available();
}

IBorealisTIRFInterface* CASDWrapperInterface::GetBorealisTIRF60()
{
  if ( IsBorealisTIRF60Available() )
  {
    if ( BorealisTIRF60Wrapper_ == nullptr )
    {
      CASDSDKLock vSDKLock;
      BorealisTIRF60Wrapper_ = new CASDWrapperBorealisTIRF( ASDInterface4_->GetBorealisTIRF60() );
    }
  }
  return BorealisTIRF60Wrapper_;
}

///////////////////////////////////////////////////////////////////////////////
// IASDInterface6
///////////////////////////////////////////////////////////////////////////////

CASDWrapperInterface::CASDWrapperInterface( IASDInterface6* ASDInterface )
  : CASDWrapperInterface( dynamic_cast< IASDInterface4* >( ASDInterface ) )
{
  ASDInterface6_ = ASDInterface;
  if ( ASDInterface6_ == nullptr )
  {
    throw std::exception( "Invalid pointer to ASDInterface6" );
  }
}

const char* CASDWrapperInterface::GetSoftwareVersion2( int ID ) const
{
  CASDSDKLock vSDKLock;
  return ASDInterface6_->GetSoftwareVersion2( ID );
}

const char* CASDWrapperInterface::GetSoftwareBuildTime2( int ID ) const
{
  CASDSDKLock vSDKLock;
  return ASDInterface6_->GetSoftwareBuildTime2( ID );
}

bool CASDWrapperInterface::IsTIRFIntensityAvailable()
{
  CASDSDKLock vSDKLock;
  return ASDInterface6_->IsTIRFIntensityAvailable();
}

ITIRFIntensityInterface* CASDWrapperInterface::GetTIRFIntensity()
{
  if ( TIRFIntensityWrapper_ == nullptr )
  {
    CASDSDKLock vSDKLock;
    TIRFIntensityWrapper_ = new CASDWrapperTIRFIntensity( ASDInterface6_->GetTIRFIntensity() );
  }
  return TIRFIntensityWrapper_;
}
