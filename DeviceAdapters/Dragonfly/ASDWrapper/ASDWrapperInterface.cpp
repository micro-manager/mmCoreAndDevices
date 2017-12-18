#include "ASDWrapperInterface.h"
#include "ASDSDKLock.h"
#include "ASDWrapperDichroicMirror.h"
#include "ASDWrapperFilterWheel.h"
#include "ASDWrapperDisk.h"
#include "ASDWrapperStatus.h"

CASDWrapperInterface::CASDWrapperInterface( IASDInterface3* ASDInterface )
  : ASDInterface_( ASDInterface),
  DichroicMirrorWrapper_( nullptr ),
  DiskWrapper_( nullptr ),
  StatusWrapper_( nullptr )
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
  throw std::logic_error( "GetBrightFieldPort() wrapper function not implemented" );
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
  throw std::logic_error( "GetAperture() wrapper function not implemented" );
}

bool CASDWrapperInterface::IsCameraPortMirrorAvailable()
{
  CASDSDKLock vSDKLock;
  return ASDInterface_->IsCameraPortMirrorAvailable();
}

ICameraPortMirrorInterface* CASDWrapperInterface::GetCameraPortMirror()
{
  throw std::logic_error( "GetCameraPortMirror() wrapper function not implemented" );
}

bool CASDWrapperInterface::IsLensAvailable( TLensType LensIndex )
{
  CASDSDKLock vSDKLock;
  return ASDInterface_->IsLensAvailable( LensIndex );
}

ILensInterface* CASDWrapperInterface::GetLens( TLensType /*LensIndex*/ )
{
  throw std::logic_error( "GetLens() wrapper function not implemented" );
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

IIllLensInterface* CASDWrapperInterface::GetIllLens( TLensType /*LensIndex*/ )
{
  throw std::logic_error( "GetIllLens() wrapper function not implemented" );
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
  throw std::logic_error( "GetTIRFPolariser() wrapper function not implemented" );
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
  throw std::logic_error( "GetSuperRes() wrapper function not implemented" );
}

bool CASDWrapperInterface::IsImagingModeAvailable()
{
  CASDSDKLock vSDKLock;
  return ASDInterface_->IsImagingModeAvailable();
}

IConfocalModeInterface3* CASDWrapperInterface::GetImagingMode()
{
  throw std::logic_error( "GetImagingMode() wrapper function not implemented" );
}

bool CASDWrapperInterface::IsTIRFAvailable()
{
  CASDSDKLock vSDKLock;
  return ASDInterface_->IsTIRFAvailable();
}

ITIRFInterface* CASDWrapperInterface::GetTIRF()
{
  throw std::logic_error( "GetTIRF() wrapper function not implemented" );
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
