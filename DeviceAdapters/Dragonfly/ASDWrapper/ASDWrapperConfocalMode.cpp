#include "ASDWrapperConfocalMode.h"
#include "ASDSDKLock.h"

CASDWrapperConfocalMode::CASDWrapperConfocalMode( IConfocalModeInterface3* ConfocalModeInterface )
  : ConfocalModeInterface_( ConfocalModeInterface )
{
  if ( ConfocalModeInterface_ == nullptr )
  {
    throw std::exception( "Invalid pointer to ConfocalModeInterface" );
  }
}

CASDWrapperConfocalMode::~CASDWrapperConfocalMode()
{
}

///////////////////////////////////////////////////////////////////////////////
// IConfocalModeInterface
///////////////////////////////////////////////////////////////////////////////

bool CASDWrapperConfocalMode::ModeNone()
{
  CASDSDKLock vSDKLock;
  return ConfocalModeInterface_->ModeNone();
}

bool CASDWrapperConfocalMode::ModeConfocalHC()
{
  CASDSDKLock vSDKLock;
  return ConfocalModeInterface_->ModeConfocalHC();
}

bool CASDWrapperConfocalMode::ModeWideField()
{
  CASDSDKLock vSDKLock;
  return ConfocalModeInterface_->ModeWideField();
}

bool CASDWrapperConfocalMode::GetMode( TConfocalMode &BrightFieldMode )
{
  CASDSDKLock vSDKLock;
  return ConfocalModeInterface_->GetMode( BrightFieldMode );
}

///////////////////////////////////////////////////////////////////////////////
// IConfocalModeInterface2
///////////////////////////////////////////////////////////////////////////////

bool CASDWrapperConfocalMode::ModeConfocalHS()
{
  CASDSDKLock vSDKLock;
  return ConfocalModeInterface_->ModeConfocalHS();
}

bool CASDWrapperConfocalMode::IsModeConfocalHSAvailable()
{
  CASDSDKLock vSDKLock;
  return ConfocalModeInterface_->IsModeConfocalHSAvailable();
}

bool CASDWrapperConfocalMode::IsFirstDisk25um()
{
  CASDSDKLock vSDKLock;
  return ConfocalModeInterface_->IsFirstDisk25um();
}

bool CASDWrapperConfocalMode::GetPinHoleSize_um( TConfocalMode ConfocalMode, int *PinHoleSize_um )
{
  CASDSDKLock vSDKLock;
  return ConfocalModeInterface_->GetPinHoleSize_um( ConfocalMode, PinHoleSize_um );
}

///////////////////////////////////////////////////////////////////////////////
// IConfocalModeInterface3
///////////////////////////////////////////////////////////////////////////////

bool CASDWrapperConfocalMode::ModeTIRF()
{
  CASDSDKLock vSDKLock;
  return ConfocalModeInterface_->ModeTIRF();
}

bool CASDWrapperConfocalMode::IsModeTIRFAvailable()
{
  CASDSDKLock vSDKLock;
  return ConfocalModeInterface_->IsModeTIRFAvailable();
}

bool CASDWrapperConfocalMode::IsConfocalModeAvailable( TConfocalMode Mode )
{
  CASDSDKLock vSDKLock;
  return ConfocalModeInterface_->IsConfocalModeAvailable( Mode );
}

