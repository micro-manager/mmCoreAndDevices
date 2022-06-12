#include "ASDWrapperConfocalMode.h"
#include "ASDSDKLock.h"

CASDWrapperConfocalMode::CASDWrapperConfocalMode( IConfocalModeInterface3* ConfocalModeInterface )
  : ConfocalModeInterface3_( ConfocalModeInterface )
{
  if ( ConfocalModeInterface3_ == nullptr )
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
  return ConfocalModeInterface3_->ModeNone();
}

bool CASDWrapperConfocalMode::ModeConfocalHC()
{
  CASDSDKLock vSDKLock;
  return ConfocalModeInterface3_->ModeConfocalHC();
}

bool CASDWrapperConfocalMode::ModeWideField()
{
  CASDSDKLock vSDKLock;
  return ConfocalModeInterface3_->ModeWideField();
}

bool CASDWrapperConfocalMode::GetMode( TConfocalMode &BrightFieldMode )
{
  CASDSDKLock vSDKLock;
  return ConfocalModeInterface3_->GetMode( BrightFieldMode );
}

///////////////////////////////////////////////////////////////////////////////
// IConfocalModeInterface2
///////////////////////////////////////////////////////////////////////////////

bool CASDWrapperConfocalMode::ModeConfocalHS()
{
  CASDSDKLock vSDKLock;
  return ConfocalModeInterface3_->ModeConfocalHS();
}

bool CASDWrapperConfocalMode::IsModeConfocalHSAvailable()
{
  CASDSDKLock vSDKLock;
  return ConfocalModeInterface3_->IsModeConfocalHSAvailable();
}

bool CASDWrapperConfocalMode::IsFirstDisk25um()
{
  CASDSDKLock vSDKLock;
  return ConfocalModeInterface3_->IsFirstDisk25um();
}

bool CASDWrapperConfocalMode::GetPinHoleSize_um( TConfocalMode ConfocalMode, int *PinHoleSize_um )
{
  CASDSDKLock vSDKLock;
  return ConfocalModeInterface3_->GetPinHoleSize_um( ConfocalMode, PinHoleSize_um );
}

///////////////////////////////////////////////////////////////////////////////
// IConfocalModeInterface3
///////////////////////////////////////////////////////////////////////////////

bool CASDWrapperConfocalMode::ModeTIRF()
{
  CASDSDKLock vSDKLock;
  return ConfocalModeInterface3_->ModeTIRF();
}

bool CASDWrapperConfocalMode::IsModeTIRFAvailable()
{
  CASDSDKLock vSDKLock;
  return ConfocalModeInterface3_->IsModeTIRFAvailable();
}

bool CASDWrapperConfocalMode::IsConfocalModeAvailable( TConfocalMode Mode )
{
  CASDSDKLock vSDKLock;
  return ConfocalModeInterface3_->IsConfocalModeAvailable( Mode );
}

///////////////////////////////////////////////////////////////////////////////
// IConfocalModeInterface4
///////////////////////////////////////////////////////////////////////////////

CASDWrapperConfocalMode::CASDWrapperConfocalMode( IConfocalModeInterface4* ConfocalModeInterface )
  : CASDWrapperConfocalMode( dynamic_cast< IConfocalModeInterface3* >( ConfocalModeInterface ) )
{
  ConfocalModeInterface4_ = ConfocalModeInterface;
  if (ConfocalModeInterface4_ == nullptr)
  {
    throw std::exception("Invalid pointer to ConfocalModeInterface4");
  }
}

bool CASDWrapperConfocalMode::ModeBorealisTIRF100()
{
  return ConfocalModeInterface4_->ModeBorealisTIRF100();
}

bool CASDWrapperConfocalMode::IsModeBorealisTIRF100Available()
{
  return ConfocalModeInterface4_->IsModeBorealisTIRF100Available();
}

bool CASDWrapperConfocalMode::ModeBorealisTIRF60()
{
  return ConfocalModeInterface4_->ModeBorealisTIRF60();
}

bool CASDWrapperConfocalMode::IsModeBorealisTIRF60Available()
{
  return ConfocalModeInterface4_->IsModeBorealisTIRF60Available();
}

bool CASDWrapperConfocalMode::SetConfocalMode(TConfocalMode Mode)
{
  return ConfocalModeInterface4_->SetConfocalMode(Mode);
}

