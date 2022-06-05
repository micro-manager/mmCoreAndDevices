#include "ASDWrapperTIRFIntensity.h"
#include "ASDSDKLock.h"

CASDWrapperTIRFIntensity::CASDWrapperTIRFIntensity(ITIRFIntensityInterface* TIRFIntensityInterface)
  : TIRFIntensityInterface_(TIRFIntensityInterface)
{
  if (TIRFIntensityInterface_ == nullptr)
  {
    throw std::exception("Invalid pointer to TIRFIntensityInterface");
  }
}

CASDWrapperTIRFIntensity::~CASDWrapperTIRFIntensity()
{
}

///////////////////////////////////////////////////////////////////////////////
// ITIRFIntensityInterface
///////////////////////////////////////////////////////////////////////////////

bool CASDWrapperTIRFIntensity::GetTIRFIntensity(int* Intensity)
{
  CASDSDKLock vSDKLock;
  return TIRFIntensityInterface_->GetTIRFIntensity(Intensity);
}

bool CASDWrapperTIRFIntensity::GetTIRFIntensityLimit(int* MinIntensity, int* MaxIntensity)
{
  CASDSDKLock vSDKLock;
  return TIRFIntensityInterface_->GetTIRFIntensityLimit(MinIntensity, MaxIntensity);
}
