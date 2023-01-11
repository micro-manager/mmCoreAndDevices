#include "ASDWrapperBorealisTIRF.h"
#include "ASDSDKLock.h"

CASDWrapperBorealisTIRF::CASDWrapperBorealisTIRF(IBorealisTIRFInterface* BorealisTIRFInterface)
  : BorealisTIRFInterface_(BorealisTIRFInterface)
{
  if (BorealisTIRFInterface_ == nullptr)
  {
    throw std::exception("Invalid pointer to BorealisTIRFInterface");
  }
}

CASDWrapperBorealisTIRF::~CASDWrapperBorealisTIRF()
{
}

///////////////////////////////////////////////////////////////////////////////
// IBorealisTIRFInterface
///////////////////////////////////////////////////////////////////////////////

bool CASDWrapperBorealisTIRF::GetBTAngle(int* Angle)
{
  CASDSDKLock vSDKLock;
  return BorealisTIRFInterface_->GetBTAngle(Angle);
}

bool CASDWrapperBorealisTIRF::SetBTAngle(int Offset)
{
  CASDSDKLock vSDKLock;
  return BorealisTIRFInterface_->SetBTAngle(Offset);
}

bool CASDWrapperBorealisTIRF::GetBTAngleLimit(int* MinAngle, int* MaxAngle)
{
  CASDSDKLock vSDKLock;
  return BorealisTIRFInterface_->GetBTAngleLimit(MinAngle, MaxAngle);
}

bool CASDWrapperBorealisTIRF::GetBTMag(int* Mag)
{
  CASDSDKLock vSDKLock;
  return BorealisTIRFInterface_->GetBTMag(Mag);
}

bool CASDWrapperBorealisTIRF::GetBTIntensity(int* Intensity)
{
  CASDSDKLock vSDKLock;
  return BorealisTIRFInterface_->GetBTIntensity(Intensity);
}

bool CASDWrapperBorealisTIRF::GetBTIntensityLimit(int* MinIntensity, int* MaxIntensity)
{
  CASDSDKLock vSDKLock;
  return BorealisTIRFInterface_->GetBTIntensityLimit(MinIntensity, MaxIntensity);
}
