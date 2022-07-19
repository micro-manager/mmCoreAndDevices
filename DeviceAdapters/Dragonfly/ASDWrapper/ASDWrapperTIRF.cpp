#include "ASDWrapperTIRF.h"
#include "ASDSDKLock.h"

CASDWrapperTIRF::CASDWrapperTIRF( ITIRFInterface* TIRFInterface )
  : TIRFInterface_( TIRFInterface )
{
  if ( TIRFInterface_ == nullptr )
  {
    throw std::exception( "Invalid pointer to TIRFInterface" );
  }
}

CASDWrapperTIRF::~CASDWrapperTIRF()
{
}

///////////////////////////////////////////////////////////////////////////////
// ITIRFInterface
///////////////////////////////////////////////////////////////////////////////

bool CASDWrapperTIRF::GetOpticalPathway( int *Magnification, double *NumericalAperture, double *RefractiveIndex, int *Scope_ID )
{
  CASDSDKLock vSDKLock;
  return TIRFInterface_->GetOpticalPathway( Magnification, NumericalAperture, RefractiveIndex, Scope_ID );
}

bool CASDWrapperTIRF::GetBandwidth( int *MinWavelength_nm, int *MaxWavelength_nm )
{
  CASDSDKLock vSDKLock;
  return TIRFInterface_->GetBandwidth( MinWavelength_nm, MaxWavelength_nm );
}

bool CASDWrapperTIRF::GetPenetration_nm( int *Depth_nm )
{
  CASDSDKLock vSDKLock;
  return TIRFInterface_->GetPenetration_nm( Depth_nm );
}

bool CASDWrapperTIRF::GetObliqueAngle_mdeg( int *ObliqueAngle_mdeg )
{
  CASDSDKLock vSDKLock;
  return TIRFInterface_->GetObliqueAngle_mdeg( ObliqueAngle_mdeg );
}

bool CASDWrapperTIRF::GetOffset( int *Offset )
{
  CASDSDKLock vSDKLock;
  return TIRFInterface_->GetOffset( Offset );
}

bool CASDWrapperTIRF::GetTIRFMode( int *TIRFMode )
{
  CASDSDKLock vSDKLock;
  return TIRFInterface_->GetTIRFMode( TIRFMode );
}

bool CASDWrapperTIRF::SetOpticalPathway( int Magnification, double NumericalAperture, double RefractiveIndex, int Scope_ID )
{
  CASDSDKLock vSDKLock;
  return TIRFInterface_->SetOpticalPathway( Magnification, NumericalAperture, RefractiveIndex, Scope_ID );
}

bool CASDWrapperTIRF::SetBandwidth( int MinWavelength_nm, int MaxWavelength_nm )
{
  CASDSDKLock vSDKLock;
  return TIRFInterface_->SetBandwidth( MinWavelength_nm, MaxWavelength_nm );
}

bool CASDWrapperTIRF::SetPenetration_nm( int Depth_nm )
{
  CASDSDKLock vSDKLock;
  return TIRFInterface_->SetPenetration_nm( Depth_nm );
}

bool CASDWrapperTIRF::SetObliqueAngle_mdeg( int ObliqueAngle_mdeg )
{
  CASDSDKLock vSDKLock;
  return TIRFInterface_->SetObliqueAngle_mdeg( ObliqueAngle_mdeg );
}

bool CASDWrapperTIRF::SetOffset( int Offset )
{
  CASDSDKLock vSDKLock;
  return TIRFInterface_->SetOffset( Offset );
}

bool CASDWrapperTIRF::SetTIRFMode( int TIRFMode )
{
  CASDSDKLock vSDKLock;
  return TIRFInterface_->SetTIRFMode( TIRFMode );
}

ITIRFConfigInterface* CASDWrapperTIRF::GetTIRFConfigInterface()
{
  throw std::logic_error( "ITIRFInterface::GetTIRFConfigInterface() wrapper function not implemented" );
}

bool CASDWrapperTIRF::GetBandwidthLimit( int *MinWavelength_nm, int *MaxWavelength_nm )
{
  CASDSDKLock vSDKLock;
  return TIRFInterface_->GetBandwidthLimit( MinWavelength_nm, MaxWavelength_nm );
}

bool CASDWrapperTIRF::GetPenetrationLimit( int *MinDepth_nm, int *MaxDepth_nm )
{
  CASDSDKLock vSDKLock;
  return TIRFInterface_->GetPenetrationLimit( MinDepth_nm, MaxDepth_nm );
}

bool CASDWrapperTIRF::GetObliqueAngleLimit( int *MinObliqueAngle_mdeg, int *MaxObliqueAngle_mdeg )
{
  CASDSDKLock vSDKLock;
  return TIRFInterface_->GetObliqueAngleLimit( MinObliqueAngle_mdeg, MaxObliqueAngle_mdeg );
}

bool CASDWrapperTIRF::GetOffsetLimit( int *MinOffset, int *MaxOffset )
{
  CASDSDKLock vSDKLock;
  return TIRFInterface_->GetOffsetLimit( MinOffset, MaxOffset );
}

