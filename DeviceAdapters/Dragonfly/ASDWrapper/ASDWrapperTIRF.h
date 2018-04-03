///////////////////////////////////////////////////////////////////////////////
// FILE:          ASDWrapperTIRF.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _ASDWRAPPERTIRF_H_
#define _ASDWRAPPERTIRF_H_

#include "ASDInterface.h"


class CASDWrapperTIRF : public ITIRFInterface
{
public:
  CASDWrapperTIRF( ITIRFInterface* TIRFInterface );
  ~CASDWrapperTIRF();

  // ITIRFInterface
  bool __stdcall GetOpticalPathway( int *Magnification, double *NumericalAperture, double *RefractiveIndex, int *Scope_ID );
  bool __stdcall GetBandwidth( int *MinWavelength_nm, int *MaxWavelength_nm );
  bool __stdcall GetPenetration_nm( int *Depth_nm );
  bool __stdcall GetObliqueAngle_mdeg( int *ObliqueAngle_mdeg );
  bool __stdcall GetOffset( int *Offset );
  bool __stdcall GetTIRFMode( int *TIRFMode );
  bool __stdcall SetOpticalPathway( int Magnification, double NumericalAperture, double RefractiveIndex, int Scope_ID );
  bool __stdcall SetBandwidth( int MinWavelength_nm, int MaxWavelength_nm );
  bool __stdcall SetPenetration_nm( int Depth_nm );
  bool __stdcall SetObliqueAngle_mdeg( int ObliqueAngle_mdeg );
  bool __stdcall SetOffset( int Offset );
  bool __stdcall SetTIRFMode( int TIRFMode );
  ITIRFConfigInterface* __stdcall GetTIRFConfigInterface();
  bool __stdcall GetBandwidthLimit( int *MinWavelength_nm, int *MaxWavelength_nm );
  bool __stdcall GetPenetrationLimit( int *MinDepth_nm, int *MaxDepth_nm );
  bool __stdcall GetObliqueAngleLimit( int *MinObliqueAngle_mdeg, int *MaxObliqueAngle_mdeg );
  bool __stdcall GetOffsetLimit( int *MinOffset, int *MaxOffset );


private:
  ITIRFInterface* TIRFInterface_;
};
#endif