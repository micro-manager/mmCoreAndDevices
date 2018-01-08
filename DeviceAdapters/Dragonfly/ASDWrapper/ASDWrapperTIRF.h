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
  bool GetOpticalPathway( int *Magnification, double *NumericalAperture, double *RefractiveIndex, int *Scope_ID );
  bool GetBandwidth( int *MinWavelength_nm, int *MaxWavelength_nm );
  bool GetPenetration_nm( int *Depth_nm );
  bool GetObliqueAngle_mdeg( int *ObliqueAngle_mdeg );
  bool GetOffset( int *Offset );
  bool GetTIRFMode( int *TIRFMode );
  bool SetOpticalPathway( int Magnification, double NumericalAperture, double RefractiveIndex, int Scope_ID );
  bool SetBandwidth( int MinWavelength_nm, int MaxWavelength_nm );
  bool SetPenetration_nm( int Depth_nm );
  bool SetObliqueAngle_mdeg( int ObliqueAngle_mdeg );
  bool SetOffset( int Offset );
  bool SetTIRFMode( int TIRFMode );
  ITIRFConfigInterface* GetTIRFConfigInterface();
  bool GetBandwidthLimit( int *MinWavelength_nm, int *MaxWavelength_nm );
  bool GetPenetrationLimit( int *MinDepth_nm, int *MaxDepth_nm );
  bool GetObliqueAngleLimit( int *MinObliqueAngle_mdeg, int *MaxObliqueAngle_mdeg );
  bool GetOffsetLimit( int *MinOffset, int *MaxOffset );


private:
  ITIRFInterface* TIRFInterface_;
};
#endif