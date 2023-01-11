///////////////////////////////////////////////////////////////////////////////
// FILE:          ALC_REV_ILE2Wrapper.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------

#ifndef _ALC_REV_ILE2WRAPPER_H_
#define _ALC_REV_ILE2WRAPPER_H_

#include "ALC_REV.h"
#include "ALC_REV_ILE2.h"
class CALC_REVObject3Wrapper;

class CALC_REV_ILE2Wrapper : public IALC_REV_ILE2
{
public:
  CALC_REV_ILE2Wrapper( IALC_REV_ILE2* ALC_REV_ILE2 );
  ~CALC_REV_ILE2Wrapper();

  // IALC_REV_ILE2
  bool __stdcall GetNumberOfLasers( int *NumLasersUnit1, int *NumLasersUnit2 );
  bool __stdcall GetNumberOfPorts( int *NumPortsUnit1, int *NumPortsUnit2 );
  bool __stdcall GetPortIndex( int *PortIndexUnit1, int *PortIndexUnit2 );
  bool __stdcall SetPortIndex( int PortIndexUnit1, int PortIndexUnit2 );
  bool __stdcall CalculatePort( int Port1, int Port2, int *Port );
  bool __stdcall ExtractPort( int Port, int *Port1, int *Port2 );
  bool __stdcall GetInterface( IALC_REVObject3 **ILE1, IALC_REVObject3 **ILE2 );
  bool __stdcall IsILE700();
  bool __stdcall GetCurrentPowerIntoFiberForDualUnit( double *Power1_mW, double *Power2_mW );
  bool __stdcall AdjustPowerIntoInputFibre();

private:
  IALC_REV_ILE2* ALC_REV_ILE2_;
  CALC_REVObject3Wrapper* ILEDevice1_;
  CALC_REVObject3Wrapper* ILEDevice2_;
};

#endif
