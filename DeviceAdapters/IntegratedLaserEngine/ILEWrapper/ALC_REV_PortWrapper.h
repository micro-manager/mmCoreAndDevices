///////////////////////////////////////////////////////////////////////////////
// FILE:          ALC_REV_PortWrapper.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------

#ifndef _ALC_REV_PORTWRAPPER_H_
#define _ALC_REV_PORTWRAPPER_H_

#include "ALC_REV.h"

class CALC_REV_PortWrapper : public IALC_REV_Port
{
public:
  CALC_REV_PortWrapper( IALC_REV_Port* ALC_REV_Port );
  ~CALC_REV_PortWrapper();

  // IALC_REV_Port
  int __stdcall InitializePort( void );
  bool __stdcall GetNumberOfPorts( int *NumberOfPorts );
  bool __stdcall GetPortIndex( int *PortIndex );
  bool __stdcall SetPortIndex( int PortIndex );

private:
  IALC_REV_Port* ALC_REV_Port_;
};

#endif
