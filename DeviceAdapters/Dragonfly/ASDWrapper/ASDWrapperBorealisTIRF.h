///////////////////////////////////////////////////////////////////////////////
// FILE:          ASDBorealisTIRF.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _ASDWRAPPERBOREALISTIRF_H_
#define _ASDWRAPPERBOREALISTIRF_H_

#include "ASDInterface.h"

class CASDWrapperBorealisTIRF : public IBorealisTIRFInterface
{
public:
  CASDWrapperBorealisTIRF(IBorealisTIRFInterface* BorealisTIRFInterface);
  ~CASDWrapperBorealisTIRF();

  // IBorealisTIRFInterface
	bool __stdcall GetBTAngle(int* Angle);
	bool __stdcall SetBTAngle(int Offset);
	bool __stdcall GetBTAngleLimit(int* MinAngle, int* MaxAngle);
	bool __stdcall GetBTMag(int* Mag);
	bool __stdcall GetBTIntensity(int* Intensity);
	bool __stdcall GetBTIntensityLimit(int* MinIntensity, int* MaxIntensity);

private:
  IBorealisTIRFInterface* BorealisTIRFInterface_;
};
#endif