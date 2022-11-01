///////////////////////////////////////////////////////////////////////////////
// FILE:          ASDTIRFIntensity.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _ASDWRAPPERTIRFINTENSITY_H_
#define _ASDWRAPPERTIRFINTENSITY_H_

#include "ASDInterface.h"

class CASDWrapperTIRFIntensity : public ITIRFIntensityInterface
{
public:
	CASDWrapperTIRFIntensity(ITIRFIntensityInterface* TIRFIntensityInterface);
	~CASDWrapperTIRFIntensity();

	// ITIRFIntensityInterface
	bool __stdcall GetTIRFIntensity(int* Intensity);
	bool __stdcall GetTIRFIntensityLimit(int* MinIntensity, int* MaxIntensity);

private:
	ITIRFIntensityInterface* TIRFIntensityInterface_;
};
#endif