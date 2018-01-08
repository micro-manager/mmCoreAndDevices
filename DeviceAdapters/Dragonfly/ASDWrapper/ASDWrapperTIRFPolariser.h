///////////////////////////////////////////////////////////////////////////////
// FILE:          ASDWrapperTIRFPolariser.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _ASDWRAPPERTIRFPOLARISER_H_
#define _ASDWRAPPERTIRFPOLARISER_H_

#include "ASDInterface.h"


class CASDWrapperTIRFPolariser : public ITIRFPolariserInterface
{
public:
  CASDWrapperTIRFPolariser( ITIRFPolariserInterface* TIRFPolariserInterface );
  ~CASDWrapperTIRFPolariser();

  // ITIRFPolariserInterface
  bool GetPosition( unsigned int& Position );
  bool SetPosition( unsigned int Position );
  bool GetLimits( unsigned int& MinPosition, unsigned int& MaxPosition );

private:
  ITIRFPolariserInterface* TIRFPolariserInterface_;
};
#endif