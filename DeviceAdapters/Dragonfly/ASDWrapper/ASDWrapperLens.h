///////////////////////////////////////////////////////////////////////////////
// FILE:          ASDWrapperLens.h
// PROJECT:       Micro-Manager
// SUBSYSTEM:     DeviceAdapters
//-----------------------------------------------------------------------------
#ifndef _ASDWRAPPERLENS_H_
#define _ASDWRAPPERLENS_H_

#include "ASDInterface.h"

class CASDWrapperFilterSet;

class CASDWrapperLens : public ILensInterface
{
public:
  CASDWrapperLens( ILensInterface* LensInterface );
  ~CASDWrapperLens();

  // ILensInterface
  bool __stdcall GetPosition( unsigned int& Position );
  bool __stdcall SetPosition( unsigned int Position );
  bool __stdcall GetLimits( unsigned int& MinPosition, unsigned int& MaxPosition );
  IFilterSet* __stdcall GetLensConfigInterface();

private:
  ILensInterface* LensInterface_;
  CASDWrapperFilterSet* FilterSetWrapper_;
};
#endif