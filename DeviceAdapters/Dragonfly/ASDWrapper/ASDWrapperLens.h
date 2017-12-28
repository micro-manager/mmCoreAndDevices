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
  bool GetPosition( unsigned int& Position );
  bool SetPosition( unsigned int Position );
  bool GetLimits( unsigned int& MinPosition, unsigned int& MaxPosition );
  IFilterSet* GetLensConfigInterface();

private:
  ILensInterface* LensInterface_;
  CASDWrapperFilterSet* FilterSetWrapper_;
};
#endif